"""
Video Frame Averaging Experiment for LSR -- YouTube-8M Edition

Uses Google's YouTube-8M dataset: pre-computed 1024-d frame-level visual
features (PCA'd Inception-v3) for real YouTube videos.

Downloads a single validation shard (~120MB), extracts frame embeddings
grouped by video, then compares retrieval on:
  (a) raw per-frame embeddings
  (b) mean-pooled clip embeddings (averaging N frames)

Shows that mean-pooling manufactures representation collapse (higher r1)
and that LSR exploits it for better diversity.

Requires: pip install tfrecord   (lightweight, no TF dependency)
Dataset:  YouTube-8M (Google, CC BY 4.0)
          https://research.google.com/youtube8m/
"""

import sys
import os
import struct
import urllib.request
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lsr import LocalSpectralRetrieval, TopKRetrieval, ThresholdRandomRetrieval, MMRRetrieval, AdaptiveLSR
from src.metrics import compute_redundancy_score, compute_recall

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data" / "yt8m"
SHARD_URL = "http://us.data.yt8m.org/2/frame/validate/validateaa.tfrecord"
SHARD_FILE = DATA_DIR / "validateaa.tfrecord"

# YouTube-8M dequantization constants
MAX_QUANTIZED_VALUE = 2.0
MIN_QUANTIZED_VALUE = -2.0


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_shard():
    """Download one YouTube-8M frame-level validation shard."""
    if SHARD_FILE.exists():
        size_mb = SHARD_FILE.stat().st_size / (1024 * 1024)
        print(f"    Shard already downloaded ({size_mb:.1f} MB)")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading {SHARD_URL}")
    print(f"    (this is ~100-150 MB, one-time download)")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            # Only print every 10%
            if int(pct) % 10 == 0 and (block_num == 0 or int((block_num-1)*block_size*100/total_size) % 10 != 0):
                print(f"    {mb:.1f} MB ({pct:.0f}%)")

    urllib.request.urlretrieve(SHARD_URL, SHARD_FILE, reporthook=progress_hook)
    size_mb = SHARD_FILE.stat().st_size / (1024 * 1024)
    print(f"    Downloaded: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Parse YouTube-8M TFRecords
# ---------------------------------------------------------------------------
def dequantize(feat_vector):
    """YouTube-8M dequantization: uint8 -> float in [-2, 2]."""
    quantized_range = MAX_QUANTIZED_VALUE - MIN_QUANTIZED_VALUE  # 4.0
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + MIN_QUANTIZED_VALUE
    return feat_vector * scalar + bias


def load_yt8m_frames(max_videos=500):
    """
    Load frame-level RGB features from a YouTube-8M TFRecord shard.

    Returns:
        videos: list of dicts with keys:
            'id': video ID string
            'labels': list of int label IDs
            'frames': (n_frames, 1024) float32 array of frame features
    """
    from tfrecord.reader import sequence_loader

    videos = []
    count = 0

    context_desc = {"id": "byte", "labels": "int"}
    feat_desc = {"rgb": "byte"}

    for context, features in sequence_loader(
        str(SHARD_FILE), None,
        context_description=context_desc,
        features_description=feat_desc,
    ):
        video_id = context["id"].decode("utf-8", errors="replace") \
            if isinstance(context["id"], bytes) else str(context["id"])
        labels_raw = context["labels"]
        labels = labels_raw.tolist() if hasattr(labels_raw, 'tolist') else list(labels_raw)

        # rgb: list of bytes objects, each 1024 bytes (uint8 features)
        rgb_list = features["rgb"]
        if len(rgb_list) == 0:
            continue

        # Convert list of bytes -> (n_frames, 1024) uint8 -> float32
        frame_arrays = [np.frombuffer(b, dtype=np.uint8) for b in rgb_list]
        frames = np.stack(frame_arrays).astype(np.float32)
        frames = dequantize(frames)

        videos.append({
            "id": video_id,
            "labels": labels,
            "frames": frames,
        })
        count += 1
        if count >= max_videos:
            break

    return videos


# ---------------------------------------------------------------------------
# Build corpora
# ---------------------------------------------------------------------------
def build_corpora(videos, frames_to_avg=5):
    """
    Build two corpora from video data:
      1. Raw frames: each frame is a separate document
      2. Mean-pooled: average N frames per clip, one doc per clip

    Also builds label-based relevance sets.

    Returns:
        raw_corpus: (total_frames, 1024) normalized
        raw_video_ids: (total_frames,) which video each frame belongs to
        pooled_corpus: (n_clips, 1024) normalized
        pooled_video_ids: (n_clips,) which video each clip belongs to
        label_map: dict mapping label_id -> set of video indices
    """
    raw_frames = []
    raw_vid_ids = []
    pooled_clips = []
    pooled_vid_ids = []
    label_map = {}

    for vid_idx, video in enumerate(videos):
        frames = video["frames"]
        n_frames = len(frames)

        # Register labels
        for lab in video["labels"]:
            if lab not in label_map:
                label_map[lab] = set()
            label_map[lab].add(vid_idx)

        # Raw frames
        for f in range(n_frames):
            raw_frames.append(frames[f])
            raw_vid_ids.append(vid_idx)

        # Mean-pooled clips: chunk frames into groups of `frames_to_avg`
        for start in range(0, n_frames, frames_to_avg):
            chunk = frames[start : start + frames_to_avg]
            pooled = chunk.mean(axis=0)
            pooled_clips.append(pooled)
            pooled_vid_ids.append(vid_idx)

    raw_corpus = np.array(raw_frames, dtype=np.float32)
    pooled_corpus = np.array(pooled_clips, dtype=np.float32)

    # L2-normalize
    raw_norms = np.linalg.norm(raw_corpus, axis=1, keepdims=True)
    raw_norms[raw_norms < 1e-8] = 1.0
    raw_corpus /= raw_norms

    pooled_norms = np.linalg.norm(pooled_corpus, axis=1, keepdims=True)
    pooled_norms[pooled_norms < 1e-8] = 1.0
    pooled_corpus /= pooled_norms

    return (
        raw_corpus, np.array(raw_vid_ids),
        pooled_corpus, np.array(pooled_vid_ids),
        label_map,
    )


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def measure_collapse(corpus, query, threshold):
    """PC1 explained variance ratio in the threshold neighborhood."""
    sims = corpus @ query
    mask = sims >= threshold
    neighborhood = corpus[mask]
    if len(neighborhood) < 5:
        return None, len(neighborhood)
    n_comp = min(5, len(neighborhood) - 1, neighborhood.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(neighborhood)
    return float(pca.explained_variance_ratio_[0]), len(neighborhood)


def retrieve_and_measure(retriever, query, corpus, k, threshold, relevant):
    """Run retrieval, return (recall, redundancy)."""
    embs, idxs = retriever.retrieve(query, corpus, k=k, threshold=threshold)
    if len(embs) == 0:
        return 0.0, 0.0
    rec = compute_recall(idxs, relevant)
    red = compute_redundancy_score(embs)
    return rec, red


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(
    max_videos=300,
    frames_to_avg=5,
    n_queries=30,
    k=10,
    threshold=0.5,
    seed=42,
):
    rng = np.random.RandomState(seed)

    print("=" * 72)
    print("VIDEO FRAME AVERAGING EXPERIMENT (YouTube-8M)")
    print("=" * 72)

    # --- Download ---
    print("\n[1] Downloading YouTube-8M frame-level features...")
    download_shard()

    # --- Parse ---
    print("\n[2] Parsing TFRecord shard...")
    videos = load_yt8m_frames(max_videos=max_videos)
    total_frames = sum(len(v["frames"]) for v in videos)
    all_labels = set()
    for v in videos:
        all_labels.update(v["labels"])
    print(f"    Videos loaded: {len(videos)}")
    print(f"    Total frames:  {total_frames}")
    print(f"    Unique labels: {len(all_labels)}")
    print(f"    Frames/video:  mean={total_frames/len(videos):.0f}, "
          f"min={min(len(v['frames']) for v in videos)}, "
          f"max={max(len(v['frames']) for v in videos)}")

    # --- Build corpora ---
    print(f"\n[3] Building corpora (averaging {frames_to_avg} frames per clip)...")
    raw_corpus, raw_vid_ids, pooled_corpus, pooled_vid_ids, label_map = \
        build_corpora(videos, frames_to_avg=frames_to_avg)
    print(f"    Raw frame corpus:   {raw_corpus.shape}")
    print(f"    Mean-pooled corpus: {pooled_corpus.shape}")

    # Pick labels that have enough videos for meaningful retrieval
    good_labels = [lab for lab, vids in label_map.items() if 5 <= len(vids) <= 80]
    if len(good_labels) < n_queries:
        print(f"    Warning: only {len(good_labels)} labels with 5-80 videos, "
              f"reducing queries")
        n_queries = min(n_queries, len(good_labels))
    query_labels = rng.choice(good_labels, size=n_queries, replace=len(good_labels) < n_queries)

    # --- Build queries ---
    # For each query: pick a random video with that label, use one of its
    # frames as the query embedding (simulates "find similar videos")
    print(f"\n[4] Running {n_queries} queries (k={k}, threshold={threshold})...")

    retrievers = {
        "Top-k":    TopKRetrieval(),
        "Random":   ThresholdRandomRetrieval(),
        "MMR":      MMRRetrieval(),
        "LSR":      LocalSpectralRetrieval(n_components=1, sampling_method="quantile"),
        "Adaptive": AdaptiveLSR(),
    }

    results = {
        corpus_type: {name: {"recall": [], "redundancy": []} for name in retrievers}
        for corpus_type in ["raw", "pooled"]
    }
    collapse_stats = {"raw": [], "pooled": []}

    for qi in range(n_queries):
        label = query_labels[qi]
        relevant_vids = label_map[label]

        # Pick a query video, use its middle frame as the query
        query_vid = rng.choice(list(relevant_vids))
        query_frames = videos[query_vid]["frames"]
        mid = len(query_frames) // 2
        query = query_frames[mid].copy()
        query /= np.linalg.norm(query) + 1e-12

        # --- Pooled corpus ---
        # Relevant = any clip from a video with the same label
        relevant_pooled = set(np.where(np.isin(pooled_vid_ids, list(relevant_vids)))[0].tolist())
        pc1_p, sz_p = measure_collapse(pooled_corpus, query, threshold)
        if pc1_p is not None:
            collapse_stats["pooled"].append((pc1_p, sz_p))

        for name, retriever in retrievers.items():
            rec, red = retrieve_and_measure(
                retriever, query, pooled_corpus, k, threshold, relevant_pooled
            )
            results["pooled"][name]["recall"].append(rec)
            results["pooled"][name]["redundancy"].append(red)

        # --- Raw frame corpus ---
        relevant_raw = set(np.where(np.isin(raw_vid_ids, list(relevant_vids)))[0].tolist())
        pc1_r, sz_r = measure_collapse(raw_corpus, query, threshold)
        if pc1_r is not None:
            collapse_stats["raw"].append((pc1_r, sz_r))

        for name, retriever in retrievers.items():
            rec, red = retrieve_and_measure(
                retriever, query, raw_corpus, k, threshold, relevant_raw
            )
            results["raw"][name]["recall"].append(rec)
            results["raw"][name]["redundancy"].append(red)

        if (qi + 1) % 10 == 0:
            print(f"    {qi+1}/{n_queries} queries done")

    # === REPORT ===
    print(f"\n{'=' * 72}")
    print("COLLAPSE DIAGNOSTICS (r1 = PC1 explained variance ratio)")
    print(f"{'=' * 72}")

    for label, stats in [("Mean-pooled", collapse_stats["pooled"]),
                          ("Raw frames", collapse_stats["raw"])]:
        if not stats:
            print(f"\n  {label}: no neighborhoods large enough")
            continue
        pc1s = [s[0] for s in stats]
        sizes = [s[1] for s in stats]
        print(f"\n  {label} ({len(stats)} queries with valid neighborhoods):")
        print(f"    r1:   mean={np.mean(pc1s):.1%}  median={np.median(pc1s):.1%}  "
              f"min={np.min(pc1s):.1%}  max={np.max(pc1s):.1%}")
        print(f"    |N|:  mean={np.mean(sizes):.0f}  "
              f"min={np.min(sizes):.0f}  max={np.max(sizes):.0f}")

    if collapse_stats["pooled"] and collapse_stats["raw"]:
        delta = np.mean([s[0] for s in collapse_stats["pooled"]]) - \
                np.mean([s[0] for s in collapse_stats["raw"]])
        print(f"\n  Mean-pooling {'increases' if delta > 0 else 'decreases'} "
              f"r1 by {abs(delta):.1%}")

    # --- Retrieval results ---
    for corpus_label, corpus_name in [("pooled", "MEAN-POOLED CLIPS"),
                                       ("raw", "RAW FRAMES")]:
        print(f"\n{'=' * 72}")
        print(f"  {corpus_name}")
        print(f"{'=' * 72}")
        print(f"\n  {'Method':<16} {'Recall':>10} {'Redundancy':>12}   (lower = more diverse)")
        print(f"  {'-'*52}")
        for name in retrievers:
            rec = np.mean(results[corpus_label][name]["recall"])
            red = np.mean(results[corpus_label][name]["redundancy"])
            print(f"  {name:<16} {rec:>10.3f} {red:>12.4f}")

    # --- LSR advantage ---
    print(f"\n{'=' * 72}")
    print("LSR DIVERSITY ADVANTAGE (redundancy reduction vs Top-k)")
    print(f"{'=' * 72}")
    for label, nice in [("pooled", "Mean-pooled"), ("raw", "Raw frames")]:
        topk_red = np.mean(results[label]["Top-k"]["redundancy"])
        lsr_red = np.mean(results[label]["LSR"]["redundancy"])
        if topk_red > 1e-9:
            pct = (topk_red - lsr_red) / topk_red * 100
            print(f"  {nice:<16} Top-k={topk_red:.4f}  LSR={lsr_red:.4f}  "
                  f"({pct:+.1f}% less redundant)")

    # --- Frame count sweep ---
    print(f"\n{'=' * 72}")
    print("FRAME COUNT SWEEP: How averaging more frames affects collapse")
    print(f"{'=' * 72}")
    print(f"\n  {'Frames':>8} {'r1 (PC1)':>10} {'Top-k Red':>12} {'LSR Red':>12} {'Delta':>10}")
    print(f"  {'-'*56}")

    lsr = LocalSpectralRetrieval(n_components=1, sampling_method="quantile")
    topk = TopKRetrieval()

    for nf in [1, 2, 3, 5, 10, 20]:
        _, _, sweep_corpus, sweep_vids, _ = build_corpora(videos, frames_to_avg=nf)
        pc1_list, topk_red_list, lsr_red_list = [], [], []

        for qi in range(n_queries):
            label = query_labels[qi]
            relevant_vids = label_map[label]
            query_vid = rng.choice(list(relevant_vids))
            qf = videos[query_vid]["frames"]
            q = qf[len(qf)//2].copy()
            q /= np.linalg.norm(q) + 1e-12

            relevant = set(np.where(np.isin(sweep_vids, list(relevant_vids)))[0].tolist())

            pc1, _ = measure_collapse(sweep_corpus, q, threshold)
            if pc1 is not None:
                pc1_list.append(pc1)

            _, red_t = retrieve_and_measure(topk, q, sweep_corpus, k, threshold, relevant)
            _, red_l = retrieve_and_measure(lsr, q, sweep_corpus, k, threshold, relevant)
            topk_red_list.append(red_t)
            lsr_red_list.append(red_l)

        pc1_mean = np.mean(pc1_list) if pc1_list else 0.0
        topk_mean = np.mean(topk_red_list)
        lsr_mean = np.mean(lsr_red_list)
        delta = (topk_mean - lsr_mean) / max(topk_mean, 1e-9) * 100
        print(f"  {nf:>8} {pc1_mean:>10.1%} {topk_mean:>12.4f} {lsr_mean:>12.4f} {delta:>+9.1f}%")

    print(f"\n  r1 = lambda_1 / sum(lambda_j)  (PC1 explained variance ratio)")
    print(f"  As frames averaged increases, per-frame noise cancels (1/N),")
    print(f"  but cross-video content spread is preserved --> r1 climbs.")
    print(f"\n{'=' * 72}")
    print("Done.")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    run_experiment()
