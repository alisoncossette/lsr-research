"""
Benchmark: LSR vs Top-K vs MMR on NFCorpus with Threshold Sweep

NFCorpus has 3,633 medical documents and 323 test queries with ~38 relevant
docs per query on average. The medical domain creates boilerplate-heavy
neighborhoods where density collapse is expected at moderate-to-high thresholds.

This script sweeps thresholds from 0.3 to 0.7 to find where LSR shines.
"""

import sys
import os
import json
import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.lsr import LocalSpectralRetrieval, AdaptiveLSR, TopKRetrieval, MMRRetrieval

K = 10
THRESHOLDS = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
MMR_LAMBDA = 0.5
MODEL_NAME = "all-MiniLM-L6-v2"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "benchmark_nfcorpus_results.json")


def avg_pairwise_cosine(embeddings):
    if len(embeddings) < 2:
        return 0.0
    sim_matrix = embeddings @ embeddings.T
    n = len(sim_matrix)
    triu_indices = np.triu_indices(n, k=1)
    return float(np.mean(sim_matrix[triu_indices]))


def pc1_variance_ratio(embeddings):
    if len(embeddings) < 3:
        return 0.0
    pca = PCA(n_components=min(3, len(embeddings), embeddings.shape[1]))
    pca.fit(embeddings)
    return float(pca.explained_variance_ratio_[0])


def recall_at_k(selected_indices, relevant_ids, corpus_id_map):
    """What fraction of relevant docs appear in the selected set?"""
    if len(relevant_ids) == 0:
        return 1.0
    selected_doc_ids = set(corpus_id_map[i] for i in selected_indices)
    covered = selected_doc_ids & relevant_ids
    return len(covered) / len(relevant_ids)


def unique_relevant_count(selected_indices, relevant_ids, corpus_id_map):
    """How many unique relevant docs are in the selected set?"""
    selected_doc_ids = set(corpus_id_map[i] for i in selected_indices)
    return len(selected_doc_ids & relevant_ids)


def variance_coverage(selected_embeddings, full_embeddings):
    if len(selected_embeddings) < 2 or len(full_embeddings) < 2:
        return 0.0
    std_selected = np.std(selected_embeddings, axis=0)
    std_full = np.std(full_embeddings, axis=0)
    mask = std_full > 1e-9
    if mask.sum() == 0:
        return 0.0
    ratios = std_selected[mask] / std_full[mask]
    return float(np.mean(np.minimum(ratios, 1.0)))


def main():
    print("=" * 70)
    print("LOADING NFCORPUS")
    print("=" * 70)
    from datasets import load_dataset

    t0 = time.time()
    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/nfcorpus-qrels", split="test")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Corpus: {len(corpus_ds)} docs")
    print(f"  Queries: {len(queries_ds)} queries")
    print(f"  Qrels: {len(qrels_ds)} judgments")

    # Build qrels lookup: query_id -> set of relevant corpus_ids
    qrels = defaultdict(set)
    for row in qrels_ds:
        if row["score"] > 0:
            qrels[row["query-id"]].add(row["corpus-id"])

    # Build corpus
    corpus_ids = [doc["_id"] for doc in corpus_ds]
    corpus_texts = [
        f"{doc['title']}. {doc['text']}" if doc["title"] else doc["text"]
        for doc in corpus_ds
    ]
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    # Build queries
    query_data = []
    for q in queries_ds:
        qid = q["_id"]
        if qid in qrels:
            text = q["text"] if q["text"] else q["title"]
            query_data.append((qid, text))

    print(f"  Queries with relevance judgments: {len(query_data)}")

    # Embed
    print()
    print("EMBEDDING CORPUS AND QUERIES...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model: {MODEL_NAME} (dim={model.get_sentence_embedding_dimension()})")

    t0 = time.time()
    corpus_embeddings = model.encode(
        corpus_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    )
    corpus_embeddings = np.array(corpus_embeddings, dtype=np.float32)
    print(f"  Embedded {len(corpus_texts)} docs in {time.time() - t0:.1f}s")

    query_texts = [q[1] for q in query_data]
    query_ids = [q[0] for q in query_data]
    t0 = time.time()
    query_embeddings = model.encode(
        query_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    )
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    print(f"  Embedded {len(query_texts)} queries in {time.time() - t0:.1f}s")

    # Initialize retrievers
    lsr = LocalSpectralRetrieval(n_components=1, sampling_method="quantile")
    adaptive = AdaptiveLSR()
    topk = TopKRetrieval()
    mmr = MMRRetrieval()

    all_results = {}

    for threshold in THRESHOLDS:
        print()
        print("=" * 70)
        print(f"THRESHOLD = {threshold}")
        print("=" * 70)

        methods = {
            "Top-K": topk,
            "MMR": mmr,
            "LSR": lsr,
            "Adaptive": adaptive,
        }
        results = {
            name: {"redundancy": [], "recall": [], "variance_coverage": [], "n_selected": []}
            for name in methods
        }
        pc1_ratios = []
        neighborhood_sizes = []
        adaptive_modes = []

        for qi in range(len(query_data)):
            q_emb = query_embeddings[qi]
            qid = query_ids[qi]
            relevant_corpus_ids = qrels[qid]

            # Check neighborhood size and collapse at this threshold
            sims = corpus_embeddings @ q_emb
            mask = sims >= threshold
            n_above = mask.sum()
            neighborhood_sizes.append(int(n_above))

            if n_above >= 3:
                neighborhood = corpus_embeddings[mask]
                pc1 = pc1_variance_ratio(neighborhood)
            else:
                pc1 = 0.0
            pc1_ratios.append(pc1)

            for name, retriever in methods.items():
                if name == "Top-K":
                    sel_embs, sel_idx = retriever.retrieve(q_emb, corpus_embeddings, K)
                elif name == "MMR":
                    sel_embs, sel_idx = retriever.retrieve(
                        q_emb, corpus_embeddings, K,
                        threshold=threshold, lambda_param=MMR_LAMBDA,
                    )
                elif name == "LSR":
                    sel_embs, sel_idx = retriever.retrieve(
                        q_emb, corpus_embeddings, K, threshold=threshold
                    )
                elif name == "Adaptive":
                    sel_embs, sel_idx = retriever.retrieve(
                        q_emb, corpus_embeddings, K, threshold=threshold
                    )
                    if hasattr(retriever, "info") and retriever.info:
                        adaptive_modes.append(retriever.info.get("mode", "unknown"))

                if len(sel_idx) == 0:
                    results[name]["redundancy"].append(np.nan)
                    results[name]["recall"].append(0.0)
                    results[name]["variance_coverage"].append(0.0)
                    results[name]["n_selected"].append(0)
                    continue

                red = avg_pairwise_cosine(sel_embs)
                rec = recall_at_k(sel_idx, relevant_corpus_ids, corpus_ids)
                vc = variance_coverage(sel_embs, corpus_embeddings[mask] if n_above > 0 else sel_embs)
                results[name]["redundancy"].append(red)
                results[name]["recall"].append(rec)
                results[name]["variance_coverage"].append(vc)
                results[name]["n_selected"].append(int(len(sel_idx)))

            if (qi + 1) % 100 == 0:
                print(f"  Processed {qi + 1}/{len(query_data)} queries...")

        # Summary
        pc1_arr = np.array(pc1_ratios)
        ns_arr = np.array(neighborhood_sizes)
        print()
        print(f"  Neighborhood stats at threshold={threshold}:")
        print(f"    Size: mean={np.mean(ns_arr):.1f}, median={np.median(ns_arr):.1f}, "
              f"max={np.max(ns_arr)}, zeros={np.sum(ns_arr == 0)}")
        print(f"    PC1:  mean={np.mean(pc1_arr):.4f}, median={np.median(pc1_arr):.4f}")
        pct50 = (pc1_arr > 0.50).sum()
        pct70 = (pc1_arr > 0.70).sum()
        print(f"    PC1 > 0.50: {pct50}/{len(pc1_arr)} ({100 * pct50 / len(pc1_arr):.1f}%)")
        print(f"    PC1 > 0.70: {pct70}/{len(pc1_arr)} ({100 * pct70 / len(pc1_arr):.1f}%)")

        if adaptive_modes:
            mode_counts = Counter(adaptive_modes)
            print(f"    Adaptive routing: {dict(mode_counts)}")

        hdr = f"  {'Method':<12} {'Redundancy':>12} {'Recall@{}'.format(K):>12} {'VarCov':>10} {'Avg N':>8}"
        print()
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        threshold_summary = {}
        for name in methods:
            red_arr = np.array(results[name]["redundancy"])
            rec_arr = np.array(results[name]["recall"])
            vc_arr = np.array(results[name]["variance_coverage"])
            ns_method = np.array(results[name]["n_selected"])
            valid_red = red_arr[~np.isnan(red_arr)]
            mean_red = float(np.mean(valid_red)) if len(valid_red) > 0 else float("nan")
            mean_rec = float(np.mean(rec_arr))
            mean_vc = float(np.mean(vc_arr))
            mean_ns = float(np.mean(ns_method))
            print(f"  {name:<12} {mean_red:>12.4f} {mean_rec:>12.4f} {mean_vc:>10.4f} {mean_ns:>8.1f}")
            threshold_summary[name] = {
                "redundancy_mean": mean_red,
                "recall_mean": mean_rec,
                "variance_coverage_mean": mean_vc,
                "avg_selected": mean_ns,
            }

        # Deltas vs Top-K
        baseline = threshold_summary["Top-K"]
        print()
        for name in ["MMR", "LSR", "Adaptive"]:
            s = threshold_summary[name]
            dr = s["redundancy_mean"] - baseline["redundancy_mean"]
            drec = s["recall_mean"] - baseline["recall_mean"]
            print(f"  {name} vs Top-K: Redundancy {dr:+.4f} | Recall {drec:+.4f}")

        all_results[str(threshold)] = {
            "pc1_mean": float(np.mean(pc1_arr)),
            "pc1_median": float(np.median(pc1_arr)),
            "pct_above_50": float(pct50 / len(pc1_arr)),
            "pct_above_70": float(pct70 / len(pc1_arr)),
            "neighborhood_size_mean": float(np.mean(ns_arr)),
            "methods": threshold_summary,
            "adaptive_routing": dict(Counter(adaptive_modes)) if adaptive_modes else {},
        }

    # Save
    output = {
        "config": {
            "dataset": "NFCorpus",
            "k": K,
            "thresholds": THRESHOLDS,
            "mmr_lambda": MMR_LAMBDA,
            "model": MODEL_NAME,
            "n_queries": len(query_data),
            "n_corpus": len(corpus_ds),
        },
        "results_by_threshold": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Results saved to {RESULTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
