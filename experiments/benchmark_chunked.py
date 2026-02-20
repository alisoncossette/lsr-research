"""
Benchmark: LSR on Chunked Documents (The Real RAG Case)

This is the scenario where LSR shines: when documents are chunked into
overlapping windows, adjacent chunks embed to nearly identical vectors.
Top-k retrieval returns k copies of the same content.

Pipeline:
  1. Chunk NFCorpus abstracts with overlapping windows
  2. Embed all chunks
  3. For each query, run top-N (N >> k) to get candidates
  4. Measure intra-list redundancy of top-k
  5. Apply LSR to select diverse k from top-N candidates
  6. Compare: top-k vs MMR(top-N) vs LSR(top-N)
"""

import sys
import os
import json
import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.lsr import LocalSpectralRetrieval, AdaptiveLSR, TopKRetrieval, MMRRetrieval

K = 10           # final number of results we want
N_CANDIDATES = 50  # initial top-N candidates from vector search
CHUNK_SIZE = 3     # sentences per chunk
OVERLAP = 2        # sentence overlap between chunks
MMR_LAMBDA = 0.5
MODEL_NAME = "all-MiniLM-L6-v2"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "benchmark_chunked_results.json")


def chunk_text_by_sentences(text, chunk_size=3, overlap=2):
    """Split text into overlapping chunks by sentence."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s) > 10]

    if len(sentences) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(sentences), step):
        chunk = ' '.join(sentences[i:i + chunk_size])
        if len(chunk) > 20:
            chunks.append(chunk)
        if i + chunk_size >= len(sentences):
            break

    return chunks if chunks else [text]


def avg_pairwise_cosine(embeddings):
    if len(embeddings) < 2:
        return 0.0
    sim_matrix = embeddings @ embeddings.T
    n = len(sim_matrix)
    triu = np.triu_indices(n, k=1)
    return float(np.mean(sim_matrix[triu]))


def max_pairwise_cosine(embeddings):
    if len(embeddings) < 2:
        return 0.0
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -1)
    return float(np.max(sim_matrix))


def pc1_variance_ratio(embeddings):
    if len(embeddings) < 3:
        return 0.0
    pca = PCA(n_components=min(3, len(embeddings), embeddings.shape[1]))
    pca.fit(embeddings)
    return float(pca.explained_variance_ratio_[0])


def unique_source_docs(indices, chunk_to_doc):
    """Count unique source documents in selected chunks."""
    return len(set(chunk_to_doc[i] for i in indices))


def recall_relevant_docs(indices, chunk_to_doc, relevant_doc_ids):
    """Fraction of relevant source documents covered."""
    if len(relevant_doc_ids) == 0:
        return 1.0
    selected_docs = set(chunk_to_doc[i] for i in indices)
    covered = selected_docs & relevant_doc_ids
    return len(covered) / len(relevant_doc_ids)


def main():
    print("=" * 70)
    print("CHUNKED DOCUMENT BENCHMARK")
    print("=" * 70)
    from datasets import load_dataset

    t0 = time.time()
    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/nfcorpus-qrels", split="test")
    print(f"  Loaded NFCorpus in {time.time() - t0:.1f}s")

    # Build qrels
    qrels = defaultdict(set)
    for row in qrels_ds:
        if row["score"] > 0:
            qrels[row["query-id"]].add(row["corpus-id"])

    # Chunk all documents
    print()
    print(f"CHUNKING DOCUMENTS (chunk_size={CHUNK_SIZE} sentences, overlap={OVERLAP})...")
    chunk_texts = []
    chunk_to_doc = []      # chunk_index -> doc_id
    chunk_to_doc_idx = []  # chunk_index -> doc_index
    doc_to_chunks = defaultdict(list)  # doc_id -> [chunk_indices]

    for doc_idx, doc in enumerate(corpus_ds):
        doc_id = doc["_id"]
        full_text = f"{doc['title']}. {doc['text']}" if doc["title"] else doc["text"]
        chunks = chunk_text_by_sentences(full_text, CHUNK_SIZE, OVERLAP)
        for chunk in chunks:
            chunk_idx = len(chunk_texts)
            chunk_texts.append(chunk)
            chunk_to_doc.append(doc_id)
            chunk_to_doc_idx.append(doc_idx)
            doc_to_chunks[doc_id].append(chunk_idx)

    print(f"  {len(corpus_ds)} docs -> {len(chunk_texts)} chunks")
    print(f"  Avg chunks per doc: {len(chunk_texts) / len(corpus_ds):.1f}")

    # Show chunk overlap example
    example_doc = corpus_ds[0]
    example_chunks = chunk_text_by_sentences(
        f"{example_doc['title']}. {example_doc['text']}", CHUNK_SIZE, OVERLAP
    )
    print(f"\n  Example doc: '{example_doc['title'][:60]}...'")
    print(f"  Chunks: {len(example_chunks)}")
    for i, c in enumerate(example_chunks[:3]):
        print(f"    Chunk {i}: '{c[:80]}...'")

    # Embed chunks
    print()
    print("EMBEDDING CHUNKS...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    t0 = time.time()
    chunk_embeddings = model.encode(
        chunk_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    )
    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
    print(f"  Embedded {len(chunk_texts)} chunks in {time.time() - t0:.1f}s")

    # Measure intra-document chunk similarity
    print()
    print("MEASURING INTRA-DOCUMENT CHUNK SIMILARITY...")
    intra_sims = []
    for doc_id, chunk_idxs in doc_to_chunks.items():
        if len(chunk_idxs) >= 2:
            doc_chunks = chunk_embeddings[chunk_idxs]
            sim = avg_pairwise_cosine(doc_chunks)
            intra_sims.append(sim)
    intra_sims = np.array(intra_sims)
    print(f"  Docs with 2+ chunks: {len(intra_sims)}")
    print(f"  Intra-doc chunk similarity:")
    print(f"    Mean:   {np.mean(intra_sims):.4f}")
    print(f"    Median: {np.median(intra_sims):.4f}")
    print(f"    P90:    {np.percentile(intra_sims, 90):.4f}")
    print(f"    Max:    {np.max(intra_sims):.4f}")
    print(f"    > 0.8:  {(intra_sims > 0.8).sum()}/{len(intra_sims)} "
          f"({100 * (intra_sims > 0.8).mean():.1f}%)")
    print(f"    > 0.9:  {(intra_sims > 0.9).sum()}/{len(intra_sims)} "
          f"({100 * (intra_sims > 0.9).mean():.1f}%)")

    # Embed queries
    query_data = [(q["_id"], q["text"] if q["text"] else q["title"]) for q in queries_ds if q["_id"] in qrels]
    query_texts_list = [q[1] for q in query_data]
    query_ids = [q[0] for q in query_data]

    print()
    print("EMBEDDING QUERIES...")
    query_embeddings = model.encode(
        query_texts_list, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    )
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    print(f"  Embedded {len(query_texts_list)} queries")

    # Run benchmark
    print()
    print("=" * 70)
    print(f"BENCHMARK: Select {K} from top-{N_CANDIDATES} candidates")
    print("=" * 70)

    # Methods that select K from top-N candidates:
    # 1. Top-K: just take the top K (ignore other candidates)
    # 2. MMR: from top-N, greedily select K balancing relevance + diversity
    # 3. LSR: from top-N, do PCA and sample along principal direction
    # 4. Adaptive: check PC1, route to LSR or MMR

    lsr = LocalSpectralRetrieval(n_components=1, sampling_method="quantile")
    adaptive = AdaptiveLSR()
    mmr = MMRRetrieval()

    metrics = {
        name: {"redundancy": [], "unique_docs": [], "recall": [], "pc1": []}
        for name in ["Top-K", "MMR", "LSR", "Adaptive"]
    }
    topk_pc1_ratios = []

    for qi in range(len(query_data)):
        q_emb = query_embeddings[qi]
        qid = query_ids[qi]
        relevant_doc_ids = qrels[qid]

        # Step 1: Get top-N candidates (simulating vector DB retrieval)
        similarities = chunk_embeddings @ q_emb
        top_n_indices = np.argsort(-similarities)[:N_CANDIDATES]
        top_n_embeddings = chunk_embeddings[top_n_indices]
        top_n_sims = similarities[top_n_indices]

        # Measure PC1 of the top-N candidate pool
        if len(top_n_indices) >= 3:
            pc1 = pc1_variance_ratio(top_n_embeddings)
        else:
            pc1 = 0.0
        topk_pc1_ratios.append(pc1)

        # --- Method 1: Top-K (just take the first K) ---
        topk_indices = top_n_indices[:K]
        topk_embs = chunk_embeddings[topk_indices]
        metrics["Top-K"]["redundancy"].append(avg_pairwise_cosine(topk_embs))
        metrics["Top-K"]["unique_docs"].append(unique_source_docs(topk_indices, chunk_to_doc))
        metrics["Top-K"]["recall"].append(recall_relevant_docs(topk_indices, chunk_to_doc, relevant_doc_ids))
        metrics["Top-K"]["pc1"].append(pc1_variance_ratio(topk_embs) if len(topk_embs) >= 3 else 0.0)

        # --- Method 2: MMR from top-N candidates ---
        # Use MMR to select K from the N candidates
        mmr_selected = [int(np.argmax(top_n_sims))]
        for _ in range(K - 1):
            best_score = -np.inf
            best_idx = -1
            for j in range(len(top_n_embeddings)):
                if j in mmr_selected:
                    continue
                redundancy = max(top_n_embeddings[j] @ top_n_embeddings[s] for s in mmr_selected)
                score = MMR_LAMBDA * top_n_sims[j] - (1 - MMR_LAMBDA) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = j
            mmr_selected.append(best_idx)
        mmr_corpus_indices = top_n_indices[mmr_selected]
        mmr_embs = chunk_embeddings[mmr_corpus_indices]
        metrics["MMR"]["redundancy"].append(avg_pairwise_cosine(mmr_embs))
        metrics["MMR"]["unique_docs"].append(unique_source_docs(mmr_corpus_indices, chunk_to_doc))
        metrics["MMR"]["recall"].append(recall_relevant_docs(mmr_corpus_indices, chunk_to_doc, relevant_doc_ids))
        metrics["MMR"]["pc1"].append(pc1_variance_ratio(mmr_embs) if len(mmr_embs) >= 3 else 0.0)

        # --- Method 3: LSR from top-N candidates ---
        # Use PCA on the top-N pool, sample along PC1
        if len(top_n_embeddings) > K:
            pca = PCA(n_components=min(3, len(top_n_embeddings)))
            pca.fit(top_n_embeddings)
            projections = pca.transform(top_n_embeddings)[:, 0]
            sorted_proj = np.argsort(projections)
            n = len(sorted_proj)
            lsr_local = np.array([sorted_proj[int(i * n / K)] for i in range(K)])
            lsr_corpus_indices = top_n_indices[lsr_local]
        else:
            lsr_corpus_indices = top_n_indices[:K]
        lsr_embs = chunk_embeddings[lsr_corpus_indices]
        metrics["LSR"]["redundancy"].append(avg_pairwise_cosine(lsr_embs))
        metrics["LSR"]["unique_docs"].append(unique_source_docs(lsr_corpus_indices, chunk_to_doc))
        metrics["LSR"]["recall"].append(recall_relevant_docs(lsr_corpus_indices, chunk_to_doc, relevant_doc_ids))
        metrics["LSR"]["pc1"].append(pc1_variance_ratio(lsr_embs) if len(lsr_embs) >= 3 else 0.0)

        # --- Method 4: Adaptive ---
        # Check PC1 of candidates, route accordingly
        if len(top_n_embeddings) > K:
            pca_check = PCA(n_components=min(3, len(top_n_embeddings)))
            pca_check.fit(top_n_embeddings)
            r1 = float(pca_check.explained_variance_ratio_[0])

            if r1 >= 0.70:
                # Strong collapse -> LSR
                adaptive_indices = lsr_corpus_indices
            elif r1 >= 0.50:
                # Moderate -> LSR with 2 components
                proj2d = pca_check.transform(top_n_embeddings)[:, :2]
                grid_k = max(2, int(np.sqrt(K)))
                q1 = np.linspace(0, 1, grid_k + 1)
                selected_2pc = []
                for gi in range(grid_k):
                    for gj in range(grid_k):
                        lo1 = np.quantile(proj2d[:, 0], q1[gi])
                        hi1 = np.quantile(proj2d[:, 0], q1[gi + 1])
                        lo2 = np.quantile(proj2d[:, 1], q1[gj])
                        hi2 = np.quantile(proj2d[:, 1], q1[gj + 1])
                        cell = np.where(
                            (proj2d[:, 0] >= lo1) & (proj2d[:, 0] <= hi1) &
                            (proj2d[:, 1] >= lo2) & (proj2d[:, 1] <= hi2)
                        )[0]
                        if len(cell) > 0:
                            selected_2pc.append(cell[0])
                        if len(selected_2pc) >= K:
                            break
                    if len(selected_2pc) >= K:
                        break
                if len(selected_2pc) < K:
                    remaining = sorted(set(range(n)) - set(selected_2pc), key=lambda i: projections[i])
                    for idx in remaining:
                        selected_2pc.append(idx)
                        if len(selected_2pc) >= K:
                            break
                adaptive_indices = top_n_indices[np.array(selected_2pc[:K])]
            else:
                # Weak collapse -> MMR
                adaptive_indices = mmr_corpus_indices
        else:
            adaptive_indices = top_n_indices[:K]

        adaptive_embs = chunk_embeddings[adaptive_indices]
        metrics["Adaptive"]["redundancy"].append(avg_pairwise_cosine(adaptive_embs))
        metrics["Adaptive"]["unique_docs"].append(unique_source_docs(adaptive_indices, chunk_to_doc))
        metrics["Adaptive"]["recall"].append(recall_relevant_docs(adaptive_indices, chunk_to_doc, relevant_doc_ids))
        metrics["Adaptive"]["pc1"].append(pc1_variance_ratio(adaptive_embs) if len(adaptive_embs) >= 3 else 0.0)

        if (qi + 1) % 100 == 0:
            print(f"  Processed {qi + 1}/{len(query_data)} queries...")

    # Results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    pc1_arr = np.array(topk_pc1_ratios)
    print()
    print(f"Top-{N_CANDIDATES} candidate pool PC1 variance ratio:")
    print(f"  Mean:   {np.mean(pc1_arr):.4f}")
    print(f"  Median: {np.median(pc1_arr):.4f}")
    print(f"  > 0.50: {(pc1_arr > 0.50).sum()}/{len(pc1_arr)} ({100 * (pc1_arr > 0.50).mean():.1f}%)")
    print(f"  > 0.70: {(pc1_arr > 0.70).sum()}/{len(pc1_arr)} ({100 * (pc1_arr > 0.70).mean():.1f}%)")

    hdr = f"  {'Method':<12} {'Redundancy':>12} {'Unique Docs':>12} {'Doc Recall':>12} {'Result PC1':>12}"
    print()
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    summary = {}
    for name in ["Top-K", "MMR", "LSR", "Adaptive"]:
        m = metrics[name]
        red = np.nanmean(m["redundancy"])
        ud = np.mean(m["unique_docs"])
        rec = np.mean(m["recall"])
        rpc1 = np.mean(m["pc1"])
        print(f"  {name:<12} {red:>12.4f} {ud:>12.1f} {rec:>12.4f} {rpc1:>12.4f}")
        summary[name] = {
            "redundancy": float(red), "unique_docs": float(ud),
            "recall": float(rec), "result_pc1": float(rpc1),
        }

    print()
    print("  (Redundancy: lower is better | Unique Docs & Recall: higher is better)")
    print()
    baseline = summary["Top-K"]
    for name in ["MMR", "LSR", "Adaptive"]:
        s = summary[name]
        dr = s["redundancy"] - baseline["redundancy"]
        dud = s["unique_docs"] - baseline["unique_docs"]
        drec = s["recall"] - baseline["recall"]
        print(f"  {name} vs Top-K: Redundancy {dr:+.4f} | Unique Docs {dud:+.1f} | Recall {drec:+.4f}")

    # Save
    output = {
        "config": {
            "dataset": "NFCorpus (chunked)",
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "k": K,
            "n_candidates": N_CANDIDATES,
            "mmr_lambda": MMR_LAMBDA,
            "model": MODEL_NAME,
            "n_queries": len(query_data),
            "n_chunks": len(chunk_texts),
            "n_original_docs": len(corpus_ds),
        },
        "intra_doc_similarity": {
            "mean": float(np.mean(intra_sims)),
            "median": float(np.median(intra_sims)),
            "p90": float(np.percentile(intra_sims, 90)),
        },
        "candidate_pool_pc1": {
            "mean": float(np.mean(pc1_arr)),
            "median": float(np.median(pc1_arr)),
            "pct_above_50": float((pc1_arr > 0.50).mean()),
            "pct_above_70": float((pc1_arr > 0.70).mean()),
        },
        "summary": summary,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
