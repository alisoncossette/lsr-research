"""
Benchmark LSR vs MMR vs Top-K on the Neo4j documentation corpus.

Reproduces Table 1 and prevalence statistics from the paper:
- Each of 15,292 documents is used as a query against the full corpus
- Top-50 neighbors form the candidate pool
- LSR, Top-K, MMR, and Adaptive select k=10 from the pool
- Results stratified by collapse level and intra-list redundancy
"""
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lsr import LocalSpectralRetrieval, AdaptiveLSR, MMRRetrieval, TopKRetrieval, power_iteration


def load_neo4j_embeddings(path):
    """Load Neo4j embeddings CSV, return embeddings matrix."""
    import csv
    embeddings = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        emb_cols = [i for i, h in enumerate(header) if h.startswith('emb_')]
        for row in reader:
            vec = [float(row[i]) for i in emb_cols]
            embeddings.append(vec)
    embeddings = np.array(embeddings, dtype=np.float64)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    embeddings = embeddings / norms
    return embeddings


def avg_pairwise_cosine(vecs):
    """Average pairwise cosine similarity among a set of vectors."""
    if len(vecs) < 2:
        return 0.0
    sim_matrix = vecs @ vecs.T
    n = len(vecs)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i, j]
            count += 1
    return total / count if count > 0 else 0.0


def compute_r1(vecs):
    """Compute PC1 explained variance ratio using power iteration."""
    if len(vecs) < 3:
        return 0.0
    centered = vecs - vecs.mean(axis=0)
    _, eigenvalue = power_iteration(centered, n_iterations=10)
    total_var = float(np.sum(centered ** 2) / len(vecs))
    return eigenvalue / total_var if total_var > 1e-12 else 0.0


def select_topk(pool, k):
    """Top-k by original ranking (already sorted by similarity)."""
    return pool[:k]


def select_mmr(pool, similarities, k, lambda_param=0.5):
    """MMR selection from pool."""
    selected = [0]  # Start with most relevant
    for _ in range(k - 1):
        best_score = -np.inf
        best_idx = -1
        for j in range(len(pool)):
            if j in selected:
                continue
            redundancy = max(pool[j] @ pool[s] for s in selected)
            score = lambda_param * similarities[j] - (1 - lambda_param) * redundancy
            if score > best_score:
                best_score = score
                best_idx = j
        if best_idx >= 0:
            selected.append(best_idx)
    return pool[selected]


def select_lsr(pool, k):
    """LSR selection: power iteration + quantile sampling."""
    centered = pool - pool.mean(axis=0)
    v1, _ = power_iteration(centered, n_iterations=10)
    projections = centered @ v1
    sorted_idx = np.argsort(projections)
    n = len(sorted_idx)
    selected = [sorted_idx[int(i * n / k)] for i in range(k)]
    return pool[selected]


def select_adaptive(pool, similarities, k, strong=0.70, moderate=0.50, lambda_param=0.5):
    """Adaptive LSR: route based on r1."""
    centered = pool - pool.mean(axis=0)
    v1, eig1 = power_iteration(centered, n_iterations=10)
    total_var = float(np.sum(centered ** 2) / len(pool))
    r1 = eig1 / total_var if total_var > 1e-12 else 0.0

    if r1 >= strong:
        return select_lsr(pool, k), "lsr"
    elif r1 >= moderate:
        return select_lsr(pool, k), "lsr_2pc"
    else:
        return select_mmr(pool, similarities, k, lambda_param), "mmr"


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'neo4j_embeddings.csv')
    print(f"Loading embeddings from {data_path}...")
    embeddings = load_neo4j_embeddings(data_path)
    N, d = embeddings.shape
    print(f"Loaded {N} documents, {d} dimensions")

    # Precompute full similarity matrix (N x N) in chunks to avoid memory issues
    # For 15k docs at 768-dim, N*N*8 bytes = ~1.7GB -- manageable
    print("Computing similarity matrix...")
    sim_matrix = embeddings @ embeddings.T
    print("Done.")

    K_SELECT = 10
    N_POOL = 50

    # Storage for per-query metrics
    r1_values = []
    intra_list_sims = []
    redundancy_topk = []
    redundancy_mmr = []
    redundancy_lsr = []
    redundancy_adaptive = []

    t0 = time.time()
    for qi in range(N):
        if qi % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  Query {qi}/{N} ({elapsed:.1f}s elapsed)")

        sims = sim_matrix[qi].copy()
        sims[qi] = -1  # exclude self

        # Top-50 pool
        top_indices = np.argsort(-sims)[:N_POOL]
        pool = embeddings[top_indices]
        pool_sims = sims[top_indices]

        # Compute r1 of the pool
        r1 = compute_r1(pool)
        r1_values.append(r1)

        # Compute intra-list similarity of top-10
        top10 = pool[:K_SELECT]
        intra_sim = avg_pairwise_cosine(top10)
        intra_list_sims.append(intra_sim)

        # Top-k selection
        topk_selected = select_topk(pool, K_SELECT)
        redundancy_topk.append(avg_pairwise_cosine(topk_selected))

        # MMR selection
        mmr_selected = select_mmr(pool, pool_sims, K_SELECT)
        redundancy_mmr.append(avg_pairwise_cosine(mmr_selected))

        # LSR selection
        lsr_selected = select_lsr(pool, K_SELECT)
        redundancy_lsr.append(avg_pairwise_cosine(lsr_selected))

        # Adaptive selection
        adaptive_selected, _ = select_adaptive(pool, pool_sims, K_SELECT)
        redundancy_adaptive.append(avg_pairwise_cosine(adaptive_selected))

    elapsed = time.time() - t0
    print(f"\nCompleted {N} queries in {elapsed:.1f}s\n")

    r1_values = np.array(r1_values)
    intra_list_sims = np.array(intra_list_sims)
    redundancy_topk = np.array(redundancy_topk)
    redundancy_mmr = np.array(redundancy_mmr)
    redundancy_lsr = np.array(redundancy_lsr)
    redundancy_adaptive = np.array(redundancy_adaptive)

    # === Prevalence statistics ===
    pct_r1_030 = float(np.mean(r1_values > 0.30) * 100)
    pct_r1_050 = float(np.mean(r1_values > 0.50) * 100)
    pct_intra_080 = float(np.mean(intra_list_sims > 0.80) * 100)

    print("=" * 60)
    print("PREVALENCE OF COLLAPSE")
    print("=" * 60)
    print(f"  r1 > 0.30: {pct_r1_030:.1f}%")
    print(f"  r1 > 0.50: {pct_r1_050:.1f}%")
    print(f"  Intra-list sim > 0.80: {pct_intra_080:.1f}%")

    # === Stratified redundancy results ===
    def report_stratum(name, mask):
        n_queries = int(np.sum(mask))
        if n_queries == 0:
            print(f"\n{name}: 0 queries")
            return {}
        tk = float(np.mean(redundancy_topk[mask]))
        mmr = float(np.mean(redundancy_mmr[mask]))
        lsr = float(np.mean(redundancy_lsr[mask]))
        adp = float(np.mean(redundancy_adaptive[mask]))

        print(f"\n{name} ({n_queries} queries, {n_queries/N*100:.1f}%)")
        print(f"  Top-k:    {tk:.3f}")
        print(f"  MMR:      {mmr:.3f} ({(mmr/tk - 1)*100:+.1f}%)")
        print(f"  LSR:      {lsr:.3f} ({(lsr/tk - 1)*100:+.1f}%)")
        print(f"  Adaptive: {adp:.3f} ({(adp/tk - 1)*100:+.1f}%)")
        return {
            "n_queries": n_queries,
            "topk": tk, "mmr": mmr, "lsr": lsr, "adaptive": adp,
            "mmr_pct": round((mmr/tk - 1)*100, 1),
            "lsr_pct": round((lsr/tk - 1)*100, 1),
            "adaptive_pct": round((adp/tk - 1)*100, 1),
        }

    print("\n" + "=" * 60)
    print("STRATIFIED REDUNDANCY (avg pairwise cosine, lower = better)")
    print("=" * 60)

    high_red = intra_list_sims > 0.80
    collapsed = r1_values > 0.30
    non_collapsed = r1_values < 0.15

    results_high = report_stratum("HIGH REDUNDANCY (intra > 0.80)", high_red)
    results_collapsed = report_stratum("COLLAPSED (r1 > 0.30)", collapsed)
    results_non = report_stratum("NON-COLLAPSED (r1 < 0.15)", non_collapsed)

    # === Overall ===
    all_mask = np.ones(N, dtype=bool)
    results_all = report_stratum("ALL QUERIES", all_mask)

    # Save results
    output = {
        "n_documents": int(N),
        "embedding_dim": int(d),
        "k_select": K_SELECT,
        "n_pool": N_POOL,
        "prevalence": {
            "pct_r1_gt_030": pct_r1_030,
            "pct_r1_gt_050": pct_r1_050,
            "pct_intra_gt_080": pct_intra_080,
        },
        "strata": {
            "high_redundancy": results_high,
            "collapsed": results_collapsed,
            "non_collapsed": results_non,
            "all": results_all,
        }
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'neo4j_corpus_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
