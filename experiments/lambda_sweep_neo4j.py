"""
Lambda Sweep Experiment on Neo4j Documentation Corpus

Tests the thesis: LSR with intent-driven lambda is a unified retrieval
framework that subsumes top-k (lambda~1), MMR (lambda~0.5), and
full-diversity LSR (lambda~0).

Key validation:
1. Redundancy across lambda values (does the dial work?)
2. Document overlap: does LSR(lambda=high) retrieve the SAME docs as top-k?
3. Does LSR(lambda=0.5) match MMR's redundancy profile?
4. Stratified by collapse level (r1)
"""
import numpy as np
import json
import os
import sys
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lsr import LocalSpectralRetrieval, MMRRetrieval, TopKRetrieval, power_iteration


def load_neo4j_embeddings(path):
    """Load Neo4j embeddings CSV, return embeddings matrix."""
    embeddings = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        emb_cols = [i for i, h in enumerate(header) if h.startswith('emb_')]
        for row in reader:
            vec = [float(row[i]) for i in emb_cols]
            embeddings.append(vec)
    embeddings = np.array(embeddings, dtype=np.float64)
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


def jaccard_overlap(set_a, set_b):
    """Jaccard similarity between two index sets."""
    a, b = set(set_a), set(set_b)
    if len(a | b) == 0:
        return 0.0
    return len(a & b) / len(a | b)


def exact_overlap(set_a, set_b):
    """Fraction of set_a that appears in set_b."""
    a, b = set(set_a), set(set_b)
    if len(a) == 0:
        return 0.0
    return len(a & b) / len(a)


def select_lsr_lambda(pool, k, lam):
    """LSR selection with lambda-biased quantile sampling."""
    centered = pool - pool.mean(axis=0)
    v1, _ = power_iteration(centered, n_iterations=10)
    projections = centered @ v1
    sorted_idx = np.argsort(projections)
    n = len(sorted_idx)

    # Lambda-biased positions
    uniform_pos = np.linspace(0, 1, k, endpoint=False) + 0.5 / k
    biased_pos = uniform_pos * (1.0 - lam) + 0.5 * lam
    idx_positions = (biased_pos * (n - 1)).astype(int)
    idx_positions = np.clip(idx_positions, 0, n - 1)

    selected = sorted_idx[idx_positions]
    # Deduplicate
    seen = set()
    result = []
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    if len(result) < k:
        center = n // 2
        for offset in range(n):
            for candidate in [center + offset, center - offset]:
                if 0 <= candidate < n and sorted_idx[candidate] not in seen:
                    seen.add(sorted_idx[candidate])
                    result.append(sorted_idx[candidate])
                if len(result) >= k:
                    break
            if len(result) >= k:
                break

    return np.array(result[:k])


def select_topk(pool, k):
    """Top-k by original ranking (already sorted by similarity)."""
    return np.arange(min(k, len(pool)))


def select_mmr(pool, similarities, k, lambda_param=0.5):
    """MMR selection from pool."""
    selected = [0]
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
    return np.array(selected)


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'neo4j_embeddings.csv')
    print(f"Loading embeddings from {data_path}...")
    embeddings = load_neo4j_embeddings(data_path)
    N, d = embeddings.shape
    print(f"Loaded {N} documents, {d} dimensions")

    print("Computing similarity matrix...")
    sim_matrix = embeddings @ embeddings.T
    print("Done.\n")

    K_SELECT = 10
    N_POOL = 50
    LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

    # Per-query storage
    r1_values = []

    # Redundancy per method
    redundancy_topk = []
    redundancy_mmr = []
    redundancy_by_lambda = {lam: [] for lam in LAMBDA_VALUES}

    # Overlap with top-k per lambda
    jaccard_vs_topk = {lam: [] for lam in LAMBDA_VALUES}
    exact_vs_topk = {lam: [] for lam in LAMBDA_VALUES}

    # Overlap with MMR for lambda=0.5
    jaccard_vs_mmr_05 = []

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

        # r1
        if len(pool) >= 3:
            centered = pool - pool.mean(axis=0)
            _, eig1 = power_iteration(centered, n_iterations=10)
            total_var = float(np.sum(centered ** 2) / len(pool))
            r1 = eig1 / total_var if total_var > 1e-12 else 0.0
        else:
            r1 = 0.0
        r1_values.append(r1)

        # Top-k baseline (indices within pool)
        topk_idx = select_topk(pool, K_SELECT)
        topk_set = set(topk_idx.tolist())
        redundancy_topk.append(avg_pairwise_cosine(pool[topk_idx]))

        # MMR baseline
        mmr_idx = select_mmr(pool, pool_sims, K_SELECT)
        mmr_set = set(mmr_idx.tolist())
        redundancy_mmr.append(avg_pairwise_cosine(pool[mmr_idx]))

        # LSR at each lambda
        for lam in LAMBDA_VALUES:
            lsr_idx = select_lsr_lambda(pool, K_SELECT, lam)
            lsr_set = set(lsr_idx.tolist())

            redundancy_by_lambda[lam].append(avg_pairwise_cosine(pool[lsr_idx]))
            jaccard_vs_topk[lam].append(jaccard_overlap(lsr_set, topk_set))
            exact_vs_topk[lam].append(exact_overlap(topk_set, lsr_set))

            if lam == 0.5:
                jaccard_vs_mmr_05.append(jaccard_overlap(lsr_set, mmr_set))

    elapsed = time.time() - t0
    print(f"\nCompleted {N} queries in {elapsed:.1f}s\n")

    r1_values = np.array(r1_values)
    redundancy_topk = np.array(redundancy_topk)
    redundancy_mmr = np.array(redundancy_mmr)

    # =====================================================================
    # RESULTS
    # =====================================================================

    print("=" * 70)
    print("LAMBDA SWEEP: REDUNDANCY (avg pairwise cosine, lower = more diverse)")
    print("=" * 70)
    print(f"  {'Method':<20} {'Redundancy':>12}  {'vs Top-k':>10}")
    print(f"  {'-'*20} {'-'*12}  {'-'*10}")

    topk_mean = float(np.mean(redundancy_topk))
    mmr_mean = float(np.mean(redundancy_mmr))
    print(f"  {'Top-k':<20} {topk_mean:>12.4f}  {'baseline':>10}")
    print(f"  {'MMR (lam=0.5)':<20} {mmr_mean:>12.4f}  {(mmr_mean/topk_mean - 1)*100:>+9.1f}%")

    for lam in LAMBDA_VALUES:
        arr = np.array(redundancy_by_lambda[lam])
        mean_red = float(np.mean(arr))
        label = f"LSR (lam={lam:.2f})"
        print(f"  {label:<20} {mean_red:>12.4f}  {(mean_red/topk_mean - 1)*100:>+9.1f}%")

    # Document overlap
    print(f"\n{'=' * 70}")
    print("DOCUMENT OVERLAP: LSR vs TOP-K")
    print("(Does high lambda retrieve the same documents as top-k?)")
    print("=" * 70)
    print(f"  {'Lambda':>8}  {'Jaccard':>10}  {'Top-k docs in LSR':>20}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*20}")
    for lam in LAMBDA_VALUES:
        jac = float(np.mean(jaccard_vs_topk[lam]))
        exact = float(np.mean(exact_vs_topk[lam]))
        print(f"  {lam:>8.2f}  {jac:>10.3f}  {exact*100:>19.1f}%")

    # Overlap with MMR at lambda=0.5
    print(f"\n  LSR(lam=0.5) vs MMR Jaccard: {float(np.mean(jaccard_vs_mmr_05)):.3f}")

    # Stratified by collapse
    print(f"\n{'=' * 70}")
    print("STRATIFIED BY COLLAPSE LEVEL")
    print("=" * 70)

    strata = [
        ("Strong collapse (r1 > 0.50)", r1_values > 0.50),
        ("Moderate collapse (0.30 < r1 < 0.50)", (r1_values > 0.30) & (r1_values <= 0.50)),
        ("Weak/no collapse (r1 < 0.15)", r1_values < 0.15),
    ]

    results_strata = {}
    for stratum_name, mask in strata:
        n_q = int(np.sum(mask))
        if n_q == 0:
            print(f"\n  {stratum_name}: 0 queries")
            continue

        print(f"\n  {stratum_name} ({n_q} queries, {n_q/N*100:.1f}%)")
        tk = float(np.mean(redundancy_topk[mask]))
        mm = float(np.mean(redundancy_mmr[mask]))
        print(f"    {'Top-k':<20} {tk:.4f}")
        print(f"    {'MMR':<20} {mm:.4f} ({(mm/tk - 1)*100:+.1f}%)")

        stratum_data = {"n_queries": n_q, "topk": tk, "mmr": mm}
        for lam in [0.0, 0.2, 0.5, 0.9, 1.0]:
            arr = np.array(redundancy_by_lambda[lam])
            lsr_val = float(np.mean(arr[mask]))
            jac_val = float(np.mean(np.array(jaccard_vs_topk[lam])[mask]))
            print(f"    {'LSR(lam='+str(lam)+')':<20} {lsr_val:.4f} ({(lsr_val/tk - 1)*100:+.1f}%)  topk-overlap: {jac_val:.3f}")
            stratum_data[f"lsr_{lam}"] = lsr_val
            stratum_data[f"jac_{lam}"] = jac_val
        results_strata[stratum_name] = stratum_data

    # Intent presets
    print(f"\n{'=' * 70}")
    print("INTENT PRESETS")
    print("=" * 70)
    for intent, lam in [("factual", 0.9), ("understanding", 0.2), ("balanced", 0.5)]:
        arr = np.array(redundancy_by_lambda[lam])
        red = float(np.mean(arr))
        jac = float(np.mean(jaccard_vs_topk[lam]))
        print(f"  intent='{intent}' (lambda={lam})")
        print(f"    Redundancy: {red:.4f} (top-k: {topk_mean:.4f}, MMR: {mmr_mean:.4f})")
        print(f"    Top-k Jaccard overlap: {jac:.3f}")

    # Save results
    output = {
        "n_documents": int(N),
        "embedding_dim": int(d),
        "k_select": K_SELECT,
        "n_pool": N_POOL,
        "overall": {
            "topk_redundancy": topk_mean,
            "mmr_redundancy": mmr_mean,
            "lambda_redundancy": {str(lam): float(np.mean(redundancy_by_lambda[lam])) for lam in LAMBDA_VALUES},
            "lambda_jaccard_vs_topk": {str(lam): float(np.mean(jaccard_vs_topk[lam])) for lam in LAMBDA_VALUES},
            "lambda_exact_vs_topk": {str(lam): float(np.mean(exact_vs_topk[lam])) for lam in LAMBDA_VALUES},
            "lsr05_vs_mmr_jaccard": float(np.mean(jaccard_vs_mmr_05)),
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'lambda_sweep_neo4j.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
