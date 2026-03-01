"""
Generate Figure 4 (Variance Coverage vs k) from real Neo4j corpus embeddings.

Uses the 15,292-document Neo4j corpus with 768-dim embeddings.
Picks queries where density collapse is present (r1 > 0.30) and
averages variance coverage across them for each k.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lsr import power_iteration

FIGS_DIR = os.path.join(os.path.dirname(__file__), "figs")
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'neo4j_embeddings.csv')

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


def load_embeddings(path):
    """Load Neo4j embeddings CSV directly into numpy."""
    import pandas as pd
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    embeddings = df[emb_cols].values.astype(np.float32)
    del df  # free memory
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return embeddings / norms


def compute_r1(vecs):
    if len(vecs) < 3:
        return 0.0
    centered = vecs - vecs.mean(axis=0)
    _, eigenvalue = power_iteration(centered, n_iterations=10)
    total_var = float(np.sum(centered ** 2) / len(vecs))
    return eigenvalue / total_var if total_var > 1e-12 else 0.0


def variance_coverage(selected, full):
    """Fraction of full neighborhood variance covered by selected subset.
    Uses per-dimension std ratio capped at 1.0, then averaged."""
    if len(selected) < 2 or len(full) < 2:
        return 0.0
    std_sel = np.std(selected, axis=0)
    std_full = np.std(full, axis=0)
    mask = std_full > 1e-9
    if mask.sum() == 0:
        return 0.0
    ratios = std_sel[mask] / std_full[mask]
    return float(np.mean(np.minimum(ratios, 1.0)))


def select_topk(pool, k):
    return pool[:k]


def select_lsr(pool, k):
    centered = pool - pool.mean(axis=0)
    v1, _ = power_iteration(centered, n_iterations=10)
    projections = centered @ v1
    sorted_idx = np.argsort(projections)
    n = len(sorted_idx)
    selected = [sorted_idx[int(i * n / k)] for i in range(k)]
    return pool[selected]


def select_mmr(pool, sims, k, lambda_param=0.5):
    selected = [0]
    for _ in range(k - 1):
        best_score = -np.inf
        best_idx = -1
        for j in range(len(pool)):
            if j in selected:
                continue
            redundancy = max(pool[j] @ pool[s] for s in selected)
            score = lambda_param * sims[j] - (1 - lambda_param) * redundancy
            if score > best_score:
                best_score = score
                best_idx = j
        if best_idx >= 0:
            selected.append(best_idx)
    return pool[selected]


def main():
    print("Loading embeddings...")
    embeddings = load_embeddings(DATA_PATH)
    N, d = embeddings.shape
    print(f"  {N} documents, {d} dimensions")

    # Threshold-based pooling: take all docs above tau
    THRESHOLD = 0.85
    MIN_POOL = 10       # need at least this many docs above tau to be interesting
    ks = [3, 5, 8, 10, 15, 20]
    MAX_QUERIES = 500

    print(f"Finding queries with >= {MIN_POOL} docs above tau={THRESHOLD}...")
    query_data = []
    for qi in range(N):
        if qi % 2000 == 0:
            print(f"  Scanning {qi}/{N}... ({len(query_data)} qualifying so far)")
        sims = embeddings[qi] @ embeddings.T
        sims[qi] = -1
        above_mask = sims >= THRESHOLD
        n_above = above_mask.sum()
        if n_above >= MIN_POOL:
            above_indices = np.where(above_mask)[0]
            # Sort by similarity descending
            order = np.argsort(-sims[above_indices])
            above_indices = above_indices[order]
            query_data.append((qi, above_indices, sims[above_indices]))
            if len(query_data) >= MAX_QUERIES:
                break

    print(f"  Found {len(query_data)} queries with >= {MIN_POOL} docs above tau={THRESHOLD}")

    # Report neighborhood stats
    pool_sizes = [len(idx) for _, idx, _ in query_data]
    all_min_sims = [float(pool_sims[-1]) for _, _, pool_sims in query_data]
    all_max_sims = [float(pool_sims[0]) for _, _, pool_sims in query_data]
    all_mean_sims = [float(pool_sims.mean()) for _, _, pool_sims in query_data]
    print(f"\n  Neighborhood stats:")
    print(f"    Pool size: mean={np.mean(pool_sizes):.1f}, min={np.min(pool_sizes)}, max={np.max(pool_sizes)}")
    print(f"    Min sim (last in pool):  mean={np.mean(all_min_sims):.4f}")
    print(f"    Max sim (1st in pool):   mean={np.mean(all_max_sims):.4f}")
    print(f"    Mean sim across pool:    mean={np.mean(all_mean_sims):.4f}")

    # Sweep k and compute variance coverage + timing
    vc_topk = {k: [] for k in ks}
    vc_lsr = {k: [] for k in ks}
    vc_mmr = {k: [] for k in ks}
    time_topk = {k: [] for k in ks}
    time_lsr = {k: [] for k in ks}
    time_mmr = {k: [] for k in ks}

    for idx, (qi, top_indices, pool_sims) in enumerate(query_data):
        if idx % 50 == 0:
            print(f"  Processing query {idx}/{len(query_data)}...")
        pool = embeddings[top_indices]

        for k in ks:
            if k > len(pool):
                continue

            t0 = time.perf_counter()
            topk_sel = select_topk(pool, k)
            time_topk[k].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            lsr_sel = select_lsr(pool, k)
            time_lsr[k].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            mmr_sel = select_mmr(pool, pool_sims, k)
            time_mmr[k].append(time.perf_counter() - t0)

            vc_topk[k].append(variance_coverage(topk_sel, pool))
            vc_lsr[k].append(variance_coverage(lsr_sel, pool))
            vc_mmr[k].append(variance_coverage(mmr_sel, pool))

    # Average across queries
    ks_plot = []
    mean_topk, mean_lsr, mean_mmr = [], [], []
    avg_time_topk, avg_time_lsr, avg_time_mmr = [], [], []
    for k in ks:
        if len(vc_topk[k]) > 0:
            ks_plot.append(k)
            mean_topk.append(np.mean(vc_topk[k]))
            mean_lsr.append(np.mean(vc_lsr[k]))
            mean_mmr.append(np.mean(vc_mmr[k]))
            avg_time_topk.append(np.mean(time_topk[k]) * 1e6)  # microseconds
            avg_time_lsr.append(np.mean(time_lsr[k]) * 1e6)
            avg_time_mmr.append(np.mean(time_mmr[k]) * 1e6)

    # Plot: two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: Variance Coverage
    ax1.plot(ks_plot, mean_lsr, 'o-', color='crimson', linewidth=2, markersize=7, label='LSR')
    ax1.plot(ks_plot, mean_mmr, 's-', color='orange', linewidth=2, markersize=7, label='MMR')
    ax1.plot(ks_plot, mean_topk, '^-', color='dodgerblue', linewidth=2, markersize=7, label='Top-k')
    ax1.set_xlabel('k (number of selected neighbors)')
    ax1.set_ylabel('Variance Coverage')
    ax1.set_title(f'Variance Coverage\n(Neo4j corpus, {len(query_data)} queries, tau={THRESHOLD})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Right: Latency
    ax2.plot(ks_plot, avg_time_lsr, 'o-', color='crimson', linewidth=2, markersize=7, label='LSR')
    ax2.plot(ks_plot, avg_time_mmr, 's-', color='orange', linewidth=2, markersize=7, label='MMR')
    ax2.plot(ks_plot, avg_time_topk, '^-', color='dodgerblue', linewidth=2, markersize=7, label='Top-k')
    ax2.set_xlabel('k (number of selected neighbors)')
    ax2.set_ylabel('Latency (μs)')
    ax2.set_title('Selection Latency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, 'variance_coverage.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved to {out_path}")

    # Print numbers
    print("\nVariance Coverage (mean across collapsed queries):")
    print(f"  {'k':>4}  {'Top-k':>8}  {'MMR':>8}  {'LSR':>8}")
    for i, k in enumerate(ks_plot):
        print(f"  {k:>4}  {mean_topk[i]:>8.4f}  {mean_mmr[i]:>8.4f}  {mean_lsr[i]:>8.4f}")

    print("\nLatency in us (mean across collapsed queries):")
    print(f"  {'k':>4}  {'Top-k':>8}  {'MMR':>8}  {'LSR':>8}  {'MMR/LSR':>8}")
    for i, k in enumerate(ks_plot):
        ratio = avg_time_mmr[i] / avg_time_lsr[i] if avg_time_lsr[i] > 0 else 0
        print(f"  {k:>4}  {avg_time_topk[i]:>8.1f}  {avg_time_mmr[i]:>8.1f}  {avg_time_lsr[i]:>8.1f}  {ratio:>7.1f}x")


if __name__ == '__main__':
    main()
