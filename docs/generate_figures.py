"""Generate all figures for the LSR paper using real Neo4j corpus embeddings.

Uses the 15,292-document Neo4j corpus with 768-dim embeddings.
Finds representative query neighborhoods exhibiting different levels
of density collapse and generates all paper figures from real data.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lsr import power_iteration

FIGS_DIR = os.path.join(os.path.dirname(__file__), "figs")
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'neo4j_embeddings.csv')
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


# ── Shared utilities ─────────────────────────────────────────────────────────

def load_embeddings(path):
    """Load Neo4j embeddings CSV directly into numpy (float32, unit-normalized)."""
    import pandas as pd
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    embeddings = df[emb_cols].values.astype(np.float32)
    del df
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return embeddings / norms


def compute_top_k_pcs(centered, k=10, n_iterations=10):
    """Compute top-k principal components via power iteration + deflation.

    Returns:
        vecs: (d, k) eigenvector matrix
        eigs: (k,) eigenvalues
        ratios: (k,) explained variance ratios
        total_var: total variance (trace of covariance)
    """
    n = len(centered)
    total_var = float(np.sum(centered ** 2) / n)
    if total_var < 1e-12:
        d = centered.shape[1]
        return np.zeros((d, 1)), np.array([0.0]), np.array([0.0]), 0.0

    k = min(k, n - 1)
    vecs, eigs = [], []
    residual = centered.copy()

    for _ in range(k):
        v, eigval = power_iteration(residual, n_iterations=n_iterations)
        vecs.append(v)
        eigs.append(eigval)
        proj = residual @ v
        residual = residual - np.outer(proj, v)

    eigs = np.array(eigs)
    ratios = eigs / total_var
    return np.column_stack(vecs), eigs, ratios, total_var


def avg_pairwise_similarity(pool):
    """Average pairwise cosine similarity within the pool (upper triangle)."""
    sim_matrix = pool @ pool.T
    n = len(pool)
    # Upper triangle, excluding diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(sim_matrix[mask].mean())


def find_representative_queries(embeddings, tau=0.85, min_pool=10):
    """Scan corpus to find queries with strong, moderate, and weak collapse.

    Collapse severity = average pairwise cosine similarity within
    the threshold neighborhood. Scans all documents, collects every
    qualifying neighborhood, then picks the best examples.

    Returns dict: {'strong': (qi, pool_idx, r1), 'moderate': ..., 'weak': ...}
    """
    N = len(embeddings)
    all_entries = []  # (qi, pool_idx, r1, avg_sim)

    for qi in range(N):
        if qi % 2000 == 0:
            print(f"    Scanning {qi}/{N}... ({len(all_entries)} qualifying)")
        sims = embeddings[qi] @ embeddings.T
        sims[qi] = -1
        above = np.where(sims >= tau)[0]
        if len(above) < min_pool:
            continue

        pool = embeddings[above]
        avg_sim = avg_pairwise_similarity(pool)

        centered = pool - pool.mean(axis=0)
        _, eigval = power_iteration(centered, n_iterations=10)
        total_var = float(np.sum(centered ** 2) / len(pool))
        r1 = eigval / total_var if total_var > 1e-12 else 0.0

        order = np.argsort(-sims[above])
        pool_idx = above[order]
        all_entries.append((qi, pool_idx, r1, avg_sim))

    print(f"    Found {len(all_entries)} qualifying neighborhoods")

    if len(all_entries) == 0:
        return {'strong': None, 'moderate': None, 'weak': None}

    # Sort by avg_sim and pick from different regions
    all_entries.sort(key=lambda x: x[3])  # ascending by avg_sim
    avg_sims = [e[3] for e in all_entries]
    print(f"    avg_sim range: [{avg_sims[0]:.4f}, {avg_sims[-1]:.4f}]")

    # Strong = highest avg_sim, Weak = lowest, Moderate = median
    strong_entry = all_entries[-1]
    weak_entry = all_entries[0]
    moderate_entry = all_entries[len(all_entries) // 2]

    def entry_to_result(e):
        return (e[0], e[1], e[2])  # (qi, pool_idx, r1)

    found = {
        'strong': entry_to_result(strong_entry),
        'moderate': entry_to_result(moderate_entry),
        'weak': entry_to_result(weak_entry),
    }

    print(f"    strong:   qi={strong_entry[0]}, pool={len(strong_entry[1])}, "
          f"r1={strong_entry[2]:.3f}, avg_sim={strong_entry[3]:.4f}")
    print(f"    moderate: qi={moderate_entry[0]}, pool={len(moderate_entry[1])}, "
          f"r1={moderate_entry[2]:.3f}, avg_sim={moderate_entry[3]:.4f}")
    print(f"    weak:     qi={weak_entry[0]}, pool={len(weak_entry[1])}, "
          f"r1={weak_entry[2]:.3f}, avg_sim={weak_entry[3]:.4f}")

    return found


def select_lsr_indices(pool, k):
    """LSR: quantile sampling along PC1. Returns indices into pool."""
    centered = pool - pool.mean(axis=0)
    v1, _ = power_iteration(centered, n_iterations=10)
    projections = centered @ v1
    sorted_idx = np.argsort(projections)
    n = len(sorted_idx)
    return np.array([sorted_idx[int(i * n / k)] for i in range(k)])


def select_mmr_indices(pool, sims, k, lambda_param=0.5):
    """MMR: greedy diversity selection. Returns indices into pool."""
    selected = [0]  # start with highest-similarity doc (pool sorted by sim)
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
    return np.array(selected)


def variance_coverage(selected, full):
    """Per-dimension std ratio capped at 1.0, averaged."""
    if len(selected) < 2 or len(full) < 2:
        return 0.0
    std_sel = np.std(selected, axis=0)
    std_full = np.std(full, axis=0)
    mask = std_full > 1e-9
    if mask.sum() == 0:
        return 0.0
    ratios = std_sel[mask] / std_full[mask]
    return float(np.mean(np.minimum(ratios, 1.0)))


# ── Figure 1: density_collapse ────────────────────────────────────────────────
def fig_density_collapse(pool, pcs, ratios):
    """2D scatter (PC1 vs PC2) + eigenvalue spectrum from real neighborhood."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    centered = pool - pool.mean(axis=0)

    # Project onto PC1, PC2
    proj_2d = centered @ pcs[:, :2]

    # Left: 2D scatter
    ax = axes[0]
    ax.scatter(proj_2d[:, 0], proj_2d[:, 1], s=15, alpha=0.5, c='steelblue', edgecolors='none')
    ax.set_xlabel('PC1 (dominant)')
    ax.set_ylabel('PC2')
    ax.set_title('Threshold Neighborhood\n(projected to PC1\u2013PC2)')

    # Covariance ellipse
    cov = np.cov(proj_2d.T)
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ell = Ellipse(xy=(proj_2d[:, 0].mean(), proj_2d[:, 1].mean()),
                  width=4*np.sqrt(vals[1]), height=4*np.sqrt(vals[0]),
                  angle=angle, fill=False, color='red', linewidth=2, linestyle='--')
    ax.add_patch(ell)
    ax.set_aspect('equal')

    # Right: eigenvalue spectrum
    ax = axes[1]
    n_bars = min(10, len(ratios))
    ax.bar(range(1, n_bars + 1), ratios[:n_bars], color='steelblue', edgecolor='navy')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(f'Eigenvalue Spectrum\n(PC1 = {ratios[0]:.0%} of variance)')
    ax.set_xticks(range(1, n_bars + 1))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'density_collapse.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  density_collapse.pdf")


# ── Figure 2: pc1_projection ─────────────────────────────────────────────────
def fig_pc1_projection(proj_1d):
    """Histogram of PC1 projections with quantile sampling boundaries."""
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.hist(proj_1d, bins=40, color='steelblue', edgecolor='navy', alpha=0.8)

    k = 8
    quantiles = np.linspace(0, 1, k + 1)
    sorted_proj = np.sort(proj_1d)
    for i in range(k):
        idx = int(quantiles[i] * (len(sorted_proj) - 1))
        val = sorted_proj[idx]
        ax.axvline(val, color='red', linewidth=1.5, alpha=0.7, linestyle='--')

    ax.set_xlabel('Projection onto PC1')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of PC1 Projections with LSR Quantile Boundaries (k=8)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'pc1_projection.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  pc1_projection.pdf")


# ── Figure 3: method_comparison ───────────────────────────────────────────────
def fig_method_comparison(pool, proj_2d, proj_1d, pool_sims):
    """3-panel scatter comparing Top-k, MMR, LSR selections in PC1-PC2 space."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    k = 8

    for ax in axes:
        ax.scatter(proj_2d[:, 0], proj_2d[:, 1], s=12, alpha=0.25, c='lightgray', edgecolors='none')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_aspect('equal')

    # Top-k: highest cosine similarity (pool already sorted by sim descending)
    topk_idx = np.arange(k)
    axes[0].scatter(proj_2d[topk_idx, 0], proj_2d[topk_idx, 1], s=60, c='dodgerblue',
                    edgecolors='navy', zorder=5, label='Top-k selected')
    axes[0].set_title('Top-k Cosine')
    axes[0].legend(loc='upper right', fontsize=9)

    # MMR
    mmr_idx = select_mmr_indices(pool, pool_sims, k)
    axes[1].scatter(proj_2d[mmr_idx, 0], proj_2d[mmr_idx, 1], s=60, c='orange',
                    edgecolors='darkorange', zorder=5, label='MMR selected')
    axes[1].set_title('MMR ($\\lambda=0.5$)')
    axes[1].legend(loc='upper right', fontsize=9)

    # LSR: quantile sampling along PC1
    lsr_idx = select_lsr_indices(pool, k)
    axes[2].scatter(proj_2d[lsr_idx, 0], proj_2d[lsr_idx, 1], s=60, c='crimson',
                    edgecolors='darkred', zorder=5, label='LSR selected')
    axes[2].set_title('LSR (PC1 quantile)')
    axes[2].legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'method_comparison.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  method_comparison.pdf")


# ── Figure 4: variance_coverage ───────────────────────────────────────────────
def fig_variance_coverage(embeddings):
    """Variance coverage vs k, averaged over many queries. Dual-panel with latency."""
    THRESHOLD = 0.85
    MIN_POOL = 10
    ks = [3, 5, 8, 10, 15, 20]
    MAX_QUERIES = 500
    N = len(embeddings)

    print("  Scanning for queries (fig4)...")
    query_data = []
    for qi in range(N):
        if qi % 2000 == 0:
            print(f"    {qi}/{N}... ({len(query_data)} qualifying)")
        sims = embeddings[qi] @ embeddings.T
        sims[qi] = -1
        above_mask = sims >= THRESHOLD
        n_above = above_mask.sum()
        if n_above >= MIN_POOL:
            above_indices = np.where(above_mask)[0]
            order = np.argsort(-sims[above_indices])
            above_indices = above_indices[order]
            query_data.append((qi, above_indices, sims[above_indices]))
            if len(query_data) >= MAX_QUERIES:
                break

    print(f"    Found {len(query_data)} queries")

    vc_topk = {k: [] for k in ks}
    vc_lsr = {k: [] for k in ks}
    vc_mmr = {k: [] for k in ks}
    time_topk = {k: [] for k in ks}
    time_lsr = {k: [] for k in ks}
    time_mmr = {k: [] for k in ks}

    for idx, (qi, top_indices, pool_sims) in enumerate(query_data):
        if idx % 100 == 0:
            print(f"    Processing {idx}/{len(query_data)}...")
        pool = embeddings[top_indices]

        # LSR: compute PCA once per neighborhood (amortized cost)
        centered = pool - pool.mean(axis=0)
        v1, _ = power_iteration(centered, n_iterations=10)
        projections = centered @ v1
        sorted_idx = np.argsort(projections)
        n_pool = len(sorted_idx)

        for k in ks:
            if k > len(pool):
                continue

            t0 = time.perf_counter()
            topk_sel = pool[:k]
            time_topk[k].append(time.perf_counter() - t0)

            # LSR selection only (PCA already computed)
            t0 = time.perf_counter()
            lsr_local = np.array([sorted_idx[int(i * n_pool / k)] for i in range(k)])
            lsr_sel = pool[lsr_local]
            time_lsr[k].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            mmr_idx = select_mmr_indices(pool, pool_sims, k)
            mmr_sel = pool[mmr_idx]
            time_mmr[k].append(time.perf_counter() - t0)

            vc_topk[k].append(variance_coverage(topk_sel, pool))
            vc_lsr[k].append(variance_coverage(lsr_sel, pool))
            vc_mmr[k].append(variance_coverage(mmr_sel, pool))

    ks_plot = []
    mean_topk, mean_lsr, mean_mmr = [], [], []
    avg_time_topk, avg_time_lsr, avg_time_mmr = [], [], []
    for k in ks:
        if len(vc_topk[k]) > 0:
            ks_plot.append(k)
            mean_topk.append(np.mean(vc_topk[k]))
            mean_lsr.append(np.mean(vc_lsr[k]))
            mean_mmr.append(np.mean(vc_mmr[k]))
            avg_time_topk.append(np.mean(time_topk[k]) * 1e6)
            avg_time_lsr.append(np.mean(time_lsr[k]) * 1e6)
            avg_time_mmr.append(np.mean(time_mmr[k]) * 1e6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(ks_plot, mean_lsr, 'o-', color='crimson', linewidth=2, markersize=7, label='LSR')
    ax1.plot(ks_plot, mean_mmr, 's-', color='orange', linewidth=2, markersize=7, label='MMR')
    ax1.plot(ks_plot, mean_topk, '^-', color='dodgerblue', linewidth=2, markersize=7, label='Top-k')
    ax1.set_xlabel('k (number of selected neighbors)')
    ax1.set_ylabel('Variance Coverage')
    ax1.set_title(f'Variance Coverage\n(Neo4j corpus, {len(query_data)} queries, $\\tau$={THRESHOLD})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.plot(ks_plot, avg_time_lsr, 'o-', color='crimson', linewidth=2, markersize=7, label='LSR')
    ax2.plot(ks_plot, avg_time_mmr, 's-', color='orange', linewidth=2, markersize=7, label='MMR')
    ax2.plot(ks_plot, avg_time_topk, '^-', color='dodgerblue', linewidth=2, markersize=7, label='Top-k')
    ax2.set_xlabel('k (number of selected neighbors)')
    ax2.set_ylabel('Latency (us)')
    ax2.set_title('Selection Latency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'variance_coverage.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  variance_coverage.pdf")

    print("\n  Variance Coverage (mean):")
    print(f"    {'k':>4}  {'Top-k':>8}  {'MMR':>8}  {'LSR':>8}")
    for i, k in enumerate(ks_plot):
        print(f"    {k:>4}  {mean_topk[i]:>8.4f}  {mean_mmr[i]:>8.4f}  {mean_lsr[i]:>8.4f}")

    print(f"\n  Latency (us):")
    print(f"    {'k':>4}  {'Top-k':>8}  {'MMR':>8}  {'LSR':>8}  {'MMR/LSR':>8}")
    for i, k in enumerate(ks_plot):
        ratio = avg_time_mmr[i] / avg_time_lsr[i] if avg_time_lsr[i] > 0 else 0
        print(f"    {k:>4}  {avg_time_topk[i]:>8.1f}  {avg_time_mmr[i]:>8.1f}  {avg_time_lsr[i]:>8.1f}  {ratio:>7.1f}x")


# ── Figure 5: eigen_spectrum ──────────────────────────────────────────────────
def fig_eigen_spectrum(queries, embeddings):
    """3-panel eigenvalue spectra for strong/moderate/weak collapse neighborhoods."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    configs = [
        ('strong', 'Strong Collapse', 'crimson'),
        ('moderate', 'Moderate Collapse', 'orange'),
        ('weak', 'Weak Collapse', 'steelblue'),
    ]

    for ax, (key, label, color) in zip(axes, configs):
        if queries[key] is None:
            ax.set_title(f'{label}\n(no example found)')
            continue

        qi, pool_idx, r1 = queries[key]
        pool = embeddings[pool_idx]
        centered = pool - pool.mean(axis=0)
        _, _, ratios, _ = compute_top_k_pcs(centered, k=10)

        n_bars = min(10, len(ratios))
        ax.bar(range(1, n_bars + 1), ratios[:n_bars], color=color, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title(f'{label}\n($r_1 = {r1*100:.0f}\\%$)')
        ax.set_xticks(range(1, n_bars + 1))
        ax.set_ylim(0, 1.0)
        ax.axhline(0.70, color='red', linestyle=':', alpha=0.5, label='70% threshold')
        ax.axhline(0.50, color='orange', linestyle=':', alpha=0.5, label='50% threshold')
        if ax == axes[0]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'eigen_spectrum.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  eigen_spectrum.pdf")


# ── Figure 6: t-SNE visualizations ───────────────────────────────────────────
def fig_tsne(pool, pool_sims):
    """t-SNE of real neighborhood with LSR vs Top-k selections highlighted."""
    # Subsample to ~150 points for clean t-SNE
    max_pts = 150
    if len(pool) > max_pts:
        pool = pool[:max_pts]
        pool_sims = pool_sims[:max_pts]

    perplexity = min(30, len(pool) // 3)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(pool)

    k = 8
    topk_idx = np.arange(k)  # pool is sorted by similarity, so first k = top-k
    lsr_idx = select_lsr_indices(pool, k)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: neighborhood structure
    ax = axes[0]
    ax.scatter(embedded[:, 0], embedded[:, 1], s=25, alpha=0.5, c='steelblue', edgecolors='none')
    ax.set_title('Threshold Neighborhood (t-SNE)')
    ax.set_xticks([]); ax.set_yticks([])

    # Right: LSR vs Top-k
    ax = axes[1]
    ax.scatter(embedded[:, 0], embedded[:, 1], s=15, alpha=0.2, c='lightgray', edgecolors='none')
    ax.scatter(embedded[topk_idx, 0], embedded[topk_idx, 1], s=80, c='dodgerblue',
               edgecolors='navy', zorder=5, label='Top-k', marker='^')
    ax.scatter(embedded[lsr_idx, 0], embedded[lsr_idx, 1], s=80, c='crimson',
               edgecolors='darkred', zorder=5, label='LSR', marker='o')
    ax.set_title('LSR vs. Top-k Selection (t-SNE)')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'tsne_combined.pdf'), bbox_inches='tight')

    # Individual panels
    for i, name in enumerate(['tsne_neighborhood.pdf', 'tsne_lsr_vs_topk.pdf']):
        fig_single, ax_single = plt.subplots(figsize=(5, 4.5))
        if i == 0:
            ax_single.scatter(embedded[:, 0], embedded[:, 1], s=25, alpha=0.5, c='steelblue', edgecolors='none')
            ax_single.set_title('Threshold Neighborhood (t-SNE)')
        else:
            ax_single.scatter(embedded[:, 0], embedded[:, 1], s=15, alpha=0.2, c='lightgray', edgecolors='none')
            ax_single.scatter(embedded[topk_idx, 0], embedded[topk_idx, 1], s=80, c='dodgerblue',
                       edgecolors='navy', zorder=5, label='Top-k', marker='^')
            ax_single.scatter(embedded[lsr_idx, 0], embedded[lsr_idx, 1], s=80, c='crimson',
                       edgecolors='darkred', zorder=5, label='LSR', marker='o')
            ax_single.set_title('LSR vs. Top-k Selection (t-SNE)')
            ax_single.legend()
        ax_single.set_xticks([]); ax_single.set_yticks([])
        fig_single.savefig(os.path.join(FIGS_DIR, name), bbox_inches='tight')
        plt.close(fig_single)

    plt.close(fig)
    print("  tsne_neighborhood.pdf")
    print("  tsne_lsr_vs_topk.pdf")


# ── Generate all ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading Neo4j corpus embeddings...")
    embeddings = load_embeddings(DATA_PATH)
    N, d = embeddings.shape
    print(f"  {N} documents, {d} dimensions")

    print("\nFinding representative query neighborhoods...")
    queries = find_representative_queries(embeddings, tau=0.85, min_pool=10)
    for key in ['strong', 'moderate', 'weak']:
        if queries[key] is not None:
            qi, pool_idx, r1 = queries[key]
            print(f"  {key}: query={qi}, pool={len(pool_idx)}, r1={r1:.3f}")
        else:
            print(f"  {key}: NOT FOUND")

    # Precompute shared data for the strong-collapse query (figs 1, 2, 3, 6)
    if queries['strong'] is None:
        print("ERROR: Could not find any qualifying neighborhood. Exiting.")
        sys.exit(1)
    strong_qi, strong_pool_idx, strong_r1 = queries['strong']
    strong_pool = embeddings[strong_pool_idx]
    strong_centered = strong_pool - strong_pool.mean(axis=0)
    strong_pcs, strong_eigs, strong_ratios, _ = compute_top_k_pcs(strong_centered, k=10)
    strong_proj_2d = strong_centered @ strong_pcs[:, :2]
    strong_proj_1d = strong_centered @ strong_pcs[:, 0]
    strong_sims = strong_pool @ embeddings[strong_qi]

    print("\nGenerating figures...")
    fig_density_collapse(strong_pool, strong_pcs, strong_ratios)
    fig_pc1_projection(strong_proj_1d)
    fig_method_comparison(strong_pool, strong_proj_2d, strong_proj_1d, strong_sims)
    fig_variance_coverage(embeddings)
    fig_eigen_spectrum(queries, embeddings)
    fig_tsne(strong_pool, strong_sims)
    print("\nDone! All figures saved to", FIGS_DIR)
