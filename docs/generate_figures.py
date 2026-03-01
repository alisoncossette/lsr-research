"""Generate all figures for the LSR paper."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

np.random.seed(42)
FIGS_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

# Common style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# ── Shared synthetic data ─────────────────────────────────────────────────────
def make_collapsed_neighborhood(n=200, d=256, collapse_strength=0.85):
    """Create a neighborhood with density collapse along one direction."""
    # Dominant direction gets most variance
    variances = np.zeros(d)
    variances[0] = collapse_strength
    variances[1] = (1 - collapse_strength) * 0.6
    variances[2:] = (1 - collapse_strength) * 0.4 / (d - 2)

    # Generate points
    points = np.random.randn(n, d) * np.sqrt(variances)

    # Add a query-centered offset
    query = np.random.randn(d) * 0.1
    points += query
    return points, query

points, query = make_collapsed_neighborhood()
pca = PCA(n_components=3)
proj_3d = pca.fit_transform(points)
proj_1d = proj_3d[:, 0]  # PC1 projections


# ── Figure 1: density_collapse ────────────────────────────────────────────────
def fig_density_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: 2D scatter showing collapse
    ax = axes[0]
    ax.scatter(proj_3d[:, 0], proj_3d[:, 1], s=15, alpha=0.5, c='steelblue', edgecolors='none')
    ax.set_xlabel('PC1 (dominant)')
    ax.set_ylabel('PC2')
    ax.set_title('Threshold Neighborhood\n(projected to PC1–PC2)')

    # Draw ellipse showing collapse
    cov = np.cov(proj_3d[:, :2].T)
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ell = Ellipse(xy=(proj_3d[:, 0].mean(), proj_3d[:, 1].mean()),
                  width=4*np.sqrt(vals[1]), height=4*np.sqrt(vals[0]),
                  angle=angle, fill=False, color='red', linewidth=2, linestyle='--')
    ax.add_patch(ell)
    ax.set_aspect('equal')

    # Right: eigenvalue spectrum showing collapse
    ax = axes[1]
    explained = pca.explained_variance_ratio_
    ax.bar(range(1, min(11, len(explained)+1)), explained[:10], color='steelblue', edgecolor='navy')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(f'Eigenvalue Spectrum\n(PC1 = {explained[0]:.0%} of variance)')
    ax.set_xticks(range(1, 11))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'density_collapse.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  density_collapse.pdf")

# ── Figure 2: pc1_projection ─────────────────────────────────────────────────
def fig_pc1_projection():
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.hist(proj_1d, bins=40, color='steelblue', edgecolor='navy', alpha=0.8)

    # Mark quantile sampling points
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
def fig_method_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    k = 8

    # Compute similarities to query for top-k
    query_proj = pca.transform(query.reshape(1, -1))[0]
    dists = np.linalg.norm(proj_3d - query_proj, axis=1)

    for ax in axes:
        ax.scatter(proj_3d[:, 0], proj_3d[:, 1], s=12, alpha=0.25, c='lightgray', edgecolors='none')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_aspect('equal')

    # Top-k: closest to query
    topk_idx = np.argsort(dists)[:k]
    axes[0].scatter(proj_3d[topk_idx, 0], proj_3d[topk_idx, 1], s=60, c='dodgerblue',
                    edgecolors='navy', zorder=5, label='Top-k selected')
    axes[0].set_title('Top-k Cosine')
    axes[0].legend(loc='upper right', fontsize=9)

    # MMR: greedy diverse selection
    selected_mmr = [np.argmin(dists)]
    for _ in range(k - 1):
        best_score = -np.inf
        best_idx = -1
        for j in range(len(points)):
            if j in selected_mmr:
                continue
            relevance = -dists[j]
            redundancy = max(1 - np.linalg.norm(proj_3d[j] - proj_3d[s]) for s in selected_mmr)
            score = 0.5 * relevance - 0.5 * redundancy
            if score > best_score:
                best_score = score
                best_idx = j
        selected_mmr.append(best_idx)

    axes[1].scatter(proj_3d[selected_mmr, 0], proj_3d[selected_mmr, 1], s=60, c='orange',
                    edgecolors='darkorange', zorder=5, label='MMR selected')
    axes[1].set_title('MMR ($\\lambda=0.5$)')
    axes[1].legend(loc='upper right', fontsize=9)

    # LSR: quantile sampling along PC1
    sorted_idx = np.argsort(proj_1d)
    lsr_idx = [sorted_idx[int(i * len(sorted_idx) / k)] for i in range(k)]
    axes[2].scatter(proj_3d[lsr_idx, 0], proj_3d[lsr_idx, 1], s=60, c='crimson',
                    edgecolors='darkred', zorder=5, label='LSR selected')
    axes[2].set_title('LSR (PC1 quantile)')
    axes[2].legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'method_comparison.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  method_comparison.pdf")

# ── Figure 4: variance_coverage ───────────────────────────────────────────────
def fig_variance_coverage():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ks = [3, 5, 8, 10, 15, 20]

    def variance_coverage(selected_pts, all_pts):
        """Fraction of total variance captured by selected subset."""
        total_var = np.var(all_pts, axis=0).sum()
        selected_var = np.var(selected_pts, axis=0).sum()
        return selected_var / total_var

    dists = np.linalg.norm(points - query, axis=1)
    sorted_idx_pc1 = np.argsort(proj_1d)

    vc_topk, vc_mmr, vc_lsr = [], [], []

    for k in ks:
        # Top-k
        topk_idx = np.argsort(dists)[:k]
        vc_topk.append(variance_coverage(points[topk_idx], points))

        # LSR
        lsr_idx = [sorted_idx_pc1[int(i * len(sorted_idx_pc1) / k)] for i in range(k)]
        vc_lsr.append(variance_coverage(points[lsr_idx], points))

        # MMR (simplified)
        selected = [np.argmin(dists)]
        for _ in range(k - 1):
            best_s, best_j = -np.inf, -1
            for j in range(len(points)):
                if j in selected:
                    continue
                rel = -dists[j]
                red = max(1 - np.linalg.norm(points[j] - points[s]) for s in selected)
                s_val = 0.5 * rel - 0.5 * red
                if s_val > best_s:
                    best_s, best_j = s_val, j
            selected.append(best_j)
        vc_mmr.append(variance_coverage(points[selected], points))

    ax.plot(ks, vc_lsr, 'o-', color='crimson', linewidth=2, markersize=7, label='LSR')
    ax.plot(ks, vc_mmr, 's-', color='orange', linewidth=2, markersize=7, label='MMR')
    ax.plot(ks, vc_topk, '^-', color='dodgerblue', linewidth=2, markersize=7, label='Top-k')

    ax.set_xlabel('k (number of selected neighbors)')
    ax.set_ylabel('Variance Coverage')
    ax.set_title('Variance Coverage vs. k')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'variance_coverage.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  variance_coverage.pdf")

# ── Figure 5: eigen_spectrum ──────────────────────────────────────────────────
def fig_eigen_spectrum():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    configs = [
        ("Strong Collapse\n($r_1 = 85\\%$)", 0.85),
        ("Moderate Collapse\n($r_1 = 60\\%$)", 0.60),
        ("Weak Collapse\n($r_1 = 35\\%$)", 0.35),
    ]
    colors = ['crimson', 'orange', 'steelblue']

    for ax, (title, strength), color in zip(axes, configs, colors):
        pts, _ = make_collapsed_neighborhood(collapse_strength=strength)
        pca_local = PCA(n_components=min(10, pts.shape[1]))
        pca_local.fit(pts)
        evr = pca_local.explained_variance_ratio_[:10]

        ax.bar(range(1, len(evr)+1), evr, color=color, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title(title)
        ax.set_xticks(range(1, 11))
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
def fig_tsne():
    # Use a smaller neighborhood for t-SNE clarity
    pts_small, q_small = make_collapsed_neighborhood(n=100, d=256)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedded = tsne.fit_transform(pts_small)

    k = 8
    dists = np.linalg.norm(pts_small - q_small, axis=1)
    topk_idx = np.argsort(dists)[:k]

    pca_local = PCA(n_components=1)
    proj = pca_local.fit_transform(pts_small).ravel()
    sorted_idx = np.argsort(proj)
    lsr_idx = [sorted_idx[int(i * len(sorted_idx) / k)] for i in range(k)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: neighborhood structure
    ax = axes[0]
    ax.scatter(embedded[:, 0], embedded[:, 1], s=20, alpha=0.4, c='lightgray', edgecolors='none')
    ax.scatter(embedded[:, 0], embedded[:, 1], s=20, alpha=0.4, c='steelblue', edgecolors='none')
    ax.set_title('Threshold Neighborhood (t-SNE)')
    ax.set_xticks([]); ax.set_yticks([])

    # Right: LSR vs top-k
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
    # Save as two separate files as the paper expects
    fig.savefig(os.path.join(FIGS_DIR, 'tsne_combined.pdf'), bbox_inches='tight')

    # Also save individual panels
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
    print("Generating figures...")
    fig_density_collapse()
    fig_pc1_projection()
    fig_method_comparison()
    fig_variance_coverage()
    fig_eigen_spectrum()
    fig_tsne()
    print("Done! All figures saved to", FIGS_DIR)