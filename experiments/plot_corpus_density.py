"""
Publication-quality t-SNE of the Neo4j corpus colored by local density.

Shows that real embedding spaces contain naturally occurring dense regions
where density collapse is expected. These are the regions where LSR matters.

Output: docs/figs/corpus_density.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import os
import sys
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lsr import power_iteration


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


def compute_local_r1(embeddings, sim_matrix, n_pool=50):
    """Compute r1 for each document's top-N neighborhood."""
    N = len(embeddings)
    r1_values = np.zeros(N)
    for qi in range(N):
        if qi % 2000 == 0:
            print(f"  Computing r1: {qi}/{N}")
        sims = sim_matrix[qi].copy()
        sims[qi] = -1
        top_idx = np.argsort(-sims)[:n_pool]
        pool = embeddings[top_idx]
        if len(pool) < 3:
            continue
        centered = pool - pool.mean(axis=0)
        _, eig1 = power_iteration(centered, n_iterations=10)
        total_var = float(np.sum(centered ** 2) / len(pool))
        r1_values[qi] = eig1 / total_var if total_var > 1e-12 else 0.0
    return r1_values


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'neo4j_embeddings.csv')
    print(f"Loading embeddings from {data_path}...")
    embeddings = load_neo4j_embeddings(data_path)
    N, d = embeddings.shape
    print(f"Loaded {N} documents, {d} dimensions")

    # --- t-SNE ---
    print(f"\nRunning t-SNE on {N} points ({d}d)... this takes a few minutes.")
    t0 = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=50,
        random_state=42,
        max_iter=1000,
        verbose=1
    )
    coords = tsne.fit_transform(embeddings)
    print(f"t-SNE complete in {time.time() - t0:.0f}s")

    # --- Local density via KDE on the 2D t-SNE coordinates ---
    print("Computing 2D density via KDE...")
    xy = coords.T
    kde = gaussian_kde(xy, bw_method=0.05)
    density = kde(xy)
    # Log-scale for better visual contrast
    log_density = np.log1p(density)

    # --- Compute r1 for each point ---
    print("Computing similarity matrix...")
    sim_matrix = embeddings @ embeddings.T
    print("Computing local r1 values...")
    r1_values = compute_local_r1(embeddings, sim_matrix, n_pool=50)

    # --- Figure: two panels ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Density
    ax1 = axes[0]
    order = np.argsort(log_density)  # plot sparse points first, dense on top
    sc1 = ax1.scatter(
        coords[order, 0], coords[order, 1],
        c=log_density[order],
        cmap='inferno',
        s=1.5,
        alpha=0.6,
        rasterized=True
    )
    ax1.set_title('Local Embedding Density', fontsize=13, fontweight='bold')
    ax1.set_xlabel('t-SNE 1', fontsize=10)
    ax1.set_ylabel('t-SNE 2', fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])
    cb1 = plt.colorbar(sc1, ax=ax1, shrink=0.8, pad=0.02)
    cb1.set_label('log density', fontsize=9)

    # Panel 2: r1 (collapse severity)
    ax2 = axes[1]
    # Use a diverging colormap: blue=low r1, red=high r1
    order2 = np.argsort(r1_values)  # plot low-r1 first, high-r1 on top
    sc2 = ax2.scatter(
        coords[order2, 0], coords[order2, 1],
        c=r1_values[order2],
        cmap='RdYlBu_r',
        s=1.5,
        alpha=0.6,
        vmin=0.0,
        vmax=0.6,
        rasterized=True
    )
    ax2.set_title('Neighborhood Collapse ($r_1$)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('t-SNE 1', fontsize=10)
    ax2.set_ylabel('t-SNE 2', fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    cb2 = plt.colorbar(sc2, ax=ax2, shrink=0.8, pad=0.02)
    cb2.set_label('$r_1$ (PC1 variance ratio)', fontsize=9)

    plt.tight_layout(w_pad=2)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figs', 'corpus_density.pdf')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved to {out_path}")

    # Also save a PNG for quick preview
    png_path = out_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"Preview: {png_path}")

    # Stats
    print(f"\n--- Density/Collapse Summary ---")
    print(f"  r1 > 0.30 (collapsed):       {np.mean(r1_values > 0.30)*100:.1f}%")
    print(f"  r1 > 0.50 (strong collapse):  {np.mean(r1_values > 0.50)*100:.1f}%")
    print(f"  Median r1:                     {np.median(r1_values):.3f}")
    print(f"  Mean r1:                       {np.mean(r1_values):.3f}")


if __name__ == '__main__':
    main()
