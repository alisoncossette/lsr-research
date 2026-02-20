"""
Create interactive 3D visualization with community detection clustering.

Usage:
    python experiments/plot_clusters_3d.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import hdbscan
from pathlib import Path

def plot_clusters_3d(
    input_file: str = "data/neo4j_embeddings.csv",
    output_file: str = "data/clusters_3d_interactive.html",
    sample_size: int = None,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    tsne_perplexity: int = 30,
    random_state: int = 42
):
    """
    Perform community detection clustering and create 3D visualization.

    Args:
        input_file: Path to CSV with embeddings
        output_file: Path to save interactive HTML
        sample_size: Number of samples to use (None for all)
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        tsne_perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
    """
    print(f"\n{'='*70}")
    print("3D CLUSTERING VISUALIZATION WITH COMMUNITY DETECTION")
    print(f"{'='*70}")

    # Load embeddings
    print(f"\n[1] Loading embeddings from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"    [OK] Loaded {len(df)} documents")

    # Extract embedding columns
    emb_cols = [col for col in df.columns if col.startswith('emb_')]
    embeddings = df[emb_cols].values
    print(f"    - Embedding dimension: {len(emb_cols)}")

    # Sample if needed
    if sample_size and len(df) > sample_size:
        print(f"\n[2] Sampling {sample_size} documents...")
        indices = np.random.choice(len(df), sample_size, replace=False)
        embeddings = embeddings[indices]
        df_sample = df.iloc[indices].copy()
        print(f"    [OK] Sampled to {len(embeddings)} documents")
    else:
        df_sample = df.copy()
        print(f"\n[2] Using all {len(df)} documents")

    # Perform HDBSCAN clustering on high-dimensional embeddings
    print(f"\n[3] Running HDBSCAN clustering...")
    print(f"    - min_cluster_size: {min_cluster_size}")
    print(f"    - min_samples: {min_samples}")
    print(f"    This may take several minutes...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    # Get cluster statistics
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters >= 0])
    n_noise = np.sum(cluster_labels == -1)

    print(f"    [OK] Clustering complete")
    print(f"    - Number of clusters found: {n_clusters}")
    print(f"    - Number of noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")

    # Print cluster sizes
    for cluster_id in sorted(unique_clusters):
        if cluster_id >= 0:
            count = np.sum(cluster_labels == cluster_id)
            print(f"    - Cluster {cluster_id}: {count} documents ({count/len(cluster_labels)*100:.1f}%)")

    # Run t-SNE in 3D for visualization
    print(f"\n[4] Running 3D t-SNE for visualization (perplexity={tsne_perplexity})...")
    print(f"    This may take several minutes...")
    tsne = TSNE(
        n_components=3,
        perplexity=tsne_perplexity,
        random_state=random_state,
        verbose=1
    )
    embeddings_3d = tsne.fit_transform(embeddings)
    print(f"    [OK] t-SNE complete")

    # Add results to dataframe
    df_sample['cluster'] = cluster_labels
    df_sample['tsne_x'] = embeddings_3d[:, 0]
    df_sample['tsne_y'] = embeddings_3d[:, 1]
    df_sample['tsne_z'] = embeddings_3d[:, 2]

    # Create hover text with truncated document text
    df_sample['hover_text'] = df_sample.apply(
        lambda row: f"Cluster {row['cluster']}<br>" +
                   (row['text'][:200] + '...' if len(str(row['text'])) > 200 else str(row['text'])),
        axis=1
    )

    # Create interactive 3D scatter plot with clusters
    print(f"\n[5] Creating interactive 3D visualization...")

    # Create separate traces for each cluster
    traces = []

    # Color palette for clusters
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Plot noise points first (in gray)
    if n_noise > 0:
        noise_data = df_sample[df_sample['cluster'] == -1]
        traces.append(go.Scatter3d(
            x=noise_data['tsne_x'],
            y=noise_data['tsne_y'],
            z=noise_data['tsne_z'],
            mode='markers',
            marker=dict(
                size=2,
                color='lightgray',
                opacity=0.3
            ),
            text=noise_data['hover_text'],
            hovertemplate='<b>Noise Point</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<br>' +
                         '<br>%{text}<extra></extra>',
            name='Noise',
            showlegend=True
        ))

    # Plot each cluster with unique color
    for i, cluster_id in enumerate(sorted(unique_clusters[unique_clusters >= 0])):
        cluster_data = df_sample[df_sample['cluster'] == cluster_id]
        color = colors[i % len(colors)]

        traces.append(go.Scatter3d(
            x=cluster_data['tsne_x'],
            y=cluster_data['tsne_y'],
            z=cluster_data['tsne_z'],
            mode='markers',
            marker=dict(
                size=4,
                color=color,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=cluster_data['hover_text'],
            hovertemplate='<b>Cluster ' + str(cluster_id) + '</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<br>' +
                         '<br>%{text}<extra></extra>',
            name=f'Cluster {cluster_id}',
            showlegend=True
        ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(
            text=f'3D Clustering Visualization with HDBSCAN Community Detection<br>' +
                 f'<sub>({len(embeddings_3d)} documents, {n_clusters} clusters, {len(emb_cols)}-dimensional embeddings)</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            zaxis_title='t-SNE Component 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1400,
        height=900,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Save interactive HTML
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"    [OK] Saved interactive plot to {output_path}")

    # Print summary
    print(f"\n[6] Summary")
    print(f"    - Documents visualized: {len(embeddings_3d)}")
    print(f"    - Embedding dimension: {len(emb_cols)}")
    print(f"    - Clusters found: {n_clusters}")
    print(f"    - Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
    print(f"    - Output file: {output_path}")
    print(f"    - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print(f"\n[7] Usage")
    print(f"    Open {output_path} in your web browser to explore:")
    print(f"    - Click and drag to rotate the 3D view")
    print(f"    - Scroll to zoom in/out")
    print(f"    - Hover over points to see document text and cluster")
    print(f"    - Click legend items to toggle cluster visibility")
    print(f"    - Use toolbar to pan, zoom, reset view")

    print(f"\n{'='*70}")
    print(f"Clustering visualization complete: {output_path}")
    print(f"{'='*70}\n")

    return str(output_path)


if __name__ == "__main__":
    import sys
    try:
        plot_file = plot_clusters_3d()
        print(f"[OK] 3D clustering plot created: {plot_file}")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
