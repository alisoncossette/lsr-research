"""
Create interactive 3D t-SNE visualization of Neo4j embeddings.

Usage:
    python experiments/plot_tsne_3d_interactive.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from pathlib import Path

def plot_tsne_3d_interactive(
    input_file: str = "data/neo4j_embeddings.csv",
    output_file: str = "data/tsne_3d_interactive.html",
    sample_size: int = None,
    perplexity: int = 30,
    random_state: int = 42
):
    """
    Create interactive 3D t-SNE visualization of embeddings.

    Args:
        input_file: Path to CSV with embeddings
        output_file: Path to save interactive HTML
        sample_size: Number of samples to use (None for all)
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
    """
    print(f"\n{'='*70}")
    print("INTERACTIVE 3D t-SNE VISUALIZATION")
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

    # Run t-SNE in 3D
    print(f"\n[3] Running 3D t-SNE (perplexity={perplexity})...")
    print(f"    This may take several minutes...")
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=random_state,
        verbose=1
    )
    embeddings_3d = tsne.fit_transform(embeddings)
    print(f"    [OK] t-SNE complete")

    # Add 3D coordinates to dataframe
    df_sample['tsne_x'] = embeddings_3d[:, 0]
    df_sample['tsne_y'] = embeddings_3d[:, 1]
    df_sample['tsne_z'] = embeddings_3d[:, 2]

    # Create hover text with truncated document text
    df_sample['hover_text'] = df_sample['text'].apply(
        lambda x: x[:200] + '...' if len(str(x)) > 200 else str(x)
    )

    # Create interactive 3D scatter plot
    print(f"\n[4] Creating interactive 3D visualization...")
    fig = go.Figure(data=[go.Scatter3d(
        x=df_sample['tsne_x'],
        y=df_sample['tsne_y'],
        z=df_sample['tsne_z'],
        mode='markers',
        marker=dict(
            size=3,
            color=df_sample.index,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Document Index"),
            opacity=0.8
        ),
        text=df_sample['hover_text'],
        hovertemplate='<b>Document %{marker.color}</b><br>' +
                      'X: %{x:.2f}<br>' +
                      'Y: %{y:.2f}<br>' +
                      'Z: %{z:.2f}<br>' +
                      '<br>%{text}<extra></extra>',
        name='Documents'
    )])

    fig.update_layout(
        title=dict(
            text=f'Interactive 3D t-SNE Visualization of Document Embeddings<br>' +
                 f'<sub>({len(embeddings_3d)} documents, {len(emb_cols)}-dimensional embeddings)</sub>',
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
        width=1200,
        height=800,
        hovermode='closest'
    )

    # Save interactive HTML
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"    [OK] Saved interactive plot to {output_path}")

    # Print summary
    print(f"\n[5] Summary")
    print(f"    - Documents visualized: {len(embeddings_3d)}")
    print(f"    - Embedding dimension: {len(emb_cols)}")
    print(f"    - t-SNE dimensions: 3D")
    print(f"    - t-SNE perplexity: {perplexity}")
    print(f"    - Output file: {output_path}")
    print(f"    - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print(f"\n[6] Usage")
    print(f"    Open {output_path} in your web browser to explore:")
    print(f"    - Click and drag to rotate the 3D view")
    print(f"    - Scroll to zoom in/out")
    print(f"    - Hover over points to see document text")
    print(f"    - Use toolbar to pan, zoom, reset view")

    print(f"\n{'='*70}")
    print(f"Interactive visualization complete: {output_path}")
    print(f"{'='*70}\n")

    return str(output_path)


if __name__ == "__main__":
    import sys
    try:
        plot_file = plot_tsne_3d_interactive()
        print(f"[OK] Interactive 3D t-SNE plot created: {plot_file}")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
