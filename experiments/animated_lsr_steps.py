"""
Animated Step-by-Step LSR Visualization

Shows the LSR algorithm in action:
1. Query projection
2. Top-K by cosine similarity (traditional)
3. Threshold neighborhood (all candidates)
4. LSR selection (PCA-based sampling)

Usage:
    python experiments/animated_lsr_steps.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lsr import LocalSpectralRetrieval, TopKRetrieval
from embeddings import MockEmbedder
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import pickle

def load_umap_model_and_coords():
    """
    Load pre-computed UMAP model and 3D coordinates.
    """
    # Load UMAP model
    model_path = Path("experiments/models/umap_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(
            f"UMAP model not found at {model_path}\n"
            "Run: python experiments/precompute_umap_space.py"
        )

    print(f"[*] Loading UMAP model from {model_path}...")
    with open(model_path, "rb") as f:
        umap_model = pickle.load(f)
    print(f"    [OK] UMAP model loaded")

    # Load 3D coordinates
    coords_path = Path("data/corpus_umap_3d.npy")
    if not coords_path.exists():
        raise FileNotFoundError(
            f"UMAP coordinates not found at {coords_path}\n"
            "Run: python experiments/precompute_umap_space.py"
        )

    print(f"[*] Loading UMAP 3D coordinates from {coords_path}...")
    umap_3d = np.load(coords_path)
    print(f"    [OK] Loaded {umap_3d.shape}")

    return umap_model, umap_3d


def create_animated_lsr_visualization(
    query_text: str = "machine learning applications",
    k: int = 20,
    threshold: float = 0.75,
    embeddings_file: str = "data/neo4j_embeddings.csv",
    output_file: str = "data/lsr_animation.html"
):
    """
    Create animated visualization showing LSR algorithm steps.

    Steps shown:
    1. Query + full corpus
    2. Top-K by cosine similarity
    3. Threshold neighborhood
    4. LSR final selection
    """
    print(f"\n{'='*70}")
    print("ANIMATED LSR STEP-BY-STEP VISUALIZATION")
    print(f"{'='*70}")

    # Load corpus data
    print(f"\n[1] Loading corpus...")
    df = pd.read_csv(embeddings_file)
    emb_cols = [col for col in df.columns if col.startswith('emb_')]
    corpus_embeddings = df[emb_cols].values
    corpus_embeddings = normalize(corpus_embeddings, norm='l2')
    doc_texts = df['text'].tolist() if 'text' in df.columns else [f"Doc {i}" for i in range(len(df))]
    doc_ids = df['doc_id'].tolist() if 'doc_id' in df.columns else list(range(len(df)))
    print(f"    [OK] Loaded {len(corpus_embeddings)} documents")

    # Load UMAP model and 3D coordinates
    print(f"\n[2] Loading UMAP model and 3D coordinates...")
    umap_model, umap_3d = load_umap_model_and_coords()

    # Generate query embedding
    print(f"\n[3] Generating query embedding...")
    embedder = MockEmbedder(embedding_dim=corpus_embeddings.shape[1])
    query_embedding = embedder.embed_single(query_text)
    query_embedding = normalize(query_embedding.reshape(1, -1), norm='l2').flatten()
    print(f"    [OK] Query: '{query_text}'")

    # Project query into UMAP 3D space (NATIVE PROJECTION!)
    print(f"\n[4] Projecting query into UMAP 3D space...")
    query_3d = umap_model.transform(query_embedding.reshape(1, -1))[0]
    print(f"    [OK] Query position: ({query_3d[0]:.2f}, {query_3d[1]:.2f}, {query_3d[2]:.2f})")

    # Run retrieval methods
    print(f"\n[5] Running retrieval methods...")

    # Compute similarities
    similarities = corpus_embeddings @ query_embedding

    # Top-K cosine
    topk = TopKRetrieval()
    _, topk_indices = topk.retrieve(query_embedding, corpus_embeddings, k)
    topk_sims = similarities[topk_indices]
    print(f"    [OK] Top-K: {len(topk_indices)} documents (cosine >= {topk_sims.min():.3f})")

    # Threshold neighborhood
    threshold_mask = similarities >= threshold
    threshold_indices = np.where(threshold_mask)[0]
    print(f"    [OK] Threshold neighborhood: {len(threshold_indices)} documents (cosine >= {threshold:.3f})")

    # LSR
    lsr = LocalSpectralRetrieval(n_components=1, sampling_method="quantile")
    _, lsr_indices = lsr.retrieve(query_embedding, corpus_embeddings, k, threshold)
    lsr_sims = similarities[lsr_indices]
    print(f"    [OK] LSR: {len(lsr_indices)} documents")

    # Create animation frames
    print(f"\n[6] Creating animation frames...")

    frames = []

    # Frame 0: Just corpus + query
    print(f"    - Frame 0: Corpus + Query")
    frames.append(go.Frame(
        data=[
            # Corpus (gray)
            go.Scatter3d(
                x=umap_3d[:, 0],
                y=umap_3d[:, 1],
                z=umap_3d[:, 2],
                mode='markers',
                marker=dict(size=2, color='lightgray', opacity=0.3),
                name='Corpus',
                hoverinfo='skip'
            ),
            # Query (green star)
            go.Scatter3d(
                x=[query_3d[0]],
                y=[query_3d[1]],
                z=[query_3d[2]],
                mode='markers',
                marker=dict(size=20, color='green', symbol='diamond'),
                name='Query',
                text=[query_text],
                hovertemplate='<b>Query</b><br>%{text}<extra></extra>'
            )
        ],
        name='Step 1: Query'
    ))

    # Frame 1: Top-K by cosine
    print(f"    - Frame 1: Top-K Cosine Similarity")
    topk_coords = umap_3d[topk_indices]
    topk_texts = [f"Doc {doc_ids[i]}<br>Cosine: {topk_sims[j]:.3f}<br>{doc_texts[i][:150]}"
                  for j, i in enumerate(topk_indices)]

    frames.append(go.Frame(
        data=[
            # Corpus (gray, dimmed)
            go.Scatter3d(
                x=umap_3d[:, 0],
                y=umap_3d[:, 1],
                z=umap_3d[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.1),
                name='Corpus',
                hoverinfo='skip'
            ),
            # Top-K (blue)
            go.Scatter3d(
                x=topk_coords[:, 0],
                y=topk_coords[:, 1],
                z=topk_coords[:, 2],
                mode='markers',
                marker=dict(size=8, color='blue', symbol='circle', line=dict(width=2, color='darkblue')),
                name=f'Top-{k} Cosine',
                text=topk_texts,
                hovertemplate='<b>Top-K</b><br>%{text}<extra></extra>'
            ),
            # Query (green star)
            go.Scatter3d(
                x=[query_3d[0]],
                y=[query_3d[1]],
                z=[query_3d[2]],
                mode='markers',
                marker=dict(size=20, color='green', symbol='diamond'),
                name='Query',
                text=[query_text],
                hovertemplate='<b>Query</b><br>%{text}<extra></extra>'
            )
        ],
        name='Step 2: Top-K Cosine'
    ))

    # Frame 2: Threshold neighborhood
    print(f"    - Frame 2: Threshold Neighborhood")
    threshold_coords = umap_3d[threshold_indices]
    threshold_sims = similarities[threshold_indices]
    threshold_texts = [f"Doc {doc_ids[i]}<br>Cosine: {threshold_sims[j]:.3f}<br>{doc_texts[i][:150]}"
                       for j, i in enumerate(threshold_indices)]

    frames.append(go.Frame(
        data=[
            # Corpus (gray, very dimmed)
            go.Scatter3d(
                x=umap_3d[:, 0],
                y=umap_3d[:, 1],
                z=umap_3d[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.05),
                name='Corpus',
                hoverinfo='skip'
            ),
            # Threshold neighborhood (orange)
            go.Scatter3d(
                x=threshold_coords[:, 0],
                y=threshold_coords[:, 1],
                z=threshold_coords[:, 2],
                mode='markers',
                marker=dict(size=5, color='orange', opacity=0.6),
                name=f'Threshold >={threshold:.2f}',
                text=threshold_texts,
                hovertemplate='<b>Threshold Neighborhood</b><br>%{text}<extra></extra>'
            ),
            # Query (green star)
            go.Scatter3d(
                x=[query_3d[0]],
                y=[query_3d[1]],
                z=[query_3d[2]],
                mode='markers',
                marker=dict(size=20, color='green', symbol='diamond'),
                name='Query',
                text=[query_text],
                hovertemplate='<b>Query</b><br>%{text}<extra></extra>'
            )
        ],
        name='Step 3: Threshold Neighborhood'
    ))

    # Frame 3: LSR selection
    print(f"    - Frame 3: LSR Selection")
    lsr_coords = umap_3d[lsr_indices]
    lsr_texts = [f"Doc {doc_ids[i]}<br>Cosine: {lsr_sims[j]:.3f}<br>{doc_texts[i][:150]}"
                 for j, i in enumerate(lsr_indices)]

    # Show threshold neighborhood in background
    frames.append(go.Frame(
        data=[
            # Corpus (gray, very dimmed)
            go.Scatter3d(
                x=umap_3d[:, 0],
                y=umap_3d[:, 1],
                z=umap_3d[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.05),
                name='Corpus',
                hoverinfo='skip'
            ),
            # Threshold neighborhood (orange, faded)
            go.Scatter3d(
                x=threshold_coords[:, 0],
                y=threshold_coords[:, 1],
                z=threshold_coords[:, 2],
                mode='markers',
                marker=dict(size=3, color='orange', opacity=0.2),
                name=f'Threshold >={threshold:.2f}',
                hoverinfo='skip'
            ),
            # LSR selection (red diamonds)
            go.Scatter3d(
                x=lsr_coords[:, 0],
                y=lsr_coords[:, 1],
                z=lsr_coords[:, 2],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond', line=dict(width=2, color='darkred')),
                name=f'LSR (k={k})',
                text=lsr_texts,
                hovertemplate='<b>LSR Selection</b><br>%{text}<extra></extra>'
            ),
            # Query (green star)
            go.Scatter3d(
                x=[query_3d[0]],
                y=[query_3d[1]],
                z=[query_3d[2]],
                mode='markers',
                marker=dict(size=20, color='green', symbol='diamond'),
                name='Query',
                text=[query_text],
                hovertemplate='<b>Query</b><br>%{text}<extra></extra>'
            )
        ],
        name='Step 4: LSR Selection'
    ))

    # Frame 4: Comparison (Top-K vs LSR)
    print(f"    - Frame 4: Comparison")
    frames.append(go.Frame(
        data=[
            # Corpus (gray, very dimmed)
            go.Scatter3d(
                x=umap_3d[:, 0],
                y=umap_3d[:, 1],
                z=umap_3d[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.05),
                name='Corpus',
                hoverinfo='skip'
            ),
            # Top-K (blue)
            go.Scatter3d(
                x=topk_coords[:, 0],
                y=topk_coords[:, 1],
                z=topk_coords[:, 2],
                mode='markers',
                marker=dict(size=6, color='blue', symbol='circle', line=dict(width=1, color='darkblue'), opacity=0.6),
                name=f'Top-{k} Cosine',
                text=topk_texts,
                hovertemplate='<b>Top-K</b><br>%{text}<extra></extra>'
            ),
            # LSR selection (red diamonds)
            go.Scatter3d(
                x=lsr_coords[:, 0],
                y=lsr_coords[:, 1],
                z=lsr_coords[:, 2],
                mode='markers',
                marker=dict(size=8, color='red', symbol='diamond', line=dict(width=2, color='darkred')),
                name=f'LSR (k={k})',
                text=lsr_texts,
                hovertemplate='<b>LSR</b><br>%{text}<extra></extra>'
            ),
            # Query (green star)
            go.Scatter3d(
                x=[query_3d[0]],
                y=[query_3d[1]],
                z=[query_3d[2]],
                mode='markers',
                marker=dict(size=20, color='green', symbol='diamond'),
                name='Query',
                text=[query_text],
                hovertemplate='<b>Query</b><br>%{text}<extra></extra>'
            )
        ],
        name='Step 5: Comparison'
    ))

    # Create figure with first frame
    print(f"\n[7] Building interactive figure...")
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    # Add animation controls
    fig.update_layout(
        title=dict(
            text=f'LSR Algorithm Step-by-Step<br>' +
                 f'<sub>Query: "{query_text}" | k={k} | threshold={threshold:.2f}</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0.95,
                x=0.1,
                xanchor='left',
                buttons=[
                    dict(
                        label='Play Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 2000, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 500}
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'y': 0.05,
            'len': 0.8,
            'x': 0.1,
            'currentvalue': {
                'prefix': 'Step: ',
                'visible': True,
                'xanchor': 'left'
            },
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f.name,
                    'method': 'animate'
                }
                for f in frames
            ]
        }],
        height=800,
        hovermode='closest'
    )

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"    [OK] Saved to {output_path}")
    print(f"    - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Summary
    print(f"\n[8] Summary:")
    print(f"    - Query: '{query_text}'")
    print(f"    - Top-K retrieved: {len(topk_indices)} (cosine >= {topk_sims.min():.3f})")
    print(f"    - Threshold neighborhood: {len(threshold_indices)} (cosine >= {threshold:.3f})")
    print(f"    - LSR selected: {len(lsr_indices)}")
    print(f"    - Overlap (Top-K & LSR): {len(set(topk_indices) & set(lsr_indices))}")

    print(f"\n{'='*70}")
    print(f"ANIMATION COMPLETE: {output_path}")
    print(f"{'='*70}")
    print(f"\nOpen {output_path} in your browser:")
    print(f"  - Click 'Play Play' to animate through steps")
    print(f"  - Use slider to jump to specific steps")
    print(f"  - Rotate/zoom the 3D view at any time")
    print(f"{'='*70}\n")

    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create animated LSR visualization')
    parser.add_argument('--query', type=str, default='machine learning applications',
                        help='Query text')
    parser.add_argument('--k', type=int, default=20,
                        help='Number of results to retrieve')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='LSR threshold')
    parser.add_argument('--output', type=str, default='data/lsr_animation.html',
                        help='Output file path')

    args = parser.parse_args()

    try:
        output_file = create_animated_lsr_visualization(
            query_text=args.query,
            k=args.k,
            threshold=args.threshold,
            output_file=args.output
        )
        print(f"[OK] Animation created: {output_file}")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
