# Local Spectral Retrieval (LSR)

**Variance-aware neighbor selection via PCA in thresholded embedding neighborhoods.**

LSR addresses the *density collapse* problem in embedding-based retrieval: when threshold-based similarity search gathers many near-duplicate documents along a low-dimensional manifold, standard top-k selection returns redundant results. LSR detects this collapse using local PCA and selects documents that maximize variance coverage along the dominant principal direction.

## What's Here

| Path | Description |
|------|-------------|
| `lsr_interactive.py` | **Interactive marimo notebook** — the best way to explore LSR. Run it to see density collapse, compare methods (top-k, LSR, MMR), and experiment with adaptive mode selection. |
| `docs/document.tex` | **Research paper** — formal treatment of the method, adaptive LSR, and empirical analysis. |
| `docs/figs/` | Paper figures (generated from `docs/generate_figures.py`). |
| `src/` | Core Python implementation: `lsr.py`, `metrics.py`, `embeddings.py`. |
| `experiments/` | Experiment scripts for HotpotQA, Natural Questions, etc. |

## Quick Start

### Interactive Notebook (Recommended)

```bash
pip install marimo numpy scikit-learn plotly
marimo edit lsr_interactive.py
```

The notebook walks through:
1. How density collapse happens in embedding neighborhoods
2. Why top-k retrieval fails in collapsed regions
3. How LSR uses PCA to select diverse, non-redundant documents
4. **Adaptive LSR** — automatic mode selection based on variance coverage
5. When to use LSR vs. MMR vs. top-k (single-hop vs. multi-hop QA)

### Paper

```bash
cd docs && pdflatex document.tex && bibtex document && pdflatex document.tex && pdflatex document.tex
```

## The LSR Algorithm

1. **Threshold retrieval**: Gather all documents with cosine similarity >= tau
2. **Local PCA**: Compute principal components of the neighborhood
3. **Project**: Map documents to their position along PC1
4. **Quantile sample**: Select k documents evenly spaced along PC1

**Adaptive LSR** checks the variance ratio before choosing a strategy:
- **PC1 >= 70%** (strong collapse): Standard LSR — ideal regime
- **PC1 50-70%** (moderate): Multi-component LSR using PC1 + PC2
- **PC1 < 50%** (weak collapse): Falls back to MMR or top-k

## Key Findings

- LSR improves multi-hop QA recall by 10-18% on HotpotQA
- Reduces pairwise redundancy by 20-40%
- Single-hop tasks (SQuAD) don't benefit much — top-k is often sufficient
- MMR is better when density collapse is absent; LSR is better when it's present
- Adaptive LSR combines both, routing per-query based on the variance signal

## Requirements

```
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
marimo>=0.9.0
plotly>=5.0.0
```

## Citation

```bibtex
@article{cossette2025lsr,
  title={Local Spectral Retrieval: Variance-Aware Neighbor Selection
         via PCA in Thresholded Embedding Neighborhoods},
  author={Cossette, Alison},
  year={2025}
}
```

## License

MIT
