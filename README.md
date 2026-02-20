# LSR: Local Spectral Retrieval

**Drop-in diversity for embedding retrieval.** LSR detects when your nearest neighbors are redundant and automatically selects a diverse, non-redundant subset — no retraining, no model changes.

## The Problem

When you retrieve the top-k nearest embeddings, you often get k copies of the same thing. Paraphrases, near-duplicates, boilerplate — they all cluster in embedding space. This is especially bad for RAG and multi-hop QA, where you need *different* pieces of evidence, not the same fact repeated five ways.

## The Fix

LSR runs PCA on the local neighborhood of your query and samples points that span the principal direction of variation. It takes a few milliseconds and slots in after your existing vector search.

**Adaptive LSR** goes further — it checks *whether* the neighborhood is actually collapsed before applying spectral sampling. If it's not, it falls back to MMR or top-k. No wasted computation on neighborhoods that don't need it.

## Install

```bash
pip install lsr-retrieval
```

Or from source:

```bash
git clone https://github.com/acossette/lsr-research.git
cd lsr-research
pip install -e .
```

## Usage

```python
from lsr import AdaptiveLSR
import numpy as np

# Your embeddings (must be unit-normalized)
corpus = np.random.randn(1000, 256)
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
query = np.random.randn(256)
query /= np.linalg.norm(query)

# Retrieve 10 diverse neighbors
retriever = AdaptiveLSR()
embeddings, indices = retriever.retrieve(query, corpus, k=10, threshold=0.3)

# Check what strategy was used
print(retriever.info)
# {'mode': 'lsr', 'pc1_variance': 0.82, 'n_candidates': 47, ...}
```

### What `mode` tells you

| Mode | PC1 Variance | What happened |
|------|-------------|---------------|
| `lsr` | >= 70% | Strong collapse detected. LSR sampled along PC1. |
| `lsr_2pc` | 50-70% | Moderate collapse. LSR sampled on PC1 + PC2 grid. |
| `mmr` | < 50% | Weak collapse. Fell back to MMR diversification. |
| `all` | — | Neighborhood had <= k points. Returned everything. |

### Using standard LSR (no adaptive routing)

```python
from lsr import LocalSpectralRetrieval

retriever = LocalSpectralRetrieval(n_components=1)
embeddings, indices = retriever.retrieve(query, corpus, k=10, threshold=0.3)
```

### Baselines included

```python
from lsr import TopKRetrieval, MMRRetrieval

topk = TopKRetrieval()
mmr = MMRRetrieval()

emb_topk, idx_topk = topk.retrieve(query, corpus, k=10)
emb_mmr, idx_mmr = mmr.retrieve(query, corpus, k=10, threshold=0.3, lambda_param=0.5)
```

## Works with any embeddings

LSR operates on the geometry of your vectors, not the content. It works with:
- **Text** (OpenAI, Cohere, sentence-transformers, etc.)
- **Images** (CLIP, DINOv2, etc.)
- **Video** (frame embeddings, clip-level features)
- **Audio** (wav2vec, CLAP, etc.)
- Any domain where you have dense embeddings and a cosine similarity threshold

## When to use LSR

| Situation | Use |
|-----------|-----|
| Multi-hop QA (HotpotQA-style) | **LSR** — you need diverse evidence |
| RAG with redundant retrieved contexts | **LSR** — reduce token waste |
| Single-hop QA (SQuAD-style) | **Top-k** — closest passage is usually right |
| Uncertain if collapse exists | **AdaptiveLSR** — it checks for you |

## Interactive Demo

```bash
pip install lsr-retrieval[notebook]
marimo edit lsr_interactive.py
```

## Paper

The full method, analysis, and experiments are in [docs/document.tex](docs/document.tex).

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
