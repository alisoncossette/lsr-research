"""
Local Spectral Retrieval (LSR) - Research Implementation

A method for improving embedding-based search by leveraging local
geometric structure in threshold-defined neighborhoods.
"""

from .lsr import (
    LocalSpectralRetrieval,
    TopKRetrieval,
    ThresholdRandomRetrieval,
    MMRRetrieval
)

from .metrics import (
    compute_redundancy_score,
    compute_variance_coverage,
    compute_recall,
    compute_ndcg,
    evaluate_retrieval
)

from .embeddings import (
    ClaudeEmbedder,
    MockEmbedder,
    SyntheticDataset
)

__version__ = "0.1.0"
__author__ = "Alison Cossette"

__all__ = [
    "LocalSpectralRetrieval",
    "TopKRetrieval",
    "ThresholdRandomRetrieval",
    "MMRRetrieval",
    "compute_redundancy_score",
    "compute_variance_coverage",
    "compute_recall",
    "compute_ndcg",
    "evaluate_retrieval",
    "ClaudeEmbedder",
    "MockEmbedder",
    "SyntheticDataset",
]
