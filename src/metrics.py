"""
Evaluation metrics for LSR and baselines.
"""

import numpy as np
from typing import List, Set, Tuple, Dict


def compute_redundancy_score(embeddings: np.ndarray) -> float:
    """
    Compute average pairwise cosine similarity (redundancy metric).
    Higher values indicate more redundancy.

    Args:
        embeddings: Retrieved embeddings (k, d)

    Returns:
        Average pairwise similarity
    """
    if len(embeddings) < 2:
        return 0.0

    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, 0)

    # Average of upper triangle
    n = len(embeddings)
    sum_sim = np.sum(similarities)
    num_pairs = n * (n - 1) / 2
    return sum_sim / (2 * num_pairs) if num_pairs > 0 else 0.0


def compute_variance_coverage(
    neighborhood: np.ndarray,
    selected_indices: np.ndarray,
    eigenvalues: np.ndarray
) -> float:
    """
    Compute fraction of local variance captured by selected neighbors.

    Args:
        neighborhood: Full neighborhood embeddings
        selected_indices: Indices of selected points within neighborhood
        eigenvalues: Eigenvalues of neighborhood covariance

    Returns:
        Fraction of variance covered (0-1)
    """
    if len(eigenvalues) == 0:
        return 0.0

    # Total variance
    total_var = np.sum(eigenvalues)

    if total_var == 0:
        return 0.0

    # Compute variance of selected points (simplified metric)
    if len(selected_indices) < 2:
        return 0.0

    selected = neighborhood[selected_indices]
    selected_mean = selected.mean(axis=0)
    neighborhood_mean = neighborhood.mean(axis=0)

    # Variance of selected points
    centered = selected - selected_mean
    selected_var = np.sum(np.sum(centered ** 2, axis=1))

    # Normalize by total neighborhood variance
    return min(selected_var / total_var, 1.0)


def compute_recall(
    retrieved_indices: np.ndarray,
    relevant_indices: Set[int]
) -> float:
    """
    Compute recall - fraction of relevant items retrieved.

    Args:
        retrieved_indices: Indices of retrieved items
        relevant_indices: Set of relevant item indices

    Returns:
        Recall (0-1)
    """
    if len(relevant_indices) == 0:
        return 0.0

    retrieved_set = set(retrieved_indices.tolist())
    num_relevant_retrieved = len(retrieved_set & relevant_indices)
    return num_relevant_retrieved / len(relevant_indices)


def compute_ndcg(
    retrieved_indices: np.ndarray,
    relevant_indices: Set[int],
    k: int = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (nDCG).

    Args:
        retrieved_indices: Indices of retrieved items (in order)
        relevant_indices: Set of relevant item indices
        k: Compute nDCG@k (if None, use all)

    Returns:
        nDCG score (0-1)
    """
    if len(relevant_indices) == 0:
        return 0.0

    if k is None:
        k = len(retrieved_indices)

    # Compute DCG
    dcg = 0.0
    for i, idx in enumerate(retrieved_indices[:k]):
        if idx in relevant_indices:
            dcg += 1.0 / np.log2(i + 2)  # +2 because position starts at 1

    # Compute ideal DCG (all relevant items ranked first)
    ideal_dcg = 0.0
    for i in range(min(k, len(relevant_indices))):
        ideal_dcg += 1.0 / np.log2(i + 2)

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def evaluate_retrieval(
    retrieved_embeddings: np.ndarray,
    retrieved_indices: np.ndarray,
    relevant_indices: Set[int],
    k: int,
    neighborhood_embeddings: np.ndarray = None,
    eigenvalues: np.ndarray = None
) -> Dict:
    """
    Comprehensive evaluation of retrieval results.

    Args:
        retrieved_embeddings: Retrieved embeddings
        retrieved_indices: Indices of retrieved items
        relevant_indices: Set of relevant item indices
        k: Number of retrieved items
        neighborhood_embeddings: Full neighborhood (for variance coverage)
        eigenvalues: Eigenvalues of neighborhood (for variance coverage)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Recall
    metrics['recall'] = compute_recall(retrieved_indices, relevant_indices)

    # nDCG
    metrics['ndcg'] = compute_ndcg(retrieved_indices, relevant_indices, k)

    # Redundancy
    if len(retrieved_embeddings) > 0:
        metrics['redundancy'] = compute_redundancy_score(retrieved_embeddings)
    else:
        metrics['redundancy'] = 0.0

    # Variance coverage
    if neighborhood_embeddings is not None and eigenvalues is not None:
        # Find local indices of retrieved items (simplified)
        metrics['variance_coverage'] = compute_variance_coverage(
            neighborhood_embeddings,
            np.arange(min(len(retrieved_embeddings), len(neighborhood_embeddings))),
            eigenvalues
        )
    else:
        metrics['variance_coverage'] = 0.0

    return metrics
