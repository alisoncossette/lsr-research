"""
Local Spectral Retrieval (LSR) Implementation

This module implements the Local Spectral Retrieval algorithm for diversity-aware
neighbor selection in embedding-based retrieval systems.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict, Optional
import warnings


class LocalSpectralRetrieval:
    """
    Local Spectral Retrieval: A retrieval method that improves embedding-based search
    by incorporating the local geometric structure of threshold-defined neighborhoods.
    """

    def __init__(self, n_components: int = 1, sampling_method: str = "quantile"):
        """
        Initialize LSR.

        Args:
            n_components: Number of principal components to use (default: 1)
            sampling_method: Method for sampling along principal direction
                           ('quantile' or 'deterministic')
        """
        self.n_components = n_components
        self.sampling_method = sampling_method
        self.pca = None
        self.neighborhood_mean = None
        self.neighborhood_cov = None
        self.eigenvalues = None
        self.eigenvectors = None

    def threshold_retrieval(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform threshold-based retrieval.

        Args:
            query_embedding: Query embedding vector (d,)
            corpus_embeddings: Corpus embeddings matrix (n, d)
            threshold: Similarity threshold

        Returns:
            neighborhood: Embeddings above threshold
            indices: Original indices of neighbors
        """
        # Compute cosine similarities
        # Assume embeddings are already normalized
        similarities = corpus_embeddings @ query_embedding

        # Get indices above threshold
        mask = similarities >= threshold
        indices = np.where(mask)[0]

        if len(indices) == 0:
            warnings.warn(f"No neighbors found above threshold {threshold}")
            return np.array([]), np.array([], dtype=int)

        neighborhood = corpus_embeddings[indices]
        return neighborhood, indices

    def fit_local_pca(self, neighborhood: np.ndarray) -> Dict:
        """
        Compute local PCA on the neighborhood.

        Args:
            neighborhood: Neighborhood embeddings (n, d)

        Returns:
            Dictionary with PCA results
        """
        if len(neighborhood) == 0:
            raise ValueError("Empty neighborhood")

        # Compute mean and centered data
        self.neighborhood_mean = neighborhood.mean(axis=0)
        centered = neighborhood - self.neighborhood_mean

        # Compute covariance
        self.neighborhood_cov = (centered.T @ centered) / len(neighborhood)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.neighborhood_cov)

        # Sort by eigenvalues in descending order
        idx = np.argsort(-eigenvalues)
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # Use sklearn PCA for consistency
        n_comps = min(self.n_components, neighborhood.shape[1], len(neighborhood))
        self.pca = PCA(n_components=n_comps)
        self.pca.fit(centered)

        return {
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'mean': self.neighborhood_mean
        }

    def project_onto_principal_direction(
        self,
        neighborhood: np.ndarray,
        component: int = 0
    ) -> np.ndarray:
        """
        Project neighborhood onto principal direction.

        Args:
            neighborhood: Neighborhood embeddings (n, d)
            component: Which principal component to use (default: 0 = first)

        Returns:
            Projections onto principal direction (n,)
        """
        centered = neighborhood - self.neighborhood_mean
        principal_vector = self.eigenvectors[:, component]
        projections = centered @ principal_vector
        return projections

    def sample_by_quantiles(
        self,
        projections: np.ndarray,
        k: int
    ) -> np.ndarray:
        """
        Sample k points by dividing projections into quantiles.

        Args:
            projections: Scalar projections (n,)
            k: Number of samples to select

        Returns:
            Indices of selected points
        """
        n = len(projections)
        if k >= n:
            return np.arange(n)

        # Sort indices by projection values
        sorted_indices = np.argsort(projections)

        # Select k points at quantiles
        selected_indices = []
        for i in range(k):
            quantile_pos = int(i * n / k)
            # Ensure we don't exceed array bounds
            quantile_pos = min(quantile_pos, n - 1)
            selected_indices.append(sorted_indices[quantile_pos])

        return np.array(selected_indices)

    def sample_deterministic(
        self,
        projections: np.ndarray,
        k: int
    ) -> np.ndarray:
        """
        Sample k points by deterministic spacing along sorted projections.

        Args:
            projections: Scalar projections (n,)
            k: Number of samples to select

        Returns:
            Indices of selected points
        """
        n = len(projections)
        if k >= n:
            return np.arange(n)

        sorted_indices = np.argsort(projections)

        # Select evenly spaced points
        step = n / k
        selected_indices = []
        for i in range(k):
            idx = int(i * step)
            idx = min(idx, n - 1)
            selected_indices.append(sorted_indices[idx])

        return np.array(selected_indices)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        k: int,
        threshold: float = 0.75
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform full LSR retrieval.

        Args:
            query_embedding: Query embedding (d,)
            corpus_embeddings: Corpus embeddings (n, d)
            k: Number of neighbors to return
            threshold: Similarity threshold

        Returns:
            selected_embeddings: k selected embeddings
            selected_indices: Original indices in corpus
        """
        # Step 1: Threshold retrieval
        neighborhood, indices = self.threshold_retrieval(
            query_embedding,
            corpus_embeddings,
            threshold
        )

        if len(neighborhood) == 0:
            return np.array([]), np.array([], dtype=int)

        if len(neighborhood) <= k:
            return neighborhood, indices

        # Step 2: Local PCA
        self.fit_local_pca(neighborhood)

        # Step 3: Project onto principal direction
        projections = self.project_onto_principal_direction(neighborhood)

        # Step 4: Variance-aware sampling
        if self.sampling_method == "quantile":
            selected_local_indices = self.sample_by_quantiles(projections, k)
        else:
            selected_local_indices = self.sample_deterministic(projections, k)

        selected_embeddings = neighborhood[selected_local_indices]
        selected_corpus_indices = indices[selected_local_indices]

        return selected_embeddings, selected_corpus_indices


class TopKRetrieval:
    """Standard top-k cosine similarity retrieval baseline."""

    def retrieve(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        k: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k most similar documents.

        Args:
            query_embedding: Query embedding (d,)
            corpus_embeddings: Corpus embeddings (n, d)
            k: Number of neighbors to return

        Returns:
            selected_embeddings: k selected embeddings
            selected_indices: Original indices in corpus
        """
        similarities = corpus_embeddings @ query_embedding
        top_indices = np.argsort(-similarities)[:k]
        return corpus_embeddings[top_indices], top_indices


class ThresholdRandomRetrieval:
    """Threshold-based retrieval with random sampling baseline."""

    def retrieve(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        k: int,
        threshold: float = 0.75,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve from threshold neighborhood with random sampling.

        Args:
            query_embedding: Query embedding (d,)
            corpus_embeddings: Corpus embeddings (n, d)
            k: Number of neighbors to return
            threshold: Similarity threshold

        Returns:
            selected_embeddings: k selected embeddings
            selected_indices: Original indices in corpus
        """
        similarities = corpus_embeddings @ query_embedding
        mask = similarities >= threshold
        indices = np.where(mask)[0]

        if len(indices) <= k:
            return corpus_embeddings[indices], indices

        selected_local_indices = np.random.choice(len(indices), k, replace=False)
        selected_indices = indices[selected_local_indices]
        return corpus_embeddings[selected_indices], selected_indices


class MMRRetrieval:
    """Maximal Marginal Relevance (MMR) retrieval baseline."""

    def retrieve(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        k: int,
        threshold: float = 0.75,
        lambda_param: float = 0.5,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve using Maximal Marginal Relevance.

        Args:
            query_embedding: Query embedding (d,)
            corpus_embeddings: Corpus embeddings (n, d)
            k: Number of neighbors to return
            threshold: Similarity threshold
            lambda_param: Balance between relevance and diversity (0-1)

        Returns:
            selected_embeddings: k selected embeddings
            selected_indices: Original indices in corpus
        """
        similarities = corpus_embeddings @ query_embedding
        mask = similarities >= threshold
        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            return np.array([]), np.array([], dtype=int)

        if len(candidate_indices) <= k:
            return corpus_embeddings[candidate_indices], candidate_indices

        candidates = corpus_embeddings[candidate_indices]
        selected_indices_local = [np.argmax(similarities[candidate_indices])]

        while len(selected_indices_local) < k:
            selected_embeddings = candidates[selected_indices_local]

            mmr_scores = np.zeros(len(candidates))
            for i, cand_embed in enumerate(candidates):
                if i in selected_indices_local:
                    mmr_scores[i] = -np.inf
                else:
                    relevance = similarities[candidate_indices[i]]
                    redundancy = np.max([
                        cand_embed @ selected_embeddings[j]
                        for j in range(len(selected_embeddings))
                    ])
                    mmr_scores[i] = lambda_param * relevance - (1 - lambda_param) * redundancy

            next_idx = np.argmax(mmr_scores)
            selected_indices_local.append(next_idx)

        selected_indices_local = np.array(selected_indices_local)
        selected_corpus_indices = candidate_indices[selected_indices_local]
        return candidates[selected_indices_local], selected_corpus_indices
