"""
Local Spectral Retrieval (LSR) Implementation

This module implements the Local Spectral Retrieval algorithm for diversity-aware
neighbor selection in embedding-based retrieval systems.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import warnings


def power_iteration(X_centered: np.ndarray, n_iterations: int = 10) -> Tuple[np.ndarray, float]:
    """
    Compute the top eigenvector and eigenvalue of X^T X / n via power iteration
    on the data matrix directly, without forming the d x d covariance matrix.

    Cost: O(T * N * d) where T is n_iterations.
    This is critical for high-dimensional embeddings (d=1536, 3072) where
    forming the covariance matrix costs O(N * d^2) and eigendecomposition O(d^3).

    Args:
        X_centered: Centered data matrix (n, d)
        n_iterations: Number of power iteration steps (default: 10)

    Returns:
        v: Top eigenvector (d,)
        eigenvalue: Corresponding eigenvalue (variance along v)
    """
    n, d = X_centered.shape
    rng = np.random.RandomState(0)
    v = rng.randn(d)
    v /= np.linalg.norm(v)

    for _ in range(n_iterations):
        # Multiply by covariance without forming it: (X^T X) v = X^T (X v)
        Xv = X_centered @ v          # (n,)  — O(nd)
        XtXv = X_centered.T @ Xv     # (d,)  — O(nd)
        norm = np.linalg.norm(XtXv)
        if norm < 1e-12:
            break
        v = XtXv / norm

    # Eigenvalue = v^T (X^T X / n) v = ||Xv||^2 / n
    Xv = X_centered @ v
    eigenvalue = float(np.dot(Xv, Xv) / n)
    return v, eigenvalue


def l1_principal_component(X_centered: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    Compute the first principal component using L1-norm maximization.

    Finds v = argmax_{||v||=1} sum_i |v^T x_i| using the bit-flipping
    algorithm of Markopoulos et al. (2017). More robust to outliers than
    standard (L2) PCA.

    Args:
        X_centered: Centered data matrix (n, d)
        max_iter: Maximum iterations for optimization
        tol: Convergence tolerance

    Returns:
        v: Unit vector maximizing sum of absolute projections (d,)
    """
    n = X_centered.shape[0]

    # Initialize with L2 direction via power iteration (not eigendecomposition)
    v, _ = power_iteration(X_centered, n_iterations=10)

    for _ in range(max_iter):
        # Compute projections and their signs
        projections = X_centered @ v
        signs = np.sign(projections)
        signs[signs == 0] = 1.0

        # Weighted sum: v_new = sum_i sign(v^T x_i) * x_i
        v_new = X_centered.T @ signs
        v_new_norm = np.linalg.norm(v_new)
        if v_new_norm < 1e-12:
            break
        v_new = v_new / v_new_norm

        # Check convergence
        if np.abs(np.abs(v_new @ v) - 1.0) < tol:
            v = v_new
            break
        v = v_new

    return v


class LocalSpectralRetrieval:
    """
    Local Spectral Retrieval: A retrieval method that improves embedding-based search
    by incorporating the local geometric structure of threshold-defined neighborhoods.
    """

    def __init__(self, n_components: int = 1, sampling_method: str = "quantile",
                 pca_norm: str = "l2"):
        """
        Initialize LSR.

        Args:
            n_components: Number of principal components to use (default: 1)
            sampling_method: Method for sampling along principal direction
                           ('quantile' or 'deterministic')
            pca_norm: Norm for PCA direction finding ('l2' or 'l1').
                     L2 (default) maximizes variance; fast via eigendecomposition.
                     L1 maximizes sum of absolute projections; robust to outliers.
        """
        self.n_components = n_components
        self.sampling_method = sampling_method
        self.pca_norm = pca_norm
        self.neighborhood_mean = None
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
        Compute local PCA on the neighborhood using power iteration.

        Cost: O(T * N * d) where T ~ 10 iterations.
        Does NOT form the d x d covariance matrix, making it efficient
        for high-dimensional embeddings (d = 1536, 3072).

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

        if self.pca_norm == "l1":
            v1 = l1_principal_component(centered)
            eigenvalue = float(np.sum((centered @ v1) ** 2) / len(neighborhood))
        else:
            v1, eigenvalue = power_iteration(centered, n_iterations=10)

        # Compute total variance without forming covariance: trace(C) = sum of squared norms / n
        total_variance = float(np.sum(centered ** 2) / len(neighborhood))
        r1 = eigenvalue / total_variance if total_variance > 1e-12 else 0.0

        self.eigenvalues = np.array([eigenvalue])
        self.eigenvectors = v1.reshape(-1, 1)  # Store as column

        return {
            'eigenvalues': self.eigenvalues,
            'explained_variance_ratio': np.array([r1]),
            'mean': self.neighborhood_mean,
            'pca_norm': self.pca_norm
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
        principal_vector = self.eigenvectors[:, component] if self.eigenvectors.ndim == 2 else self.eigenvectors
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


class AdaptiveLSR:
    """
    Adaptive Local Spectral Retrieval.

    Automatically selects the best retrieval strategy based on the
    variance structure of the threshold neighborhood:

    - Strong collapse (PC1 >= 70%): LSR along PC1
    - Moderate collapse (PC1 50-70%): LSR along PC1 + PC2
    - Weak collapse (PC1 < 50%): Falls back to MMR
    """

    def __init__(
        self,
        strong_threshold: float = 0.70,
        moderate_threshold: float = 0.50,
        lambda_param: float = 0.5,
        sampling_method: str = "quantile",
        pca_norm: str = "l2",
    ):
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold
        self.lambda_param = lambda_param
        self.sampling_method = sampling_method
        self.pca_norm = pca_norm
        self._last_info = None

    @property
    def info(self) -> Optional[Dict]:
        """Diagnostic info from the last retrieve() call."""
        return self._last_info

    def retrieve(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        k: int,
        threshold: float = 0.75,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve k diverse neighbors, automatically choosing the best strategy.

        Args:
            query_embedding: Query embedding (d,). Must be unit-normalized.
            corpus_embeddings: Corpus embeddings (n, d). Must be unit-normalized.
            k: Number of neighbors to return.
            threshold: Cosine similarity threshold for neighborhood construction.

        Returns:
            selected_embeddings: (k, d) array of selected embeddings.
            selected_indices: (k,) array of corpus indices.
        """
        # Step 1: threshold retrieval
        similarities = corpus_embeddings @ query_embedding
        mask = similarities >= threshold
        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            self._last_info = {"mode": "empty", "pc1_variance": 0.0, "n_candidates": 0}
            return np.array([]), np.array([], dtype=int)

        if len(candidate_indices) <= k:
            self._last_info = {
                "mode": "all",
                "pc1_variance": 0.0,
                "n_candidates": len(candidate_indices),
            }
            return corpus_embeddings[candidate_indices], candidate_indices

        neighborhood = corpus_embeddings[candidate_indices]

        # Step 2: local PCA via power iteration — O(T*N*d), no d x d covariance
        centered = neighborhood - neighborhood.mean(axis=0)

        if self.pca_norm == "l1":
            v1 = l1_principal_component(centered)
            eig1 = float(np.sum((centered @ v1) ** 2) / len(neighborhood))
        else:
            v1, eig1 = power_iteration(centered, n_iterations=10)

        total_var = float(np.sum(centered ** 2) / len(neighborhood))
        r1 = eig1 / total_var if total_var > 1e-12 else 0.0
        projections_pc1 = centered @ v1

        # Step 3: route based on variance ratio
        if r1 >= self.strong_threshold:
            mode = "lsr"
            selected = self._lsr_select_fast(projections_pc1, k)
        elif r1 >= self.moderate_threshold:
            mode = "lsr_2pc"
            # For 2-component, compute second eigenvector via deflation + power iteration
            deflated = centered - np.outer(projections_pc1, v1)
            v2, _ = power_iteration(deflated, n_iterations=10)
            projections_pc2 = centered @ v2
            selected = self._lsr_select_2pc(projections_pc1, projections_pc2, k)
        else:
            mode = "mmr"
            selected = self._mmr_select(
                neighborhood, similarities[candidate_indices], k
            )

        self._last_info = {
            "mode": mode,
            "pc1_variance": r1,
            "n_candidates": len(candidate_indices),
            "explained_variance_ratio": [r1],
            "pca_norm": self.pca_norm if mode != "mmr" else "n/a",
        }

        selected_embeddings = neighborhood[selected]
        selected_corpus_indices = candidate_indices[selected]
        return selected_embeddings, selected_corpus_indices

    def _lsr_select_fast(self, projections: np.ndarray, k: int) -> np.ndarray:
        """Select k points by quantile sampling along pre-computed PC1 projections."""
        sorted_indices = np.argsort(projections)
        n = len(sorted_indices)
        return np.array([sorted_indices[int(i * n / k)] for i in range(k)])

    def _lsr_select_2pc(
        self, proj_pc1: np.ndarray, proj_pc2: np.ndarray, k: int
    ) -> np.ndarray:
        """Select k points by 2D grid sampling along PC1 and PC2."""
        n = len(proj_pc1)
        grid_k = max(2, int(np.sqrt(k)))
        q1 = np.linspace(0, 1, grid_k + 1)
        selected = []
        for i in range(grid_k):
            for j in range(grid_k):
                lo1 = np.quantile(proj_pc1, q1[i])
                hi1 = np.quantile(proj_pc1, q1[i + 1])
                lo2 = np.quantile(proj_pc2, q1[j])
                hi2 = np.quantile(proj_pc2, q1[j + 1])
                cell_mask = (
                    (proj_pc1 >= lo1) & (proj_pc1 <= hi1)
                    & (proj_pc2 >= lo2) & (proj_pc2 <= hi2)
                )
                cell_indices = np.where(cell_mask)[0]
                if len(cell_indices) > 0:
                    selected.append(cell_indices[0])
                if len(selected) >= k:
                    break
            if len(selected) >= k:
                break

        # Fill remaining slots with PC1 quantile sampling if needed
        if len(selected) < k:
            sorted_indices = np.argsort(proj_pc1)
            remaining = set(range(n)) - set(selected)
            remaining_by_pc1 = sorted(remaining, key=lambda i: proj_pc1[i])
            step = max(1, len(remaining_by_pc1) // (k - len(selected)))
            for idx in remaining_by_pc1[::step]:
                selected.append(idx)
                if len(selected) >= k:
                    break

        return np.array(selected[:k])

    def _mmr_select(
        self, neighborhood: np.ndarray, relevances: np.ndarray, k: int
    ) -> np.ndarray:
        """Fallback MMR selection for weakly collapsed neighborhoods."""
        selected = [int(np.argmax(relevances))]

        for _ in range(k - 1):
            best_score = -np.inf
            best_idx = -1
            for j in range(len(neighborhood)):
                if j in selected:
                    continue
                redundancy = max(
                    neighborhood[j] @ neighborhood[s] for s in selected
                )
                score = (
                    self.lambda_param * relevances[j]
                    - (1 - self.lambda_param) * redundancy
                )
                if score > best_score:
                    best_score = score
                    best_idx = j
            selected.append(best_idx)

        return np.array(selected)


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
