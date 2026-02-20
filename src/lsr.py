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

    # Convenience presets mapping intent to lambda
    INTENT_PRESETS = {
        "factual": 0.9,
        "understanding": 0.2,
        "balanced": 0.5,
    }

    def __init__(self, n_components: int = 1, sampling_method: str = "quantile",
                 pca_norm: str = "l2",
                 variance_target: Optional[float] = None,
                 max_components: int = 5,
                 lambda_param: float = 0.0,
                 intent: Optional[str] = None):
        """
        Initialize LSR.

        Args:
            n_components: Number of principal components to use (default: 1).
                         Ignored if variance_target is set.
            sampling_method: Method for sampling along principal direction
                           ('quantile' or 'deterministic')
            pca_norm: Norm for PCA direction finding ('l2' or 'l1').
                     L2 (default) maximizes variance; fast via eigendecomposition.
                     L1 maximizes sum of absolute projections; robust to outliers.
            variance_target: If set, compute enough PCs to explain this fraction
                           of total variance (e.g. 0.80 = 80%). Overrides
                           n_components. Capped by max_components.
            max_components: Maximum number of PCs to compute (default: 5).
                          Limits both computational cost and accumulated
                          deflation error in power iteration.
            lambda_param: Relevance-diversity tradeoff (0.0 = max diversity,
                        1.0 = max relevance/consensus). Controls how samples
                        are distributed along principal components: 0 spreads
                        uniformly across the full range, 1 concentrates near
                        the center (most typical documents). Default: 0.0.
            intent: Convenience preset that sets lambda_param automatically.
                   'factual' (lambda=0.9), 'understanding' (lambda=0.2),
                   'balanced' (lambda=0.5). Overrides lambda_param if set.
        """
        self.n_components = n_components
        self.sampling_method = sampling_method
        self.pca_norm = pca_norm
        self.variance_target = variance_target
        self.max_components = max_components
        if intent is not None:
            if intent not in self.INTENT_PRESETS:
                raise ValueError(f"Unknown intent '{intent}'. Choose from: {list(self.INTENT_PRESETS.keys())}")
            self.lambda_param = self.INTENT_PRESETS[intent]
        else:
            self.lambda_param = lambda_param
        self.neighborhood_mean = None
        self.eigenvalues = None
        self.eigenvectors = None
        self._last_info = None

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
        Compute local PCA on the neighborhood using power iteration
        with successive deflation for multi-PC extraction.

        Cost per PC: O(T * N * d) where T ~ 10 iterations.
        Does NOT form the d x d covariance matrix, making it efficient
        for high-dimensional embeddings (d = 1536, 3072).

        The number of PCs is determined by:
        - variance_target (if set): compute PCs until cumulative explained
          variance >= target, up to max_components
        - n_components (otherwise): compute exactly this many PCs

        Args:
            neighborhood: Neighborhood embeddings (n, d)

        Returns:
            Dictionary with PCA results
        """
        if len(neighborhood) == 0:
            raise ValueError("Empty neighborhood")

        self.neighborhood_mean = neighborhood.mean(axis=0)
        centered = neighborhood - self.neighborhood_mean

        # Total variance: trace(C) = sum of squared norms / n
        total_variance = float(np.sum(centered ** 2) / len(neighborhood))
        if total_variance < 1e-12:
            self.eigenvalues = np.array([0.0])
            self.eigenvectors = np.zeros((centered.shape[1], 1))
            return {'eigenvalues': self.eigenvalues,
                    'explained_variance_ratio': np.array([0.0]),
                    'mean': self.neighborhood_mean, 'pca_norm': self.pca_norm}

        # Determine how many PCs to compute
        if self.variance_target is not None:
            n_target = self.max_components
        else:
            n_target = self.n_components
        n_target = min(n_target, len(neighborhood) - 1)

        # Extract PCs via successive deflation
        vecs, eigs = [], []
        residual = centered.copy()
        cumulative_r = 0.0

        for i in range(n_target):
            if self.pca_norm == "l1" and i == 0:
                v = l1_principal_component(residual)
            else:
                v, _ = power_iteration(residual, n_iterations=10)

            eig = float(np.sum((centered @ v) ** 2) / len(neighborhood))
            r = eig / total_variance

            vecs.append(v)
            eigs.append(eig)
            cumulative_r += r

            # Deflate residual
            proj = residual @ v
            residual = residual - np.outer(proj, v)

            # Stop early if variance target met
            if self.variance_target is not None and cumulative_r >= self.variance_target:
                break

        self.eigenvalues = np.array(eigs)
        self.eigenvectors = np.column_stack(vecs)  # (d, m)
        ratios = self.eigenvalues / total_variance

        return {
            'eigenvalues': self.eigenvalues,
            'explained_variance_ratio': ratios,
            'cumulative_variance': cumulative_r,
            'n_components_used': len(eigs),
            'mean': self.neighborhood_mean,
            'pca_norm': self.pca_norm,
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
        k: int,
        lambda_override: Optional[float] = None
    ) -> np.ndarray:
        """
        Sample k points by dividing projections into quantiles,
        with lambda-controlled center biasing.

        When lambda=0: uniform quantile spacing (max diversity).
        When lambda=1: all samples from the center (max relevance).

        Args:
            projections: Scalar projections (n,)
            k: Number of samples to select
            lambda_override: If set, use this lambda instead of self.lambda_param

        Returns:
            Indices of selected points
        """
        n = len(projections)
        if k >= n:
            return np.arange(n)

        lam = lambda_override if lambda_override is not None else self.lambda_param

        # Sort indices by projection values
        sorted_indices = np.argsort(projections)

        # Generate uniform positions in [0, 1], then bias toward center
        uniform_positions = np.linspace(0, 1, k, endpoint=False) + 0.5 / k
        # Interpolate toward center (0.5): biased = uniform*(1-lam) + 0.5*lam
        biased_positions = uniform_positions * (1.0 - lam) + 0.5 * lam

        # Convert to array indices
        idx_positions = (biased_positions * (n - 1)).astype(int)
        idx_positions = np.clip(idx_positions, 0, n - 1)

        selected_indices = sorted_indices[idx_positions]
        # Deduplicate (can happen at high lambda when positions converge)
        seen = set()
        result = []
        for idx in selected_indices:
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
        # Fill any gaps from center outward
        if len(result) < k:
            center_idx = n // 2
            for offset in range(n):
                for candidate in [center_idx + offset, center_idx - offset]:
                    if 0 <= candidate < n and sorted_indices[candidate] not in seen:
                        seen.add(sorted_indices[candidate])
                        result.append(sorted_indices[candidate])
                    if len(result) >= k:
                        break
                if len(result) >= k:
                    break

        return np.array(result[:k])

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

        # Step 2: Local PCA (computes 1 or more PCs)
        pca_info = self.fit_local_pca(neighborhood)
        n_pcs = len(self.eigenvalues)

        # Step 3: Sample
        if n_pcs == 1:
            # Single-PC path (original, fast)
            projections = self.project_onto_principal_direction(neighborhood)
            if self.sampling_method == "quantile":
                selected_local_indices = self.sample_by_quantiles(projections, k)
            else:
                selected_local_indices = self.sample_deterministic(projections, k)
        else:
            # Multi-PC proportional allocation
            selected_local_indices = self._multi_pc_sample(neighborhood, k)

        self._last_info = {
            'n_components_used': n_pcs,
            'explained_variance_ratio': pca_info['explained_variance_ratio'].tolist(),
            'cumulative_variance': pca_info.get('cumulative_variance', float(pca_info['explained_variance_ratio'][0])),
            'pca_norm': self.pca_norm,
        }

        selected_embeddings = neighborhood[selected_local_indices]
        selected_corpus_indices = indices[selected_local_indices]

        return selected_embeddings, selected_corpus_indices

    def _multi_pc_sample(self, neighborhood: np.ndarray, k: int) -> np.ndarray:
        """
        Proportional allocation across multiple PCs.

        Allocates k samples proportionally to eigenvalues (more samples to
        higher-variance directions), then quantile-samples each PC independently,
        skipping already-selected points.
        """
        centered = neighborhood - self.neighborhood_mean
        n = len(neighborhood)
        m = len(self.eigenvalues)

        # Compute projections for each PC
        all_projections = [centered @ self.eigenvectors[:, i] for i in range(m)]

        # Allocate samples: proportional to eigenvalue, min 1 per PC
        alloc = self._allocate_proportional(self.eigenvalues, k)

        # Quantile-sample each PC, skipping already-selected
        selected = set()
        selected_list = []

        lam = self.lambda_param

        for pc_idx in range(m):
            k_i = alloc[pc_idx]
            if k_i == 0:
                continue

            proj = all_projections[pc_idx]
            sorted_indices = np.argsort(proj)
            available = [idx for idx in sorted_indices if idx not in selected]

            if len(available) == 0:
                continue

            picks = min(k_i, len(available))
            # Lambda-biased quantile positions
            uniform_pos = np.linspace(0, 1, picks, endpoint=False) + 0.5 / picks
            biased_pos = uniform_pos * (1.0 - lam) + 0.5 * lam
            for i in range(picks):
                pos = int(biased_pos[i] * (len(available) - 1))
                pos = min(pos, len(available) - 1)
                chosen = available[pos]
                selected.add(chosen)
                selected_list.append(chosen)

        # Fill any remaining slots from PC1 (dedup overflow)
        if len(selected_list) < k:
            sorted_pc1 = np.argsort(all_projections[0])
            for idx in sorted_pc1:
                if idx not in selected:
                    selected.add(idx)
                    selected_list.append(idx)
                if len(selected_list) >= k:
                    break

        return np.array(selected_list[:k])

    @staticmethod
    def _allocate_proportional(eigenvalues: np.ndarray, k: int) -> List[int]:
        """Allocate k samples across PCs proportionally to eigenvalues, min 1 each."""
        m = len(eigenvalues)
        if k <= m:
            return [1 if i < k else 0 for i in range(m)]

        total = float(np.sum(eigenvalues))
        if total < 1e-12:
            base = k // m
            alloc = [base] * m
            for i in range(k - base * m):
                alloc[i] += 1
            return alloc

        # 1 per PC + distribute remainder by eigenvalue share
        remaining = k - m
        weights = eigenvalues / total
        fractional = remaining * weights
        floors = np.floor(fractional).astype(int)
        alloc = [1 + int(f) for f in floors]

        # Distribute leftover by largest fractional remainder
        leftover = k - sum(alloc)
        if leftover > 0:
            remainders = fractional - floors
            for idx in np.argsort(-remainders)[:leftover]:
                alloc[idx] += 1

        return alloc


class AdaptiveLSR:
    """
    Adaptive Local Spectral Retrieval.

    Two operating modes:

    **Threshold mode** (default, ``variance_target=None``):
        Routes queries based on fixed r1 cutoffs:
        - Strong collapse (r1 >= 70%): LSR along PC1
        - Moderate collapse (r1 50-70%): LSR along PC1 + PC2
        - Weak collapse (r1 < 50%): Falls back to MMR

    **Variance-target mode** (``variance_target=0.80``):
        Computes PCs via successive deflation + power iteration until the
        cumulative explained variance reaches the target.  Allocates k
        samples proportionally across the retained PCs.  No MMR fallback;
        the number of components is data-driven.

        Practical ceiling: ``max_components`` (default 5) caps the number
        of PCs to avoid diminishing returns and accumulated deflation error
        in the power-iteration estimates.
    """

    def __init__(
        self,
        strong_threshold: float = 0.70,
        moderate_threshold: float = 0.50,
        lambda_param: float = 0.5,
        sampling_method: str = "quantile",
        pca_norm: str = "l2",
        variance_target: Optional[float] = None,
        max_components: int = 5,
    ):
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold
        self.lambda_param = lambda_param
        self.sampling_method = sampling_method
        self.pca_norm = pca_norm
        self.variance_target = variance_target
        self.max_components = max_components
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
