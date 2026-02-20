"""
Embedding generation using Claude API.
"""

import os
import json
import numpy as np
from anthropic import Anthropic
from typing import List, Tuple
import warnings


class ClaudeEmbedder:
    """Generate embeddings using Claude API."""

    def __init__(self, model: str = "claude-opus-4-1-20250805"):
        """
        Initialize Claude embedder.

        Args:
            model: Claude model to use
        """
        self.client = Anthropic()
        self.model = model
        self.embedding_dim = 1024

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using Claude."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a concise semantic summary of the following text for embedding purposes. Return only 100-200 words of semantic content:\n\n{text}"
                }
            ]
        )
        summary = message.content[0].text
        embedding = self._text_to_embedding(summary)
        return embedding

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding using hash-based approach."""
        text_lower = text.lower()
        embedding = np.zeros(self.embedding_dim)
        for i, char in enumerate(text_lower):
            hash_val = hash(char + str(i)) % self.embedding_dim
            embedding[hash_val] += ord(char) / 256.0
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.embed_single(text)
            embeddings.append(embedding)
        return np.array(embeddings)


class MockEmbedder:
    """Mock embedder for quick testing without API calls."""

    def __init__(self, embedding_dim: int = 1024, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate mock embedding for a single text."""
        hash_val = hash(text) % (2 ** 31)
        rng = np.random.RandomState(hash_val)
        embedding = rng.randn(self.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.embed_single(text)
            embeddings.append(embedding)
        return np.array(embeddings)


class SyntheticDataset:
    """Generate synthetic QA dataset with density collapse."""

    def __init__(self, n_queries: int = 10, n_docs_per_query: int = 100, 
                 embedding_dim: int = 1024, seed: int = 42, density_collapse_factor: float = 0.8):
        """
        Initialize synthetic dataset with configurable density collapse.

        Args:
            n_queries: Number of queries
            n_docs_per_query: Documents per query
            embedding_dim: Embedding dimension
            seed: Random seed
            density_collapse_factor: How much to collapse density (0-1, higher = more collapse)
        """
        self.n_queries = n_queries
        self.n_docs_per_query = n_docs_per_query
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.density_collapse_factor = density_collapse_factor

    def generate(self) -> Tuple[np.ndarray, np.ndarray, List[set]]:
        """
        Generate synthetic dataset with intentional density collapse.
        
        Creates multiple clusters of relevant documents (paraphrases) along
        a principal direction, simulating real-world density collapse.

        Returns:
            query_embeddings: Query embeddings (n_queries, embedding_dim)
            corpus_embeddings: Document embeddings (n_queries * n_docs, embedding_dim)
            relevant_docs: List of sets with relevant doc indices per query
        """
        query_embeddings = []
        corpus_embeddings = []
        relevant_docs = []

        for q_idx in range(self.n_queries):
            # Create query embedding
            query = self.rng.randn(self.embedding_dim).astype(np.float32)
            query = query / np.linalg.norm(query)
            query_embeddings.append(query)

            # Create principal direction for density collapse
            principal_dir = self.rng.randn(self.embedding_dim).astype(np.float32)
            principal_dir = principal_dir / np.linalg.norm(principal_dir)

            # Number of relevant clusters (paraphrase groups)
            n_clusters = self.rng.randint(2, 4)
            docs_per_cluster = self.rng.randint(3, 8)
            n_relevant = n_clusters * docs_per_cluster

            relevant_indices = set()

            for d_idx in range(self.n_docs_per_query):
                corpus_idx = q_idx * self.n_docs_per_query + d_idx

                if d_idx < n_relevant:
                    # Create relevant documents in clusters along principal direction
                    cluster_id = d_idx // docs_per_cluster
                    
                    # Position along principal direction
                    cluster_pos = (cluster_id - n_clusters / 2) * 0.3
                    
                    # Start with query
                    doc = query.copy()
                    
                    # Move along principal direction (density collapse)
                    doc = doc + principal_dir * cluster_pos * self.density_collapse_factor
                    
                    # Add small noise within cluster (paraphrases)
                    cluster_noise = self.rng.randn(self.embedding_dim) * 0.08
                    doc = doc + cluster_noise
                    
                    # Normalize
                    doc = doc / np.linalg.norm(doc)
                    relevant_indices.add(corpus_idx)
                else:
                    # Irrelevant documents: random embeddings
                    doc = self.rng.randn(self.embedding_dim)
                    doc = doc / np.linalg.norm(doc)

                corpus_embeddings.append(doc)

            relevant_docs.append(relevant_indices)

        query_embeddings = np.array(query_embeddings)
        corpus_embeddings = np.array(corpus_embeddings)

        return query_embeddings, corpus_embeddings, relevant_docs
