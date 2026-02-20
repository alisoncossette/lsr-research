"""
Demo script for Local Spectral Retrieval (LSR) research recreation.

This script demonstrates LSR on synthetic data and compares it with baselines.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

from lsr import (
    LocalSpectralRetrieval,
    TopKRetrieval,
    ThresholdRandomRetrieval,
    MMRRetrieval
)
from metrics import evaluate_retrieval
from embeddings import SyntheticDataset, MockEmbedder


def run_single_query_experiment(
    retriever,
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    relevant_indices: set,
    k: int = 10,
    threshold: float = 0.75,
    retriever_name: str = "Unknown"
) -> Dict:
    """
    Run retrieval experiment for a single query.

    Args:
        retriever: Retrieval method instance
        query_embedding: Query embedding
        corpus_embeddings: Corpus embeddings
        relevant_indices: Set of relevant document indices
        k: Number of documents to retrieve
        threshold: Similarity threshold (for some methods)
        retriever_name: Name of retriever for logging

    Returns:
        Dictionary with results
    """
    try:
        retrieved_embeddings, retrieved_indices = retriever.retrieve(
            query_embedding,
            corpus_embeddings,
            k=k,
            threshold=threshold
        )

        if len(retrieved_embeddings) == 0:
            print(f"  {retriever_name}: No results retrieved")
            return {"error": "No results"}

        metrics = evaluate_retrieval(
            retrieved_embeddings,
            retrieved_indices,
            relevant_indices,
            k
        )
        metrics["n_retrieved"] = len(retrieved_embeddings)
        return metrics

    except Exception as e:
        print(f"  {retriever_name}: Error - {str(e)}")
        return {"error": str(e)}


def run_experiments(
    n_queries: int = 5,
    n_docs_per_query: int = 100,
    embedding_dim: int = 1024,
    k: int = 10,
    threshold: float = 0.75,
    seed: int = 42
) -> Dict:
    """
    Run full retrieval experiments.

    Args:
        n_queries: Number of queries
        n_docs_per_query: Documents per query
        embedding_dim: Embedding dimension
        k: Number of documents to retrieve
        threshold: Similarity threshold
        seed: Random seed

    Returns:
        Dictionary with all results
    """
    print("\n" + "="*70)
    print("LOCAL SPECTRAL RETRIEVAL (LSR) - DEMO EXPERIMENTS")
    print("="*70)

    # Generate synthetic dataset
    print("\n[1] Generating synthetic dataset...")
    dataset = SyntheticDataset(
        n_queries=n_queries,
        n_docs_per_query=n_docs_per_query,
        embedding_dim=embedding_dim,
        seed=seed
    )
    query_embeddings, corpus_embeddings, relevant_docs = dataset.generate()
    print(f"  - Queries: {len(query_embeddings)}")
    print(f"  - Documents: {len(corpus_embeddings)}")
    print(f"  - Embedding dim: {embedding_dim}")

    # Initialize retrievers
    print("\n[2] Initializing retrievers...")
    retrievers = {
        "LSR (quantile)": LocalSpectralRetrieval(n_components=1, sampling_method="quantile"),
        "LSR (deterministic)": LocalSpectralRetrieval(n_components=1, sampling_method="deterministic"),
        "Top-k": TopKRetrieval(),
        "Threshold + Random": ThresholdRandomRetrieval(),
        "MMR": MMRRetrieval(lambda_param=0.5)
    }
    print(f"  - {len(retrievers)} retrieval methods")

    # Run experiments
    print("\n[3] Running experiments...")
    results = {method: [] for method in retrievers.keys()}

    for query_idx in range(n_queries):
        print(f"\n  Query {query_idx + 1}/{n_queries}:")
        query = query_embeddings[query_idx]
        relevant = relevant_docs[query_idx]

        for method_name, retriever in retrievers.items():
            result = run_single_query_experiment(
                retriever,
                query,
                corpus_embeddings,
                relevant,
                k=k,
                threshold=threshold,
                retriever_name=method_name
            )
            results[method_name].append(result)
            if "error" not in result:
                print(f"    {method_name}: Recall={result['recall']:.3f}, "
                      f"nDCG={result['ndcg']:.3f}, Redundancy={result['redundancy']:.3f}")

    return results


def aggregate_results(results: Dict) -> Dict:
    """
    Aggregate results across queries.

    Args:
        results: Dictionary of results per method

    Returns:
        Dictionary with aggregated metrics
    """
    aggregated = {}

    for method, query_results in results.items():
        # Filter out error results
        valid_results = [r for r in query_results if "error" not in r]

        if len(valid_results) == 0:
            aggregated[method] = {"error": "No valid results"}
            continue

        metrics = {}
        for metric_name in ["recall", "ndcg", "redundancy", "variance_coverage", "n_retrieved"]:
            values = [r.get(metric_name, 0) for r in valid_results]
            metrics[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }

        aggregated[method] = metrics

    return aggregated


def print_results(aggregated: Dict):
    """
    Print formatted results.

    Args:
        aggregated: Aggregated results dictionary
    """
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\nRecall (fraction of relevant docs retrieved):")
    print("-" * 70)
    print(f"{'Method':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    for method, metrics in aggregated.items():
        if "error" in metrics:
            print(f"{method:<25} ERROR")
        else:
            recall = metrics.get("recall", {})
            print(f"{method:<25} {recall.get('mean', 0):.4f}    "
                  f"{recall.get('std', 0):.4f}    "
                  f"{recall.get('min', 0):.4f}    "
                  f"{recall.get('max', 0):.4f}")

    print("\nnDCG@k (ranking quality):")
    print("-" * 70)
    print(f"{'Method':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    for method, metrics in aggregated.items():
        if "error" in metrics:
            print(f"{method:<25} ERROR")
        else:
            ndcg = metrics.get("ndcg", {})
            print(f"{method:<25} {ndcg.get('mean', 0):.4f}    "
                  f"{ndcg.get('std', 0):.4f}    "
                  f"{ndcg.get('min', 0):.4f}    "
                  f"{ndcg.get('max', 0):.4f}")

    print("\nRedundancy Score (lower is better):")
    print("-" * 70)
    print(f"{'Method':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    for method, metrics in aggregated.items():
        if "error" in metrics:
            print(f"{method:<25} ERROR")
        else:
            redundancy = metrics.get("redundancy", {})
            print(f"{method:<25} {redundancy.get('mean', 0):.4f}    "
                  f"{redundancy.get('std', 0):.4f}    "
                  f"{redundancy.get('min', 0):.4f}    "
                  f"{redundancy.get('max', 0):.4f}")


def save_results(results: Dict, filename: str = "results.json"):
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        filename: Output filename
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable = convert_to_serializable(results)

    with open(filename, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    # Run experiments with moderate dataset size for demo
    results = run_experiments(
        n_queries=5,
        n_docs_per_query=100,
        embedding_dim=1024,
        k=10,
        threshold=0.75,
        seed=42
    )

    # Aggregate and print results
    aggregated = aggregate_results(results)
    print_results(aggregated)

    # Save results
    save_results({"detailed": results, "aggregated": aggregated}, "results.json")

    print("\n" + "="*70)
    print("Demo completed! Check results.json for detailed results.")
    print("="*70)
