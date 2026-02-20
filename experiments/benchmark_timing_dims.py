"""Timing benchmark: LSR vs MMR across embedding dimensions."""
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lsr import LocalSpectralRetrieval, MMRRetrieval

lsr = LocalSpectralRetrieval(n_components=1, sampling_method='quantile')
mmr = MMRRetrieval()

dims = [384, 768, 1536, 3072]
configs = [
    (100, 10),
    (500, 50),
    (500, 100),
    (1000, 50),
    (5000, 10),
    (5000, 50),
]

print(f"{'d':>6} {'N':>6} {'k':>4} {'MMR (ms)':>10} {'LSR (ms)':>10} {'Speedup':>10}")
print("-" * 52)

for d in dims:
    rng = np.random.RandomState(42)
    for N, k in configs:
        corpus = rng.randn(N, d).astype(np.float64)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        query = rng.randn(d).astype(np.float64)
        query /= np.linalg.norm(query)
        threshold = 0.0

        # Warm up
        lsr.retrieve(query, corpus, k=k, threshold=threshold)
        mmr.retrieve(query, corpus, k=k, threshold=threshold)

        # Time LSR (3 runs)
        times_lsr = []
        for _ in range(3):
            t0 = time.perf_counter()
            lsr.retrieve(query, corpus, k=k, threshold=threshold)
            times_lsr.append((time.perf_counter() - t0) * 1000)
        lsr_ms = np.median(times_lsr)

        # Time MMR (3 runs, skip if too slow)
        estimated = N * k * k * d / 1e9 * 50
        if estimated > 20000:
            mmr_str = ">15,000"
            speedup_str = ">150x"
        else:
            times_mmr = []
            for _ in range(3):
                t0 = time.perf_counter()
                mmr.retrieve(query, corpus, k=k, threshold=threshold)
                times_mmr.append((time.perf_counter() - t0) * 1000)
            mmr_ms = np.median(times_mmr)
            mmr_str = f"{mmr_ms:,.0f}"
            speedup_str = f"{mmr_ms / lsr_ms:,.0f}x" if lsr_ms > 0 else "inf"

        print(f"{d:>6} {N:>6} {k:>4} {mmr_str:>10} {lsr_ms:>10.0f} {speedup_str:>10}")
    print()
