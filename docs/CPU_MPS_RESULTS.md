# Performance Benchmarks for PyTorch LayerNorm

## PyTorch LayerNorm (Forward + Backward)

**CASE: (100, 512)**

| Metric | CPU | MPS | CPU |
|--------|-----|-----|-----|
| Operation | PyTorch LayerNorm | PyTorch LayerNorm | Manual LayerNorm |
| Median (ms) | 0.213 | 1.00 | 0.868 |
| IQR (ms) | 0.01 | 0.03 | 0.037 |
| IQR Range | 0.21 - 0.22 | 0.99 - 1.02 | 0.852 - 0.890 |
| Iterations | 4572 | 977 | 1132 |
| Improvement % | - | -369.48% | - |

**CASE: (1000, 1024)**

| Metric | CPU | MPS | CPU |
|--------|-----|-----|-----|
| Operation | PyTorch LayerNorm | PyTorch LayerNorm | Manual LayerNorm |
| Median (ms) | 3.09 | 2.26 | 15.61 |
| IQR (ms) | 0.21 | 0.13 | 0.22 |
| IQR Range | 2.96 - 3.17 | 2.23 - 2.36 | 15.48 - 15.71 |
| Iterations | 32 | 399 | 65 |
| Improvement % | - | +26.86% | - |

**CASE: (5000, 256)**

| Metric | CPU | MPS | CPU |
|--------|-----|-----|-----|
| Operation | PyTorch LayerNorm | PyTorch LayerNorm | Manual LayerNorm |
| Median (ms) | 3.55 | 2.55 | 19.92 |
| IQR (ms) | 0.17 | 0.25 | 0.40 |
| IQR Range | 3.47 - 3.64 | 2.52 - 2.78 | 19.73 - 20.13 |
| Iterations | 28 | 769 | 51 |
| Improvement % | - | +28.17% | - |

**CASE: (10000, 10000)**

| Metric | CPU | MPS | CPU |
|--------|-----|-----|-----|
| Operation | PyTorch LayerNorm | PyTorch LayerNorm | Manual LayerNorm |
| Median (ms) | 363.74 | 171.04 | 1620.00 |
| IQR (ms) | 28.45 | 1.08 | 10.00 |
| IQR Range | 358.04 - 386.49 | 170.55 - 171.63 | 1620.00 - 1630.00 |
| Iterations | 12 | 6 | 4 |
| Improvement % | - | +52.98% | - |

## Summary

For larger workloads (1000+ samples), MPS (Mac's GPU) performs about 26-28% faster than CPU when running PyTorch's LayerNorm (forward + backward), and even up to 52% faster on very large samples such as 10000. For smaller workloads (100 samples), CPU outperforms MPS due to GPU overhead, resulting in ~369% slower performance on MPS. The manual LayerNorm (C++) seems to perform better on smaller samples such as 100, but struggles a lot with exponential higher runtimes in the larger sample cases.

**Note on Manual LayerNorm (C++)**: The manual implementation uses `float64` precision (required by the C++ backend), while PyTorch's LayerNorm uses `float32`. This precision difference results in approximately 2x computational overhead for double precision operations, which should be considered when comparing performance. The manual LayerNorm performs relatively well on smaller samples (0.933ms vs 0.213ms PyTorch CPU for case (100, 512)), but scales less efficiently on larger workloads, likely due to both precision overhead and optimization differences.