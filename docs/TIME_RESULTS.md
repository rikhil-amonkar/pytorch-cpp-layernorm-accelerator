# Time Results: Baseline vs Optimized Implementation

Results from 1000 iterations, averaged across 3 test cases.

Test Cases: (100, 512), (1000, 1024), (5000, 256).

Margin of Error (w/ Tests): +/- 0.0001 seconds.

## Forward Pass

| Test Case | PyTorch (seconds) | Manual Baseline (seconds) | Manual Optimized - BEST (seconds) | Improvement |
|-----------|--------------|---------------|-------------|-------------|
| (100, 512) | 0.0000 | 0.0004 | 0.0003 | +25.0% |
| (1000, 1024) | 0.0004 | 0.0080 | 0.0062 | +22.5% |
| (5000, 256) | 0.0006 | 0.0103 | 0.0077 | +25.2% |

## Backward Pass

| Test Case | PyTorch (seconds) | Manual Baseline (seconds) | Manual Optimized (seconds) | Improvement |
|-----------|--------------|---------------|-------------|-------------|
| (100, 512) | 0.0001 | 0.0009 | 0.0005 | +44.4% |
| (1000, 1024) | 0.0019 | 0.0222 | 0.0094 | +57.7% |
| (5000, 256) | 0.0022 | 0.0244 | 0.0115 | +52.9% |
