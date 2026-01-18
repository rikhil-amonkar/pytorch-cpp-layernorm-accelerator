# Time Results: Baseline vs Optimized Implementation

Results from 1000 iterations, averaged across 3 test cases.

Test Cases: (100, 512), (1000, 1024), (5000, 256).

Margin of Error (w/ Tests): +/- 0.0001 seconds.

## Forward Pass

| Test Case | PyTorch (seconds) | Manual Baseline (seconds) | Manual Optimized - BEST (seconds) | Improvement |
|-----------|--------------|---------------|-------------|-------------|
| (100, 512) | 0.0000 | 0.0001 | 0.0001 | 0.0% |
| (1000, 1024) | 0.0005 | 0.0021 | 0.0019 | +10.5% |
| (5000, 256) | 0.0005 | 0.0020 | 0.0021 | -5.0% |

## Backward Pass

| Test Case | PyTorch (seconds) | Manual Baseline (seconds) | Manual Optimized (seconds) | Improvement |
|-----------|--------------|---------------|-------------|-------------|
| (100, 512) | 0.0004 | 0.0010 | 0.0005 | +50.0% |
| (1000, 1024) | 0.0087 | 0.0242 | 0.0101 | +58.3% |
| (5000, 256) | 0.0106 | 0.0253 | 0.0122 | +51.8% |
