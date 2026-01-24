# Performance Benchmarks

## Element-wise Multiplication (`x * y`)

| Device | Operation | Median (ms) | IQR (ms) | IQR Range | Iterations |
|--------|-----------|-------------|----------|-----------|--------------|
| CPU | `x * y` | 31.17 | 0.84 | 30.71-31.54 | 32 |
| MPS | `x * y` | 12.12 | 0.12 | 12.07-12.20 | 82 |

MPS (Mac's GPU) performs about 86% faster than CPU on a simple element-wise tensor multiplication.