import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

# set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# check pytorch version
print(f"\nPyTorch version: {torch.__version__}")

# create sample data (batch_size, sequence_length, embedding_data)
batch_size, seq_length, embedding_dim = 4, 10, 512
x = torch.randn(batch_size, seq_length, embedding_dim)

# initialize layer normalization (normalize over last dimension)
ln = nn.LayerNorm(embedding_dim)

# apply normilization
output = ln(x)

# print input and output shapes (test)
print(f"\nInput Shape: {x.shape}")
print(f"Outut Shape: {output.shape}")

# check normalization (each batch and sequence position)
sample_mean = output[0, 0, :].mean()   # first sample, first position (should average ~0)
sample_std = output[0, 0, :].std()  # first sample, first position (should average ~1)

# print single position stats for mean and std (test)
print(f"\nSingle Position Mean: {sample_mean:.4f} (should be around 0)")
print(f"Single Position STD: {sample_std:.4f} (should be around 1)")

# verify across all positions
all_means = output.mean(dim=-1)  # mean across embedding dimension
all_stds = output.std(dim=-1)  # std across embedding dimension

# print all position means and stds (test)
print(f"\nAll Position Means - Min: {all_means.min():.4f}, Max: {all_means.max():.4f}")
print(f"All Position STDs - Min: {all_stds.min():.4f}, Max: {all_stds.max():.4f}")


