# Minimal LayerNorm Forward + Backward via Python

## Execute and Validate Forward Pass Behavior (sample_layernorm_forward.py)
- Write a .py program using PyTorch to understand a sample data's forward pass through layer normalization.
- Set a random seed for reproducibility and verify PyTorch version.
- Create sample data via parameters for input tensor dimensions (batch_size, sequence_length, embedding_data).
- Initialize layer norm function and pass input tensor through.
- Check normalization for each batch and sequence to verify if mean averages around 0 and standard deviation sits around 1.
- Check all means and stds across all positions using the minimum and maximum values found in the output tensor. 

## Build and Execute Full LayerNorm Manually with Forward + Backward Pass (layernorm_manual_test.py)

### Important: LayerNorm vs BatchNorm - Key Difference
- **LayerNorm normalizes across the FEATURE dimension (last axis, axis=-1)**, NOT the batch dimension.
- **BatchNorm normalizes across the BATCH dimension (first axis, axis=0)**.
- For input shape (batch_size, sequence_length, embedding_dim):
  - LayerNorm: Each (batch, sequence) position gets its own mean/variance computed over embedding_dim features.
  - BatchNorm: Each feature gets one mean/variance computed over all (batch, sequence) positions.
- **All operations (mean, variance, gradient sums) use axis=-1 (last axis) for LayerNorm**, not axis=0.
- The mathematical operations are identical, but the axis of normalization is different.

- Create sample data via parameters for input tensor dimensions (batch_size, sequence_length, embedding_data).
- Write forward pass function which takes in the sample tensor, gamma, beta, and the epsilon constant values as input.
    - Step 1: Get the shape of the tensor and then calculate the mean of the input tensor across the feature dimension (last axis) - mean = mean(x, axis=-1, keepdims=True).
    - Step 2: Calculate the mean-centered input (xmu) - subtract mean from input: xmu = x - mean.
    - Step 3: Calculate the squared deviations (sq) - square the mean-centered values: sq = xmu^2.
    - Step 4: Calculate the variance (var) - mean of squared deviations across feature dimension: var = mean(sq, axis=-1, keepdims=True).
    - Step 5: Add epsilon constant to variance to prevent division by zero: var_eps = var + eps.
    - Step 6: Calculate the square root of variance (sqrtvar) - sqrtvar = sqrt(var_eps).
    - Step 7: Calculate the inverse variance (ivar) - ivar = 1 / sqrtvar.
    - Step 8: Normalize the input (xhat) - multiply mean-centered input by inverse variance: xhat = xmu * ivar.
    - Step 9: Scale with learnable parameter gamma (gammax) - multiply normalized input by gamma: gammax = gamma * xhat.
    - Step 10: Shift with learnable parameter beta (output) - add beta to scaled values: output = gammax + beta.
    - Store all intermediate values (xhat, gamma, xmu, ivar, sqrtvar, var, eps, mean, x) into a cache for backward pass.
    - Return output and cache.
- Write the backward pass function which takes in dout (gradient of loss w.r.t. output) and the cache from the forward pass as input.
    - Unfold the variables that are stored in the cache (xhat, gamma, xmu, ivar, sqrtvar, var, eps, mean, x).
    - Get the dimensions of the input/output (usually same: batch_size, sequence_length, embedding_dim).
    - Step 1: Calculate gradient w.r.t. beta (dbeta) - sum dout across the feature dimension (axis=-1, keepdims=True) since output = gamma * xhat + beta. NOTE: This is axis=-1 for LayerNorm, NOT axis=0 like BatchNorm.
    - Step 2: Calculate gradient w.r.t. gamma (dgamma) - sum (dout * xhat) across the feature dimension (axis=-1, keepdims=True) since output = gamma * xhat + beta. NOTE: This is axis=-1 for LayerNorm, NOT axis=0 like BatchNorm.
    - Step 3: Calculate gradient w.r.t. xhat (dxhat) - multiply dout by gamma since the gradient flows through the scaling operation.
    - Step 4: Calculate gradient w.r.t. inverse variance (divar) - sum (dxhat * xmu) across the feature dimension (axis=-1, keepdims=True) since xhat = xmu * ivar. NOTE: axis=-1 for LayerNorm.
    - Step 5: Calculate first component of gradient w.r.t. xmu (dxmu1) - multiply dxhat by ivar.
    - Step 6: Calculate gradient w.r.t. sqrtvar (dsqrtvar) - apply chain rule: -1/(sqrtvar^2) * divar since ivar = 1/sqrtvar.
    - Step 7: Calculate gradient w.r.t. variance (dvar) - apply chain rule: 0.5 * 1/sqrt(var+eps) * dsqrtvar since sqrtvar = sqrt(var+eps).
    - Step 8: Calculate gradient w.r.t. squared deviations (dsq) - broadcast dvar with factor 1/D where D is the feature dimension size.
    - Step 9: Calculate second component of gradient w.r.t. xmu (dxmu2) - multiply 2 * xmu * dsq since sq = xmu^2.
    - Step 10: Combine gradient components for xmu (dxmu = dxmu1 + dxmu2) and calculate gradient w.r.t. mean (dmu) - negative sum of dxmu across feature dimension (axis=-1, keepdims=True) since xmu = x - mean. NOTE: axis=-1 for LayerNorm.
    - Step 11: Calculate gradient component from mean computation (dx2) - broadcast dmu with factor 1/D across feature dimension.
    - Step 12: Calculate final gradient w.r.t. input (dx) - sum dxmu and dx2: dx = dxmu + dx2.
    - Return dx, dgamma, dbeta.

## Create a Unit Test to Compare Built-In LayerNorm vs Manual Implementation
- Define forward pass and backward pass functions as well as PyTorch versions.
- Set random seeds for reproducibility and define test tensor shapes.
- Test the forward pass function.
    - Create an input NumPy data array.
    - Convert to a PyTorch tensor.
    - Run the forward function.
    - Run the PyTorch LayerNorm built-in function.
    - Compare the outputs using assertions.
    - Print success/failure messages.
- Test the backward pass function.
    - Create input data with requires_grad=True for PyTorch.
    - Run forward on both functions.
    - Create dummy output gradient.
    - Run backward pass on both built-in and manual functions.
    - Compare all three gradients (dx, dgamma, dbeta).
    - Use assertions to verify matches.
- Test property values.
    - Test that mean averages to about 0.
    - Test that std sits around 1.