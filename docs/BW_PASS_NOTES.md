# Backward Pass + Autograd for LayerNorm in C++

## Forward Pass Math Recap

# Math Context

1. Compute mean
    - Average value of all the features.
2. Compute variance
    - Check how far features are spread apart from the mean.
3. Normalize
    - Scale features to have a mean of 0 and unit variance.
4. Scale and shift
    - Learnable parameters to let the neural network undo normalization if needed.

## Backward Pass Math

# Math Behind Backpass (backward_pass_layernorm.cpp)

1. Gradients for the learnable parameters (gamma, beta)
    - This is known as linear scaling.
    - The derivative just passes through or multiplies by the other input of multiplication.
    - Chain rule is a big component in the calculation.
2. Gradient through normalization
    - Local gradient with respect to the numerator.
    - Local gradient with respect to the denominator.
    - Chain rule applications.
3. Gradient with respect to variance
    - Derivative with respect to variance.
    - Propogate how changing variance changes normalization.
    - Chain rule applications.
4. Gradient with respect to mean
    - Mean appears in numerator and in variance.
    - Derivative with respect to mean.
    - Increasing mean shifts all normalized features down.
5. Gradient with respect to input tensor
    - Direct derivative through numerator.
    - Indirect derivative through variance.
    - Indirect derivative through mean.
    - Summing mean and variance contributions gives the total gradient.

# Explanation of Selected Dimensions for Forward vs. Backward

Forward pass uses an axis of -1 (last dimension) because LayerNorm normalizes across features, with mean being calculated per sample across features, and variance also per sample across features.

Backward pass (gradient accumulation) instead uses the axis of 0 (first dimension) due to the parameter gradients (dgamma, dbeta) because you sum across samples. The sum across samples to get one gradient per features leads to dbeta as well as dgamma. Backpass still uses the last dimension on the axis of -1 for intermediate normalization gradients such as divar and dmu (variance and mean). 