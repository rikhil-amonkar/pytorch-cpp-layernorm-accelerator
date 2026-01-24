#include <torch/torch.h>
#include <iostream>
#include <cstdlib>
#include <forward.h>

using namespace std;

// function definition for performing forward pass on input tensors via layernorm (with CUDA)
forwardOutput forwardPassLayerNormCUDA(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float epsilon) {

    // warn about cuda inavailability (just template for now)
    cerr << "WARNING: CUDA is not yet supported for this Layernorm, but will be used in the future. Use CPU instead!" << endl;  // cuda warning (change in future)
    exit(1);  // default return

}