#include <torch/torch.h>
#include <iostream>
#include <cstdlib>
#include <backward.h>

using namespace std;

// function definition for performing backward pass on output and cache via layernorm (with CUDA)
backwardOutput backwardPassLayerNormCUDA(torch::Tensor dout, vector<torch::Tensor> cache, float epsilon) {

    // warn about cuda inavailability (just template for now)
    cerr << "WARNING: CUDA is not yet supported for this Layernorm, but will be used in the future. Use CPU instead!" << endl;  // cuda warning (change in future)
    exit(1);  // default return

}