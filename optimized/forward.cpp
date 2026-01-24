#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include "forward.h"

using namespace std;

// function definition for performing forward pass on tensor via layer norm
forwardOutput forwardPassLayerNormCPU(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float epsilon) {

    // validate tensors against input contract (rules)
    TORCH_CHECK(gamma.numel() == x.size(-1), "Gamma dimensions are unqual to last dimension of input tensor.")  // gamma lives on last dimension
    TORCH_CHECK(beta.numel() == x.size(-1), "Beta dimensions are unqual to last dimension of input tensor.")  // gamma lives on last dimension
    TORCH_CHECK(x.dtype() == gamma.dtype() && x.dtype() == beta.dtype(), "Input tensor and learnable parameters have unequal data types.")  // data pointer needs same data type
    TORCH_CHECK(x.device() == gamma.device() && x.device() == beta.device(), "Input tensor and learnable parameters are not located on the same devices.")  // need same memory location to act on
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(), "Input tensor or learnable parameters are not contiguous.")  // data layout must be same in order to access 

    // derive n-samples and last dimension (used for normalization)
    int dims = x.size(-1);  // last
    int n = x.numel() / dims;  // all except last

    // initialize an output tensor to store results
    torch::Tensor output = torch::empty_like(x);  // contract copy of input

    // create a cache vector to store intermediate operation results
    vector<torch::Tensor> cache{};  // empty

    // create initial data pointers for tensors (input/output)
    float *ptr_x = x.data_ptr<float>();
    float *ptr_out = output.data_ptr<float>();

    // create calc tensors needed per-row/group
    torch::Tensor mu = torch::empty({n}, x.dtype());  // mean
    torch::Tensor sqrtvar = torch::empty({n}, x.dtype());;  // squared variance (std)
    torch::Tensor ivar = torch::empty({n}, x.dtype());  // inverse variance

    // create calc tensors needed per-feature for each row/group
    torch::Tensor xmu = torch::empty_like(x);  // center mean
    torch::Tensor xhat = torch::empty_like(x);  // normalization

    // initialize data pointers for each calc tensor
    float *ptr_mu = mu.data_ptr<float>();
    float *ptr_sqrtvar = sqrtvar.data_ptr<float>();
    float *ptr_ivar = ivar.data_ptr<float>();
    float *ptr_xmu = xmu.data_ptr<float>();
    float *ptr_xhat = xhat.data_ptr<float>();

    // initialize data pointers for learnable parameter tensors
    float *ptr_gam = gamma.data_ptr<float>();
    float *ptr_bet = beta.data_ptr<float>();

    // iterate through all groups/samples (rows)
    for (int i = 0; i < n; i++) {

        // calculate mean across features in dimension (center data, remove bias)
        float mu_sum = 0.0f;  // track current row sum
        for (int j = 0; j < dims; j++) {
            mu_sum += ptr_x[(i * dims) + j];  // move past prev rows then correct column
        }
        ptr_mu[i] = mu_sum / dims;  // store mean

        // subtract mean from each feature in row (shift) and prep for std
        float var_sum = 0.0f;  // track current row sum
        for (int j = 0; j < dims; j++) {
            ptr_xmu[(i * dims) + j] = ptr_x[(i * dims) + j] - ptr_mu[i];  // subtract
            var_sum += (ptr_xmu[(i * dims) + j]) * (ptr_xmu[(i * dims) + j]);  // sum square center means
        }

        // add numerical stability via epsilon constant (convert var to std)
        ptr_sqrtvar[i] = sqrt(((var_sum / dims) + epsilon));  // add constant (prevent div by 0)

        // invert standard deviation (for each row)
        ptr_ivar[i] = 1.0f / ptr_sqrtvar[i];

        // execute normalization and apply learnable parameters
        for (int j = 0; j < dims; j++) {
            ptr_xhat[(i * dims) + j] = ptr_xmu[(i * dims) + j] * ptr_ivar[i];
            ptr_out[(i * dims) + j] = (ptr_gam[j] * ptr_xhat[(i * dims) + j]) + ptr_bet[j];  // gamma and beta only live on last dim
        }

    }

    // append intermediate operation tensors to cache vector
    cache.insert(end(cache), {gamma, xhat, xmu, sqrtvar, ivar});  // add tensor cache to end of vector

    // (output tensor, intermediate tensors, eps constant)
    return {output, cache, epsilon};  // return struct data types

}

// function to check device and pass based on cpu vs cuda (cuda in future)
forwardOutput forwardPassLayerNorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float epsilon) {

    // check which device tensor was stored on
    if (x.device().is_cpu()) {
        return forwardPassLayerNormCPU(x, gamma, beta, epsilon);  // cpu function
    } else if (x.device().is_cuda()) {
        cerr << "WARNING: CUDA is not yet implemented. Will use in future. Use CPU instead!" << endl;  // warning incase cuda
        exit(1);  // default return
    } else {
        cerr << "WARNING: Unsupported device type. Will not be able to run program. Use CPU instead!" << endl;  // incorrect type case
        exit(1);  // default return
    }

}
