#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "forward.h"
using namespace std;

// function definition for performing forward pass on tensor via layer norm
forwardOutput forwardPassLayerNorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float epsilon) {

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

    // initialize a cache vectors to output calculation info
    vector<torch::Tensor> cache_tensors;

    // create initial data pointers for tensors (input/output)
    float *ptr_x = x.data_ptr<float>();
    float *ptr_out = output.data_ptr<float>();

    // initialize calc vectors needed per-row/group
    vector<float> mu(n);  // mean
    vector<float> var(n);  // variance
    vector<float> std(n);  // squared variance (std)
    vector<float> ivar(n);  // inverse variance

    // initialize calc tensors needed per-feature for each row/group
    torch::Tensor xmu = torch::empty_like(x);  // center mean
    torch::Tensor sq = torch::empty_like(x);  // variance prep
    torch::Tensor xhat = torch::empty_like(x);  // normalization

    // initialize data pointers for each calc tensor
    float *ptr_xmu = xmu.data_ptr<float>();
    float *ptr_sq = sq.data_ptr<float>();
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
        mu[i] = mu_sum / dims;  // store mean

        // subtract mean from each feature in row (shift) and prep for std
        for (int j = 0; j < dims; j++) {
            ptr_xmu[(i * dims) + j] = ptr_x[(i * dims) + j] - mu[i];  // subtract
            ptr_sq[(i * dims) + j] = ptr_xmu[(i * dims) + j] * ptr_xmu[(i * dims) + j];  // square each center feature to prep for variance
        }

        // calculate variance across features in dimension (spread of features)
        float var_sum = 0.0f;  // track current row sum
        for (int j = 0; j < dims; j++) {
            var_sum += ptr_sq[(i * dims) + j];  // sum square center means
        }
        var[i] = var_sum / dims;  // store variance

        // add numerical stability via epsilon constant (convert var to std)
        std[i] = sqrt((var[i] + epsilon));  // add constant (prevent div by 0)

        // invert standard deviation (for each row)
        ivar[i] = 1 / std[i];

        // execute normalization and apply learnable parameters
        for (int j = 0; j < dims; j++) {
            ptr_xhat[(i * dims) + j] = ptr_xmu[(i * dims) + j] * ivar[i];
            ptr_out[(i * dims) + j] = (ptr_gam[j] * ptr_xhat[(i * dims) + j]) + ptr_bet[j];  // gamma and beta only live on last dim
        }

    }

    // store result tensors in cache after changes (for back pass)
    cache_tensors.insert(end(cache_tensors), {xhat, xmu, sq, x, gamma, beta});  // add tensor cache to end of vector

    return {output, mu, var, std, ivar, cache_tensors, epsilon};  // return struct data types (output, floats (vectors), tensors)

}
