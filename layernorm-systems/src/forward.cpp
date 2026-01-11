#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "forward.h"

using namespace std;

// function definition for performing forward pass on tensor via layer norm
forwardOutput forwardPassLayerNorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double epsilon) {

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
    double *ptr_x = x.data_ptr<double>();
    double *ptr_out = output.data_ptr<double>();

    // create calc tensors needed per-row/group
    torch::Tensor mu = torch::empty_like(x);  // mean
    torch::Tensor var = torch::empty_like(x);  // variance
    torch::Tensor sqrtvar = torch::empty_like(x);  // squared variance (std)
    torch::Tensor ivar = torch::empty_like(x);  // inverse variance

    // create calc tensors needed per-feature for each row/group
    torch::Tensor xmu = torch::empty_like(x);  // center mean
    torch::Tensor sq = torch::empty_like(x);  // variance prep
    torch::Tensor xhat = torch::empty_like(x);  // normalization

    // initialize data pointers for each calc tensor
    double *ptr_mu = mu.data_ptr<double>();
    double *ptr_var = var.data_ptr<double>();
    double *ptr_sqrtvar = sqrtvar.data_ptr<double>();
    double *ptr_ivar = ivar.data_ptr<double>();
    double *ptr_xmu = xmu.data_ptr<double>();
    double *ptr_sq = sq.data_ptr<double>();
    double *ptr_xhat = xhat.data_ptr<double>();

    // initialize data pointers for learnable parameter tensors
    double *ptr_gam = gamma.data_ptr<double>();
    double *ptr_bet = beta.data_ptr<double>();

    // iterate through all groups/samples (rows)
    for (int i = 0; i < n; i++) {

        // calculate mean across features in dimension (center data, remove bias)
        double mu_sum = 0.0f;  // track current row sum
        for (int j = 0; j < dims; j++) {
            mu_sum += ptr_x[(i * dims) + j];  // move past prev rows then correct column
        }
        ptr_mu[i] = mu_sum / dims;  // store mean

        // subtract mean from each feature in row (shift) and prep for std
        for (int j = 0; j < dims; j++) {
            ptr_xmu[(i * dims) + j] = ptr_x[(i * dims) + j] - ptr_mu[i];  // subtract
            ptr_sq[(i * dims) + j] = ptr_xmu[(i * dims) + j] * ptr_xmu[(i * dims) + j];  // square each center feature to prep for variance
        }

        // calculate variance across features in dimension (spread of features)
        double var_sum = 0.0f;  // track current row sum
        for (int j = 0; j < dims; j++) {
            var_sum += ptr_sq[(i * dims) + j];  // sum square center means
        }
        ptr_var[i] = var_sum / dims;  // store variance

        // add numerical stability via epsilon constant (convert var to std)
        ptr_sqrtvar[i] = sqrt((ptr_var[i] + epsilon));  // add constant (prevent div by 0)

        // invert standard deviation (for each row)
        ptr_ivar[i] = 1 / ptr_sqrtvar[i];

        // execute normalization and apply learnable parameters
        for (int j = 0; j < dims; j++) {
            ptr_xhat[(i * dims) + j] = ptr_xmu[(i * dims) + j] * ptr_ivar[i];
            ptr_out[(i * dims) + j] = (ptr_gam[j] * ptr_xhat[(i * dims) + j]) + ptr_bet[j];  // gamma and beta only live on last dim
        }

    }

    // append intermediate operation tensors to cache vector
    cache.insert(end(cache), {gamma, xhat, xmu, sqrtvar, ivar, var});  // add tensor cache to end of vector

    // (output tensor, intermediate tensors, eps constant)
    return {output, cache, epsilon};  // return struct data types

}
