#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "forward_pass_layernorm.h"
using namespace std;

// function definition for performing forward pass on tensor via layernorm
forwardOutput forwardPassLayerNorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float epsilon) {

    // validate tensors against input contract (rules)
    TORCH_CHECK(gamma.numel() == x.size(-1), "Gamma dimensions are unequal to last dimension of input tensor.");  // gamma lives on last dimension
    TORCH_CHECK(beta.numel() == x.size(-1), "Beta dimensions are unequal to last dimension of input tensor.");  // gamma lives on last dimension
    TORCH_CHECK(x.dtype() == gamma.dtype() && x.dtype() == beta.dtype(), "Input tensor and learnable parameters have unequal data types.");  // data pointer needs same data type
    TORCH_CHECK(x.device() == gamma.device() && x.device() == beta.device(), "Input tensor and learnable parameters are not located on the same devices.");  // need same memory location to act on
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(), "Input tensor or learnable parameters are not contiguous.");  // data layout must be same in order to access 

    // derive n-samples and last dimension (used for normalization)
    int dims = x.size(-1);  // last
    int n = x.numel() / dims;  // all except last

    // initialize an output tensor to store results
    torch::Tensor output = torch::empty_like(x);  // contract copy of input

    // initialize a cache vectors to output calculation info
    vector<torch::Tensor> cache{};  // empty

    // create initial data pointers for tensors (input/output)
    float *ptr_x = x.data_ptr<float>();
    float *ptr_out = output.data_ptr<float>();

    // create calc tensors needed per-row/group
    torch::Tensor mu = torch::empty({n}, x.options());  // mean
    torch::Tensor var = torch::empty({n}, x.options());  // variance
    torch::Tensor sqrtvar = torch::empty({n}, x.options());  // squared variance (std)
    torch::Tensor ivar = torch::empty({n}, x.options());  // inverse variance

    // create calc tensors needed per-feature for each row/group
    torch::Tensor xmu = torch::empty_like(x);  // center mean
    torch::Tensor sq = torch::empty_like(x);  // variance prep
    torch::Tensor xhat = torch::empty_like(x);  // normalization

    // initialize data pointers for each calc tensor
    float *ptr_mu = mu.data_ptr<float>();  // 1-D (N,)
    float *ptr_var = var.data_ptr<float>();  // 1-D (N,)
    float *ptr_sqrtvar = sqrtvar.data_ptr<float>();  // 1-D (N,)
    float *ptr_ivar = ivar.data_ptr<float>();  // 1-D (N,)
    float *ptr_xmu = xmu.data_ptr<float>();  // 2-D (N, D)
    float *ptr_sq = sq.data_ptr<float>();  // 2-D (N, D)
    float *ptr_xhat = xhat.data_ptr<float>();  // 2-D (N, D)

    // initialize data pointers for learnable parameter tensors
    float *ptr_gam = gamma.data_ptr<float>();  // 1-D (D,)
    float *ptr_bet = beta.data_ptr<float>();  // 1-D (D,)

    // iterate through all groups/samples (rows)
    for (int i = 0; i < n; i++) {

        // calculate mean across features in dimension (center data, remove bias)
        float mu_sum = 0.0f;  // track current row sum
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
        float var_sum = 0.0f;  // track current row sum
        for (int j = 0; j < dims; j++) {
            var_sum += ptr_sq[(i * dims) + j];  // sum square center means
        }
        ptr_var[i] = var_sum / dims;  // store variance

        // add numerical stability via epsilon constant (convert var to sqrtvar)
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

    // cache_tensors.push_back(xhat);
    // cache_tensors.push_back(xmu);
    // cache_tensors.push_back(sq);
    // cache_tensors.push_back(x);
    // cache_tensors.push_back(gamma);
    // cache_tensors.push_back(beta);

    // (output tensor, learnable tensor, 3 calc tensors, 2 calc vectors, float constant)
    return {output, cache, epsilon};  // return struct data types (output, floats (vectors), tensors)

}

// int sample() {
    
//     cout << "\nForward Pass Function (LayerNorm C++)!" << endl;

//     // create sample input tensor
//     int n_samples = 40, embedding_dims = 512;
//     torch::Tensor x = torch::rand({n_samples, embedding_dims}, torch::dtype(torch::kFloat32).device(torch::kCPU));  // 5x3, float32, cpu
//     torch::Tensor gamma = torch::ones(embedding_dims);  // gamma
//     torch::Tensor beta = torch::zeros(embedding_dims);  // beta 

//     // call forward pass
//     forwardOutput result = forwardPassLayerNorm(x, gamma, beta);  // struct result type

//     // extract mean and std to check values across first row
//     float mean = result.output[0].mean().item<float>();
//     float std = result.output[0].std().item<float>();

//     // print results
//     cout << "\n===================================" << endl;
//     cout << "Input Tensor SIZE: " << x.sizes() << endl;
//     cout << format("Output Tensor MEAN: {:.4f} (should be ~0)", mean) << endl;
//     cout << format("Output Tensor STD: {:.4f} (should be ~1)", std) << endl;
//     cout << "===================================\n" << endl;

//     return 0;

// }