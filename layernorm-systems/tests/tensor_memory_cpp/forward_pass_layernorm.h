#ifndef FORWARD_PASS_LAYERNORM_H
#define FORWARD_PASS_LAYERNORM_H

#include <torch/torch.h>
#include <vector>
using namespace std;

// structure to group multiple data type returns
struct forwardOutput {
    torch::Tensor output;
    vector<float> mu;
    vector<float> var;
    vector<float> std;
    vector<float> ivar;
    vector<torch::Tensor> cache_tensors;
    float epsilon;
};

// forward pass function decleration
forwardOutput forwardPassLayerNorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float epsilon = 1e-5);

#endif  // FORWARD_PASS_LAYERNORM_H