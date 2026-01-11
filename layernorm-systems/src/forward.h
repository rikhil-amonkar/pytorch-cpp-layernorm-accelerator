#ifndef FORWARD_PASS_LAYERNORM_H
#define FORWARD_PASS_LAYERNORM_H

#include <torch/torch.h>
#include <vector>

using namespace std;

// structure to group multiple data type returns
struct forwardOutput {
    torch::Tensor output;
    vector<torch::Tensor> cache;
    double epsilon;
};

// forward pass function decleration
forwardOutput forwardPassLayerNorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double epsilon = 1e-5);

#endif  // FORWARD_PASS_LAYERNORM_H