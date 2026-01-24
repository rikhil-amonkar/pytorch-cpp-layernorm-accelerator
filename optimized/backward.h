#ifndef BACKWARD_PASS_LAYERNORM_H
#define BACKWARD_PASS_LAYERNORM_H

#include <torch/torch.h>
#include <vector>

using namespace std;

// structure to group multiple data type returns
struct backwardOutput {
    torch::Tensor dx;
    torch::Tensor dgamma;
    torch::Tensor dbeta;
};

// Backward pass function decleration
backwardOutput backwardPassLayerNorm(torch::Tensor dout, vector<torch::Tensor> cache, float epsilon);

#endif  // BACKWARD_PASS_LAYERNORM_H