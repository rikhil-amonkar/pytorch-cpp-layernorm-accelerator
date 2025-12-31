#include <catch2/catch_test_macros.hpp>  // test case framework
#include <catch2/catch_approx.hpp>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <tuple>
#include "forward_pass_layernorm.h"
using namespace std;

// unit test using test case framework from catch2
TEST_CASE("Manual and PyTorch tensor outputs are computed") {

    cout << "\nUNIT TEST: Forward Pass LayerNorm (C++) - Manual vs. PyTorch" << endl;

    // create sample input tensors
    vector<tuple<int, int>> cases = {
        // {1, 1},
        // {2, 1},
        // {1, 8},
        // {8, 1},  // hard case
        // {3, 7},
        // {16, 128},
        {40, 512},
        // {17, 513},
        // {64, 1024},
        // {1024, 1}  // hard case}
    };

    // iterate through cases and test each
    for (const auto& [n_samples, embedding_dims] : cases) {  // read-only, auto int, reference (not copy), [first, second], all in vector

        // create sample input tensor based on case
        torch::Tensor x = torch::rand({n_samples, embedding_dims}, torch::dtype(torch::kFloat32).device(torch::kCPU));

        // create sample learnable parameters
        torch::Tensor gamma = torch::ones(embedding_dims);
        torch::Tensor beta = torch::zeros(embedding_dims);

        // run tensor through built-in forward pass (pytorch)
        vector<int64_t> normalized = {embedding_dims};
        torch::nn::LayerNormOptions options(normalized);
        torch::nn::LayerNorm layer_norm(options);
        layer_norm->weight.data() = torch::ones(embedding_dims);  // gamma
        layer_norm->bias.data() = torch::zeros(embedding_dims);  // beta
        torch::Tensor torch_result = layer_norm->forward(x);

        // run tensor through manual forward pass (libtorch)
        torch::Tensor manual_result = forwardPassLayerNorm(x, gamma, beta).output;

        // determine mean and std of both results
        float mean_torch = torch_result[0].mean().item<float>();
        float std_torch = torch_result[0].std().item<float>();
        float mean_manual = manual_result[0].mean().item<float>();
        float std_manual = manual_result[0].std().item<float>();

        // display results
        cout << "\n===========================================" << endl;
        cout << "Input Tensor SIZE: " << x.sizes() << endl;
        cout << "===========================================" << endl;
        cout << format("PyTorch FW LayerNorm MEAN: {:.4f} (should be ~0)", mean_torch) << endl;
        cout << format("Manual FW LayerNorm MEAN: {:.4f} (should be ~0)", mean_manual) << endl;
        cout << "===========================================" << endl;
        cout << format("PyTorch FW LayerNorm STD: {:.4f} (should be ~1)", std_torch) << endl;
        cout << format("Manual FW LayerNorm STD: {:.4f} (should be ~1)", std_manual) << endl;
        cout << "===========================================" << endl;

        // check output tensors against unit test catches
        REQUIRE(manual_result.allclose(torch_result, 1e-5, 1e-8));  // check tensors with small error margin

    }

}