#include <catch2/catch_test_macros.hpp>  // test case framework
#include <catch2/catch_approx.hpp>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <tuple>
#include "../forward_pass/forward_pass_layernorm.h"
#include "backward_pass_layernorm.h"
using namespace std;

// unit test using test case framework from catch2
TEST_CASE("Manual and PyTorch tensor outputs are computed") {

    // set random seed for reproducibility (same inputs every run)
    torch::manual_seed(42);

    cout << "\nUNIT TEST: Backward Pass LayerNorm (C++) - Manual vs. PyTorch" << endl;

    // create sample input tensors
    vector<tuple<int, int>> cases = {
        {1, 1},
        {2, 1},
        {1, 8},
        {8, 1},
        {3, 7},  // dx issue ()  --> tolerence
        {16, 128},  // dx issue ()  --> tolerence
        {40, 512},  // gamma issue () --> size issue
        {17, 513},  //  gamma issue () --> size issue
        {64, 1024},  // gamma issue () --> size issue
        {1024, 1}
    };

    // display all test cases
    cout << "\nTEST CASES:" << endl;
    for (size_t i = 0; i < cases.size(); i++) {  // size_t is unsigned, int is signed
        cout << "CASE " << i + 1 << ": [" << get<0>(cases[i]) << ", " << get<1>(cases[i]) << "]" << endl;  // get is used for tuple indexes
    }

    // iterate through cases and test each
    for (const auto& [n_samples, embedding_dims] : cases) {  // read-only, auto int, reference (not copy), [first, second], all in vector

        /* SAMPLE INPUT DATA FOR BOTH TESTS */

        // create sample input tensor based on case
        torch::Tensor x = torch::randn({n_samples, embedding_dims}, torch::dtype(torch::kFloat32).device(torch::kCPU));  // [-inf, inf] with mean 0

        // create sample learnable parameters
        torch::Tensor gamma = torch::ones(embedding_dims);
        torch::Tensor beta = torch::zeros(embedding_dims);

        /* BUILT-IN FORWARD AND BACKWARD PASS CALCULATIONS */

        // run tensor through built-in forward pass (pytorch with set options)
        vector<int64_t> normalized = {embedding_dims};
        torch::nn::LayerNormOptions options(normalized);
        options.eps(1e-5);  // epsilon
        torch::nn::LayerNorm layer_norm(options);
        layer_norm->weight.data() = torch::ones(embedding_dims);  // gamma
        layer_norm->bias.data() = torch::zeros(embedding_dims);  // beta
        x.requires_grad_(true);  // tensor gradient tracking
        layer_norm->weight.requires_grad_(true);  // gamma gradient tracking
        layer_norm->bias.requires_grad_(true);  // beta gradient tracking
        torch::Tensor torch_result = layer_norm->forward(x);  // forward pass (built-in)
        torch::Tensor loss = torch_result.sum();  // simple sum computation
        loss.backward();  // backward pass (built-in)

        // extract output gradients from built-in backward pass
        torch::Tensor dx_torch = x.grad();  // dx
        torch::Tensor dgamma_torch = layer_norm->weight.grad();  // dgamma
        torch::Tensor dbeta_torch = layer_norm->bias.grad();  // dbeta

        /* MANUAL FORWARD AND BACKWARD PASS CALCULATIONS */

        // run tensor through manual forward pass (libtorch)
        torch::Tensor x_manual = x.clone().detach();  // fresh copy of x
        forwardOutput manual_fw_res = forwardPassLayerNorm(x_manual, gamma, beta, 1e-5);
        torch::Tensor manual_output = manual_fw_res.output;  // output
        vector<torch::Tensor> manual_cache = manual_fw_res.cache;  // cache
        float manual_epsilon = manual_fw_res.epsilon;  // epsilon

        // determine mean and std of both results
        // float mean_torch = torch_result.mean().item<float>();
        // float std_torch = torch_result.std().item<float>();
        // float mean_manual = manual_output.mean().item<float>();
        // float std_manual = manual_output.std().item<float>();

        // create sample loss output with default of 1 (easy to compare)
        torch::Tensor manual_dout = torch::ones_like(manual_output);  // same size as output

        // run loss tensor through manual backward pass (libtorch)
        backwardOutput manual_bw_res = backwardPassLayerNorm(manual_dout, manual_cache, manual_epsilon);
        torch::Tensor dx_manual = manual_bw_res.dx;  // dx
        torch::Tensor dgamma_manual = manual_bw_res.dgamma;  // dgamma
        torch::Tensor dbeta_manual = manual_bw_res.dbeta;  // dbeta

        // print all tensors for debugging (means, stds, and backward pass outputs)
        cout << "\n===========================================" << endl;
        cout << "Input Tensor SIZE: " << x_manual.sizes() << endl;
        // cout << "===========================================" << endl;
        // cout << format("Manual FW LayerNorm MEAN: {:.4f} (should be ~0)", mean_manual) << endl;
        // cout << format("PyTorch FW LayerNorm MEAN: {:.4f} (should be ~0)", mean_torch) << endl;
        // cout << "===========================================" << endl;
        // cout << format("Manual FW LayerNorm STD: {:.4f} (should be ~1)", std_manual) << endl;
        // cout << format("PyTorch FW LayerNorm STD: {:.4f} (should be ~1)", std_torch) << endl;
        // cout << "===========================================" << endl;
        // cout << "Manual BW LayerNorm DX:\n" << dx_manual << endl;
        // cout << "PyTorch BW LayerNorm DX:\n" << dx_torch << endl;
        // cout << "===========================================" << endl;
        cout << "Manual BW LayerNorm DGAMMA:\n" << dgamma_manual.mean().item<float>() << endl;
        cout << "PyTorch BW LayerNorm DGAMMA:\n" << dgamma_torch.mean().item<float>() << endl;
        // cout << "===========================================" << endl;
        // cout << "Manual BW LayerNorm DBETA:\n" << dbeta_manual << endl;
        // cout << "PyTorch BW LayerNorm DBETA:\n" << dbeta_torch << endl;
        cout << "===========================================" << endl;

        /* COMPARE MANUAL VS. BUILT IN BACKWARD (AND FORWARD) PASS */

        auto diff = (dx_manual - dx_torch).abs();
        cout << "max dx diff = " << diff.max().item<float>() << endl;
        cout << "mean dx diff = " << diff.mean().item<float>() << endl;
        cout << "L2 dx diff = " << diff.norm().item<float>() << endl;

        // check output tensors against unit test catches (forward output)
        REQUIRE(manual_output.allclose(torch_result, 1e-4, 1e-7));  // check tensors with small error margin

        // compare output tensor gradients from backward pass methods
        REQUIRE(dx_manual.allclose(dx_torch, 1e-4, 1e-4));  // check tensors with small error margin
        REQUIRE(dgamma_manual.allclose(dgamma_torch, 1e-4, 1e-5));  // check tensors with small error margin
        REQUIRE(dbeta_manual.allclose(dbeta_torch, 1e-4, 1e-7));  // check tensors with small error margin

    }

}