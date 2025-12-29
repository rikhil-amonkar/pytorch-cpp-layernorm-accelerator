import torch
import torch.nn as nn
import numpy as np
from layernorm_manual_test import forward, backward  # import pass functions

# create list of special test cases
test_cases = [
    (1, 1),
    (2, 1),
    (1, 8),
    # (8, 1),  # hard case
    (3, 7),
    (16, 128),
    (40, 512),
    (17, 513),
    (64, 1024),
    # (1024, 1)  # hard case
]

# create sample data tensor inputs
def createSampleData(case):
    
    print("\nCreating sample data...")
    
    # set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # create sample data (samples, dimensions)
    n_samples, embedding_dim = case
    x = torch.randn(n_samples, embedding_dim)  # pytorch tensor for built-in layernorm
    
    return x, embedding_dim

# initialize pytorch layernorm (built-in)
def pyTorchLayerNorm(dims):
    
    print("Initializing PyTorch LayerNorm function...")

    # initialize layer normalization (built-in via pytorch)
    ptln = nn.LayerNorm(dims)
    
    return ptln

# unit test for forward pass function (built-in vs manual)
def forwardPassUnitTest(x, dims, ptln):
    
    print("\n=== RUNNING FORWARD PASS UNIT TEST ===")
    
    x_np = x.numpy()  # convert tensor to np for manual layernorm

    # set pytorch layernorm to use default ones/zeroes (similar to manual)
    with torch.no_grad():
        ptln.weight.data = torch.ones(dims)  # gamma for built-in
        ptln.bias.data = torch.zeros(dims)  # beta for built-in
        
    # call the built-in layernorm function
    output = ptln(x)  # called on pytorch tensor    
        
    # define manual gamma and beta from layernorm implementation (numpy)
    gamma = np.ones(dims)  # gamma for manual
    beta = np.zeros(dims)  # beta for manual

    # call the manual layernorm forward pass function
    output_manual, cache = forward(x_np, gamma, beta, epsilon=1e-5)

    # convert pytorch output to numpy for equal comparison
    output_pytorch = output.detach().numpy()

    # calculate max value difference between outputs
    max_diff = np.max(np.abs(output_manual - output_pytorch))
    
    # calculate means and stds for built-in and manual outputs
    torch_means = output_pytorch[0, :].mean()
    torch_stds = output_pytorch[0, :].std()
    man_means = output_manual[0, :].mean()
    man_stds = output_manual[0, :].std()
    
    print(f"FW Single Position TORCH | Mean: {torch_means:.4f}, STD: {torch_stds:.4f}")
    print(f"FW Single Position MANUAL | Mean: {man_means:.4f}, STD: {man_stds:.4f}")

    # use assertion to check success or failure for matching values
    assert np.allclose(output_manual, output_pytorch, atol=1e-5), f"FAILURE: Outputs do not match! Max diff: {max_diff}"  # checks if two np arrays are equal within a tolerance
    print("SUCCESS: Manual forward pass matches PyTorch!")
    
    return output_manual, cache
    
# unit test for backward pass function (built-in vs manual)
def backwardPassUnitTest(x, ptln, output_manual, cache):

    print("\n=== RUNNING BACKWARD PASS UNIT TEST ===")

    # create input that requires gradients
    x_np = x.numpy()  # convert tensor to np for manual layernorm
    x_torch_grad = torch.from_numpy(x_np).clone().requires_grad_(True)  # track gradients

    # call forward pass function with pytorch
    output_torch = ptln(x_torch_grad)

    # create dummy gradient for backward pass input
    dout_np = np.random.randn(*output_manual.shape)
    dout_torch = torch.from_numpy(dout_np)

    # call backward pass function from manual implementation
    dx_manual, dgamma_manual, dbeta_manual = backward(dout_np, cache)

    # call pytorch built-in backward pass function
    output_torch.backward(dout_torch)

    # compare gradients
    dx_pytorch = x_torch_grad.grad.numpy()
    dgamma_pytorch = ptln.weight.grad.numpy()
    dbeta_pytorch = ptln.bias.grad.numpy()
    
    # calculate means and stds for built-in and manual outputs
    torch_means = output_torch[0, :].mean()
    torch_stds = output_torch[0, :].std() if len(output_torch[0, :]) > 1 else torch.tensor(0.0)  # handle single-element edge case
    man_means = output_manual[0, :].mean()
    man_stds = output_manual[0, :].std() if len(output_manual[0, :]) > 1 else torch.tensor(0.0)  # handle single-element edge case
    
    print(f"BW Single Position TORCH | Mean: {torch_means:.4f}, STD: {torch_stds:.4f}")
    print(f"BW Single Position MANUAL | Mean: {man_means:.4f}, STD: {man_stds:.4f}")

    # use assertions on all outputs to compare values
    assert np.allclose(dx_manual, dx_pytorch, atol=1e-5), f"FAIL: dx does not match!"
    assert np.allclose(dgamma_manual, dgamma_pytorch, atol=1e-5), f"FAIL: dgamma does not match!"
    assert np.allclose(dbeta_manual, dbeta_pytorch, atol=1e-5), f"FAIL: dbeta does not match!"
    print("SUCCESS: Manual backward pass matches PyTorch!")
    
# run through all unit tests
def runTests():
    print("\nStarting unit tests...")
    print(f"TEST CASES: {test_cases}\n")
    for case in test_cases:
        print("="*50)
        print(f"\nCASE: {case}")
        x, dims = createSampleData(case)  # create input tensor
        ptln = pyTorchLayerNorm(dims)  # initialize pytorch layernorm
        output_manual, cache = forwardPassUnitTest(x, dims, ptln)  # test forward pass
        backwardPassUnitTest(x, ptln, output_manual, cache)  # test backward pass 
        print("\n" + "="*50)
    
if __name__ == "__main__":
    runTests()