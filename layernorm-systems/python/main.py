import torch
import torch.nn as nn
import numpy as np
from torch.utils.cpp_extension import load

# define path to c++ extensions
CPP_PATH = "./layernorm-systems/src"

# load c++ extensions
forward_pass_ext = load(
    name="layernorm_module",  # module name
    sources=[
        f"{CPP_PATH}/extension.cpp",  # link to pybind signature
        f"{CPP_PATH}/forward.cpp"  # link to function definition
    ],
    verbose=True  # compiler output
)

# create list of special test cases
test_cases = [
    (1, 1), (2, 1), (1, 8), (8, 1),
    (3, 7), (16, 128), (40, 512),
    (17, 513), (64, 1024), (1024, 1)
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
    gamma = torch.ones(embedding_dim)
    beta = torch.zeros(embedding_dim)
    
    return x, embedding_dim, gamma, beta

# initialize pytorch layernorm (built-in)
def pyTorchLayerNorm(dims):
    
    print("Initializing PyTorch LayerNorm function...")

    # initialize layer normalization (built-in via pytorch)
    ptln = nn.LayerNorm(dims)
    
    # set pytorch layernorm to use default ones/zeroes (similar to manual)
    with torch.no_grad():
        ptln.weight.data = torch.ones(dims)  # gamma for built-in
        ptln.bias.data = torch.zeros(dims)  # beta for built-in
    
    return ptln

# unit test for forward pass function (built-in vs manual)
def forwardPassUnitTest(x, ptln, gamma, beta):
    
    print("\n=== RUNNING FORWARD PASS UNIT TEST ===\n")
    pass_check = False        
            
    # call the built-in layernorm function
    output_pytorch = ptln(x)  # called on pytorch tensor    

    # call the manual layernorm forward pass function
    result = forward_pass_ext.forward(x, gamma, beta, epsilon=1e-5)
    output_manual = result.output
    
    # convert both outputs to numpy arrays to compare
    output_manual_np = output_manual.detach().numpy()
    output_pytorch_np = output_pytorch.detach().numpy()

    # use assertion to check success or failure for matching values
    assert np.allclose(output_manual_np, output_pytorch_np, atol=1e-4, rtol=1e-7), f"FAILURE: Outputs do not match!"
    print("SUCCESS: Manual forward pass matches PyTorch!")
    pass_check = True  # update if assertion passed
    
    return pass_check
        
# run through all unit tests
def runTests():
    
    print("\nStarting unit tests...")
    print(f"TEST CASES: {test_cases}\n")
    passed = 0

    # loop through test cases and check tensors
    for case in test_cases:
        
        print("="*50)
        print(f"\nCASE: {case}")
        
        # create sample data and feed through passes to then check
        x, dims, gamma, beta = createSampleData(case)  # create input tensor
        ptln = pyTorchLayerNorm(dims)  # initialize pytorch layernorm
        pass_check = forwardPassUnitTest(x, ptln, gamma, beta)  # test forward pass
        
        # update passed cases
        if pass_check:
            passed += 1
            
        print("\n" + "="*50)
        
    print(f"\n***** RESULTS: {passed}/{len(test_cases)} cases passed! *****")
    
if __name__ == "__main__":
    runTests()
