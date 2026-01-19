import torch
import torch.nn as nn
import numpy as np
from torch.utils.cpp_extension import load

# define path to c++ extensions
CPP_PATH = "./layernorm-systems/optimized"

# load c++ extensions
layernorm_ext = load(
    name="layernorm_module",  # module name
    sources=[
        f"{CPP_PATH}/extension.cpp",  # link to pybind signature
        f"{CPP_PATH}/forward.cpp",  # link to forward function definition
        f"{CPP_PATH}/backward.cpp"  # link to forward function definition
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
    x = torch.randn((n_samples, embedding_dim), dtype=torch.float64)  # pytorch tensor for built-in layernorm
    target = torch.randn((n_samples, embedding_dim), dtype=torch.float64)  # pytorch target tensor for loss
    gamma = torch.ones((embedding_dim), dtype=torch.float64)
    beta = torch.zeros((embedding_dim), dtype=torch.float64)
    
    return x, target, embedding_dim, gamma, beta

# initialize pytorch layernorm (built-in)
def pyTorchLayerNorm(dims):
    
    print("Initializing PyTorch LayerNorm function...")

    # initialize layer normalization (built-in via pytorch)
    ptln = nn.LayerNorm(dims)
    
    # set pytorch layernorm to use default ones/zeroes (similar to manual)
    with torch.no_grad():
        ptln.weight.data = torch.ones((dims), dtype=torch.float64)  # gamma for built-in
        ptln.bias.data = torch.zeros((dims), dtype=torch.float64)  # beta for built-in
    
    return ptln

# unit test for forward pass function (built-in vs manual)
def forwardPassUnitTest(x, ptln, gamma, beta):
    
    print("\n=== RUNNING FORWARD PASS UNIT TEST ===\n")
    pass_check = False        
            
    # call the built-in layernorm function
    output_pytorch = ptln(x)  # called on pytorch tensor    

    # call the manual layernorm forward pass function
    forw_result = layernorm_ext.forward(x, gamma, beta, epsilon=1e-5)
    output_manual_out = forw_result.output
    output_manual_cache = forw_result.cache
    output_manual_eps = forw_result.epsilon

    # convert both outputs to numpy arrays to compare
    output_manual_np = output_manual_out.detach().numpy()
    output_pytorch_np = output_pytorch.detach().numpy()

    # use assertion to check success or failure for matching values
    assert np.allclose(output_manual_np, output_pytorch_np, atol=1e-4, rtol=1e-7), f"FAILURE: Outputs do not match!"
    print("SUCCESS: Manual forward pass matches PyTorch!")
    pass_check = True  # update if assertion passed
    
    return pass_check, output_pytorch, output_manual_out, output_manual_cache, output_manual_eps

# unit test for backward pass function (built-in vs manual)
def backwardPassUnitTest(x, target, ptln, output_manual, cache_manual, epsilon):
    
    print("\n=== RUNNING BACKWARD PASS UNIT TEST ===\n")
    pass_check = False        
    
    # compute gradients for pytorch but with cloned tensors
    x_pt = x.clone().detach().requires_grad_(True)
    
    # ensure parameters are tracking gradients
    ptln.weight.requires_grad_(True)
    ptln.bias.requires_grad_(True)
    
    # reset gradients form any forward pass tests
    if x_pt.grad is not None:
        x_pt.grad.zero_()
    if ptln.weight.grad is not None:
        ptln.weight.grad.zero_()
    if ptln.bias.grad is not None:
        ptln.bias.grad.zero_()
        
    # call the built-in layernorm function
    output_pytorch_grad = ptln(x_pt)  # called on pytorch tensor    

    # compute built-in loss using mean squared error
    loss_pytorch = ((output_pytorch_grad - target) ** 2).sum()  # MSE sum
    
    # compute gradients from backward pass using loss
    loss_pytorch.backward()  # built-in back pass
    
    # extract gradient outputs from pytorch backward pass (w.r.t. input)
    dx_pytorch = x_pt.grad
    dgamma_pytorch = ptln.weight.grad  # internal gamma
    dbeta_pytorch = ptln.bias.grad  # internal beta

    # compute gradient of loss (w.r.t.) manual output (not loss itself)
    dout_manual = 2.0 * (output_manual - target)  # manually to do, pytorch does auto
        
    # call manual backward pass with forward output tensors
    back_result = layernorm_ext.backward(dout_manual, cache_manual, epsilon)
    dx_manual = back_result.dx
    dgamma_manual = back_result.dgamma
    dbeta_manual = back_result.dbeta

    # convert all outputs to numpy arrays to compare
    dx_pytorch_np = dx_pytorch.detach().numpy()
    dx_manual_np = dx_manual.detach().numpy()
    dgamma_pytorch_np = dgamma_pytorch.detach().numpy()
    dgamma_manual_np = dgamma_manual.detach().numpy()
    dbeta_pytorch_np = dbeta_pytorch.detach().numpy()
    dbeta_manual_np = dbeta_manual.detach().numpy()

    # use assertion to check success or failure for matching values
    assert np.allclose(dx_manual_np, dx_pytorch_np, atol=1e-4, rtol=1e-7), f"FAILURE: Output DX does not match!"
    assert np.allclose(dgamma_manual_np, dgamma_pytorch_np, atol=1e-4, rtol=1e-7), f"FAILURE: Output GAMMA does not match!"
    assert np.allclose(dbeta_manual_np, dbeta_pytorch_np, atol=1e-4, rtol=1e-7), f"FAILURE: Output BETA does not match!"
    print("SUCCESS: Manual backward pass matches PyTorch!")
    pass_check = True  # update if assertion passed
    
    return pass_check
        
# run through all unit tests
def runTests():
    
    print("\nStarting unit tests...")
    print(f"TEST CASES: {test_cases}\n")
    passed_fw = 0
    passed_bw = 0

    # loop through test cases and check tensors
    for case in test_cases:
        
        print("="*50)
        print(f"\nCASE: {case}")
        
        # create sample data and feed through passes to then check
        x, target, dims, gamma, beta = createSampleData(case)  # create input tensor
        ptln = pyTorchLayerNorm(dims)  # initialize pytorch layernorm
        
        # test forward pass
        pass_check_fw, _, output_manual, cache_manual, epsilon = forwardPassUnitTest(
            x, ptln, gamma, beta
        )
        
        # test backward pass
        pass_check_bw = backwardPassUnitTest(
            x, target, ptln, output_manual, cache_manual, epsilon
        )

        # update passed cases
        if pass_check_fw:
            passed_fw += 1
        if pass_check_bw:
            passed_bw += 1
                        
        print("\n" + "="*50)
        
    print(f"\n***** FW RESULTS: {passed_fw}/{len(test_cases)} forward pass cases passed! *****")
    print(f"***** BW RESULTS: {passed_fw}/{len(test_cases)} backward pass cases passed! *****")

if __name__ == "__main__":
    runTests()
