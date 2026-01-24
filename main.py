import numpy as np
import torch
import torch.nn as nn
from py.autograd import LayerNorm  # custom layernorm

# limit randomness
torch.manual_seed(42)

# run through test cases
def runTests(case):

    n, dims = case

    # input data (pytorch func)
    pt_x = torch.rand((n, dims), dtype=torch.float32, requires_grad=True)  # input
    pt_target = torch.rand((n, dims), dtype=torch.float32)  # target for loss
    
    ptln = nn.LayerNorm(dims)

    # reset params/grads
    with torch.no_grad():
        ptln.weight.data = torch.ones(dims, dtype=torch.float32, requires_grad=True)  # pt gamma
        ptln.bias.data = torch.zeros(dims, dtype=torch.float32, requires_grad=True)  # pt beta
                
    # pytorch forward pass
    ptout = ptln(pt_x)

    # compute loss
    ptloss = ((ptout - pt_target) ** 2).sum()

    # pytorch backward pass
    ptloss.backward()

    pt_dx = pt_x.grad  # dx
    pt_dgamma = ptln.weight.grad  # dgamma
    pt_dbeta = ptln.bias.grad  # dbeta

    # input data (custom func)
    m_x = pt_x.clone().detach().to(torch.float32).requires_grad_(True)  # input (leaf)
    m_gamma = torch.ones((dims), dtype=torch.float32, requires_grad=True)  # scale
    m_beta = torch.zeros((dims), dtype=torch.float32, requires_grad=True)  # shift 
    m_target = pt_target.clone().detach().to(torch.float32)  # target for loss (leaf)

    # custom forward pass
    mout = LayerNorm.apply(m_x, m_gamma, m_beta, 1e-5)

    # compute loss
    criterion = nn.MSELoss(reduction="sum")
    mloss = criterion(mout, m_target)

    # custom backward pass
    mloss.backward()

    assert np.allclose(m_x.grad.detach().numpy(), pt_dx.detach().numpy(), atol=1e-4, rtol=1e-4), f"FAILURE: Output DX does not match!"
    assert np.allclose(m_gamma.grad.detach().numpy(), pt_dgamma.detach().numpy(), atol=1e-4, rtol=1e-4), f"FAILURE: Output GAMMA does not match!"
    assert np.allclose(m_beta.grad.detach().numpy(), pt_dbeta.detach().numpy(), atol=1e-4, rtol=1e-4), f"FAILURE: Output BETA does not match!"
    
    return True
        
if __name__ == "__main__":
    
    # sample test cases
    cases = [
        (100, 512),
        (1000, 1024),
        (5000, 256)
    ]
    
    print("\nCustom C++ PyTorch Layernorm Extension!")
    print("\nRunning a test...")
    
    for case in cases:
        print("="*55)
        print("CASE:", case)
        print(f"PASSED:", runTests(case))  # evaluate
        print("="*55)
        
    print("\nThanks for using my LayerNorm! Feel free to play around with multiple test cases.")
    print("- Rikhil Amonkar")
    



    
    
    
    
