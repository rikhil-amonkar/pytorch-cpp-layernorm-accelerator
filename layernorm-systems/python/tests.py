import numpy as np
import torch
import torch.nn as nn
from autograd import LayerNorm
# from torch.autograd import gradchecks
import time

# run through test cases
def runTests(case):

    # limit randomness
    torch.manual_seed(42)

    # create sample data (track gradients)
    n, dims = case

    # ========== PYTORCH LAYERNORM ==========

    # pt input tensor data
    pt_x = torch.rand((n, dims), dtype=torch.float64, requires_grad=True)  # input
    pt_target = torch.rand((n, dims), dtype=torch.float64)  # target for loss
    
    # pt backward pass start timer
    pt_fw_start = time.perf_counter()

    # pytorch layer normalization function
    ptln = nn.LayerNorm(dims)
    
    # pt forward pass end timer
    pt_fw_end = time.perf_counter()

    # reset pytorch learn-param grads
    with torch.no_grad():
        ptln.weight.data = torch.ones(dims, dtype=torch.float64, requires_grad=True)  # pt gamma
        ptln.bias.data = torch.zeros(dims, dtype=torch.float64, requires_grad=True)  # pt beta
        
    # pass tensor through pt forward layernorm
    ptout = ptln(pt_x)

    # calculate pt loss (mse)
    ptloss = ((ptout - pt_target) ** 2).sum()  # mean sq err sum
    
    # pt backward pass end timer
    pt_bw_start = time.perf_counter()

    # pass loss grad through pt backward layernorm
    ptloss.backward()
    
    # pt backward pass end timer
    pt_bw_end = time.perf_counter()

    # extract output tensors from pt passes
    pt_dx = pt_x.grad  # dx
    pt_dgamma = ptln.weight.grad  # dgamma
    pt_dbeta = ptln.bias.grad  # dbeta

    # ========== MANUAL LAYERNORM ==========

    # manual input tensor data
    m_x = pt_x.clone().detach().to(torch.float64).requires_grad_(True)  # input (leaf)
    m_gamma = torch.ones((dims), dtype=torch.float64, requires_grad=True)  # scale
    m_beta = torch.zeros((dims), dtype=torch.float64, requires_grad=True)  # shift 
    m_target = pt_target.clone().detach().to(torch.float64)  # target for loss (leaf)
    
    # manual forward pass start timer
    m_fw_start = time.perf_counter()

    # pass through custom layernorm operation (forward pass)
    mout = LayerNorm.apply(m_x, m_gamma, m_beta, 1e-5)
    
    # manual forward pass start timer
    m_fw_end = time.perf_counter()

    # compute loss against target
    criterion = nn.MSELoss(reduction="sum")  # set mse sum (default is div)
    mloss = criterion(mout, m_target)
    
    # manual backward pass start timer
    m_bw_start = time.perf_counter()

    # call backward pass via gradient outputs
    mloss.backward()
    
    # manual backward pass end timer
    m_bw_end = time.perf_counter()

    # pass through autograd built-in check
    # test = gradcheck(LayerNorm.apply, (m_x, m_gamma, m_beta, 1e-5), eps=1e-6, atol=1e-4)
    # print("GRAD CHECK:", test)

    # ========== RESULTS ==========

    # print("=" * 50)

    # # pt results
    # print("PYTORCH RESULTS\n")
    # print("DX:\n", pt_dx.detach().numpy())
    # print("DGAMMA:\n", pt_dgamma.detach().numpy())
    # print("DBETA:\n", pt_dbeta.detach().numpy())

    # print("=" * 50)

    # # manual results
    # print("MANUAL RESULTS\n")
    # print("DX:\n", m_x.grad.detach().numpy())
    # print("DGAMMA:\n", m_gamma.grad.detach().numpy())
    # print("DBETA:\n", m_beta.grad.detach().numpy())

    # print("=" * 50)
    # print("=" * 50)

    # abs tol (atol) --> near-zero values, how close they are
    # rel tol (rtol) --> percent error (magnitude of num)

    # use assertion to check success or failure for matching values
    assert np.allclose(m_x.grad.detach().numpy(), pt_dx.detach().numpy(), atol=1e-7, rtol=1e-10), f"FAILURE: Output DX does not match!"
    assert np.allclose(m_gamma.grad.detach().numpy(), pt_dgamma.detach().numpy(), atol=1e-7, rtol=1e-10), f"FAILURE: Output GAMMA does not match!"
    assert np.allclose(m_beta.grad.detach().numpy(), pt_dbeta.detach().numpy(), atol=1e-7, rtol=1e-10), f"FAILURE: Output BETA does not match!"
    # print("SUCCESS: Manual backward pass matches PyTorch!")
    
    # time results
    print(f"PyTorch Baseline Runtime for Forward Pass: {(pt_fw_end - pt_fw_start):.4f} seconds.")
    print(f"Manual Baseline Runtime for Forward Pass: {(m_fw_end - m_fw_start):.4f} seconds.")
    print(f"PyTorch Baseline Runtime for Backward Pass: {(pt_bw_end - pt_bw_start):.4f} seconds.")
    print(f"Manual Baseline Runtime for Backward Pass: {(m_bw_end - m_bw_start):.4f} seconds.")
    print("="*50)

    # print("=" * 50)
        
if __name__ == "__main__":
    
    # sample test cases (baseline)
    cases = [
        (100, 512),
        (1000, 1024),
        (5000, 256)
    ]
    
    # loop through cases (super strict)
    for case in cases:
        print("="*50)
        print("CASE:", case)
        print("="*50)
        runTests(case)
    
    print("\nAll test cases succeeded!")
    
"""
RESULTS FROM TEST CASES (baseline)

==================================================
CASE: (100, 512)
==================================================
PyTorch Baseline Runtime for Forward Pass: 0.0002 seconds.
Manual Baseline Runtime for Forward Pass: 0.0006 seconds.
PyTorch Baseline Runtime for Backward Pass: 0.0112 seconds.
Manual Baseline Runtime for Backward Pass: 0.0013 seconds.
==================================================
==================================================
CASE: (1000, 1024)
==================================================
PyTorch Baseline Runtime for Forward Pass: 0.0003 seconds.
Manual Baseline Runtime for Forward Pass: 0.0087 seconds.
PyTorch Baseline Runtime for Backward Pass: 0.0156 seconds.
Manual Baseline Runtime for Backward Pass: 0.0236 seconds.
==================================================
==================================================
CASE: (5000, 256)
==================================================
PyTorch Baseline Runtime for Forward Pass: 0.0001 seconds.
Manual Baseline Runtime for Forward Pass: 0.0108 seconds.
PyTorch Baseline Runtime for Backward Pass: 0.0095 seconds.
Manual Baseline Runtime for Backward Pass: 0.0267 seconds.
==================================================

All test cases succeeded!
"""
    
    
    
    
