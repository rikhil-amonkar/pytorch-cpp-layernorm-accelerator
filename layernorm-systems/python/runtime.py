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
    pt_x = torch.rand((n, dims), dtype=torch.float32, requires_grad=True)  # input
    pt_target = torch.rand((n, dims), dtype=torch.float32)  # target for loss
    
    # pytorch layer normalization function
    ptln = nn.LayerNorm(dims)

    # reset pytorch learn-param grads
    with torch.no_grad():
        ptln.weight.data = torch.ones(dims, dtype=torch.float32, requires_grad=True)  # pt gamma
        ptln.bias.data = torch.zeros(dims, dtype=torch.float32, requires_grad=True)  # pt beta
        
    # pt backward pass start timer
    pt_fw_start = time.perf_counter()
        
    # pass tensor through pt forward layernorm
    ptout = ptln(pt_x)
    
    # pt forward pass end timer
    pt_fw_end = time.perf_counter()

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
    m_x = pt_x.clone().detach().to(torch.float32).requires_grad_(True)  # input (leaf)
    m_gamma = torch.ones((dims), dtype=torch.float32, requires_grad=True)  # scale
    m_beta = torch.zeros((dims), dtype=torch.float32, requires_grad=True)  # shift 
    m_target = pt_target.clone().detach().to(torch.float32)  # target for loss (leaf)
    
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

    # abs tol (atol) --> near-zero values, how close they are
    # rel tol (rtol) --> percent error (magnitude of num)
    # float32 has less precision than float64 --> but faster access

    # use assertion to check success or failure for matching values
    assert np.allclose(m_x.grad.detach().numpy(), pt_dx.detach().numpy(), atol=1e-4, rtol=1e-4), f"FAILURE: Output DX does not match!"
    assert np.allclose(m_gamma.grad.detach().numpy(), pt_dgamma.detach().numpy(), atol=1e-4, rtol=1e-4), f"FAILURE: Output GAMMA does not match!"
    assert np.allclose(m_beta.grad.detach().numpy(), pt_dbeta.detach().numpy(), atol=1e-4, rtol=1e-4), f"FAILURE: Output BETA does not match!"
    # print("SUCCESS: Manual backward pass matches PyTorch!")
    
    # calculate elapsed times
    pt_fw_elapsed = pt_fw_end - pt_fw_start
    pt_bw_elapsed = pt_bw_end - pt_bw_start
    m_fw_elapsed = m_fw_end - m_fw_start
    m_bw_elapsed = m_bw_end - m_bw_start
    
    # accumlate times to average
    return pt_fw_elapsed, pt_bw_elapsed, m_fw_elapsed, m_bw_elapsed
        
if __name__ == "__main__":
    
    # sample test cases (baseline)
    cases = [
        (100, 512),
        (1000, 1024),
        (5000, 256)
    ]
    
    # eval runs (for time averages)
    iterations = 1000
    
    # loop through cases (super strict)
    final_elapsed = {}
    for case in cases:
        pt_fw_elap_sum = 0
        pt_bw_elap_sum = 0
        m_fw_elap_sum = 0
        m_bw_elap_sum = 0
        for i in range(iterations):  # total test iterations
                pt_fw_elapsed, pt_bw_elapsed, m_fw_elapsed, m_bw_elapsed = runTests(case)
                pt_fw_elap_sum += pt_fw_elapsed
                pt_bw_elap_sum += pt_bw_elapsed
                m_fw_elap_sum += m_fw_elapsed
                m_bw_elap_sum += m_bw_elapsed
                
        # store current case finals
        final_elapsed[case] = [(pt_fw_elap_sum / iterations), (pt_bw_elap_sum / iterations), (m_fw_elap_sum / iterations), (m_bw_elap_sum / iterations)]
    
    print("="*55)
    print(f"Total of {iterations} elapsed time tests have completed.")
    
    # display time results
    if len(final_elapsed) > 0:
        for case, elapsed in final_elapsed.items():
            pt_fw_avg_time, pt_bw_avg_time, m_fw_avg_time, m_bw_avg_time = elapsed
            print("="*55)
            print("CASE:", case)
            print("="*55)
            print(f"AVG. PyTorch FW Elapsed Runtime: {pt_fw_avg_time:.4f} seconds.")
            print(f"AVG. Manual FW Elapsed Runtime: {m_fw_avg_time:.4f} seconds.")
            print(f"AVG. PyTorch BW Elapsed Runtime: {pt_bw_avg_time:.4f} seconds.")
            print(f"AVG. Manual BW Elapsed Runtime: {m_bw_avg_time:.4f} seconds.")
            print("="*55)
    else:
        print("No results!")
        exit()
    



    
    
    
    
