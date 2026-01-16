import numpy as np
import torch
import torch.nn as nn
from autograd import LayerNorm
from torch.autograd import gradcheck

# limit randomness
torch.manual_seed(42)

# create sample data (track gradients)
n, dims = 2, 3

# ========== PYTORCH LAYERNORM ==========

# pt input tensor data
pt_x = torch.rand((n, dims), dtype=torch.float32, requires_grad=True)  # input
pt_target = torch.rand((n, dims), dtype=torch.float32)  # target for loss

# pytorch layer normalization function
ptln = nn.LayerNorm(dims)

# reset pytorch learn-param grads
with torch.no_grad():
    ptln.weight.data = torch.ones(dims)  # pt gamma
    ptln.bias.data = torch.zeros(dims)  # pt beta
    
# ensure grads are tracked
ptln.weight.requires_grad_(True)
ptln.bias.requires_grad_(True)
    
# pass tensor through pt forward layernorm
ptout = ptln(pt_x)

# calculate pt loss (mse)
ptloss = ((ptout - pt_target) ** 2).sum()  # mean sq err sum

# pass loss grad through pt backward layernorm
ptloss.backward()

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

# pass through custom layernorm operation (forward pass)
mout = LayerNorm.apply(m_x, m_gamma, m_beta, 1e-5)

# compute loss against target
criterion = nn.MSELoss(reduction="sum")  # set mse sum (default is div)
mloss = criterion(mout, m_target)

# call backward pass via gradient outputs
mloss.backward()

# pass through autograd built-in check
# test = gradcheck(LayerNorm.apply, (m_x, m_gamma, m_beta, 1e-5), eps=1e-6, atol=1e-4)
# print("GRAD CHECK:", test)

# ========== RESULTS ==========

print("=" * 50)

# pt results
print("PYTORCH RESULTS\n")
print("DX:\n", pt_dx.detach().numpy())
print("DGAMMA:\n", pt_dgamma.detach().numpy())
print("DBETA:\n", pt_dbeta.detach().numpy())

print("=" * 50)

# manual results
print("MANUAL RESULTS\n")
print("DX:\n", m_x.grad.detach().numpy())
print("DGAMMA:\n", m_gamma.grad.detach().numpy())
print("DBETA:\n", m_beta.grad.detach().numpy())

print("=" * 50)
print("=" * 50)

# abs tol (atol) --> near-zero values, how close they are
# rel tol (rtol) --> percent error (magnitude of num)

# use assertion to check success or failure for matching values
assert np.allclose(m_x.grad.detach().numpy(), pt_dx.detach().numpy(), atol=1e-5, rtol=1e-8), f"FAILURE: Output DX does not match!"
assert np.allclose(m_gamma.grad.detach().numpy(), pt_dgamma.detach().numpy(), atol=1e-5, rtol=1e-8), f"FAILURE: Output GAMMA does not match!"
assert np.allclose(m_beta.grad.detach().numpy(), pt_dbeta.detach().numpy(), atol=1e-5, rtol=1e-8), f"FAILURE: Output BETA does not match!"
print("SUCCESS: Manual backward pass matches PyTorch!")

print("=" * 50)
