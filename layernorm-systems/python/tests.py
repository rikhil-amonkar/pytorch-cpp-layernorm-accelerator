import torch
import torch.nn as nn
from autograd import LayerNorm
from torch.autograd import gradcheck

# limit randomness
torch.manual_seed(42)

# create sample data (track gradients)
x = torch.rand((2, 3), dtype=torch.float64, requires_grad=True)  # input
gamma = torch.rand((3), dtype=torch.float64, requires_grad=True)  # scale
beta = torch.rand((3), dtype=torch.float64, requires_grad=True)  # shift 
target = torch.rand((2, 3), dtype=torch.float64)  # target for loss

# pass through custom layernorm operation (forward pass)
output = LayerNorm.apply(x, gamma, beta, 1e-5)

# compute loss against target
criterion = nn.MSELoss()
loss = criterion(output, target)

# call backward pass via gradient outputs
loss.backward()

# pass through autograd built-in check
test = gradcheck(LayerNorm.apply, (x, gamma, beta, 1e-5), eps=1e-6, atol=1e-4)

# display results
print("DX:\n", x.grad.detach().numpy())
print("DGAMMA:\n", gamma.grad.detach().numpy())
print("DBETA:\n", beta.grad.detach().numpy())
print("GRAD CHECK:", test)

