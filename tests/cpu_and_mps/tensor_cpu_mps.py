import torch
from torch.utils.benchmark import Timer

# sample tensor function
def createTensors(device):
    
    # create random float tensors
    x = torch.rand((10000, 10000), dtype=torch.float32)
    y = torch.rand((10000, 10000), dtype=torch.float32)
    
    # select device
    x = x.to(device)
    y = y.to(device)
    
    return x, y

# test with cpu
device = torch.device("mps")
x, y = createTensors(device)
print(f"Using Device: {device}")

# create a timer instance
timer = Timer(
    stmt="x * y",
    globals={"x": x, "y": y}
)
result = timer.blocked_autorange(min_run_time=1)  # min iterations
print(result)


