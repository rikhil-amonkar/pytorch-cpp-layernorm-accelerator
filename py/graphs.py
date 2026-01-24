import matplotlib.pyplot as plt
import numpy as np
 
# set data
test_cases = ['(100, 512)', '(1000, 1024)', '(5000, 256)', '(10000, 10000)']
runtimesA = [0.213, 3.09, 3.55, 363.74]
runtimesB = [0.03, 2.26, 2.55, 171.04]
runtimesC = [0.868, 15.61, 19.92, 1620]

x = np.arange(len(test_cases))  # match cases
width = 0.25  # rect width
 
# create figure and style
fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, runtimesA, width, label='PyTorch (CPU)', alpha=0.8)
rects2 = ax.bar(x, runtimesB, width, label='PyTorch (MPS)', alpha=0.8)
rects3 = ax.bar(x + width, runtimesC, width, label='Manual (CPU)', alpha=0.8)

# log scale for large vals
ax.set_yscale('log')

# fix labels and style
ax.set_xticks(x, test_cases)
ax.set_xlabel('Test Case (Sample Size, Dimension Size)', fontsize=12, fontweight='bold')
ax.set_ylabel('Runtime (seconds, log scale)', fontsize=12, fontweight='bold')
ax.set_title('LayerNorm Runtime Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# add padding
plt.tight_layout()
 
plt.show()