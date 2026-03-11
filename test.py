import torch

import torch

# Define dimensions
d1, d2, d3 = 2, 3, 4
# Create an empty tensor to fill
x = torch.zeros(d1, d2, d3)

current_value = 1

# Nested loops to fill the tensor
for i in range(d1):        # First dimension (4)
    for j in range(d2):    # Second dimension (5)
        for k in range(d3): # Third dimension (6)
            x[i][j][k] = current_value
            current_value += 1

print(x.transpose(0, 1))
print(f"Final Shape: {x.shape}")

