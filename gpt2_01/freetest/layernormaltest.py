import torch
import torch.nn as nn

# input = torch.randn(2, 5, 10, 10)
input = torch.arange(32.).reshape(2,4,2,2)
print(input)
# With Learnable Parameters
print(input.size()[1:])
m = nn.LayerNorm(input.size()[1:])
# Without Learnable Parameters
# m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
# Normalize over last two dimensions
output = m(input)
print(output)
print("***************")
m = nn.LayerNorm([2, 2])
output = m(input)
print(output)
print("***************")
# Normalize over last dimension of size 10
m = nn.LayerNorm(2)
# Activating the module
output = m(input)
print(output)