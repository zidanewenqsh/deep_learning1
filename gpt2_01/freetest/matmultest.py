import torch
a = torch.randn(2,3,4,5)
b = torch.randn(2,3,4,4)
c = torch.matmul(b,a)
print(c.shape)