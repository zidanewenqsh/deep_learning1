import torch
a = torch.arange(24).reshape(2,3,4)
b,c = torch.chunk(a,2,dim=-1)
print(b)
print(c)