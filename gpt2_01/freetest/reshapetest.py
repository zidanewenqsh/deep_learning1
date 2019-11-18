import torch
a = torch.arange(24).reshape(2,3,4)
print(a)
b = torch.arange(6).reshape(2,3)
print(b)
a1 = a.reshape(6,4)
b1 = b.reshape(-1)
print(a1)
print(b1)