import torch
from torch import nn

ems = nn.Embedding(100,4)
a = ems(torch.tensor([1,99,5,2,99,0,1,2,0]))
print(a)