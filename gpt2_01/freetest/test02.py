import torch
import torch.nn as nn
import numpy as np
fc_1 = nn.Conv2d(1,1,3,2,0)
a = torch.randn(1,1,10,10)
b = fc_1(a)
print(b.size())
fc_2 = nn.ConvTranspose2d(1,1,3,2,0,output_padding=1)
c = fc_2(b)
print(c.size())
print(fc_1.weight)
for n,p in fc_1.named_parameters():
    print(n)
    print(p)
pi = np.pi
print(pi)
e = np.e
print(e)
print(pi**4+pi**5)
print(e**6)
