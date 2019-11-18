import torch
import numpy as np
import torch.nn as nn
#important
# a = torch.arange(11.)
ln = nn.LayerNorm([2,2,2])
ln_2 = nn.LayerNorm([3])
for n,p in ln_2.named_parameters():
    print(n)
    print(p)
# ln = nn.LayerNorm([2])
# ln = nn.LayerNorm(2)
nn.init.constant_(ln.weight,1)
nn.init.constant_(ln.bias,0)
# a = torch.Tensor([1,2,3,4]).reshape(1,1,2,2)
a = torch.Tensor([1,2,3,4,5,6,7,8]).reshape(1,2,2,2)
b = torch.Tensor([5,6,7]).reshape(1,1,3)
x = np.array([1.,2,3])
m1 = torch.mean(a)
m2 = torch.mean(b)
a1 = torch.std(a,unbiased=False)
b1 = torch.std(b,unbiased=False)

ln1 = ln(a)
ln2 = ln_2(b)
print(ln1)
print("*******")
print(ln2)
# std2 = torch.std(a,unbiased=True)
# sd1 = np.std(x,ddof=0)
# sd2 = np.std(x,ddof=1)
print("****************")
print(a1,b1)
c1 = (a-m1)/a1
d1 = (b-m2)/b1
print(c1)
print(d1)

