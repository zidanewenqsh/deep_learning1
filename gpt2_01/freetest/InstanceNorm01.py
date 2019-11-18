import torch
import torch.nn as nn
#important
import torch
import numpy as np
import torch.nn as nn
#important
# a = torch.arange(11.)
# ln = nn.LayerNorm([2,2,2])
ln_1 = nn.InstanceNorm2d(2)
ln_2 = nn.InstanceNorm1d(2)
ln_3 = nn.LayerNorm([2,2])
ln_4 = nn.LayerNorm([2,2,2])
ln_5 = nn.InstanceNorm3d(1)
# ln = nn.LayerNorm([2])
# ln = nn.LayerNorm(2)
# nn.init.constant_(ln_1.weight,1)
# nn.init.constant_(ln_1.bias,0)
# a = torch.Tensor([1,2,3,4]).reshape(1,1,2,2)
a = torch.Tensor([1,2,3,4,5,6,7,8]).reshape(1,2,2,2)
b = torch.Tensor([1,2,3,4,5,6,7,8]).reshape(2,2,2)
c = torch.Tensor([1,2,3,4,5,6,7,8]).reshape(1,2,2,2)
d = torch.Tensor([1,2,3,4,5,6,7,8]).reshape(1,1,2,2,2)
x = np.array([1.,2,3])
m1 = torch.mean(a)
m2 = torch.mean(b)
a1 = torch.std(a,unbiased=False)
b1 = torch.std(b,unbiased=False)


ln1 = ln_1(a)
ln2 = ln_2(b)
ln3= ln_3(c)
ln4 = ln_4(c)
ln5 = ln_5(d)
print(ln1)
print("1*******")
print(ln2)
print("2*******")
print(ln3)
print("3*******")
print(ln4)
print("4*******")
print(ln5)
# std2 = torch.std(a,unbiased=True)
# sd1 = np.std(x,ddof=0)
# sd2 = np.std(x,ddof=1)
print("****************")
print(a1,b1)
c1 = (a-m1)/a1
d1 = (b-m2)/b1
print(c1)
print(d1)

