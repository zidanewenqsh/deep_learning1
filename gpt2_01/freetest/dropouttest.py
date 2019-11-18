import torch
import torch.nn as nn

a = torch.randn(2,10,2,2)
dt1 = nn.Dropout(0.2)
dt2 = nn.Dropout2d(0.5)
dt3 = nn.Dropout3d(0.5)
b1 = dt1(a)
b2 = dt2(a)
b3 = dt3(a)
print(a)
print("1************")
# print(b1)
# print("2************")
# print(b2)
# print("3************")
print(b3)