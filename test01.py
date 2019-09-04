import numpy as np
import torch
a = [1,2,3,4,5]
b = np.array(a,dtype='f8')
print(b.dtype)
c = torch.Tensor(b)
print(c.dtype)
a = np.zeros(shape=(3,3))
print(a.dtype)
a[1,1] = 1
print(a)
b = torch.Tensor(a)
print(b.dtype)