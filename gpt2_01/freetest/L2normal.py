import numpy as np
import torch
a = np.random.rand(5,5)
b = torch.from_numpy(a)
np.set_printoptions(precision=4, suppress=True)
# mask = a<0
# print(mask)
# print(a[mask])
# print(mask.shape)
# print(a[mask].shape)
# a[mask] = np.random.randn()
# print(np.random.randn(*mask.shape))
i = 0
# while np.any(a<0):
#     mask = a<0
#     mask_ = b<0
#     print(mask_.size())
#     a[mask]=np.random.randn()
#     # b[mask_] = torch.randn()
#     i+=1

print(a)
norm_a = np.sqrt(np.sum(np.square(a),axis=1,keepdims=True))
# print(norm_a)
a_ = a/norm_a
# print(a)
print(a_)
print(np.sqrt(np.sum(np.square(a_),axis=1)))
print(np.dot(a_, a_.T))
print(i)
print(b.shape)
c = torch.nn.functional.normalize(b,2,1)
print(c)
print(a_)
print(torch.sqrt(torch.sum(c**2,dim=1,keepdim=True)))
c1 = 0
print(torch.norm(b,2,1,True))
print(norm_a)