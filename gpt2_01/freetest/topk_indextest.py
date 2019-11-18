import torch
# a = torch.arange(24).reshape(2,3,4)
a = torch.randint(1,100,size=(5,5,5))
b = a[:,-1]
print(a)
# print(b)
print(b.shape)
c = torch.topk(a,3,-1)
print(c)
print(type(c))
print(c[0].shape)