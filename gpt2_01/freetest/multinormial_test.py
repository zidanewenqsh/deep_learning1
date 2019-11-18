import torch

a = torch.Tensor([0, 10, 3, 0])
a = torch.arange(16.).reshape(4,4)
b = torch.multinomial(a, 2)
# c = torch.multinomial(a, 4)
#RuntimeError: invalid multinomial distribution (with replacement=False, not enough non-negative category to sample)
c = torch.multinomial(a, 2, replacement=True)
print(b)
print(c)
a = torch.randn(10,5)
b = torch.arange(0, a.size(1))[None,:].repeat(a.size(0),1)
print(a.shape)
print(b)
