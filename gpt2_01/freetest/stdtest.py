import torch
import numpy as np
import torch.nn as nn

# a = torch.arange(11.)
a = torch.Tensor([1,2,3])
x = np.array([1.,2,3])
m = torch.mean(a)
print(torch.mean(a))
std1 = torch.std(a,unbiased=False)
std2 = torch.std(a,unbiased=True)
sd1 = np.std(x,ddof=0)
sd2 = np.std(x,ddof=1)
print(std1,std2)
std1_ = torch.std(a)
std2_ = np.std(x)
print("a",std1_,std2_)
print(sd1,sd2)
sum1 = torch.sum((a-m)**2)
sqrt1 = torch.sqrt(sum1)
print(sum1,sqrt1)
print(torch.sqrt((sum1/3)))