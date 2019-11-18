import torch
a = [1, 2, 3, 4]
b = [4, 5, 6, 7]
a.extend([x for x in b if x not in a])
print(a)


def listextend(a: list, b: list) -> list:
   a.extend([x for x in b if x not in a])

c = [1,4,2,8]
listextend(a,c)
print(a)
print(a.index(6))
a1 = torch.Tensor(a)
print(a1)
chr_pass = ['\n']
c = '\n'
print(c in chr_pass)