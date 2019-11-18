import os

line = open(os.path.join(r"D:\PycharmProjects\gpt2_01\datas\vocabs\vocab1.txt"),"r+",encoding="utf-8").read()
print(len(line.split()))