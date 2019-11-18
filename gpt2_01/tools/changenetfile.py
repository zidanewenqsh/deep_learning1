import os

net_file_r = r'C:\Users\wen\PycharmProjects\myproject_20190910\liev\scipy\test0.py'
net_file_w = r'C:\Users\wen\PycharmProjects\myproject_20190910\liev\scipy\Least_squares_fitting.py'
with open(net_file_w,'w',encoding='utf-8') as f1:
    with open(net_file_r,'r',encoding='utf-8') as f:
        for line in f.readlines():
            print(line[3:], end='', file=f1)