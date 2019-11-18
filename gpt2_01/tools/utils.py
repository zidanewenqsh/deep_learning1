import torch
import numpy as np
import os

def creatMask(pos_num):
    mask = torch.zeros(1,1,pos_num,pos_num)
    for i in range(pos_num):
        for j in range(i+1):
            mask[:,:,i,j]=1
    return mask

def makedir(path):
    '''
    如果文件夹不存在，就创建
    :param path:路径
    :return:路径名
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def toTensor(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, torch.FloatTensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, (list, tuple)):
        return torch.tensor(list(data)).float()  # 针对列表和元组，注意避免list里是tensor的情况
    elif isinstance(data, torch.Tensor):
        return data.float()
    return


def toNumpy(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, (list, tuple)):
        return np.array(list(data))  # 针对列表和元组
    return


def toList(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.numpy().tolist()
    elif isinstance(data, (list, tuple)):
        return list(data)  # 针对列表和元组
    return


