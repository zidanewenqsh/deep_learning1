import torch
import torch.nn as nn
import os
from torch.utils import data
from configs.config import *

class Dataset(data.Dataset):
    def __init__(self, data_dir=r"../tokenized"):
        self.data_dir = data_dir
        self.dataset = []
        for data_name in os.listdir(data_dir):
            data_file = os.path.join(data_dir, data_name)
            with open(data_file) as f:
                ws = [int(x) for x in f.read().split()]
                ws_len = len(ws)
                start = 0
                while ws_len-start>=pos_num+1:
                    self.dataset.append(ws[start:start+pos_num+1])
                    start += stride
                else:
                    self.dataset.append(ws[ws_len-pos_num-1:ws_len])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_=torch.tensor(self.dataset[index]).long()
        return data_[:-1], data_[1:]

if __name__ == '__main__':
    print(0)
    data_dir = r"../tokenized"
    dataset_ = Dataset(data_dir)
    dataloader_ = data.DataLoader(dataset_,batch_size=512,shuffle=True,drop_last=True)
    print(len(dataloader_))
    for i, (x, y) in enumerate(dataloader_):
        print(x.shape,y.shape)
        break