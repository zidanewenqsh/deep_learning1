import torch
import torch.nn as nn
import jieba
from torch.utils import data
import os
import numpy as np


class CBOW(nn.Module):
    def __init__(self, word_num=100):
        super(CBOW, self).__init__()
        self.ems = nn.Embedding(word_num, 16)
        self.linear_1 = nn.Linear(16, 16)
        self.linear_2 = nn.Linear(16, 16)
        self.loss_fn = nn.MSELoss()

    def forward(self, w1_idx: torch.Tensor, w2_idx: torch.Tensor) -> torch.Tensor:
        w1 = self.ems(w1_idx)
        w2 = self.ems(w2_idx)
        w = self.linear_1(w1) + self.linear_2(w2)
        # print(w.size())
        return w

    def getloss(self, w_out: torch.Tensor, w_idx: torch.Tensor) -> torch.Tensor:
        target = self.ems(w_idx)
        return self.loss_fn(w_out, target)


class Dataset(data.Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.dataset = []
        self.datalist = []
        chr_pass = ['\n']
        with open(self.datapath, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                self.dataset.append(line)
        for data_ in self.dataset:
            sents = jieba.cut(data_)
            seg_list = ','.join(sents).split(',')
            self.datalist.extend([x for x in seg_list if x not in self.datalist])
        print(self.datalist)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data_0 = self.dataset[item]
        sents_ = jieba.cut(data_0)
        seg_list_ = ','.join(sents_).split(',')
        index_list = []
        for word in seg_list_:
            index_list.append(self.datalist.index(word))
        return seg_list_, torch.tensor(index_list, dtype=torch.long)


class Trainer:
    def __init__(self, net_savepath="net.pth", isCuda=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.net_savepath = net_savepath
        if os.path.exists(self.net_savepath):
            self.net = torch.load(self.net_savepath)
        else:
            self.net = CBOW()
        if isCuda:
            self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.dataset = Dataset("datasets.txt")
        self.dataloader = data.DataLoader(self.dataset, 1, shuffle=True)
        self.epoch = 0

    def forward(self):
        i = 0
        while True:
            # print("i",i)
            for j, (seg_list, index_list) in enumerate(self.dataloader):
                # print("j",j)
                for k, idx in enumerate(index_list):
                    # print("index",index_list)

                    # print("idxs",idxs)
                    for m in range(len(idx)):
                        # print("idx",idx)
                        if m > 0 and m < len(idx) - 1:
                            w_out = self.net(idx[m - 1], idx[m + 1])

                        elif m == 0:
                            w_out = self.net(torch.tensor(0, dtype=torch.long), idx[m + 1])
                            # print(w_out.size(), idx[m + 1].size())

                        elif m == len(idx) - 1:
                            w_out = self.net(idx[m - 1], torch.tensor(0, dtype=torch.long))
                            # print(w_out.size())
                            # print(torch.tensor([0],dtype=torch.long))
                            # print(idx[m - 1],type(idx[m - 1]),idx[m - 1].dtype,idx[m - 1].size())

                            # print("w_out", w_out.size(), w_out.dtype)
                        loss = self.net.getloss(w_out, idx[m])

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        # print("%d/%d/%d/%d loss: %.4f" % (i, j, k, m, loss.detach()))
            if i % 10 == 0:
                print("%d loss: %.6f" % (i, loss.detach()))
                # print("{0} loss: {1:.6f}".format(i,loss.detach()))
            i += 1


if __name__ == '__main__':
    # cbow = CBOW()
    # a = torch.tensor([1], dtype=torch.long)
    # b = torch.tensor([3], dtype=torch.long)
    # print(cbow(a, b))
    # loss = cbow.getloss(cbow(a, b), torch.tensor([2]))
    # print(loss)
    train = Trainer()
    train.forward()
