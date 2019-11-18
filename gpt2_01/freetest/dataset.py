import torch
from torch import nn
from torchvision import transforms
import numpy as np
from torch.utils import data
import jieba


class Dataset(data.Dataset):
    def __init__(self,datapath):
        self.datapath = datapath
        self.dataset = []
        self.datalist = []
        chr_pass = ['\n']
        with open(self.datapath,'r',encoding='utf-8') as file:
            for line in file.readlines():
                self.dataset.append(line)
        for data_ in self.dataset:
            sents = jieba.cut(data_)
            seg_list = ','.join(sents).split(',')
            # print('/n' not in chr_pass)
            self.datalist.extend([x for x in seg_list if x not in self.datalist])
            # for word in seg_list:
            #     print(word not in chr_pass)
            #     if word not in chr_pass and word not in self.datalist:
            #         self.datalist.append(word)

                # if word not in self.datalist and word not in chr_pass:
                #     self.datalist.append(word)
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
        return seg_list_,torch.tensor(index_list,dtype=torch.long)

    # @staticmethod
    # def listextend(a: list, b: list) -> None:
    #     a.extend([x for x in b if x not in a])
if __name__ == '__main__':
    dataset = Dataset("datasets.txt")
    dataloader = data.DataLoader(dataset,1,shuffle=False)
    for x, y in dataloader:
        print(x)
        print("*********")
        print(y)
        print("******************")