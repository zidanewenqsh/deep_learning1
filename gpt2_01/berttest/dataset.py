import os
from torch.utils import data
from configs import config as cfg
import torch

class MyDataset(data.Dataset):
    def __init__(self, data_dir="../tokenized"):
        self.dataset = []
        self.data_dir = data_dir
        for filename in os.listdir(data_dir):
            with open(os.path.join(data_dir,filename),'r',encoding='utf-8') as f:
                ws = [int(x) for x in f.readline().split()]

                # print(f.readlines())
                # ws = [int(x) for x in f.readlines()]
                # print(ws[:10])
                ws_len = len(ws)
                # print(ws_len)
                start = 0
                # while

                while ws_len - start > cfg.pos_num + 1:
                    # print(ws_len - start, cfg.pos_num + 1)
                    self.dataset.append(ws[start:start + cfg.pos_num + 1])
                    # print(ws[start:start + cfg.pos_num + 1])
                    # print(len(ws[start:start + cfg.pos_num + 1]))
                    start += cfg.stride
                    # if start>1000:
                    #     break
                else:
                    if ws_len > cfg.pos_num + 1:
                        # print(ws_len)
                        self.dataset.append(ws[ws_len - cfg.pos_num - 1:])
                        # print(len(ws[ws_len - cfg.pos_num - 1:]), ws_len - cfg.pos_num - 1, ws_len)

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        data = torch.tensor(self.dataset[index])
        return data[0:-1], data[1:]

if __name__ == '__main__':
    myDataset = MyDataset()
    print(len(myDataset))
    print(myDataset[0][0].shape)
    myDataloader = data.DataLoader(myDataset, batch_size=512, shuffle=True, drop_last=True)
    print(len(myDataloader))
    for i, (data_1, data_2) in enumerate(myDataloader):
        print(data_1.size())
        print("111")
        print(data_2.size())
        print("222")
        break
    # print(len(myDataset))
    # print(myDataset[0])
    # f1 = r'D:\PycharmProjects\bert\tokenized\1.txt'
    # with open(f1) as f:
    #     print(type(f.readline()))
    #     print(type(f.readlines()))
