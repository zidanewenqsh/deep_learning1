import torch
import torch.nn as nn
from torch.utils import data
from torch import optim
from src.nets import Gpt2
# from berttest.module import Gpt2

import os
from datasets.dataset import Dataset
from configs.config import *


class Trainer(nn.Module):
    def __init__(self, net_savefile=r"../params/net.pth", param_savefile=r"../params/net.pt", data_dir=r"../tokenized"):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_savefile = net_savefile
        self.param_savefile = param_savefile
        self.data_dir = data_dir

        self.net = Gpt2(True)

        if os.path.exists(self.net_savefile):
            self.net = torch.load(self.net_savefile)
            print("net load successfully")
        elif os.path.exists(self.param_savefile):
            self.net.load_state_dict(torch.load(self.param_savefile))
            print("net params load successfully")

        self.net.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters())
        self.dataset = Dataset(data_dir)
        self.dataloader = data.DataLoader(self.dataset, batch_size=4, shuffle=True, drop_last=True)
        # self.epoch = 10000

    def forward(self, epoch: int) -> int:
        for i in range(epoch):
            for j, (x, y_) in enumerate(self.dataloader):
                x = x.to(self.device)
                y_ = y_.to(self.device)
                p = torch.arange(x.size(-1)).reshape(1, -1).repeat(x.size(0), 1).to(self.device)
                y = self.net(x, p)
                y = y.reshape(-1, vocab_num)
                y_ = y_.reshape(-1)

                loss = self.loss_fn(y, y_)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if j % 10 == 0:
                    print("epoch: %d, batch:%d, loss: %.4f" % (i, j, loss.detach()))
                    torch.save(self.net, self.net_savefile)
                    # print("net save successfully")
                    torch.save(self.net.state_dict(),self.param_savefile)

        return 0


if __name__ == '__main__':
    trainer = Trainer()
    trainer(1)
