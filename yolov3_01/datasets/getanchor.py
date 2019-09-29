import torch
import numpy as np
from torch.utils import data
from torch import nn
from matplotlib import pyplot as plt
from tool import utils
import os


class Dataset(data.Dataset):
    def __init__(self, datalist):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        return torch.Tensor(datalist[index][3:])


#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = dict()

        self.layer_0 = nn.Parameter(torch.Tensor([5.5, 5.5]) / 6)
        self.layer_1 = nn.Parameter(torch.Tensor([5.0, 5.7]) / 6)
        self.layer_2 = nn.Parameter(torch.Tensor([5.7, 4.8]) / 6)
        self.layer_3 = nn.Parameter(torch.Tensor([4.8, 4.8]) / 6)
        self.layer_4 = nn.Parameter(torch.Tensor([4.6, 5.0]) / 6)
        self.layer_5 = nn.Parameter(torch.Tensor([5.0, 4.2]) / 6)
        self.layer_6 = nn.Parameter(torch.Tensor([4.1, 4.1]) / 6)
        self.layer_7 = nn.Parameter(torch.Tensor([3.9, 4.3]) / 6)
        self.layer_8 = nn.Parameter(torch.Tensor([4.3, 3.4]) / 6)

        self.layer[0] = self.layer_0
        self.layer[1] = self.layer_1
        self.layer[2] = self.layer_2
        self.layer[3] = self.layer_3
        self.layer[4] = self.layer_4
        self.layer[5] = self.layer_5
        self.layer[6] = self.layer_6
        self.layer[7] = self.layer_7
        self.layer[8] = self.layer_8

    def getresult(self, key, result_, datas_):
        result = result_
        datas = datas_
        data_ = datas[0] * datas[1]
        w_target = torch.exp(self.layer[key][0]*6)
        h_target = torch.exp(self.layer[key][1]*6)
        w_target_1 = torch.round(torch.exp(self.layer[key][0] * 6))
        h_target_1 = torch.round(torch.exp(self.layer[key][1] * 6))
        target_ = w_target * h_target
        target_1 = w_target_1*h_target_1

        w_min = torch.min(datas[0],w_target)
        h_min = torch.min(datas[1],h_target)
        inter = w_min*h_min
        union = data_ + target_ - inter
        iou_value = inter/union

        w_min_1 = torch.min(datas[0], w_target_1)
        h_min_1 = torch.min(datas[1], h_target_1)
        inter_1 = w_min_1 * h_min_1
        union_1 = data_ + target_1 - inter_1
        iou_value_1 = inter_1 / union_1

        result += torch.reciprocal(iou_value)
        return result, iou_value_1

    def forward(self, datass):
        ioulist = []
        # ioulist_1 = []
        result = torch.Tensor([0, ])
        for datas in datass:

            w, h = datas
            max_len = max(w, h)
            ratio = w / h

            datas = datas
            if max_len > 190:
                if ratio > 2:
                    key = 2
                elif ratio < 0.5:
                    key = 1
                else:
                    key = 0
                result = self.getresult(key, result, datas)[0]
                ioulist.append(self.getresult(key, result, datas)[1])

            elif max_len > 94:
                if ratio > 2:
                    key = 5
                elif ratio < 0.5:
                    key = 4
                else:
                    key = 3
                result = self.getresult(key, result, datas)[0]
                ioulist.append(self.getresult(key, result, datas)[1])

            else:
                if ratio > 2:
                    key = 8
                elif ratio < 0.5:
                    key = 7
                else:
                    key = 6
                result = self.getresult(key, result, datas)[0]
                ioulist.append(self.getresult(key, result, datas)[1])

        return result / datass.size(0), ioulist



def shownetparam(net, i, outputfile="anchor.txt"):
    with open(outputfile, 'a') as f:

        datalist = []
        for n, p in net.named_parameters():
            datalist.extend(torch.round(torch.exp(6*p).detach()))
        print(datalist,file=f)


if __name__ == '__main__':
    torch.set_printoptions(threshold=1000000, sci_mode=False)
    label_file = r"D:\PycharmProjects\yolov3_01\labels\label_02.txt"
    save_dir = r"D:\PycharmProjects\yolov3_01\save\getanchorhisto"

    net = Net()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    datalist = []
    datalist2 = []
    data_w = []
    data_h = []
    data_max = []
    data_min = []
    with open(label_file) as f:
        for line in f.readlines():
            strs = line.strip().split()
            _boxes = np.array([int(float(x)) for x in strs[1:]])
            boxes = np.split(_boxes, len(_boxes) // 5)
            datalist.extend(list(boxes))
    dataset = Dataset(datalist)

    for x in dataset:
        print(x)


    batchsize = len(dataset)
    dataloader = data.DataLoader(dataset, batchsize, shuffle=False, drop_last=True)
    print(len(dataloader))
    test = False
    if test:
        for j, datas_ in enumerate(dataloader):
            print(datas_)
            cord = datas_.detach()


    train = True

    if train:
        outputfile = "anchor1.txt"
        with open(outputfile, 'w') as f:
            pass
        init = True


        for i in range(100):

            for j, datas in enumerate(dataloader):
                if init:
                    y, ioulist = net(datas)
                    ioulist = [x.detach() for x in ioulist]
                    ioulist = torch.Tensor(ioulist)
                    init = False



                mask = ioulist<0.55

                y, _ = net(datas[mask])
                _, ioulist = net(datas)

                ioulist = [x.detach() for x in ioulist]
                ioulist = torch.Tensor(ioulist)

                if y.size(0)>0:
                    optimizer.zero_grad()
                    y.backward()
                    optimizer.step()


                if i % 1 == 0 and j==0:
                    print(torch.max(ioulist), torch.min(ioulist),torch.sum(ioulist),torch.mean((ioulist>0.49).float()))
                    print("{0}/{1}, loss: {2}".format(j, i, y.detach()))
                    index = torch.nonzero(ioulist<0.5)
                    print(index)
                # if i % 100 == 0 and j ==0:
                    plt_savename = "histo_{0}_{1}.jpg".format(i,j)
                    plt_savefile = os.path.join(save_dir, plt_savename)
                    shownetparam(net, i, outputfile)
                    plt.clf()
                    plt.xlim(0.3,1.05)
                    plt.ylim(0,5.5)
                    plt.hist(ioulist.numpy(), 76, density=0, facecolor="green", edgecolor="black", alpha=0.7)
                    plt.savefig(plt_savefile)
                    plt.show()
                    plt.pause(0.1)
            index = torch.nonzero(ioulist < 0.5)

            if index.size(0) < 3 and torch.min(ioulist)>0.5:
                print(i)
                shownetparam(net, i, outputfile)
                plt.hist(ioulist.numpy(), 76, density=0, facecolor="green", edgecolor="black", alpha=0.7)
                #
                plt.show()
                plt.pause(0.1)
                print(index)

                break


    analysis = False
    if analysis:
        for x in datalist:
            datalist2.extend(x[3:])
            data_w.append(x[3])
            data_h.append(x[4])
            data_max.append(max(x[3], x[4]))
            data_min.append(min(x[3], x[4]))
        print(datalist2)
        print(data_w)
        print(data_h)
        datalist2 = np.array(datalist2)
        print(np.max(datalist2), np.min(datalist2))
        histo = np.histogram(datalist2, 10, range=(33, 403))
        print(histo)
        ratio = np.array(data_w) / np.array(data_h)
        # print(ratio)

        plt.subplot(3, 1, 1)
        # plt.hist(datalist2, 10, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.hist(data_w, 10, density=0, facecolor="red", edgecolor="black", alpha=0.7)
        plt.hist(data_h, 10, density=0, facecolor="green", edgecolor="black", alpha=0.7)
        plt.subplot(3, 1, 2)
        plt.hist(ratio, 10, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.subplot(3, 1, 3)
        plt.hist(data_max, 10, density=0, facecolor="red", edgecolor="black", alpha=0.7)
        plt.hist(data_min, 10, density=0, facecolor="green", edgecolor="black", alpha=0.7)

        plt.show()
        plt.pause(0.1)
