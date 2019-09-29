import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils import data
import time

from src1.yolodataset_3 import YoloDataset

from src.darknet53 import MainNet
import matplotlib.pyplot as plt
from tool import utils
import argparse
import configparser
from PIL import Image, ImageDraw, ImageFont

from detect.detector import Detector
import cv2
from src import cfg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFGFILE = "..\src\cfg.ini"


SAVE_DIR = r""
PIC_DIR = r""
LABEL_PATH = r""
NETFILE_EXTENTION = "pt"
ALPHA = 0.9
CONTINUETRAIN = False
NEEDTEST = False
NEEDSAVE = False
NEEDSHOW = False
EPOCH = 1
BATCHSIZE = 4
NUMWORKERS = 1

THREHOLD = 0.5
LR = 1e-3
ISCUDA = True
SAVEDIR_EPOCH = r""
TEST_IMG = r""

RECORDPOINT = 10
TESTPOINT = 100


class Yolov3Trainer:
    def __init__(self, net, netfile_name, cfgfile=None):
        self.net = net
        self.netfile_name = netfile_name
        print(cfgfile)
        if cfgfile != None:
            self.cfginit(cfgfile)
            print(1)
        print(SAVE_DIR)
        utils.makedir(SAVE_DIR)
        parser = argparse.ArgumentParser(description="base class for network training")
        self.args = self.argparser(parser)

        net_savefile = "{0}.{1}".format(self.netfile_name, NETFILE_EXTENTION)
        self.save_dir = os.path.join(SAVE_DIR, "nets")
        utils.makedir(self.save_dir)
        self.save_path = os.path.join(self.save_dir, net_savefile)
        self.savepath_epoch = os.path.join(SAVEDIR_EPOCH, net_savefile)

        if os.path.exists(self.save_path) and CONTINUETRAIN:

            try:
                self.net.load_state_dict(torch.load(self.save_path))
                print("net param load successful")

            except:
                self.net = torch.load(self.save_path)
                print("net load successful")


        else:
            self.net.paraminit()
            print("param initial complete")

        if ISCUDA:
            self.net = self.net.to(DEVICE)

        if NEEDTEST:
            self.detecter = Detector()

        self.logdir = os.path.join(SAVE_DIR, "log")

        utils.makedir(self.logdir)
        self.logfile = os.path.join(self.logdir, "{0}.txt".format(self.netfile_name))
        if not os.path.exists(self.logfile):
            with open(self.logfile, 'w') as f:
                print("%.2f %d    " % (0.00, 0), end='\r', file=f)
                print("logfile created")

        self.optimizer = optim.Adam(self.net.parameters())

        # 损失函数定义
        self.conf_loss_fn = nn.BCEWithLogitsLoss()  # 定义置信度损失函数
        self.center_loss_fn = nn.BCEWithLogitsLoss()  # 定义中心点损失函数
        self.wh_loss_fn = nn.MSELoss()  # 宽高损失
        # self.cls_loss_fn = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失
        self.cls_loss_fn = nn.CrossEntropyLoss()

        self.detecter = Detector()

        print("initial complete")

    def cfginit(self, cfgfile):
        config = configparser.ConfigParser()
        config.read(cfgfile)
        items_ = config.items(self.netfile_name)

        for key, value in items_:
            if key.upper() in globals().keys():
                try:
                    globals()[key.upper()] = config.getint(self.netfile_name, key.upper())
                except:
                    try:
                        globals()[key.upper()] = config.getfloat(self.netfile_name, key.upper())
                    except:
                        try:
                            globals()[key.upper()] = config.getboolean(self.netfile_name, key.upper())
                        except:
                            globals()[key.upper()] = config.get(self.netfile_name, key.upper())


    def argparser(self, parser):
        """default argparse, please customize it by yourself. """

        parser.add_argument("-e", "--epoch", type=int, default=EPOCH, help="number of epochs")
        parser.add_argument("-b", "--batch_size", type=int, default=BATCHSIZE, help="mini-batch size")
        parser.add_argument("-n", "--num_workers", type=int, default=NUMWORKERS,
                            help="number of threads used during batch generation")
        parser.add_argument("-l", "--lr", type=float, default=LR, help="learning rate for gradient descent")
        parser.add_argument("-r", "--record_point", type=int, default=RECORDPOINT, help="print frequency")
        parser.add_argument("-t", "--test_point", type=int, default=TESTPOINT,
                            help="interval between evaluations on validation set")
        parser.add_argument("-a", "--alpha", type=float, default=ALPHA, help="ratio of conf and offset loss")
        parser.add_argument("-d", "--threshold", type=float, default=THREHOLD, help="threhold")

        return parser.parse_args()

    def _loss_fn(self, output, target, alpha):

        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        mask_obj = target[..., 0] > 0
        mask_noobj = target[..., 0] == 0

        output_obj, target_obj = output[mask_obj], target[mask_obj]

        loss_obj_conf = self.conf_loss_fn(output_obj[:, 0], target_obj[:, 0])
        loss_obj_center = self.center_loss_fn(output_obj[:, 1:3], target_obj[:, 1:3])
        loss_obj_wh = self.wh_loss_fn(output_obj[:, 3:5], target_obj[:, 3:5])
        loss_obj_cls = self.cls_loss_fn(output_obj[:, 5:], target_obj[:, 5].long())
        loss_obj = loss_obj_conf + loss_obj_center + loss_obj_wh + loss_obj_cls

        output_noobj, target_noobj = output[mask_noobj], target[mask_noobj]
        loss_noobj = self.conf_loss_fn(output_noobj[:, 0], target_noobj[:, 0])

        loss = alpha * loss_obj + (1 - alpha) * loss_noobj
        return loss

    def logging(self, result, dataloader_len, RECORDPOINT):

        with open(self.logfile, "r+") as f:

            if f.readline() == "":
                batchcount = 0
                f.seek(0, 0)
                print("%.2f %d        " % (0.00, 0), end='\r', file=f)

            else:
                f.seek(0, 0)
                batchcount = int(f.readline().split()[-1].strip()) + RECORDPOINT


            f.seek(0, 0)
            print("%.2f %d " % (batchcount / dataloader_len, batchcount), end='', file=f)

            f.seek(0, 2)
            print(result, file=f)

    def getstatistics(self):
        datalist = []
        with open(self.logfile) as f:
            for line in f.readlines():
                if not line[0].isdigit():
                    datalist.append(eval(line))
        return datalist

    def scalarplotting(self, datalist, key):
        save_dir = os.path.join(SAVE_DIR, key)
        utils.makedir(save_dir)
        save_name = "{0}.jpg".format(key)

        save_file = os.path.join(save_dir, save_name)
        values = []
        for data_dict in datalist:
            if data_dict:
                values.append(data_dict[key])
        if len(values) != 0:
            plt.plot(values)
            plt.savefig(save_file)
            plt.show()

    def FDplotting(self, net):
        save_dir = os.path.join(SAVE_DIR, "params")
        utils.makedir(save_dir)
        save_name = "{0}_param.jpg".format(self.netfile_name)
        save_file = os.path.join(SAVE_DIR, save_name)
        params = []
        for param in net.parameters():
            params.extend(param.view(-1).cpu().detach().numpy())
        params = np.array(params)
        histo = np.histogram(params, 10, range=(np.min(params), np.max(params)))
        plt.plot(histo[1][1:], histo[0])
        plt.savefig(save_file)
        plt.show()

    def train(self):
        dataset = YoloDataset(LABEL_PATH, PIC_DIR)
        train_loader = data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.num_workers,
                                       drop_last=True)
        dataloader_len = len(train_loader)

        start_time = time.time()

        if os.path.exists(self.logfile):
            with open(self.logfile) as f:
                if f.readline() != "":
                    f.seek(0, 0)
                    batch_count = int(float(f.readline().split()[1]))

        for i in range(self.args.epoch):

            for j, (target13, target26, target52, img_data) in enumerate(train_loader):

                self.net.train()
                if ISCUDA:
                    target13 = target13.to(DEVICE)
                    target26 = target26.to(DEVICE)
                    target52 = target52.to(DEVICE)
                    img_data = img_data.to(DEVICE)

                output_13, output_26, output_52 = self.net(img_data)

                loss_13 = self._loss_fn(output_13, target13, alpha=ALPHA)
                loss_26 = self._loss_fn(output_26, target26, alpha=ALPHA)
                loss_52 = self._loss_fn(output_52, target52, alpha=ALPHA)
                loss = loss_13 + loss_26 + loss_52
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                if j % self.args.record_point == 0:

                    checktime = time.time() - start_time

                    result = "{'epoch':%d,'batch':%d,'loss':%.5f,'loss_13':%.5f,'loss_26':%.5f,'loss_52':%f',total_time':%.2f,'time':%s}" % (
                        i, j, loss, loss_13, loss_26, loss_52, checktime,
                        time.strftime("%Y%m%d%H%M%S", time.localtime()))
                    print(result)

                    # self.logging(result, dataloader_len, self.args.record_point)
                    if NEEDSAVE:
                        # torch.save(self.net.state_dict(), self.save_path)
                        torch.save(self.net, self.save_path)
                        print("net save successful")

                if NEEDTEST and j % self.args.test_point == 0:
                    self.net.eval()

                    batch_count = i
                    self.test(batch_count,j)
            if NEEDSAVE:
                torch.save(self.net.state_dict(), self.savepath_epoch)
                # torch.save(self.net, self.savepath_epoch)
                # print("an epoch save successful")

    def test(self, batch_count,j):
        with torch.no_grad():
            self.net.eval()

            img = Image.open(TEST_IMG)
            # img_ = cv2.imread(TEST_IMG)

            last_boxes = self.detecter.detect(img, self.args.threshold, net=self.net)

            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font="arial.ttf", size=10, encoding="utf-8")

            if np.any(last_boxes):
                for box in last_boxes:
                    xybox = box[:4].astype("i4")
                    text_x, text_y = list(box[:2])[0], list(box[:2])[1] - 10
                    text_conf = list(box[:2])[0] + 30
                    draw.text((text_x, text_y), cfg.COCO_DICT[int(box[5])], fill=(255, 0, 0), font=font)
                    draw.text((text_conf, text_y), "%.2f" % box[4], fill=(255, 0, 0), font=font)
                    draw.rectangle(list(xybox), outline="green", width=2)

            # img.show()
            if NEEDSAVE:
                testpic_savedir = os.path.join(SAVE_DIR, "testpic", self.netfile_name)
                utils.makedir(testpic_savedir)
                testpic_savefile = os.path.join(testpic_savedir, "{0}_{1}.jpg".format(batch_count,j))
                img.save(testpic_savefile)

            if NEEDSHOW:
                plt.clf()
                plt.axis("off")
                plt.imshow(img)
                plt.pause(0.1)


