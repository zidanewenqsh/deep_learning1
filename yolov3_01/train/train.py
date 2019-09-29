import torch
import torch.nn as nn
from src.darknet53 import MainNet
from src.trainmain_2 import Yolov3Trainer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFGFILE = r"..\train\cfg.ini"
import numpy as np

if __name__ == '__main__':


    torch.set_printoptions(precision=4, threshold=np.inf, sci_mode=False)
    net = MainNet(10)
    trainer = Yolov3Trainer(net, "yolo_00_46", CFGFILE)
    trainer.train()
