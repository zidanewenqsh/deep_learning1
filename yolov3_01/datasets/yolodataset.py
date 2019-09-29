import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import src.cfg as cfg
import math

transform = transforms.Compose([
    transforms.Resize((416, 416)),
    # transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])


def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1
    return b


class YoloDataset(data.Dataset):
    def __init__(self, label_path, pic_path, face_size=416, transforms=transform):
        self.label_path = label_path
        self.pic_path = pic_path
        self.face_size = face_size
        self.transform = transforms
        self.dataset = []

        with open(label_path, 'r') as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}
        line = self.dataset[index]
        strs = line.strip().split()

        _img_data = Image.open(os.path.join(self.pic_path, strs[0]))
        img_data = self.transform(_img_data)

        _boxes = np.array([int(float(x)) for x in strs[1:]])
        boxes = np.split(_boxes, len(_boxes) // 5)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            # 造好一个标签字典
            labels[feature_size] = torch.zeros(size=(feature_size, feature_size, 3, 6), dtype=torch.float)

            for box in boxes:
                cls, cx, cy, b_w, b_h = box

                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_HEIGHT)

                for i, anchor in enumerate(anchors):
                    p_w, p_h = anchor
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    box_area = b_w * b_h
                    inter = min(b_w, p_w) * min(b_h, p_h)
                    union = anchor_area + box_area - inter
                    conf = inter / union
                    # print(conf)
                    tw = math.log(b_w / p_w)
                    th = math.log(b_h / p_h)

                    labels[feature_size][int(cy_index), int(cx_index), i] = torch.tensor(
                        [conf, cx_offset, cy_offset, tw, th, int(cls)]).float()

        return labels[13], labels[26], labels[52], img_data


