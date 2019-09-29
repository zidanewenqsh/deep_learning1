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
            # labels[feature_size] = torch.zeros(size=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM),dtype=torch.float)
            labels[feature_size] = torch.zeros(size=(feature_size, feature_size, 3, 6), dtype=torch.float)

            # print("labels",labels[feature_size].dtype)
            for box in boxes:
                cls, cx, cy, b_w, b_h = box

                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_HEIGHT)
                # print("cx_index", cx_index, cy_index, cx_offset, cy_offset)
                for i, anchor in enumerate(anchors):
                    p_w, p_h = anchor
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    box_area = b_w * b_h
                    inter = min(b_w, p_w) * min(b_h, p_h)
                    union = anchor_area + box_area - inter
                    iou = inter / union
                    conf = 1/(1+math.exp(-10*(iou-0.4)))
                    # print(conf)
                    tw = math.log(b_w / p_w)
                    th = math.log(b_h / p_h)
                    # print("conf",conf)
                    # print(type(cfg.CLASS_NUM))
                    # print(type(cls))
                    # print(cls)
                    # print(one_hot(cfg.CLASS_NUM, cls))
                    # a = [conf, cx_offset, cy_offset, tw, th, *one_hot(cfg.CLASS_NUM, cls)]
                    # print(a)
                    # labels[feature_size][int(cx_index), int(cy_index), i] = torch.tensor(
                    #     [conf, cx_offset, cy_offset, tw, th, *one_hot(cfg.CLASS_NUM, cls)]).float()
                    labels[feature_size][int(cy_index), int(cx_index), i] = torch.tensor(
                        [conf, cx_offset, cy_offset, tw, th, int(cls)]).float()
                    # print("labels1", labels[feature_size].dtype,type(labels[feature_size]))

        return labels[13], labels[26], labels[52], img_data


if __name__ == '__main__':
    torch.set_printoptions(threshold=1000000, sci_mode=False)
    label_path = r"..\labels\label_02_21.txt"
    pic_path = r"D:\datasets\yolodatasets\datasets_20190801\datasets_resize"
    # label_path = r"..\param\coco_label.txt"
    pic_path = r"../picdata"

    dataset = YoloDataset(label_path, pic_path)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    #print(len(dataset))
    for data in dataloader:
        #print(type(data))
        d1, d2, d3, _ = data
        #print(d1.size())

        m1 = d1[..., 0] > 0
        m2 = d2[..., 0] > 0
        m3 = d3[..., 0] > 0
        nz1 = torch.nonzero(m1)
        nz2 = torch.nonzero(m2)
        nz3 = torch.nonzero(m3)

        # print(nz1)
        # print(d1[m1])
        # print(nz2)
        # print(d2[m2])
        # print(nz3)
        # print(d3[m3])

        # index = torch.argmax(d1[..., 0],dim=-2)
        # print(index)
        # print(d1[..., 0].size())
        # print(len(data))
        # print(data[0].size())
        # print(data[0])
        # break

    # for label_13,label_26,label_52,img_data in dataloader:
    #     print(label_13.size())
    #     print(label_26.size())
    #     print(label_52.size())
    #     print(img_data.size())
    #     print(label_13.dtype)
    #     print(label_26.dtype)
    #     print(label_52.dtype)
    #     print("******")
