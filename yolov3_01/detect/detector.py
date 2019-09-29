import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from src.darknet53 import MainNet
import torch.nn as nn
from src import cfg
from tool import utils
import matplotlib.pyplot as plt
import cv2
import time


class Detector:
    def __init__(self, net_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if net_path != None:
            if os.path.exists(net_path):
                try:
                    self.net = torch.load(net_path)
                    self.net = self.net.to(self.device)
                    self.net.eval()
                except:
                    self.net = MainNet(10)
                    self.net.load_state_dict(torch.load(net_path))
                    self.net = self.net.to(self.device)
                    self.net.eval()
            else:
                raise FileNotFoundError

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        self.anchors = cfg.ANCHORS_GROUP

    def detect(self, image, thresh, net=None):

        if net != None:
            self.net = net.to(self.device)
        input = self.transform(image)

        input.unsqueeze_(dim=0)

        output_13, output_26, output_52 = self.net(input.to(self.device))

        idx_13, vecs_13 = self._filter(output_13, thresh)
        idx_26, vecs_26 = self._filter(output_26, thresh)
        idx_52, vecs_52 = self._filter(output_52, thresh)

        box_13 = self._parse(idx_13, vecs_13, 32, self.anchors[13])
        box_26 = self._parse(idx_26, vecs_26, 16, self.anchors[26])
        box_52 = self._parse(idx_52, vecs_52, 8, self.anchors[52])

        box_list = []

        for box_ in [box_13, box_26, box_52]:
            if box_.shape[0] != 0:
                box_list.append(box_)

        if len(box_list) > 0:
            boxes_all = np.concatenate(box_list, axis=0)

            last_boxes = []
            # last_boxes1 = []
            for n in range(input.size(0)):
                n_boxes = []
                boxes_n = boxes_all[boxes_all[:, 6] == n]
                for cls in range(cfg.CLASS_NUM):
                    boxes_c = boxes_n[boxes_n[:, 5] == cls]
                    if boxes_c.shape[0] > 0:
                        n_boxes.extend(utils.nms(boxes_c, 0.3))
                    else:
                        pass
                last_boxes.extend(np.stack(n_boxes))
                # last_boxes1.append(np.stack(n_boxes))
            last_boxes = np.stack(last_boxes)

            return last_boxes
        return

    def _filter(self, output, thresh):

        output = output.permute(0, 3, 2, 1)  # NWHC
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        output = output.cpu().detach()
        torch.sigmoid_(output[..., 0:3])
        mask = torch.gt(output[..., 0], thresh)

        if torch.any(mask):
            idxs = torch.nonzero(mask)

            vecs = output[mask]

            return idxs.numpy(), vecs.numpy()

        return np.array([]), np.array([])

    def _parse(self, idxs, vecs, t, anchors):

        if idxs.shape[0] == 0:
            return np.array([])
        anchors = np.array(anchors)  # self.anchors这么写不行，因为self.anchors 是个字典。所以需要将anchor作为参数传进来

        n = idxs[:, 0]

        anchor_index = idxs[:, 3]

        conf = vecs[:, 0]

        cls = np.argmax(vecs[:, 5:], axis=1)
        cx = (idxs[:, 1] + vecs[:, 1]) * t
        cy = (idxs[:, 2] + vecs[:, 2]) * t

        w = anchors[anchor_index, 0] * np.exp(vecs[:, 3])
        h = anchors[anchor_index, 1] * np.exp(vecs[:, 4])
        w_half, h_half = w / 2, h / 2
        x1, y1, x2, y2 = cx - w_half, cy - h_half, cx + w_half, cy + h_half
        return np.stack((x1, y1, x2, y2, conf, cls, n), axis=1)

    def PILshow(self, img, last_boxes, draw, font, fill=(255, 0, 0), outline="blue", width=2, savepath=None,
                pltshow=False, imgshow=True):
        if np.any(last_boxes):
            for box in last_boxes:
                xybox = box[:4].astype("i4")
                text_x, text_y = list(box[:2])[0], list(box[:2])[1] - 15
                text_conf = list(box[:2])[0] + 35
                draw.text((text_x, text_y), cfg.COCO_DICT[int(box[5])], fill=fill, font=font)
                draw.text((text_conf, text_y), "%.2f" % box[4], fill=fill, font=font)
                draw.rectangle(list(xybox), outline=outline, width=width)
            if savepath != None:
                img.save(savepath)
        if pltshow:
            plt.clf()
            plt.imshow(img)
            plt.pause(0.1)
        elif imgshow:
            img.show()

    def cv2show(self, pic_file, last_boxes, color=(0, 0, 255), thickness=1, name='0.jpg',savedir = None,needshow=True):
        # cv2
        img = cv2.imread(pic_file)
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = "%s.jpg" % name.split('.')[0].split('_')[-1]
        if np.any(last_boxes):
            for box in last_boxes:
                cord = box[:4].astype("i4")
                pt1 = tuple(cord[:2])
                pt2 = tuple(cord[2:])
                text_x, text_y = int(list(box[:2])[0]), int(list(box[:2])[1]) - 10
                text_conf = int(list(box[:2])[0]) + 80
                cv2.rectangle(img, pt1, pt2, color, thickness)
                cv2.putText(img,cfg.COCO_DICT[int(box[5])],(text_x, text_y),font,0.5,color,thickness)
                cv2.putText(img,  "{:.2f}".format(box[4]), (text_conf, text_y), font, 0.5, color, thickness)
        if savedir != None and name.endswith(".jpg"):

            savepath = os.path.join(savedir,name)
            cv2.imwrite(savepath, img)
        if needshow:
            cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



'''


net_02.pt 正确网络
net_2_1 原网络 [98,15]
tensor([[0, 6, 6, 0],
        [0, 6, 6, 1],
        [0, 6, 6, 2]], device='cuda:0')
torch.Size([3, 6])
torch.Size([98, 15])
'''

if __name__ == '__main__':
    # np.set_printoptions(precision=4, threshold=np.inf, suppress=True)
    # net_path = r"D:\PycharmProjects\yolov3_01\save\20190925\nets\yolo_02.pth"
    # pic_name = r"pic_021.jpg"
    # pic_dir = r"D:\datasets\yolodatasets\datasets_20190801\datasets_resize"
    # save_dir = r"D:\PycharmProjects\yolov3_01\save\img"
    # save_path = os.path.join(save_dir, pic_name)
    #
    # torch.set_printoptions(threshold=np.inf, sci_mode=False)
    # detecter = Detector(net_path)
    # pic_file = os.path.join(pic_dir, pic_name)
    #
    # img = Image.open(pic_file)
    # last_boxes = detecter.detect(img, 0.6)
    #
    # draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(font="arial.ttf", size=12, encoding="utf-8")
    # save_dir1 = r"D:\PycharmProjects\yolov3_01\save\cv2img"
    # color = (0, 255, 0)
    # detecter.cv2show(pic_file, last_boxes, color, 2, pic_name, save_dir1)
    # color = (0, 0, 255)
    # detecter.PILshow(img, last_boxes, draw, font, fill=color, outline="yellow", savepath=save_path, pltshow=False)

    np.set_printoptions(precision=4, threshold=np.inf, suppress=True)
    net_path = r"D:\PycharmProjects\yolov3_01\save\20190925\nets\yolo_02.pth"

    pic_dir = r"D:\datasets\yolodatasets\datasets_20190801\datasets_resize"
    save_dir = r"D:\PycharmProjects\yolov3_01\save\img1"
    save_dir1 = r"D:\PycharmProjects\yolov3_01\save\cv2img1"
    utils.makedir(save_dir)
    utils.makedir(save_dir1)

    # pic_name = r"pic_008.jpg"
    # save_path = os.path.join(save_dir,pic_name)
    torch.set_printoptions(threshold=np.inf, sci_mode=False)
    detecter = Detector(net_path)

    for pic_name in os.listdir(pic_dir):
        pic_file = os.path.join(pic_dir, pic_name)
        save_path = os.path.join(save_dir, pic_name)
        # save_path1 = os.path.join(save_dir1, pic_name)
        img = Image.open(pic_file)
        last_boxes = detecter.detect(img, 0.6)

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font="arial.ttf", size=12, encoding="utf-8")

        color = (0, 255, 0)
        detecter.cv2show(pic_file, last_boxes, color, 2, pic_name, save_dir1, needshow=False)
        color = (0, 0, 255)
        detecter.PILshow(img, last_boxes, draw, font, fill=color, outline="yellow", savepath=save_path, pltshow=False,
                         imgshow=False)


