import os
import numpy as np
from PIL import Image,ImageDraw
label_path = r"D:\PycharmProjects\yolov3_01\labels\label_02_1.txt"
pic_file = r"D:\PycharmProjects\yolov3_01\picdata\pic_001.jpg"
datalist = list()
with open(label_path) as f:
    for i,line in enumerate(f.readlines()):
        print(line)
        datalist.append(line)
for datas in datalist:
    datas_ = datas.split()
    print(datas_)
    name = datas_[0]
    boxes_ = np.array(datas_[1:], dtype=np.float)
    print(boxes_)
    print(type(boxes_))
    print(len(boxes_))
    print(boxes_.size)
    boxes = np.split(boxes_, boxes_.size//5)
    print(boxes)
with Image.open(pic_file) as img:
    draw = ImageDraw.Draw(img)

    for box_ in boxes:
        cx, cy, w, h = box_[1:]
        x1 = int(cx-w//2)
        y1 = int(cy -h//2)
        x2 = int(x1+w)
        y2 = int(y1+h)
        box = (x1,y1,x2,y2)
        print(box)
        draw.rectangle(list(box),outline="blue", width=3)
    img.show()
