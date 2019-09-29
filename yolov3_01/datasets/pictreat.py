import torch
import torch.nn as nn
from PIL import Image,ImageDraw
import numpy as np
import os

pic_dir = r"D:\datasets\yolodatasets\datasets_20190801\datasets_tomake"
pic_savedir = r"D:\datasets\yolodatasets\datasets_20190801\datasets_resize"
img_blank = Image.new(mode="RGB",size=(416,416),color=(0,0,0))
width,height = img_blank.size
print(width,height)

for i,img_name in enumerate(os.listdir(pic_dir)):
    img_b = Image.new(mode="RGB", size=(416, 416), color=(0, 0, 0))
    cx, cy = 208,208
    img_path = os.path.join(pic_dir,img_name)
    img_name_ = "pic_%03d.jpg" % i
    img_save_path = os.path.join(pic_savedir,img_name_)
    with Image.open(img_path) as img_file:
        w,h = img_file.size
        while w > 416 or h > 416:
            w = int(w * 0.99)
            h = int(h * 0.99)
        x_ = cx - w//2
        y_ = cy - h//2
        print(img_name)
        print(w,h)
        img_file_ = img_file.resize((w,h))
        img_b.paste(img_file_,(x_,y_))
        print(img_save_path)
        img_b.save(img_save_path)
        # img_b.show()




# with Image.open(img_1_) as img_1:
#     # w,h = img_1.size
#     # while w>416 or h>416:
#     #     w = int(w*0.99)
#     #     h = int(h*0.99)
#     img3 = img_1.resize((100,100))
#     img2 = img_1.paste(img3)
#     img2.save("a.jpg")
    # img_1.show()