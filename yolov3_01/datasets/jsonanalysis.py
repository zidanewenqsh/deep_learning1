import json
import os
from PIL import Image
from src import cfg
from tool import utils
import numpy as np
import torch

json_dir = r"D:\datasets\yolodatasets\datasets_20190801\datasets_json\outputs"
pic_dir = r"D:\datasets\yolodatasets\datasets_20190801\datasets_resize"
checksavedir = r"D:\datasets\yolodatasets\datasets_20190801\img_check"
datalist = []
cordlist = []


for json_name in os.listdir(json_dir):
    datalist_each = []
    json_file = os.path.join(json_dir, json_name)

    try:
        with open(json_file, 'r', encoding='utf-8') as load_f:
            load_dict = json.load(load_f)
            outputs_object = load_dict['outputs']['object']
            pic_name = load_dict['path'].split('\\')[-1]


            datalist_each.append(pic_name)

            for object_data in outputs_object:
                name = object_data['name']
                bndbox = object_data['bndbox']

                x1, y1, x2, y2 = bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']
                cls = cfg.COCO_CLASS.index(name)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                b_w = x2 - x1
                b_h = y2 - y1

                datalist_each.extend([cls, cx, cy, b_w, b_h])
                cordlist.append([x1,y1,x2,y2])
    except:
        print("the problem file is :", json_file)
        continue

    datalist.append(datalist_each)


