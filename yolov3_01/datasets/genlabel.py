import torch
import os
import numpy as np
from PIL import Image, ImageDraw
import json
from src import cfg


def img_resize(pic_path, pic_savepath):
    '''

    :param pic_path:
    :param pic_savepath:
    :return:
    '''
    for pic_name in os.listdir(pic_path):
        pic_file = os.path.join(pic_path, pic_name)
        pic_savefile = os.path.join(pic_savepath, pic_name)
        with Image.open(pic_file) as img:
            img = img.resize((416, 416))
            img.save(pic_savefile)


def makedir(path):
    '''
    如果文件夹不存在，就创建
    :param path:路径
    :return:路径名
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    return path


# 找到json文件读取其中数据
def makeLabel(json_dir, savefile, mode='w'):
    '''
    生成标签文件，将makeLable和checkdataset两个函数联用
    :param rootpath:根目录
    :param savename:标签保存名
    :param needcheck:是否需要对源数据进行检查，默认False
    :param preclear:是否需要预先对datasets文件夹和save文件夹进行清空，默认False
    :param removeProblem:是否需要移除不良数据，默认False
    :param mode:文件读写模式，默认'w'，备用参数
    :return:data的数量
    步骤：
        1.执行检查步骤，判断要生成标签的文件夹datasets是否存在，返回布尔值和json和jpg路径
        2.datasets存在，则生成保存的文件夹，生成数据列表，遍历json文件夹，读取数据并将结果按格式加入列表
        3.将列表转为数组，按文件名由小到大排序，并将结果保存，并返回数据个数
    '''
    # json_dir = r"D:\Users\Administrator\Pictures\yolov3\json\outputs"
    datalist = []
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
        except:
            print("the problem file is :", json_file)
            continue

        datalist.append(datalist_each)
    print(len(datalist))
    for data in datalist:
        print(data)

    with open(savefile, mode) as f:
        for data_line in datalist:

            for data_ in data_line:

                print(data_,end=' ')
                print(data_, end=' ',file=f)
            print('',file=f)


    return len(datalist)



