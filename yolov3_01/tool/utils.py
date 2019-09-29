import torch
import numpy as np
import os

def makedir(path):
    '''
    如果文件夹不存在，就创建
    :param path:路径
    :return:路径名
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def toTensor(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, torch.FloatTensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, (list, tuple)):
        return torch.tensor(list(data)).float()  # 针对列表和元组，注意避免list里是tensor的情况
    elif isinstance(data, torch.Tensor):
        return data.float()
    return


def toNumpy(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, (list, tuple)):
        return np.array(list(data))  # 针对列表和元组
    return


def toList(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.numpy().tolist()
    elif isinstance(data, (list, tuple)):
        return list(data)  # 针对列表和元组
    return


def area(box):
    box = toNumpy(box)
    return np.multiply(np.subtract(box[3], box[1]), np.subtract(box[2], box[0]))


def areas(boxes):
    boxes = toNumpy(boxes)
    # print(boxes)
    return np.multiply(np.subtract(boxes[:, 3], boxes[:, 1]), np.subtract(boxes[:, 2], boxes[:, 0]))


def iou(box, boxes, mode='inter'):
    box = toNumpy(box)
    boxes = toNumpy(boxes)
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)
    box_area = area(box)
    boxes_areas = areas(boxes)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    #重点错处
    # inter = np.multiply(np.subtract(xx2, xx1), np.subtract(yy2, yy1))
    inter = np.multiply(np.maximum((xx2-xx1), 0), np.maximum((yy2- yy1),0))
    # print("inter",inter)
    # print("union",np.subtract(np.add(box_area, boxes_areas), inter))
    # print("area",box_area,boxes_areas)
    if mode == 'inter':
        return np.divide(inter, np.subtract(np.add(box_area, boxes_areas), inter))
    elif mode == 'min':
        return np.divide(inter, np.minimum(box_area, boxes_areas))


def nms(boxes, threh=0.3, mode='inter'):
    boxes = toNumpy(boxes)
    if boxes.ndim == 1:
        return np.expand_dims(boxes, axis=0)
    keep_boxes = []
    if boxes.shape[0] == 0:
        return keep_boxes
    sort_boxes = boxes[np.argsort(-boxes[:, 4])]
    # print(sort_boxes.shape)
    # print(sort_boxes[0])
    while sort_boxes.shape[0] > 0:
        # print(boxes.shape[0])
        _box = sort_boxes[0]
        keep_boxes.append(_box)
        if boxes.shape[0] > 1:
            _boxes = sort_boxes[1:]
            _iou = iou(_box, _boxes, mode)
            sort_boxes = _boxes[np.less(_iou, threh)]
        else:
            break
    return keep_boxes


if __name__ == '__main__':
    torch.manual_seed(100)
    a = np.array([24, 23, 55, 66])
    b = torch.from_numpy(a).float()
    bs = torch.Tensor([[2, 2, 30, 30, 40], [3, 3, 25, 25, 60], [18, 18, 27, 27, 15]])
    print(iou(a, bs))
    print(nms(bs))