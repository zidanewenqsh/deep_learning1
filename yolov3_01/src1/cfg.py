COCO_ANN_FILE = 'data/coco/instances_train2017.json'

LABEL_FILE = "data/coco_label.txt"
IMG_BASE_DIR = "data/images"

DEVICE = "cuda"

IMG_HEIGHT = 416
IMG_WIDTH = 416

COCO_CLASS = ["person",
              "cat",
              "dog",
              "zebra",
              "giraffe",
              "banana",
              "airplane",
              "computer",
              "horse",
              "sheep"
              ]
CLASS_NUM = len(COCO_CLASS)
COCO_DICT = {0: "person",
             1: "cat",
             2: "dog",
             3: "zebra",
             4: "giraffe",
             5: "banana",
             6: "airplane",
             7: "computer",
             8: "horse",
             9: "sheep"
             }
ANCHORS_GROUP = {
    13: [[239, 239], [148, 229], [279, 113]],
    26: [[136, 125], [89, 159], [117, 53]],
    52: [[58, 58], [49, 74], [74, 30]]
}


ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
