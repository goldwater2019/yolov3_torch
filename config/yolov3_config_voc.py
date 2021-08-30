# coding=utf-8
# project
import os

PROJECT_ROOT = os.path.join(__file__, "../")

DATA_PATH = os.path.join(PROJECT_ROOT, "data/VOC")
PROJECT_PATH = PROJECT_ROOT

DATA = {"CLASSES": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor'],
        "NUM": 20}

# model
MODEL = {"ANCHORS": [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj 52*52 416/8
                     [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj 26 * 26
                     [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]],  # Anchors for big obj
         "STRIDES": [8, 16, 32],  # original image over stride
         "ANCHORS_PER_SCLAE": 3
         }

# train
TRAIN = {
    "TRAIN_IMG_SIZE": 416,
    "AUGMENT": True,
    "BATCH_SIZE": 4,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "EPOCHS": 50,
    "NUMBER_WORKERS": 4,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2  # or None
}

# test
TEST = {
    "TEST_IMG_SIZE": 416,
    "BATCH_SIZE": 4,
    "NUMBER_WORKERS": 2,
    "CONF_THRESH": 0.01,
    "NMS_THRESH": 0.5,
    "MULTI_SCALE_TEST": False,
    "FLIP_TEST": False
}

if __name__ == '__main__':
    print(__name__)
    print(__file__)
    print(PROJECT_ROOT)
