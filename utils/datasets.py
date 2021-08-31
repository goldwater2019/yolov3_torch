# coding=utf-8
import os
import sys

sys.path.append("..")
sys.path.append("../utils")
import torch
from torch.utils.data import Dataset, DataLoader
import config.yolov3_config_voc as cfg
import cv2
import numpy as np
import random
# from . import data_augment as dataAug
# from . import tools

import utils.data_augment as dataAug
import utils.tools as tools


class VocDataset(Dataset):
    def __init__(self, anno_file_type, img_size=416):
        """
        @param anno_file_type: 标注数据类型. train/test
        @param img_size: 图片大小(正方形)

        cfg = {"CLASSES": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor'],
        "NUM": 20}

        """
        self.img_size = img_size  # For Multi-training
        self.classes = cfg.DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.__annotations = self.__load_annotations(anno_file_type)  # 加载标注数据

    def __len__(self):
        """
        dataset 默认重载方法
        获得数据共有多少数据
        """
        return len(self.__annotations)

    def __getitem__(self, item):
        """
        @param item: 获得第 item 条数据
        通过这个方法来获得数据
        """
        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        # 得到的是 img, bboxes_org
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW
        # 重新转置img -> 得到原始图片

        item_mix = random.randint(0, len(self.__annotations) - 1)
        # 随机选择一个 mix 的图片索引
        img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)
        # 参与 mix 的torch图片 img_mix
        # box 是 bboxes_mix

        img, bboxes = dataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        # img: [c,h,w]
        # bboxes: [n_boxes, 6]
        # 6 = 5 + 1(confidence)
        del img_org, bboxes_org, img_mix, bboxes_mix

        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__creat_label(bboxes)

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()

        return img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __load_annotations(self, anno_type):
        """
        @param anno_type: 标注类型. train/test
        @return: list[str]
        str: filename x1,y1,x2,y2,label x1,y1,x2,y2,label
        """

        assert anno_type in ['train', 'test'], "You must choice one of the 'train' or 'test' for anno_type parameter"
        anno_path = os.path.join(cfg.PROJECT_PATH, 'data', anno_type + "_annotation.txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))
        assert len(annotations) > 0, "No images found in {}".format(anno_path)

        return annotations

    def __parse_annotation(self, annotation):
        """
        @param annotation: [image_path xmin,ymin,xmax,ymax,class_index xmin,ymin,xmax,ymax,class_index]
        @return (image, bboxes): image, shape of [img_size, img_size,3](hwc), bboxes, shape of (n_boxes, x,y,x,y,class_index)
        Data augument.
        :param annotation: Image' path and bboxes' coordinates, categories.
        ex. [image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...]
        :return: Return the enhanced image and bboxes. bbox'shape is [xmin, ymin, xmax, ymax, class_ind]
        """
        anno = annotation.strip().split(' ')

        img_path = anno[0]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array(
            [list(map(float, box.split(','))) for box in anno[1:]])  # box: xmin,ymin,xmax,ymax,class_index
        # bboxes: an array of item
        # item: an array of box

        img, bboxes = dataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))  # 对图片进行随机翻转
        img, bboxes = dataAug.RandomCrop()(np.copy(img), np.copy(bboxes))  # 对图片进行随机抠图
        img, bboxes = dataAug.RandomAffine()(np.copy(img), np.copy(bboxes))  # 对图片进行随机仿射变换(移动)
        img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))  # 对图片进行旋转

        return img, bboxes

    def __creat_label(self, bboxes):
        """
        根据传进来的 GT bbox, 生成在不同尺度上的anchor 检测框
        @param bboxes: [n_boxes, 6]
                        6: [xmin, ymin, xmax, ymax, class_index, confidence]

                        1. 按照顺序遍历 bbox, (xmin,ymin,xmax,ymax) 转化成(x_center, y_center, w, h)
                        2. 根据 stride 等比例缩放(x,y,w,h)
                        3. 计算每个检测层的 anchor 和实际的 bbox, 选择最大的 anchor 来预测 bbox
                            如果所有的 iou 都小于 0.3, 选择最大的那个 iou 来预测 bbox
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.

        """

        anchors = np.array(cfg.MODEL["ANCHORS"])
        # 不同 scale 的长宽
        # [small_anchors, mid_anchors, large_anchors]
        strides = np.array(cfg.MODEL["STRIDES"])
        # 不同 scale 下面需要缩放的比例
        # np.array([8, 16, 32])
        train_output_size = self.img_size / strides
        # [52, 26, 13]
        # the size of train_output_size
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]
        # 每一个尺度应该要生成的 anchor
        # 这里是一个常数 3
        n_scales = strides.shape[0]  # add by xinsen, 获得 n_scale 的数量, 让数据根据参数可调整
        assert strides.shape[0] == anchors.shape[0]

        label = [
            np.zeros((int(train_output_size[i]), int(train_output_size[i]), anchors_per_scale, 6 + self.num_classes))
            for i in range(n_scales)]
        # label 是每个 scale 下面的 scaled_label 的集合
        # 每个 scaled_label:
        # shape of (
        #           img_size/stride,
        #           img_size/stride,
        #           anchors_per_scale,
        #           4(x_min, y_min, x_max, y_max)+1(class_index)+1(confidence)+self.num_classes
        #           )
        # label[small_scaled_label, mid_scaled_label, large_scaled_label]
        for i in range(n_scales):
            label[i][..., 5] = 1.0
            # 每一个 GT 的 confidence 都是 1.0

        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]  # Darknet the max_num is 30
        # 用一个定长的 bboxes_xywh 来表示一张增强后的图片的 bbox
        bbox_count = np.zeros((n_scales,))
        # 记录每个 scale 下面 bbox 的数量

        for bbox in bboxes:  # (x_min, y_min, x_max, y_max, class_index, confidence)
            bbox_coor = bbox[:4]
            # (x_min, y_min, x_max, y_max)
            bbox_class_ind = int(bbox[4])
            # class_index
            bbox_mix = bbox[5]
            # bbox confidence

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)
            # 使用 labelSmothing 的方式平滑标签
            # smothing label(delta=0.001)
            # 0->delta/num_class
            # 1->1-delta

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # (bbox_coor[2:] + bbox_coor[:2]) * 0.5: (x_center, y_center)
            # bbox_coor[2:] - bbox_coor[:2]: (w,h)
            # bbox_wywh: (4,)
            # print("bbox_xywh: ", bbox_xywh)

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
            # 不同尺度下的 bbox_xywh
            # 每个 scale 下面, center of (x,y) -> (x/scale, y/scale)
            #                measure (w,h) -> (w/scale, h/scale)
            # bbox_xywh_scaled: shape of (n_scales, 4)

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 0.5 for compensation
                # 每一个 scale 下面 box 的 x_center,y_center
                # TODO 为什么要加0.5的补偿
                #   中心偏移(0.5,0.5)之后, 给定当前感受野下的宽和高, 比较 gt 的 bbox 和当前 bbox 的 iou
                #   如果 iou 大于一个给定的 threshold, 则说明给定感受野下的 anchor 可以认为是正样本
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                # iou_scale: [3,1]
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3  # [3,1]

                if np.any(iou_mask):
                    # 有任何一个 box 的 iou > 0.3就能人为这个 box 框到了目标
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    # 说明当前 scale 下面某个视野存在目标
                    # 将目标的索引取出来

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    # iou>0.3的标签的 bbox 为 xywh(without scale)
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


if __name__ == "__main__":

    voc_dataset = VocDataset(anno_file_type="train", img_size=416)
    dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1, num_workers=0)

    for i, (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(dataloader):
        if i == 0:
            print(img.shape)
            print(label_sbbox.shape)
            print(label_mbbox.shape)
            print(label_lbbox.shape)
            print(sbboxes.shape)
            print(mbboxes.shape)
            print(lbboxes.shape)

            if img.shape[0] == 1:
                labels = np.concatenate([label_sbbox.reshape(-1, 26), label_mbbox.reshape(-1, 26),
                                         label_lbbox.reshape(-1, 26)], axis=0)
                labels_mask = labels[..., 4] > 0
                labels = np.concatenate([labels[labels_mask][..., :4], np.argmax(labels[labels_mask][..., 6:],
                                                                                 axis=-1).reshape(-1, 1)], axis=-1)

                print(labels.shape)
                tools.plot_box(labels, img, id=1)
