# coding=utf-8
import random

import cv2
import numpy as np


class RandomHorizontalFilp(object):
    """
    随机水平翻转对象
    """

    def __init__(self, p=0.5):
        """
        @param p: 发生随机变换的概率
        初始化
        """
        self.p = p

    def __call__(self, img, bboxes):
        """
        对图像, bboxes 进行转换
        当且仅当随机产生的随机数<设定的阈值的时候发生翻转
        @param img: [h,w,c], an array representation of an image with opencv format
        """
        if random.random() < self.p:
            _, w_img, _ = img.shape
            # img = np.fliplr(img)
            img = img[:, ::-1, :]  # 水平方向进行翻转
            bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
            # w_img 图片的宽度
            # bboxes[i, [2,0]] 第 i 个 box的 xmax 和 xmin
        return img, bboxes


class RandomCrop(object):
    """
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        """
        @param img: [h,w,c], an array representation of an image with opencv format
        """
        if random.random() < self.p:
            h_img, w_img, _ = img.shape

            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            # np.min(bboxes[:, 0:2], axis=0)
            # bboxes[i, 0:2] 是第 i 个 box 的 xmin,ymin
            # bboxes[:,0:2]: shape with [n_boxes, 2]
            # np.min(bboxes[:,0:2], axis=0) -> 所有框中的 [min(xmin), min(ymin)]
            # np.max(bboxes[:, 2:4], axis=0) -> 所有框中的 [max(xmax), max(ymax)]
            # max_bbox: [4,1]
            max_l_trans = max_bbox[0]
            # min(xmin)
            max_u_trans = max_bbox[1]
            # min(ymin)
            max_r_trans = w_img - max_bbox[2]
            # w-max(xmax) -> 最大 x 方向可腐蚀的 pixel
            max_d_trans = h_img - max_bbox[3]
            # h-max(ymax) -> 最大 y 方向可腐蚀的 pixel

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            # 此处应该是 min(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))?
            crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            img = img[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
            # 从图片中抠出来相应的地方

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            # 相应的 box 的位置减去 xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
            # 相应的 box 的位置减去 ymin
        return img, bboxes


class RandomAffine(object):
    """
    随机仿射变换

    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            # 得到可以包含所有bbox的最大bbox
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            # max_bbox[0], max_bbox[1] 表示最大的 box 的(xmin, ymin)
            # max_bbox[2], max_bbox[3] 表示最大的 box 的(xmax, ymax)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            # 最大可向右移动 max_r_trnas 个单位
            max_d_trans = h_img - max_bbox[3]
            # 最大可向下移动 max_d_trans 个单位

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w_img, h_img))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            # 所有的 box 向右移动 tx 个单位
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
            # 所有的 box 向上移动 ty 个单位
        return img, bboxes


class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """

    def __init__(self, target_shape, correct_box=True):
        """
        @param target_shape: 目标图像的 shape,
                                <code>self.h_target = self.w_target = target_shape</code>
        @param correct_box[boolean]: <code> self.correct_box = correct_box
        """
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes):
        """
        @param img: [h,w,c] opencv 格式的图片张量
        @param bboxes: [n_boxes, xmin, ymin, xmax, ymax]
        """
        h_org, w_org, _ = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 将 opencv 原始的 BGR 视野的图片转换成 RGB 视野的图片
        # img: [h,w,3]

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        # resize_ratio 表示将图片等比例缩放的比例
        resize_w = int(resize_ratio * w_org)
        # resize 之后的 w
        resize_h = int(resize_ratio * h_org)
        # resize 之后的 h
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        # 填充之后的图默认使用 128 标记(中间数)
        dw = int((self.w_target - resize_w) / 2)
        # 宽上面做边需要 padding 的单位
        dh = int((self.h_target - resize_h) / 2)
        # 高上面下边需要 padding 的单位
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image


class Mixup(object):
    """
    将两张图片按照一定的比例进行 mixup
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        """
        @param img_org: 原始图片
        @param bboxes_org: 原始 box. [n_boxes, 4]
        @param img_mix: 混合图片
        @param bboxes_mix: 混合 box: [n_boxes, 4]
        """
        if random.random() > self.p:
            lam = np.random.beta(1.5, 1.5)
            img = lam * img_org + (1 - lam) * img_mix
            # 将两张图片加权混合(mixedUp)
            # 假设 lam=0.77
            # 表示混合之后的图片0.77来自原图, 0.23 来自于混合图
            bboxes_org = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1)
            # bboxes_org: [n_boxes_org, 5]
            # 5: 4+1
            bboxes_mix = np.concatenate(
                [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1)
            # bboxes_org: [n_boxes_mix, 5]
            bboxes = np.concatenate([bboxes_org, bboxes_mix])
            # bboxes: [n_boxes, 5]

        else:
            img = img_org
            bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)

        return img, bboxes


class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes
