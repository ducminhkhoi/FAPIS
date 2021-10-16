from collections import Sequence

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch

import cv2

from pycocotools import mask as cocomask
from pycocotools import coco

def prepare_rf(img, ann, cat, mask=None, enlarge=False):
    # while True:
    #     index = np.random.randint(len(ann))
    #     cat_rf = ann[index]['category_id']
    #     if cat_rf == cat and not ann[index]['iscrowd']:
    #         break
    # x, y, w, h = np.array(ann[index]['bbox']).astype(int)
    x, y, w, h = np.array(ann['bbox']).astype(int)

    if enlarge:
        H, W = img.shape[:2]
        delta_x, delta_y = int(w * 0.5), int(h * 0.5)
        x1, y1 = max(0, x - delta_x), max(0, y - delta_y)
        x2, y2 = min(W, x + w + 1 + delta_x), min(H, y + h + 1 + delta_y)
    else:
        x1, y1, x2, y2 = x, y, x+w+1, y+h+1

    crop = img[y1:y2, x1:x2, :]
    if mask is not None:
        crop_mask = mask[y1:y2, x1:x2]
        return crop, crop_mask
        
    return crop


def prepare_rf_test(img, ann, cls_name, mask=None, enlarge=False):
    x, y, w, h = np.array(ann['bbox']).astype(int)
    H, W = img.shape[:2]

    if enlarge:
        delta_x, delta_y = int(w * 0.5), int(h * 0.5)
        x1, y1 = max(0, x - delta_x), max(0, y - delta_y)
        x2, y2 = min(W, x + w + 1 + delta_x), min(H, y + h + 1 + delta_y)
    else:
        x1, y1, x2, y2 = x, y, x+w+1, y+h+1

    crop = img[y1:y2, x1:x2, :]
    if mask is not None:
        crop_mask = mask[y1:y2, x1:x2]
        return crop, crop_mask
    
    # # vis rf_img
    # b, g, r = cv2.split(crop)
    # show_img = cv2.merge([r, g, b])
    # import matplotlib.pyplot as plt
    # plt.imshow(show_img)

    # plt.text(-1, -1, '{}'.format(cls_name), fontsize=30)
    # plt.show()
    return crop


def zero_pad(img):
    _, h, w = img.shape
    max_ = np.max(img.shape)
    pad = np.pad(
        img, ((0, 0), (0, max_ - h), (0, max_ - w)),
        'constant',
        constant_values=(0, 0))
    return pad


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()
