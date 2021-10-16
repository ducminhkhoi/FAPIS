
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import builder
from mmcv.cnn import xavier_init, normal_init
import mmcv

from mmdet.core import force_fp32, multi_apply, mask_target, auto_fp16
from ..builder import build_loss
import numpy as np
import pycocotools.mask as mask_util
from .sparsemax import Sparsemax
import matplotlib.pyplot as plt
from torchvision.ops import roi_align
from ..utils import bias_init_with_prob, Scale, ConvModule
import math
from .unet_parts import inconv, down, up, outconv
from scipy.optimize import linear_sum_assignment as hungarian
import time
# from lapsolver import solve_dense as hungarian


def dice_loss(input, target):
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1)
    c = torch.sum(target * target, 1)
    d = (2 * a) / (b + c + 1e-16)
    return (1-d).mean()


class PrototypeNet(nn.Module):
    def __init__(self, protonet, num_parts, constraint_part=False):
        super().__init__()
        if protonet is None:
            protonet = UNetD2(in_channels=256, n_classes=num_parts)

        self.protonet = protonet
        self.constraint_part = constraint_part
        self.num_parts = num_parts

        if constraint_part:
            W, H = torch.load(f'data/NMF_{num_parts}_1000.pth')
            H = H / H.max(1)[:, None]
            self.nmf_parts = torch.from_numpy(H)[None, None, ...]

    def forward(self, x, proto_coeff, training=True):
        prototypes = self.protonet(x)

        proto_coeff = proto_coeff[..., None, None]

        prototypes = prototypes.relu()
        prototypes = prototypes / (F.adaptive_max_pool2d(prototypes, 1) + 1e-16) # max_norm
        mask_pred = (proto_coeff.sigmoid() * prototypes).sum(1) 

        # ### for visualization
        # N, C, H, W = prototypes.shape
        # parts = prototypes.reshape(N, 4, -1, H, W).permute(0, 1, 3, 2, 4).reshape(N, 4*H, -1).data.cpu().numpy()
        # masks = mask_pred.data.cpu().numpy()

        # for i in range(20):
        #     plt.matshow(parts[i])
        #     plt.savefig(f'vis/parts_0.01_{i}.jpg')

        #     # plt.matshow(masks[i])
        #     # plt.savefig(f'vis/mask_0.01_{i}.jpg')

        #     plt.close('all')

        # exit()

        if training:
            loss_part = None 
            if self.constraint_part:
                N, C, H, W = prototypes.shape
                protos = prototypes.flatten(2)#.sigmoid()

                with torch.no_grad():
                    temp_proto = protos.unsqueeze(2)
                    nmf_parts = self.nmf_parts.to(temp_proto.device)

                    inter = (temp_proto * nmf_parts).sum(-1)
                    union = temp_proto.sum(-1) + nmf_parts.sum(-1) - inter
                    ious = inter / (union + 1e-16)
                    ious = ious.data.cpu().numpy()

                    gt_parts = torch.cat([nmf_parts[0, 0][hungarian(1-ious[i])[1]] for i in range(N)])

                loss_part = dice_loss(protos.reshape(N*C, H*W), gt_parts)

            return mask_pred, loss_part
        else:
            return mask_pred

    def loss(self, mask_pred, mask_targets):
        loss_mask = dice_loss(mask_pred, mask_targets)

        loss_dict = dict(loss_mask=loss_mask)

        return loss_dict

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)

        return mask_targets

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, real_labels, mask_thr_binary,
                ori_shape, scale_factor, rescale, gt_masks=None):

        mask_pred = mask_pred.unsqueeze(1)
        mask_pred = mask_pred.cpu().numpy()

        mask_pred = mask_pred.astype(np.float32)

        cls_segms = [[] for _ in range(80)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1
        real_labels = real_labels.cpu().numpy()

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        im_mask_all = np.zeros((img_h, img_w), dtype=np.uint8)

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            real_label = real_labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            if gt_masks is not None:
                bbox_mask = mmcv.imresize(gt_masks[i][0].cpu().numpy(), (w, h))
            else:
                bbox_mask = mmcv.imresize(mask_pred_, (w, h))

            bbox_mask = (bbox_mask > mask_thr_binary).astype(
                np.uint8)
            real_h, real_w = im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w].shape
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask[:real_h, :real_w]

            im_mask_all[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[real_label].append(rle)

        return cls_segms, im_mask_all

class UNetD2(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2):
        super(UNetD2, self).__init__()
        self.inc = inconv(in_channels, 16 * w)
        self.down1 = down(16 * w, 32 * w)
        self.down2 = down(32 * w, 32 * w)
        self.up1 = up(64 * w, 16 * w)
        self.up2 = up(32 * w, 16 * w)
        self.outc = outconv(16 * w, n_classes)

    def forward(self, x):
        x1 = self.inc(x) # 16
        x2 = self.down1(x1) # 32
        x3 = self.down2(x2) # 32
        x = self.up1(x3, x2) # 16
        x = self.up2(x, x1) # 16
        x = self.outc(x)
        return x