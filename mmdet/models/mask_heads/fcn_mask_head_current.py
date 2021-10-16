import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import mask_target, force_fp32, auto_fp16

import matplotlib.pyplot as plt 
import kornia
import math


@HEADS.register_module
class FCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_maskopt=False,
                 num_protos=32,
                 generate_weight=False,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.generate_weight = generate_weight
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_protos = num_protos
        self.fp16_enabled = False
        loss_mask.use_sigmoid = True
        loss_mask.use_mask = False
        self.loss_mask = build_loss(loss_mask)
        # self.loss_mask = nn.CrossEntropyLoss()

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = self.num_protos
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits_0 = nn.Conv2d(logits_in_channel, logits_in_channel, 3, padding=1)
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        # self.conv_logits = nn.Sequential(
        #     nn.Conv2d(self.conv_out_channels, self.conv_out_channels, 3, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(self.conv_out_channels, self.num_protos, 1),
        # )

        self.proto_coeff = nn.Sequential(
            nn.Conv2d(self.conv_out_channels, self.conv_out_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(self.conv_out_channels, self.num_protos, 1),
        )

        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits, self.conv_logits_0]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x, query_feat=None, ref_feat=None, num_pos=None, num_cls_share=None):
        for conv in self.convs:
            x = conv(x)

        proto_feat = F.adaptive_avg_pool2d(x, 1)
        proto_coeff = self.proto_coeff(proto_feat)

        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        x = self.conv_logits_0(x).relu()
        prototypes = self.conv_logits(x)

        # assemble prototypes with proto_coeff
        # mask_pred = (prototypes.relu() * proto_coeff.tanh()).sum(1)
        mask_pred = (prototypes.relu() * proto_coeff.sigmoid()).sum(1)

        return mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels, num_pos=None):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets, torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets.gt(0.5).float() / 2 + 0.5)
            loss_mask = loss_mask + math.log(0.5)

        # loss_mask = torch.clamp(loss_mask, 0, 10)
        # print(loss_mask.item())

        loss['loss_mask'] = loss_mask
        
        return loss

    def get_seg_masks(self, mask_pred, mask_feats, ref_feats, det_bboxes, det_labels, real_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale, gt_masks=None):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
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

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            real_label = real_labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic and not self.use_maskopt:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            if gt_masks is not None:
                bbox_mask = mmcv.imresize(gt_masks[i][0].cpu().numpy(), (w, h))
            else:
                bbox_mask = mmcv.imresize(mask_pred_, (w, h))

            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[real_label].append(rle)

        return cls_segms, mask_pred[:, 0:1]
