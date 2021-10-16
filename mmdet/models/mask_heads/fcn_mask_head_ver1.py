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
# import kornia


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
                 generate_weight=False,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.generate_weight = False
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
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)

        # TODO: change here
        self.use_maskopt = use_maskopt
        # if use_maskopt:
        #     self.edge_det = kornia.filters.Sobel() 
        #     upsample_in_channels = (
        #         self.conv_out_channels if self.num_convs > 0 else in_channels)

        #     out_channels = 1 if self.class_agnostic else self.num_classes

        #     self.convs = nn.ModuleList()
        #     for i in range(self.num_convs):
        #         in_channels = (
        #             self.in_channels if i == 0 else self.conv_out_channels)
        #         padding = (self.conv_kernel_size - 1) // 2
        #         self.convs.append(
        #             ConvModule(
        #                 in_channels,
        #                 self.conv_out_channels,
        #                 self.conv_kernel_size,
        #                 padding=padding,
        #                 conv_cfg=conv_cfg,
        #                 norm_cfg=norm_cfg))

        #     if self.upsample_method is None:
        #         self.upsample = None
        #     elif self.upsample_method == 'deconv':
        #         self.upsample = nn.ConvTranspose2d(
        #             upsample_in_channels,
        #             self.conv_out_channels,
        #             self.upsample_ratio,
        #             stride=self.upsample_ratio)
        #     else:
        #         self.upsample = nn.Upsample(
        #             scale_factor=self.upsample_ratio, mode=self.upsample_method)

        #     logits_in_channel = (
        #         self.conv_out_channels
        #         if self.upsample_method == 'deconv' else upsample_in_channels)
        #     self.conv_logits = nn.Conv2d(logits_in_channel, out_channels+1, 1)

        # else:
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

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)

        # if self.generate_weight:
        #     self.weights_predict = nn.Sequential(
        #         nn.Linear(256 * 2, 1024),
        #         nn.ReLU(True),
        #         nn.Dropout(),
        #         nn.Linear(1024, 1024),
        #         nn.ReLU(True),
        #         nn.Dropout(),
        #         nn.Linear(1024, (self.conv_out_channels+1)*self.num_classes), # +1 for bias
        #     )

        #     loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1)
        #     self.loss_cls = build_loss(loss_cls)

        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.ws = 3

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x, query_feat=None, ref_feat=None, num_pos=None, num_cls_share=None):
        # if num_cls_share is None:
        #     num_cls_share = len(self.convs)

        # for conv in self.convs[:num_cls_share]:
        #     x = conv(x)

        # if self.generate_weight:
        #     cls_feat = F.adaptive_avg_pool2d(x, 1)
        #     cls_feat = cls_feat.view(cls_feat.size(0), 1, -1)

        #     predicted = self.weights_predict(torch.cat([query_feat, ref_feat], 1))
        #     weight, bias = predicted.view(-1, self.num_classes, 
        #                         self.conv_out_channels+1).split(self.conv_out_channels, 2)

        #     cls_score = ((weight * cls_feat).sum(2, keepdim=True) + bias).view(-1, self.num_classes)

        # if num_pos is not None:
        #     x = x[:num_pos]

        # for conv in self.convs[num_cls_share:]:
        #     x = conv(x)

        for conv in self.convs:
            x = conv(x)

        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)

        if self.use_maskopt:
            edge_pred, mask_pred = mask_pred.split(2, dim=1)
            return edge_pred, mask_pred

        if self.generate_weight:
            return cls_score, mask_pred

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
        # if self.generate_weight:
        #     cls_score, mask_pred = mask_pred
        #     label_weights = torch.ones(len(cls_score)).to(labels.device)

        #     avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        #     loss['loss_cls_mask'] = self.loss_cls(
        #         cls_score,
        #         labels,
        #         label_weights,
        #         avg_factor=avg_factor,
        #         reduction_override=None)

        # if num_pos is not None:
        #     labels = labels[:num_pos]
            
        # if self.use_maskopt:
        #     edge_pred, mask_pred = mask_pred
            
        #     device = edge_pred.device
        #     N, H, W = mask_targets.shape
        #     with torch.no_grad():
        #         edges = self.edge_det(mask_targets.unsqueeze(0)).squeeze(0) 

        #         edges[:, 0, :] = torch.where(mask_targets[:, 0, :]==1, mask_targets[:, 0, :], edges[:, 0, :])
        #         edges[:, :, 0] = torch.where(mask_targets[:, :, 0]==1, mask_targets[:, :, 0], edges[:, :, 0])
        #         edges[:, H-1, :] = torch.where(mask_targets[:, H-1, :]==1, mask_targets[:, H-1, :], edges[:, H-1, :])
        #         edges[:, :, W-1] = torch.where(mask_targets[:, :, W-1]==1, mask_targets[:, :, W-1], edges[:, :, W-1])

        #         edge_targets = (edges > 0.25).long()

        #         weight = torch.tensor([(edges==1).sum(), (edges==0).sum()]).float() / edges.numel()

        #         edge_area = F.conv2d(edges.unsqueeze(1).float(), torch.ones(1, 1, self.ws, self.ws).to(device), 
        #                         padding=self.ws//2)

        #     loss_edge = F.cross_entropy(edge_pred, edge_targets, weight.to(device))
        #     loss['loss_edge'] = loss_edge

        #     # loss_mask = F.binary_cross_entropy_with_logits(mask_pred[edge_area > 0], 
        #     #                 mask_targets.unsqueeze(1)[edge_area > 0].float())

        #     loss_mask = F.binary_cross_entropy_with_logits(mask_pred, 
        #                     mask_targets.unsqueeze(1).float())

        # else:
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets, torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)

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

        if isinstance(mask_pred, tuple):
            edge_pred, mask_pred = mask_pred

        # if False:
        #     edge_pred, mask_pred = mask_pred
        #     edge_pred = edge_pred.argmax(1, keepdim=True).float()
        #     device = mask_pred.device

        #     # edge_area = F.conv2d(edge_pred, torch.ones(1, 1, self.ws, self.ws).to(device), padding=self.ws//2)

        #     # a = torch.where(edge_area > 0, mask_pred.sigmoid() * 2 - 1, torch.zeros_like(mask_pred))
        #     # a = torch.where(edge_area > 0, mask_pred.tanh(), torch.zeros_like(mask_pred))

        #     a_0 = torch.where(mask_pred > 0, torch.ones_like(mask_pred), -torch.ones_like(mask_pred)) # can change to binary

        #     a = torch.where(edge_pred > 0, a_0, torch.zeros_like(mask_pred))

        #     # b = F.cosine_similarity(mask_feats.unsqueeze(1), ref_feats, dim=2)
        #     # b = F.interpolate(b, a.shape[-2:], mode='bilinear', align_corners=True)

        #     alpha, beta, gamma, delta, lambd = 1, 1, 1, 1, 1e-1
        #     n_iters = 100

        #     # c = alpha * a + beta * b.mean(1, keepdim=True) 

        #     # f = torch.tensor([  [0, 1/4, 0],
        #     #                     [1/4, 0, 1/4],
        #     #                     [0, 1/4, 0]])[None, None, :, :].to(device)

        #     f = torch.tensor([  [0, 1, 0],
        #                         [1, 0, 1],
        #                         [0, 1, 0]])[None, None, :, :].float().to(device)

        #     H, W = a.shape[-2:]
        #     divide = torch.ones(H, W) * 1/4
        #     divide[0, :] = 1/3
        #     divide[H-1, :] = 1/3
        #     divide[:, 0] = 1/3
        #     divide[:, W-1] = 1/3
        #     divide[0, 0] = 1/2
        #     divide[0, W-1] = 1/2
        #     divide[H-1, 0] = 1/2
        #     divide[H-1, W-1] = 1/2
        #     divide = divide[None, None, :, :].float().to(device)

        #     # plt.matshow(edge_pred[0, 0].data.cpu().numpy())
        #     # plt.savefig('edge.jpg')

        #     # plt.matshow(a_0[0, 0].data.cpu().numpy())
        #     # plt.savefig('qual1.jpg')

        #     d = a_0    
                
        #     for i in range(n_iters):
        #         d_avg = F.conv2d(d, f, padding=1) * divide
        #         # exp = alpha * a * torch.exp(-(a*d).sum(dim=[2,3], keepdim=True))
        #         sigmoid = torch.sigmoid(-(a*d).sum(dim=[2,3], keepdim=True))
        #         exp = alpha * a * sigmoid * (1 - sigmoid)
        #         d = exp + d_avg

        #     #     print(d.min().item(), d.max().item())

        #     # plt.matshow(d[0, 0].data.cpu().numpy())
        #     # plt.savefig('qual_end.jpg')     
        #     # exit()

        #     mask_pred = (d + 1) / 2

        #     # d_old = mask_pred

        #     # for i in range(n_iters): 
        #     #     d_g = (gamma + delta) * d 
        #     #     d_g -= delta*F.conv2d(d, f, padding=1)
        #     #     d_g -= c * torch.exp(-(c * d).sum(dim=[0, 1, 2, 3], keepdim=True))
        #     #     # d_g -= c * torch.exp(-(c * d))
        #     #     # d_g -= alpha * a * torch.exp(-alpha * (a * d))
        #     #     # d_g -= beta * b * torch.exp(-beta * (b * d))
        #     #     d = d - lambd * d_g

        #     #     if torch.norm(d - d_old) < 0.01:
        #     #         break

        #     #     d_old = d
                
        #     # mask_pred = d

        # else:
        mask_pred = mask_pred.sigmoid()

        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.cpu().numpy()
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
