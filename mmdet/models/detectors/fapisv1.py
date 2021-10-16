import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler

import matplotlib.pyplot as plt
from .label_propagate import propagate_to_edge, PathIndex
from .prototype import PrototypeNet

import numpy as np
import mmcv
import pycocotools.mask as mask_util
# torch.autograd.set_detect_anomaly(True)


@DETECTORS.register_module
class FAPISv2(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 mask_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 k_shot=1,
                 use_rf_mask=False):
        super(FAPISv2, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.num_protos = bbox_head.num_protos

        bbox_head.transform_ref = True #TODO change here

        self.bbox_head = builder.build_head(bbox_head)
        self.k_shot = k_shot
        self.use_rf_mask = use_rf_mask
        self.generate_weight = False
        self.use_boundary = bbox_head.use_boundary
        self.use_prototype = bbox_head.use_prototype

        self.mask_head = None
        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor

            if self.use_prototype:
                mask_head['num_classes'] = self.num_protos

            self.mask_head = builder.build_head(mask_head)
            self.bbox_assigner = build_assigner(train_cfg.rcnn.assigner)
            self.bbox_sampler = build_sampler(
                train_cfg.rcnn.sampler, context=self)

            if self.use_prototype:
                self.protonet = PrototypeNet(self.mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(FAPISv2, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

        if self.mask_head is not None:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def matching(self, If, Rf, Rf_mask):
        out = []
        ref_feats = []
        num_layers = len(If)
        batch_size = len(If[0])

        if self.use_rf_mask:
            for i in range(len(Rf)):
                Rf_mask_reshape = F.interpolate(Rf_mask.float(), Rf[i].shape[-2:])

                for Rf_, Rf_mask_ in zip(Rf[i], Rf_mask_reshape):
                    if Rf_mask_.sum() == 0: 
                        results = Rf_.mean(dim=[1,2])
                    else:
                        results = (Rf_ * Rf_mask_).sum(dim=[1,2]) / Rf_mask_.sum()
                    ref_feats.append(results)

        else:
            for i in range(len(Rf)):
                ref_feats.append(self.avg(Rf[i]))

        ref_feats = torch.stack(ref_feats).view(num_layers, batch_size, self.k_shot, -1, 1, 1)

        for i in range(num_layers):
            out.append(If[i] * ref_feats[i].mean(1))

        return out, ref_feats

    # extract_feat in siamese way
    def extract_feat(self, img, img_meta):
        If = self.backbone(img)
        if self.with_neck:
            If = self.neck(If)

        Rf = []
        Rf_mask = []

        if self.use_rf_mask:
            for i in range(len(img_meta)):
                for rf_img, rf_mask in zip(img_meta[i]['rf_img'], img_meta[i]['rf_mask']):
                    Rf.append(torch.from_numpy(rf_img).unsqueeze(0).to(img.device))
                    Rf_mask.append(torch.from_numpy(rf_mask).to(img.device))
            Rf_mask = torch.stack(Rf_mask)
        else:
            for i in range(len(img_meta)):
                for rf_img in img_meta[i]['rf_img']:
                    Rf.append(torch.from_numpy(rf_img).unsqueeze(0).to(img.device))

        Rf = torch.cat(Rf, dim=0)
        Rf = self.backbone(Rf)
        if self.with_neck:
            Rf = self.neck(Rf)

        If_new, ref_feats = self.matching(If, Rf, Rf_mask)
        return tuple(If), tuple(If_new), ref_feats 

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):

        x_ori, x, ref_feats = self.extract_feat(img, img_metas)

        x_used = x # x or x_ori

        ref_feats = [x_.mean(1) for x_ in ref_feats]

        outs = self.bbox_head(x_used, ref_feats)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if self.mask_head is not None or self.use_prototype:
            # sampling boxes for training
            proposal_cfg = self.train_cfg.get('rpn_proposal', None)
            proposal_inputs = outs + (img_metas, proposal_cfg)
            proposal_list = self.bbox_head.get_bboxes(*proposal_inputs)

            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            
            sampling_results = []
            sampling_protos = []

            for i in range(num_imgs):
                proposal = proposal_list[i][0]
                if len(proposal) == 0:
                    proposal = torch.cat([gt_bboxes[i], torch.ones_like(gt_bboxes[i][:, :1])], 1)

                assign_result = self.bbox_assigner.assign(proposal,
                                                    gt_bboxes[i],
                                                    gt_bboxes_ignore[i],
                                                    gt_labels[i])

                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal,
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                sampling_results.append(sampling_result)

            if self.use_prototype: # filter out unused images in sampling results, gt_masks and x
                used_images = [i for i in range(num_imgs) if sampling_results[i].pos_inds.max() < len(proposal_list[i][2])]

                sampling_protos = torch.cat([proposal_list[i][2][sampling_results[i].pos_inds] for i in used_images])

                if len(used_images) < num_imgs:
                    print('unused images:', set(range(num_imgs))-set(used_images))
                    sampling_results = [sampling_results[i] for i in used_images]
                    gt_masks = [gt_masks[i] for i in used_images]
                    x_used = [z[used_images] for z in x_used]

            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            num_pos = len(pos_rois)
            pos_indices = pos_rois[:, 0].long()

            pos_mask_feats, pos_return_lvls = self.mask_roi_extractor(
                        x_used[:self.mask_roi_extractor.num_inputs], pos_rois,
                        im_size=img.shape[-2:], return_lvls=True)

        if self.use_prototype:
            # sampling_protos = torch.cat([proposal_list[i][2][sampling_results[i].pos_inds] for i in range(num_imgs)])
            
            mask_pred = self.protonet(pos_mask_feats, sampling_protos)

            mask_targets = self.protonet.get_target(sampling_results, gt_masks, self.train_cfg.rcnn)

            loss_mask = self.protonet.loss(mask_pred, mask_targets) # change here

            losses.update(loss_mask)

        elif self.mask_head is not None:

            mask_pred = self.mask_head(pos_mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results, gt_masks, self.train_cfg.rcnn)

            loss_mask = self.mask_head.loss(mask_pred, mask_targets)

            losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, rescale=False, gt_masks=None):
        x_ori, x, ref_feats = self.extract_feat(img, img_meta)

        x_used = x # x or x_ori

        ref_feats = [x_.mean(1) for x_ in ref_feats]

        outs = self.bbox_head(x_used, ref_feats)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        
        # only have one image in test
        if self.use_prototype:
            det_bboxes, det_labels, det_protos = self.bbox_head.get_bboxes(*bbox_inputs)[0]
        else:
            det_bboxes, det_labels = self.bbox_head.get_bboxes(*bbox_inputs)[0]

        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        label = img_meta[0]['label']
        img_id = img_meta[0]['img_id']

        real_labels = torch.full_like(det_labels, label).long() - 1

        bbox_results = bbox2result(det_bboxes, real_labels, 81)

        segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]

        if det_bboxes.shape[0] > 0:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats, return_lvls = self.mask_roi_extractor(
                    x_used[:len(self.mask_roi_extractor.featmap_strides)], mask_rois, im_size=img.shape[-2:], return_lvls=True) 
            
            if self.use_prototype:
                mask_pred = self.protonet(mask_feats, det_protos)
                segm_result, _ = self.protonet.get_seg_masks(mask_pred, _bboxes, det_labels,
                                    real_labels, self.test_cfg.rcnn.mask_thr_binary, ori_shape, scale_factor, rescale)

            elif self.mask_head is not None:
                mask_pred = self.mask_head(mask_feats)

                segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                        det_labels,
                                                        real_labels,
                                                        self.test_cfg.rcnn,
                                                        ori_shape, scale_factor,
                                                        rescale)

        return (bbox_results, segm_result), img_id, label

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def to_segment_results(self, mask_pred, det_bboxes, det_labels, real_labels, mask_thr_binary,
                ori_shape, scale_factor, rescale, gt_masks=None):
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
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

            im_mask_all[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[real_label].append(rle)

        return cls_segms, im_mask_all

    
