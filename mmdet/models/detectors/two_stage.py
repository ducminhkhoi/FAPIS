import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
import matplotlib.pyplot as plt
from .semantic_seg import SemSegFPNHead, focal_loss


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 k_shot=1,
                 use_l1=False, 
                 use_rf_mask=False,
                 use_semantic=False,
                 correlate_after=False):
        super(TwoStageDetector, self).__init__()

        self.conv = nn.Conv2d(512, 256, 1)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.k_shot = k_shot
        self.use_l1 = use_l1
        self.use_rf_mask = use_rf_mask
        self.use_semantic = use_semantic
        self.correlate_after = correlate_after
        
        if self.use_semantic:
            # self.sem_seg_head = SemSegFPNHead(loss_weight=0.1)
            self.sem_seg_head = nn.Sequential(nn.Conv2d(257, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(256, 2, kernel_size=1, stride=1))

            for m in self.sem_seg_head.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def matching_old(self, If, Rf):
        out = []
        for i in range(len(Rf)):
            rf_avg = self.avg(Rf[i])
            delta = If[i] - rf_avg
            concat = torch.cat((If[i], delta.abs()), dim=1)
            out.append(self.conv(concat))
        return out

    def matching(self, If, Rf, Rf_mask, aug_rf=None):
        out = []
        ref_feats = []
        num_layers = len(If)
        batch_size = len(If[0])
        n_features = Rf[0].shape[1]

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

        ref_feats = torch.stack(ref_feats).view(num_layers, batch_size, self.k_shot, -1)

        if aug_rf is not None: 
            ref_feats = torch.cat([ref_feats, aug_rf], 2)

        ref_feats = ref_feats.view(num_layers, batch_size, -1, n_features, 1, 1)
        list_deltas = []

        for i in range(num_layers):
            delta = (If[i] - ref_feats[i].mean(1)).abs()
            list_deltas.append(delta)

            if self.correlate_after:
                out.append(If[i] * ref_feats[i].mean(1))
            else:
                out.append(self.conv(torch.cat((If[i], delta), dim=1)))

        return out, ref_feats, list_deltas

    # extract_feat in siamese way
    def extract_feat(self, img, img_meta, augment_ref=False):
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

        Rf_img = torch.cat(Rf, dim=0)
        Rf = self.backbone(Rf_img)
        if self.with_neck:
            Rf = self.neck(Rf)

        # augment rf images by randomly placing rf instance in query instance
        if augment_ref: # in testing only, only have one input
            aug_rf = self.augment_ref(Rf_img, Rf_mask, img)
        else:
            aug_rf = None

        If_new, ref_feats, list_deltas = self.matching(If, Rf, Rf_mask, aug_rf)
        return tuple(If), tuple(If_new), ref_feats, list_deltas

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x_ori, x, ref_feats, deltas = self.extract_feat(img, img_meta)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            if self.correlate_after:
                rpn_outs = self.rpn_head(x_ori) # change here for correlate after
            else:
                rpn_outs = self.rpn_head(x) # change here for correlate after

            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposal

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats, return_lvls = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois, return_lvls=True)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            # TODO: Change here
            indices = rois[:, 0].long()

            num_layers, batch_size, k_shot, C = ref_feats.shape[:4] # comment this line
            ref_feats = ref_feats[..., 0, 0]
            query_feats = torch.stack([x_.mean(dim=[-1, -2]) for x_ in x]) # comment this line

            query_feats = query_feats[return_lvls, indices] # comment this line
            ref_feats_ = ref_feats[return_lvls, indices] # comment this line

            cls_score, bbox_pred = self.bbox_head(bbox_feats, ref_feats_, query_feats) # correct, new

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)

            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])

            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)

            losses.update(loss_mask)

            if self.use_semantic:
                # semantic seg
                B, C, H, W = img.shape
                masks = torch.zeros(B, H, W).to(img.device)

                with torch.no_grad():
                    for i in range(B):
                        mask_i = torch.from_numpy(gt_masks[i]).max(0)[0].to(img.device)[None, None, ...].float()
                        mask_i = F.interpolate(mask_i, (H, W))[0, 0]
                        masks[i] = mask_i

                # _, loss_sem = self.sem_seg_head(x, masks.long())

                sim = F.relu(F.cosine_similarity(x_ori[-1], ref_feats[-1].mean(1)[..., None, None]))
                s = F.interpolate(sim.unsqueeze(1), x_ori[0].shape[-2:], mode='bilinear', align_corners=True)
                s_max = F.adaptive_max_pool2d(s, 1)
                s = s / (s_max + 1e-16)

                output = self.sem_seg_head(torch.cat([x_ori[0], s], 1))
                output = F.interpolate(output, masks.shape[-2:], mode='bilinear', align_corners=True)

                # loss_sem = {'loss_sem_seg': F.cross_entropy(output, masks.long())}
                loss_sem = {'loss_sem_seg': focal_loss(output, masks.long())}

                losses.update(loss_sem)

        return losses

    def augment_ref(self, Rf_img, Rf_mask, img):
        flips = [True, False, True, False]
        scales = [1/8, 1/8, 1/4, 1/2] # relative to img
        sizes = [3, 2, 2, 1]

        Rf_H, Rf_W = Rf_img.shape[-2:]
        H, W = img.shape[-2:]

        list_imgs = []
        list_masks = []

        for size, scale, flip in zip(sizes, scales, flips):
            real_size = int(H * scale), int(W * scale)
            Rf_img_scale = F.interpolate(Rf_img, size=real_size, mode='bilinear', align_corners=True)
            Rf_mask_scale = F.interpolate(Rf_mask.float(), size=real_size)

            if flip:
                Rf_img_scale = Rf_img_scale.flip(3)
                Rf_mask_scale = Rf_mask_scale.flip(3)

            Rf_mask_scale_ = Rf_mask_scale.expand_as(Rf_img_scale)

            real_rf_h, real_rf_w = Rf_img_scale.shape[-2:]

            real_H, real_W = H - 2 * real_rf_h, W - 2 * real_rf_w
            new_img = img.data.clone()
            new_mask = torch.zeros(1, 1, H, W).to(img.device)

            for h in range(size):
                for w in range(size):
                    if size == 1:
                        real_h, real_w = (H - real_rf_h) // 2, (W - real_rf_w) // 2
                    else:
                        real_h = int(real_rf_h // 2 + (real_H // (size-1)) * h)
                        real_w = int(real_rf_w // 2 + (real_W // (size-1)) * w)

                    new_img[:, :, real_h:real_h+real_rf_h, real_w:real_w+real_rf_w] = torch.where(Rf_mask_scale_ > 0,
                            Rf_img_scale, new_img[:, :, real_h:real_h+real_rf_h, real_w:real_w+real_rf_w])

                    new_mask[:, :, real_h:real_h+real_rf_h, real_w:real_w+real_rf_w] = torch.where(Rf_mask_scale > 0,
                            torch.ones_like(Rf_mask_scale), torch.zeros_like(Rf_mask_scale))
                
            list_imgs.append(new_img)
            list_masks.append(new_mask)

        aug_rf_imgs = torch.cat(list_imgs)
        aug_rf_masks = torch.cat(list_masks)

        aug_rf_imgs = self.backbone(aug_rf_imgs)
        if self.with_neck:
            aug_rf = self.neck(aug_rf_imgs)

        ref_feats = []
        num_layers = len(aug_rf)
        batch_size = img.shape[0]

        for i in range(num_layers):
            Rf_mask_reshape = F.interpolate(aug_rf_masks.float(), aug_rf[i].shape[-2:])

            for Rf_, Rf_mask_ in zip(aug_rf[i], Rf_mask_reshape):
                if Rf_mask_.sum() == 0: 
                    results = Rf_.mean(dim=[1,2])
                else:
                    results = (Rf_ * Rf_mask_).sum(dim=[1,2]) / Rf_mask_.sum()
                ref_feats.append(results)

        ref_feats = torch.stack(ref_feats).view(num_layers, batch_size, len(scales), -1)

        return ref_feats

    def simple_test(self, img, img_meta, proposals=None, rescale=False, gt_masks=None):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x_ori, x, ref_feats, deltas = self.extract_feat(img, img_meta, augment_ref=False)

        if self.correlate_after:
            proposal_list = self.simple_test_rpn(
                                    x_ori, img_meta, self.test_cfg.rpn) if proposals is None else proposals     
        else:
            proposal_list = self.simple_test_rpn(
                                    x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, x_ori, img_meta, proposal_list, ref_feats, deltas, self.test_cfg.rcnn, rescale=rescale)

        det_labels, real_labels = det_labels
        # import ipdb;ipdb.set_trace()
        # bbox_results = bbox2result(det_bboxes, real_labels,
        #                            self.bbox_head.num_classes)
        bbox_results = bbox2result(det_bboxes, real_labels, 81)

        sem_seg_result = None

        if not self.with_mask:
            return bbox_results
        else:
            segm_results, mask_pred = self.simple_test_mask(
                x, x_ori, ref_feats, img_meta, det_bboxes, det_labels, real_labels, gt_masks, rescale=rescale)
            img_id = img_meta[0]['img_id']
            label = img_meta[0]['label']

            if self.use_semantic and mask_pred is not None:
                mask_rois = bbox2roi([det_bboxes])

                sim = F.relu(F.cosine_similarity(x_ori[-1], ref_feats[-1].mean(1)))
                s = F.interpolate(sim.unsqueeze(1), x_ori[0].shape[-2:], mode='bilinear', align_corners=True)
                s_max = F.adaptive_max_pool2d(s, 1)
                s = s / (s_max + 1e-16)
                output = self.sem_seg_head(torch.cat([x_ori[0], s], 1))

                output_ = F.interpolate(output, img[0].shape[-2:], mode='bilinear', align_corners=True)
                sem_seg_result = output_.argmax(1)

                semantic_roi = self.mask_roi_extractor([output], mask_rois) 
                semantic_roi = F.interpolate(semantic_roi, mask_pred.shape[-2:], mode='bilinear', align_corners=True)
                semantic_roi = semantic_roi.argmax(1, keepdim=True)

                mask_pred = torch.from_numpy(mask_pred).to(semantic_roi.device)
                N = mask_pred.size(0)
                semantic_roi = (semantic_roi > 0.5).view(N, -1)
                mask_pred = (mask_pred > 0.5).view(N, -1)
                mask_score = ((semantic_roi & mask_pred).float().sum(1) / (semantic_roi | mask_pred).float().sum(1)).tolist()

                for bbox, score in zip(bbox_results[real_labels[0]], mask_score):
                    bbox[-1] = bbox[-1] + score

            return (bbox_results, segm_results), img_id, label #, sem_seg_result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
