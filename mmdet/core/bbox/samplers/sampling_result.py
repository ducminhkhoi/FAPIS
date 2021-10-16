import torch


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

        self.bboxes_ = bboxes
        self.gt_bboxes = gt_bboxes
        self.assign_result = assign_result
        self.gt_flags = gt_flags

    def update_pos_inds(self, pos_inds):
        self.pos_inds = pos_inds
        self.pos_bboxes = self.bboxes_[pos_inds]
        self.pos_is_gt = self.gt_flags[pos_inds]

        self.pos_assigned_gt_inds = self.assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_bboxes = self.gt_bboxes[self.pos_assigned_gt_inds, :]
        if self.assign_result.labels is not None:
            self.pos_gt_labels = self.assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])
