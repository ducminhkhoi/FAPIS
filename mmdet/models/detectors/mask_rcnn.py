from .two_stage import TwoStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class MaskRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 use_l1=False,
                 k_shot=1,
                 use_rf_mask=False,
                 use_semantic=False,
                 correlate_after=False,
                 use_prototype=False):
        super(MaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            use_l1=use_l1,
            k_shot=k_shot,
            use_rf_mask=use_rf_mask,
            use_semantic=use_semantic,
            correlate_after=correlate_after)
