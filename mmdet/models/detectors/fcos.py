from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class FCOS(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_head,
                 mask_roi_extractor,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 k_shot=1,
                 use_rf_mask=False,
                 correlate_after=False,
                 ):
        super(FCOS, self).__init__(backbone, neck, bbox_head, 
                                    mask_head, mask_roi_extractor,
                                    train_cfg,
                                   test_cfg, pretrained,
                                   k_shot, use_rf_mask)
