from __future__ import division

import argparse
import os
from mmcv import Config

from mmdet import __version__
from mmdet.datasets import build_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--load_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--generate_weight',
        action='store_true',
        help='whether to generate the weight for classification head')
    parser.add_argument(
        '--use_bg',
        action='store_true',
        help='whether to use query bg to generate the weight for classification head')
    parser.add_argument(
        '--use_diff',
        action='store_true',
        help='whether to use diff between query bg and fg to generate the weight for classification head')
    parser.add_argument(
        '--use_l1',
        action='store_true',
        help='whether to use l1 instead of elementwise l1 for concatenating')
    parser.add_argument(
        '--enlarge_ref',
        action='store_true',
        help='whether to enlarge the ref image for better capturing the  image content')
    parser.add_argument(
        '--use_rf_mask',
        action='store_true',
        help='whether to use ref mask instead of ref bbox')
    parser.add_argument(
        '--use_maskopt',
        action='store_true',
        help='whether to use mask optimization (2nd contribution) instead of standard mask prediction')
    parser.add_argument(
        '--use_combine',
        action='store_true',
        help='whether to use ref_img feature combined with current feature to predict')
    parser.add_argument(
        '--shape_aware',
        action='store_true',
        help='whether to use shape aware for fcos loss')
    parser.add_argument(
        '--use_boundary',
        action='store_true',
        help='whether to use boundary for mask prediction')
    parser.add_argument(
        '--use_prototype',
        action='store_true',
        help='whether to use prototype as in Yolact for mask prediction')
    parser.add_argument(
        '--correlate_after',
        action='store_true',
        help='whether to correlate query feature map and support feature vector after rpn')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--num_protos', type=int, default=32)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.load_from is not None:
        cfg.load_from = args.load_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # cfg.model.bbox_head.generate_weight = args.generate_weight
    # cfg.model.bbox_head.use_bg = args.use_bg
    cfg.model.bbox_head.use_boundary = args.use_boundary
    cfg.model.bbox_head.use_prototype = args.use_prototype
    cfg.model.bbox_head.num_protos = args.num_protos
    # cfg.model.bbox_head.use_combine = args.use_combine
    # cfg.model.bbox_head.shape_aware = args.shape_aware
    # cfg.model.bbox_head.use_diff = args.use_diff
    # cfg.model.use_l1 = args.use_l1
    cfg.model.k_shot = args.k_shot
    cfg.model.use_rf_mask = args.use_rf_mask
    # cfg.model.correlate_after = args.correlate_after
    # cfg.model.mask_head.use_maskopt = args.use_maskopt
    # cfg.model.mask_head.generate_weight = args.generate_weight
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    cfg.data.train.part = args.part
    cfg.data.train.enlarge_ref = args.enlarge_ref
    train_dataset = build_dataset(cfg.data.train)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
