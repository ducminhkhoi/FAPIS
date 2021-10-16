import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist, show_result
from mmdet.core import results2json, coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core import tensor2imgs

import numpy as np
from mmcv.image import imread, imwrite


def single_gpu_test(model, data_loader, show=False, not_show_progress=False):
    model.eval()
    results = []
    img_ids = []
    cat_ids = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    save_data = {}

    folder = args.out.replace('results.pkl', 'results')
    os.makedirs(folder, exist_ok=True)

    for i, data in enumerate(data_loader):

        idx, img_id, category_id, rf_ann_id  = data['rf_ann_id']
        idx = idx.item()
        rf_ann_id = [x.item() for x in rf_ann_id]
        img_id = img_id.item()
        category_id = category_id.item()
        save_data[idx] = (img_id, category_id, rf_ann_id)
        label = dataset.cat2label[category_id]

        del data['rf_ann_id']

        with torch.no_grad():
            result, img_id, cat_id = model(return_loss=False, rescale=not show, **data)

        # print('\t', img_id, cat_id)
        # [1, 5, 9, 14, 18, 22, 27, 33, 37, 41, 46, 50, 54, 58, 62, 67, 74, 78, 82, 87]

        results.append(result)
        img_ids.append(img_id)
        cat_ids.append(cat_id)

        if show:
            rf_imgs = data['img_meta'][0].data[0][0]['rf_img']
            rf_masks = data['img_meta'][0].data[0][0]['rf_mask']

            for k, (rf_img, rf_mask) in enumerate(zip(rf_imgs, rf_masks)):
                rf_img = torch.from_numpy(rf_img)[None, ...]
                rf_img = tensor2imgs(rf_img, **dataset.img_norm_cfg)[0]
                rf_mask = rf_mask[0].astype(np.bool)
                # rf_img[rf_mask] = rf_img[rf_mask] * 0.5 + np.array([[255, 0, 0]], dtype=np.uint8) * 0.5
                imwrite(rf_img, f"{folder}/{idx:04d}_ref_{k}.jpg")

            out_file = f"{folder}/{idx:04d}.jpg"
            model.module.show_result(data, result, dataset.img_norm_cfg, out_file=out_file, score_thr=0.2)

        if not not_show_progress:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size):
                prog_bar.update()

    # print(len(save_data))
    # save_data_file = f'test_data/coco_{args.k_shot}_{args.part}_{args.number}.pkl'
    # torch.save(save_data, save_data_file)
    # exit()
    return results, img_ids, cat_ids


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--checkpoint2',
        help='output result file name without extension',
        default=None,
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument(
        '--generate_weight',
        action='store_true',
        help='whether to generate the weight for classification head')
    parser.add_argument(
        '--use_diff',
        action='store_true',
        help='whether to use diff between query bg and fg to generate the weight for classification head')
    parser.add_argument(
        '--use_l1',
        action='store_true',
        help='whether to use l1 instead of elementwise l1 for concatenating')
    parser.add_argument(
        '--use_rf_mask',
        action='store_true',
        help='whether to use ref mask instead of ref bbox')
    parser.add_argument(
        '--enlarge_ref',
        action='store_true',
        help='whether to enlarge the ref image for better capturing the  image content')
    parser.add_argument(
        '--use_bg',
        action='store_true',
        help='whether to use query bg to generate the weight for classification head')
    parser.add_argument(
        '--use_maskopt',
        action='store_true',
        help='whether to use mask optimization (2nd contribution) instead of standard mask prediction')
    parser.add_argument(
        '--use_boundary',
        type=float,
        default=0.,
        # action='store_true',
        help='whether to use boundary for mask prediction')
    parser.add_argument(
        '--use_prototype',
        action='store_true',
        help='whether to use prototype as in Yolact for mask prediction')
    parser.add_argument(
        '--not_show_progress',
        action='store_true',
        help='whether to use prototype as in Yolact for mask prediction')
    parser.add_argument(
        '--correlate_after',
        action='store_true',
        help='whether to correlate query feature map and support feature vector after rpn')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--number', type=int, default=0)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--num_protos', type=int, default=32)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    cfg.data.test.part = args.part 
    cfg.data.test.k_shot = args.k_shot
    cfg.data.test.number = args.number
    cfg.data.test.enlarge_ref = args.enlarge_ref
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.bbox_head.generate_weight = args.generate_weight #comment here for box_head generate
    # cfg.model.bbox_head.use_bg = args.use_bg
    cfg.model.bbox_head.use_boundary = args.use_boundary
    cfg.model.bbox_head.use_prototype= args.use_prototype
    cfg.model.bbox_head.num_protos = args.num_protos
    # cfg.model.bbox_head.generate_weight = args.generate_weight
    # cfg.model.bbox_head.use_diff = args.use_diff
    # cfg.model.use_l1 = args.use_l1
    cfg.model.k_shot = args.k_shot
    # cfg.model.correlate_after = args.correlate_after
    cfg.model.use_rf_mask = args.use_rf_mask
    # cfg.model.mask_head.use_maskopt = args.use_maskopt
    # cfg.model.mask_head.generate_weight = args.generate_weight

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # print('load checkpoint 1 completed')
    # if args.checkpoint2 is not None:
    #     checkpoint = load_checkpoint(model, args.checkpoint2, map_location='cpu')
    #     print('load checkpoint 2 completed')

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs, img_ids, cat_ids = single_gpu_test(model, data_loader, args.show, args.not_show_progress)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs, args.out, img_ids)
                    coco_eval(result_files, eval_types, dataset.coco, img_ids, cat_ids)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        coco_eval(result_files, eval_types, dataset.coco)

    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            results2json(dataset, outputs, args.json_out)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)


if __name__ == '__main__':
    args = parse_args()
    main()
