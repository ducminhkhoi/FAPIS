# One-shot Instance Segmentation



## Introduction
This is **pytorch** version of [Siamese Mask-RCNN](https://arxiv.org/abs/1811.11507) and we use [mmdetection](https://github.com/open-mmlab/mmdetection) toolbox to finish it.

The official code can be found in [siamese mask-rcnn](https://github.com/bethgelab/siamese-mask-rcnn)

We only support single-gpu training and single-gpu testing now.

The distributed training code may be updated recently

## Installation

Please follow the installation in **README_mmdetection.md** or the steps in [mmdetection](https://github.com/open-mmlab/mmdetection)

## Get Started

We only support single-gpu training and single-gpu testing

### Prepare COCO dataset

```shell
ln -s $path/to/coco data/coco
```

### single-gpu training

```shell
python tools/train.py configs/siamese_mask_rcnn.py
```

### single-gpu testing

```shell
python tools/test.py configs/siamese_mask_rcnn.py work_dirs/siamese_mask_rcnn_train/latest.pth --out results.pkl --eval bbox segm
```

### show the result

```shell
python tools/test.py configs/siamese_mask_rcnn.py work_dirs/siamese_mask_rcnn_train/latest.pth --show
```

## Citation

The official code can be found in [siamese mask-rcnn](https://github.com/bethgelab/siamese-mask-rcnn)

```
@article{michaelis_one-shot_2018,
    title = {One-Shot Instance Segmentation},
    author = {Michaelis, Claudio and Ustyuzhaninov, Ivan and Bethge, Matthias and Ecker, Alexander S.},
    year = {2018},
    journal = {arXiv},
    url = {http://arxiv.org/abs/1811.11507}
}
```

This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection) toolbox.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li,
             Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng,
             Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu,
             Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, Dahua Lin},
  journal = {arXiv preprint arXiv:1906.07155},
  year    = {2019}
}
```

Thanks for their contributions

## Collaborators

The collaborators of this project are [mi804](https://github.com/mi804) and [SteveLin0418](https://github.com/SteveLin0418) and [me](https://github.com/phj128)
