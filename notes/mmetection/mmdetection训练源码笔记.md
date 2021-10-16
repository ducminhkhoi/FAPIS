# mmdetection训练流程

### 单GPU训练的方式：

```
python tools/train.py ${CONFIG_FILE}
```

以上未指定work_dir。例子：

```shell
python ./tools/train.py configs/mask_rcnn_r50_fpn_1x.py
```

即调用train.py文件， 训练的参数文件为CONFIG_FILE

### 训练流程

以下只列出了一些重要,流程1.2.3.都在tools/train.py

以下的build都使用了register

##### 1. build_detector

```python
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # 反回了一个类，类的具体类型是由传入cfg.model决定的，
    # cfg.model是参数文件中的一个字典，里面的type决定了类型：	MaskRcnn。该类定义在mmdet/models/detectors/mask_rcnn.py中
```

##### 2. build_dataset

```python
    train_dataset = build_dataset(cfg.data.train) # 构造dataset类。cfg中的dataset_type = 'CocoDataset'决定了他是一个CocoDataset类，该类定义在mmdet/datasets/coco.py中，该类继承了CustomDataset类，定义在custom.py中
    model.CLASSES = train_dataset.CLASSES # 将dataset中的类别复制到model中
```

##### 3. train_detector

```python
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)
```

train_detector 定义在mmdet/apis/train.py中，他根据参数，定义为dist_train)多gpu_还是非dist_train(单gpu)

```python
# train_detector
if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)

```

以下以_dist_train为例，讲解流程

##### 3.1 dataloader

```python
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    # dataloader是一个列表,他存了各种模式下的dataloader，如train，val
```

##### 3.2 put model on gpu

```pyhon
    model = MMDistributedDataParallel(model.cuda())
```

##### 3.3 build runner

```python
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)# 此函数定义在同一个文件中，根据参数构造optimizer
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)# 此函数为mmcv中的一个类，batch_processor为train.py中的另一个函数，用于每一个batch的数据传入model中进行forward_train，并计算loss，进行返回，用于反向传播
```

##### 3.4 run

```python
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    # 本函数进行主要的训练过程，更多关于mmcv_runner的函数mmcv_Runner.md
    
    # cfg.workflow和config文件中的 workflow = [('train', 1)]对应，表示epoc的进行方式
    # 如果为workflow = [('train', 2),('val',1)]，表示两个epoc进行train，一个epoc进行val，交替进行。
```