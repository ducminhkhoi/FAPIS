## mmcv Runner（mmdetecion训练过程）

#### Runner.run()

```python
def run(self, data_loaders, workflow, max_epochs, **kwargs):
    """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
    assert isinstance(data_loaders, list)
    assert mmcv.is_list_of(workflow, tuple)
    assert len(data_loaders) == len(workflow)

    self._max_epochs = max_epochs
    work_dir = self.work_dir if self.work_dir is not None else 'NONE'
    self.logger.info('Start running, host: %s, work_dir: %s',
                     get_host_info(), work_dir)
    self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
    self.call_hook('before_run')      #定义在/hooks/lr_updater.py中

    while self.epoch < max_epochs:
        for i, flow in enumerate(workflow):     # workflow中定义的模式，比如maskrcnn中的
            # [('train', 1)]，表示一直进行train的epoc，以下以maskrcnn中的config为例
            mode, epochs = flow              # mode = 'train', epochs = 1
            if isinstance(mode, str):  # self.train()
                if not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run an epoch'.
                        format(mode))
                    epoch_runner = getattr(self, mode)
                    elif callable(mode):  # custom train()
                        epoch_runner = mode       # epoch_runner对应上了Runner中名为train的函数
                        else:
                            raise TypeError('mode in workflow must be a str or '
                                            'callable function, not {}'.format(
                                                type(mode)))
                            for _ in range(epochs):
                                if mode == 'train' and self.epoch >= max_epochs:
                                    return
                                epoch_runner(data_loaders[i], **kwargs)    # 调用train函数

                                time.sleep(1)  # wait for some hooks like loggers to finish
                                self.call_hook('after_run')        # 调用after run的hook回调函数
```

#### Runner.train

​     iter代表使用一个batch训练，epoc代表训练完一次data_set

每个iter包括前向传播和反向传播过程，分别由batch_processor和after_train_iter这个hook（与optimizer有关）定义

```python
def train(self, data_loader, **kwargs):
    self.model.train()       #每一个epoc 调用一次，调用model的train函数
    self.mode = 'train'
    self.data_loader = data_loader      
    self._max_iters = self._max_epochs * len(data_loader)    #到这里，dataloader应该是已经存     了data了，len(data_loader)应该代表 totoal_data / batch_size，即data_batch的数量

    self.call_hook('before_train_epoch')      #定义在/hooks/lr_updater.py中，用于更新learning_rate
    for i, data_batch in enumerate(data_loader):   
        self._inner_iter = i
        self.call_hook('before_train_iter')    #定义在/hooks/lr_updater.py中，用于更新lr
        outputs = self.batch_processor(
            self.model, data_batch, train_mode=True, **kwargs)      # 讲数据传入batch_processor进行前向传播
            # 每个iter调用一次
        if not isinstance(outputs, dict):
            raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
                self.outputs = outputs
                self.call_hook('after_train_iter')   # 函数after_train_iter定义在/hooks/optimizer.py中，反向传播过程在这里进行
                self._iter += 1

                self.call_hook('after_train_epoch')  #函数after_train_epoc定义在/hooks/checkpoint.py中，用于存储每个epoc训练完成后的参数（权重）
                self._epoch += 1
```

#### after_train_iter

```python
def after_train_iter(self, runner):
    runner.optimizer.zero_grad()
    runner.outputs['loss'].backward()
    if self.grad_clip is not None:
        self.clip_grads(runner.model.parameters())
        runner.optimizer.step()
```

