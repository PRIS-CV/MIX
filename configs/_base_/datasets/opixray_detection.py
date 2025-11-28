_base_ = '../default_runtime.py'

# 训练的轮数
epoch =12

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',interval=1, max_keep_ckpts=3),
    visualization=dict(
        draw=True, test_out_dir='testresult', type='DetVisualizationHook')
   )

# 设置tensorboard后端可以用tensorboard查看
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 修改数据集相关配置
dataset_type = 'OPIXrayDataset'
data_root = 'D:/BGM复现/data/'  # 改成你的数据集文件根目录
metainfo = {
    'classes': ("Folding_Knife", "Straight_Knife", "Scissor", "Utility_Knife", "Multi-tool_Knife" ), #你的类别,我只有一类
    'palette': [
        (220, 20, 60), #调色盘
    ]
}

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    sampler=dict(shuffle=True, type='DefaultSampler'),
    pin_memory=True,
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train_annotations.json', # 你需要修改的训练集地址
        data_prefix=dict(img='train_image'), # 数据的前缀，即data_root后面的内容，你的图片的位置
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args)
        )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
         metainfo=metainfo,
        ann_file='annotations/test_annotations.json', #此处是测试集的json文件位置
        data_prefix=dict(img='test_image'), # 图片的位置
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# 我没有验证集，所以将验证集设置成和测试集相同
val_dataloader = test_dataloader

# 修改评价指标相关配置
test_evaluator =dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test_annotations.json',# 修改为你测试集的标注文件位置
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
val_evaluator = test_evaluator


# 训练循环的配置，每一个epoch都验证一次
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# 优化器配置，lr为学习率，如果损失值变为nan，可以调小学习率试一下
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')

# 多尺度学习率，可以自行修改删去
param_scheduler = [
    # 热身，训练学习率逐步增长
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    # 学习率调控 ,33,43,50 轮后学习率均变为原来的0.3倍
    dict(
        begin=0,
        by_epoch=True,
        end=epoch,
        gamma=0.3,
        milestones=[
            33,
            43,
            50,
        ],
        type='MultiStepLR'),
]

auto_scale_lr = dict(base_batch_size=2, enable=False)
log_processor = dict(by_epoch=True)
