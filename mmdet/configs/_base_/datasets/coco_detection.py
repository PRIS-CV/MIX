# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import LoadImageFromFile
from mmengine.dataset.sampler import DefaultSampler

from mmdet.datasets import AspectRatioBatchSampler, CocoDataset
from mmdet.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                       RandomFlip, Resize)
from mmdet.evaluation import CocoMetric

# dataset settings
dataset_type = CocoDataset
data_root = '/home/liuxintong/LMUData/opixray/'

backend_args = None

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type='MIX',
        bgm_params=dict(
            m_patch_range=(1, 5),
            alpha_range=(0.1, 0.5),
            max_trials=100
        ),
        layer_params=dict(
            base_layers_config=dict(
                metal=dict(
                    attenuation_range=(0.4, 0.8),
                    saturation_scale=(1.1, 1.5),
                    brightness_scale=(0.8, 1.2)
                ),
                organic=dict(
                    attenuation_range=(0.1, 0.3),
                    saturation_scale=(0.8, 1.2),
                    brightness_scale=(0.9, 1.1)
                ),
                mix=dict(
                    attenuation_range=(0.1, 0.2),
                    saturation_scale=(0.9, 1.1),
                    brightness_scale=(0.9, 1.1)
                )
            )
        ),
        apply_prob=dict(
            bg=0.1,   # 背景补丁概率
            gt=0.1    # GT补丁概率
        )
    ),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs),
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train_annotation.json',
        data_prefix=dict(img='train_image/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test_annotation.json',
        data_prefix=dict(img='test_image/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'annotations/test_annotation.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


