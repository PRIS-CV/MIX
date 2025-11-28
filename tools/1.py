from mmdet.datasets import CocoDataset
from mmengine import Config

cfg = Config.fromfile('/home/liuxintong/BGM/mmdetection/configs/atss/atss_r50_fpn_1x_coco.py')

# 构建训练集
train_dataset_cfg = cfg.train_dataloader.dataset
train_dataset = CocoDataset(
    ann_file=train_dataset_cfg.ann_file,
    pipeline=train_dataset_cfg.pipeline,
    data_root=train_dataset_cfg.data_root,
    data_prefix=train_dataset_cfg.data_prefix,
    # filter_empty_gt=train_dataset_cfg.get('filter_cfg', {}).get('filter_empty_gt', True)
)

# 构建验证集
val_dataset_cfg = cfg.val_dataloader.dataset
val_dataset = CocoDataset(
    ann_file=val_dataset_cfg.ann_file,
    pipeline=val_dataset_cfg.pipeline,
    data_root=val_dataset_cfg.data_root,
    data_prefix=val_dataset_cfg.data_prefix,
    test_mode=True
)

print(f"Train dataset length: {len(train_dataset)}")
print(f"Val dataset length: {len(val_dataset)}")
