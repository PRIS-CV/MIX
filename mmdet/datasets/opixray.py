from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class OPIXrayDataset(CocoDataset):

    CLASSES = ("Straight_Knife","Folding_Knife",  "Scissor", "Utility_Knife", "Multi-tool_Knife")

