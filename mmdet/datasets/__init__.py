# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.ade20k import (ADE20KInstanceDataset, ADE20KPanopticDataset, ADE20KSegDataset)
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.datasets.base_semseg_dataset import BaseSegDataset
from mmdet.datasets.base_video_dataset import BaseVideoDataset
from mmdet.datasets.cityscapes import CityscapesDataset
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.coco_caption import CocoCaptionDataset
from mmdet.datasets.coco_panoptic import CocoPanopticDataset
from mmdet.datasets.coco_semantic import CocoSegDataset
from mmdet.datasets.crowdhuman import CrowdHumanDataset
from mmdet.datasets.dataset_wrappers import ConcatDataset, MultiImageMixDataset
from mmdet.datasets.deepfashion import DeepFashionDataset
from mmdet.datasets.dod import DODDataset
from mmdet.datasets.dsdl import DSDLDetDataset
from mmdet.datasets.flickr30k import Flickr30kDataset
from mmdet.datasets.isaid import iSAIDDataset
from mmdet.datasets.lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
from mmdet.datasets.mdetr_style_refcoco import MDETRStyleRefCocoDataset
from mmdet.datasets.mot_challenge_dataset import MOTChallengeDataset
from mmdet.datasets.objects365 import Objects365V1Dataset, Objects365V2Dataset
from mmdet.datasets.odvg import ODVGDataset
from mmdet.datasets.openimages import OpenImagesChallengeDataset, OpenImagesDataset
from mmdet.datasets.refcoco import RefCocoDataset
from mmdet.datasets.reid_dataset import ReIDDataset
from mmdet.datasets.samplers import (AspectRatioBatchSampler, ClassAwareSampler,
                       CustomSampleSizeSampler, GroupMultiSourceSampler,
                       MultiSourceSampler, TrackAspectRatioBatchSampler,
                       TrackImgSampler)
from mmdet.datasets.utils import get_loading_pipeline
from mmdet.datasets.v3det import V3DetDataset
from mmdet.datasets.voc import VOCDataset
from mmdet.datasets.wider_face import WIDERFaceDataset
from mmdet.datasets.xml_style import XMLDataset
from mmdet.datasets.youtube_vis_dataset import YouTubeVISDataset


__all__ = [
    'XMLDataset', 'CocoDataset', 'DeepFashionDataset', 'VOCDataset',
    'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset', 'LVISV1Dataset',
    'WIDERFaceDataset', 'get_loading_pipeline', 'CocoPanopticDataset',
    'MultiImageMixDataset', 'OpenImagesDataset', 'OpenImagesChallengeDataset',
    'AspectRatioBatchSampler', 'ClassAwareSampler', 'MultiSourceSampler',
    'GroupMultiSourceSampler', 'BaseDetDataset', 'CrowdHumanDataset',
    'Objects365V1Dataset', 'Objects365V2Dataset', 'DSDLDetDataset',
    'BaseVideoDataset', 'MOTChallengeDataset', 'TrackImgSampler',
    'ReIDDataset', 'YouTubeVISDataset', 'TrackAspectRatioBatchSampler',
    'ADE20KPanopticDataset', 'CocoCaptionDataset', 'RefCocoDataset',
    'BaseSegDataset', 'ADE20KSegDataset', 'CocoSegDataset',
    'ADE20KInstanceDataset', 'iSAIDDataset', 'V3DetDataset', 'ConcatDataset',
    'ODVGDataset', 'MDETRStyleRefCocoDataset', 'DODDataset',
    'CustomSampleSizeSampler', 'Flickr30kDataset'
]
