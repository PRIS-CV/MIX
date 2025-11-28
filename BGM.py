import random
import numpy as np
import torch
from typing import List, Tuple
from mmengine.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import HorizontalBoxes  # 确保导入


@TRANSFORMS.register_module()
class BGM(BaseTransform):
    """ 自定义 BGM（Background Mixup）数据增强方法，适配 COCO 格式

    该方法包含：
    1. **Self Patch Mixup (SPM)**：从背景区域提取 patch，随机移动并进行 mixup。
    2. **Color Patch Mixup (CPM)**：在图像上随机添加彩色 patch，并进行混合。

    Args:
        num_patches_range (Tuple[int, int]): Patch 数量范围 (min, max)。
        alpha_range (Tuple[float, float]): 透明度范围 (min, max)。
        patch_area_ratio_range (Tuple[float, float]): Patch 面积占图像的比例 (min, max)。
        apply_ratio_range (Tuple[float, float]): SPM 占比范围 (min, max)。
        apply_prob (float): 以 `apply_prob` 概率应用 BGM。
    """

    def __init__(self, num_patches_range: Tuple[int, int] = (1, 5),
                 alpha_range: Tuple[float, float] = (0.1, 0.3),
                 patch_area_ratio_range: Tuple[float, float] = (0.6, 0.8),
                 apply_ratio_range: Tuple[float, float] = (0.3, 0.7),
                 apply_prob: float = 0.4):
        self.num_patches_range = num_patches_range
        self.alpha_range = alpha_range
        self.patch_area_ratio_range = patch_area_ratio_range
        self.apply_ratio_range = apply_ratio_range
        self.apply_prob = apply_prob

    def transform(self, results: dict) -> dict:
        """ 执行 BGM 变换 """
        img = results['img']
        h, w, _ = img.shape
        gt_boxes = results.get('gt_bboxes', [])

        # 1️⃣ 以 `apply_prob` 概率决定是否应用 BGM
        if random.random() > self.apply_prob:
            return results  # 直接返回原图

        # **转换 COCO 格式的 gt_bboxes** `[x, y, w, h] → [x1, y1, x2, y2]`
        gt_boxes = self._convert_coco_bbox(gt_boxes)

        # 2️⃣ 随机选择 Patch 数量
        num_patches = random.randint(*self.num_patches_range)

        # 3️⃣ 随机选择 SPM 的占比
        apply_ratio = random.uniform(*self.apply_ratio_range)
        n_spm = round(num_patches * apply_ratio)  # Self Patch Mixup 数量
        n_cpm = num_patches - n_spm  # Color Patch Mixup 数量

        # 4️⃣ 计算 Patch 最大尺寸
        patch_area_ratio = random.uniform(*self.patch_area_ratio_range)
        max_patch_size = int(np.sqrt(patch_area_ratio * h * w))  # 计算 Patch 最大边长

        # 5️⃣ 执行 SPM
        for _ in range(n_spm):
            patch, patch_coords = self._select_background_patch(img, gt_boxes, h, w, max_patch_size)
            if patch is None:
                continue
            new_coords = self._random_move_patch(patch_coords, h, w)
            alpha = random.uniform(*self.alpha_range)
            img = self._apply_patch_mixup(img, patch, new_coords, alpha)

        # 6️⃣ 执行 CPM
        img = self._apply_color_patch_mixup(img, h, w, n_cpm, max_patch_size)

        results['img'] = img
        return results

    def _convert_coco_bbox(self, gt_boxes):
        """ 将 COCO 格式 `[x, y, w, h]` 转换为 `[x1, y1, x2, y2]` """
        # print(f"⚠️ gt_boxes: {gt_boxes}")  # 调试信息

        # 1️⃣ 检测 gt_boxes 是否为 `HorizontalBoxes` 类型
        if isinstance(gt_boxes, HorizontalBoxes):
            gt_boxes = gt_boxes.tensor.cpu().numpy().tolist()  # 转 NumPy 数组，再转列表

        # 2️⃣ 兼容 PyTorch Tensor 和 NumPy
        elif isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy().tolist()
        elif isinstance(gt_boxes, np.ndarray):
            gt_boxes = gt_boxes.tolist()

        # 3️⃣ 过滤无效的 box，并转换格式
        filtered_boxes = []
        for box in gt_boxes:
            if isinstance(box, (list, tuple)) and len(box) == 4:
                filtered_boxes.append([box[0], box[1], box[2], box[3]])  # 直接保持 `x1, y1, x2, y2` 格式
            else:
                print(f"⚠️ 跳过格式错误的 box: {box}")  # 过滤错误数据
        return filtered_boxes

    def _is_overlap_with_gt(self, patch_coords, gt_boxes):
        """ 检查 Patch 是否与 GT box 重叠 """
        x1_p, y1_p, x2_p, y2_p = patch_coords
        for box in gt_boxes:
            if len(box) != 4:
                continue  # 过滤格式不正确的 box
            x1_b, y1_b, x2_b, y2_b = box
            if max(x1_p, x1_b) < min(x2_p, x2_b) and max(y1_p, y1_b) < min(y2_p, y2_b):
                return True
        return False

    def _select_background_patch(self, img, gt_boxes, h, w, max_patch_size):
        """ 选取不与目标框重叠的背景 Patch """
        for _ in range(50):  # 限制尝试次数，防止死循环
            # 确保 patch_size 不超过 h 和 w
            patch_size = min(random.randint(30, max_patch_size), h, w)

            # 如果 patch_size 超出边界，跳过当前循环
            if w - patch_size <= 0 or h - patch_size <= 0:
                continue
            x1, y1 = random.randint(0, w - patch_size), random.randint(0, h - patch_size)
            x2, y2 = x1 + patch_size, y1 + patch_size

            if not self._is_overlap_with_gt((x1, y1, x2, y2), gt_boxes):
                return img[y1:y2, x1:x2], (x1, y1, x2, y2)
        return None, None

    def _random_move_patch(self, patch_coords, h, w):
        """ 使用随机偏移进行 Patch 移动 """
        x1, y1, x2, y2 = patch_coords
        patch_w, patch_h = x2 - x1, y2 - y1
        dx = random.randint(-patch_w // 2, patch_w // 2)
        dy = random.randint(-patch_h // 2, patch_h // 2)
        new_x1 = max(0, min(w - patch_w, x1 + dx))
        new_y1 = max(0, min(h - patch_h, y1 + dy))
        return new_x1, new_y1, new_x1 + patch_w, new_y1 + patch_h

    def _apply_patch_mixup(self, img, patch, new_coords, alpha):
        """ 应用 Mixup 透明度融合 """
        x1, y1, x2, y2 = new_coords
        img[y1:y2, x1:x2] = (alpha * patch + (1 - alpha) * img[y1:y2, x1:x2]).astype(np.uint8)
        return img

    def _apply_color_patch_mixup(self, img, h, w, num_patches, max_patch_size):
        """ 生成随机颜色块并进行混合 """
        for _ in range(num_patches):
            # 确保 patch_size 不超过 h 和 w
            patch_size = min(random.randint(30, max_patch_size), h, w)

            # 确保 x1, y1 的随机范围有效
            if w - patch_size <= 0 or h - patch_size <= 0:
                continue  # 跳过无效 patch

            x1, y1 = random.randint(0, w - patch_size), random.randint(0, h - patch_size)
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            alpha = random.uniform(*self.alpha_range)

            img[y1:y1 + patch_size, x1:x1 + patch_size] = (
                    (1 - alpha) * img[y1:y1 + patch_size, x1:x1 + patch_size] + alpha * color
            ).astype(np.uint8)
        return img

