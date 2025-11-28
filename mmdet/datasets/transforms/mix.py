import cv2
import numpy as np
import random
from mmengine.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import torch

@TRANSFORMS.register_module()
class MIX(BaseTransform):
    """稳定版MIX增强：支持背景增强和目标增强，结合物理衰减与颜色扰动"""

    def __init__(self,

                 bgm_params: dict = {'m_patch_range': (1, 5), 'alpha_range': (0.1, 0.5), 'max_trials': 100},
                 layer_params: dict = {
                     'base_layers_config': {
                         "metal": {"attenuation_range": (0.4, 0.8), "saturation_scale": (1.1, 1.5),
                                   "brightness_scale": (0.8, 1.2)},
                         "organic": {"attenuation_range": (0.1, 0.3), "saturation_scale": (0.8, 1.2),
                                     "brightness_scale": (0.9, 1.1)},
                         "mix": {"attenuation_range": (0.1, 0.2), "saturation_scale": (0.9, 1.1),
                                        "brightness_scale": (0.9, 1.1)}
                     }
                 },
                 apply_prob={
                     'bg': 0.1,  # 背景补丁概率
                     'gt': 0.1  # GT补丁概率
                 },
                 # fixed_coords: list = None
                 ):


        self.m_patch_range = bgm_params['m_patch_range']
        self.bgm_alpha_range = bgm_params['alpha_range']
        self.max_trials = bgm_params['max_trials']

        self.base_layers = layer_params['base_layers_config']

        self.material_ranges = {
            "metal": {"lower": [90, 50, 50], "upper": [130, 255, 255]},
            "organic": {"lower": [0, 50, 50], "upper": [30, 255, 255]},
            "mix": {"lower": [35, 50, 50], "upper": [85, 255, 255]}
        }
        self.apply_prob = apply_prob if isinstance(apply_prob, dict) else {
            'bg': apply_prob,
            'gt': apply_prob
        }

    def _convert_gt_boxes(self, gt_boxes):
        if hasattr(gt_boxes, 'tensor'):
            return gt_boxes.tensor.cpu().numpy().tolist()
        elif isinstance(gt_boxes, torch.Tensor):
            return gt_boxes.cpu().tolist()
        elif isinstance(gt_boxes, np.ndarray):
            return gt_boxes.tolist()
        elif isinstance(gt_boxes, list):
            return gt_boxes
        else:
            return []

    def transform(self, results: dict) -> dict:
        img = results['img']
        gt_boxes_raw = results.get('gt_bboxes', [])
        gt_boxes = self._convert_gt_boxes(gt_boxes_raw)

        final_img = img.copy().astype(np.float32)

        # ===== GT补丁增强 =====
        if random.random() <= self.apply_prob.get('gt', 0.0):
            gt_patches = self._extract_gt_patches(img, gt_boxes)
            if gt_patches:
                final_img = self._apply_gt_patches(final_img, gt_patches)
                results["gt_patches"] = gt_patches

        # ===== 背景补丁增强 =====
        if random.random() <= self.apply_prob.get('bg', 0.0):
            bg_patches = self._extract_bgm_patches(img, gt_boxes)
            if bg_patches:
                final_img = self._apply_all_patches_mixup(final_img, bg_patches)
                results["bg_patches"] = bg_patches


        # ===== 更新结果 =====
        results['img'] = np.clip(final_img, 0, 255).astype(np.uint8)
        return results


    def _extract_bgm_patches(self, img: np.ndarray, gt_boxes: list) -> list:
        patches = []
        m_patch = random.randint(*self.m_patch_range)
        for _ in range(m_patch):
            patch, coords = self._select_patch(img, gt_boxes)
            if patch is not None:
                alpha = random.uniform(*self.bgm_alpha_range)
                patches.append({'data': patch, 'coords': coords, 'alpha': alpha, "type": "bg"})
        return patches

    def _select_patch(self, img: np.ndarray, gt_boxes: list) -> tuple:
        h, w = img.shape[:2]
        for _ in range(self.max_trials):
            size = random.randint(100, 200)
            x1 = random.randint(0, w - size)
            y1 = random.randint(0, h - size)
            x2, y2 = x1 + size, y1 + size
            if self._is_overlap((x1, y1, x2, y2), gt_boxes):
                    continue

            gray_patch = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if np.mean(gray_patch) > 240 or np.var(gray_patch) < 10:
                continue

            return img[y1:y2, x1:x2], (x1, y1, x2, y2)
        return None, None

    def _is_overlap(self, patch_coords, gt_boxes):
        x1_p, y1_p, x2_p, y2_p = patch_coords
        for box in gt_boxes:
            if len(box) != 4: continue
            x1_b, y1_b, x2_b, y2_b = box
            if max(x1_p, x1_b) < min(x2_p, x2_b) and max(y1_p, y1_b) < min(y2_p, y2_b):
                return True
        return False

    # ========== GT补丁提取 ==========
    def _extract_gt_patches(self, img: np.ndarray, gt_boxes: list) -> list:
        patches = []
        for box in gt_boxes:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = map(int, box)
            patch = img[y1:y2, x1:x2]
            if patch.size > 0:
                patches.append({'data': patch, 'coords': (x1, y1, x2, y2), "type": "gt"})
        return patches

    def _adjust_patch_layers(self, patch: np.ndarray, return_original: bool = False) -> np.ndarray:
        """
        分层增强（统一模式版）：
        - 每个 patch 只随机选择一次模式：
            'attenuation'：厚度层物理衰减增强
            'color'：HSV颜色扰动
        - 所有材料统一执行同一模式
        """
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
        masks = self._generate_masks(hsv)
        adjusted = patch.astype(np.float32)
        I0 = 255.0
        mode = random.choice(['attenuation', 'color'])
        if mode == 'attenuation':
            # 厚度层增强
            patch_float = patch.astype(np.float32)
            for material, params in self.base_layers.items():
                mask = masks[material][:, :, np.newaxis]
                mu = random.uniform(*params['attenuation_range'])

                normalized = np.clip(patch_float / I0, 1e-4, 1.0)
                thickness = -np.log(normalized) / mu

                scale = random.uniform(0.9, 1.1)
                noise = np.random.normal(0, 0.02, thickness.shape)
                thickness_aug = np.clip(thickness * scale + noise, 0, 5.0)

                patch_processed = I0 * np.exp(-mu * thickness_aug)
                mask_3c = np.repeat(mask, 3, axis=2)
                adjusted = adjusted * (1 - mask_3c) + patch_processed * mask_3c

        else:
            for material, params in self.base_layers.items():
                mask = masks[material][:, :, np.newaxis]
                patch_hsv = hsv.copy()  # ✅ 每个材质独立调整
                sat = random.uniform(*params['saturation_scale'])
                bri = random.uniform(*params['brightness_scale'])
                patch_hsv[:, :, 1] = np.clip(patch_hsv[:, :, 1] * sat, 0, 255)
                patch_hsv[:, :, 2] = np.clip(patch_hsv[:, :, 2] * bri, 0, 255)
                patch_bgr = cv2.cvtColor(patch_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                adjusted = adjusted * (1 - mask) + patch_bgr.astype(np.float32) * mask
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _adjust_gt_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        对 GT patch 进行分层增强（仅金属材料）
        """
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
        masks = self._generate_masks(hsv)
        adjusted = patch.astype(np.float32)
        material = "metal"
        if material in self.base_layers:
            mask = masks[material][:, :, np.newaxis]
            patch_hsv = hsv.copy()
            params = self.base_layers[material]

            sat = random.uniform(*params['saturation_scale'])
            bri = random.uniform(*params['brightness_scale'])
            patch_hsv[:, :, 1] = np.clip(patch_hsv[:, :, 1] * sat, 0, 255)
            patch_hsv[:, :, 2] = np.clip(patch_hsv[:, :, 2] * bri, 0, 255)
            patch_bgr = cv2.cvtColor(patch_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            mask_3c = np.repeat(mask, 3, axis=2)
            adjusted = adjusted * (1 - mask_3c) + patch_bgr.astype(np.float32) * mask_3c

        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _generate_masks(self, hsv_img: np.ndarray) -> dict:
        masks = {}
        lower = np.array(self.material_ranges["metal"]["lower"])
        upper = np.array(self.material_ranges["metal"]["upper"])
        masks["metal"] = cv2.inRange(hsv_img, lower, upper) / 255.0
        lower = np.array(self.material_ranges["organic"]["lower"])
        upper = np.array(self.material_ranges["organic"]["upper"])
        masks["organic"] = cv2.inRange(hsv_img, lower, upper) / 255.0
        lower = np.array(self.material_ranges["mix"]["lower"])
        upper = np.array(self.material_ranges["mix"]["upper"])
        masks["mix"] = cv2.inRange(hsv_img, lower, upper) / 255.0
        return masks

    def _apply_all_patches_mixup(self, img: np.ndarray, patches: list) -> np.ndarray:
        h, w = img.shape[:2]
        out_img = img.astype(np.float32)

        for patch in patches:
            patch_data = patch["data"].astype(np.uint8)  # 原始裁剪区域
            patch["orig_data"] = patch_data.copy()
            ph, pw = patch_data.shape[:2]

            if ph >= h or pw >= w:
                continue

            enhanced_patch = self._adjust_patch_layers(patch_data)
            patch["enhanced_data"] = enhanced_patch  # 保存增强后的 patch
            patch["orig_data"] = patch_data.copy()  # 保存原始 patch

            for _ in range(self.max_trials):
                x = random.randint(0, w - pw)
                y = random.randint(0, h - ph)
                roi = out_img[y:y + ph, x:x + pw]
                roi_gray = cv2.cvtColor(out_img[y:y + ph, x:x + pw].astype(np.uint8), cv2.COLOR_BGR2GRAY)
                if np.mean(roi_gray) < 240:  # ROI不是空白
                    break

            alpha = patch.get("alpha", 0.5)

            if patch["type"] == "bg":
                blended = alpha * enhanced_patch + (1 - alpha) * roi
                out_img[y:y + ph, x:x + pw] = blended
            elif patch["type"] == "cl":
                Ck = np.full((ph, pw, 3), patch_data[0, 0, :], dtype=np.float32)
                blended = (1 - alpha) * roi + alpha * Ck
                out_img[y:y + ph, x:x + pw] = blended
            else:
                blended = alpha * enhanced_patch + (1 - alpha) * roi
                out_img[y:y + ph, x:x + pw] = blended

            patch["coords"] = (x, y, x + pw, y + ph)

        return np.clip(out_img, 0, 255).astype(np.uint8)

    def _apply_gt_patches(self, base_img: np.ndarray, patches: list) -> np.ndarray:
        base_img = base_img.astype(np.float32)
        for patch_info in patches:
            patch = patch_info['data'].astype(np.float32)
            x1, y1, x2, y2 = patch_info['coords']
            ph, pw = patch.shape[:2]
            if (y2 - y1) != ph or (x2 - x1) != pw:
                continue
            lambda_val = random.uniform(0.1, 0.3)
            base_img[y1:y2, x1:x2] = lambda_val * patch + (1 - lambda_val) * base_img[y1:y2, x1:x2]
        return np.clip(base_img, 0, 255).astype(np.uint8)
