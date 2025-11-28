import cv2
import numpy as np
import mmcv
import random
from BGM import BGM  # 确保 BGM.py 在当前目录或 Python path 中
from mmengine.registry import TRANSFORMS

# 读取测试图片
image_path = 'D:/BGM复现/data/train_image/009002.jpg'  # 替换成你的测试图片路径
img = mmcv.imread(image_path)

# 伪造 COCO 格式的 bbox（假设物体在图片中心）
gt_bboxes = [[100, 100, 200, 200], [300, 300, 400, 400]]  # (x, y, w, h) 格式

# 构造数据字典
data = {'img': img, 'gt_bboxes': gt_bboxes}

# ✅ 创建 BGM 变换（参数可调）
bgm_transform = BGM(
    num_patches_range=(2, 5),   # 生成 2~5 个 Patch
    alpha_range=(0.2, 0.4),      # 透明度 0.2~0.4
    patch_area_ratio_range=(0.4, 0.6),  # Patch 面积占比 40%~60%
    apply_ratio_range=(0.5, 0.7),  # SPM 比例 50%~70%
    apply_prob=1.0  # 设为 1.0，保证每次都应用 BGM
)

# 运行 BGM 变换
transformed_data = bgm_transform.transform(data)

# 获取增强后的图片
augmented_img = transformed_data['img']

# ✅ 显示图像
cv2.imshow('BGM Augmented Image', augmented_img)
cv2.waitKey(0)  # 按任意键关闭窗口
cv2.destroyAllWindows()

# ✅ 保存结果
cv2.imwrite('bgm_augmented.jpg', augmented_img)
print("增强后的图片已保存为 'bgm_augmented.jpg'")
