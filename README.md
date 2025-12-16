# About Bimodal-Cross-Fusion
Code release for "MIX-based Foreground and Background Patch Augmentation Guided by Physics and Material Properties for X-ray Detection"

The performance of deep learning-based models for X-ray prohibited item detection heavily relies on large-scale, diverse datasets, which are often unavailable. While data augmentation offers a promising solution, prevalent methods ignore the fundamental principles of X-ray imaging, leading to artifacts such as distorted material properties and unnatural thickness perturbations. To bridge this gap, we present MIX, a physics-grounded data augmentation pipeline. The core idea of MIX is to manipulate image attributes in a way that reflects real-world physical variations. Our contributions are twofold: (1) to address material ambiguity, MIX modulates foreground pseudo-colors by directly manipulating hue and saturation, informed by the relationship between color and effective atomic number, which forces the model to learn more robust material representations; and (2) to simulate variations in object density and thickness, MIX introduces a novel thickness perturbation technique based on X-ray attenuation principles, significantly improving the model’s adaptability to geometric changes. Our proposed method seamlessly integrates with existing detectors and yields substantial performance gains across multiple benchmarks. Our work not only provides an effective augmentation solution but also highlights the critical need for domain-specific approaches in X-ray computer vision.
## Environment configuration
* Requirements
```python
python 3.8.20
pytorch 2.1.0
mmdet 3.3.0
mmcv 2.1.0
```

## Dataset
The experiments were conducted on two prohibited item datasets, CLCXray and OPIXray. 
## Train
```
cd MIX/mmdetection
python /tools/train.py /configs/atss/atss_r50_fpn_1x_coco.py
```
## Test
```
python /tools/train.py /configs/atss/atss_r50_fpn_1x_coco.py
```
