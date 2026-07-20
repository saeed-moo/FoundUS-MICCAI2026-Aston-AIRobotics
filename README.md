# FoundUS 2026 — Team Aston-AIRobotics

Unified multi-task ultrasound landmark detection across 9 tasks
(A4C, AOP, FA, FUGC, HC, IVC, PLAX, PSAX, fetal femur).

## Method
DINOv3-Large encoder (`vit_large_patch16_dinov3.lvd1689m`) adapted with
LoRA (rank 16, 2.39M/305M trainable = 0.78%), feeding 9 task-specific
heatmap heads. Input 512x512 letterbox. Split-screen cropping applied to
cardiac views during training.

## Results (held-out split)
Average MRE 8.61 px. Per task: A4C 3.93, AOP 6.36, FA 8.68, FUGC 6.82,
HC 19.12, IVC 2.76, PLAX 3.35, PSAX 1.70, fetal femur 24.81.

## Layout
- `training/` — training script, dataset, model factory
- `inference/model.py` — challenge submission interface (`Model.predict`)
- `preprocessing/` — split-screen detection and cropping

## Requirements
torch>=2.1, timm>=1.0.20, peft>=0.19, albumentations, opencv-python,
pandas, numpy, tqdm

## Training

## Inference

## Citation
Moosivand, S., Wang, Z. Encoder Capacity Is the Binding Constraint in
Unified Multi-Task Ultrasound Landmark Detection. MICCAI 2026 MWM Workshop.
