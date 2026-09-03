# FoundUS 2026 — Team Aston-AIRobotics

Unified multi-task ultrasound landmark detection across 9 tasks
(A4C, AOP, FA, FUGC, HC, IVC, PLAX, PSAX, fetal femur).

Accompanies our accepted MICCAI 2026 MWM workshop paper,
*"Encoder Capacity Is the Binding Constraint in Unified Multi-Task Ultrasound
Landmark Detection."*

## Method

A single **DINOv3-Large** encoder (`vit_large_patch16_dinov3.lvd1689m`) adapted
with **LoRA** (rank 16, 2.39M / 305M trainable = 0.78%), feeding 9 task-specific
heatmap heads. Input 512×512 (letterbox resize + pad, ImageNet normalization);
128×128 output heatmaps decoded to landmark coordinates. Trained with an MSE
heatmap loss, AdamW (encoder LoRA LR 2e-5, heads LR 1e-3), CosineAnnealingLR,
batch 8, up to 250 epochs, seed 42. Split-screen detection and cropping are
applied to cardiac views during training.

## Key findings (from the paper)

- **Encoder capacity is the binding constraint.** In a controlled ablation
  (one axis varied at a time), changes to loss or output resolution do not move
  the metric; only the encoder swap DINOv2-Small → DINOv3-Large + LoRA does,
  cutting average MRE from 11.01 to 8.61 px (HC 30.5 → 19.1).
- **Split-screen cropping** on cardiac views gives large gains: IVC 40.2 → 2.76,
  PSAX 27.4 → 1.70 px.
- **Covariate shift.** Local held-out 8.61 px vs. challenge server 25.60 px; a
  distortion experiment shows aspect-ratio change alone (1.48 → 1.90) triples
  error (10.5 → 35.1 px) — acquisition geometry, not encoder capacity, is the
  generalization limit.

## Results (held-out split)

Average MRE **8.61 px**. Per task:

| A4C | AOP | FA | FUGC | HC | IVC | PLAX | PSAX | fetal femur |
|-----|-----|-----|------|-----|-----|------|------|-------------|
| 3.93 | 6.36 | 8.68 | 6.82 | 19.12 | 2.76 | 3.35 | 1.70 | 24.81 |

Preliminary challenge score: 25.60.

## Layout


## Requirements


Install: `pip install -r requirements.txt`

## Training

```bash
python training/train_finetune.py
# key config (env-overridable): NUM_EPOCHS, MODEL_SAVE_PATH
# DINOv3 pretrained encoder weights loaded from PRETRAINED_ENCODER_PATH
```

## Inference

`Model.predict()` in `inference/model.py` runs the model on the input images and
writes `regression_predictions.json` (normalized + pixel landmark coordinates
per task), the format expected by the challenge.

## Docker

`seven816/foundus:v1.0` (linux/amd64; DINOv3 cache baked in for offline
inference).

## Citation

> Moosivand, S., Wang, Z. *Encoder Capacity Is the Binding Constraint in Unified
> Multi-Task Ultrasound Landmark Detection.* MICCAI 2026 MWM Workshop.
