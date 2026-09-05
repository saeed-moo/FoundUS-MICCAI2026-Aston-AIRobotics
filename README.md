# FoundUS 2026 — Team Aston-AIRobotics

Unified multi-task ultrasound landmark detection across 9 tasks (A4C, AOP, FA, FUGC, HC, IVC, PLAX, PSAX, fetal femur).

This is our submission code for the Foundation Model for Ultrasound Biometry challenge (MICCAI 2026). It accompanies our accepted MWM workshop paper, *"Encoder Capacity Is the Binding Constraint in Unified Multi-Task Ultrasound Landmark Detection."*

## Method

A single DINOv3-Large encoder (`vit_large_patch16_dinov3.lvd1689m`) adapted with LoRA (rank 16; 2.39M of 305M parameters trained, 0.78%), feeding 9 task-specific heatmap heads. Images are letterboxed to 512x512 (ImageNet normalization) and the model predicts 128x128 heatmaps that are decoded to landmark coordinates. Training uses an MSE heatmap loss, AdamW (encoder LoRA at LR 2e-5, heads at 1e-3), a cosine schedule, batch size 8, up to 250 epochs. Split-screen detection and cropping are applied to cardiac views.

## Key findings

The paper runs a controlled ablation, varying one factor at a time. Changing the loss or the output resolution does not help; only increasing encoder capacity does. Swapping DINOv2-Small for DINOv3-Large + LoRA cuts average MRE from 11.01 to 8.61 px (HC from 30.5 to 19.1).

Split-screen cropping on cardiac views gives large gains: IVC 40.2 to 2.76, PSAX 27.4 to 1.70 px.

We also find a generalization gap driven by acquisition geometry: 8.61 px locally versus 25.60 px on the challenge server. A distortion test shows that stretching the aspect ratio alone (1.48 to 1.90) triples the error (10.5 to 35.1 px), so geometry coverage, not encoder capacity, is the remaining limit.

![Validation convergence: DINOv3-L + LoRA vs. DINOv2-S baseline](fig2_convergence.png)

## Results (held-out split)

Average MRE 8.61 px.

| A4C | AOP | FA | FUGC | HC | IVC | PLAX | PSAX | fetal femur |
|-----|-----|-----|------|-----|-----|------|------|-------------|
| 3.93 | 6.36 | 8.68 | 6.82 | 19.12 | 2.76 | 3.35 | 1.70 | 24.81 |

Preliminary challenge score: 25.60.

## Layout

training/ train_finetune.py, model_factory.py, dataset.py, utils.py
inference/ model.py (challenge submission interface, Model.predict)
preprocessing/ detect_splitscreen.py, apply_crop.py
requirements.txt


## Requirements

`torch>=2.1, timm>=1.0.20, peft>=0.19, albumentations, opencv-python, pandas, numpy, tqdm`

Install with `pip install -r requirements.txt`.

## Training and inference

Train with `python training/train_finetune.py` (DINOv3 weights loaded from `PRETRAINED_ENCODER_PATH`; epochs configurable via `NUM_EPOCHS`). At inference, `Model.predict()` in `inference/model.py` writes `regression_predictions.json` with normalized and pixel landmark coordinates per task.

## Docker

`seven816/foundus:v1.0` (linux/amd64; DINOv3 cache baked in for offline

inference).

## Citation

Moosivand, S., Wang, Z. *Encoder Capacity Is the Binding Constraint in Unified Multi-Task Ultrasound Landmark Detection.* MICCAI 2026 MWM Workshop.
