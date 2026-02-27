# AutoHDR Hackathon — Session Archive

**Date**: Feb 22, 2026
**Competition**: Kaggle Automatic Lens Correction Challenge
**Deadline**: Feb 22, 2026 at 2:00 PM

---

## Submission History

| Attempt | Approach | Score | Issue |
|---------|----------|-------|-------|
| v1 CNN | ResNet50 → k1,k2 regression (MSE loss) | 2.86/100 | Model collapsed to mean predictions (k1≈0, k2≈0). 642/1000 hard fails. |
| Classical Hough | Per-image line detection + param optimization | 1.27/100 | Over-correction. Maximizing line straightness ≠ correct distortion params. |
| E2E Differentiable + Hough Refinement | ResNet50 E2E trained + Hough post-processing | 1.27/100 | Hough refinement over-corrected good CNN predictions. |

---

## Best Model: E2E Differentiable Pipeline

### Architecture
- **Backbone**: ResNet50 (ImageNet pre-trained)
- **Head**: AdaptiveAvgPool → FC(2048,512) → ReLU → Dropout(0.3) → FC(512,2) → Tanh × 0.5
- **Output**: k1, k2 in range [-0.5, 0.5]
- **Differentiable undistort**: `F.grid_sample()` with radial distortion grid for end-to-end backprop

### Training Details
- **GPU**: RunPod RTX 4090 ($0.59/hr)
- **Data**: 23,120 pairs (21,964 train / 1,156 val), 256×256 resolution
- **Optimizer**: AdamW, lr=3e-4, CosineAnnealing over 50 epochs
- **Batch size**: 24
- **Mixed precision**: Yes (torch.amp)
- **Time per epoch**: ~6 minutes

### Loss Schedule (3 phases)
| Phase | Epochs | Losses | line_weight |
|-------|--------|--------|-------------|
| 1 | 1-10 | Charbonnier + SSIM | 0.0 |
| 2 | 11-25 | Charbonnier + SSIM + LineStraightness | 0.05 |
| 3 | 26-50 | Charbonnier + SSIM + LineStraightness | 0.1 |

### Training Results
- **Best epoch**: 10 (val loss: 0.0190)
- **Final epoch 43**: val loss 0.0406
- **k1 std**: 0.044 (healthy diversity, no mean collapse)
- **k2 std**: 0.168 (healthy diversity)

### CNN-Only Predictions (before Hough)
- k1: mean=0.021, std=0.044
- k2: mean=-0.146, std=0.168

### After Hough Refinement (what was submitted)
- k1: mean=0.111, std=0.059, range=[-0.16, 0.24]
- k2: mean=-0.181, std=0.172, range=[-0.80, 0.69]
- **Hough shifted k1 by ~+0.09 to +0.15 and k2 by ~-0.10 to -0.20 from CNN predictions**

---

## Root Cause of 1.27 Score

The **Hough refinement** in `predict_and_correct()` was the problem:
- Searches +/-0.15 around k1_cnn and +/-0.3 around k2_cnn (~150 combinations)
- Picks whichever (k1, k2) maximizes Hough line straightness score
- Problem: maximizing Hough line straightness does NOT equal correct distortion parameters
- The refinement was massively over-correcting, turning barrel distortion into pincushion
- This is the same failure mode as the classical approach (also scored 1.27)

### Evidence (from predictions_e2e.csv)
```
Example image: CNN predicted k1=0.007, k2=-0.146
              Hough changed to k1=0.127, k2=-0.346
              Shift: k1 +0.12, k2 -0.20 (massive over-correction)
```

---

## Immediate Fix (NOT YET DONE)

**Submit with CNN-only predictions (no Hough refinement)**:
1. Modify `predict_and_correct()` in `train_e2e.py` to skip Hough loop
2. Or use `predictions_e2e.csv` columns `k1_cnn`/`k2_cnn` directly
3. Re-apply `cv2.undistort()` with CNN-only params
4. Re-zip and re-submit

Alternative fixes:
- Tighten Hough search range to +/-0.02 k1, +/-0.05 k2
- Use best epoch model (epoch 10) instead of last epoch
- Submit raw images (no correction) as a baseline to understand scoring

---

## RunPod Pod Details

- **Pod ID**: vh44vzpy8xiflh
- **SSH**: `root@103.196.86.102 -p 11289 -i ~/.ssh/runpod_key`
- **GPU**: RTX 4090, $0.59/hr
- **Container disk**: 100GB
- **Status**: STILL RUNNING (billing active)

### Pod file locations
```
/workspace/train_e2e.py          — training script
/workspace/resnet50.pth          — ImageNet weights
/workspace/train-data/           — 46,238 training files (23,119 pairs)
/workspace/test-data/            — 1,000 test images
/workspace/checkpoints/best_model.pth  — best model (epoch 10, val 0.0190)
/workspace/output/corrected_e2e.zip    — corrected images (scored 1.27)
/workspace/output/predictions_e2e.csv  — per-image predictions
/workspace/training.log                — full training log
```

---

## Local File Locations

```
/Users/saiteja/Documents/Dev/AutoHDR/
├── src/train_e2e.py              — E2E training + inference script (604 lines)
├── src/classical_correction.py   — Classical Hough approach
├── src/create_presentation.py    — Presentation slide generator
├── outputs/
│   ├── best_model_e2e.pth        — Best model weights (local copy)
│   ├── corrected_e2e.zip         — 564MB corrected test images
│   ├── predictions_e2e.csv       — Per-image predictions (k1_cnn, k2_cnn, k1_final, k2_final)
│   └── submission.csv            — Kaggle submission CSV (scored 1.27)
├── presentation/                 — 8 PNG slides explaining pipeline
├── lens-correction-train-cleaned/ — 36GB training data (23,118 pairs)
└── test-originals/               — 803MB test images (1000 files)
```

---

## Key Script: train_e2e.py

### Critical functions
- `build_radial_undistort_grid(k1, k2, h, w)` — builds distortion grid for F.grid_sample
- `differentiable_undistort(img, k1, k2)` — applies undistortion differentiably
- `predict_and_correct(model, test_dir, output_dir)` — inference with Hough refinement (THE BUG)
- `train(data_dir, epochs, batch_size, ...)` — 3-phase training loop

### The buggy Hough refinement code (in predict_and_correct):
```python
# Searches +/-0.15 k1 and +/-0.3 k2 around CNN prediction
for dk1 in np.linspace(-0.15, 0.15, 13):
    for dk2 in np.linspace(-0.3, 0.3, 13):
        # ... applies undistort, measures Hough line straightness
        # Picks combo with highest straightness score
```

---

## What Worked
- E2E differentiable training successfully learned diverse k1/k2 predictions (no mean collapse)
- 3-phase loss scheduling kept training stable
- Data transfer pipeline (runpodctl send/receive) worked for 36GB
- Training completed in ~5 hours on RTX 4090

## What Failed
- Hough refinement at inference time over-corrected everything
- Maximizing Hough line straightness ≠ matching ground truth distortion
- This negated the good CNN predictions

## Lessons Learned
1. Don't add post-processing that optimizes a proxy metric (line straightness) when the evaluation metric is different (edge similarity, SSIM, etc.)
2. Trust the CNN predictions — they had healthy diversity and reasonable values
3. Always submit a "CNN-only" version before adding refinement
4. Test refinement on validation set before applying to test set
