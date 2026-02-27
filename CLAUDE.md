# AutoHDR Hackathon — Automatic Lens Correction

## Competition Overview

**Platform**: Kaggle — [automatic-lens-correction](https://www.kaggle.com/competitions/automatic-lens-correction/overview)
**Host**: AutoHDR (AI Real Estate Photo Editing startup)
**Event**: AutoHDR Hackathon, Austin TX, Feb 20–22, 2026
**Prizes**: 1st $5K, 2nd $2.5K, 3rd $500 + job interviews ($120K–$180K)
**Compute Budget**: $100 reimbursed per team (up to $500 with approval)

## Problem Statement

Modern cameras/lenses introduce predictable **barrel distortion**. Professional workflows fix this using manufacturer lens profiles (e.g., Adobe Lightroom). When no lens profile is available (common with new or niche lenses), photos show bent lines, warped interiors, and broken perspective — especially problematic in real estate photography.

**Task**: Build a model that automatically corrects lens distortion in photos **without** needing the manufacturer's lens profile.

- Input: Distorted RAW-format images
- Output: Lens-corrected images
- Training data: Paired (distorted original + corrected ground truth) images
- Test data: Distorted images only (corrections are scored against hidden ground truth)

## Evaluation Metric

Submissions scored using **MAE** between per-image scores and perfect scores (1.0). Lower MAE = better. Perfect = 0.0.

### Per-Image Scoring (multi-metric, geometry-focused):

| Metric              | Weight | What It Measures                          |
|---------------------|--------|-------------------------------------------|
| Edge Similarity     | 40%    | Multi-scale Canny edge F1 score           |
| Line Straightness   | 22%    | Hough line angle distribution match       |
| Gradient Orientation| 18%    | Gradient direction histogram similarity   |
| SSIM                | 15%    | Structural similarity index               |
| Pixel Accuracy      | 5%     | Mean absolute pixel difference            |

**80% of the score is geometry-focused.** Color, noise, compression artifacts matter less.

### Hard Fail Conditions (image scores 0.0):
- Maximum regional difference exceeds threshold (catastrophic local error)
- Edge similarity below minimum (structurally broken output)

## Submission Pipeline

1. Correct all test images using your model/algorithm
2. Upload corrected images (as a zip) to the scoring service
3. Scoring service returns `submission.csv` with per-image scores
4. Download CSV and submit on Kaggle

### submission.csv format:
```csv
image_id,score
08084abe-e9be-4277-8aa1-e1d910312a58_g0,0.9729
10ab83b0-c13a-4b7c-a08e-2d8b2e250a7e_g0,0.9748
```

Note: You do NOT compute scores yourself. The scoring service generates the CSV.

## Final Submission (2:00 PM Sunday)

- 1-minute video overview of solution
- Code/model files
- Submit via: https://forms.gle/7xNL4d8DpAb711hM9

## Technical Context

### What is Lens Distortion?
- **Barrel distortion**: Straight lines curve outward (common with wide-angle lenses used in real estate)
- **Pincushion distortion**: Lines curve inward (telephoto lenses)
- **Radial distortion model**: Described by coefficients k1, k2, k3 in the Brown-Conrady model
- Distortion is radial from the image center and mathematically predictable

### Key Research
- [Blind Geometric Distortion Correction (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Blind_Geometric_Distortion_Correction_on_Images_Through_Deep_Learning_CVPR_2019_paper.pdf) — U-Net predicts distortion flow field + inverse warping
- [Awesome Image Distortion Correction](https://github.com/subeeshvasu/Awesome-Image-Distortion-Correction) — Curated resource list
- [LearnOpenCV — Understanding Lens Distortion](https://learnopencv.com/understanding-lens-distortion/) — Classical approach with OpenCV

### Approach Options (ranked by suitability)

1. **CNN → Distortion Coefficients → OpenCV Undistort** (Recommended)
   - Train ResNet/EfficientNet to regress k1, k2 (radial distortion coefficients)
   - Apply `cv2.undistort()` for mathematically precise correction
   - Geometry-perfect, fast inference, interpretable
   - Assumes radial distortion model (likely valid for this problem)

2. **U-Net / Encoder-Decoder Displacement Field**
   - Predict per-pixel (dx, dy) flow field
   - Warp image using predicted displacements
   - More flexible, handles non-radial distortions
   - Harder to train, needs more compute

3. **Classical: Line Detection + Parameter Optimization**
   - Detect lines (Hough transform), optimize distortion params to straighten them
   - No training needed, but may not generalize well
   - Good quick baseline

4. **Hybrid: Param estimation + learned refinement**
   - Predict distortion params with CNN, apply classical correction, refine residuals with U-Net

### Key Libraries
- `OpenCV` — `cv2.undistort()`, `cv2.getOptimalNewCameraMatrix()`, calibration functions
- `rawpy` — Reading RAW image files
- `PyTorch` / `torchvision` — Model training
- `scikit-image` — SSIM, edge detection, image metrics
- `kornia` — Differentiable image processing (useful for end-to-end training)

## Competition Rules

### Competition-Specific Terms
- **Title**: Automatic Lens Correction Challenge
- **Sponsor**: AutoHDR
- **Website**: https://www.kaggle.com/competitions/automatic-lens-correction

### Team & Submission Limits
- Maximum team size: **5**
- Team mergers allowed (by team leader, combined submissions must not exceed limit)
- Max **5 submissions per day**
- Select up to **2 Final Submissions** for judging

### Data Rules
- Competition data for **non-commercial use only** (competition, Kaggle forums, academic research)
- Do NOT share competition data with non-participants
- **External data allowed** — must be publicly available and equally accessible to all participants at no/minimal cost
- Automated ML Tools (AutoML, H2O, etc.) are **allowed**

### Intellectual Property & Confidentiality (IMPORTANT)
- **All submissions become AutoHDR's property** — irrevocable, worldwide assignment of all code, models, weights, algorithms, documentation, videos, datasets, and derivative works
- Participant warrants submissions don't contain restricted third-party IP (unless disclosed and approved in advance)
- **Confidential Information** (datasets, evaluation methods, scoring systems, model architectures, infrastructure) must remain strictly confidential
- Must **delete all sponsor-provided materials** after hackathon or upon request
- No reverse engineering, decompiling, or deriving sponsor's methods/technology

### Winner Obligations
- Deliver: final model software code (training code, inference code, computational environment description)
- Grant **open source license** (OSI-approved, no commercial restrictions)
- Sign prize acceptance docs and tax forms (W-9 / W-8BEN)
- Must be able to reproduce the winning submission from the description provided

### External Data & Tools
- Pre-trained models and external data: **fair game**
- Must be "reasonably accessible to all" at minimal cost
- AutoML tools explicitly allowed with proper licensing
- Cannot use proprietary datasets costing more than prize value

### Eligibility
- Employees/interns/contractors of AutoHDR and Kaggle may participate but **cannot win prizes**
- U.S.-based participants with valid work authorization (no sponsorship needed)

### Governing Law
- California law, Santa Clara County courts

### Final Deliverables (2:00 PM Sunday, Feb 22)
- 1-minute video overview of solution
- Code/model files
- Submit via: https://forms.gle/7xNL4d8DpAb711hM9

### Prize Details
- 1st Place: $5,000 + full-time job offer ($120K–$180K)
- 2nd Place: $2,500 + potential job offer
- 3rd Place: $500 + potential job offer
- Top 3 guaranteed job interviews
- All clever/creative solutions may get interview consideration

### Compute Reimbursement
- Up to $100 per team automatically reimbursed
- Up to $500 with pre-approval from AutoHDR team
- Email receipt + amount + PayPal email to: ap@AutoHDR.com

## Project Structure (planned)

```
AutoHDR/
├── CLAUDE.md              # This file
├── data/
│   ├── train/
│   │   ├── distorted/     # Training distorted images
│   │   └── corrected/     # Training ground truth
│   └── test/              # Test distorted images
├── src/
│   ├── dataset.py         # Data loading and preprocessing
│   ├── model.py           # Model architecture
│   ├── train.py           # Training loop
│   ├── predict.py         # Inference on test set
│   └── utils.py           # Metrics, visualization helpers
├── notebooks/
│   └── eda.ipynb          # Exploratory data analysis
├── outputs/
│   ├── corrected/         # Corrected test images for submission
│   └── submission.csv     # Final submission file
└── requirements.txt
```
