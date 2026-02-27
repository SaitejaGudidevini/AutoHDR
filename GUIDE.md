# How to Win the AutoHDR Lens Correction Hackathon — Simple Guide

## What's the Problem?

Imagine you take a photo of a room with a wide-angle camera lens.

The photo looks slightly "bent" — straight walls look curved, door frames bow outward, and the room looks like it's being viewed through a fishbowl. This is called **barrel distortion**.

```
What the room actually looks like:     What the camera captures:

 ┌──────────────────┐                  ╭──────────────────╮
 │                  │                  │                  │
 │   ┌────────┐    │                  │   ╭────────╮    │
 │   │  Door  │    │                  │   │  Door  │    │
 │   │        │    │                  │   │        │    │
 │   └────────┘    │                  │   ╰────────╯    │
 │                  │                  │                  │
 └──────────────────┘                  ╰──────────────────╯

 Straight lines                        Lines are curved (distorted)
```

### Why Does This Happen?

Every camera lens bends light slightly. Wide-angle lenses (used heavily in real estate photography) bend it more. The bending follows a predictable mathematical pattern — it's strongest at the edges and barely noticeable at the center.

### How Do Professionals Fix It Today?

Adobe Lightroom has a database of "lens profiles" — one for each camera + lens combination. When you import a photo, Lightroom looks up your lens, applies the matching correction, and the lines become straight again.

**The problem**: New lenses come out faster than profiles get created. When there's no profile, the photo stays distorted.

### What AutoHDR Wants

A single AI model that can fix distortion from **any** lens — no profile needed. Just feed it a distorted photo, and it outputs a corrected one.

---

## What You're Given

### Training Data
Pairs of images:
- `{id}_original.jpg` — the distorted photo (what the camera captured)
- `{id}_generated.jpg` — the corrected version (what it should look like)

Your job: **learn the pattern** from these pairs so you can correct new photos you've never seen.

### Test Data
- Distorted photos only (no corrected version)
- You must produce the corrected versions
- A scoring service grades your corrections against hidden ground truth

---

## How You're Scored (This is Critical)

Your corrected images are compared to the hidden ground truth using 5 metrics:

```
                        WHAT MATTERS MOST
                        ─────────────────

    ┌─────────────────────────────────────────────────────┐
    │  Edge Similarity          ████████████████  40%     │
    │  (Do edges match?)                                  │
    │                                                     │
    │  Line Straightness        █████████  22%            │
    │  (Are straight lines                                │
    │   actually straight?)                               │
    │                                                     │
    │  Gradient Orientation     ███████  18%              │
    │  (Do structural                                     │
    │   directions match?)                                │
    ├─────────────────────────────────────────────────────┤
    │  GEOMETRY TOTAL                          = 80%      │
    ├─────────────────────────────────────────────────────┤
    │  SSIM                     ██████  15%               │
    │  (Overall structure                                 │
    │   similarity)                                       │
    │                                                     │
    │  Pixel Accuracy           ██  5%                    │
    │  (Exact pixel match)                                │
    ├─────────────────────────────────────────────────────┤
    │  APPEARANCE TOTAL                        = 20%      │
    └─────────────────────────────────────────────────────┘
```

### What This Means in Plain English

- **80% of your score = geometry**. Did you straighten the lines? Did you fix the curves? Are edges where they should be?
- **Only 20% = how it looks**. Color accuracy, noise, exact pixel values barely matter.
- **You don't need pixel-perfect output.** A geometrically correct image with slightly different brightness still scores high.

### Instant Fail Conditions
Your image scores **0.0** (zero) if:
- A region of the image is catastrophically wrong (huge local error)
- The edges are completely broken (structurally destroyed)

**Lesson**: It's better to make a small, safe correction than a wild one that ruins part of the image.

---

## The Math Behind Distortion (Simplified)

Barrel distortion follows a simple formula:

```
Every pixel has a distance (r) from the center of the image.

The pixel gets shifted by:
    new_position = old_position × (1 + k1×r² + k2×r⁴)

Where:
    k1, k2 = "distortion coefficients" (small numbers like -0.3, 0.1)
    r = distance from image center (normalized)
```

- **k1 negative** → barrel distortion (lines curve outward)
- **k1 positive** → pincushion distortion (lines curve inward)
- **k1 = 0** → no distortion

The key insight: **the entire distortion is described by just 2-3 numbers (k1, k2, k3)**.

If you can predict those numbers from a photo, you can perfectly undo the distortion using OpenCV's `cv2.undistort()` function.

---

## Approaches to Win (from simplest to most complex)

### Approach 1: Predict the Numbers (Recommended)
```
Distorted Image → CNN (ResNet) → predicts k1, k2 → OpenCV undistort → Corrected Image
```

**Why this is great:**
- The correction is mathematically exact (perfect geometry)
- Small model, fast to train (even on Apple Silicon)
- Matches the scoring metric perfectly (geometry-focused)
- Only need to predict 2-3 numbers per image

**How to get training labels:**
- Compare each (distorted, corrected) pair
- Find matching features between them
- Fit the distortion formula to find k1, k2
- Now you have: image → [k1, k2] training pairs

### Approach 2: Pixel-to-Pixel Correction (U-Net)
```
Distorted Image → U-Net → Corrected Image directly
```

**Why this is flexible:**
- Handles any type of distortion (not just radial)
- End-to-end learning

**Why this is harder:**
- Needs more compute and training time
- Has to learn the geometry implicitly
- Output quality depends on resolution

### Approach 3: Classical (No Training)
```
Distorted Image → Detect lines → Optimize k1,k2 to make lines straight → Apply correction
```

**Why this is useful:**
- No training needed at all
- Good quick baseline
- Works on individual images

**Why this might not win:**
- Slow per image
- Fails if image has few straight lines
- Less robust than learned approaches

### Approach 4: Hybrid (Best of Both Worlds)
```
Distorted Image → CNN predicts k1,k2 → OpenCV undistort → U-Net refines residual errors → Final Image
```

---

## How to Excel — Winning Strategy

### 1. Speed Matters: Get a Baseline Score Fast
- Don't spend 2 days perfecting one approach
- Get something submitted in the first few hours
- Iterate from there

### 2. Understand the Scoring Metric
- **Edge similarity is 40%** — your correction must preserve edges
- Use edge-aware training losses (Sobel, Canny-based losses)
- Validate by computing edge maps of your output vs ground truth

### 3. Avoid the Instant-Fail Traps
- Never output an image that's wildly different in any region
- If unsure, make a conservative (smaller) correction
- Check for black borders or artifacts after undistorting

### 4. Smart Training Tricks
- **Pre-trained backbones**: Use ImageNet-pretrained ResNet/EfficientNet — don't train from scratch
- **Data augmentation**: Flips, slight rotations, brightness changes (but be careful — these don't change distortion)
- **Resolution**: Train at 256x256 or 512x512 for speed, but apply correction at full resolution

### 5. Post-Processing Matters
- After undistorting, you might get black borders (pixels that don't exist after correction)
- Crop or inpaint these borders
- Make sure output resolution matches expected size

### 6. Ensemble If Time Allows
- Run 2-3 different models
- Average their predicted coefficients
- Or: take the correction that has the best self-consistency (straightest lines)

### 7. What Judges Want to See (for the Job Interview)
- Clean, well-organized code
- Clear understanding of the problem
- Creative or clever solutions
- Ability to explain your approach in 1 minute (video submission)

---

## Common Pitfalls to Avoid

| Pitfall | Why It Hurts | What to Do Instead |
|---------|-------------|-------------------|
| Training on full-resolution images | Too slow, runs out of memory | Train at 256-512px, apply correction at full res |
| Ignoring black borders after undistort | Scoring service might penalize | Crop or inpaint borders |
| Over-correcting | Pincushion is worse than barrel | Use conservative predictions |
| Pixel-perfect loss only (MSE) | Doesn't match the 80% geometry metric | Add edge + SSIM losses |
| Not submitting early | You don't know your baseline score | Submit something in the first 3-4 hours |
| Training from scratch | Wastes time, worse features | Always use pre-trained backbones |
| Ignoring the "generated" naming | Confusing which is distorted vs corrected | `original` = distorted, `generated` = corrected |

---

## Your 2-Day Timeline

```
FRIDAY (Today)
├── Evening: Understand data, explore pairs, visualize distortion
│
SATURDAY
├── Morning: Build + train parameter prediction model (Approach 1)
├── Afternoon: First submission → see baseline score
├── Evening: Iterate — improve model, try alternatives
│
SUNDAY
├── Morning: Final training runs, ensemble if possible
├── Noon: Final submission, record 1-min video
└── 2:00 PM: Submit video + code via Google Form
```

---

## Quick Reference

| Term | Meaning |
|------|---------|
| Barrel distortion | Lines curve outward (like a barrel) |
| k1, k2 | Numbers that describe how much distortion there is |
| `cv2.undistort()` | OpenCV function that removes distortion given k1, k2 |
| `original` | The distorted photo (input) |
| `generated` | The corrected photo (target/ground truth) |
| MAE | Mean Absolute Error — lower is better, 0.0 is perfect |
| SSIM | Structural Similarity — measures if images look structurally alike |
| Edge F1 | How well edges in your output match edges in ground truth |
