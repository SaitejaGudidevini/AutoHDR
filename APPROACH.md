# Core Approach: CNN → Distortion Coefficients → OpenCV Undistort

## The Big Idea in One Sentence

Instead of teaching an AI to "draw" the corrected image pixel by pixel, we teach it to **figure out what went wrong** (find 2 numbers), and then use **math to perfectly fix it**.

---

## Step-by-Step Breakdown

### Step 1: Understanding What Distortion Actually Is

When light passes through a lens, it bends. This bending follows a simple math formula:

```
For any pixel at distance r from the center of the image:

    distorted_position = real_position × (1 + k1×r² + k2×r⁴)
```

That's it. The entire distortion is controlled by just **two numbers**: `k1` and `k2`.

Let's see what different values do:

```
k1 = -0.3 (negative)           k1 = 0 (zero)              k1 = +0.3 (positive)
BARREL DISTORTION               NO DISTORTION               PINCUSHION DISTORTION

  ╭──────────────╮             ┌──────────────┐             ┌──────────────┐
  │ ╭──────────╮ │             │ ┌──────────┐ │             │ ╱──────────╲ │
  │ │          │ │             │ │          │ │             │╱            ╲│
  │ │          │ │             │ │          │ │             │╲            ╱│
  │ ╰──────────╯ │             │ └──────────┘ │             │ ╲──────────╱ │
  ╰──────────────╯             └──────────────┘             └──────────────┘

  Lines bow OUTWARD             Lines are STRAIGHT           Lines bow INWARD
  (wide-angle lenses)           (perfect)                    (telephoto lenses)
```

### Why This Is Powerful

If you know k1 and k2, you can **perfectly reverse** the distortion. It's not an approximation — it's mathematically exact. OpenCV has a function `cv2.undistort()` that does this reversal.

```
The formula works both ways:

    DISTORT:    real_position → distorted_position   (camera does this)
    UNDISTORT:  distorted_position → real_position   (we want to do this)
```

---

### Step 2: The Problem — We Don't Know k1 and k2

When someone uploads a photo, we don't know what lens they used. We don't know k1 or k2. We just have a distorted image.

**So we need to GUESS k1 and k2 just by looking at the image.**

This is where the CNN (neural network) comes in.

---

### Step 3: Training the CNN to Predict k1, k2

#### What is a CNN?

A Convolutional Neural Network (CNN) is a type of AI that's really good at understanding images. Think of it as:

```
Image → [millions of learned filters] → understanding of what's in the image → output
```

We're using **ResNet50** — a CNN pre-trained on millions of images (ImageNet). It already knows how to "see" edges, shapes, lines, curves, textures. We just need to teach it one more thing: **how much are the lines bent?**

#### How We Train It

```
TRAINING PHASE (what happens on the GPU):

    For each training pair:

    1. We have:
       - original.jpg    (distorted image)
       - generated.jpg   (corrected image)

    2. We figure out k1, k2 by comparing the two:
       - Find matching features in both images (SIFT)
       - Optimize: "what k1, k2 would transform original → generated?"
       - Now we have the ground truth label: [k1, k2]

    3. We train the CNN:
       ┌─────────────────┐
       │ distorted image  │──→ ResNet50 ──→ predicts [k1_pred, k2_pred]
       └─────────────────┘                          │
                                                    ↓ compare
       Ground truth: [k1_true, k2_true] ←──── Loss = (pred - true)²
                                                    │
                                                    ↓
                                          Update CNN weights
                                          (make prediction closer
                                           to truth next time)

    Repeat for all 23,000 training images, multiple times (epochs).
```

After training, the CNN has learned: **"when I see lines curving outward this much, k1 is approximately -0.25"**

#### The Architecture

```
                        ResNet50 (Pre-trained on ImageNet)
                        ──────────────────────────────────

Input Image             ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
(256×256×3)     ──→     │ Conv    │──→│ Block 1 │──→│ Block 2 │──→│ Block 3 │──→
                        │ Layer   │   │ (64ch)  │   │ (128ch) │   │ (256ch) │
                        └─────────┘   └─────────┘   └─────────┘   └─────────┘
                                                                        │
                                                                        ↓
                        ┌──────────┐   ┌─────────┐   ┌─────────┐
Output: [k1, k2] ←──   │ Linear  │←──│ Block 5 │←──│ Block 4 │
(2 numbers)             │ 2048→2  │   │ (2048ch)│   │ (512ch) │
                        └──────────┘   └─────────┘   └─────────┘

The "Blocks" are groups of convolutional layers that progressively
understand the image at different scales:

Block 1: Detects edges, basic textures
Block 2: Detects corners, patterns
Block 3: Detects parts of objects (walls, doors)
Block 4: Detects full objects and spatial relationships
Block 5: Understands global image structure (overall curvature!)

The final Linear layer maps Block 5's understanding → [k1, k2]
```

**Key idea**: We DON'T train the whole network from scratch. ResNet50 already knows how to see. We just replace the last layer (which originally classified "cat vs dog") with a new layer that outputs [k1, k2] (regression instead of classification).

This is called **transfer learning** — borrowing knowledge from a pre-trained model.

---

### Step 4: Inference (Using the Trained Model)

Once trained, using the model is dead simple:

```
TEST PHASE (what happens at submission time):

    For each test image:

    ┌──────────────────┐
    │  Distorted test  │
    │  image           │
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────┐
    │  Trained ResNet50 │──→ predicts k1 = -0.28, k2 = 0.05
    └──────────────────┘
             │
             ↓
    ┌──────────────────┐
    │  cv2.undistort() │    Uses the predicted k1, k2 to
    │  (OpenCV math)   │    mathematically reverse the distortion
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────┐
    │  Corrected image │──→ Save as output
    │  (straight lines)│
    └──────────────────┘
```

### What cv2.undistort() Does Under the Hood

```python
# This is what OpenCV does internally:

for each pixel (x, y) in the output image:
    # 1. Normalize coordinates (center = 0,0)
    x_norm = (x - center_x) / focal_length
    y_norm = (y - center_y) / focal_length

    # 2. Compute distance from center
    r² = x_norm² + y_norm²

    # 3. Apply distortion model (find where this pixel came from)
    x_distorted = x_norm × (1 + k1×r² + k2×r⁴)
    y_distorted = y_norm × (1 + k1×r² + k2×r⁴)

    # 4. Sample the pixel from the distorted image
    output[y, x] = distorted_image[y_distorted, x_distorted]
```

It goes pixel by pixel and asks: "where did this pixel come from in the distorted image?" Then it copies it to the right place.

---

## Step 2 Deep Dive: How We Extract k1, k2 from Training Pairs

This is the trickiest part. We have paired images (distorted + corrected) and need to figure out what k1, k2 values were used to correct it.

### Method: Feature Matching + Optimization

```
STEP 2a: Find matching points between the two images
─────────────────────────────────────────────────────

    Distorted Image                 Corrected Image
    ┌───────────────────┐           ┌───────────────────┐
    │    ★              │           │   ★               │
    │         ●         │           │        ●          │
    │  ▲                │           │  ▲                │
    │              ■    │           │             ■     │
    │      ◆            │           │      ◆            │
    └───────────────────┘           └───────────────────┘

    SIFT/ORB feature detector finds these points (★●▲■◆)
    in both images and matches them.

    Result: pairs of coordinates
    Point 1: (102, 45) in distorted  →  (105, 43) in corrected
    Point 2: (230, 120) in distorted →  (225, 118) in corrected
    Point 3: (50, 200) in distorted  →  (55, 195) in corrected
    ... hundreds of matched points


STEP 2b: Optimize k1, k2 to explain the point shifts
─────────────────────────────────────────────────────

    We try different values of k1, k2 and see which ones
    best explain how the points moved:

    Trial 1: k1=-0.1, k2=0.0  →  predicted shifts don't match  →  error: 15.2
    Trial 2: k1=-0.2, k2=0.0  →  closer but not quite         →  error: 8.7
    Trial 3: k1=-0.3, k2=0.05 →  almost perfect match!         →  error: 1.2
    Trial 4: k1=-0.28, k2=0.04 → best match                   →  error: 0.8  ← winner!

    We use scipy.optimize to do this search efficiently.
```

### Alternative Method: Grid Search with SSIM

If feature matching is noisy, there's a simpler (but slower) approach:

```
For each training pair:
    For k1 in [-0.5, -0.49, -0.48, ..., 0.48, 0.49, 0.5]:
        For k2 in [-0.3, -0.29, ..., 0.29, 0.3]:
            corrected = cv2.undistort(distorted, k1, k2)
            score = SSIM(corrected, ground_truth)

    Best k1, k2 = the ones with highest SSIM score
```

This is brute force but very reliable. We can make it fast with coarse-to-fine search.

---

## Why This Approach Matches the Scoring Metric Perfectly

```
Scoring Metric              Why Our Approach Nails It
──────────────              ────────────────────────

Edge Similarity (40%)       cv2.undistort() preserves ALL edges perfectly.
                            It just moves them to the right place.
                            No blurring, no artifacts, no hallucinated edges.

Line Straightness (22%)     This is LITERALLY what undistortion does —
                            it makes curved lines straight. If k1, k2
                            are correct, lines will be mathematically straight.

Gradient Orientation (18%)  Undistortion preserves gradient directions.
                            It's a geometric transformation, not a filter.

SSIM (15%)                  Structure is perfectly preserved because we're
                            just remapping pixels, not generating new ones.

Pixel Accuracy (5%)         Pixels are moved to correct positions.
                            Only interpolation introduces tiny differences.
```

Compare with a U-Net approach where the network has to **generate** every pixel — it might blur edges, hallucinate textures, or introduce subtle artifacts that hurt on all 5 metrics.

---

## What Could Go Wrong (and How We Handle It)

### Problem 1: Distortion Isn't Purely Radial
Some lenses have tangential distortion (p1, p2 parameters) or decentering.

**Solution**: Add p1, p2 to our prediction: CNN → [k1, k2, p1, p2]. OpenCV handles all of these.

### Problem 2: Black Borders After Undistortion
When you undistort, pixels at the edges get pulled inward, leaving black triangles in the corners.

```
Before undistort:          After undistort:
┌──────────────┐           ┌──────────────┐
│              │           │▚            ▚│  ← black corners
│              │           │              │
│              │           │              │
│              │           │▚            ▚│  ← black corners
└──────────────┘           └──────────────┘
```

**Solution**: Use `cv2.getOptimalNewCameraMatrix()` to crop or scale the image to hide borders. The ground truth images likely handle this too.

### Problem 3: Some Photos Have No Distortion
The founder said some photos are already clean.

**Solution**: The CNN will predict k1≈0, k2≈0 for these. `cv2.undistort()` with k1=0, k2=0 returns the image unchanged. This is handled automatically.

### Problem 4: CNN Predicts Wrong Coefficients
If the prediction is off, the correction will be wrong.

**Solution**:
- Use a robust loss function (Huber loss instead of MSE — less sensitive to outliers)
- Train with data augmentation
- Validate on held-out set
- Conservative predictions are better than wild ones (avoid hard-fail conditions)

---

## The Full Pipeline — Code Overview

```python
# ============================================
# PHASE 1: Extract k1, k2 from training pairs
# ============================================

for original, generated in training_pairs:
    # Find matching features
    matches = sift_match(original, generated)

    # Optimize k1, k2 to explain the transformation
    k1, k2 = optimize_distortion_params(matches)

    # Save: image_path → [k1, k2]
    labels.append((image_path, k1, k2))

# ============================================
# PHASE 2: Train CNN to predict k1, k2
# ============================================

model = ResNet50(pretrained=True)
model.fc = Linear(2048, 2)  # Replace classifier with [k1, k2] regressor

for epoch in range(100):
    for images, true_k1k2 in dataloader:
        predicted_k1k2 = model(images)
        loss = huber_loss(predicted_k1k2, true_k1k2)
        loss.backward()
        optimizer.step()

# ============================================
# PHASE 3: Correct test images
# ============================================

for test_image in test_images:
    k1, k2 = model.predict(test_image)

    camera_matrix = estimate_camera_matrix(test_image)
    dist_coeffs = [k1, k2, 0, 0]  # [k1, k2, p1, p2]

    corrected = cv2.undistort(test_image, camera_matrix, dist_coeffs)
    save(corrected)
```

---

## Key Terms Glossary

| Term | Simple Meaning |
|------|---------------|
| **CNN** | A neural network designed to understand images |
| **ResNet50** | A specific CNN with 50 layers, pre-trained on 1M+ images |
| **Transfer Learning** | Using a pre-trained model and adapting it for a new task |
| **k1, k2** | Two numbers that describe how much a lens bends light |
| **Barrel distortion** | When k1 < 0, lines curve outward like a barrel |
| **cv2.undistort()** | OpenCV function that reverses distortion given k1, k2 |
| **SIFT** | Algorithm that finds distinctive points in images for matching |
| **Regression** | Predicting continuous numbers (vs classification = predicting categories) |
| **Loss function** | How we measure "how wrong" the CNN's prediction is |
| **Epoch** | One full pass through all training data |
| **Camera matrix** | Numbers describing the camera's internal properties (focal length, center point) |
| **Focal length (fx, fy)** | How "zoomed in" the camera is |
| **Principal point (cx, cy)** | The optical center of the image (usually near the pixel center) |

---

## Why This Approach Will Impress the Judges

1. **Elegant** — Reducing a complex image problem to predicting 2 numbers
2. **Mathematically grounded** — Uses the actual physics of how lenses work
3. **Explainable** — Easy to describe in a 1-minute video
4. **Production-ready** — This is how you'd actually deploy it at AutoHDR
5. **Robust** — Handles edge cases (no distortion, varying lenses) naturally
6. **Fast** — Inference in milliseconds per image (they process 93K photos/day)
