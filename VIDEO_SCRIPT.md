# 1-Minute Video Script — AutoHDR Lens Correction

**Total time: 60 seconds | ~130 words**

---

### Slide 1 (0:00 – 0:12)

> "Hi, I'm Sai Teja. For the AutoHDR Lens Correction Challenge, I took a physics-first approach. Wide-angle lenses bend straight lines outward — barrel distortion — a big problem in real estate photography."

### Slide 2 (0:12 – 0:22)

> "The key insight: all radial distortion is controlled by just two numbers, k1 and k2. Know these, and OpenCV reverses the distortion perfectly."

### Slide 3 (0:22 – 0:35)

> "So my pipeline is simple. A ResNet50 predicts k1 and k2, then OpenCV corrects the image. Training is end-to-end — the undistortion is differentiable using PyTorch grid sample, so gradients flow back to the predictions."

### Slide 4 (0:35 – 0:48)

> "Trained on twenty-three thousand pairs on an RTX 4090. The loss combines pixel, structural, and line straightness terms — matching the competition's eighty percent geometry scoring."

### Slide 5 (0:48 – 0:60)

> "Inference takes five milliseconds per image. Physics-based, geometry-preserving, and production-ready at AutoHDR's scale. Thank you."
