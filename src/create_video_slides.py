"""
Generate clean, visually compelling slides for the 1-minute video submission.
5 slides, designed for screen recording.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUTPUT_DIR = '/Users/saiteja/Documents/Dev/AutoHDR/video_slides'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
BG = '#0d1117'
CARD_BG = '#161b22'
ACCENT_BLUE = '#58a6ff'
ACCENT_GREEN = '#3fb950'
ACCENT_ORANGE = '#d29922'
ACCENT_RED = '#f85149'
ACCENT_PURPLE = '#bc8cff'
TEXT_PRIMARY = '#e6edf3'
TEXT_SECONDARY = '#8b949e'
BORDER = '#30363d'

def setup_slide(title, subtitle=None):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Title
    ax.text(8, 8.2, title, fontsize=28, fontweight='bold',
            ha='center', va='center', color=TEXT_PRIMARY)
    if subtitle:
        ax.text(8, 7.5, subtitle, fontsize=14,
                ha='center', va='center', color=TEXT_SECONDARY, style='italic')

    return fig, ax

def add_box(ax, x, y, w, h, title, body, color, title_size=13, body_size=10):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                           facecolor=CARD_BG, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.35, title, fontsize=title_size, ha='center', va='center',
            fontweight='bold', color=color)
    if body:
        ax.text(x + w/2, y + h/2 - 0.15, body, fontsize=body_size, ha='center', va='center',
                color=TEXT_SECONDARY, family='monospace', linespacing=1.6)

def save_slide(fig, name, num):
    path = os.path.join(OUTPUT_DIR, f'slide_{num:02d}_{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG,
                pad_inches=0.3)
    plt.close(fig)
    print(f'Saved: {path}')


# ============================================================
# SLIDE 1: Title + Problem
# ============================================================
fig, ax = setup_slide('Automatic Lens Correction', 'AutoHDR Hackathon Challenge')

# Left: Distorted
rect1 = FancyBboxPatch((1, 2.5), 5.5, 4, boxstyle="round,pad=0.15",
                         facecolor=CARD_BG, edgecolor=ACCENT_RED, linewidth=2)
ax.add_patch(rect1)
ax.text(3.75, 6, 'Distorted Input', fontsize=15, ha='center', fontweight='bold', color=ACCENT_RED)

# Draw curved grid lines to show barrel distortion
for i, y_base in enumerate(np.linspace(3.2, 5.5, 5)):
    x = np.linspace(1.5, 6, 60)
    amp = 0.25 * (1 - abs(y_base - 4.35) / 1.5)
    curve = amp * np.sin(np.pi * (x - 1.5) / 4.5)
    ax.plot(x, y_base + curve, color=ACCENT_RED, linewidth=1.5, alpha=0.6)

for x_base in np.linspace(2, 5.5, 5):
    y = np.linspace(3, 5.8, 60)
    amp = 0.2 * (1 - abs(x_base - 3.75) / 2.0)
    curve = amp * np.sin(np.pi * (y - 3) / 2.8)
    ax.plot(x_base + curve, y, color=ACCENT_RED, linewidth=1.5, alpha=0.6)

# Arrow
ax.annotate('', xy=(9, 4.5), xytext=(7, 4.5),
            arrowprops=dict(arrowstyle='->', color=ACCENT_BLUE, lw=3))
ax.text(8, 5.2, 'Our Model', fontsize=14, ha='center', fontweight='bold',
        color=ACCENT_BLUE)

# Right: Corrected
rect2 = FancyBboxPatch((9.5, 2.5), 5.5, 4, boxstyle="round,pad=0.15",
                         facecolor=CARD_BG, edgecolor=ACCENT_GREEN, linewidth=2)
ax.add_patch(rect2)
ax.text(12.25, 6, 'Corrected Output', fontsize=15, ha='center', fontweight='bold', color=ACCENT_GREEN)

# Draw straight grid lines
for y_base in np.linspace(3.2, 5.5, 5):
    ax.plot([10, 14.5], [y_base, y_base], color=ACCENT_GREEN, linewidth=1.5, alpha=0.6)
for x_base in np.linspace(10.5, 14, 5):
    ax.plot([x_base, x_base], [3, 5.8], color=ACCENT_GREEN, linewidth=1.5, alpha=0.6)

# Bottom text
ax.text(8, 1.8, 'Scoring: 80% Geometry-Focused', fontsize=14, ha='center',
        fontweight='bold', color=ACCENT_ORANGE)
ax.text(8, 1.1, 'Edge Similarity (40%)  |  Line Straightness (22%)  |  Gradient Orientation (18%)  |  SSIM (15%)',
        fontsize=10, ha='center', color=TEXT_SECONDARY)

# Author
ax.text(8, 0.4, 'Sai Teja', fontsize=12, ha='center', color=TEXT_SECONDARY)

save_slide(fig, 'problem', 1)


# ============================================================
# SLIDE 2: Key Insight — Two Numbers
# ============================================================
fig, ax = setup_slide('The Key Insight', 'All lens distortion is described by just 2 numbers')

# Formula box (center)
formula_rect = FancyBboxPatch((2, 4.5), 12, 2, boxstyle="round,pad=0.2",
                                facecolor='#1a2332', edgecolor=ACCENT_BLUE, linewidth=2.5)
ax.add_patch(formula_rect)
ax.text(8, 5.9, 'Brown-Conrady Radial Distortion Model', fontsize=14,
        ha='center', fontweight='bold', color=ACCENT_BLUE)
ax.text(8, 5.0, r'$r_{distorted}$  =  r  ×  ( 1  +  k₁ · r²  +  k₂ · r⁴ )',
        fontsize=20, ha='center', va='center', color=TEXT_PRIMARY, fontweight='bold')

# Three examples below
examples = [
    (2.5, 'k₁ < 0', 'Barrel\nDistortion', ACCENT_RED, 'barrel'),
    (7.5, 'k₁ = 0', 'No\nDistortion', TEXT_SECONDARY, 'none'),
    (12.5, 'k₁ > 0', 'Pincushion\nDistortion', ACCENT_PURPLE, 'pincushion'),
]

for cx, label, desc, color, dtype in examples:
    rect = FancyBboxPatch((cx - 1.8, 1.0), 3.6, 2.8, boxstyle="round,pad=0.12",
                           facecolor=CARD_BG, edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(cx, 3.5, label, fontsize=14, ha='center', fontweight='bold', color=color)

    # Draw distortion pattern
    center_y = 2.2
    for line_y in np.linspace(center_y - 0.6, center_y + 0.6, 4):
        x = np.linspace(cx - 1.3, cx + 1.3, 40)
        if dtype == 'barrel':
            amp = 0.15 * (1 - abs(line_y - center_y) / 0.8)
            curve = amp * np.sin(np.pi * (x - (cx - 1.3)) / 2.6)
            ax.plot(x, line_y + curve, color=color, linewidth=1.2, alpha=0.7)
        elif dtype == 'pincushion':
            amp = 0.15 * (1 - abs(line_y - center_y) / 0.8)
            curve = -amp * np.sin(np.pi * (x - (cx - 1.3)) / 2.6)
            ax.plot(x, line_y + curve, color=color, linewidth=1.2, alpha=0.7)
        else:
            ax.plot(x, [line_y]*40, color=color, linewidth=1.2, alpha=0.7)

    ax.text(cx, 1.2, desc, fontsize=10, ha='center', va='center', color=TEXT_SECONDARY)

# Arrow annotation
ax.text(8, 0.3, 'If you know k₁ and k₂ → OpenCV cv2.undistort() reverses it perfectly',
        fontsize=13, ha='center', color=ACCENT_GREEN, fontweight='bold')

save_slide(fig, 'insight', 2)


# ============================================================
# SLIDE 3: Pipeline Architecture
# ============================================================
fig, ax = setup_slide('End-to-End Pipeline')

# Pipeline boxes
pipeline = [
    (0.5, 4, 2.8, 2.2, 'Input\nImage', '256×256 RGB', ACCENT_RED),
    (4, 4, 3, 2.2, 'ResNet50\nBackbone', 'Pre-trained ImageNet\n24.6M parameters', ACCENT_BLUE),
    (7.8, 4, 2.8, 2.2, 'Regression\nHead', 'FC layers\n→ Tanh × 0.5', ACCENT_ORANGE),
    (11.4, 4, 3, 2.2, 'Differentiable\nUndistort', 'F.grid_sample()\nRadial model', ACCENT_GREEN),
]

for x, y, w, h, title, body, color in pipeline:
    add_box(ax, x, y, w, h, title, body, color, title_size=14, body_size=10)

# Arrows between pipeline boxes
arrow_pairs = [(3.3, 4), (7, 7.8), (10.6, 11.4)]
for x_end, x_start in arrow_pairs:
    ax.annotate('', xy=(x_start, 5.1), xytext=(x_end, 5.1),
                arrowprops=dict(arrowstyle='->', color=TEXT_SECONDARY, lw=2.5))

# Output labels
ax.text(10.2, 4.5, 'k₁, k₂', fontsize=16, ha='center', fontweight='bold', color=ACCENT_ORANGE)

# Output arrow
ax.annotate('', xy=(15.5, 5.1), xytext=(14.4, 5.1),
            arrowprops=dict(arrowstyle='->', color=ACCENT_GREEN, lw=2.5))
ax.text(15.6, 5.1, '✓', fontsize=24, ha='center', va='center', color=ACCENT_GREEN)

# Bottom: Training loop explanation
loop_rect = FancyBboxPatch((1, 1, ), 14, 2.2, boxstyle="round,pad=0.15",
                             facecolor=CARD_BG, edgecolor=ACCENT_PURPLE, linewidth=2)
ax.add_patch(loop_rect)
ax.text(8, 2.8, 'End-to-End Training: Gradients flow through the undistortion',
        fontsize=14, ha='center', fontweight='bold', color=ACCENT_PURPLE)

ax.text(4, 1.8, 'Loss = Charbonnier(pixel)', fontsize=11,
        ha='center', color=ACCENT_BLUE, fontweight='bold')
ax.text(8, 1.8, '+ SSIM(structure)', fontsize=11,
        ha='center', color=ACCENT_GREEN, fontweight='bold')
ax.text(12, 1.8, '+ Line Straightness(geometry)', fontsize=11,
        ha='center', color=ACCENT_ORANGE, fontweight='bold')

ax.text(8, 1.1, 'Compare corrected output vs ground truth → backpropagate → update ResNet50 weights',
        fontsize=10, ha='center', color=TEXT_SECONDARY)

save_slide(fig, 'pipeline', 3)


# ============================================================
# SLIDE 4: Training Details
# ============================================================
fig, ax = setup_slide('Training', 'RTX 4090  |  23,120 image pairs  |  50 epochs')

# Left: Loss scheduling chart
chart_ax = fig.add_axes([0.06, 0.15, 0.42, 0.55])
chart_ax.set_facecolor(CARD_BG)

epochs = np.arange(1, 51)
char_w = np.ones(50)
ssim_w = np.ones(50) * 0.1
line_w = np.zeros(50)
line_w[10:25] = 0.05
line_w[25:] = 0.1

chart_ax.fill_between(epochs, 0, line_w, alpha=0.4, color=ACCENT_ORANGE)
chart_ax.plot(epochs, line_w, color=ACCENT_ORANGE, linewidth=2.5, label='Line Straightness')
chart_ax.fill_between(epochs, 0, ssim_w, alpha=0.3, color=ACCENT_GREEN)
chart_ax.plot(epochs, ssim_w, color=ACCENT_GREEN, linewidth=2, label='SSIM')

chart_ax.axvline(x=10, color=TEXT_SECONDARY, linestyle='--', alpha=0.4)
chart_ax.axvline(x=25, color=TEXT_SECONDARY, linestyle='--', alpha=0.4)

chart_ax.text(5, 0.11, 'Phase 1\nPixel Only', fontsize=9, ha='center',
              color=TEXT_SECONDARY, fontweight='bold')
chart_ax.text(17.5, 0.11, 'Phase 2\n+Geometry', fontsize=9, ha='center',
              color=TEXT_SECONDARY, fontweight='bold')
chart_ax.text(37.5, 0.11, 'Phase 3\nFull Geo', fontsize=9, ha='center',
              color=TEXT_SECONDARY, fontweight='bold')

chart_ax.set_xlabel('Epoch', fontsize=10, color=TEXT_SECONDARY)
chart_ax.set_ylabel('Loss Weight', fontsize=10, color=TEXT_SECONDARY)
chart_ax.set_title('Loss Scheduling Strategy', fontsize=13,
                    fontweight='bold', color=TEXT_PRIMARY, pad=10)
chart_ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=BORDER,
                labelcolor=TEXT_SECONDARY)
chart_ax.set_xlim(1, 50)
chart_ax.set_ylim(0, 0.15)
chart_ax.tick_params(colors=TEXT_SECONDARY)
for spine in chart_ax.spines.values():
    spine.set_color(BORDER)
chart_ax.grid(True, alpha=0.15, color=TEXT_SECONDARY)

# Right: Training stats
stats_x = 9.5

add_box(ax, 8.5, 5.0, 6.5, 1.8, 'Model Architecture', None, ACCENT_BLUE)
ax.text(11.75, 5.6, 'ResNet50 (pre-trained) → FC(2048→512→128→2) → Tanh×0.5',
        fontsize=10, ha='center', color=TEXT_SECONDARY, family='monospace')

add_box(ax, 8.5, 2.8, 6.5, 1.8, 'Training Config', None, ACCENT_GREEN)
stats_text = ('Optimizer: AdamW (lr=3e-4)\n'
              'Scheduler: Cosine Annealing\n'
              'Batch: 24  |  Resolution: 256×256')
ax.text(11.75, 3.35, stats_text, fontsize=10, ha='center', color=TEXT_SECONDARY,
        family='monospace', linespacing=1.5)

add_box(ax, 8.5, 0.6, 6.5, 1.8, 'Key Result', None, ACCENT_ORANGE)
ax.text(11.75, 1.15, 'Best val loss: 0.019 (epoch 10)\nk₁ std=0.044  k₂ std=0.168  (diverse predictions!)',
        fontsize=10, ha='center', color=TEXT_SECONDARY, family='monospace', linespacing=1.5)

save_slide(fig, 'training', 4)
ax.remove()


# ============================================================
# SLIDE 5: Results + Closing
# ============================================================
fig, ax = setup_slide('Results & Deployment', 'Fast, accurate, and production-ready')

# Inference speed
add_box(ax, 0.5, 4.5, 4.5, 2.3, 'Inference Speed', None, ACCENT_GREEN, title_size=15)
ax.text(2.75, 5.3, '~5ms', fontsize=36, ha='center', fontweight='bold', color=ACCENT_GREEN)
ax.text(2.75, 4.8, 'per image (CNN prediction)', fontsize=10, ha='center', color=TEXT_SECONDARY)

# Approach
add_box(ax, 5.5, 4.5, 5, 2.3, 'Why This Works', None, ACCENT_BLUE, title_size=15)
points = ('✓ Physics-based correction\n'
          '✓ Mathematically exact undistort\n'
          '✓ No pixel hallucination\n'
          '✓ Geometry-preserving')
ax.text(8, 5.15, points, fontsize=11, ha='center', va='center',
        color=TEXT_SECONDARY, linespacing=1.8)

# Scale
add_box(ax, 11, 4.5, 4.5, 2.3, 'Production Scale', None, ACCENT_ORANGE, title_size=15)
ax.text(13.25, 5.3, '93K', fontsize=36, ha='center', fontweight='bold', color=ACCENT_ORANGE)
ax.text(13.25, 4.8, 'photos/day at AutoHDR', fontsize=10, ha='center', color=TEXT_SECONDARY)

# How it works summary
summary_rect = FancyBboxPatch((1, 1.5), 14, 2.3, boxstyle="round,pad=0.2",
                                facecolor=CARD_BG, edgecolor=ACCENT_PURPLE, linewidth=2)
ax.add_patch(summary_rect)
ax.text(8, 3.3, 'The Elegant Solution', fontsize=16, ha='center', fontweight='bold',
        color=ACCENT_PURPLE)

# Mini pipeline in summary
mini_steps = [
    (2, 'Distorted\nImage', ACCENT_RED),
    (5.2, 'ResNet50\nPredict k₁,k₂', ACCENT_BLUE),
    (8.8, 'cv2.undistort()\nMath correction', ACCENT_GREEN),
    (12.5, 'Corrected\nImage', ACCENT_GREEN),
]
for x, text, color in mini_steps:
    mini_rect = FancyBboxPatch((x, 1.7), 2.5, 1.3, boxstyle="round,pad=0.1",
                                facecolor=BG, edgecolor=color, linewidth=1.5)
    ax.add_patch(mini_rect)
    ax.text(x + 1.25, 2.35, text, fontsize=10, ha='center', va='center',
            fontweight='bold', color=color)

for i in range(3):
    x_start = mini_steps[i][0] + 2.5
    x_end = mini_steps[i+1][0]
    ax.annotate('', xy=(x_end, 2.35), xytext=(x_start, 2.35),
                arrowprops=dict(arrowstyle='->', color=TEXT_SECONDARY, lw=2))

# Thank you
ax.text(8, 0.5, 'Thank you!  —  Sai Teja', fontsize=16, ha='center',
        fontweight='bold', color=TEXT_PRIMARY)

save_slide(fig, 'results', 5)


print(f'\n✅ All video slides saved to: {OUTPUT_DIR}/')
print('Files:')
for i in range(1, 6):
    print(f'  slide_{i:02d}_*.png')
