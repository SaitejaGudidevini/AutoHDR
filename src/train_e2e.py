"""
End-to-End Differentiable Lens Correction
ResNet50 → k1, k2 → Differentiable Undistort → Loss(corrected, ground_truth)

Loss = Charbonnier + SSIM + Line Straightness (scheduled)
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.amp import autocast, GradScaler
from PIL import Image
import pandas as pd
from tqdm import tqdm
import time
import glob
import zipfile
from collections import defaultdict

# ============================================================
# Differentiable Undistortion (no kornia dependency)
# ============================================================

def build_radial_undistort_grid(k1, k2, h, w, fx=None, fy=None, cx=None, cy=None):
    """
    Build a sampling grid that undoes radial distortion.
    All operations are differentiable through k1, k2.

    Returns grid in [-1, 1] format for F.grid_sample()
    """
    if fx is None:
        fx = float(w)
    if fy is None:
        fy = float(w)
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    # Create normalized pixel coordinates
    y_coords = torch.linspace(0, h - 1, h, device=k1.device)
    x_coords = torch.linspace(0, w - 1, w, device=k1.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Normalize to camera coordinates
    x_norm = (xx - cx) / fx  # [H, W]
    y_norm = (yy - cy) / fy  # [H, W]

    # Radial distance squared
    r2 = x_norm ** 2 + y_norm ** 2  # [H, W]

    # Radial distortion factor: 1 + k1*r^2 + k2*r^4
    # k1, k2 are [B] tensors
    batch_size = k1.shape[0]
    r2 = r2.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]

    k1 = k1.view(-1, 1, 1)  # [B, 1, 1]
    k2 = k2.view(-1, 1, 1)  # [B, 1, 1]

    radial = 1.0 + k1 * r2 + k2 * r2 * r2  # [B, H, W]

    # Distorted coordinates (where to sample FROM in the input)
    x_dist = x_norm.unsqueeze(0) * radial  # [B, H, W]
    y_dist = y_norm.unsqueeze(0) * radial  # [B, H, W]

    # Convert back to pixel coordinates, then to [-1, 1] for grid_sample
    x_pixel = x_dist * fx + cx
    y_pixel = y_dist * fy + cy

    # Normalize to [-1, 1]
    x_grid = 2.0 * x_pixel / (w - 1) - 1.0
    y_grid = 2.0 * y_pixel / (h - 1) - 1.0

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]
    return grid


def differentiable_undistort(images, k1, k2):
    """
    Apply radial undistortion to a batch of images.
    images: [B, C, H, W]
    k1, k2: [B] tensors
    Returns: [B, C, H, W] undistorted images
    """
    B, C, H, W = images.shape
    grid = build_radial_undistort_grid(k1, k2, H, W)
    undistorted = F.grid_sample(images, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return undistorted


# ============================================================
# Loss Functions
# ============================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.epsilon * self.epsilon))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        mu_pred = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.window_size // 2)
        mu_target = F.avg_pool2d(target, self.window_size, stride=1, padding=self.window_size // 2)

        mu_pred_sq = mu_pred * mu_pred
        mu_target_sq = mu_target * mu_target
        mu_cross = mu_pred * mu_target

        sigma_pred = F.avg_pool2d(pred * pred, self.window_size, stride=1, padding=self.window_size // 2) - mu_pred_sq
        sigma_target = F.avg_pool2d(target * target, self.window_size, stride=1, padding=self.window_size // 2) - mu_target_sq
        sigma_cross = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=self.window_size // 2) - mu_cross

        ssim = ((2 * mu_cross + self.C1) * (2 * sigma_cross + self.C2)) / \
               ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred + sigma_target + self.C2))

        return 1.0 - ssim.mean()


class LineStraightnessLoss(nn.Module):
    """
    Differentiable approximation of line straightness.
    Uses Sobel gradients to measure edge orientation consistency.
    In well-corrected images, edges should align to horizontal/vertical.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        """
        Compare gradient orientation distributions between pred and target.
        Both should have similar edge structures if correction is right.
        """
        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        # Compute gradients
        gx_pred = F.conv2d(pred_gray, self.sobel_x, padding=1)
        gy_pred = F.conv2d(pred_gray, self.sobel_y, padding=1)
        gx_target = F.conv2d(target_gray, self.sobel_x, padding=1)
        gy_target = F.conv2d(target_gray, self.sobel_y, padding=1)

        # Gradient magnitude (for weighting — focus on strong edges)
        mag_pred = torch.sqrt(gx_pred ** 2 + gy_pred ** 2 + 1e-6)
        mag_target = torch.sqrt(gx_target ** 2 + gy_target ** 2 + 1e-6)

        # Gradient orientation (using atan2)
        angle_pred = torch.atan2(gy_pred, gx_pred)
        angle_target = torch.atan2(gy_target, gx_target)

        # Angular difference (handle wraparound)
        angle_diff = angle_pred - angle_target
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

        # Weight by edge magnitude (both pred and target should have strong edges)
        weight = torch.min(mag_pred, mag_target)
        weight = weight / (weight.sum(dim=[2, 3], keepdim=True) + 1e-6)

        # Weighted angular difference
        loss = (weight * angle_diff.abs()).sum(dim=[2, 3]).mean()

        return loss


# ============================================================
# Model
# ============================================================

class DistortionRegressorE2E(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)

        # Try loading pretrained weights
        weight_paths = [
            '/workspace/resnet50.pth',
            '/runpod-volume/resnet50.pth',
        ]
        for wp in weight_paths:
            if os.path.exists(wp):
                self.backbone.load_state_dict(torch.load(wp, weights_only=True, map_location='cpu'))
                print(f'Loaded pretrained weights from {wp}')
                break
        else:
            # Download if available
            try:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                print('Downloaded pretrained weights')
            except:
                print('No pretrained weights, using random init')

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Tanh(),  # Clamp output to [-1, 1]
        )
        self.k_scale = 0.5  # Scale tanh output to [-0.5, 0.5]

    def forward(self, x):
        params = self.backbone(x)
        return params * self.k_scale  # [B, 2] = [k1, k2]


# ============================================================
# Dataset
# ============================================================

class PairedDataset(Dataset):
    """Loads paired (distorted, corrected) images directly."""

    def __init__(self, pairs, img_size=384):
        self.pairs = pairs  # list of (original_path, generated_path)
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig_path, gen_path = self.pairs[idx]

        orig = Image.open(orig_path).convert('RGB')
        gen = Image.open(gen_path).convert('RGB')

        orig_tensor = self.transform(orig)
        gen_tensor = self.transform(gen)

        # Random horizontal flip (apply same flip to both)
        if torch.rand(1) > 0.5:
            orig_tensor = torch.flip(orig_tensor, [2])
            gen_tensor = torch.flip(gen_tensor, [2])

        return orig_tensor, gen_tensor


# ============================================================
# Training
# ============================================================

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Find data
    TRAIN_DIR = '/workspace/train-data'
    if not os.path.exists(TRAIN_DIR):
        print(f'ERROR: {TRAIN_DIR} not found')
        sys.exit(1)

    # Parse pairs
    train_files = sorted(os.listdir(TRAIN_DIR))
    pairs_dict = defaultdict(dict)
    for f in train_files:
        if not f.endswith('.jpg'):
            continue
        parts = f.rsplit('_', 1)
        if len(parts) == 2:
            image_id = parts[0]
            img_type = parts[1].replace('.jpg', '')
            pairs_dict[image_id][img_type] = os.path.join(TRAIN_DIR, f)

    complete = [(v['original'], v['generated'])
                for v in pairs_dict.values()
                if 'original' in v and 'generated' in v]

    print(f'Training pairs: {len(complete)}')

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(complete))
    split = int(len(complete) * 0.95)
    train_pairs = [complete[i] for i in indices[:split]]
    val_pairs = [complete[i] for i in indices[split:]]

    print(f'Train: {len(train_pairs)} | Val: {len(val_pairs)}')

    IMG_SIZE = 256  # Smaller for speed with differentiable undistort
    BATCH_SIZE = 24

    train_dataset = PairedDataset(train_pairs, img_size=IMG_SIZE)
    val_dataset = PairedDataset(val_pairs, img_size=IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    # Model
    model = DistortionRegressorE2E().to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Losses
    charbonnier = CharbonnierLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    line_loss = LineStraightnessLoss().to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = GradScaler('cuda')

    EPOCHS = 50
    best_val_loss = float('inf')
    save_dir = '/workspace/checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        # Loss scheduling
        lambda_char = 1.0
        lambda_ssim = 0.1
        if epoch < 10:
            lambda_line = 0.0  # Pure pixel loss first
        elif epoch < 25:
            lambda_line = 0.05  # Gradually add geometry
        else:
            lambda_line = 0.1  # Full geometry weight

        # Train
        model.train()
        train_losses = []
        train_k1s = []
        train_k2s = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for orig, gen in pbar:
            orig = orig.to(device)
            gen = gen.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                # Predict k1, k2
                params = model(orig)
                k1 = params[:, 0]
                k2 = params[:, 1]

                # Differentiable undistort
                corrected = differentiable_undistort(orig, k1, k2)

                # Losses
                l_char = charbonnier(corrected, gen)
                l_ssim = ssim_loss(corrected, gen)
                l_line = line_loss(corrected, gen) if lambda_line > 0 else torch.tensor(0.0, device=device)

                loss = lambda_char * l_char + lambda_ssim * l_ssim + lambda_line * l_line

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_k1s.extend(k1.detach().cpu().numpy().tolist())
            train_k2s.extend(k2.detach().cpu().numpy().tolist())

            pbar.set_postfix(loss=f'{loss.item():.4f}', k1=f'{k1.mean().item():.3f}', k2=f'{k2.mean().item():.3f}')

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        val_k1s = []
        val_k2s = []

        with torch.no_grad():
            for orig, gen in val_loader:
                orig = orig.to(device)
                gen = gen.to(device)

                params = model(orig)
                k1 = params[:, 0]
                k2 = params[:, 1]

                corrected = differentiable_undistort(orig, k1, k2)

                l_char = charbonnier(corrected, gen)
                l_ssim = ssim_loss(corrected, gen)
                l_line = line_loss(corrected, gen) if lambda_line > 0 else torch.tensor(0.0, device=device)

                loss = lambda_char * l_char + lambda_ssim * l_ssim + lambda_line * l_line
                val_losses.append(loss.item())
                val_k1s.extend(k1.cpu().numpy().tolist())
                val_k2s.extend(k2.cpu().numpy().tolist())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        k1_std = np.std(train_k1s)
        k2_std = np.std(train_k2s)
        lr = optimizer.param_groups[0]['lr']

        improved = ''
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            improved = ' ** BEST **'

        print(f'Epoch {epoch+1:2d}/{EPOCHS} | '
              f'Train: {avg_train:.4f} | Val: {avg_val:.4f} | '
              f'k1 std: {k1_std:.4f} | k2 std: {k2_std:.4f} | '
              f'line_w: {lambda_line} | LR: {lr:.1e}{improved}')

    print(f'\nTraining done. Best val loss: {best_val_loss:.4f}')
    return os.path.join(save_dir, 'best_model.pth')


# ============================================================
# Inference with Hough Refinement
# ============================================================

def hough_straightness(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (512, 384))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.5), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)

    if lines is None or len(lines) < 3:
        return 0.0

    angles = []
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi % 180
        angles.append(angle)
        lengths.append(length)

    angles = np.array(angles)
    lengths = np.array(lengths)
    weights = lengths / lengths.sum()

    dist_h = np.minimum(angles, 180 - angles)
    dist_v = np.abs(angles - 90)
    dist_cardinal = np.minimum(dist_h, dist_v)
    alignment = np.clip(1.0 - dist_cardinal / 45.0, 0, 1)

    return np.sum(alignment * weights)


def predict_and_correct(model_path, test_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DistortionRegressorE2E().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    os.makedirs(output_dir, exist_ok=True)
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f'Correcting {len(test_files)} test images...')

    results = []
    for f in tqdm(test_files):
        img_path = os.path.join(test_dir, f)
        img_full = cv2.imread(img_path)
        if img_full is None:
            continue

        h, w = img_full.shape[:2]

        # CNN prediction
        img_rgb = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            params = model(img_tensor).cpu().numpy()[0]

        k1_cnn, k2_cnn = float(params[0]), float(params[1])

        # Hough refinement: search around CNN prediction
        best_k1, best_k2 = k1_cnn, k2_cnn
        best_score = -1

        img_small = cv2.resize(img_full, (512, 384))
        h_s, w_s = img_small.shape[:2]
        cam = np.array([[w_s, 0, w_s/2], [0, w_s, h_s/2], [0, 0, 1]], dtype=np.float64)

        # Also test k1=0, k2=0 (no correction)
        score_orig = hough_straightness(img_small)
        best_score = score_orig
        best_k1, best_k2 = 0.0, 0.0

        for dk1 in np.arange(-0.15, 0.16, 0.03):
            for dk2 in np.arange(-0.3, 0.31, 0.05):
                k1_try = k1_cnn + dk1
                k2_try = k2_cnn + dk2

                dist = np.array([k1_try, k2_try, 0, 0, 0], dtype=np.float64)
                undist = cv2.undistort(img_small, cam, dist)
                score = hough_straightness(undist)

                if score > best_score:
                    best_score = score
                    best_k1, best_k2 = k1_try, k2_try

        # Apply correction at full resolution
        if best_k1 == 0.0 and best_k2 == 0.0:
            # No correction beats all options — keep original
            corrected = img_full
        else:
            camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.array([best_k1, best_k2, 0, 0, 0], dtype=np.float64)
            new_cam, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha=0.3)
            corrected = cv2.undistort(img_full, camera_matrix, dist_coeffs, None, new_cam)

            if roi != (0, 0, 0, 0):
                x, y, rw, rh = roi
                if rw > 0 and rh > 0:
                    corrected = corrected[y:y+rh, x:x+rw]
                    corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)

        out_path = os.path.join(output_dir, f)
        cv2.imwrite(out_path, corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])

        results.append({
            'filename': f,
            'k1_cnn': k1_cnn, 'k2_cnn': k2_cnn,
            'k1_final': best_k1, 'k2_final': best_k2,
            'line_score': best_score,
            'corrected': not (best_k1 == 0.0 and best_k2 == 0.0)
        })

    # Save predictions
    pd.DataFrame(results).to_csv(os.path.join(output_dir, '..', 'predictions_e2e.csv'), index=False)

    # Stats
    corrected_r = [r for r in results if r['corrected']]
    print(f'\nCorrected: {len(corrected_r)}/{len(results)}')
    if corrected_r:
        k1s = [r['k1_final'] for r in corrected_r]
        k2s = [r['k2_final'] for r in corrected_r]
        print(f'k1: mean={np.mean(k1s):.4f}, std={np.std(k1s):.4f}, range=[{np.min(k1s):.4f}, {np.max(k1s):.4f}]')
        print(f'k2: mean={np.mean(k2s):.4f}, std={np.std(k2s):.4f}, range=[{np.min(k2s):.4f}, {np.max(k2s):.4f}]')

    # Create zip
    zip_path = os.path.join(output_dir, '..', 'corrected_e2e.zip')
    files = sorted(glob.glob(os.path.join(output_dir, '*.jpg')))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, os.path.basename(f))

    print(f'Created {zip_path} ({os.path.getsize(zip_path) / 1024 / 1024:.1f} MB)')
    return zip_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='both')
    parser.add_argument('--model', default='/workspace/checkpoints/best_model.pth')
    parser.add_argument('--test-dir', default='/workspace/test-data')
    parser.add_argument('--output-dir', default='/workspace/output/corrected')
    args = parser.parse_args()

    if args.mode in ('train', 'both'):
        model_path = train()
    else:
        model_path = args.model

    if args.mode in ('predict', 'both'):
        predict_and_correct(model_path, args.test_dir, args.output_dir)
