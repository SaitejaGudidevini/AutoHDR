"""
Classical Hough Line Optimization for Lens Distortion Correction
No neural network — optimizes k1, k2 per-image to maximize line straightness.
"""

import os
import sys
import glob
import numpy as np
import cv2
from scipy.optimize import minimize
from tqdm import tqdm
import zipfile
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def detect_lines(gray, resize_dim=512):
    """Detect lines using Canny + HoughLinesP."""
    h, w = gray.shape
    scale = resize_dim / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))

    blurred = cv2.GaussianBlur(small, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=30, maxLineGap=10)
    return lines, small.shape[:2], scale


def line_straightness_score(lines):
    """
    Score how well lines align to horizontal/vertical.
    Higher = straighter = better correction.
    """
    if lines is None or len(lines) < 3:
        return 0.0

    angles = []
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angle = angle % 180  # Normalize to [0, 180)
        angles.append(angle)
        lengths.append(length)

    angles = np.array(angles)
    lengths = np.array(lengths)

    # Weight by line length — longer lines matter more
    weights = lengths / lengths.sum()

    # Score: how close to 0, 90, or 180 degrees (h/v alignment)
    # Use distance to nearest cardinal direction
    dist_to_h = np.minimum(angles, 180 - angles)  # distance to 0/180
    dist_to_v = np.abs(angles - 90)  # distance to 90
    dist_to_cardinal = np.minimum(dist_to_h, dist_to_v)

    # Convert to score: 0 degrees off = 1.0, 45 degrees off = 0.0
    alignment = np.clip(1.0 - dist_to_cardinal / 45.0, 0, 1)
    score = np.sum(alignment * weights)

    return score


def apply_undistort(gray, k1, k2, h, w):
    """Apply undistortion with given k1, k2."""
    fx = fy = float(w)
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    return cv2.undistort(gray, camera_matrix, dist_coeffs)


def optimize_single_image(image_path, resize_dim=512):
    """
    Find optimal k1, k2 for a single image using Hough line optimization.
    Returns (k1, k2, score).
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, 0.0, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_orig, w_orig = gray.shape

    # Resize for speed
    scale = resize_dim / max(h_orig, w_orig)
    h_small = int(h_orig * scale)
    w_small = int(w_orig * scale)
    gray_small = cv2.resize(gray, (w_small, h_small))

    # First check: does the original already have good lines?
    lines_orig, _, _ = detect_lines(gray, resize_dim)
    score_orig = line_straightness_score(lines_orig)

    # Coarse grid search
    best_score = score_orig
    best_k1, best_k2 = 0.0, 0.0

    for k1 in np.arange(-0.4, 0.41, 0.05):
        for k2 in np.arange(-0.6, 0.61, 0.1):
            if k1 == 0.0 and k2 == 0.0:
                continue

            undist = apply_undistort(gray_small, k1, k2, h_small, w_small)
            lines = cv2.HoughLinesP(
                cv2.Canny(cv2.GaussianBlur(undist, (5, 5), 1.5), 50, 150),
                1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10
            )
            score = line_straightness_score(lines)

            if score > best_score:
                best_score = score
                best_k1, best_k2 = k1, k2

    # If no improvement found, return original (no correction)
    if best_k1 == 0.0 and best_k2 == 0.0:
        return 0.0, 0.0, score_orig

    # Fine optimization around best grid point
    def neg_score(params):
        k1, k2 = params
        undist = apply_undistort(gray_small, k1, k2, h_small, w_small)
        lines = cv2.HoughLinesP(
            cv2.Canny(cv2.GaussianBlur(undist, (5, 5), 1.5), 50, 150),
            1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10
        )
        return -line_straightness_score(lines)

    result = minimize(neg_score, [best_k1, best_k2],
                      method='Nelder-Mead',
                      options={'xatol': 0.005, 'fatol': 1e-5, 'maxiter': 100})

    k1_opt, k2_opt = result.x
    final_score = -result.fun

    # Safety check: if optimized correction barely beats original, don't correct
    if final_score < score_orig + 0.02:
        return 0.0, 0.0, score_orig

    return k1_opt, k2_opt, final_score


def correct_image(image_path, k1, k2, output_path):
    """Apply correction and save."""
    img = cv2.imread(image_path)
    if img is None:
        return False

    if k1 == 0.0 and k2 == 0.0:
        # No correction needed — copy original
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True

    h, w = img.shape[:2]
    fx = fy = float(w)
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )

    corrected = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop to valid region if ROI is valid, then resize back
    if roi != (0, 0, 0, 0):
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            corrected = corrected[y:y + rh, x:x + rw]
            corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(output_path, corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def process_one(args):
    """Worker function for parallel processing."""
    image_path, output_path = args
    filename = os.path.basename(image_path)
    try:
        k1, k2, score = optimize_single_image(image_path)
        correct_image(image_path, k1, k2, output_path)
        return filename, k1, k2, score, True
    except Exception as e:
        # On failure, copy original
        try:
            img = cv2.imread(image_path)
            if img is not None:
                cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except:
            pass
        return filename, 0.0, 0.0, 0.0, False


def main():
    PROJECT_DIR = '/Users/saiteja/Documents/Dev/AutoHDR'
    TEST_DIR = os.path.join(PROJECT_DIR, 'test-originals')
    OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'corrected_classical')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f'Found {len(test_files)} test images')

    # Prepare args
    args_list = []
    for f in test_files:
        args_list.append((
            os.path.join(TEST_DIR, f),
            os.path.join(OUTPUT_DIR, f)
        ))

    # Process with multiprocessing
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f'Using {n_workers} workers')

    results = []
    corrected_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_one, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures)):
            filename, k1, k2, score, success = future.result()
            results.append({
                'filename': filename,
                'k1': k1, 'k2': k2,
                'line_score': score,
                'corrected': k1 != 0.0 or k2 != 0.0
            })
            if k1 != 0.0 or k2 != 0.0:
                corrected_count += 1
            else:
                skipped_count += 1

    print(f'\nDone! Corrected: {corrected_count}, Kept original: {skipped_count}')

    # Save predictions
    pred_path = os.path.join(PROJECT_DIR, 'outputs', 'predictions_classical.csv')
    with open(pred_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'k1', 'k2', 'line_score', 'corrected'])
        writer.writeheader()
        for r in sorted(results, key=lambda x: x['filename']):
            writer.writerow(r)
    print(f'Saved predictions to {pred_path}')

    # Create zip
    zip_path = os.path.join(PROJECT_DIR, 'outputs', 'corrected_classical.zip')
    corrected_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*.jpg')))
    print(f'Zipping {len(corrected_files)} files...')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in corrected_files:
            zf.write(f, os.path.basename(f))

    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f'Created {zip_path} ({zip_mb:.1f} MB)')

    # Stats
    k1_vals = [r['k1'] for r in results if r['corrected']]
    k2_vals = [r['k2'] for r in results if r['corrected']]
    if k1_vals:
        print(f'\nCorrected image stats:')
        print(f'  k1: mean={np.mean(k1_vals):.4f}, std={np.std(k1_vals):.4f}, range=[{np.min(k1_vals):.4f}, {np.max(k1_vals):.4f}]')
        print(f'  k2: mean={np.mean(k2_vals):.4f}, std={np.std(k2_vals):.4f}, range=[{np.min(k2_vals):.4f}, {np.max(k2_vals):.4f}]')


if __name__ == '__main__':
    main()
