# =========================================================
# Full CLAHE + Metrics Evaluation
# =========================================================

import os
import cv2
import numpy as np
from glob import glob
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt

# -----------------------------
# CLAHE Enhancement Function (Grayscale)
# -----------------------------
def clahe_enhancement(image_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image.

    Parameters:
    - image_gray: input grayscale image (numpy array 0-255)
    - clip_limit: threshold for contrast limiting
    - tile_grid_size: size of grid for histogram equalization

    Returns:
    - enhanced image (uint8)
    """
    img = image_gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img)
    return enhanced

# -----------------------------
# Metric Functions
# -----------------------------
def psnr_metric(orig, proc):
    mse_val = np.mean((orig.astype(np.float32) - proc.astype(np.float32)) ** 2)
    if mse_val == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse_val))

def mse_metric(orig, proc):
    return np.mean((orig.astype(np.float32) - proc.astype(np.float32)) ** 2)

def ssim_metric(orig, proc):
    return ssim(orig, proc, data_range=255)

def epi_metric(orig_gray, proc_gray):
    grad_orig = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 1)
    grad_proc = cv2.Sobel(proc_gray, cv2.CV_64F, 1, 1)
    numerator = np.sum(grad_orig * grad_proc)
    denominator = np.sqrt(np.sum(grad_orig**2) * np.sum(grad_proc**2) + 1e-8)
    return numerator / denominator

def cii_metric(orig_gray, proc_gray):
    contrast_orig = orig_gray.std()
    contrast_proc = proc_gray.std()
    return contrast_proc / (contrast_orig + 1e-8)

# -----------------------------
# Paths
# -----------------------------
input_main_folder = "/content/drive/MyDrive/Colab Notebooks/entropy_normalized_images"
output_main_folder = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/images_clahe"
os.makedirs(output_main_folder, exist_ok=True)

# -----------------------------
# Recursively find all .tif images
# -----------------------------
image_paths = glob(os.path.join(input_main_folder, '**', '*.tif'), recursive=True)

# -----------------------------
# Store overall metrics
# -----------------------------
psnr_list, mse_list, ssim_list, epi_list, cii_list = [], [], [], [], []

# -----------------------------
# Process images and compute metrics
# -----------------------------
for img_path in image_paths:
    # Load image
    img_color = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # Convert to LAB and split channels
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)

    # Apply CLAHE enhancement on L channel
    l_channel_enhanced = clahe_enhancement(l_channel, clip_limit=2.0, tile_grid_size=(8, 8))

    # Merge channels and convert back to RGB
    lab_enhanced = cv2.merge((l_channel_enhanced, a_channel, b_channel))
    color_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # -----------------------------
    # Compute metrics
    # -----------------------------
    psnr_val = psnr_metric(l_channel, l_channel_enhanced)
    mse_val = mse_metric(l_channel, l_channel_enhanced)
    ssim_val = ssim_metric(l_channel, l_channel_enhanced)
    epi_val = epi_metric(l_channel, l_channel_enhanced)
    cii_val = cii_metric(l_channel, l_channel_enhanced)

    # Append to lists for overall metrics
    psnr_list.append(psnr_val)
    mse_list.append(mse_val)
    ssim_list.append(ssim_val)
    epi_list.append(epi_val)
    cii_list.append(cii_val)

    print(f"Metrics for {os.path.basename(img_path)}:")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  MSE: {mse_val:.2f}")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  EPI: {epi_val:.4f}")
    print(f"  CII: {cii_val:.4f}")

    # -----------------------------
    # Save enhanced image
    # -----------------------------
    relative_path = os.path.relpath(img_path, input_main_folder)
    output_path = os.path.join(output_main_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(color_enhanced, cv2.COLOR_RGB2BGR))

    # -----------------------------
    # Display original and enhanced
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(color_enhanced)
    plt.title("CLAHE Enhanced")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Display overall metrics
# -----------------------------
print("\n=== Overall Metrics Across All Images ===")
print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")
print(f"Average MSE: {np.mean(mse_list):.2f}")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")
print(f"Average EPI: {np.mean(epi_list):.4f}")
print(f"Average CII: {np.mean(cii_list):.4f}")
