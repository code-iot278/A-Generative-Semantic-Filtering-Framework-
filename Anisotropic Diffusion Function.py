# =========================================================
# Full Anisotropic Diffusion + Metrics Evaluation
# =========================================================

import os
import cv2
import numpy as np
from glob import glob
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt

# -----------------------------
# Anisotropic Diffusion Function (Grayscale)
# -----------------------------
def anisotropic_diffusion(image_gray, num_iter=10, kappa=20, gamma=0.1, option=1):
    """
    Apply Perona-Malik Anisotropic Diffusion to a grayscale image.

    Parameters:
    - image_gray: input grayscale image (numpy array 0-255)
    - num_iter: number of iterations
    - kappa: conduction coefficient (controls sensitivity to edges)
    - gamma: integration constant (0 < gamma <= 0.25 for stability)
    - option: 1 for exponential, 2 for inverse quadratic conduction function

    Returns:
    - diffused image (uint8)
    """
    img = image_gray.astype(np.float32)
    for _ in range(num_iter):
        # Pad the image to handle borders
        img_pad = np.pad(img, ((1, 1), (1, 1)), mode='edge')

        # Compute differences
        deltaN = img_pad[:-2, 1:-1] - img
        deltaS = img_pad[2:, 1:-1] - img
        deltaE = img_pad[1:-1, 2:] - img
        deltaW = img_pad[1:-1, :-2] - img

        # Compute conduction
        if option == 1:
            cN = np.exp(-(deltaN / kappa) ** 2)
            cS = np.exp(-(deltaS / kappa) ** 2)
            cE = np.exp(-(deltaE / kappa) ** 2)
            cW = np.exp(-(deltaW / kappa) ** 2)
        elif option == 2:
            cN = 1.0 / (1.0 + (deltaN / kappa) ** 2)
            cS = 1.0 / (1.0 + (deltaS / kappa) ** 2)
            cE = 1.0 / (1.0 + (deltaE / kappa) ** 2)
            cW = 1.0 / (1.0 + (deltaW / kappa) ** 2)

        # Update image
        img += gamma * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)

    # Normalize to 0-255 and convert to uint8
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

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
output_main_folder = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/images_anisotropic"
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

    # Apply Anisotropic Diffusion on L channel
    l_channel_diffused = anisotropic_diffusion(l_channel, num_iter=15, kappa=30, gamma=0.15, option=1)

    # Merge channels and convert back to RGB
    lab_diffused = cv2.merge((l_channel_diffused, a_channel, b_channel))
    color_diffused = cv2.cvtColor(lab_diffused, cv2.COLOR_LAB2RGB)

    # -----------------------------
    # Compute metrics
    # -----------------------------
    psnr_val = psnr_metric(l_channel, l_channel_diffused)
    mse_val = mse_metric(l_channel, l_channel_diffused)
    ssim_val = ssim_metric(l_channel, l_channel_diffused)
    epi_val = epi_metric(l_channel, l_channel_diffused)
    cii_val = cii_metric(l_channel, l_channel_diffused)

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
    # Save diffused image
    # -----------------------------
    relative_path = os.path.relpath(img_path, input_main_folder)
    output_path = os.path.join(output_main_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(color_diffused, cv2.COLOR_RGB2BGR))

    # -----------------------------
    # Display original and diffused
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(color_diffused)
    plt.title("Anisotropic Diffusion")
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
