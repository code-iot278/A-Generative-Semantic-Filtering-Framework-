# ============================================================
# Anisotropic Diffusion + Kuwahara Filtering | Folder-wise
# Python / Colab Compatible
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# ------------------------------------------------------------
# Anisotropic Diffusion (Perona–Malik)
# ------------------------------------------------------------
def anisotropic_diffusion(img, niter=15, kappa=30, gamma=0.2):
    img = img.astype(np.float32)
    diff = img.copy()
    for _ in range(niter):
        north = np.roll(diff, -1, axis=0) - diff
        south = np.roll(diff,  1, axis=0) - diff
        east  = np.roll(diff, -1, axis=1) - diff
        west  = np.roll(diff,  1, axis=1) - diff

        cN = np.exp(-(north/kappa)**2)
        cS = np.exp(-(south/kappa)**2)
        cE = np.exp(-(east /kappa)**2)
        cW = np.exp(-(west /kappa)**2)

        diff += gamma * (cN*north + cS*south + cE*east + cW*west)
    return diff

# ------------------------------------------------------------
# Kuwahara Filter
# ------------------------------------------------------------
def kuwahara_filter(img, window=5):
    pad = window // 2
    padded = np.pad(img, pad, mode='reflect')
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            regions = []
            for dy in [0, pad]:
                for dx in [0, pad]:
                    region = padded[i+dy:i+dy+pad+1, j+dx:j+dx+pad+1]
                    regions.append((np.var(region), np.mean(region)))
            output[i, j] = min(regions, key=lambda x: x[0])[1]
    return output

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
input_main_folder = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/images_curvelet"
output_main_folder = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/images_diff_kuwahara"
os.makedirs(output_main_folder, exist_ok=True)

# Supported image formats
valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# ------------------------------------------------------------
# Recursively find all images
# ------------------------------------------------------------
image_paths = []
for ext in valid_ext:
    image_paths.extend(glob(os.path.join(input_main_folder, '**', f'*{ext}'), recursive=True))

# ------------------------------------------------------------
# Process all images
# ------------------------------------------------------------
for img_path in image_paths:
    img_color = cv2.imread(img_path)
    if img_color is None:
        continue

    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Apply Anisotropic Diffusion
    diffused = anisotropic_diffusion(img_gray, niter=15, kappa=30, gamma=0.2)

    # Apply Kuwahara Filter
    kuwahara = kuwahara_filter(diffused.astype(np.uint8), window=5)
    kuwahara = cv2.normalize(kuwahara, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Preserve original color
    final_result = cv2.bitwise_and(img_rgb, img_rgb, mask=kuwahara)

    # Prepare output folder structure
    relative_path = os.path.relpath(img_path, input_main_folder)
    output_path = os.path.join(output_main_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save processed image
    cv2.imwrite(output_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    print(f"Processed and saved: {output_path}")

    # Optional: Display first few images
    if image_paths.index(img_path) < 3:  # only first 3 for speed
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(diffused, cmap='gray')
        plt.title("Anisotropic Diffusion")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(final_result)
        plt.title("Diffusion + Kuwahara")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
