# ============================================================
# Curvelet-Domain Noise Suppression (FDCT-Inspired) | Folder
# Python / Colab Compatible
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor
import os
from glob import glob

# ------------------------------------------------------------
# Curvelet-Inspired Multiscale Directional Denoising
# ------------------------------------------------------------
def curvelet_domain_denoising(image_gray, scales=(0.05, 0.1, 0.2), orientations=16, thresh_factor=0.2):
    img = image_gray.astype(np.float32) / 255.0
    response = np.zeros_like(img)

    # Multiscale + multi-directional analysis
    for scale in scales:
        for theta in np.linspace(0, np.pi, orientations, endpoint=False):
            real, imag = gabor(img, frequency=scale, theta=theta)
            magnitude = np.sqrt(real**2 + imag**2)
            response = np.maximum(response, magnitude)

    # Thresholding (noise suppression)
    threshold = thresh_factor * np.mean(response)
    response[response < threshold] = 0

    # Normalize to 8-bit
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return response

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
input_main_folder = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/images_morph"
output_main_folder = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/images_curvelet"
os.makedirs(output_main_folder, exist_ok=True)

# Supported image formats
valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# ------------------------------------------------------------
# Process all images recursively
# ------------------------------------------------------------
image_paths = []
for ext in valid_ext:
    image_paths.extend(glob(os.path.join(input_main_folder, '**', f'*{ext}'), recursive=True))

for img_path in image_paths:
    # Load image
    img_color = cv2.imread(img_path)
    if img_color is None:
        continue

    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Apply Curvelet-Domain Noise Suppression
    curvelet_mask = curvelet_domain_denoising(
        img_gray, scales=(0.05, 0.1, 0.2), orientations=16, thresh_factor=0.2
    )

    # Preserve color using mask
    curvelet_result = cv2.bitwise_and(img_rgb, img_rgb, mask=curvelet_mask)

    # Prepare output folder structure
    relative_path = os.path.relpath(img_path, input_main_folder)
    output_path = os.path.join(output_main_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save processed image
    cv2.imwrite(output_path, cv2.cvtColor(curvelet_result, cv2.COLOR_RGB2BGR))
    print(f"Processed and saved: {output_path}")

    # Optional: Display first few images
    if image_paths.index(img_path) < 3:  # Display only first 3 for speed
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(curvelet_result)
        plt.title("Curvelet-Domain Noise Suppression")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
