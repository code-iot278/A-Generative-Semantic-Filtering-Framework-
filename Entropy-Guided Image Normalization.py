# ============================================================
# Entropy-Guided Image Normalization for All Images in Folder
# Python / Colab Compatible | Recursively Processes Subfolders
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
import os

# ------------------------------------------------------------
# Entropy-Guided Normalization Function
# ------------------------------------------------------------
def entropy_guided_normalization(image_gray, entropy_radius=7, alpha=0.6):
    """
    Entropy-guided adaptive image normalization
    """
    img = image_gray.astype(np.uint8)

    # Step 1: Local entropy estimation
    ent = entropy(img, disk(entropy_radius)).astype(np.float32)
    ent_norm = cv2.normalize(ent, None, 0, 1, cv2.NORM_MINMAX)

    # Step 2: Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Step 3: Entropy-guided blending
    normalized = alpha * ent_norm * enhanced + (1 - alpha * ent_norm) * img
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return normalized

# ------------------------------------------------------------
# Input / Output Folders
# ------------------------------------------------------------
input_folder = "/content/drive/MyDrive/Colab Notebooks/filtered_images"
output_folder = "/content/drive/MyDrive/Colab Notebooks/entropy_normalized_images"
os.makedirs(output_folder, exist_ok=True)

# ------------------------------------------------------------
# Recursively process all images
# ------------------------------------------------------------
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, filename)
            img_color = cv2.imread(img_path)
            if img_color is None:
                print(f"Failed to read: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            # Apply entropy-guided normalization
            entropy_norm = entropy_guided_normalization(img_gray, entropy_radius=7, alpha=0.6)

            # Preserve color using entropy-guided mask
            entropy_color = cv2.bitwise_and(img_rgb, img_rgb, mask=entropy_norm)

            # ------------------------------------------------
            # Save normalized image (preserve folder structure)
            # ------------------------------------------------
            rel_path = os.path.relpath(root, input_folder)
            save_dir = os.path.join(output_folder, rel_path)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"entropy_{filename}")
            cv2.imwrite(save_path, cv2.cvtColor(entropy_color, cv2.COLOR_RGB2BGR))

            # Optional: Display side-by-side for verification
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_rgb)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(entropy_color)
            plt.title("Entropy-Guided Normalization")
            plt.axis("off")

            plt.suptitle(filename)
            plt.tight_layout()
            plt.show()

            print(f"Processed: {img_path}")

print("All images processed and saved!")