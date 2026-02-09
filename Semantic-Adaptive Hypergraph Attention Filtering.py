# ============================================================
# Semantic-Adaptive Hypergraph Attention Filtering
# Recursively process all images in a main folder
# Python / Colab Compatible
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import rgb2lab
import os

# ------------------------------------------------------------
# Hypergraph Attention Filtering Function
# ------------------------------------------------------------
def hypergraph_attention_filter(image_rgb, n_segments=300, compactness=20, alpha=0.9):
    img_lab = rgb2lab(image_rgb)
    h, w, _ = image_rgb.shape
    segments = slic(image_rgb, n_segments=n_segments, compactness=compactness, start_label=0)
    output = np.zeros_like(img_lab)

    for seg_id in np.unique(segments):
        mask = segments == seg_id
        pixels = img_lab[mask]
        if len(pixels) < 5:
            output[mask] = pixels
            continue
        center = np.mean(pixels, axis=0)
        distances = np.linalg.norm(pixels - center, axis=1)
        attention = np.exp(-distances)
        attention /= np.sum(attention) + 1e-8
        filtered = alpha * np.sum(attention[:, None] * pixels, axis=0) + (1 - alpha) * center
        output[mask] = filtered

    output_rgb = cv2.cvtColor(output.astype(np.float32), cv2.COLOR_Lab2RGB)
    output_rgb = np.clip(output_rgb, 0, 1)
    return (output_rgb * 255).astype(np.uint8)

# ------------------------------------------------------------
# Input / Output Folders
# ------------------------------------------------------------
input_folder = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/images_diff_kuwahara"
output_folder = "/content/drive/MyDrive/Colab Notebooks/filtered_images"
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

            # Apply filtering
            filtered_img = hypergraph_attention_filter(img_rgb, n_segments=300, compactness=20, alpha=0.7)

            # Preserve folder structure in output
            rel_path = os.path.relpath(root, input_folder)
            save_dir = os.path.join(output_folder, rel_path)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"filtered_{filename}")
            cv2.imwrite(save_path, cv2.cvtColor(filtered_img, cv2.COLOR_RGB2BGR))

            # Display side-by-side
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_rgb)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(filtered_img)
            plt.title("Filtered Image")
            plt.axis("off")

            plt.suptitle(filename)
            plt.show()

            print(f"Processed and displayed: {img_path}")

print("All images processed and displayed!")
