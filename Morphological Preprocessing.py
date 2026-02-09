import cv2
import os
import matplotlib.pyplot as plt

# ---------------------------------
# Morphological Preprocessing
# ---------------------------------
def morphological_preprocessing(image_gray, kernel_size=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

# ---------------------------------
# Paths
# ---------------------------------
main_folder = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/images 1"
output_folder = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/images_morph"

os.makedirs(output_folder, exist_ok=True)

# Supported image formats
valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# ---------------------------------
# Traverse folders
# ---------------------------------
for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.lower().endswith(valid_ext):

            img_path = os.path.join(root, file)

            # Load color image
            img_color = cv2.imread(img_path)
            if img_color is None:
                continue

            # Convert for processing
            img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            # Morphological preprocessing
            processed_gray = morphological_preprocessing(img_gray)
            processed_color = cv2.bitwise_and(img_rgb, img_rgb, mask=processed_gray)

            # ---------------------------------
            # Prepare output path
            # ---------------------------------
            relative_path = os.path.relpath(img_path, main_folder)
            save_path = os.path.join(output_folder, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save processed image
            cv2.imwrite(save_path, cv2.cvtColor(processed_color, cv2.COLOR_RGB2BGR))
            print(f"Processed and saved: {save_path}")

            # ---------------------------------
            # Optional: Display
            # ---------------------------------
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_rgb)
            plt.title(f"Original\n{file}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(processed_color)
            plt.title("Morphologically Preprocessed")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

