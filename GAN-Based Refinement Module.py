# ============================================================
# Refinement Normalization (Color-Preserving) | Full Code
# Python / Colab Compatible
# ============================================================
# =========================================================
# DCGAN – INPUT IMAGES → TRAIN → SAVE SINGLE .PKL → GENERATE
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from glob import glob
from torch.utils.data import Dataset, DataLoader
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.metrics import structural_similarity as ssim
# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
# ------------------------------
# CONFIG
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

data_dir = "/content/drive/MyDrive/Colab Notebooks/entropy_normalized_images"
save_path = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/gan_model.pkl"

image_size = 64
batch_size = 64
latent_dim = 100
epochs = 50
lr = 0.0002

# ------------------------------
# CUSTOM DATASET (OpenCV Loader)
# ------------------------------
class TiffDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = glob(f"{root_dir}/**/*.*", recursive=True)
        self.files = [f for f in self.files if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path)
        if img is None:
            # Fallback black image if unreadable
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label

# ------------------------------
# TRANSFORMS
# ------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = TiffDataset(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

print(f"Total images: {len(dataset)}")
print(f"Total batches: {len(loader)}")

# ------------------------------
# GENERATOR
# ------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# ------------------------------
# DISCRIMINATOR
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), 1)

# ------------------------------
# INIT MODELS AND OPTIMIZERS
# ------------------------------
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# ------------------------------
# TRAINING LOOP
# ------------------------------
for epoch in range(epochs):
    d_loss_epoch = 0.0
    g_loss_epoch = 0.0

    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)
        b = real_imgs.size(0)

        real_labels = torch.ones(b, 1, device=device)
        fake_labels = torch.zeros(b, 1, device=device)

        # ---- Discriminator ----
        z = torch.randn(b, latent_dim, 1, 1, device=device)
        fake_imgs = G(z)

        d_loss = criterion(D(real_imgs), real_labels) + criterion(D(fake_imgs.detach()), fake_labels)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # ---- Generator ----
        g_loss = criterion(D(fake_imgs), real_labels)
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        d_loss_epoch += d_loss.item()
        g_loss_epoch += g_loss.item()

    num_batches = max(1, len(loader))
    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss_epoch/num_batches:.4f} | G Loss: {g_loss_epoch/num_batches:.4f}")

# ------------------------------
# SAVE SINGLE .PKL FILE
# ------------------------------
torch.save({
    "generator": G.state_dict(),
    "discriminator": D.state_dict(),
    "opt_G": opt_G.state_dict(),
    "opt_D": opt_D.state_dict(),
    "latent_dim": latent_dim,
    "epochs": epochs
}, save_path)

print("✅ SINGLE GAN MODEL SAVED:", save_path)

# -----------------------------
# Refinement Normalization Function (Grayscale)
# -----------------------------
def refinement_normalization(image_gray, entropy_radius=7, alpha=0.2):
    img = image_gray.astype(np.uint8)
    ent = entropy(img, disk(entropy_radius)).astype(np.float32)
    ent_norm = cv2.normalize(ent, None, 0, 1, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    normalized = alpha * ent_norm * enhanced + (1 - alpha * ent_norm) * img
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return normalized

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
    # Edge Preservation Index (simple approximation using Sobel)
    grad_orig = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 1)
    grad_proc = cv2.Sobel(proc_gray, cv2.CV_64F, 1, 1)
    numerator = np.sum(grad_orig * grad_proc)
    denominator = np.sqrt(np.sum(grad_orig**2) * np.sum(grad_proc**2) + 1e-8)
    return numerator / denominator

def cii_metric(orig_gray, proc_gray):
    # Contrast Improvement Index (RMS contrast)
    contrast_orig = orig_gray.std()
    contrast_proc = proc_gray.std()
    return contrast_proc / (contrast_orig + 1e-8)

# -----------------------------
# Paths
# -----------------------------
input_main_folder = "/content/drive/MyDrive/Colab Notebooks/entropy_normalized_images"
output_main_folder = "/content/drive/MyDrive/Colab Notebooks/UC MercedLand for Earth Scanning/images_normalized"
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

    # Apply refinement normalization on L channel
    l_channel_norm = refinement_normalization(l_channel, entropy_radius=7, alpha=0.2)

    # Merge channels and convert back to RGB
    lab_normalized = cv2.merge((l_channel_norm, a_channel, b_channel))
    color_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)

    # -----------------------------
    # Compute metrics
    # -----------------------------
    l_orig = l_channel
    l_proc = l_channel_norm

    psnr_val = psnr_metric(l_orig, l_proc)
    mse_val = mse_metric(l_orig, l_proc)
    ssim_val = ssim_metric(l_orig, l_proc)
    epi_val = epi_metric(l_orig, l_proc)
    cii_val = cii_metric(l_orig, l_proc)

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
    # Prepare output folder and save
    # -----------------------------
    relative_path = os.path.relpath(img_path, input_main_folder)
    output_path = os.path.join(output_main_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(color_normalized, cv2.COLOR_RGB2BGR))

    # -----------------------------
    # Display original and normalized
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(color_normalized)
    plt.title("Refinement Normalization")
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