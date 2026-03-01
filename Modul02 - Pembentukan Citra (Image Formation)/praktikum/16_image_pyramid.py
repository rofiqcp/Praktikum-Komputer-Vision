"""
==========================================================================
PERCOBAAN 16: IMAGE PYRAMID
==========================================================================
Image pyramid adalah representasi gambar multi-resolusi.
- Gaussian Pyramid: blur → downsample berulang
- Laplacian Pyramid: detail antar level (untuk blending)

Fungsi:
- cv2.pyrDown(src) → Gaussian blur + downsample 2×
- cv2.pyrUp(src) → Upsample 2× + Gaussian blur
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()
img = cv2.resize(img, (512, 512))

print("=" * 60)
print("PERCOBAAN 16: IMAGE PYRAMID")
print("=" * 60)

# ============================================================
# 1. Gaussian Pyramid (downsample)
# ============================================================
print("\n--- 1. Gaussian Pyramid ---")

gauss_pyr = [img]
current = img.copy()
for i in range(5):
    current = cv2.pyrDown(current)
    gauss_pyr.append(current)
    print(f"  Level {i+1}: {current.shape[1]}×{current.shape[0]}")

# ============================================================
# 2. Laplacian Pyramid (detail setiap level)
# ============================================================
print("\n--- 2. Laplacian Pyramid ---")

lap_pyr = []
for i in range(len(gauss_pyr) - 1):
    # Upsampled dari level berikutnya
    up = cv2.pyrUp(gauss_pyr[i + 1])
    # Samakan ukuran (bisa beda 1 piksel)
    h_cur, w_cur = gauss_pyr[i].shape[:2]
    up = cv2.resize(up, (w_cur, h_cur))
    # Laplacian = perbedaan antara level saat ini dan upsampled
    lap = cv2.subtract(gauss_pyr[i], up)
    lap_pyr.append(lap)
    print(f"  Laplacian {i}: {lap.shape[1]}×{lap.shape[0]}")
# Level terakhir = sisa (low frequency)
lap_pyr.append(gauss_pyr[-1])

# ============================================================
# 3. Rekonstruksi dari Laplacian Pyramid
# ============================================================
print("\n--- 3. Rekonstruksi ---")

# Mulai dari level terendah
recon = lap_pyr[-1]
for i in range(len(lap_pyr) - 2, -1, -1):
    up = cv2.pyrUp(recon)
    h_lap, w_lap = lap_pyr[i].shape[:2]
    up = cv2.resize(up, (w_lap, h_lap))
    recon = cv2.add(up, lap_pyr[i])

diff = np.mean(cv2.absdiff(img, recon))
print(f"  Error rekonstruksi: {diff:.4f}")

# ============================================================
# 4. Laplacian Blending (blend 2 gambar)
# ============================================================
print("\n--- 4. Laplacian Blending ---")

# Buat gambar kedua
img2 = cv2.imread(os.path.join(IMAGE_DIR, "kotak_warna.png"))
if img2 is None:
    img2 = np.zeros_like(img)
    img2[:, :, 2] = 200
img2 = cv2.resize(img2, (512, 512))

# Gaussian pyramids
gp1 = [img.copy()]
gp2 = [img2.copy()]
for i in range(5):
    gp1.append(cv2.pyrDown(gp1[-1]))
    gp2.append(cv2.pyrDown(gp2[-1]))

# Laplacian pyramids
lp1, lp2 = [], []
for i in range(5):
    up1 = cv2.resize(cv2.pyrUp(gp1[i+1]), (gp1[i].shape[1], gp1[i].shape[0]))
    up2 = cv2.resize(cv2.pyrUp(gp2[i+1]), (gp2[i].shape[1], gp2[i].shape[0]))
    lp1.append(cv2.subtract(gp1[i], up1))
    lp2.append(cv2.subtract(gp2[i], up2))
lp1.append(gp1[-1])
lp2.append(gp2[-1])

# Blend: setengah kiri dari img1, setengah kanan dari img2
lp_blend = []
for l1, l2 in zip(lp1, lp2):
    h_l, w_l = l1.shape[:2]
    blended = np.hstack([l1[:, :w_l//2], l2[:, w_l//2:]])
    lp_blend.append(blended)

# Rekonstruksi dari blended pyramid
result = lp_blend[-1]
for i in range(len(lp_blend) - 2, -1, -1):
    up = cv2.pyrUp(result)
    h_b, w_b = lp_blend[i].shape[:2]
    up = cv2.resize(up, (w_b, h_b))
    result = cv2.add(up, lp_blend[i])

# Direct blend (tanpa pyramid) untuk perbandingan
direct = np.hstack([img[:, :256], img2[:, 256:]])
print("  Laplacian blending vs direct blend")

# ============================================================
# 5. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Gaussian pyramid levels
for i in range(4):
    level = cv2.resize(gauss_pyr[i], (128, 128))
    axes[0, i].imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"Gauss L{i} ({gauss_pyr[i].shape[1]}²)")
    axes[0, i].axis("off")

# Laplacian + blending
lap_vis = cv2.normalize(lap_pyr[0], None, 0, 255, cv2.NORM_MINMAX)
axes[1, 0].imshow(cv2.cvtColor(lap_vis, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Laplacian L0")
axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(direct, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Direct Blend")
axes[1, 1].axis("off")

axes[1, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Laplacian Blend")
axes[1, 2].axis("off")

axes[1, 3].imshow(cv2.cvtColor(recon, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Rekonstruksi")
axes[1, 3].axis("off")

plt.suptitle("Percobaan 16: Image Pyramid", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "16_image_pyramid_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
