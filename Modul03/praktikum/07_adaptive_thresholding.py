"""
==========================================================================
PERCOBAAN 07: ADAPTIVE THRESHOLDING
==========================================================================
Adaptive thresholding menghitung threshold lokal untuk setiap piksel
berdasarkan neighborhood-nya. Cocok untuk gambar dengan pencahayaan
tidak merata.

Fungsi:
- cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType,
                        blockSize, C)
  - ADAPTIVE_THRESH_MEAN_C: threshold = mean lokal - C
  - ADAPTIVE_THRESH_GAUSSIAN_C: threshold = Gaussian-weighted mean - C
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

img = cv2.imread(os.path.join(IMAGE_DIR, "dokumen.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!"); exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 07: ADAPTIVE THRESHOLDING")
print("=" * 60)

# ============================================================
# 1. Mean Adaptive Thresholding
# ============================================================
print("\n--- 1. Mean Adaptive ---")

# blockSize harus ganjil, C adalah konstanta yang dikurangi
adapt_mean = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 5)
white_pct = np.sum(adapt_mean == 255) / adapt_mean.size * 100
print(f"  blockSize=11, C=5: {white_pct:.1f}% putih")

# ============================================================
# 2. Gaussian Adaptive Thresholding
# ============================================================
print("\n--- 2. Gaussian Adaptive ---")

adapt_gauss = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 5)
white_pct_g = np.sum(adapt_gauss == 255) / adapt_gauss.size * 100
print(f"  blockSize=11, C=5: {white_pct_g:.1f}% putih")

# ============================================================
# 3. Variasi blockSize
# ============================================================
print("\n--- 3. Variasi blockSize ---")

block_sizes = [3, 7, 11, 21, 51, 101]
block_results = []

for bs in block_sizes:
    result = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, bs, 5)
    block_results.append(result)
    print(f"  blockSize={bs:3d}: area putih={np.mean(result == 255) * 100:.1f}%")

# ============================================================
# 4. Variasi Konstanta C
# ============================================================
print("\n--- 4. Variasi Konstanta C ---")

c_values = [-5, 0, 5, 10, 20, 30]
c_results = []

for c in c_values:
    result = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, c)
    c_results.append(result)
    print(f"  C={c:3d}: area putih={np.mean(result == 255) * 100:.1f}%")

# ============================================================
# 5. Perbandingan Global vs Adaptive
# ============================================================
print("\n--- 5. Global vs Adaptive ---")

# Global Otsu
_, global_otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Buat gambar dengan pencahayaan tidak merata
h, w = gray.shape
# Gradient gelap-terang dari kiri ke kanan
gradient = np.tile(np.linspace(0.3, 1.0, w).astype(np.float32), (h, 1))
uneven = (gray.astype(np.float32) * gradient).astype(np.uint8)

# Global threshold pada gambar tidak merata
_, global_uneven = cv2.threshold(uneven, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Adaptive pada gambar tidak merata
adapt_uneven = cv2.adaptiveThreshold(uneven, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 21, 5)
print("  Global gagal pada pencahayaan tidak merata")
print("  Adaptive berhasil menyesuaikan threshold lokal")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Perbandingan metode
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(global_otsu, cmap='gray')
axes[0, 1].set_title("Global Otsu")
axes[0, 1].axis("off")

axes[0, 2].imshow(adapt_mean, cmap='gray')
axes[0, 2].set_title("Adaptive Mean")
axes[0, 2].axis("off")

axes[0, 3].imshow(adapt_gauss, cmap='gray')
axes[0, 3].set_title("Adaptive Gaussian")
axes[0, 3].axis("off")

# Baris 2: Variasi blockSize
for i, (bs, res) in enumerate(zip([3, 11, 51, 101],
                                   [block_results[0], block_results[2],
                                    block_results[4], block_results[5]])):
    axes[1, i].imshow(res, cmap='gray')
    axes[1, i].set_title(f"block={bs}")
    axes[1, i].axis("off")

# Baris 3: Pencahayaan tidak merata
axes[2, 0].imshow(uneven, cmap='gray')
axes[2, 0].set_title("Tidak Merata")
axes[2, 0].axis("off")

axes[2, 1].imshow(global_uneven, cmap='gray')
axes[2, 1].set_title("Global (Gagal)")
axes[2, 1].axis("off")

axes[2, 2].imshow(adapt_uneven, cmap='gray')
axes[2, 2].set_title("Adaptive (Berhasil)")
axes[2, 2].axis("off")

axes[2, 3].axis("off")

plt.suptitle("Percobaan 07: Adaptive Thresholding", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "07_adaptive_threshold_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 07")
print("=" * 60)
print("""
1. Adaptive threshold menghitung T lokal per piksel
2. MEAN_C: T = mean(neighborhood) - C
3. GAUSSIAN_C: T = Gaussian-weighted mean - C
4. blockSize kecil → detail tinggi, noise lebih banyak
5. blockSize besar → hasil lebih halus, kurang detail
6. C positif → lebih banyak area putih (foreground)
7. Sangat efektif untuk pencahayaan tidak merata
""")
