"""
==========================================================================
PERCOBAAN 09: GAUSSIAN BLUR
==========================================================================
Gaussian blur menggunakan kernel Gaussian untuk menghaluskan gambar.
Lebih alami dari box blur karena distribusi Gaussian memberi
bobot lebih besar pada piksel dekat pusat.

Fungsi:
- cv2.GaussianBlur(src, ksize, sigmaX) → Gaussian blur
- cv2.getGaussianKernel(ksize, sigma) → Gaussian 1D kernel
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(os.path.join(IMAGE_DIR, "kota.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!"); exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 09: GAUSSIAN BLUR")
print("=" * 60)

# ============================================================
# 1. Gaussian Blur Dasar
# ============================================================
print("\n--- 1. Gaussian Blur Dasar ---")

# ksize harus ganjil, sigma=0 → dihitung otomatis dari ksize
blur_3 = cv2.GaussianBlur(img, (3, 3), 0)
blur_7 = cv2.GaussianBlur(img, (7, 7), 0)
blur_15 = cv2.GaussianBlur(img, (15, 15), 0)
blur_31 = cv2.GaussianBlur(img, (31, 31), 0)
print("  ksize: 3, 7, 15, 31 (sigma otomatis)")

# ============================================================
# 2. Variasi Sigma
# ============================================================
print("\n--- 2. Variasi Sigma ---")

# Sigma besar → blur lebih kuat meskipun ksize sama
sigmas = [0.5, 1.0, 2.0, 5.0, 10.0]
sigma_results = []
for sigma in sigmas:
    # ksize besar agar sigma punya ruang cukup
    result = cv2.GaussianBlur(gray, (21, 21), sigma)
    sigma_results.append(result)
    print(f"  σ={sigma:5.1f}: mean diff dari original = {cv2.absdiff(gray, result).mean():.2f}")

# ============================================================
# 3. Gaussian Kernel 1D → 2D
# ============================================================
print("\n--- 3. Gaussian Kernel ---")

# Dapatkan Gaussian kernel 1D
kernel_1d = cv2.getGaussianKernel(7, 1.5)
# Kernel 2D = outer product dari 1D × 1D
kernel_2d = kernel_1d @ kernel_1d.T
print(f"  Kernel 1D shape: {kernel_1d.shape}")
print(f"  Kernel 2D shape: {kernel_2d.shape}")
print(f"  Sum kernel 2D: {kernel_2d.sum():.6f}")
# Terapkan kernel 2D manual
manual_gauss = cv2.filter2D(gray, -1, kernel_2d)
opencv_gauss = cv2.GaussianBlur(gray, (7, 7), 1.5)
diff = cv2.absdiff(manual_gauss, opencv_gauss).mean()
print(f"  Perbedaan manual vs OpenCV: {diff:.4f}")

# ============================================================
# 4. Gaussian vs Box Blur
# ============================================================
print("\n--- 4. Gaussian vs Box Blur ---")

# Box blur
box = cv2.blur(gray, (11, 11))
# Gaussian blur
gauss = cv2.GaussianBlur(gray, (11, 11), 0)
# Perbedaan
diff_box_gauss = cv2.absdiff(box, gauss)
print(f"  Mean perbedaan Box vs Gaussian: {diff_box_gauss.mean():.2f}")
print(f"  Max perbedaan: {diff_box_gauss.max()}")

# ============================================================
# 5. Iterative Gaussian Blur
# ============================================================
print("\n--- 5. Iterative Blur ---")

# Beberapa kali Gaussian kecil ≈ satu kali Gaussian besar
iter_img = gray.copy()
for i in range(5):
    iter_img = cv2.GaussianBlur(iter_img, (5, 5), 1.0)
# Bandingkan dengan single large blur
# sigma efektif dari N iterasi σ_eff = σ * sqrt(N)
single_large = cv2.GaussianBlur(gray, (15, 15), 1.0 * np.sqrt(5))
diff_iter = cv2.absdiff(iter_img, single_large).mean()
print(f"  5x(σ=1) ≈ 1x(σ={1.0 * np.sqrt(5):.2f})")
print(f"  Perbedaan: {diff_iter:.2f}")

# ============================================================
# 6. Performa: Timing
# ============================================================
print("\n--- 6. Timing ---")

for ksize in [3, 11, 31, 51]:
    t0 = time.time()
    for _ in range(100):
        cv2.GaussianBlur(gray, (ksize, ksize), 0)
    elapsed = (time.time() - t0) * 10  # ms per call
    print(f"  ksize={ksize:2d}: {elapsed:.2f} ms avg")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

titles_imgs = [
    ("Original", img),
    ("Gauss 3×3", blur_3),
    ("Gauss 15×15", blur_15),
    ("Gauss 31×31", blur_31),
]

for i, (title, im) in enumerate(titles_imgs):
    axes[0, i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(title)
    axes[0, i].axis("off")

# Baris 2: Sigma comparison + kernel
for i, (s, res) in enumerate(zip([0.5, 2.0, 5.0, 10.0],
                                  [sigma_results[0], sigma_results[2],
                                   sigma_results[3], sigma_results[4]])):
    axes[1, i].imshow(res, cmap='gray')
    axes[1, i].set_title(f"σ={s}")
    axes[1, i].axis("off")

plt.suptitle("Percobaan 09: Gaussian Blur", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "09_gaussian_blur_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 09")
print("=" * 60)
print("""
1. cv2.GaussianBlur(src, ksize, sigma) → Gaussian smoothing
2. sigma=0 → sigma dihitung otomatis dari ksize
3. Gaussian kernel = distribusi bell curve (bobot pusat besar)
4. ksize lebih besar / sigma lebih besar → blur lebih kuat
5. Gaussian = separable filter (efisien: 2D → dua 1D)
6. N iterasi σ kecil ≈ 1 iterasi σ_eff = σ·√N
""")
