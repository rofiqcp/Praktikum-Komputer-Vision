"""
==========================================================================
PERCOBAAN 10: MEDIAN DAN BILATERAL FILTER
==========================================================================
- Median filter: mengganti piksel dengan median dari neighborhoodnya.
  Sangat efektif menghilangkan salt-and-pepper noise.
- Bilateral filter: menghaluskan sambil mempertahankan edge.
  Memberi bobot berdasarkan jarak DAN kesamaan intensitas.

Fungsi:
- cv2.medianBlur(src, ksize) → median filter
- cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace) → bilateral
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
print("PERCOBAAN 10: MEDIAN DAN BILATERAL FILTER")
print("=" * 60)

# ============================================================
# 1. Median Filter — Salt & Pepper Noise
# ============================================================
print("\n--- 1. Median vs Salt & Pepper ---")

# Tambahkan salt and pepper noise
sp = gray.copy()
prob = 0.05
salt = np.random.random(gray.shape) < prob
pepper = np.random.random(gray.shape) < prob
sp[salt] = 255
sp[pepper] = 0
print(f"  Noise probability: {prob*100}% salt, {prob*100}% pepper")

# Median filter (ksize harus ganjil)
median_3 = cv2.medianBlur(sp, 3)
median_5 = cv2.medianBlur(sp, 5)
median_7 = cv2.medianBlur(sp, 7)
median_11 = cv2.medianBlur(sp, 11)

# Gaussian untuk perbandingan
gauss_sp = cv2.GaussianBlur(sp, (5, 5), 0)

# PSNR
psnr_noisy = cv2.PSNR(gray, sp)
psnr_median3 = cv2.PSNR(gray, median_3)
psnr_median5 = cv2.PSNR(gray, median_5)
psnr_gauss = cv2.PSNR(gray, gauss_sp)
print(f"  PSNR noisy:     {psnr_noisy:.2f} dB")
print(f"  PSNR median(3): {psnr_median3:.2f} dB")
print(f"  PSNR median(5): {psnr_median5:.2f} dB")
print(f"  PSNR Gaussian:  {psnr_gauss:.2f} dB")

# ============================================================
# 2. Median Filter — Gaussian Noise
# ============================================================
print("\n--- 2. Median vs Gaussian Noise ---")

gauss_noise = gray.astype(np.int16) + np.random.normal(0, 30, gray.shape).astype(np.int16)
gauss_noisy = np.clip(gauss_noise, 0, 255).astype(np.uint8)

median_gn = cv2.medianBlur(gauss_noisy, 5)
gauss_gn = cv2.GaussianBlur(gauss_noisy, (5, 5), 0)

print(f"  PSNR noisy:   {cv2.PSNR(gray, gauss_noisy):.2f} dB")
print(f"  PSNR median:  {cv2.PSNR(gray, median_gn):.2f} dB")
print(f"  PSNR Gauss:   {cv2.PSNR(gray, gauss_gn):.2f} dB")

# ============================================================
# 3. Bilateral Filter Dasar
# ============================================================
print("\n--- 3. Bilateral Filter ---")

# d=diameter, sigmaColor=kesamaan warna, sigmaSpace=kedekatan spasial
# d=-1 menggunakan sigmaSpace untuk menghitung d
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
print("  d=9, sigmaColor=75, sigmaSpace=75")

# ============================================================
# 4. Variasi Parameter Bilateral
# ============================================================
print("\n--- 4. Variasi Parameter Bilateral ---")

# Variasi sigmaColor
for sc in [25, 75, 150]:
    result = cv2.bilateralFilter(img, d=9, sigmaColor=sc, sigmaSpace=75)
    diff = cv2.absdiff(img, result).mean()
    print(f"  sigmaColor={sc:3d}: mean diff={diff:.2f}")

# Variasi sigmaSpace
for ss in [25, 75, 150]:
    result = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=ss)
    diff = cv2.absdiff(img, result).mean()
    print(f"  sigmaSpace={ss:3d}: mean diff={diff:.2f}")

# ============================================================
# 5. Perbandingan Edge Preservation
# ============================================================
print("\n--- 5. Edge Preservation ---")

# Hitung Sobel pada original
sobel_orig = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)

# Gaussian blur lalu Sobel
gauss_blur = cv2.GaussianBlur(gray, (9, 9), 3)
sobel_gauss = cv2.Sobel(gauss_blur, cv2.CV_64F, 1, 1, ksize=3)

# Bilateral lalu Sobel
bilateral_gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
sobel_bilateral = cv2.Sobel(bilateral_gray, cv2.CV_64F, 1, 1, ksize=3)

print(f"  Edge energy original:  {np.sum(sobel_orig**2):.0f}")
print(f"  Edge energy Gaussian:  {np.sum(sobel_gauss**2):.0f}")
print(f"  Edge energy bilateral: {np.sum(sobel_bilateral**2):.0f}")
print("  Bilateral mempertahankan edge lebih baik!")

# ============================================================
# 6. Timing Comparison
# ============================================================
print("\n--- 6. Timing ---")

t0 = time.time()
for _ in range(50):
    cv2.GaussianBlur(gray, (9, 9), 0)
t_gauss = (time.time() - t0) / 50 * 1000

t0 = time.time()
for _ in range(50):
    cv2.medianBlur(gray, 9)
t_median = (time.time() - t0) / 50 * 1000

t0 = time.time()
for _ in range(50):
    cv2.bilateralFilter(gray, 9, 75, 75)
t_bilateral = (time.time() - t0) / 50 * 1000

print(f"  Gaussian:  {t_gauss:.2f} ms")
print(f"  Median:    {t_median:.2f} ms")
print(f"  Bilateral: {t_bilateral:.2f} ms")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Salt & pepper noise
axes[0, 0].imshow(sp, cmap='gray')
axes[0, 0].set_title("S&P Noise")
axes[0, 0].axis("off")

axes[0, 1].imshow(median_3, cmap='gray')
axes[0, 1].set_title("Median 3×3")
axes[0, 1].axis("off")

axes[0, 2].imshow(median_5, cmap='gray')
axes[0, 2].set_title("Median 5×5")
axes[0, 2].axis("off")

axes[0, 3].imshow(gauss_sp, cmap='gray')
axes[0, 3].set_title("Gaussian (bandingkan)")
axes[0, 3].axis("off")

# Baris 2: Bilateral
axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Original")
axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Bilateral")
axes[1, 1].axis("off")

blur_comp = cv2.GaussianBlur(img, (9, 9), 3)
axes[1, 2].imshow(cv2.cvtColor(blur_comp, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Gaussian Blur")
axes[1, 2].axis("off")

# Sisa edge setelah filter
axes[1, 3].imshow(np.abs(sobel_bilateral), cmap='gray', vmin=0, vmax=100)
axes[1, 3].set_title("Edge (Bilateral)")
axes[1, 3].axis("off")

# Baris 3: Gaussian noise
axes[2, 0].imshow(gauss_noisy, cmap='gray')
axes[2, 0].set_title("Gaussian Noise")
axes[2, 0].axis("off")

axes[2, 1].imshow(median_gn, cmap='gray')
axes[2, 1].set_title("Median")
axes[2, 1].axis("off")

axes[2, 2].imshow(gauss_gn, cmap='gray')
axes[2, 2].set_title("Gaussian Filter")
axes[2, 2].axis("off")

axes[2, 3].imshow(bilateral_gray, cmap='gray')
axes[2, 3].set_title("Bilateral")
axes[2, 3].axis("off")

plt.suptitle("Percobaan 10: Median dan Bilateral Filter", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "10_median_bilateral_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 10")
print("=" * 60)
print("""
1. Median filter: terbaik untuk salt & pepper noise
2. Median mengganti piksel dengan median neighbourhood
3. Bilateral filter: smoothing + edge preservation
4. sigmaColor → seberapa mirip warna yang diblur bersama
5. sigmaSpace → seberapa jauh piksel yang ikut dihitung
6. Bilateral lebih lambat dari Gaussian/median tapi lebih baik
""")
