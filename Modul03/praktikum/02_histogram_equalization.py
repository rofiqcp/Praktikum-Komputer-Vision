"""
==========================================================================
PERCOBAAN 02: HISTOGRAM EQUALIZATION
==========================================================================
Histogram equalization menyebarkan distribusi intensitas secara merata
sehingga kontras gambar meningkat secara global.

Fungsi:
- cv2.calcHist(images, channels, mask, histSize, ranges) → hitung histogram
- cv2.equalizeHist(src) → equalisasi histogram (grayscale)
- plt.hist() → visualisasi distribusi piksel
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

img = cv2.imread(os.path.join(IMAGE_DIR, "kota.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!"); exit()

# Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 02: HISTOGRAM EQUALIZATION")
print("=" * 60)

# ============================================================
# 1. Histogram Gambar Asli
# ============================================================
print("\n--- 1. Histogram Asli ---")

# Hitung histogram menggunakan OpenCV
hist_orig = cv2.calcHist([gray], [0], None, [256], [0, 256])
print(f"  Mean intensitas: {gray.mean():.1f}")
print(f"  Std deviasi: {gray.std():.1f}")
print(f"  Min/Max: {gray.min()}/{gray.max()}")

# ============================================================
# 2. Equalisasi Histogram
# ============================================================
print("\n--- 2. Equalisasi Histogram ---")

# Terapkan equalisasi histogram
eq = cv2.equalizeHist(gray)
hist_eq = cv2.calcHist([eq], [0], None, [256], [0, 256])
print(f"  Mean setelah eq: {eq.mean():.1f}")
print(f"  Std setelah eq: {eq.std():.1f}")
print(f"  Min/Max setelah eq: {eq.min()}/{eq.max()}")

# ============================================================
# 3. Equalisasi pada Gambar Gelap
# ============================================================
print("\n--- 3. Equalisasi Gambar Gelap ---")

# Buat versi gelap dari gambar
dark = cv2.convertScaleAbs(gray, alpha=0.4, beta=-20)
dark_eq = cv2.equalizeHist(dark)
print(f"  Gelap mean: {dark.mean():.1f} → Eq mean: {dark_eq.mean():.1f}")

# ============================================================
# 4. Equalisasi pada Gambar Terang
# ============================================================
print("\n--- 4. Equalisasi Gambar Terang ---")

# Buat versi terang dari gambar
bright = cv2.convertScaleAbs(gray, alpha=0.8, beta=100)
bright_eq = cv2.equalizeHist(bright)
print(f"  Terang mean: {bright.mean():.1f} → Eq mean: {bright_eq.mean():.1f}")

# ============================================================
# 5. Equalisasi Manual (Implementasi sendiri)
# ============================================================
print("\n--- 5. Equalisasi Manual ---")

# Hitung histogram
hist_manual = np.zeros(256, dtype=np.int64)
for val in gray.ravel():
    hist_manual[val] += 1

# Hitung CDF (Cumulative Distribution Function)
cdf = hist_manual.cumsum()
# Normalisasi CDF ke range [0, 255]
cdf_normalized = ((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min()))
cdf_normalized = cdf_normalized.astype(np.uint8)

# Terapkan mapping
eq_manual = cdf_normalized[gray]

# Bandingkan dengan OpenCV
diff = cv2.absdiff(eq, eq_manual).mean()
print(f"  Perbedaan manual vs OpenCV: {diff:.4f}")

# ============================================================
# 6. Equalisasi pada Gambar Berwarna (YCrCb)
# ============================================================
print("\n--- 6. Equalisasi Gambar Berwarna ---")

# Konversi ke YCrCb (Y = luminance)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# Equalisasi hanya channel Y (luminance)
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
# Konversi kembali ke BGR
eq_color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
print("  Equalisasi channel Y pada YCrCb")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Original vs Equalized + histogram
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].hist(gray.ravel(), 256, [0, 256], color='steelblue', alpha=0.8)
axes[0, 1].set_title("Histogram Original")
axes[0, 1].set_xlim(0, 256)

axes[0, 2].imshow(eq, cmap='gray')
axes[0, 2].set_title("Equalized")
axes[0, 2].axis("off")

axes[0, 3].hist(eq.ravel(), 256, [0, 256], color='coral', alpha=0.8)
axes[0, 3].set_title("Histogram Equalized")
axes[0, 3].set_xlim(0, 256)

# Baris 2: Dark vs Dark Eq
axes[1, 0].imshow(dark, cmap='gray')
axes[1, 0].set_title("Gelap")
axes[1, 0].axis("off")

axes[1, 1].hist(dark.ravel(), 256, [0, 256], color='steelblue', alpha=0.8)
axes[1, 1].set_title("Hist Gelap")
axes[1, 1].set_xlim(0, 256)

axes[1, 2].imshow(dark_eq, cmap='gray')
axes[1, 2].set_title("Eq Gelap")
axes[1, 2].axis("off")

axes[1, 3].hist(dark_eq.ravel(), 256, [0, 256], color='coral', alpha=0.8)
axes[1, 3].set_title("Hist Eq Gelap")
axes[1, 3].set_xlim(0, 256)

# Baris 3: Warna
axes[2, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("Original Warna")
axes[2, 0].axis("off")

axes[2, 1].imshow(cv2.cvtColor(eq_color, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("Eq Warna (YCrCb)")
axes[2, 1].axis("off")

# CDF plot
axes[2, 2].plot(cdf / cdf.max() * 255, color='steelblue', label='CDF Original')
cdf_eq = cv2.calcHist([eq], [0], None, [256], [0, 256]).cumsum()
axes[2, 2].plot(cdf_eq / cdf_eq.max() * 255, color='coral', label='CDF Eq')
axes[2, 2].set_title("CDF")
axes[2, 2].legend()
axes[2, 2].set_xlim(0, 256)

axes[2, 3].axis("off")

plt.suptitle("Percobaan 02: Histogram Equalization", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "02_histogram_equalization_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 02")
print("=" * 60)
print("""
1. Histogram equalization menyebarkan intensitas piksel merata
2. cv2.equalizeHist() bekerja pada gambar grayscale saja
3. Untuk gambar berwarna: konversi ke YCrCb, eq channel Y
4. CDF (Cumulative Distribution Function) menjadi lookup table
5. Efektif meningkatkan kontras gambar gelap/terang
6. Implementasi manual: histogram → CDF → normalisasi → mapping
""")
