"""
==========================================================================
PERCOBAAN 18: HISTOGRAM GAMBAR
==========================================================================
Program ini mempelajari cara menghitung dan memvisualisasikan histogram
gambar. Histogram menunjukkan distribusi intensitas piksel.

Fungsi utama:
- cv2.calcHist([img], [channel], mask, [histSize], [ranges])
  → Menghitung histogram gambar
  - channel  : 0=Blue, 1=Green, 2=Red (untuk BGR)
  - histSize : jumlah bin (biasanya 256)
  - ranges   : rentang nilai (biasanya [0,256])
- cv2.equalizeHist(src) → Equalization histogram (meratakan distribusi)
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

print("=" * 60)
print("PERCOBAAN 18: HISTOGRAM GAMBAR")
print("=" * 60)

# Membaca gambar
img = cv2.imread(os.path.join(IMAGE_DIR, "lena.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

img = cv2.resize(img, (300, 300))
# Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ============================================================
# 1. Histogram grayscale
# ============================================================
print("\n--- 1. Histogram Grayscale ---")

# cv2.calcHist([images], [channels], mask, [histSize], [ranges])
# [gray]   : gambar input dalam list
# [0]      : channel ke-0 (satu-satunya channel)
# None     : tanpa mask (hitung seluruh gambar)
# [256]    : 256 bin (satu per nilai intensitas)
# [0, 256] : rentang nilai dari 0 sampai 255
hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])

print(f"  Shape histogram: {hist_gray.shape}")
print(f"  Total piksel: {hist_gray.sum():.0f}")
print(f"  Piksel intensitas 0 (hitam): {hist_gray[0][0]:.0f}")
print(f"  Piksel intensitas 255 (putih): {hist_gray[255][0]:.0f}")

# ============================================================
# 2. Histogram per-channel BGR
# ============================================================
print("\n--- 2. Histogram Per-Channel BGR ---")

warna = ('b', 'g', 'r')
nama_warna = ('Blue', 'Green', 'Red')

# Menghitung histogram untuk setiap channel
for i, (w, n) in enumerate(zip(warna, nama_warna)):
    # Menghitung histogram channel ke-i
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    print(f"  {n}: mean={hist.mean():.1f}, max={hist.max():.0f}")

# ============================================================
# 3. Histogram dengan mask (ROI tertentu)
# ============================================================
print("\n--- 3. Histogram dengan Mask ---")

# Membuat mask lingkaran (hanya hitung histogram di area lingkaran)
mask = np.zeros(gray.shape, dtype=np.uint8)
# cv2.circle membuat lingkaran putih di tengah mask
h, w = gray.shape
cv2.circle(mask, (w // 2, h // 2), 100, 255, -1)

# Menghitung histogram hanya di area mask putih
hist_masked = cv2.calcHist([gray], [0], mask, [256], [0, 256])
# Histogram tanpa mask (seluruh gambar)
hist_full = cv2.calcHist([gray], [0], None, [256], [0, 256])

print(f"  Piksel dalam mask: {cv2.countNonZero(mask)}")
print(f"  Total hist masked: {hist_masked.sum():.0f}")
print(f"  Total hist full:   {hist_full.sum():.0f}")

# ============================================================
# 4. Equalisasi histogram
# ============================================================
print("\n--- 4. Equalisasi Histogram ---")

# cv2.equalizeHist meratakan distribusi intensitas
# Membuat gambar gelap dan terang untuk demo
gray_gelap = cv2.convertScaleAbs(gray, alpha=0.5, beta=-30)
gray_terang = cv2.convertScaleAbs(gray, alpha=1.0, beta=70)

# Equalisasi pada gambar gelap
eq_gelap = cv2.equalizeHist(gray_gelap)
# Equalisasi pada gambar terang
eq_terang = cv2.equalizeHist(gray_terang)
# Equalisasi pada gambar normal
eq_normal = cv2.equalizeHist(gray)

print(f"  Gelap → equalized: mean {gray_gelap.mean():.1f} → {eq_gelap.mean():.1f}")
print(f"  Terang → equalized: mean {gray_terang.mean():.1f} → {eq_terang.mean():.1f}")
print(f"  Normal → equalized: mean {gray.mean():.1f} → {eq_normal.mean():.1f}")

# ============================================================
# 5. Histogram dengan jumlah bin berbeda
# ============================================================
print("\n--- 5. Jumlah Bin Berbeda ---")

# Bin = 256 → histogram paling detail
hist_256 = cv2.calcHist([gray], [0], None, [256], [0, 256])
# Bin = 64 → detail sedang
hist_64 = cv2.calcHist([gray], [0], None, [64], [0, 256])
# Bin = 16 → kasar
hist_16 = cv2.calcHist([gray], [0], None, [16], [0, 256])

print(f"  256 bin: {hist_256.shape}")
print(f"  64 bin:  {hist_64.shape}")
print(f"  16 bin:  {hist_16.shape}")

# ============================================================
# 6. Statistik dari histogram
# ============================================================
print("\n--- 6. Statistik ---")

# Menghitung mean dari histogram
# Setiap bin berbobot sesuai nilainya
bobot = np.arange(256)
total_piksel = hist_gray.flatten().sum()

# Mean = Σ(nilai × frekuensi) / total
mean_hist = np.sum(bobot * hist_gray.flatten()) / total_piksel
# Variance
var_hist = np.sum(((bobot - mean_hist) ** 2) * hist_gray.flatten()) / total_piksel
# Standard deviation
std_hist = np.sqrt(var_hist)

print(f"  Mean dari histogram: {mean_hist:.2f}")
print(f"  Std dari histogram:  {std_hist:.2f}")
print(f"  Verifikasi (NumPy mean): {gray.mean():.2f}")
print(f"  Verifikasi (NumPy std):  {gray.std():.2f}")

# ============================================================
# 7. Visualisasi lengkap
# ============================================================

fig = plt.figure(figsize=(20, 16))

# --- Gambar original + grayscale ---
ax1 = fig.add_subplot(3, 4, 1)
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original")
ax1.axis("off")

ax2 = fig.add_subplot(3, 4, 2)
ax2.imshow(gray, cmap="gray")
ax2.set_title("Grayscale")
ax2.axis("off")

# --- Histogram grayscale ---
ax3 = fig.add_subplot(3, 4, 3)
ax3.plot(hist_gray, color='black', linewidth=0.7)
ax3.set_title("Histogram Gray")
ax3.set_xlim([0, 256])

# --- Histogram per-channel ---
ax4 = fig.add_subplot(3, 4, 4)
for i, w in enumerate(warna):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    ax4.plot(hist, color=w, linewidth=0.7, label=nama_warna[i])
ax4.set_title("Histogram BGR")
ax4.set_xlim([0, 256])
ax4.legend()

# --- Masked histogram ---
ax5 = fig.add_subplot(3, 4, 5)
ax5.imshow(cv2.bitwise_and(gray, mask), cmap="gray")
ax5.set_title("Area Mask")
ax5.axis("off")

ax6 = fig.add_subplot(3, 4, 6)
ax6.plot(hist_full, 'b-', linewidth=0.7, label='Full', alpha=0.5)
ax6.plot(hist_masked, 'r-', linewidth=0.7, label='Masked')
ax6.set_title("Full vs Mask")
ax6.legend()
ax6.set_xlim([0, 256])

# --- Equalisasi ---
ax7 = fig.add_subplot(3, 4, 7)
ax7.imshow(gray_gelap, cmap="gray")
ax7.set_title("Gelap")
ax7.axis("off")

ax8 = fig.add_subplot(3, 4, 8)
ax8.imshow(eq_gelap, cmap="gray")
ax8.set_title("Equalized")
ax8.axis("off")

# --- Histogram sebelum/sesudah equalisasi ---
ax9 = fig.add_subplot(3, 4, 9)
hist_pre = cv2.calcHist([gray_gelap], [0], None, [256], [0, 256])
ax9.plot(hist_pre, 'r-', linewidth=0.7)
ax9.set_title("Hist Gelap")
ax9.set_xlim([0, 256])

ax10 = fig.add_subplot(3, 4, 10)
hist_post = cv2.calcHist([eq_gelap], [0], None, [256], [0, 256])
ax10.plot(hist_post, 'g-', linewidth=0.7)
ax10.set_title("Hist Equalized")
ax10.set_xlim([0, 256])

# --- Jumlah bin berbeda ---
ax11 = fig.add_subplot(3, 4, 11)
ax11.bar(range(16), hist_16.flatten(), color='steelblue')
ax11.set_title("16 Bin")

ax12 = fig.add_subplot(3, 4, 12)
ax12.bar(range(64), hist_64.flatten(), color='coral', width=1)
ax12.set_title("64 Bin")

plt.suptitle("Percobaan 18: Histogram Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "18_histogram_gambar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 18")
print("=" * 60)
print("  cv2.calcHist()     → Hitung histogram")
print("  cv2.equalizeHist() → Ratakan distribusi")
print("  mask → hitung histogram area tertentu")
print("  histSize → jumlah bin (detail histogram)")
print("  Histogram → peta distribusi intensitas piksel")
print("=" * 60)
