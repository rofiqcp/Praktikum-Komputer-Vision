"""
==========================================================================
PERCOBAAN 17: BRIGHTNESS DAN CONTRAST SEDERHANA
==========================================================================
Program ini mempelajari cara mengatur kecerahan (brightness) dan
kontras gambar menggunakan operasi linear dan non-linear.

Rumus dasar:
  g(x,y) = α * f(x,y) + β
  - α (alpha/gain) → mengontrol kontras (α > 1 = kontras naik)
  - β (beta/bias)  → mengontrol brightness (β > 0 = lebih terang)

Fungsi utama:
- cv2.convertScaleAbs(src, alpha, beta) → Kontras + brightness
- np.clip() → Membatasi nilai piksel ke rentang valid
- cv2.normalize() → Normalisasi rentang nilai
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
print("PERCOBAAN 17: BRIGHTNESS DAN CONTRAST")
print("=" * 60)

# Membaca gambar
img = cv2.imread(os.path.join(IMAGE_DIR, "foto_gelap.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

img = cv2.resize(img, (300, 300))
print(f"[INFO] Gambar: {img.shape}")

# ============================================================
# 1. Mengatur brightness menggunakan cv2.add
# ============================================================
print("\n--- 1. Brightness dengan cv2.add ---")

# cv2.add melakukan penjumlahan dengan saturasi (max=255)
# Membuat array konstan untuk ditambahkan
terang = cv2.add(img, np.full_like(img, 60))
gelap = cv2.subtract(img, np.full_like(img, 60))

print(f"  Terang (+60): mean = {terang.mean():.1f}")
print(f"  Gelap  (-60): mean = {gelap.mean():.1f}")

# ============================================================
# 2. Brightness dengan cv2.convertScaleAbs
# ============================================================
print("\n--- 2. Brightness dengan convertScaleAbs ---")

# cv2.convertScaleAbs(src, alpha=1.0, beta=0)
# Menghitung: |alpha * src + beta| lalu konversi ke uint8
# alpha=1 artinya kontras tetap, beta mengubah brightness
img_bright = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
img_dark = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)

print(f"  Beta=+50 (terang): mean = {img_bright.mean():.1f}")
print(f"  Beta=-50 (gelap):  mean = {img_dark.mean():.1f}")

# ============================================================
# 3. Mengatur kontras
# ============================================================
print("\n--- 3. Kontras ---")

# alpha > 1.0 → kontras meningkat
img_kontras_tinggi = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
# alpha < 1.0 → kontras menurun
img_kontras_rendah = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
# alpha > 1 + beta < 0 → kontras naik tanpa terlalu terang
img_kontras_balanced = cv2.convertScaleAbs(img, alpha=1.5, beta=-60)

print(f"  Alpha=1.5 (kontras tinggi): std = {img_kontras_tinggi.std():.1f}")
print(f"  Alpha=0.5 (kontras rendah): std = {img_kontras_rendah.std():.1f}")
print(f"  Alpha=1.5 Beta=-60 (balanced): std = {img_kontras_balanced.std():.1f}")

# ============================================================
# 4. Kombinasi brightness + kontras
# ============================================================
print("\n--- 4. Kombinasi Brightness + Kontras ---")

kombinasi = [
    (1.0, 0, "Original"),
    (1.3, 20, "Cerah+Kontras"),
    (0.7, -10, "Gelap+Flat"),
    (1.5, -40, "Kontras Tinggi"),
    (0.5, 50, "Terang+Flat"),
    (2.0, -100, "Sangat Kontras"),
]

hasil_kombinasi = []
for alpha, beta, nama in kombinasi:
    # Terapkan transformasi linear
    hasil = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    hasil_kombinasi.append((hasil, nama))
    print(f"  α={alpha:.1f} β={beta:+d} → {nama}")

# ============================================================
# 5. Brightness/kontras manual dengan NumPy
# ============================================================
print("\n--- 5. Manual dengan NumPy ---")

alpha_m = 1.3
beta_m = 20

# Konversi ke float untuk menghindari overflow
img_float = img.astype(np.float64)
# Rumus: g(x) = alpha * f(x) + beta
img_manual = alpha_m * img_float + beta_m
# np.clip membatasi nilai ke rentang [0, 255]
img_manual = np.clip(img_manual, 0, 255).astype(np.uint8)

# Bandingkan dengan cv2.convertScaleAbs
img_cv = cv2.convertScaleAbs(img, alpha=alpha_m, beta=beta_m)
perbedaan = np.mean(cv2.absdiff(img_manual, img_cv))
print(f"  Perbedaan manual vs cv2: {perbedaan:.4f}")

# ============================================================
# 6. Normalisasi kontras (auto contrast)
# ============================================================
print("\n--- 6. Auto Contrast (Normalisasi) ---")

# Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.normalize memetakan rentang nilai ke [0, 255]
# NORM_MINMAX: min→0, max→255 (memanfaatkan seluruh rentang)
gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
print(f"  Sebelum: min={gray.min()}, max={gray.max()}")
print(f"  Sesudah: min={gray_norm.min()}, max={gray_norm.max()}")

# Normalisasi per channel untuk gambar berwarna
b, g, r = cv2.split(img)
b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
img_auto = cv2.merge([b_norm, g_norm, r_norm])

# ============================================================
# 7. Look-Up Table (LUT) untuk brightness
# ============================================================
print("\n--- 7. LUT Brightness ---")

def buat_lut_brightness(beta):
    """Membuat Look-Up Table untuk brightness adjustment."""
    # Tabel 256 nilai (0-255)
    tabel = np.arange(256, dtype=np.int16) + beta
    # Clip ke rentang valid
    tabel = np.clip(tabel, 0, 255).astype(np.uint8)
    return tabel

# Membuat LUT dengan brightness +40
lut_bright = buat_lut_brightness(40)
# cv2.LUT menerapkan look-up table ke gambar (sangat cepat)
img_lut_bright = cv2.LUT(img, lut_bright)

# LUT brightness -40
lut_dark = buat_lut_brightness(-40)
img_lut_dark = cv2.LUT(img, lut_dark)

print("  LUT +40 dan -40 diterapkan (sangat cepat)")

# ============================================================
# 8. Visualisasi
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Brightness
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 1].imshow(cv2.cvtColor(terang, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Brightness +60")
axes[0, 2].imshow(cv2.cvtColor(gelap, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Brightness -60")
axes[0, 3].imshow(cv2.cvtColor(img_auto, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Auto Contrast")

# Baris 2: Kontras
axes[1, 0].imshow(cv2.cvtColor(img_kontras_tinggi, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Kontras α=1.5")
axes[1, 1].imshow(cv2.cvtColor(img_kontras_rendah, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Kontras α=0.5")
axes[1, 2].imshow(cv2.cvtColor(img_kontras_balanced, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("α=1.5 β=-60")
axes[1, 3].imshow(cv2.cvtColor(hasil_kombinasi[5][0], cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("α=2.0 β=-100")

# Baris 3: LUT + kombinasi
axes[2, 0].imshow(cv2.cvtColor(img_lut_bright, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("LUT +40")
axes[2, 1].imshow(cv2.cvtColor(img_lut_dark, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("LUT -40")
axes[2, 2].imshow(gray, cmap="gray")
axes[2, 2].set_title("Gray Original")
axes[2, 3].imshow(gray_norm, cmap="gray")
axes[2, 3].set_title("Gray Normalized")

for ax in axes.flat:
    ax.axis("off")

plt.suptitle("Percobaan 17: Brightness & Contrast", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "17_brightness_contrast_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 17")
print("=" * 60)
print("  g(x) = α * f(x) + β")
print("  α (alpha) → kontras (>1 naik, <1 turun)")
print("  β (beta)  → brightness (>0 terang, <0 gelap)")
print("  cv2.convertScaleAbs → fungsi utama")
print("  cv2.normalize → auto contrast (NORM_MINMAX)")
print("  cv2.LUT → cepat untuk transformasi look-up")
print("=" * 60)
