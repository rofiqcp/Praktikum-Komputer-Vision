"""
==========================================================================
PERCOBAAN 03: CLAHE (Contrast Limited Adaptive Histogram Equalization)
==========================================================================
CLAHE membagi gambar menjadi tile kecil dan melakukan equalisasi
lokal di setiap tile, dengan pembatasan kontras untuk menghindari
amplifikasi noise. Lebih baik dari global equalization.

Fungsi:
- cv2.createCLAHE(clipLimit, tileGridSize) → buat objek CLAHE
- clahe.apply(src) → terapkan CLAHE
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

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 03: CLAHE")
print("=" * 60)

# ============================================================
# 1. Global Equalization vs CLAHE
# ============================================================
print("\n--- 1. Global vs CLAHE ---")

# Equalisasi global
eq_global = cv2.equalizeHist(gray)

# CLAHE default (clipLimit=2.0, tileGridSize=(8,8))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
eq_clahe = clahe.apply(gray)

print(f"  Original std: {gray.std():.1f}")
print(f"  Global eq std: {eq_global.std():.1f}")
print(f"  CLAHE std: {eq_clahe.std():.1f}")

# ============================================================
# 2. Variasi clipLimit
# ============================================================
print("\n--- 2. Variasi clipLimit ---")

clip_values = [1.0, 2.0, 4.0, 8.0, 20.0, 40.0]
clip_results = []

for clip in clip_values:
    # clipLimit membatasi amplifikasi kontras per tile
    clahe_v = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    result = clahe_v.apply(gray)
    clip_results.append(result)
    print(f"  clipLimit={clip:5.1f}: mean={result.mean():.1f}, std={result.std():.1f}")

# ============================================================
# 3. Variasi tileGridSize
# ============================================================
print("\n--- 3. Variasi tileGridSize ---")

tile_sizes = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
tile_results = []

for tsize in tile_sizes:
    # Tile lebih kecil → lebih lokal, lebih banyak adaptasi
    clahe_t = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tsize)
    result = clahe_t.apply(gray)
    tile_results.append(result)
    print(f"  tileGridSize={tsize}: mean={result.mean():.1f}")

# ============================================================
# 4. CLAHE pada Gambar Berwarna (LAB)
# ============================================================
print("\n--- 4. CLAHE pada Gambar Berwarna ---")

# Konversi ke LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# Terapkan CLAHE pada channel L (lightness)
clahe_color = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
lab[:, :, 0] = clahe_color.apply(lab[:, :, 0])
# Konversi kembali ke BGR
clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
print("  CLAHE pada channel L (LAB color space)")

# ============================================================
# 5. CLAHE pada Gambar Gelap
# ============================================================
print("\n--- 5. CLAHE pada Gambar Gelap ---")

dark = cv2.convertScaleAbs(gray, alpha=0.3, beta=-10)
dark_global = cv2.equalizeHist(dark)
dark_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(dark)
print(f"  Gelap mean: {dark.mean():.1f}")
print(f"  Global eq: {dark_global.mean():.1f}")
print(f"  CLAHE: {dark_clahe.mean():.1f}")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Perbandingan utama
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(eq_global, cmap='gray')
axes[0, 1].set_title("Global EQ")
axes[0, 1].axis("off")

axes[0, 2].imshow(eq_clahe, cmap='gray')
axes[0, 2].set_title("CLAHE (clip=2)")
axes[0, 2].axis("off")

axes[0, 3].hist(eq_clahe.ravel(), 256, [0, 256], color='coral')
axes[0, 3].set_title("Histogram CLAHE")
axes[0, 3].set_xlim(0, 256)

# Baris 2: Variasi clipLimit
for i, (clip, res) in enumerate(zip([1.0, 4.0, 20.0, 40.0],
                                     [clip_results[0], clip_results[2],
                                      clip_results[4], clip_results[5]])):
    axes[1, i].imshow(res, cmap='gray')
    axes[1, i].set_title(f"clip={clip}")
    axes[1, i].axis("off")

# Baris 3: Warna + gelap
axes[2, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("Original Warna")
axes[2, 0].axis("off")

axes[2, 1].imshow(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("CLAHE Warna (LAB)")
axes[2, 1].axis("off")

axes[2, 2].imshow(dark, cmap='gray')
axes[2, 2].set_title("Gelap")
axes[2, 2].axis("off")

axes[2, 3].imshow(dark_clahe, cmap='gray')
axes[2, 3].set_title("CLAHE Gelap")
axes[2, 3].axis("off")

plt.suptitle("Percobaan 03: CLAHE", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "03_clahe_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 03")
print("=" * 60)
print("""
1. CLAHE = adaptive histogram equalization + clip limit
2. clipLimit mengontrol batas amplifikasi kontras (anti-noise)
3. tileGridSize mengontrol ukuran area lokal
4. Untuk gambar berwarna: terapkan CLAHE pada channel L (LAB)
5. CLAHE lebih baik dari global EQ untuk gambar dengan pencahayaan
   tidak merata
""")
