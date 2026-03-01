"""
==========================================================================
PERCOBAAN 12: DETEKSI TEPI SOBEL
==========================================================================
Operator Sobel mendeteksi tepi (edge) menggunakan turunan pertama.
Menghasilkan gradient horizontal (Gx) dan vertikal (Gy).
Magnitude: G = sqrt(Gx² + Gy²), Arah: θ = arctan(Gy/Gx)

Fungsi:
- cv2.Sobel(src, ddepth, dx, dy, ksize) → gradient Sobel
- cv2.Scharr(src, ddepth, dx, dy) → versi lebih akurat (ksize=3)
- cv2.magnitude(Gx, Gy) → hitung magnitude
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

img = cv2.imread(os.path.join(IMAGE_DIR, "garis_tepi.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 12: DETEKSI TEPI SOBEL")
print("=" * 60)

# ============================================================
# 1. Sobel Horizontal (dx=1, dy=0) → deteksi tepi vertikal
# ============================================================
print("\n--- 1. Sobel Horizontal (Gx) ---")

# CV_64F agar tidak kehilangan informasi negatif
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
# Konversi ke absolut untuk visualisasi
abs_sobel_x = np.abs(sobel_x).astype(np.uint8)
print(f"  Gx range: [{sobel_x.min():.0f}, {sobel_x.max():.0f}]")

# ============================================================
# 2. Sobel Vertikal (dx=0, dy=1) → deteksi tepi horizontal
# ============================================================
print("\n--- 2. Sobel Vertikal (Gy) ---")

sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
abs_sobel_y = np.abs(sobel_y).astype(np.uint8)
print(f"  Gy range: [{sobel_y.min():.0f}, {sobel_y.max():.0f}]")

# ============================================================
# 3. Magnitude Gradient
# ============================================================
print("\n--- 3. Magnitude dan Arah ---")

# Magnitude: G = sqrt(Gx² + Gy²)
magnitude = cv2.magnitude(sobel_x, sobel_y)
mag_uint8 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Arah gradient: θ = arctan2(Gy, Gx)
direction = np.arctan2(sobel_y, sobel_x)
# Konversi ke derajat [0, 360]
dir_deg = (np.degrees(direction) + 360) % 360
print(f"  Magnitude range: [{magnitude.min():.0f}, {magnitude.max():.0f}]")
print(f"  Direction range: [{dir_deg.min():.1f}°, {dir_deg.max():.1f}°]")

# ============================================================
# 4. Variasi ksize
# ============================================================
print("\n--- 4. Variasi ksize ---")

ksize_list = [1, 3, 5, 7]
ksize_results = []

for ks in ksize_list:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ks)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ks)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ksize_results.append(mag_norm)
    print(f"  ksize={ks}: max magnitude={mag.max():.0f}")

# ============================================================
# 5. Scharr Operator (lebih akurat dari Sobel ksize=3)
# ============================================================
print("\n--- 5. Scharr Operator ---")

scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
scharr_mag = cv2.magnitude(scharr_x, scharr_y)
scharr_norm = cv2.normalize(scharr_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Bandingkan Sobel ksize=3 vs Scharr
diff = cv2.absdiff(ksize_results[1], scharr_norm).mean()
print(f"  Scharr max magnitude: {scharr_mag.max():.0f}")
print(f"  Mean diff Sobel(3) vs Scharr: {diff:.2f}")

# ============================================================
# 6. Sobel pada Gambar Berwarna
# ============================================================
print("\n--- 6. Sobel Berwarna ---")

# Hitung Sobel per channel
edges_bgr = np.zeros_like(img)
for c in range(3):
    gx = cv2.Sobel(img[:, :, c], cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img[:, :, c], cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edges_bgr[:, :, c] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
print("  Sobel diterapkan per channel BGR")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Dasar
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(abs_sobel_x, cmap='gray')
axes[0, 1].set_title("Sobel X (Gx)")
axes[0, 1].axis("off")

axes[0, 2].imshow(abs_sobel_y, cmap='gray')
axes[0, 2].set_title("Sobel Y (Gy)")
axes[0, 2].axis("off")

axes[0, 3].imshow(mag_uint8, cmap='gray')
axes[0, 3].set_title("Magnitude")
axes[0, 3].axis("off")

# Baris 2: ksize variasi
for i, (ks, res) in enumerate(zip(ksize_list, ksize_results)):
    axes[1, i].imshow(res, cmap='gray')
    axes[1, i].set_title(f"ksize={ks}")
    axes[1, i].axis("off")

# Baris 3: Scharr + direction + warna
axes[2, 0].imshow(scharr_norm, cmap='gray')
axes[2, 0].set_title("Scharr")
axes[2, 0].axis("off")

# Arah gradient sebagai heatmap
axes[2, 1].imshow(dir_deg, cmap='hsv')
axes[2, 1].set_title("Arah Gradient")
axes[2, 1].axis("off")

axes[2, 2].imshow(cv2.cvtColor(edges_bgr, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title("Sobel Berwarna")
axes[2, 2].axis("off")

axes[2, 3].axis("off")

plt.suptitle("Percobaan 12: Deteksi Tepi Sobel", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "12_sobel_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 12")
print("=" * 60)
print("""
1. cv2.Sobel(src, ddepth, dx, dy, ksize) → turunan pertama
2. Gx (dx=1,dy=0) mendeteksi tepi vertikal
3. Gy (dx=0,dy=1) mendeteksi tepi horizontal
4. Magnitude G = √(Gx² + Gy²), Arah θ = arctan2(Gy, Gx)
5. ksize lebih besar → bisa mendeteksi tepi pada skala berbeda
6. Scharr lebih akurat dari Sobel untuk ksize=3
""")
