"""
==========================================================================
PERCOBAAN 04: TRANSFORMASI AFFINE
==========================================================================
Transformasi affine memetakan 3 titik sumber ke 3 titik tujuan.
Mempertahankan garis paralel tetapi memungkinkan rotasi, skala,
translasi, dan shear secara bersamaan.

Fungsi:
- cv2.getAffineTransform(src_pts, dst_pts) → Matriks affine 2×3
- cv2.warpAffine(src, M, dsize) → Terapkan transformasi
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

img = cv2.imread(os.path.join(IMAGE_DIR, "grid.png"))
if img is None:
    img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()

img = cv2.resize(img, (300, 300))
h, w = img.shape[:2]

print("=" * 60)
print("PERCOBAAN 04: TRANSFORMASI AFFINE")
print("=" * 60)

# ============================================================
# 1. Transformasi affine dasar (3 titik)
# ============================================================
print("\n--- 1. Affine Dasar ---")

# 3 titik sumber (segitiga pada gambar asli)
src_pts = np.float32([[50, 50], [200, 50], [50, 200]])
# 3 titik tujuan (segitiga pada gambar hasil)
dst_pts = np.float32([[10, 100], [200, 50], [100, 250]])

# cv2.getAffineTransform menghitung matriks affine 2×3
M = cv2.getAffineTransform(src_pts, dst_pts)
print(f"  Matriks Affine:\n{M}")

# Terapkan transformasi
img_affine1 = cv2.warpAffine(img, M, (w, h))

# ============================================================
# 2. Shearing horizontal
# ============================================================
print("\n--- 2. Shearing Horizontal ---")

# Matriks shear: [[1, shx, 0], [0, 1, 0]]
shx = 0.3  # Faktor shear horizontal
M_shear_h = np.float32([[1, shx, 0],
                         [0, 1,   0]])
img_shear_h = cv2.warpAffine(img, M_shear_h, (int(w + h * abs(shx)), h))
print(f"  Shear horizontal: shx={shx}")

# ============================================================
# 3. Shearing vertikal
# ============================================================
print("\n--- 3. Shearing Vertikal ---")

shy = 0.3  # Faktor shear vertikal
M_shear_v = np.float32([[1, 0,   0],
                         [shy, 1, 0]])
img_shear_v = cv2.warpAffine(img, M_shear_v, (w, int(h + w * abs(shy))))
print(f"  Shear vertikal: shy={shy}")

# ============================================================
# 4. Rotasi + translasi + skala sekaligus
# ============================================================
print("\n--- 4. Rotasi + Translasi + Skala ---")

sudut = 25
skala = 0.8
tx, ty = 30, 20
rad = np.deg2rad(sudut)
cos_a = skala * np.cos(rad)
sin_a = skala * np.sin(rad)

M_combo = np.float32([
    [cos_a, -sin_a, tx + w/2*(1-cos_a) + h/2*sin_a],
    [sin_a,  cos_a, ty + h/2*(1-cos_a) - w/2*sin_a]
])
img_combo = cv2.warpAffine(img, M_combo, (w, h))
print(f"  Sudut={sudut}°, Skala={skala}, Translasi=({tx},{ty})")

# ============================================================
# 5. Refleksi (pencerminan) sebagai affine transform
# ============================================================
print("\n--- 5. Refleksi via Affine ---")

# Refleksi horizontal: [[−1, 0, w], [0, 1, 0]]
M_ref_h = np.float32([[-1, 0, w - 1],
                        [0, 1, 0]])
img_ref_h = cv2.warpAffine(img, M_ref_h, (w, h))

# Refleksi vertikal: [[1, 0, 0], [0, −1, h]]
M_ref_v = np.float32([[1, 0, 0],
                        [0, -1, h - 1]])
img_ref_v = cv2.warpAffine(img, M_ref_v, (w, h))

print("  Refleksi H dan V via matriks affine")

# ============================================================
# 6. Inverse affine transform
# ============================================================
print("\n--- 6. Inverse Affine ---")

# cv2.invertAffineTransform menghitung kebalikan transformasi
M_inv = cv2.invertAffineTransform(M)
img_inv = cv2.warpAffine(img_affine1, M_inv, (w, h))
diff = np.mean(cv2.absdiff(img, img_inv))
print(f"  Perbedaan original vs inverse: {diff:.2f}")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 1].imshow(cv2.cvtColor(img_affine1, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Affine (3 titik)")
axes[0, 2].imshow(cv2.cvtColor(img_shear_h, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"Shear H ({shx})")
axes[0, 3].imshow(cv2.cvtColor(img_shear_v, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title(f"Shear V ({shy})")

axes[1, 0].imshow(cv2.cvtColor(img_combo, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Rot+Trans+Scale")
axes[1, 1].imshow(cv2.cvtColor(img_ref_h, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Refleksi H")
axes[1, 2].imshow(cv2.cvtColor(img_ref_v, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Refleksi V")
axes[1, 3].imshow(cv2.cvtColor(img_inv, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Inverse Affine")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Percobaan 04: Transformasi Affine", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "04_transformasi_affine_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
