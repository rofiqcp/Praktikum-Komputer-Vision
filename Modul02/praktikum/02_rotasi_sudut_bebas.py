"""
==========================================================================
PERCOBAAN 02: ROTASI DENGAN SUDUT BEBAS
==========================================================================
Rotasi gambar dengan sudut bebas, titik pusat custom, dan penanganan
canvas agar gambar tidak terpotong.

Fungsi:
- cv2.getRotationMatrix2D(center, angle, scale) → Matriks rotasi 2×3
- cv2.warpAffine(src, M, dsize) → Terapkan rotasi
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

img = cv2.imread(os.path.join(IMAGE_DIR, "bintang.png"))
if img is None:
    img = cv2.imread(os.path.join(IMAGE_DIR, "grid.png"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()

img = cv2.resize(img, (300, 300))
h, w = img.shape[:2]
pusat = (w // 2, h // 2)

print("=" * 60)
print("PERCOBAAN 02: ROTASI SUDUT BEBAS")
print("=" * 60)

# ============================================================
# 1. Rotasi berbagai sudut
# ============================================================
print("\n--- 1. Berbagai Sudut ---")
sudut_list = [15, 37, 72, 123, 200, 315]
hasil = {}
for s in sudut_list:
    M = cv2.getRotationMatrix2D(pusat, s, 1.0)
    hasil[s] = cv2.warpAffine(img, M, (w, h))
    print(f"  Sudut {s}°")

# ============================================================
# 2. Rotasi tanpa crop dengan perhitungan canvas otomatis
# ============================================================
print("\n--- 2. Rotasi Tanpa Crop ---")

def rotasi_full(img, sudut):
    """Rotasi gambar dengan canvas diperbesar agar tidak crop."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), sudut, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(img, M, (nw, nh), borderValue=(255, 255, 255))

img_full_37 = rotasi_full(img, 37)
img_full_72 = rotasi_full(img, 72)
print(f"  37° full: {img_full_37.shape[1]}×{img_full_37.shape[0]}")

# ============================================================
# 3. Rotasi dari titik pusat berbeda
# ============================================================
print("\n--- 3. Pusat Rotasi Berbeda ---")
pusat_list = [(0, 0), (w, 0), (w // 2, h // 2), (w, h)]
hasil_pusat = {}
for p in pusat_list:
    M = cv2.getRotationMatrix2D(p, 30, 1.0)
    hasil_pusat[p] = cv2.warpAffine(img, M, (w, h))
    print(f"  Pusat {p}")

# ============================================================
# 4. Rotasi + scaling bersamaan
# ============================================================
print("\n--- 4. Rotasi + Scale ---")
scale_list = [0.5, 0.75, 1.0, 1.25, 1.5]
hasil_scale = {}
for s in scale_list:
    M = cv2.getRotationMatrix2D(pusat, 45, s)
    hasil_scale[s] = cv2.warpAffine(img, M, (w, h))
    print(f"  45° scale={s}")

# ============================================================
# 5. Membangun matriks rotasi manual
# ============================================================
print("\n--- 5. Matriks Rotasi Manual ---")
sudut_rad = np.deg2rad(30)
cos_t = np.cos(sudut_rad)
sin_t = np.sin(sudut_rad)
cx, cy = w / 2, h / 2
# Matriks rotasi sekitar pusat (cx, cy):
# M = [[cos, -sin, cx*(1-cos)+cy*sin],
#      [sin,  cos, cy*(1-cos)-cx*sin]]
M_manual = np.float32([
    [cos_t, -sin_t, cx * (1 - cos_t) + cy * sin_t],
    [sin_t,  cos_t, cy * (1 - cos_t) - cx * sin_t]
])
M_cv = cv2.getRotationMatrix2D((cx, cy), 30, 1.0)
print(f"  Manual:\n{M_manual}")
print(f"  OpenCV:\n{M_cv}")
print(f"  Identik: {np.allclose(M_manual, M_cv)}")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
for i, s in enumerate([15, 37, 72]):
    axes[0, i+1].imshow(cv2.cvtColor(hasil[s], cv2.COLOR_BGR2RGB))
    axes[0, i+1].set_title(f"{s}° (crop)")

axes[1, 0].imshow(cv2.cvtColor(img_full_37, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("37° (full)")
axes[1, 1].imshow(cv2.cvtColor(img_full_72, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("72° (full)")
axes[1, 2].imshow(cv2.cvtColor(hasil_scale[0.5], cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("45° scale=0.5")
axes[1, 3].imshow(cv2.cvtColor(hasil_scale[1.5], cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("45° scale=1.5")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Percobaan 02: Rotasi Sudut Bebas", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "02_rotasi_sudut_bebas_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
