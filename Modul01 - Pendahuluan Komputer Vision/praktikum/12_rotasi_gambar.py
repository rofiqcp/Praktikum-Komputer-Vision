"""
==========================================================================
PERCOBAAN 12: ROTASI GAMBAR
==========================================================================
Program ini mempelajari cara merotasi gambar menggunakan transformasi
affine. Rotasi dilakukan menggunakan matriks rotasi 2×3.

Fungsi utama:
- cv2.getRotationMatrix2D(center, angle, scale) → Matriks rotasi 2×3
- cv2.warpAffine(src, M, dsize) → Terapkan transformasi affine

Parameter:
- center : titik pusat rotasi (x, y)
- angle  : sudut rotasi dalam derajat (positif = counter-clockwise)
- scale  : faktor skala (1.0 = ukuran sama)
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
print("PERCOBAAN 12: ROTASI GAMBAR")
print("=" * 60)

# Membaca gambar
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

tinggi, lebar = img.shape[:2]
print(f"[INFO] Gambar asli: {lebar}×{tinggi}")

# ============================================================
# 1. Rotasi dasar (pusat = tengah gambar)
# ============================================================
print("\n--- 1. Rotasi Dasar ---")

# Menentukan titik pusat rotasi (tengah gambar)
pusat = (lebar // 2, tinggi // 2)

# cv2.getRotationMatrix2D(center, angle, scale)
# Mengembalikan matriks transformasi 2×3 untuk rotasi
# angle positif = putar counter-clockwise (berlawanan jarum jam)
M_45 = cv2.getRotationMatrix2D(pusat, 45, 1.0)
print(f"  Matriks rotasi 45°:\n{M_45}")

# cv2.warpAffine(src, M, dsize) menerapkan transformasi affine
# dsize = ukuran output (width, height)
img_rot45 = cv2.warpAffine(img, M_45, (lebar, tinggi))
print("  Rotasi 45° berlawanan jarum jam")

# ============================================================
# 2. Rotasi berbagai sudut
# ============================================================
print("\n--- 2. Rotasi Berbagai Sudut ---")

sudut_list = [0, 30, 45, 90, 135, 180, 270, -45]
hasil_rotasi = {}

for sudut in sudut_list:
    # Membuat matriks rotasi untuk setiap sudut
    M = cv2.getRotationMatrix2D(pusat, sudut, 1.0)
    # Menerapkan rotasi
    img_rot = cv2.warpAffine(img, M, (lebar, tinggi))
    hasil_rotasi[sudut] = img_rot
    print(f"  Rotasi {sudut:4d}° selesai")

# ============================================================
# 3. Rotasi tanpa cropping (memperbesar canvas)
# ============================================================
print("\n--- 3. Rotasi Tanpa Cropping ---")

def rotasi_tanpa_crop(img, sudut):
    """Merotasi gambar tanpa memotong bagian yang keluar batas."""
    h, w = img.shape[:2]
    pusat = (w // 2, h // 2)

    # Membuat matriks rotasi
    M = cv2.getRotationMatrix2D(pusat, sudut, 1.0)

    # Menghitung ukuran baru agar seluruh gambar terlihat
    # Menggunakan sudut absolut dalam radian
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # Lebar dan tinggi baru
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Menggeser pusat rotasi ke pusat canvas baru
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Menerapkan rotasi dengan canvas baru
    return cv2.warpAffine(img, M, (new_w, new_h))

# Rotasi 30° tanpa crop
img_rot30_nocrop = rotasi_tanpa_crop(img, 30)
print(f"  30° tanpa crop: {img_rot30_nocrop.shape[1]}×{img_rot30_nocrop.shape[0]}")

# Rotasi 45° tanpa crop
img_rot45_nocrop = rotasi_tanpa_crop(img, 45)
print(f"  45° tanpa crop: {img_rot45_nocrop.shape[1]}×{img_rot45_nocrop.shape[0]}")

# ============================================================
# 4. Rotasi dengan skala (zoom + rotasi)
# ============================================================
print("\n--- 4. Rotasi + Zoom ---")

# Rotasi 30° dengan zoom 0.5x (perkecil)
M_zoom_out = cv2.getRotationMatrix2D(pusat, 30, 0.5)
img_rot_zoom_out = cv2.warpAffine(img, M_zoom_out, (lebar, tinggi))

# Rotasi 30° dengan zoom 1.5x (perbesar)
M_zoom_in = cv2.getRotationMatrix2D(pusat, 30, 1.5)
img_rot_zoom_in = cv2.warpAffine(img, M_zoom_in, (lebar, tinggi))

print("  Rotasi 30° + zoom 0.5× dan 1.5×")

# ============================================================
# 5. Rotasi dengan titik pusat berbeda
# ============================================================
print("\n--- 5. Pusat Rotasi Berbeda ---")

# Pusat di sudut kiri atas (0, 0)
M_sudut = cv2.getRotationMatrix2D((0, 0), 30, 1.0)
img_rot_sudut = cv2.warpAffine(img, M_sudut, (lebar, tinggi))

# Pusat di kanan bawah
M_kanan = cv2.getRotationMatrix2D((lebar, tinggi), 30, 1.0)
img_rot_kanan = cv2.warpAffine(img, M_kanan, (lebar, tinggi))

print("  Rotasi dengan pusat di (0,0) dan (w,h)")

# ============================================================
# 6. Rotasi 90° menggunakan cv2.rotate() (lebih cepat)
# ============================================================
print("\n--- 6. Rotasi 90° Cepat (cv2.rotate) ---")

# cv2.rotate() hanya untuk rotasi 90°, 180°, 270° (lebih cepat dari warpAffine)
# cv2.ROTATE_90_CLOCKWISE     : 90° searah jarum jam
# cv2.ROTATE_180              : 180°
# cv2.ROTATE_90_COUNTERCLOCKWISE : 90° berlawanan jarum jam

img_90cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img_180 = cv2.rotate(img, cv2.ROTATE_180)
img_90ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

print(f"  90° CW:  {img_90cw.shape[1]}×{img_90cw.shape[0]}")
print(f"  180°:    {img_180.shape[1]}×{img_180.shape[0]}")
print(f"  90° CCW: {img_90ccw.shape[1]}×{img_90ccw.shape[0]}")

# ============================================================
# 7. Visualisasi
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Rotasi berbagai sudut
for i, sudut in enumerate([0, 45, 90, 180]):
    axes[0, i].imshow(cv2.cvtColor(hasil_rotasi[sudut], cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"Rotasi {sudut}°")
    axes[0, i].axis("off")

# Baris 2: Tanpa crop, zoom
axes[1, 0].imshow(cv2.cvtColor(img_rot45, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("45° (dengan crop)")
axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(img_rot45_nocrop, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("45° (tanpa crop)")
axes[1, 1].axis("off")

axes[1, 2].imshow(cv2.cvtColor(img_rot_zoom_out, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("30° + Zoom 0.5×")
axes[1, 2].axis("off")

axes[1, 3].imshow(cv2.cvtColor(img_rot_zoom_in, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("30° + Zoom 1.5×")
axes[1, 3].axis("off")

# Baris 3: Pusat berbeda, rotate cepat
axes[2, 0].imshow(cv2.cvtColor(img_rot_sudut, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("Pusat: (0,0)")
axes[2, 0].axis("off")

axes[2, 1].imshow(cv2.cvtColor(img_90cw, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("90° CW (cv2.rotate)")
axes[2, 1].axis("off")

axes[2, 2].imshow(cv2.cvtColor(img_180, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title("180° (cv2.rotate)")
axes[2, 2].axis("off")

axes[2, 3].imshow(cv2.cvtColor(img_90ccw, cv2.COLOR_BGR2RGB))
axes[2, 3].set_title("90° CCW (cv2.rotate)")
axes[2, 3].axis("off")

plt.suptitle("Percobaan 12: Rotasi Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "12_rotasi_gambar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 12")
print("=" * 60)
print("  1. cv2.getRotationMatrix2D() → Matriks rotasi 2×3")
print("  2. cv2.warpAffine()          → Terapkan rotasi")
print("  3. Rotasi tanpa crop         → Perbesar canvas dulu")
print("  4. Rotasi + scale            → Zoom sambil putar")
print("  5. cv2.rotate()              → 90°/180°/270° cepat")
print("=" * 60)
