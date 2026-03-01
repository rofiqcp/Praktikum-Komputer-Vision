"""
==========================================================================
PERCOBAAN 10: RESIZE DAN SCALING GAMBAR
==========================================================================
Program ini mempelajari cara mengubah ukuran gambar (resize/scaling)
menggunakan berbagai metode interpolasi.

Fungsi utama:
- cv2.resize(src, dsize, fx, fy, interpolation)
  dsize = ukuran output (width, height) — PERHATIAN: width dulu!
  fx, fy = faktor skala horizontal dan vertikal
  interpolation = metode interpolasi

Metode Interpolasi:
- INTER_NEAREST  : Nearest neighbor (cepat, kasar, untuk mask)
- INTER_LINEAR   : Bilinear (default, bagus untuk memperkecil)
- INTER_CUBIC    : Bicubic (lebih halus, lebih lambat)
- INTER_LANCZOS4 : Lanczos 8×8 (kualitas terbaik, paling lambat)
- INTER_AREA     : Resampling (terbaik untuk memperkecil)
==========================================================================
"""

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 10: RESIZE DAN SCALING GAMBAR")
print("=" * 60)

# Membaca gambar
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

tinggi, lebar = img.shape[:2]
print(f"[INFO] Gambar asli: {lebar}×{tinggi}")

# ============================================================
# 1. Resize dengan ukuran absolut (dsize)
# ============================================================
print("\n--- 1. Resize dengan Ukuran Absolut ---")

# cv2.resize(src, (width, height)) — PERHATIAN: (lebar, tinggi), bukan (tinggi, lebar)!
img_320x240 = cv2.resize(img, (320, 240))
print(f"  Resize ke 320×240: {img_320x240.shape}")

img_800x600 = cv2.resize(img, (800, 600))
print(f"  Resize ke 800×600: {img_800x600.shape}")

img_100x100 = cv2.resize(img, (100, 100))
print(f"  Resize ke 100×100: {img_100x100.shape}")

# ============================================================
# 2. Resize dengan faktor skala (fx, fy)
# ============================================================
print("\n--- 2. Resize dengan Faktor Skala ---")

# fx = faktor skala horizontal, fy = faktor skala vertikal
# dsize=(0,0) berarti gunakan fx dan fy untuk menentukan ukuran
img_half = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
print(f"  50% (fx=0.5, fy=0.5): {img_half.shape}")

img_double = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)
print(f"  200% (fx=2.0, fy=2.0): {img_double.shape}")

# Resize non-proporsional (mengubah aspect ratio)
img_stretch = cv2.resize(img, (0, 0), fx=2.0, fy=0.5)
print(f"  Stretch (fx=2.0, fy=0.5): {img_stretch.shape}")

# ============================================================
# 3. Resize dengan menjaga aspect ratio
# ============================================================
print("\n--- 3. Resize Proporsional ---")

def resize_proporsional(img, target_width=None, target_height=None):
    """Resize gambar dengan menjaga aspect ratio."""
    h, w = img.shape[:2]

    if target_width is not None:
        # Menghitung tinggi baru berdasarkan rasio lebar
        ratio = target_width / w
        new_size = (target_width, int(h * ratio))
    elif target_height is not None:
        # Menghitung lebar baru berdasarkan rasio tinggi
        ratio = target_height / h
        new_size = (int(w * ratio), target_height)
    else:
        return img

    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

# Resize ke lebar 300 dengan menjaga proporsi
img_prop_w = resize_proporsional(img, target_width=300)
print(f"  Target lebar 300: {img_prop_w.shape[1]}×{img_prop_w.shape[0]}")

# Resize ke tinggi 200 dengan menjaga proporsi
img_prop_h = resize_proporsional(img, target_height=200)
print(f"  Target tinggi 200: {img_prop_h.shape[1]}×{img_prop_h.shape[0]}")

# ============================================================
# 4. Perbandingan metode interpolasi
# ============================================================
print("\n--- 4. Perbandingan Metode Interpolasi ---")

# Memperkecil gambar dulu, lalu memperbesar kembali
# (untuk melihat perbedaan interpolasi dengan jelas)
img_kecil = cv2.resize(img, (80, 60))

# Memperbesar kembali dengan berbagai metode
metode_interp = [
    (cv2.INTER_NEAREST, "NEAREST"),
    (cv2.INTER_LINEAR, "LINEAR"),
    (cv2.INTER_CUBIC, "CUBIC"),
    (cv2.INTER_LANCZOS4, "LANCZOS4"),
]

hasil_interp = {}
for interp, nama in metode_interp:
    # Mengukur waktu eksekusi
    start = time.time()
    # Memperbesar 8× dari gambar kecil
    img_besar = cv2.resize(img_kecil, (640, 480), interpolation=interp)
    waktu = time.time() - start

    hasil_interp[nama] = img_besar
    print(f"  {nama:10s}: {waktu*1000:.2f} ms")

# ============================================================
# 5. INTER_AREA: Terbaik untuk memperkecil
# ============================================================
print("\n--- 5. INTER_AREA untuk Downsampling ---")

# INTER_AREA menghasilkan downsampling yang lebih bersih karena
# menghitung rata-rata area, bukan sampling titik

# Memperkecil gambar 10× dengan berbagai metode
ukuran_kecil = (lebar // 10, tinggi // 10)

img_nearest_small = cv2.resize(img, ukuran_kecil, interpolation=cv2.INTER_NEAREST)
img_linear_small = cv2.resize(img, ukuran_kecil, interpolation=cv2.INTER_LINEAR)
img_area_small = cv2.resize(img, ukuran_kecil, interpolation=cv2.INTER_AREA)

print(f"  Diperkecil ke {ukuran_kecil[0]}×{ukuran_kecil[1]}")
print("  INTER_AREA biasanya menghasilkan kualitas terbaik saat memperkecil")

# ============================================================
# 6. Visualisasi perbandingan interpolasi (upscaling)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Gambar asli
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"Asli ({lebar}×{tinggi})")
axes[0, 0].axis("off")

# Gambar kecil
axes[0, 1].imshow(cv2.cvtColor(img_kecil, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Diperkecil (80×60)")
axes[0, 1].axis("off")

# NEAREST (kasar, piksel terlihat jelas)
axes[0, 2].imshow(cv2.cvtColor(hasil_interp["NEAREST"], cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("NEAREST\n(Cepat, kotak-kotak)")
axes[0, 2].axis("off")

# LINEAR
axes[1, 0].imshow(cv2.cvtColor(hasil_interp["LINEAR"], cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("LINEAR (Bilinear)\n(Default, cukup halus)")
axes[1, 0].axis("off")

# CUBIC
axes[1, 1].imshow(cv2.cvtColor(hasil_interp["CUBIC"], cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("CUBIC (Bicubic)\n(Lebih halus, lebih lambat)")
axes[1, 1].axis("off")

# LANCZOS4
axes[1, 2].imshow(cv2.cvtColor(hasil_interp["LANCZOS4"], cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("LANCZOS4\n(Kualitas tertinggi, paling lambat)")
axes[1, 2].axis("off")

plt.suptitle("Percobaan 10: Perbandingan Metode Interpolasi (80×60 → 640×480)",
             fontsize=14, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "10_resize_interpolasi_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# 7. Visualisasi downsampling
# ============================================================

fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))

axes2[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes2[0].set_title(f"Asli ({lebar}×{tinggi})")
axes2[0].axis("off")

for i, (im, nama) in enumerate([
    (img_nearest_small, "NEAREST"),
    (img_linear_small, "LINEAR"),
    (img_area_small, "AREA (Terbaik)")
]):
    axes2[i+1].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    axes2[i+1].set_title(f"{nama}\n({im.shape[1]}×{im.shape[0]})")
    axes2[i+1].axis("off")

plt.suptitle("Downsampling: Perbandingan Metode", fontsize=14, fontweight="bold")
plt.tight_layout()

output_path2 = os.path.join(OUTPUT_DIR, "10_downsampling_hasil.png")
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Downsampling disimpan di: {output_path2}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 10")
print("=" * 60)
print("Resize gambar:")
print("  1. cv2.resize(img, (w,h))   → Ukuran absolut")
print("  2. cv2.resize(img, (0,0), fx=, fy=) → Faktor skala")
print("\nRekomendasi interpolasi:")
print("  Memperbesar → INTER_CUBIC atau INTER_LANCZOS4")
print("  Memperkecil → INTER_AREA")
print("  Kecepatan   → INTER_NEAREST (untuk mask/label)")
print("  Default     → INTER_LINEAR")
print("=" * 60)
