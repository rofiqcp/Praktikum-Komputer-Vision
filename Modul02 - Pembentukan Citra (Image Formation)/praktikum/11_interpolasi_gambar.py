"""
==========================================================================
PERCOBAAN 11: INTERPOLASI GAMBAR
==========================================================================
Interpolasi menghitung nilai piksel baru saat resize/transformasi.
Metode berbeda menghasilkan kualitas dan kecepatan berbeda.

Metode interpolasi OpenCV:
- INTER_NEAREST  : Tetangga terdekat (cepat, jagged)
- INTER_LINEAR   : Bilinear (default, smooth)
- INTER_CUBIC    : Bicubic 4×4 (lebih smooth)
- INTER_LANCZOS4 : Lanczos 8×8 (terbaik, lambat)
- INTER_AREA     : Terbaik untuk downsampling
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

img = cv2.imread(os.path.join(IMAGE_DIR, "baboon.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()
img = cv2.resize(img, (400, 400))

print("=" * 60)
print("PERCOBAAN 11: INTERPOLASI GAMBAR")
print("=" * 60)

# ============================================================
# 1. Upsample kecil → besar (melihat perbedaan kualitas)
# ============================================================
print("\n--- 1. Upsample 40→400 ---")

# Perkecil ke 40×40 dulu
img_kecil = cv2.resize(img, (40, 40))

metode = {
    "NEAREST":  cv2.INTER_NEAREST,
    "LINEAR":   cv2.INTER_LINEAR,
    "CUBIC":    cv2.INTER_CUBIC,
    "LANCZOS4": cv2.INTER_LANCZOS4,
}

hasil_up = {}
for nama, m in metode.items():
    t0 = time.time()
    hasil_up[nama] = cv2.resize(img_kecil, (400, 400), interpolation=m)
    dt = (time.time() - t0) * 1000
    print(f"  {nama:10s}: {dt:.2f} ms")

# ============================================================
# 2. Downsample 400→100
# ============================================================
print("\n--- 2. Downsample 400→100 ---")

metode_down = {**metode, "AREA": cv2.INTER_AREA}
hasil_down = {}
for nama, m in metode_down.items():
    hasil_down[nama] = cv2.resize(img, (100, 100), interpolation=m)
    print(f"  {nama}")

# ============================================================
# 3. Zoom pada area detail (crop & compare)
# ============================================================
print("\n--- 3. Perbandingan Detail ---")

# Crop area kecil untuk melihat perbedaan
for nama, im in hasil_up.items():
    crop = im[150:250, 150:250]
    psnr = cv2.PSNR(img[150:250, 150:250], crop)
    print(f"  {nama:10s}: PSNR = {psnr:.2f} dB")

# ============================================================
# 4. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original 400×400")
axes[0, 1].imshow(cv2.cvtColor(img_kecil, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Kecil 40×40")

for i, nama in enumerate(["NEAREST", "LINEAR"]):
    axes[0, i+2].imshow(cv2.cvtColor(hasil_up[nama], cv2.COLOR_BGR2RGB))
    axes[0, i+2].set_title(f"Up: {nama}")

for i, nama in enumerate(["CUBIC", "LANCZOS4"]):
    axes[1, i].imshow(cv2.cvtColor(hasil_up[nama], cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"Up: {nama}")

# Perbandingan zoom crop
for i, nama in enumerate(["NEAREST", "LANCZOS4"]):
    crop = hasil_up[nama][150:250, 150:250]
    axes[1, i+2].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[1, i+2].set_title(f"Zoom: {nama}")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Percobaan 11: Metode Interpolasi", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "11_interpolasi_gambar_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
