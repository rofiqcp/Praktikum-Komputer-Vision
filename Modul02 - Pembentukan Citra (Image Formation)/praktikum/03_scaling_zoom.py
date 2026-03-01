"""
==========================================================================
PERCOBAAN 03: SCALING DAN ZOOM
==========================================================================
Scaling mengubah ukuran gambar. Zoom-in memperbesar area tertentu,
zoom-out memperkecil seluruh gambar.

Fungsi:
- cv2.resize(src, dsize, fx, fy, interpolation)
- cv2.pyrUp(src) → Perbesar 2× dengan Gaussian pyramid
- cv2.pyrDown(src) → Perkecil 2× dengan Gaussian pyramid
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

img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()

img = cv2.resize(img, (400, 400))
h, w = img.shape[:2]

print("=" * 60)
print("PERCOBAAN 03: SCALING DAN ZOOM")
print("=" * 60)

# ============================================================
# 1. Resize dengan ukuran absolut
# ============================================================
print("\n--- 1. Resize Absolut ---")
ukuran_list = [(100, 100), (200, 200), (400, 400), (600, 600)]
hasil_abs = {}
for uk in ukuran_list:
    hasil_abs[uk] = cv2.resize(img, uk)
    print(f"  {w}×{h} → {uk[0]}×{uk[1]}")

# ============================================================
# 2. Resize dengan faktor skala
# ============================================================
print("\n--- 2. Resize Faktor Skala ---")
faktor = [0.25, 0.5, 1.5, 2.0]
hasil_faktor = {}
for f in faktor:
    # fx dan fy: faktor skala horizontal dan vertikal
    hasil_faktor[f] = cv2.resize(img, None, fx=f, fy=f)
    sh = hasil_faktor[f].shape
    print(f"  Faktor {f:.2f}×: {sh[1]}×{sh[0]}")

# ============================================================
# 3. Perbandingan metode interpolasi
# ============================================================
print("\n--- 3. Perbandingan Interpolasi ---")

# Perkecil dulu lalu perbesar untuk melihat perbedaan kualitas
img_kecil = cv2.resize(img, (50, 50))

interp = {
    "NEAREST":  cv2.INTER_NEAREST,
    "LINEAR":   cv2.INTER_LINEAR,
    "CUBIC":    cv2.INTER_CUBIC,
    "LANCZOS4": cv2.INTER_LANCZOS4,
}

hasil_interp = {}
for nama, metode in interp.items():
    t0 = time.time()
    hasil_interp[nama] = cv2.resize(img_kecil, (400, 400), interpolation=metode)
    dt = (time.time() - t0) * 1000
    print(f"  {nama:10s}: {dt:.2f} ms")

# ============================================================
# 4. Zoom-in (crop + resize)
# ============================================================
print("\n--- 4. Zoom-In (Crop + Resize) ---")

def zoom_in(img, cx, cy, factor):
    """Zoom ke titik (cx, cy) dengan faktor zoom."""
    h, w = img.shape[:2]
    nw = int(w / factor)
    nh = int(h / factor)
    x1 = max(0, cx - nw // 2)
    y1 = max(0, cy - nh // 2)
    x2 = min(w, x1 + nw)
    y2 = min(h, y1 + nh)
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)

zoom_2x = zoom_in(img, w // 2, h // 2, 2.0)
zoom_4x = zoom_in(img, w // 2, h // 2, 4.0)
zoom_kiri = zoom_in(img, w // 4, h // 4, 3.0)
print("  Zoom 2×, 4×, dan kiri-atas 3×")

# ============================================================
# 5. Gaussian Pyramid (pyrDown / pyrUp)
# ============================================================
print("\n--- 5. Gaussian Pyramid ---")

# cv2.pyrDown: perkecil gambar 2× dengan Gaussian smoothing
pyr_down1 = cv2.pyrDown(img)
pyr_down2 = cv2.pyrDown(pyr_down1)
pyr_down3 = cv2.pyrDown(pyr_down2)

# cv2.pyrUp: perbesar gambar 2× (tidak mengembalikan detail)
pyr_up1 = cv2.pyrUp(pyr_down2)

print(f"  Level 0: {img.shape[1]}×{img.shape[0]}")
print(f"  Level 1: {pyr_down1.shape[1]}×{pyr_down1.shape[0]}")
print(f"  Level 2: {pyr_down2.shape[1]}×{pyr_down2.shape[0]}")
print(f"  Level 3: {pyr_down3.shape[1]}×{pyr_down3.shape[0]}")

# ============================================================
# 6. Non-uniform scaling (aspek ratio berubah)
# ============================================================
print("\n--- 6. Non-Uniform Scaling ---")
img_lebar = cv2.resize(img, (600, 200))
img_tinggi = cv2.resize(img, (200, 600))
print(f"  Lebar: {img_lebar.shape[1]}×{img_lebar.shape[0]}")
print(f"  Tinggi: {img_tinggi.shape[1]}×{img_tinggi.shape[0]}")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original 400×400")
axes[0, 1].imshow(cv2.cvtColor(zoom_2x, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Zoom 2×")
axes[0, 2].imshow(cv2.cvtColor(zoom_4x, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Zoom 4×")
axes[0, 3].imshow(cv2.cvtColor(zoom_kiri, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Zoom Kiri-Atas 3×")

for i, (nama, im) in enumerate(hasil_interp.items()):
    axes[1, i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"50→400 {nama}")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Percobaan 03: Scaling & Zoom", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "03_scaling_zoom_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
