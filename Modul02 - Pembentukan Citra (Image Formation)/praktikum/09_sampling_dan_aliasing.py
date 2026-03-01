"""
==========================================================================
PERCOBAAN 09: SAMPLING DAN ALIASING
==========================================================================
Sampling mengurangi resolusi gambar. Aliasing terjadi ketika
frekuensi tinggi tidak di-filter sebelum downsampling.

Konsep: Nyquist theorem - frekuensi sampling harus ≥ 2× frekuensi sinyal.
Anti-aliasing: blur gambar sebelum downsampling.

Fungsi: cv2.resize, cv2.GaussianBlur, cv2.pyrDown
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
print("PERCOBAAN 09: SAMPLING DAN ALIASING")
print("=" * 60)

# ============================================================
# 1. Membuat gambar dengan frekuensi tinggi (garis-garis halus)
# ============================================================
print("\n--- 1. Gambar Frekuensi Tinggi ---")

# Pola garis vertikal (frekuensi tinggi)
img_lines = np.zeros((400, 400), dtype=np.uint8)
for x in range(0, 400, 2):
    img_lines[:, x] = 255
print("  Pola garis: periode 2 piksel")

# Pola checkerboard (frekuensi tinggi)
cb = np.zeros((400, 400), dtype=np.uint8)
for y in range(400):
    for x in range(400):
        if (x + y) % 2 == 0:
            cb[y, x] = 255

# Pola sinusoidal dengan frekuensi meningkat
sinus = np.zeros((400, 400), dtype=np.uint8)
for x in range(400):
    freq = 1 + x * 0.1  # Frekuensi meningkat ke kanan
    for y in range(400):
        val = 128 + 127 * np.sin(2 * np.pi * freq * y / 400)
        sinus[y, x] = int(val)

# ============================================================
# 2. Downsampling TANPA anti-aliasing (aliasing)
# ============================================================
print("\n--- 2. Downsampling Tanpa AA ---")

# Resize langsung tanpa blur → muncul aliasing (moiré pattern)
garis_small_no_aa = cv2.resize(img_lines, (100, 100), interpolation=cv2.INTER_NEAREST)
cb_small_no_aa = cv2.resize(cb, (100, 100), interpolation=cv2.INTER_NEAREST)
sinus_small_no_aa = cv2.resize(sinus, (100, 100), interpolation=cv2.INTER_NEAREST)
print("  400→100 dengan INTER_NEAREST (aliased)")

# ============================================================
# 3. Downsampling DENGAN anti-aliasing
# ============================================================
print("\n--- 3. Downsampling Dengan AA ---")

# Blur dulu dengan GaussianBlur → lalu resize
garis_blur = cv2.GaussianBlur(img_lines, (7, 7), 2)
garis_small_aa = cv2.resize(garis_blur, (100, 100), interpolation=cv2.INTER_LINEAR)

cb_blur = cv2.GaussianBlur(cb, (7, 7), 2)
cb_small_aa = cv2.resize(cb_blur, (100, 100), interpolation=cv2.INTER_LINEAR)

sinus_blur = cv2.GaussianBlur(sinus, (7, 7), 2)
sinus_small_aa = cv2.resize(sinus_blur, (100, 100), interpolation=cv2.INTER_LINEAR)
print("  GaussianBlur(7,7) → resize 100×100 (anti-aliased)")

# ============================================================
# 4. Gaussian Pyramid vs Direct Resize
# ============================================================
print("\n--- 4. Pyramid vs Direct ---")

img_real = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
if img_real is None:
    img_real = cv2.cvtColor(sinus, cv2.COLOR_GRAY2BGR)
img_real = cv2.resize(img_real, (400, 400))

# Direct resize ke 50×50
direct_50 = cv2.resize(img_real, (50, 50), interpolation=cv2.INTER_NEAREST)
# Pyramid: 400→200→100→50 (tiap level blur + downsample)
pyr1 = cv2.pyrDown(img_real)   # 200
pyr2 = cv2.pyrDown(pyr1)       # 100
pyr3 = cv2.pyrDown(pyr2)       # 50

# Perbesar kembali untuk perbandingan
direct_big = cv2.resize(direct_50, (400, 400), interpolation=cv2.INTER_NEAREST)
pyr_big = cv2.resize(pyr3, (400, 400), interpolation=cv2.INTER_NEAREST)
print("  Direct 400→50: aliased")
print("  Pyramid 400→200→100→50: anti-aliased")

# ============================================================
# 5. INTER_AREA (terbaik untuk downsampling)
# ============================================================
print("\n--- 5. INTER_AREA ---")

# cv2.INTER_AREA menggunakan area resampling → anti-aliasing bawaan
garis_area = cv2.resize(img_lines, (100, 100), interpolation=cv2.INTER_AREA)
cb_area = cv2.resize(cb, (100, 100), interpolation=cv2.INTER_AREA)
print("  INTER_AREA: anti-aliasing otomatis saat downsampling")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Original
axes[0, 0].imshow(img_lines, cmap='gray')
axes[0, 0].set_title("Garis (orig)")
axes[0, 1].imshow(cb[:100, :100], cmap='gray')
axes[0, 1].set_title("Checker (crop)")
axes[0, 2].imshow(sinus, cmap='gray')
axes[0, 2].set_title("Sinus (freq naik)")
axes[0, 3].imshow(cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Grid Real")

# Baris 2: Tanpa AA
axes[1, 0].imshow(garis_small_no_aa, cmap='gray')
axes[1, 0].set_title("Garis NEAREST ↓")
axes[1, 1].imshow(cb_small_no_aa, cmap='gray')
axes[1, 1].set_title("Checker NEAREST ↓")
axes[1, 2].imshow(sinus_small_no_aa, cmap='gray')
axes[1, 2].set_title("Sinus NEAREST ↓")
axes[1, 3].imshow(cv2.cvtColor(direct_big, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Direct ↓ (aliased)")

# Baris 3: Dengan AA
axes[2, 0].imshow(garis_small_aa, cmap='gray')
axes[2, 0].set_title("Garis+Blur AA ↓")
axes[2, 1].imshow(garis_area, cmap='gray')
axes[2, 1].set_title("Garis INTER_AREA")
axes[2, 2].imshow(cb_small_aa, cmap='gray')
axes[2, 2].set_title("Checker+Blur AA")
axes[2, 3].imshow(cv2.cvtColor(pyr_big, cv2.COLOR_BGR2RGB))
axes[2, 3].set_title("Pyramid ↓ (AA)")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Percobaan 09: Sampling & Aliasing", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "09_sampling_aliasing_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
