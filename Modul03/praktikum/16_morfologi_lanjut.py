"""
==========================================================================
PERCOBAAN 16: MORFOLOGI LANJUT (OPENING, CLOSING, GRADIENT)
==========================================================================
Operasi morfologi lanjut mengkombinasikan erosi dan dilasi:
- Opening = Erosi → Dilasi (hapus noise kecil, pertahankan ukuran)
- Closing = Dilasi → Erosi (tutup lubang kecil, pertahankan ukuran)
- Gradient = Dilasi - Erosi (outline / tepi objek)

Fungsi:
- cv2.morphologyEx(src, op, kernel, iterations)
  op: MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT
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

biner = cv2.imread(os.path.join(IMAGE_DIR, "biner_noise.png"), cv2.IMREAD_GRAYSCALE)
if biner is None:
    print("[ERROR] Jalankan download_image.py!"); exit()

print("=" * 60)
print("PERCOBAAN 16: MORFOLOGI LANJUT")
print("=" * 60)

se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ============================================================
# 1. Opening (Erosi → Dilasi)
# ============================================================
print("\n--- 1. Opening ---")

opening = cv2.morphologyEx(biner, cv2.MORPH_OPEN, se)
# Bandingkan dengan manual
manual_open = cv2.dilate(cv2.erode(biner, se), se)
diff = cv2.absdiff(opening, manual_open).max()
print(f"  Area sebelum: {np.sum(biner == 255)}")
print(f"  Area setelah opening: {np.sum(opening == 255)}")
print(f"  Manual vs morphologyEx: diff={diff}")

# ============================================================
# 2. Closing (Dilasi → Erosi)
# ============================================================
print("\n--- 2. Closing ---")

closing = cv2.morphologyEx(biner, cv2.MORPH_CLOSE, se)
print(f"  Area setelah closing: {np.sum(closing == 255)}")

# ============================================================
# 3. Morphological Gradient (Dilasi - Erosi = Outline)
# ============================================================
print("\n--- 3. Morphological Gradient ---")

gradient = cv2.morphologyEx(biner, cv2.MORPH_GRADIENT, se)
print(f"  Gradient piksel: {np.sum(gradient > 0)}")

# ============================================================
# 4. Variasi Iterasi
# ============================================================
print("\n--- 4. Variasi Iterasi ---")

open_results = []
close_results = []
for itr in [1, 2, 3, 5]:
    op = cv2.morphologyEx(biner, cv2.MORPH_OPEN, se, iterations=itr)
    cl = cv2.morphologyEx(biner, cv2.MORPH_CLOSE, se, iterations=itr)
    open_results.append(op)
    close_results.append(cl)
    print(f"  iter={itr}: open={np.sum(op==255)}, close={np.sum(cl==255)}")

# ============================================================
# 5. Opening + Closing (noise removal lengkap)
# ============================================================
print("\n--- 5. Opening → Closing ---")

# Opening menghapus noise kecil di background
# Closing menutup lubang kecil di foreground
clean = cv2.morphologyEx(biner, cv2.MORPH_OPEN, se, iterations=1)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, se, iterations=1)
print(f"  Area bersih: {np.sum(clean == 255)}")

# ============================================================
# 6. Internal dan External Gradient
# ============================================================
print("\n--- 6. Internal & External Gradient ---")

# Internal gradient: Original - Erosi (tepi sisi dalam)
eroded = cv2.erode(clean, se)
internal = cv2.subtract(clean, eroded)

# External gradient: Dilasi - Original (tepi sisi luar)
dilated = cv2.dilate(clean, se)
external = cv2.subtract(dilated, clean)

# Full gradient = dilasi - erosi
full_grad = cv2.subtract(dilated, eroded)
print(f"  Internal edge: {np.sum(internal > 0)} piksel")
print(f"  External edge: {np.sum(external > 0)} piksel")

# ============================================================
# 7. Aplikasi pada Gambar Grayscale
# ============================================================
print("\n--- 7. Morfologi Grayscale ---")

gray_img = cv2.imread(os.path.join(IMAGE_DIR, "kota.jpg"), cv2.IMREAD_GRAYSCALE)
if gray_img is not None:
    gray_img = cv2.resize(gray_img, (512, 512))
    se_g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_g = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, se_g)
    close_g = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, se_g)
    grad_g = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, se_g)
    print("  Opening, Closing, Gradient pada grayscale")

# ============================================================
# 8. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Operasi dasar
axes[0, 0].imshow(biner, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(opening, cmap='gray')
axes[0, 1].set_title("Opening")
axes[0, 1].axis("off")

axes[0, 2].imshow(closing, cmap='gray')
axes[0, 2].set_title("Closing")
axes[0, 2].axis("off")

axes[0, 3].imshow(gradient, cmap='gray')
axes[0, 3].set_title("Gradient")
axes[0, 3].axis("off")

# Baris 2: Clean + gradients
axes[1, 0].imshow(clean, cmap='gray')
axes[1, 0].set_title("Open→Close")
axes[1, 0].axis("off")

axes[1, 1].imshow(internal, cmap='gray')
axes[1, 1].set_title("Internal Edge")
axes[1, 1].axis("off")

axes[1, 2].imshow(external, cmap='gray')
axes[1, 2].set_title("External Edge")
axes[1, 2].axis("off")

axes[1, 3].imshow(full_grad, cmap='gray')
axes[1, 3].set_title("Full Gradient")
axes[1, 3].axis("off")

# Baris 3: Grayscale morphology
if gray_img is not None:
    axes[2, 0].imshow(gray_img, cmap='gray')
    axes[2, 0].set_title("Gray Original")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(open_g, cmap='gray')
    axes[2, 1].set_title("Gray Opening")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(close_g, cmap='gray')
    axes[2, 2].set_title("Gray Closing")
    axes[2, 2].axis("off")

    axes[2, 3].imshow(grad_g, cmap='gray')
    axes[2, 3].set_title("Gray Gradient")
    axes[2, 3].axis("off")

plt.suptitle("Percobaan 16: Morfologi Lanjut", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "16_morfologi_lanjut_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 16")
print("=" * 60)
print("""
1. Opening = Erosi → Dilasi → hapus noise kecil
2. Closing = Dilasi → Erosi → tutup lubang kecil
3. Gradient = Dilasi - Erosi → outline objek
4. Internal gradient: Original - Erosi
5. External gradient: Dilasi - Original
6. Open → Close: pipeline pembersihan noise lengkap
7. Bisa diterapkan pada grayscale (bukan hanya biner)
""")
