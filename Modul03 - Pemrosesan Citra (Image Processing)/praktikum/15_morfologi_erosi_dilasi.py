"""
==========================================================================
PERCOBAAN 15: MORFOLOGI DASAR (EROSI DAN DILASI)
==========================================================================
Operasi morfologi bekerja pada gambar biner menggunakan structuring
element (kernel bentuk). Dua operasi dasar:
- Erosi: mengecilkan foreground, menghilangkan noise
- Dilasi: memperbesar foreground, mengisi lubang kecil

Fungsi:
- cv2.erode(src, kernel, iterations) → erosi
- cv2.dilate(src, kernel, iterations) → dilasi
- cv2.getStructuringElement(shape, ksize) → buat kernel
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
print("PERCOBAAN 15: MORFOLOGI DASAR (EROSI & DILASI)")
print("=" * 60)

# ============================================================
# 1. Structuring Elements
# ============================================================
print("\n--- 1. Structuring Elements ---")

# Tiga bentuk structuring element
se_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
se_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
se_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print("  RECT 5×5:")
print(se_rect)
print("  CROSS 5×5:")
print(se_cross)
print("  ELLIPSE 5×5:")
print(se_ellipse)

# ============================================================
# 2. Erosi
# ============================================================
print("\n--- 2. Erosi ---")

erode_1 = cv2.erode(biner, se_rect, iterations=1)
erode_2 = cv2.erode(biner, se_rect, iterations=2)
erode_3 = cv2.erode(biner, se_rect, iterations=3)

# Hitung area foreground (putih)
orig_area = np.sum(biner == 255)
print(f"  Original area: {orig_area} piksel")
print(f"  Erode 1×: {np.sum(erode_1 == 255)} piksel")
print(f"  Erode 2×: {np.sum(erode_2 == 255)} piksel")
print(f"  Erode 3×: {np.sum(erode_3 == 255)} piksel")

# ============================================================
# 3. Dilasi
# ============================================================
print("\n--- 3. Dilasi ---")

dilate_1 = cv2.dilate(biner, se_rect, iterations=1)
dilate_2 = cv2.dilate(biner, se_rect, iterations=2)
dilate_3 = cv2.dilate(biner, se_rect, iterations=3)

print(f"  Dilate 1×: {np.sum(dilate_1 == 255)} piksel")
print(f"  Dilate 2×: {np.sum(dilate_2 == 255)} piksel")
print(f"  Dilate 3×: {np.sum(dilate_3 == 255)} piksel")

# ============================================================
# 4. Pengaruh Bentuk Kernel
# ============================================================
print("\n--- 4. Pengaruh Bentuk Kernel ---")

erode_rect = cv2.erode(biner, se_rect, iterations=2)
erode_cross = cv2.erode(biner, se_cross, iterations=2)
erode_ellipse = cv2.erode(biner, se_ellipse, iterations=2)

print(f"  Rect:    {np.sum(erode_rect == 255)} piksel")
print(f"  Cross:   {np.sum(erode_cross == 255)} piksel")
print(f"  Ellipse: {np.sum(erode_ellipse == 255)} piksel")

# ============================================================
# 5. Variasi Ukuran Kernel
# ============================================================
print("\n--- 5. Variasi Ukuran Kernel ---")

for size in [3, 5, 7, 11]:
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    e = cv2.erode(biner, se, iterations=1)
    d = cv2.dilate(biner, se, iterations=1)
    print(f"  SE {size}×{size}: erode={np.sum(e==255)}, dilate={np.sum(d==255)}")

# ============================================================
# 6. Erosi untuk Menghilangkan Noise
# ============================================================
print("\n--- 6. Noise Removal ---")

# Erosi menghilangkan noise kecil (salt) pada foreground
cleaned_erode = cv2.erode(biner, se_rect, iterations=1)
# Tapi objek juga mengecil → dilasi untuk mengembalikan ukuran
cleaned = cv2.dilate(cleaned_erode, se_rect, iterations=1)
print("  Erode → Dilate: membersihkan noise kecil (= Opening)")

# ============================================================
# 7. Dilasi untuk Menutup Lubang
# ============================================================
print("\n--- 7. Gap Filling ---")

# Dilasi menutup lubang kecil pada foreground
filled_dilate = cv2.dilate(biner, se_rect, iterations=1)
# Tapi objek membesar → erosi untuk mengembalikan ukuran
filled = cv2.erode(filled_dilate, se_rect, iterations=1)
print("  Dilate → Erode: menutup lubang kecil (= Closing)")

# ============================================================
# 8. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Erosi
axes[0, 0].imshow(biner, cmap='gray')
axes[0, 0].set_title("Original (Noise)")
axes[0, 0].axis("off")

axes[0, 1].imshow(erode_1, cmap='gray')
axes[0, 1].set_title("Erode 1×")
axes[0, 1].axis("off")

axes[0, 2].imshow(erode_2, cmap='gray')
axes[0, 2].set_title("Erode 2×")
axes[0, 2].axis("off")

axes[0, 3].imshow(erode_3, cmap='gray')
axes[0, 3].set_title("Erode 3×")
axes[0, 3].axis("off")

# Baris 2: Dilasi
axes[1, 0].imshow(biner, cmap='gray')
axes[1, 0].set_title("Original")
axes[1, 0].axis("off")

axes[1, 1].imshow(dilate_1, cmap='gray')
axes[1, 1].set_title("Dilate 1×")
axes[1, 1].axis("off")

axes[1, 2].imshow(dilate_2, cmap='gray')
axes[1, 2].set_title("Dilate 2×")
axes[1, 2].axis("off")

axes[1, 3].imshow(dilate_3, cmap='gray')
axes[1, 3].set_title("Dilate 3×")
axes[1, 3].axis("off")

# Baris 3: Bentuk kernel + cleaning
axes[2, 0].imshow(erode_rect, cmap='gray')
axes[2, 0].set_title("Erode: Rect")
axes[2, 0].axis("off")

axes[2, 1].imshow(erode_cross, cmap='gray')
axes[2, 1].set_title("Erode: Cross")
axes[2, 1].axis("off")

axes[2, 2].imshow(cleaned, cmap='gray')
axes[2, 2].set_title("Cleaning (E→D)")
axes[2, 2].axis("off")

axes[2, 3].imshow(filled, cmap='gray')
axes[2, 3].set_title("Filling (D→E)")
axes[2, 3].axis("off")

plt.suptitle("Percobaan 15: Morfologi Dasar (Erosi & Dilasi)", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "15_erosi_dilasi_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 15")
print("=" * 60)
print("""
1. Erosi (E = A ⊖ B): mengecilkan foreground
2. Dilasi (D = A ⊕ B): memperbesar foreground
3. Structuring element: RECT, CROSS, ELLIPSE
4. iterations > 1 → efek lebih kuat
5. Erode → Dilate = Opening (hapus noise kecil)
6. Dilate → Erode = Closing (tutup lubang kecil)
""")
