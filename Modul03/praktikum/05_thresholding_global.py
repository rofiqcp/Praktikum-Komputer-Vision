"""
==========================================================================
PERCOBAAN 05: THRESHOLDING GLOBAL
==========================================================================
Thresholding global mengkonversi gambar grayscale menjadi biner
menggunakan satu nilai threshold yang sama untuk seluruh gambar.

Fungsi:
- cv2.threshold(src, thresh, maxval, type) → thresholding global
  Tipe: THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC,
        THRESH_TOZERO, THRESH_TOZERO_INV
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

img = cv2.imread(os.path.join(IMAGE_DIR, "dokumen.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!"); exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 05: THRESHOLDING GLOBAL")
print("=" * 60)

# ============================================================
# 1. Lima Tipe Thresholding
# ============================================================
print("\n--- 1. Lima Tipe Thresholding ---")

thresh_val = 127
types = [
    (cv2.THRESH_BINARY, "BINARY"),
    (cv2.THRESH_BINARY_INV, "BINARY_INV"),
    (cv2.THRESH_TRUNC, "TRUNC"),
    (cv2.THRESH_TOZERO, "TOZERO"),
    (cv2.THRESH_TOZERO_INV, "TOZERO_INV"),
]

type_results = []
for ttype, name in types:
    # threshold() mengembalikan (nilai_thresh, gambar_hasil)
    ret, result = cv2.threshold(gray, thresh_val, 255, ttype)
    type_results.append((name, result))
    unique = len(np.unique(result))
    print(f"  {name:15s}: ret={ret:.0f}, unique_vals={unique}")

# ============================================================
# 2. Variasi Nilai Threshold
# ============================================================
print("\n--- 2. Variasi Nilai Threshold ---")

thresh_list = [50, 80, 100, 127, 160, 200]
thresh_results = []

for t in thresh_list:
    ret, result = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    # Hitung persentase piksel putih (foreground)
    white_pct = np.sum(result == 255) / result.size * 100
    thresh_results.append(result)
    print(f"  T={t:3d}: {white_pct:.1f}% piksel putih")

# ============================================================
# 3. Thresholding Manual (NumPy)
# ============================================================
print("\n--- 3. Thresholding Manual ---")

# Implementasi manual
T = 127
manual_binary = np.where(gray > T, 255, 0).astype(np.uint8)

# Verifikasi dengan OpenCV
_, opencv_binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
diff = cv2.absdiff(manual_binary, opencv_binary).max()
print(f"  Perbedaan manual vs OpenCV: {diff}")

# ============================================================
# 4. Thresholding pada Gambar Berwarna (per channel)
# ============================================================
print("\n--- 4. Thresholding per Channel ---")

b, g, r = cv2.split(img)
_, b_thresh = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
_, g_thresh = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
_, r_thresh = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)
# Gabungkan kembali
color_thresh = cv2.merge([b_thresh, g_thresh, r_thresh])
print("  Threshold T=127 di setiap channel BGR")

# ============================================================
# 5. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Lima tipe
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

for i, (name, res) in enumerate(type_results[:3]):
    axes[0, i + 1].imshow(res, cmap='gray')
    axes[0, i + 1].set_title(name)
    axes[0, i + 1].axis("off")

# Baris 2: Sisa tipe + variasi threshold
for i in range(2):
    name, res = type_results[3 + i]
    axes[1, i].imshow(res, cmap='gray')
    axes[1, i].set_title(name)
    axes[1, i].axis("off")

axes[1, 2].imshow(thresh_results[0], cmap='gray')
axes[1, 2].set_title("T=50")
axes[1, 2].axis("off")

axes[1, 3].imshow(thresh_results[3], cmap='gray')
axes[1, 3].set_title("T=127")
axes[1, 3].axis("off")

# Baris 3: Variasi threshold + warna
axes[2, 0].imshow(thresh_results[1], cmap='gray')
axes[2, 0].set_title("T=80")
axes[2, 0].axis("off")

axes[2, 1].imshow(thresh_results[5], cmap='gray')
axes[2, 1].set_title("T=200")
axes[2, 1].axis("off")

axes[2, 2].imshow(cv2.cvtColor(color_thresh, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title("Per-Channel")
axes[2, 2].axis("off")

# Histogram + threshold line
axes[2, 3].hist(gray.ravel(), 256, [0, 256], color='steelblue', alpha=0.7)
axes[2, 3].axvline(x=127, color='red', linestyle='--', label='T=127')
axes[2, 3].set_title("Histogram + T")
axes[2, 3].legend()

plt.suptitle("Percobaan 05: Thresholding Global", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "05_thresholding_global_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 05")
print("=" * 60)
print("""
1. BINARY: piksel > T → 255, lainnya → 0
2. BINARY_INV: piksel > T → 0, lainnya → 255
3. TRUNC: piksel > T → T, lainnya tetap
4. TOZERO: piksel > T → tetap, lainnya → 0
5. TOZERO_INV: piksel > T → 0, lainnya tetap
6. Pemilihan T yang tepat sangat penting untuk hasil
""")
