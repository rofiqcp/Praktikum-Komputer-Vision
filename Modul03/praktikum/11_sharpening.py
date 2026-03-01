"""
==========================================================================
PERCOBAAN 11: SHARPENING (PENAJAMAN GAMBAR)
==========================================================================
Sharpening mempertajam detail gambar dengan menekankan tepi (edge).
Teknik utama:
- Unsharp Mask: original + k*(original - blurred)
- Laplacian sharpening: original - Laplacian
- Custom kernel sharpening

Fungsi:
- cv2.GaussianBlur() → untuk unsharp mask
- cv2.Laplacian() → turunan kedua
- cv2.filter2D() → kernel custom
- cv2.addWeighted() → blending
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

img = cv2.imread(os.path.join(IMAGE_DIR, "teks_buram.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!"); exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 11: SHARPENING")
print("=" * 60)

# ============================================================
# 1. Unsharp Mask
# ============================================================
print("\n--- 1. Unsharp Mask ---")

# Langkah: blur → hitung detail → tambah detail ke original
blurred = cv2.GaussianBlur(gray, (9, 9), 3)
# Detail = original - blurred
detail = cv2.subtract(gray, blurred)
# Sharpened = original + k * detail
k_values = [0.5, 1.0, 1.5, 2.0, 3.0]
unsharp_results = []
for k in k_values:
    sharpened = cv2.addWeighted(gray, 1.0 + k, blurred, -k, 0)
    unsharp_results.append(sharpened)
    print(f"  k={k:.1f}: std={sharpened.std():.1f}")

# ============================================================
# 2. Laplacian Sharpening
# ============================================================
print("\n--- 2. Laplacian Sharpening ---")

# Hitung Laplacian (turunan kedua)
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
# Sharpened = original - Laplacian
lap_sharp = gray.astype(np.float64) - laplacian
lap_sharp = np.clip(lap_sharp, 0, 255).astype(np.uint8)
print(f"  Laplacian range: [{laplacian.min():.0f}, {laplacian.max():.0f}]")

# ============================================================
# 3. Kernel Sharpening Custom
# ============================================================
print("\n--- 3. Kernel Sharpening Custom ---")

# Kernel 1: mild sharpening
kernel1 = np.array([[ 0, -1,  0],
                     [-1,  5, -1],
                     [ 0, -1,  0]], dtype=np.float32)
sharp1 = cv2.filter2D(gray, -1, kernel1)

# Kernel 2: strong sharpening (8-connected)
kernel2 = np.array([[-1, -1, -1],
                     [-1,  9, -1],
                     [-1, -1, -1]], dtype=np.float32)
sharp2 = cv2.filter2D(gray, -1, kernel2)

# Kernel 3: very strong
kernel3 = np.array([[-1, -2, -1],
                     [-2, 13, -2],
                     [-1, -2, -1]], dtype=np.float32)
sharp3 = cv2.filter2D(gray, -1, kernel3)

print("  Kernel 1 (mild): center=5")
print("  Kernel 2 (strong): center=9")
print("  Kernel 3 (very strong): center=13")

# ============================================================
# 4. Sharpening pada Gambar Berwarna
# ============================================================
print("\n--- 4. Sharpening Berwarna ---")

# Unsharp mask pada gambar berwarna
blurred_color = cv2.GaussianBlur(img, (9, 9), 3)
sharp_color = cv2.addWeighted(img, 1.5, blurred_color, -0.5, 0)

# Sharpening via LAB (hanya channel L)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
lab[:, :, 0] = cv2.addWeighted(
    lab[:, :, 0], 1.5,
    cv2.GaussianBlur(lab[:, :, 0], (9, 9), 3), -0.5, 0
)
sharp_lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
print("  Direct BGR vs LAB-only sharpening")

# ============================================================
# 5. High-Boost Filtering
# ============================================================
print("\n--- 5. High-Boost Filtering ---")

# High-boost = A * original - lowpass
# Jika A > 1, amplifikasi high-frequency
A_values = [1.5, 2.0, 3.0]
lowpass = cv2.GaussianBlur(gray, (7, 7), 2)
hb_results = []
for A in A_values:
    hb = cv2.addWeighted(gray, A, lowpass, -(A - 1), 0)
    hb_results.append(hb)
    print(f"  A={A}: high-boost filter")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Unsharp mask
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original (Buram)")
axes[0, 0].axis("off")

for i, (k, res) in enumerate(zip([1.0, 2.0, 3.0],
                                  [unsharp_results[1], unsharp_results[3],
                                   unsharp_results[4]])):
    axes[0, i + 1].imshow(res, cmap='gray')
    axes[0, i + 1].set_title(f"Unsharp k={k}")
    axes[0, i + 1].axis("off")

# Baris 2: Kernel + Laplacian
axes[1, 0].imshow(lap_sharp, cmap='gray')
axes[1, 0].set_title("Laplacian Sharp")
axes[1, 0].axis("off")

axes[1, 1].imshow(sharp1, cmap='gray')
axes[1, 1].set_title("Kernel Mild")
axes[1, 1].axis("off")

axes[1, 2].imshow(sharp2, cmap='gray')
axes[1, 2].set_title("Kernel Strong")
axes[1, 2].axis("off")

axes[1, 3].imshow(sharp3, cmap='gray')
axes[1, 3].set_title("Kernel V.Strong")
axes[1, 3].axis("off")

# Baris 3: Warna
axes[2, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("Original Warna")
axes[2, 0].axis("off")

axes[2, 1].imshow(cv2.cvtColor(sharp_color, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("Sharp BGR")
axes[2, 1].axis("off")

axes[2, 2].imshow(cv2.cvtColor(sharp_lab, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title("Sharp LAB")
axes[2, 2].axis("off")

axes[2, 3].imshow(hb_results[1], cmap='gray')
axes[2, 3].set_title("High-Boost A=2")
axes[2, 3].axis("off")

plt.suptitle("Percobaan 11: Sharpening (Penajaman)", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "11_sharpening_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 11")
print("=" * 60)
print("""
1. Unsharp mask: sharp = original + k*(original - blurred)
2. Laplacian sharpening: sharp = original - Laplacian
3. Kernel sharpening: center besar, tetangga negatif
4. k/A lebih besar → sharpening lebih kuat (tapi bisa artifact)
5. Untuk gambar berwarna, sharpen di channel L (LAB) lebih aman
6. High-boost: generalisasi unsharp mask (A > 1)
""")
