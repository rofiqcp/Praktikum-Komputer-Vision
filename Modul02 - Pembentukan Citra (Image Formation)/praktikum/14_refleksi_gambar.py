"""
==========================================================================
PERCOBAAN 14: REFLEKSI GAMBAR
==========================================================================
Refleksi mencerminkan gambar terhadap sumbu atau garis tertentu
menggunakan matriks transformasi affine.

- Refleksi horizontal: M = [[-1,0,w],[0,1,0]]
- Refleksi vertikal:   M = [[1,0,0],[0,-1,h]]
- Refleksi diagonal:   Kombinasi rotasi + flip
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

img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()
img = cv2.resize(img, (300, 300))
h, w = img.shape[:2]

print("=" * 60)
print("PERCOBAAN 14: REFLEKSI GAMBAR")
print("=" * 60)

# ============================================================
# 1. Refleksi horizontal (sumbu Y)
# ============================================================
print("\n--- 1. Refleksi Horizontal ---")
# Matriks: x' = -x + (w-1), y' = y
M_h = np.float32([[-1, 0, w-1], [0, 1, 0]])
img_ref_h = cv2.warpAffine(img, M_h, (w, h))
# Sama dengan cv2.flip(img, 1)
print("  M = [[-1,0,w-1],[0,1,0]]")

# ============================================================
# 2. Refleksi vertikal (sumbu X)
# ============================================================
print("\n--- 2. Refleksi Vertikal ---")
M_v = np.float32([[1, 0, 0], [0, -1, h-1]])
img_ref_v = cv2.warpAffine(img, M_v, (w, h))
print("  M = [[1,0,0],[0,-1,h-1]]")

# ============================================================
# 3. Refleksi terhadap garis diagonal (y=x)
# ============================================================
print("\n--- 3. Refleksi Diagonal ---")
# Transpose gambar: (x,y) → (y,x)
img_diag = cv2.transpose(img)
img_diag = cv2.resize(img_diag, (w, h))
print("  cv2.transpose → refleksi y=x")

# ============================================================
# 4. Refleksi terhadap garis miring (sudut θ)
# ============================================================
print("\n--- 4. Refleksi Garis Miring ---")

def refleksi_garis(img, sudut_derajat):
    """Refleksi terhadap garis melalui pusat dengan sudut θ."""
    h, w = img.shape[:2]
    theta = np.deg2rad(sudut_derajat)
    cos2 = np.cos(2 * theta)
    sin2 = np.sin(2 * theta)
    # Matriks refleksi terhadap garis y = x*tan(θ) melalui pusat
    cx, cy = w / 2, h / 2
    M = np.float32([
        [cos2, sin2, cx * (1 - cos2) - cy * sin2],
        [sin2, -cos2, cy * (1 + cos2) - cx * sin2]
    ])
    return cv2.warpAffine(img, M, (w, h))

img_ref_30 = refleksi_garis(img, 30)
img_ref_45 = refleksi_garis(img, 45)
img_ref_60 = refleksi_garis(img, 60)
print("  Refleksi garis 30°, 45°, 60°")

# ============================================================
# 5. Efek cermin (split + flip)
# ============================================================
print("\n--- 5. Efek Cermin ---")
# Ambil separuh kiri, cerminkan ke kanan
kiri = img[:, :w//2]
kiri_flip = cv2.flip(kiri, 1)
img_cermin_h = np.hstack([kiri, kiri_flip])

# Ambil separuh atas, cerminkan ke bawah
atas = img[:h//2, :]
atas_flip = cv2.flip(atas, 0)
img_cermin_v = np.vstack([atas, atas_flip])
print("  Efek cermin horizontal dan vertikal")

# ============================================================
# 6. Kaleidoskop (4 refleksi)
# ============================================================
print("\n--- 6. Efek Kaleidoskop ---")
kuadran = img[:h//2, :w//2]
kanan = cv2.flip(kuadran, 1)
atas_row = np.hstack([kuadran, kanan])
bawah_row = cv2.flip(atas_row, 0)
kaleidoskop = np.vstack([atas_row, bawah_row])
print("  4-way kaleidoskop")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 1].imshow(cv2.cvtColor(img_ref_h, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Refleksi Horizontal")
axes[0, 2].imshow(cv2.cvtColor(img_ref_v, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Refleksi Vertikal")
axes[0, 3].imshow(cv2.cvtColor(img_diag, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Refleksi Diagonal")

axes[1, 0].imshow(cv2.cvtColor(img_ref_45, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Refleksi Garis 45°")
axes[1, 1].imshow(cv2.cvtColor(img_cermin_h, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Efek Cermin H")
axes[1, 2].imshow(cv2.cvtColor(img_cermin_v, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Efek Cermin V")
axes[1, 3].imshow(cv2.cvtColor(kaleidoskop, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Kaleidoskop")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Percobaan 14: Refleksi Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "14_refleksi_gambar_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
