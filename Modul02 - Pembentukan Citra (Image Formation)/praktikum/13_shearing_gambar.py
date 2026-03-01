"""
==========================================================================
PERCOBAAN 13: SHEARING GAMBAR
==========================================================================
Shearing memiringkan gambar searah sumbu X atau Y.
Matriks shear:
  Horizontal: [[1, shx, 0], [0, 1, 0]]   → miring ke kanan/kiri
  Vertikal:   [[1, 0, 0], [shy, 1, 0]]   → miring ke atas/bawah

Fungsi: cv2.warpAffine(src, M_shear, dsize)
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

img = cv2.imread(os.path.join(IMAGE_DIR, "kotak_warna.png"))
if img is None:
    img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()
img = cv2.resize(img, (300, 300))
h, w = img.shape[:2]

print("=" * 60)
print("PERCOBAAN 13: SHEARING GAMBAR")
print("=" * 60)

# ============================================================
# 1. Shear horizontal positif
# ============================================================
print("\n--- 1. Shear Horizontal ---")
shx_values = [-0.5, -0.3, 0, 0.3, 0.5]
hasil_h = {}
for shx in shx_values:
    M = np.float32([[1, shx, 0], [0, 1, 0]])
    nw = int(w + abs(shx) * h)
    offset = int(abs(shx) * h) if shx < 0 else 0
    M[0, 2] = offset
    hasil_h[shx] = cv2.warpAffine(img, M, (nw, h))
    print(f"  shx={shx:+.1f} → {nw}×{h}")

# ============================================================
# 2. Shear vertikal
# ============================================================
print("\n--- 2. Shear Vertikal ---")
shy_values = [-0.3, 0.3]
hasil_v = {}
for shy in shy_values:
    M = np.float32([[1, 0, 0], [shy, 1, 0]])
    nh = int(h + abs(shy) * w)
    offset = int(abs(shy) * w) if shy < 0 else 0
    M[1, 2] = offset
    hasil_v[shy] = cv2.warpAffine(img, M, (w, nh))
    print(f"  shy={shy:+.1f} → {w}×{nh}")

# ============================================================
# 3. Shear gabungan (X + Y)
# ============================================================
print("\n--- 3. Gabungan ---")
M_gabung = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
nw = int(w + 0.2 * h)
nh = int(h + 0.2 * w)
img_gabung = cv2.warpAffine(img, M_gabung, (nw, nh))
print(f"  shx=0.2 shy=0.2 → {nw}×{nh}")

# ============================================================
# 4. Aplikasi: efek italic teks
# ============================================================
print("\n--- 4. Efek Italic ---")
teks_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
cv2.putText(teks_img, "KOMPUTER VISION", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
M_italic = np.float32([[1, -0.3, 30], [0, 1, 0]])
teks_italic = cv2.warpAffine(teks_img, M_italic, (300, 100),
                               borderValue=(255, 255, 255))
print("  Teks normal → italic")

# ============================================================
# 5. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 1].imshow(cv2.cvtColor(hasil_h[0.3], cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Shear H +0.3")
axes[0, 2].imshow(cv2.cvtColor(hasil_h[-0.3], cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Shear H -0.3")
axes[0, 3].imshow(cv2.cvtColor(hasil_v[0.3], cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Shear V +0.3")

axes[1, 0].imshow(cv2.cvtColor(hasil_v[-0.3], cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Shear V -0.3")
axes[1, 1].imshow(cv2.cvtColor(img_gabung, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Gabungan 0.2/0.2")
axes[1, 2].imshow(cv2.cvtColor(teks_img, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Teks Normal")
axes[1, 3].imshow(cv2.cvtColor(teks_italic, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Teks Italic (shear)")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Percobaan 13: Shearing Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "13_shearing_gambar_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
