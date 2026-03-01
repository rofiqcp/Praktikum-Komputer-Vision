"""
==========================================================================
PERCOBAAN 15: MATRIKS TRANSFORMASI HOMOGEN
==========================================================================
Matriks transformasi homogen (3×3) menyatukan translasi, rotasi,
skala, dan shear dalam satu matriks. Komposisi transformasi dilakukan
dengan perkalian matriks.

Konsep: [x'] = [a b tx] [x]
        [y']   [c d ty] [y]
        [1 ]   [0 0 1 ] [1]
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
print("PERCOBAAN 15: MATRIKS TRANSFORMASI HOMOGEN")
print("=" * 60)

# ============================================================
# 1. Matriks dasar
# ============================================================
print("\n--- 1. Matriks Dasar ---")

# Identitas
M_id = np.float64([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Translasi
tx, ty = 50, 30
M_trans = np.float64([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

# Rotasi (pusat origin)
theta = np.deg2rad(30)
M_rot = np.float64([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0, 0, 1]
])

# Skala
sx, sy = 0.8, 1.2
M_scale = np.float64([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

# Shear
M_shear = np.float64([[1, 0.3, 0], [0, 1, 0], [0, 0, 1]])

print("  Translasi, Rotasi, Skala, Shear")

# ============================================================
# 2. Komposisi transformasi (perkalian matriks)
# ============================================================
print("\n--- 2. Komposisi Transformasi ---")

# Rotasi di pusat gambar = Translate → Rotate → Translate-back
cx, cy = w / 2, h / 2
M_to_origin = np.float64([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
M_from_origin = np.float64([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

# Komposisi: M = T_back @ R @ T_origin
# np.dot atau @ untuk perkalian matriks
M_rot_center = M_from_origin @ M_rot @ M_to_origin
print(f"  Rotasi 30° di pusat:\n{M_rot_center[:2]}")

# Terapkan (gunakan hanya baris 0-1 untuk warpAffine)
img_rot = cv2.warpAffine(img, M_rot_center[:2], (w, h))

# ============================================================
# 3. Transformasi gabungan: Rotate + Scale + Translate
# ============================================================
print("\n--- 3. Gabungan R+S+T ---")

# Urutan: Scale → Rotate → Translate
M_gabung = M_trans @ M_from_origin @ M_rot @ M_scale @ M_to_origin
img_gabung = cv2.warpAffine(img, M_gabung[:2], (w, h))
print("  Scale(0.8,1.2) → Rot(30°) → Trans(50,30)")

# ============================================================
# 4. Urutan transformasi berbeda → hasil berbeda
# ============================================================
print("\n--- 4. Urutan Berbeda ---")

# Rotate lalu Translate
M_rt = M_trans @ M_from_origin @ M_rot @ M_to_origin
img_rt = cv2.warpAffine(img, M_rt[:2], (w, h))

# Translate lalu Rotate
M_tr = M_from_origin @ M_rot @ M_to_origin @ M_trans
img_tr = cv2.warpAffine(img, M_tr[:2], (w, h))

diff = np.mean(cv2.absdiff(img_rt, img_tr))
print(f"  Rot→Trans vs Trans→Rot: berbeda = {diff:.1f}")

# ============================================================
# 5. Inverse transformasi
# ============================================================
print("\n--- 5. Inverse ---")

# Inverse matriks homogen
M_inv = np.linalg.inv(M_gabung)
img_inv = cv2.warpAffine(img_gabung, M_inv[:2], (w, h))
diff_inv = np.mean(cv2.absdiff(img, img_inv))
print(f"  Inverse error: {diff_inv:.2f}")

# ============================================================
# 6. Transformasi titik
# ============================================================
print("\n--- 6. Transformasi Titik ---")

titik = np.float64([100, 100, 1])  # Koordinat homogen
titik_baru = M_gabung @ titik
print(f"  Titik (100,100) → ({titik_baru[0]:.1f}, {titik_baru[1]:.1f})")
titik_back = M_inv @ titik_baru
print(f"  Inverse → ({titik_back[0]:.1f}, {titik_back[1]:.1f})")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")

img_t = cv2.warpAffine(img, M_trans[:2], (w, h))
axes[0, 1].imshow(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Translasi")

axes[0, 2].imshow(cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Rotasi 30°")

img_s = cv2.warpAffine(img, (M_from_origin @ M_scale @ M_to_origin)[:2], (w, h))
axes[0, 3].imshow(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Scale (0.8,1.2)")

img_sh = cv2.warpAffine(img, M_shear[:2], (w, h))
axes[1, 0].imshow(cv2.cvtColor(img_sh, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Shear 0.3")
axes[1, 1].imshow(cv2.cvtColor(img_gabung, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("S→R→T Gabungan")
axes[1, 2].imshow(cv2.cvtColor(img_rt, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Rot→Trans")
axes[1, 3].imshow(cv2.cvtColor(img_tr, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Trans→Rot (beda!)")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Percobaan 15: Matriks Transformasi Homogen", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "15_matriks_transformasi_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
