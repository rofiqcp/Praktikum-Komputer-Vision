"""
==========================================================================
PERCOBAAN 01: TRANSLASI GAMBAR
==========================================================================
Translasi adalah pergeseran gambar secara horizontal (x) dan vertikal (y)
menggunakan matriks transformasi affine 2×3.

Matriks Translasi:
  M = [[1, 0, tx],    tx = pergeseran horizontal (+kanan, -kiri)
       [0, 1, ty]]    ty = pergeseran vertikal   (+bawah, -atas)

Fungsi: cv2.warpAffine(src, M, dsize)
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
print("PERCOBAAN 01: TRANSLASI GAMBAR")
print("=" * 60)

# Membaca gambar grid (cocok untuk melihat pergeseran)
img = cv2.imread(os.path.join(IMAGE_DIR, "grid.png"))
if img is None:
    img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!")
    exit()

img = cv2.resize(img, (300, 300))
h, w = img.shape[:2]

# ============================================================
# 1. Translasi ke kanan dan bawah
# ============================================================
print("\n--- 1. Translasi Dasar ---")

# Matriks translasi: geser 50px ke kanan, 30px ke bawah
# np.float32 penting karena warpAffine membutuhkan float
tx, ty = 50, 30
M = np.float32([[1, 0, tx],
                [0, 1, ty]])

# cv2.warpAffine menerapkan transformasi affine
img_geser = cv2.warpAffine(img, M, (w, h))
print(f"  tx={tx}, ty={ty} → kanan-bawah")

# ============================================================
# 2. Translasi ke berbagai arah
# ============================================================
print("\n--- 2. Berbagai Arah ---")

arah = {
    "kanan_bawah": (60, 40),
    "kiri_atas": (-60, -40),
    "kanan_atas": (60, -40),
    "kiri_bawah": (-60, 40),
}

hasil_arah = {}
for nama, (tx, ty) in arah.items():
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    hasil_arah[nama] = cv2.warpAffine(img, M, (w, h))
    print(f"  {nama}: tx={tx:+d}, ty={ty:+d}")

# ============================================================
# 3. Translasi tanpa kehilangan area (canvas lebih besar)
# ============================================================
print("\n--- 3. Translasi dengan Canvas Besar ---")

tx, ty = 80, 60
M = np.float32([[1, 0, tx], [0, 1, ty]])
# Canvas lebih besar agar gambar tidak terpotong
new_w = w + abs(tx)
new_h = h + abs(ty)
img_full = cv2.warpAffine(img, M, (new_w, new_h))
print(f"  Canvas: {w}×{h} → {new_w}×{new_h}")

# ============================================================
# 4. Translasi menggunakan NumPy (alternatif)
# ============================================================
print("\n--- 4. Translasi NumPy ---")

tx_np, ty_np = 40, 30
# Membuat canvas kosong
img_np = np.zeros_like(img)
# Salin area yang valid setelah pergeseran
# Area sumber: [0 : h-ty, 0 : w-tx]
# Area target: [ty : h, tx : w]
if ty_np > 0 and tx_np > 0:
    img_np[ty_np:h, tx_np:w] = img[0:h-ty_np, 0:w-tx_np]
print(f"  NumPy translasi: tx={tx_np}, ty={ty_np}")

# Verifikasi kedua metode menghasilkan hasil yang sama
M_ver = np.float32([[1, 0, tx_np], [0, 1, ty_np]])
img_cv = cv2.warpAffine(img, M_ver, (w, h))
diff = np.mean(cv2.absdiff(img_np, img_cv))
print(f"  Perbedaan NumPy vs CV: {diff:.4f}")

# ============================================================
# 5. Animasi translasi (beberapa frame)
# ============================================================
print("\n--- 5. Animasi Translasi ---")

frames = []
for i in range(6):
    # Geser secara bertahap
    tx_anim = i * 15
    ty_anim = i * 10
    M_anim = np.float32([[1, 0, tx_anim], [0, 1, ty_anim]])
    frame = cv2.warpAffine(img, M_anim, (w, h))
    frames.append(frame)
    print(f"  Frame {i}: tx={tx_anim}, ty={ty_anim}")

# ============================================================
# 6. Translasi dengan border mode berbeda
# ============================================================
print("\n--- 6. Border Mode ---")

tx, ty = 80, 60
M = np.float32([[1, 0, tx], [0, 1, ty]])

# borderMode menentukan apa yang mengisi area kosong
img_border_const = cv2.warpAffine(img, M, (w, h),
    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 255))
img_border_reflect = cv2.warpAffine(img, M, (w, h),
    borderMode=cv2.BORDER_REFLECT)
img_border_wrap = cv2.warpAffine(img, M, (w, h),
    borderMode=cv2.BORDER_WRAP)

print("  CONSTANT (merah), REFLECT, WRAP")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 1].imshow(cv2.cvtColor(hasil_arah["kanan_bawah"], cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Kanan+Bawah")
axes[0, 2].imshow(cv2.cvtColor(hasil_arah["kiri_atas"], cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Kiri+Atas")
axes[0, 3].imshow(cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Canvas Besar")

axes[1, 0].imshow(cv2.cvtColor(img_border_const, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Border CONSTANT")
axes[1, 1].imshow(cv2.cvtColor(img_border_reflect, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Border REFLECT")
axes[1, 2].imshow(cv2.cvtColor(img_border_wrap, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Border WRAP")
axes[1, 3].imshow(cv2.cvtColor(frames[4], cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Animasi Frame 4")

for ax in axes.flat:
    ax.axis("off")

plt.suptitle("Percobaan 01: Translasi Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "01_translasi_gambar_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

print("\n" + "=" * 60)
print("RINGKASAN: M = [[1,0,tx],[0,1,ty]]")
print("  tx>0 → kanan, tx<0 → kiri")
print("  ty>0 → bawah, ty<0 → atas")
print("  cv2.warpAffine(src, M, (w,h))")
print("=" * 60)
