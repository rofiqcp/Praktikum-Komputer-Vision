"""
==========================================================================
PERCOBAAN 05: TRANSFORMASI PERSPEKTIF
==========================================================================
Transformasi perspektif memetakan 4 titik sumber ke 4 titik tujuan
menggunakan matriks 3×3. Tidak mempertahankan garis paralel.

Fungsi:
- cv2.getPerspectiveTransform(src, dst) → Matriks perspektif 3×3
- cv2.warpPerspective(src, M, dsize) → Terapkan transformasi
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
    img = cv2.imread(os.path.join(IMAGE_DIR, "grid.png"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()

img = cv2.resize(img, (300, 300))
h, w = img.shape[:2]

print("=" * 60)
print("PERCOBAAN 05: TRANSFORMASI PERSPEKTIF")
print("=" * 60)

# ============================================================
# 1. Perspektif dasar (4 titik)
# ============================================================
print("\n--- 1. Perspektif Dasar ---")

# 4 titik sudut gambar asli
src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
# 4 titik tujuan (efek miring)
dst = np.float32([[50, 30], [250, 0], [280, 290], [20, 260]])

# cv2.getPerspectiveTransform: matriks 3×3 dari 4 pasang titik
M = cv2.getPerspectiveTransform(src, dst)
print(f"  Matriks Perspektif 3×3:\n{M}")

# cv2.warpPerspective menerapkan transformasi perspektif
img_persp = cv2.warpPerspective(img, M, (w, h))

# ============================================================
# 2. Koreksi perspektif (dokumen scanner)
# ============================================================
print("\n--- 2. Document Scanner ---")

# Simulasi: 4 sudut dokumen miring di foto
src_doc = np.float32([[30, 20], [220, 0], [250, 290], [0, 270]])
# Tujuan: persegi panjang sempurna
dst_doc = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

M_scan = cv2.getPerspectiveTransform(src_doc, dst_doc)
img_scan = cv2.warpPerspective(img, M_scan, (w, h))
print("  Koreksi dokumen miring → lurus")

# ============================================================
# 3. Efek bird's eye view (pandangan atas)
# ============================================================
print("\n--- 3. Bird's Eye View ---")

src_bird = np.float32([[50, 0], [250, 0], [300, 300], [0, 300]])
dst_bird = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
M_bird = cv2.getPerspectiveTransform(src_bird, dst_bird)
img_bird = cv2.warpPerspective(img, M_bird, (w, h))
print("  Trapesium → persegi (bird's eye)")

# ============================================================
# 4. Efek zoom perspektif
# ============================================================
print("\n--- 4. Zoom Perspektif ---")

# Zoom in perspektif (sisi jauh membesar)
src_zoom = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst_zoom = np.float32([[40, 40], [w-40, 40], [w, h], [0, h]])
M_zoom = cv2.getPerspectiveTransform(src_zoom, dst_zoom)
img_zoom = cv2.warpPerspective(img, M_zoom, (w, h))

# Efek tilt ke belakang
dst_tilt = np.float32([[60, 0], [w-60, 0], [w, h], [0, h]])
M_tilt = cv2.getPerspectiveTransform(src_zoom, dst_tilt)
img_tilt = cv2.warpPerspective(img, M_tilt, (w, h))
print("  Zoom perspektif dan tilt")

# ============================================================
# 5. Inverse perspective
# ============================================================
print("\n--- 5. Inverse Perspektif ---")

# Gunakan flag WARP_INVERSE_MAP untuk inverse
M_inv = cv2.getPerspectiveTransform(dst, src)
img_recovered = cv2.warpPerspective(img_persp, M_inv, (w, h))
diff = np.mean(cv2.absdiff(img, img_recovered))
print(f"  Perbedaan original vs recovered: {diff:.2f}")

# ============================================================
# 6. Transformasi 4 sudut interaktif
# ============================================================
print("\n--- 6. Variasi Transformasi ---")

variasi = [
    ("Condong Kanan", [[0,0],[w,0],[w-60,h],[60,h]]),
    ("Condong Kiri",  [[60,0],[w-60,0],[w,h],[0,h]]),
    ("Trapesium Atas", [[50,0],[w-50,0],[w,h],[0,h]]),
    ("Trapesium Bawah",[[0,0],[w,0],[w-50,h],[50,h]]),
]

hasil_var = {}
for nama, pts in variasi:
    dst_v = np.float32(pts)
    M_v = cv2.getPerspectiveTransform(src, dst_v)
    hasil_var[nama] = cv2.warpPerspective(img, M_v, (w, h))
    print(f"  {nama}")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 1].imshow(cv2.cvtColor(img_persp, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Perspektif")
axes[0, 2].imshow(cv2.cvtColor(img_scan, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Document Scanner")
axes[0, 3].imshow(cv2.cvtColor(img_bird, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Bird's Eye")

axes[1, 0].imshow(cv2.cvtColor(img_tilt, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Tilt")
for i, (nama, im) in enumerate(list(hasil_var.items())[:3]):
    axes[1, i+1].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    axes[1, i+1].set_title(nama)

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Percobaan 05: Transformasi Perspektif", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "05_transformasi_perspektif_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
