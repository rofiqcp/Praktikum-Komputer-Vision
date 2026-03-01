"""
==========================================================================
PERCOBAAN 08: KOREKSI DISTORSI LENSA
==========================================================================
Distorsi lensa menyebabkan garis lurus melengkung. Program ini
mensimulasikan dan mengoreksi distorsi radial dan tangensial.

Fungsi:
- cv2.undistort(src, mtx, dist) → Koreksi distorsi
- cv2.initUndistortRectifyMap() → Peta koreksi (untuk remap)
- cv2.remap() → Terapkan peta koreksi
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

img = cv2.imread(os.path.join(IMAGE_DIR, "grid.png"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()
img = cv2.resize(img, (400, 400))
h, w = img.shape[:2]

print("=" * 60)
print("PERCOBAAN 08: KOREKSI DISTORSI LENSA")
print("=" * 60)

# ============================================================
# 1. Simulasi distorsi barrel (k1 > 0)
# ============================================================
print("\n--- 1. Simulasi Distorsi ---")

def simulasi_distorsi(img, k1=0, k2=0, p1=0, p2=0):
    """Simulasi distorsi lensa menggunakan remap."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    # Grid koordinat piksel
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            # Normalisasi koordinat ke [-1, 1]
            xn = (x - cx) / cx
            yn = (y - cy) / cy
            r2 = xn ** 2 + yn ** 2
            r4 = r2 ** 2
            # Model distorsi radial + tangensial
            radial = 1 + k1 * r2 + k2 * r4
            xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2)
            yd = yn * radial + p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn
            map_x[y, x] = xd * cx + cx
            map_y[y, x] = yd * cy + cy

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# Barrel distortion (k1 positif → membengkak)
img_barrel = simulasi_distorsi(img, k1=0.3)
# Pincushion distortion (k1 negatif → menyempit)
img_pincushion = simulasi_distorsi(img, k1=-0.3)
# Distorsi kuat
img_strong = simulasi_distorsi(img, k1=0.5, k2=0.3)
print("  Barrel (k1=0.3), Pincushion (k1=-0.3), Kuat (k1=0.5,k2=0.3)")

# ============================================================
# 2. Koreksi menggunakan cv2.undistort
# ============================================================
print("\n--- 2. cv2.undistort ---")

# Parameter kamera simulasi
fx, fy = 300, 300
cx_cam, cy_cam = w / 2, h / 2
camera_matrix = np.float64([[fx, 0, cx_cam], [0, fy, cy_cam], [0, 0, 1]])

# Koefisien distorsi [k1, k2, p1, p2, k3]
dist_barrel = np.float64([0.3, 0, 0, 0, 0])

# cv2.undistort mengoreksi gambar yang terdistorsi
img_corrected = cv2.undistort(img_barrel, camera_matrix, dist_barrel)
print("  Barrel → koreksi")

# ============================================================
# 3. Koreksi dengan initUndistortRectifyMap + remap
# ============================================================
print("\n--- 3. Map-based Undistort ---")

# Metode lebih efisien untuk banyak gambar (hitung map sekali)
# cv2.initUndistortRectifyMap menghitung peta x,y untuk koreksi
new_mtx, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_barrel, (w, h), 1, (w, h)
)
mapx, mapy = cv2.initUndistortRectifyMap(
    camera_matrix, dist_barrel, None, new_mtx, (w, h), cv2.CV_32FC1
)
img_remap = cv2.remap(img_barrel, mapx, mapy, cv2.INTER_LINEAR)
print(f"  ROI setelah undistort: {roi}")

# ============================================================
# 4. Distorsi tangensial
# ============================================================
print("\n--- 4. Distorsi Tangensial ---")
img_tang = simulasi_distorsi(img, p1=0.1, p2=0.05)
print("  Tangensial: p1=0.1, p2=0.05")

# ============================================================
# 5. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original (Grid)")
axes[0, 1].imshow(cv2.cvtColor(img_barrel, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Barrel (k1=0.3)")
axes[0, 2].imshow(cv2.cvtColor(img_pincushion, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Pincushion (k1=-0.3)")
axes[0, 3].imshow(cv2.cvtColor(img_strong, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Kuat (k1=0.5,k2=0.3)")

axes[1, 0].imshow(cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Barrel → Undistort")
axes[1, 1].imshow(cv2.cvtColor(img_remap, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Remap Undistort")
axes[1, 2].imshow(cv2.cvtColor(img_tang, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Tangensial")
axes[1, 3].axis("off")

for ax in axes.flat:
    ax.set_aspect("equal")

plt.suptitle("Percobaan 08: Koreksi Distorsi Lensa", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "08_koreksi_distorsi_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
