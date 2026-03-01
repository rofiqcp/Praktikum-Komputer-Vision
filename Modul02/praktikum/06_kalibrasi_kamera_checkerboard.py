"""
==========================================================================
PERCOBAAN 06: KALIBRASI KAMERA DENGAN CHECKERBOARD
==========================================================================
Kalibrasi kamera mengestimasi parameter intrinsik (focal length,
principal point, distorsi) menggunakan pola checkerboard.

Fungsi:
- cv2.findChessboardCorners() → Deteksi sudut checkerboard
- cv2.cornerSubPix() → Sub-pixel refinement
- cv2.drawChessboardCorners() → Visualisasi
- cv2.calibrateCamera() → Estimasi parameter kamera
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
print("PERCOBAAN 06: KALIBRASI KAMERA")
print("=" * 60)

# ============================================================
# 1. Membuat beberapa gambar checkerboard dari sudut berbeda
# ============================================================
print("\n--- 1. Membuat Data Kalibrasi ---")

# Ukuran checkerboard interior corners
ROWS, COLS = 7, 10
CELL = 40

# Membuat checkerboard dasar
cb = np.zeros(((ROWS + 1) * CELL, (COLS + 1) * CELL), dtype=np.uint8)
for r in range(ROWS + 1):
    for c in range(COLS + 1):
        if (r + c) % 2 == 0:
            cb[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = 255

# Border putih
cb_bordered = cv2.copyMakeBorder(cb, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
cb_bgr = cv2.cvtColor(cb_bordered, cv2.COLOR_GRAY2BGR)

# Buat beberapa view dari sudut berbeda (simulasi)
views = []
h_cb, w_cb = cb_bgr.shape[:2]
src = np.float32([[0, 0], [w_cb, 0], [w_cb, h_cb], [0, h_cb]])

perspektif_list = [
    np.float32([[20, 10], [w_cb-10, 0], [w_cb-20, h_cb], [10, h_cb-10]]),
    np.float32([[0, 20], [w_cb, 10], [w_cb-30, h_cb-10], [30, h_cb]]),
    np.float32([[40, 0], [w_cb-10, 30], [w_cb, h_cb-20], [10, h_cb-30]]),
    np.float32([[10, 30], [w_cb-30, 10], [w_cb-10, h_cb], [20, h_cb-20]]),
]

for i, dst in enumerate(perspektif_list):
    M = cv2.getPerspectiveTransform(src, dst)
    view = cv2.warpPerspective(cb_bgr, M, (w_cb, h_cb), borderValue=(200, 200, 200))
    views.append(view)
    print(f"  View {i + 1} dibuat")

# ============================================================
# 2. Deteksi sudut checkerboard
# ============================================================
print("\n--- 2. Deteksi Sudut ---")

# Titik 3D objek (koordinat nyata checkerboard)
# Diasumsikan z=0 (papan datar), x,y = posisi sudut
objp = np.zeros((ROWS * COLS, 3), np.float32)
# np.mgrid membuat grid koordinat
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)

obj_points = []  # Titik 3D di dunia nyata
img_points = []  # Titik 2D di gambar
detected_views = []

# Kriteria terminasi untuk cornerSubPix
# (tipe, max_iterasi, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for i, view in enumerate(views):
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)

    # cv2.findChessboardCorners mendeteksi sudut interior checkerboard
    # Returns: (found, corners)
    found, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), None)

    if found:
        obj_points.append(objp)

        # cv2.cornerSubPix memperbaiki posisi sudut ke presisi sub-pixel
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners_refined)

        # cv2.drawChessboardCorners menggambar sudut yang terdeteksi
        view_drawn = view.copy()
        cv2.drawChessboardCorners(view_drawn, (COLS, ROWS), corners_refined, found)
        detected_views.append(view_drawn)

        print(f"  View {i + 1}: TERDETEKSI ({len(corners)} sudut)")
    else:
        print(f"  View {i + 1}: Tidak terdeteksi")

# ============================================================
# 3. Kalibrasi kamera
# ============================================================
print("\n--- 3. Kalibrasi Kamera ---")

if len(obj_points) >= 2:
    # cv2.calibrateCamera mengestimasi parameter kamera
    # Returns: ret, camera_matrix, dist_coeffs, rvecs, tvecs
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    print(f"  RMS re-projection error: {ret:.4f}")
    print(f"\n  Camera Matrix (Intrinsik):")
    print(f"    fx = {mtx[0,0]:.2f}")
    print(f"    fy = {mtx[1,1]:.2f}")
    print(f"    cx = {mtx[0,2]:.2f}")
    print(f"    cy = {mtx[1,2]:.2f}")
    print(f"\n  Koefisien Distorsi: {dist.flatten()[:5]}")

    # ============================================================
    # 4. Undistort gambar
    # ============================================================
    print("\n--- 4. Undistort ---")
    
    # cv2.undistort mengoreksi distorsi lensa
    img_undist = cv2.undistort(views[0], mtx, dist)
    
    # cv2.getOptimalNewCameraMatrix untuk crop optimal setelah undistort
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w_cb, h_cb), 1, (w_cb, h_cb)
    )
    img_undist2 = cv2.undistort(views[0], mtx, dist, None, new_mtx)
    print("  Gambar di-undistort")
else:
    print("  [WARNING] Tidak cukup view untuk kalibrasi")
    mtx = np.eye(3)
    dist = np.zeros(5)

# ============================================================
# 5. Visualisasi
# ============================================================
n_views = min(len(detected_views), 4)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Baris 1: Checkerboard & deteksi
axes[0, 0].imshow(cv2.cvtColor(cb_bgr, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Checkerboard Asli")
for i in range(min(3, n_views)):
    axes[0, i + 1].imshow(cv2.cvtColor(detected_views[i], cv2.COLOR_BGR2RGB))
    axes[0, i + 1].set_title(f"View {i + 1} (corners)")

# Baris 2: Kalibrasi info
axes[1, 0].imshow(cv2.cvtColor(views[0], cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("View 1 Original")

if len(obj_points) >= 2:
    axes[1, 1].imshow(cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Undistorted")

# Plot camera matrix sebagai heatmap
axes[1, 2].imshow(mtx, cmap='hot')
axes[1, 2].set_title("Camera Matrix")
for (j, i), val in np.ndenumerate(mtx):
    axes[1, 2].text(i, j, f'{val:.0f}', ha='center', va='center', fontsize=8)

# Plot distortion coefficients
if len(obj_points) >= 2:
    axes[1, 3].bar(range(5), dist.flatten()[:5])
    axes[1, 3].set_title("Distortion Coeffs")
else:
    axes[1, 3].axis("off")

for ax in axes.flat:
    if not ax.has_data():
        ax.axis("off")

plt.suptitle("Percobaan 06: Kalibrasi Kamera", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "06_kalibrasi_kamera_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
