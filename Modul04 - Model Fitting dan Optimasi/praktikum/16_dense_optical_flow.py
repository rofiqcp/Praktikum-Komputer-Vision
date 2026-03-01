"""
==========================================================================
PERCOBAAN 16: DENSE OPTICAL FLOW (FARNEBACK)
==========================================================================
Dense optical flow menghitung vektor perpindahan untuk SETIAP piksel,
berbeda dengan Lucas-Kanade yang hanya melacak titik fitur tertentu.
Metode Farneback mengaproksimasi neighborhood setiap piksel dengan
polinomial kuadrat dan meminimalkan error perpindahan.

Fungsi utama:
- cv2.calcOpticalFlowFarneback() : dense optical flow (semua piksel)
- cv2.cartToPolar()              : konversi (dx,dy) ke (magnitude, angle)
- cv2.cvtColor() HSV→BGR        : visualisasi flow sebagai warna
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 16: DENSE OPTICAL FLOW (FARNEBACK)")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Memuat dua frame
# ============================================================
print("\n--- 1. Memuat Dua Frame ---")

frame1_path = os.path.join(IMAGE_DIR, "frame1.png")
frame2_path = os.path.join(IMAGE_DIR, "frame2.png")

if not os.path.exists(frame1_path) or not os.path.exists(frame2_path):
    print("[ERROR] frame1.png / frame2.png tidak ditemukan. Jalankan download_image.py!"); exit()

frame1 = cv2.imread(frame1_path)
frame2 = cv2.imread(frame2_path)

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
print(f"  Frame 1: {frame1.shape}")
print(f"  Frame 2: {frame2.shape}")

# ============================================================
# 2. Farneback Dense Optical Flow
# ============================================================
print("\n--- 2. Farneback Optical Flow ---")

# cv2.calcOpticalFlowFarneback parameters:
# prev, next: frame grayscale
# flow: output (None = compute from scratch)
# pyr_scale: skala piramida (0.5 = piramida klasik)
# levels: jumlah level piramida
# winsize: ukuran averaging window
# iterations: iterasi per level
# poly_n: ukuran neighborhood polynomial (biasanya 5 atau 7)
# poly_sigma: smoothing sigma untuk polinomial
# flags: opsi tambahan

flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2,
    flow=None,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# flow shape: (H, W, 2) — berisi (dx, dy) untuk setiap piksel
print(f"  Flow shape: {flow.shape}")
print(f"  Flow dx range: [{flow[:,:,0].min():.2f}, {flow[:,:,0].max():.2f}]")
print(f"  Flow dy range: [{flow[:,:,1].min():.2f}, {flow[:,:,1].max():.2f}]")

# ============================================================
# 3. Visualisasi HSV (warna = arah, brightness = magnitude)
# ============================================================
print("\n--- 3. Visualisasi HSV ---")

# Konversi flow ke magnitude dan angle
mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

# Buat gambar HSV
# Hue = arah flow (0-360°)
# Saturation = 255 (penuh)
# Value = magnitude (dinormalisasi)
hsv = np.zeros((*gray1.shape, 3), dtype=np.uint8)
hsv[:, :, 0] = ang * 180 / np.pi / 2  # hue: 0-180 di OpenCV
hsv[:, :, 1] = 255                      # saturasi penuh
hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # value

# Konversi ke BGR untuk tampilan
img_flow_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite(os.path.join(OUTPUT_DIR, "16_flow_hsv.png"), img_flow_hsv)

print(f"  Mean magnitude: {mag.mean():.2f}")
print(f"  Max magnitude: {mag.max():.2f}")

# ============================================================
# 4. Visualisasi Quiver (panah)
# ============================================================
print("\n--- 4. Visualisasi Quiver ---")

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

# Subsample agar panah tidak terlalu padat
step = 15
h, w = gray1.shape
y, x = np.mgrid[step//2:h:step, step//2:w:step]
fx = flow[step//2:h:step, step//2:w:step, 0]
fy = flow[step//2:h:step, step//2:w:step, 1]

# Quiver plot
ax.quiver(x, y, fx, fy, color='lime', angles='xy', scale_units='xy', scale=0.5)
ax.set_title("Dense Optical Flow (Quiver)")
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "16_flow_quiver.png"), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 5. Pengaruh Parameter - winsize
# ============================================================
print("\n--- 5. Pengaruh Window Size ---")

winsizes = [5, 11, 15, 21, 31]
flow_results = {}

for ws in winsizes:
    flow_ws = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, ws, 3, 5, 1.2, 0
    )
    mag_ws, _ = cv2.cartToPolar(flow_ws[:, :, 0], flow_ws[:, :, 1])
    flow_results[ws] = (flow_ws, mag_ws)
    print(f"  winsize={ws:2d}: mean_mag={mag_ws.mean():.3f}, max_mag={mag_ws.max():.3f}")

# ============================================================
# 6. Pengaruh Parameter - poly_n
# ============================================================
print("\n--- 6. Pengaruh poly_n ---")

poly_ns = [3, 5, 7]
poly_sigmas = [0.8, 1.2, 1.5]

for pn, ps in zip(poly_ns, poly_sigmas):
    flow_pn = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, pn, ps, 0
    )
    mag_pn, _ = cv2.cartToPolar(flow_pn[:, :, 0], flow_pn[:, :, 1])
    print(f"  poly_n={pn}, poly_sigma={ps}: mean_mag={mag_pn.mean():.3f}")

# ============================================================
# 7. Flow warping (verifikasi kualitas)
# ============================================================
print("\n--- 7. Flow Warping ---")

# Gunakan flow untuk warp frame1 → frame2 (estimasi)
h_img, w_img = gray1.shape
y_coords, x_coords = np.mgrid[0:h_img, 0:w_img].astype(np.float32)

# Lokasi baru = lokasi asli + flow
map_x = x_coords + flow[:, :, 0]
map_y = y_coords + flow[:, :, 1]

# cv2.remap: warp gambar menggunakan mapping
warped = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)

# Hitung error antara warped dan frame2
error = cv2.absdiff(warped, frame2)
error_gray = cv2.cvtColor(error, cv2.COLOR_BGR2GRAY)
mean_error = error_gray.mean()

cv2.imwrite(os.path.join(OUTPUT_DIR, "16_warped.png"), warped)
cv2.imwrite(os.path.join(OUTPUT_DIR, "16_warp_error.png"), error)

print(f"  Mean warp error: {mean_error:.2f}")
print(f"  Max warp error: {error_gray.max()}")

# ============================================================
# 8. Visualisasi gabungan
# ============================================================
print("\n--- 8. Visualisasi Gabungan ---")

fig, axes = plt.subplots(2, 4, figsize=(24, 12))

axes[0, 0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Frame 1")
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Frame 2")
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(img_flow_hsv, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Flow HSV")
axes[0, 2].axis('off')

axes[0, 3].imshow(mag, cmap='hot')
axes[0, 3].set_title("Flow Magnitude")
axes[0, 3].axis('off')

axes[1, 0].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Warped Frame 1")
axes[1, 0].axis('off')

axes[1, 1].imshow(error_gray, cmap='hot')
axes[1, 1].set_title(f"Warp Error (mean={mean_error:.1f})")
axes[1, 1].axis('off')

# Komponen x dan y flow
axes[1, 2].imshow(flow[:, :, 0], cmap='coolwarm')
axes[1, 2].set_title("Flow dx")
axes[1, 2].axis('off')

axes[1, 3].imshow(flow[:, :, 1], cmap='coolwarm')
axes[1, 3].set_title("Flow dy")
axes[1, 3].axis('off')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "16_dense_flow_all.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Disimpan: {output_path}")

print("\n" + "=" * 60)
print("PERCOBAAN 16 SELESAI")
print("=" * 60)
