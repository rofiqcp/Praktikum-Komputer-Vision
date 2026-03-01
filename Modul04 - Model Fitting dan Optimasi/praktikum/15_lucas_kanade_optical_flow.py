"""
==========================================================================
PERCOBAAN 15: LUCAS-KANADE OPTICAL FLOW
==========================================================================
Lucas-Kanade adalah metode sparse optical flow yang melacak pergerakan
titik-titik fitur antar frame. Asumsi: perpindahan kecil dan konstan
di dalam window kecil (local constraint).

Fungsi utama:
- cv2.goodFeaturesToTrack()     : deteksi corner (Shi-Tomasi) untuk tracking
- cv2.calcOpticalFlowPyrLK()    : hitung Lucas-Kanade optical flow (piramida)
- cv2.cornerSubPix()            : refine corner ke sub-pixel
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
print("PERCOBAAN 15: LUCAS-KANADE OPTICAL FLOW")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Membuat atau memuat dua frame
# ============================================================
print("\n--- 1. Memuat Dua Frame ---")

frame1_path = os.path.join(IMAGE_DIR, "frame1.png")
frame2_path = os.path.join(IMAGE_DIR, "frame2.png")

if not os.path.exists(frame1_path):
    print("[ERROR] frame1.png tidak ditemukan. Jalankan download_image.py!"); exit()

if not os.path.exists(frame2_path):
    print("[ERROR] frame2.png tidak ditemukan. Jalankan download_image.py!"); exit()

frame1 = cv2.imread(frame1_path)
frame2 = cv2.imread(frame2_path)
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

print(f"  Frame 1: {frame1.shape}")
print(f"  Frame 2: {frame2.shape}")

# ============================================================
# 2. Deteksi Feature untuk Tracking
# ============================================================
print("\n--- 2. Deteksi Feature (Shi-Tomasi) ---")

# Parameter untuk cv2.goodFeaturesToTrack (Shi-Tomasi corner detector)
feature_params = dict(
    maxCorners=100,      # maksimum jumlah corner
    qualityLevel=0.3,    # threshold kualitas (fraksi dari corner terkuat)
    minDistance=7,        # jarak minimum antar corner
    blockSize=7          # ukuran neighborhood untuk perhitungan corner
)

# Deteksi corner di frame pertama
p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
print(f"  Jumlah corner terdeteksi: {len(p0)}")

# Tampilkan corner pada frame 1
img_corners = frame1.copy()
for pt in p0:
    x, y = pt.ravel()
    cv2.circle(img_corners, (int(x), int(y)), 5, (0, 255, 0), -1)

cv2.imwrite(os.path.join(OUTPUT_DIR, "15_corners_frame1.png"), img_corners)

# ============================================================
# 3. Lucas-Kanade Optical Flow
# ============================================================
print("\n--- 3. Lucas-Kanade Optical Flow ---")

# Parameter untuk cv2.calcOpticalFlowPyrLK
lk_params = dict(
    winSize=(15, 15),     # ukuran window pencarian
    maxLevel=2,           # jumlah level piramida (0 = tanpa piramida)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Hitung optical flow dari frame1 ke frame2
# p1: posisi baru di frame2
# st: status (1 = berhasil dilacak)
# err: error tracking
p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

# Filter titik yang berhasil dilacak
good_new = p1[st == 1]
good_old = p0[st == 1]

print(f"  Titik terdeteksi: {len(p0)}")
print(f"  Titik berhasil dilacak: {len(good_new)}")
print(f"  Mean error: {err[st == 1].mean():.2f}")

# ============================================================
# 4. Visualisasi Flow Vectors
# ============================================================
print("\n--- 4. Visualisasi Flow Vectors ---")

img_flow = frame2.copy()
mask_flow = np.zeros_like(frame1)

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    a, b, c, d = int(a), int(b), int(c), int(d)
    
    # Hitung displacement
    dx, dy = a - c, b - d
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Gambar garis flow (dari posisi lama ke baru)
    mask_flow = cv2.line(mask_flow, (a, b), (c, d), (0, 255, 0), 2)
    # Gambar titik baru
    img_flow = cv2.circle(img_flow, (a, b), 5, (0, 0, 255), -1)
    # Gambar titik lama
    img_flow = cv2.circle(img_flow, (c, d), 3, (255, 0, 0), -1)
    
    if i < 5:
        print(f"  Titik {i}: ({c},{d}) → ({a},{b}), Δ=({dx},{dy}), mag={magnitude:.1f}")

# Gabungkan mask flow dengan gambar
img_flow_overlay = cv2.add(img_flow, mask_flow)
cv2.imwrite(os.path.join(OUTPUT_DIR, "15_lk_flow.png"), img_flow_overlay)

# ============================================================
# 5. Pengaruh Window Size
# ============================================================
print("\n--- 5. Pengaruh Window Size ---")

win_sizes = [5, 11, 15, 21, 31]
results_win = {}

for ws in win_sizes:
    lk_params_ws = dict(
        winSize=(ws, ws),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    p1_ws, st_ws, err_ws = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params_ws)
    
    good_count = np.sum(st_ws == 1)
    mean_err = err_ws[st_ws == 1].mean() if good_count > 0 else float('inf')
    results_win[ws] = (good_count, mean_err)
    
    print(f"  winSize={ws:2d}: tracked={good_count}, mean_err={mean_err:.2f}")

# ============================================================
# 6. Pengaruh Pyramid Level
# ============================================================
print("\n--- 6. Pengaruh Pyramid Level ---")

levels = [0, 1, 2, 3, 4]
results_pyr = {}

for lvl in levels:
    lk_params_lvl = dict(
        winSize=(15, 15),
        maxLevel=lvl,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    p1_lvl, st_lvl, err_lvl = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params_lvl)
    
    good_count = np.sum(st_lvl == 1)
    mean_err = err_lvl[st_lvl == 1].mean() if good_count > 0 else float('inf')
    results_pyr[lvl] = (good_count, mean_err)
    
    print(f"  maxLevel={lvl}: tracked={good_count}, mean_err={mean_err:.2f}")

# ============================================================
# 7. Backward verification (forward-backward error)
# ============================================================
print("\n--- 7. Forward-Backward Verification ---")

# Forward: frame1 → frame2
p1_fwd, st_fwd, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
# Backward: frame2 → frame1
p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, p1_fwd, None, **lk_params)

# Hitung forward-backward error
fb_error = np.sqrt(np.sum((p0 - p0_back) ** 2, axis=2)).ravel()

# Filter titik dengan FB error rendah (tracking stabil)
fb_threshold = 1.0
good_fb = fb_error < fb_threshold

print(f"  Forward-backward error (mean): {fb_error.mean():.2f}")
print(f"  Titik dengan FB error < {fb_threshold}: {np.sum(good_fb)}/{len(fb_error)}")

# Visualisasi FB error
img_fb = frame2.copy()
for i in range(len(p0)):
    if st_fwd[i] == 1:
        a, b = p1_fwd[i].ravel()
        color = (0, 255, 0) if fb_error[i] < fb_threshold else (0, 0, 255)
        cv2.circle(img_fb, (int(a), int(b)), 5, color, -1)

cv2.imwrite(os.path.join(OUTPUT_DIR, "15_fb_verification.png"), img_fb)

# ============================================================
# 8. Visualisasi gabungan
# ============================================================
print("\n--- 8. Visualisasi Gabungan ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Frame 1")

axes[0, 1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Frame 2")

axes[0, 2].imshow(cv2.cvtColor(img_flow_overlay, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Lucas-Kanade Flow")

# Plot displacement magnitudes
mags = [np.sqrt((n[0]-o[0])**2 + (n[1]-o[1])**2) for n, o in zip(good_new, good_old)]
axes[1, 0].hist(mags, bins=20, color='steelblue', edgecolor='black')
axes[1, 0].set_xlabel("Displacement")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Displacement Distribution")

# Plot window size vs error
axes[1, 1].plot(list(results_win.keys()),
                [v[1] for v in results_win.values()], 'ro-')
axes[1, 1].set_xlabel("Window Size")
axes[1, 1].set_ylabel("Mean Error")
axes[1, 1].set_title("Win Size vs Error")

axes[1, 2].imshow(cv2.cvtColor(img_fb, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("FB Verification\n(Green=good, Red=bad)")

for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]]:
    ax.axis('off')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "15_lucas_kanade_all.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Disimpan: {output_path}")

print("\n" + "=" * 60)
print("PERCOBAAN 15 SELESAI")
print("=" * 60)
