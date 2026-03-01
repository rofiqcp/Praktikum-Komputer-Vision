"""
==========================================================================
PERCOBAAN 12: STABILISASI VIDEO (VIDEO STABILIZATION)
==========================================================================
Program ini mempelajari pipeline stabilisasi video menggunakan
optical flow. Video yang tidak stabil (goyang/panning) dikoreksi
agar terlihat lebih halus.

Pipeline Stabilisasi Video:
1. Deteksi fitur di setiap frame (goodFeaturesToTrack)
2. Tracking fitur antar frame (calcOpticalFlowPyrLK)
3. Estimasi transformasi (estimateAffinePartial2D)
4. Akumulasi transformasi untuk mendapatkan trajectory
5. Haluskan trajectory (moving average)
6. Terapkan koreksi pada setiap frame

Fungsi utama yang dipelajari:
- cv2.goodFeaturesToTrack()        : Mendeteksi fitur corner
- cv2.calcOpticalFlowPyrLK()       : Tracking fitur antar frame
- cv2.estimateAffinePartial2D()    : Estimasi transformasi affine parsial
- cv2.warpAffine()                 : Menerapkan transformasi pada frame
- np.cumsum()                      : Akumulasi transformasi

Hasil: Perbandingan video asli vs video yang telah distabilkan
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan video
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file
import os

# Mengimpor matplotlib untuk menyimpan visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input dan output
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 12: STABILISASI VIDEO (VIDEO STABILIZATION)")
print("=" * 60)

# ============================================================
# 1. Membuka video panning (video yang bergoyang)
# ============================================================

# Membuka file video panning yang akan distabilkan
video_path = os.path.join(IMAGE_DIR, "video_panning.avi")
cap = cv2.VideoCapture(video_path)

# Memeriksa apakah video berhasil dibuka
if not cap.isOpened():
    print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Membaca properti video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"[INFO] Video berhasil dibuka: {width}x{height}")
print(f"[INFO] FPS: {fps}, Total frame: {total_frames}")

# ============================================================
# 2. Langkah 1 - Menghitung transformasi antar frame
# ============================================================

# Membaca frame pertama
ret, prev_frame = cap.read()
if not ret:
    print("[ERROR] Gagal membaca frame pertama.")
    exit()

# Mengkonversi ke grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Array untuk menyimpan transformasi setiap frame
# Setiap transformasi disimpan sebagai [dx, dy, da] (translasi x, y, dan rotasi)
transforms = []

# Parameter untuk deteksi fitur
feature_params = dict(
    maxCorners=200,       # Banyak fitur untuk estimasi yang robust
    qualityLevel=0.01,    # Threshold kualitas rendah agar banyak fitur
    minDistance=30,        # Jarak minimum antar fitur
    blockSize=3           # Ukuran blok komputasi
)

# Parameter untuk Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

print("[INFO] Langkah 1: Menghitung transformasi antar frame...")

# Counter frame
frame_count = 0

while True:
    # Membaca frame berikutnya
    ret, curr_frame = cap.read()
    if not ret:
        break

    # Mengkonversi frame saat ini ke grayscale
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi fitur corner pada frame sebelumnya
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Memeriksa apakah fitur ditemukan
    if prev_pts is None or len(prev_pts) < 4:
        # Jika terlalu sedikit fitur, gunakan transformasi identitas
        transforms.append([0, 0, 0])
        prev_gray = curr_gray.copy()
        frame_count += 1
        continue

    # Menghitung optical flow untuk tracking fitur
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_params
    )

    # Memfilter hanya titik yang berhasil di-track
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]

    # ============================================================
    # Estimasi transformasi affine parsial (translasi + rotasi + skala)
    # cv2.estimateAffinePartial2D() mengembalikan matriks 2x3:
    # [[cos(a)*s, -sin(a)*s, tx],
    #  [sin(a)*s,  cos(a)*s, ty]]
    # ============================================================
    if len(good_prev) >= 4:
        # Mengestimasi transformasi affine parsial menggunakan RANSAC
        m, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)

        if m is not None:
            # Mengekstrak translasi (dx, dy)
            dx = m[0, 2]
            dy = m[1, 2]
            # Mengekstrak rotasi (da) dari matriks transformasi
            da = np.arctan2(m[1, 0], m[0, 0])
        else:
            dx, dy, da = 0, 0, 0
    else:
        dx, dy, da = 0, 0, 0

    # Menyimpan transformasi frame ini
    transforms.append([dx, dy, da])

    # Memperbarui frame sebelumnya
    prev_gray = curr_gray.copy()
    frame_count += 1

# Menutup video capture
cap.release()

# Mengkonversi list ke array NumPy
transforms = np.array(transforms)
print(f"[INFO] Selesai menghitung {len(transforms)} transformasi.")

# ============================================================
# 3. Langkah 2 - Menghitung trajectory dan menghaluskannya
# ============================================================

# Menghitung trajectory kumulatif (akumulasi transformasi)
trajectory = np.cumsum(transforms, axis=0)

print("[INFO] Langkah 2: Menghaluskan trajectory...")

# Fungsi untuk menghaluskan trajectory menggunakan moving average
def smooth_trajectory(trajectory, radius=15):
    """
    Menghaluskan trajectory menggunakan moving average filter.
    radius: jumlah frame sebelum dan sesudah yang digunakan
    """
    # Membuat array untuk trajectory yang sudah dihaluskan
    smoothed = np.copy(trajectory)
    
    # Menerapkan moving average untuk setiap komponen (dx, dy, da)
    for i in range(3):
        # Menggunakan convolution dengan kernel rata-rata
        kernel_size = 2 * radius + 1
        kernel = np.ones(kernel_size) / kernel_size
        
        # Padding agar ukuran output sama
        padded = np.pad(trajectory[:, i], (radius, radius), mode='edge')
        smoothed[:, i] = np.convolve(padded, kernel, mode='valid')
    
    return smoothed

# Menghaluskan trajectory dengan radius 15 frame
SMOOTHING_RADIUS = 15
smoothed_trajectory = smooth_trajectory(trajectory, radius=SMOOTHING_RADIUS)

# Menghitung koreksi = trajectory halus - trajectory asli
correction = smoothed_trajectory - trajectory

# Menerapkan koreksi pada transformasi
transforms_corrected = transforms + correction

print(f"[INFO] Smoothing radius: {SMOOTHING_RADIUS} frame")

# ============================================================
# 4. Langkah 3 - Menerapkan stabilisasi pada video
# ============================================================

print("[INFO] Langkah 3: Menerapkan koreksi stabilisasi...")

# Membuka video lagi untuk membaca frame-frame
cap = cv2.VideoCapture(video_path)
ret, _ = cap.read()  # Skip frame pertama (referensi)

# Menyimpan frame untuk visualisasi
original_frames = []
stabilized_frames = []
vis_indices = []

# Menentukan frame mana yang akan divisualisasi
target_vis = [5, 20, 40, 60]

for i in range(len(transforms_corrected)):
    # Membaca frame berikutnya
    ret, frame = cap.read()
    if not ret:
        break

    # Mengekstrak transformasi terkoreksi
    dx = transforms_corrected[i, 0]
    dy = transforms_corrected[i, 1]
    da = transforms_corrected[i, 2]

    # Membuat matriks transformasi affine 2x3
    cos_a = np.cos(da)
    sin_a = np.sin(da)
    m = np.array([
        [cos_a, -sin_a, dx],
        [sin_a,  cos_a, dy]
    ], dtype=np.float64)

    # Menerapkan transformasi pada frame menggunakan warpAffine
    stabilized = cv2.warpAffine(frame, m, (width, height))

    # Menyimpan frame untuk visualisasi
    if i in target_vis:
        original_frames.append(frame.copy())
        stabilized_frames.append(stabilized.copy())
        vis_indices.append(i)

# Menutup video capture
cap.release()

print(f"[INFO] Stabilisasi selesai untuk {len(transforms_corrected)} frame.")

# ============================================================
# 5. Visualisasi perbandingan original vs stabilized
# ============================================================

# Membuat figure: original vs stabilized
num_vis = min(4, len(original_frames))
fig1, axes1 = plt.subplots(2, num_vis, figsize=(16, 8))

# Memastikan axes selalu 2D
if num_vis == 1:
    axes1 = axes1.reshape(2, 1)

for idx in range(num_vis):
    # Baris atas: frame original
    orig_rgb = cv2.cvtColor(original_frames[idx], cv2.COLOR_BGR2RGB)
    axes1[0, idx].imshow(orig_rgb)
    axes1[0, idx].set_title(f"Original Frame #{vis_indices[idx]}", fontsize=10)
    axes1[0, idx].axis("off")

    # Baris bawah: frame stabilized
    stab_rgb = cv2.cvtColor(stabilized_frames[idx], cv2.COLOR_BGR2RGB)
    axes1[1, idx].imshow(stab_rgb)
    axes1[1, idx].set_title(f"Stabilized Frame #{vis_indices[idx]}", fontsize=10)
    axes1[1, idx].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 12: Video Stabilization\nAtas = Original, Bawah = Stabilized",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan figure pertama
output_path_1 = os.path.join(OUTPUT_DIR, "12_video_stabilization_frames.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Hasil frame disimpan di: {output_path_1}")

# ============================================================
# 6. Visualisasi trajectory
# ============================================================

# Membuat figure: trajectory sebelum dan setelah smoothing
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10))

# Label untuk setiap komponen
labels = ['Translasi X (piksel)', 'Translasi Y (piksel)', 'Rotasi (radian)']
colors_orig = ['r', 'g', 'b']
colors_smooth = ['darkred', 'darkgreen', 'darkblue']

for i in range(3):
    # Menggambar trajectory asli
    axes2[i].plot(trajectory[:, i], color=colors_orig[i], alpha=0.5,
                  label='Trajectory Asli', linewidth=1)
    # Menggambar trajectory yang sudah dihaluskan
    axes2[i].plot(smoothed_trajectory[:, i], color=colors_smooth[i],
                  label='Trajectory Halus', linewidth=2)
    axes2[i].set_ylabel(labels[i], fontsize=10)
    axes2[i].legend(fontsize=9)
    axes2[i].grid(True, alpha=0.3)

# Menambahkan label sumbu x pada subplot terakhir
axes2[2].set_xlabel("Nomor Frame", fontsize=10)

# Menambahkan judul utama
plt.suptitle("Percobaan 12: Trajectory Kamera\nMerah/Hijau/Biru = Asli, Gelap = Setelah Smoothing",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan figure kedua
output_path_2 = os.path.join(OUTPUT_DIR, "12_trajectory_comparison.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Hasil trajectory disimpan di: {output_path_2}")

# Menampilkan statistik
print(f"\n[INFO] Statistik Transformasi:")
print(f"  - Rata-rata translasi X : {transforms[:, 0].mean():.3f} piksel/frame")
print(f"  - Rata-rata translasi Y : {transforms[:, 1].mean():.3f} piksel/frame")
print(f"  - Rata-rata rotasi      : {np.degrees(transforms[:, 2].mean()):.4f} derajat/frame")
print(f"  - Maks koreksi X        : {correction[:, 0].max():.3f} piksel")
print(f"  - Maks koreksi Y        : {correction[:, 1].max():.3f} piksel")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 12")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.goodFeaturesToTrack()      → Deteksi corner untuk tracking")
print("  2. cv2.calcOpticalFlowPyrLK()     → Tracking fitur antar frame")
print("  3. cv2.estimateAffinePartial2D()  → Estimasi transformasi affine")
print("     - Mengembalikan matriks 2x3 (translasi + rotasi + skala)")
print("     - Menggunakan RANSAC untuk robustness")
print("  4. cv2.warpAffine()               → Menerapkan transformasi affine")
print("  5. np.cumsum()                    → Menghitung trajectory kumulatif")
print("Pipeline Stabilisasi:")
print("  Step 1: Hitung transformasi antar frame berurutan")
print("  Step 2: Akumulasi → trajectory, lalu smooth (moving average)")
print("  Step 3: Koreksi = smooth - original, terapkan warpAffine")
print("=" * 60)
