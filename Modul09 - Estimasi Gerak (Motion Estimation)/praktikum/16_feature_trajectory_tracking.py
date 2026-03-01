"""
==========================================================================
PERCOBAAN 16: TRACKING TRAJECTORY FITUR MULTI-FRAME
==========================================================================
Program ini mempelajari cara melacak fitur di banyak frame dan
menggambar trajectory lengkap (jejak perjalanan) setiap fitur.
Warna trajectory dikodekan berdasarkan waktu sehingga dapat dilihat
perjalanan fitur dari awal (biru) hingga akhir (merah).

Pipeline:
1. Deteksi fitur di frame pertama (goodFeaturesToTrack)
2. Tracking fitur frame per frame (calcOpticalFlowPyrLK)
3. Simpan semua posisi untuk setiap fitur
4. Gambar trajectory lengkap dengan warna berdasarkan waktu

Fungsi utama yang dipelajari:
- cv2.goodFeaturesToTrack()       : Mendeteksi fitur corner
- cv2.calcOpticalFlowPyrLK()      : Tracking fitur antar frame
- cv2.polylines()                 : Menggambar polyline trajectory
- cv2.circle()                    : Menggambar posisi fitur
- matplotlib colormap             : Pewarnaan berdasarkan waktu

Hasil: Visualisasi trajectory penuh semua fitur yang di-track
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan video
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file
import os

# Mengimpor matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mengimpor colormap dari matplotlib
from matplotlib import cm

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input dan output
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 16: TRACKING TRAJECTORY FITUR MULTI-FRAME")
print("=" * 60)

# ============================================================
# 1. Membuka video dan membaca frame pertama
# ============================================================

# Menggunakan video bola untuk tracking trajectory
video_path = os.path.join(IMAGE_DIR, "video_bola.avi")
cap = cv2.VideoCapture(video_path)

# Memeriksa apakah video berhasil dibuka
if not cap.isOpened():
    print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Membaca properti video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"[INFO] Video berhasil dibuka: {width}x{height}")
print(f"[INFO] FPS: {fps}, Total frame: {total_frames}")

# Membaca frame pertama
ret, first_frame = cap.read()
if not ret:
    print("[ERROR] Gagal membaca frame pertama.")
    exit()

# Mengkonversi frame pertama ke grayscale
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Mendeteksi fitur di frame pertama
# ============================================================

# Parameter deteksi fitur
feature_params = dict(
    maxCorners=50,        # 50 titik fitur untuk track yang jelas
    qualityLevel=0.3,     # Kualitas minimum 30%
    minDistance=10,        # Jarak minimum 10 piksel antar fitur
    blockSize=7           # Ukuran blok komputasi
)

# Mendeteksi fitur corner pada frame pertama
initial_pts = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)

# Memeriksa apakah fitur ditemukan
if initial_pts is None or len(initial_pts) == 0:
    print("[ERROR] Tidak ada fitur yang terdeteksi.")
    exit()

# Jumlah fitur awal yang terdeteksi
num_features = len(initial_pts)
print(f"[INFO] Jumlah fitur terdeteksi: {num_features}")

# ============================================================
# 3. Inisialisasi penyimpanan trajectory
# ============================================================

# List untuk menyimpan trajectory setiap fitur
# trajectories[i] = list of (x, y) positions for feature i
trajectories = [[] for _ in range(num_features)]

# Menyimpan posisi awal setiap fitur
for i in range(num_features):
    x, y = initial_pts[i].ravel()
    trajectories[i].append((float(x), float(y)))

# Array status aktif (True jika fitur masih bisa di-track)
active = np.ones(num_features, dtype=bool)

# Parameter untuk Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# ============================================================
# 4. Tracking fitur frame per frame
# ============================================================

print("[INFO] Memulai tracking fitur...")

# Variabel untuk frame sebelumnya
prev_gray = first_gray.copy()
prev_pts = initial_pts.copy()

# Counter frame dan penyimpan frame untuk visualisasi
frame_count = 0
last_frame = first_frame.copy()
snapshot_frames = [(0, first_frame.copy())]

while True:
    # Membaca frame berikutnya
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Mengkonversi ke grayscale
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Menyiapkan titik aktif untuk tracking
    active_indices = np.where(active)[0]

    if len(active_indices) == 0:
        print(f"[WARN] Semua fitur hilang di frame {frame_count}.")
        break

    # Mengambil titik yang masih aktif
    active_pts = prev_pts[active_indices]

    # Menghitung optical flow untuk titik aktif
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, active_pts, None, **lk_params
    )

    # Memperbarui trajectory dan status
    for idx, orig_idx in enumerate(active_indices):
        if status[idx] == 1:
            # Fitur berhasil di-track → simpan posisi baru
            x, y = new_pts[idx].ravel()
            trajectories[orig_idx].append((float(x), float(y)))
            # Memperbarui posisi di array pts
            prev_pts[orig_idx] = new_pts[idx]
        else:
            # Fitur hilang → tandai sebagai tidak aktif
            active[orig_idx] = False

    # Menyimpan frame untuk visualisasi pada beberapa titik
    if frame_count in [total_frames // 4, total_frames // 2, total_frames * 3 // 4]:
        snapshot_frames.append((frame_count, frame.copy()))

    # Menyimpan frame terakhir
    last_frame = frame.copy()

    # Memperbarui frame sebelumnya
    prev_gray = curr_gray.copy()

# Menyimpan frame terakhir
snapshot_frames.append((frame_count, last_frame.copy()))

# Menutup video capture
cap.release()

# Menghitung statistik
active_count = np.count_nonzero(active)
print(f"[INFO] Tracking selesai: {frame_count} frame diproses.")
print(f"[INFO] Fitur masih aktif: {active_count}/{num_features}")

# ============================================================
# 5. Menggambar trajectory pada frame terakhir
# ============================================================

print("[INFO] Menggambar trajectory pada frame terakhir...")

# Membuat salinan frame terakhir untuk menggambar trajectory
trajectory_image = last_frame.copy()

# Membuat colormap berdasarkan waktu (biru → merah)
# Menggunakan jet colormap: biru=awal, merah=akhir
cmap = cm.get_cmap('coolwarm')

for i in range(num_features):
    traj = trajectories[i]
    
    # Hanya gambar trajectory yang cukup panjang (minimal 5 titik)
    if len(traj) < 5:
        continue

    # Menggambar setiap segmen trajectory dengan warna berdasarkan waktu
    for j in range(1, len(traj)):
        # Menghitung rasio waktu (0.0 = awal, 1.0 = akhir)
        time_ratio = j / (len(traj) - 1)
        
        # Mendapatkan warna dari colormap (BGR format untuk OpenCV)
        rgba = cmap(time_ratio)
        color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        
        # Menggambar garis dari posisi sebelumnya ke saat ini
        pt1 = (int(traj[j-1][0]), int(traj[j-1][1]))
        pt2 = (int(traj[j][0]), int(traj[j][1]))
        cv2.line(trajectory_image, pt1, pt2, color, 2)

    # Menggambar titik awal (hijau) dan akhir (merah)
    start_pt = (int(traj[0][0]), int(traj[0][1]))
    end_pt = (int(traj[-1][0]), int(traj[-1][1]))
    cv2.circle(trajectory_image, start_pt, 5, (0, 255, 0), -1)   # Hijau = awal
    cv2.circle(trajectory_image, end_pt, 5, (0, 0, 255), -1)     # Merah = akhir

# ============================================================
# 6. Visualisasi trajectory overlay
# ============================================================

# Figure 1: Trajectory pada frame terakhir
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Frame pertama dengan titik fitur awal
frame1_vis = first_frame.copy()
for i in range(num_features):
    pt = (int(trajectories[i][0][0]), int(trajectories[i][0][1]))
    cv2.circle(frame1_vis, pt, 5, (0, 255, 0), -1)
axes1[0].imshow(cv2.cvtColor(frame1_vis, cv2.COLOR_BGR2RGB))
axes1[0].set_title(f"Frame Pertama: {num_features} fitur terdeteksi", fontsize=11)
axes1[0].axis("off")

# Subplot 2: Frame terakhir dengan trajectory
axes1[1].imshow(cv2.cvtColor(trajectory_image, cv2.COLOR_BGR2RGB))
axes1[1].set_title(f"Frame Terakhir: Trajectory {active_count} fitur aktif", fontsize=11)
axes1[1].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 16: Tracking Trajectory Fitur\n"
             "Biru=awal, Merah=akhir; Hijau●=start, Merah●=end",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan figure pertama
output_path_1 = os.path.join(OUTPUT_DIR, "16_feature_trajectory.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Trajectory disimpan di: {output_path_1}")

# ============================================================
# 7. Visualisasi progresif trajectory
# ============================================================

# Figure 2: Trajectory pada beberapa snapshot
num_snap = len(snapshot_frames)
fig2, axes2 = plt.subplots(1, num_snap, figsize=(5 * num_snap, 5))

if num_snap == 1:
    axes2 = [axes2]

for snap_idx, (fnum, snap_frame) in enumerate(snapshot_frames):
    # Membuat salinan frame
    vis = snap_frame.copy()
    
    # Menggambar trajectory hingga frame ini
    for i in range(num_features):
        traj = trajectories[i]
        # Menghitung berapa banyak titik hingga frame ini
        max_pts = min(len(traj), fnum + 1)
        
        if max_pts < 2:
            continue

        for j in range(1, max_pts):
            time_ratio = j / max(max_pts - 1, 1)
            rgba = cmap(time_ratio)
            color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            pt1 = (int(traj[j-1][0]), int(traj[j-1][1]))
            pt2 = (int(traj[j][0]), int(traj[j][1]))
            cv2.line(vis, pt1, pt2, color, 2)

    # Menampilkan pada subplot
    axes2[snap_idx].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes2[snap_idx].set_title(f"Frame #{fnum}", fontsize=10)
    axes2[snap_idx].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 16: Perkembangan Trajectory Seiring Waktu",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan figure kedua
output_path_2 = os.path.join(OUTPUT_DIR, "16_trajectory_progress.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Progres trajectory disimpan di: {output_path_2}")

# ============================================================
# 8. Statistik trajectory
# ============================================================

# Menghitung panjang total setiap trajectory
traj_lengths = []
traj_displacements = []
for i in range(num_features):
    traj = trajectories[i]
    if len(traj) < 2:
        continue
    
    # Panjang total (jumlah jarak antar titik berurutan)
    total_len = 0
    for j in range(1, len(traj)):
        dx = traj[j][0] - traj[j-1][0]
        dy = traj[j][1] - traj[j-1][1]
        total_len += np.sqrt(dx**2 + dy**2)
    traj_lengths.append(total_len)
    
    # Displacement (jarak dari awal ke akhir)
    dx = traj[-1][0] - traj[0][0]
    dy = traj[-1][1] - traj[0][1]
    traj_displacements.append(np.sqrt(dx**2 + dy**2))

print(f"\n[INFO] ============ STATISTIK TRAJECTORY ============")
print(f"  Jumlah trajectory   : {len(traj_lengths)}")
print(f"  Panjang rata-rata   : {np.mean(traj_lengths):.2f} piksel")
print(f"  Panjang terpanjang  : {np.max(traj_lengths):.2f} piksel")
print(f"  Displacement rata-rata: {np.mean(traj_displacements):.2f} piksel")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 16")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.goodFeaturesToTrack()    → Deteksi fitur awal")
print("  2. cv2.calcOpticalFlowPyrLK()   → Track fitur frame ke frame")
print("  3. cv2.line()                   → Gambar segmen trajectory")
print("  4. cv2.polylines()              → Gambar polyline")
print("  5. matplotlib colormap          → Pewarnaan berdasarkan waktu")
print("Konsep Trajectory Tracking:")
print("  - Track fitur di BANYAK frame, bukan hanya 2")
print("  - Simpan semua posisi → gambar trajectory penuh")
print("  - Color-code: biru=awal, merah=akhir (temporal)")
print("  - Fitur bisa hilang (di-occlude/keluar frame)")
print("  - Panjang vs displacement menunjukkan linearitas gerakan")
print("=" * 60)
