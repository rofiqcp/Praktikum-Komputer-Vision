"""
==========================================================================
PERCOBAAN 16: SIMULASI STEREO MATCHING REAL-TIME
==========================================================================
Program ini mempelajari cara melakukan stereo matching secara real-time
pada video sintetis dengan objek bergerak. Performa StereoBM dan
StereoSGBM dibandingkan dalam hal FPS (frame per second) dan kualitas
disparity map pada setiap frame.

Konsep utama:
- Stereo matching real-time memerlukan keseimbangan akurasi vs kecepatan
- StereoBM lebih cepat namun kurang akurat dibanding StereoSGBM
- FPS (Frames Per Second) mengukur kecepatan pemrosesan
- Parameter numDisparities dan blockSize mempengaruhi akurasi dan kecepatan
- Video sintetis memungkinkan evaluasi tanpa hardware stereo kamera

Fungsi utama yang dipelajari:
- cv2.StereoBM_create()       : Membuat matcher Block Matching
- cv2.StereoSGBM_create()     : Membuat matcher Semi-Global BM
- cv2.VideoWriter()            : Menulis video output
- cv2.normalize()              : Normalisasi disparity untuk visualisasi
- cv2.applyColorMap()          : Menerapkan colormap pada disparity
- time.time()                  : Mengukur waktu komputasi per frame

Hasil: Perbandingan FPS dan kualitas stereo matching BM vs SGBM pada
       video sintetis, statistik performa, dan sample frame output
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk membuat dan menyimpan visualisasi
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu komputasi
import time

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan judul percobaan
print("=" * 60)
print("PERCOBAAN 16: SIMULASI STEREO MATCHING REAL-TIME")
print("=" * 60)

# ============================================================
# 1. Mendefinisikan Parameter Simulasi
# ============================================================

# Menampilkan informasi tahap parameter
print("\n--- Parameter Simulasi ---")

# Mendefinisikan ukuran frame video
frame_h, frame_w = 240, 320

# Mendefinisikan jumlah total frame yang akan digenerate
total_frames = 30

# Mendefinisikan parameter stereo
num_disp = 48

# Mendefinisikan ukuran blok matching
block_size_bm = 15
block_size_sgbm = 5

# Menampilkan parameter simulasi
print(f"[INFO] Ukuran frame: {frame_w}x{frame_h}")
print(f"[INFO] Total frame: {total_frames}")
print(f"[INFO] numDisparities: {num_disp}")
print(f"[INFO] Block size BM: {block_size_bm}")
print(f"[INFO] Block size SGBM: {block_size_sgbm}")


# ============================================================
# 2. Memuat Gambar Stereo Real untuk Simulasi Real-Time
# ============================================================

# Menampilkan informasi tahap memuat gambar
print("\n--- Memuat Gambar Stereo Real ---")

# Mendefinisikan path gambar stereo
path_left  = os.path.join(IMAGE_DIR, "stereo_left.png")
path_right = os.path.join(IMAGE_DIR, "stereo_right.png")

# Memuat dan resize gambar stereo ke ukuran frame simulasi
_img_l = cv2.imread(path_left)
_img_r = cv2.imread(path_right)

# Download otomatis jika file tidak tersedia
if _img_l is None or _img_r is None:
    print("[WARN] Gambar stereo tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    _img_l = cv2.imread(path_left)
    _img_r = cv2.imread(path_right)
if _img_l is None:
    raise FileNotFoundError("[ERROR] stereo_left.png tidak tersedia. Jalankan: python download_image.py")
if _img_r is None:
    raise FileNotFoundError("[ERROR] stereo_right.png tidak tersedia. Jalankan: python download_image.py")

# Resize ke ukuran frame simulasi
img_left_real  = cv2.resize(_img_l, (frame_w, frame_h))
img_right_real = cv2.resize(_img_r, (frame_w, frame_h))

print(f"[INFO] Gambar stereo dimuat dan di-resize ke {frame_w}x{frame_h}")
print("[INFO] Generator frame real-time siap")

# ============================================================
# 3. Membuat Stereo Matchers (BM dan SGBM)
# ============================================================

# Menampilkan informasi tahap pembuatan matcher
print("\n--- Membuat Stereo Matchers ---")

# Membuat objek StereoBM
stereo_bm = cv2.StereoBM_create(
    numDisparities=num_disp,
    blockSize=block_size_bm
)

# Menampilkan informasi StereoBM
print(f"[BM] numDisparities={num_disp}, blockSize={block_size_bm}")

# Menghitung parameter P1 dan P2 untuk SGBM
P1 = 8 * 3 * block_size_sgbm ** 2
P2 = 32 * 3 * block_size_sgbm ** 2

# Membuat objek StereoSGBM
stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size_sgbm,
    P1=P1,
    P2=P2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=16,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Menampilkan informasi StereoSGBM
print(f"[SGBM] numDisparities={num_disp}, blockSize={block_size_sgbm}")
print(f"[SGBM] P1={P1}, P2={P2}")

# ============================================================
# 4. Memproses Frame dan Mengukur FPS
# ============================================================

# Menampilkan informasi tahap pemrosesan
print("\n--- Memproses Frame Stereo ---")

# Menginisialisasi list untuk menyimpan FPS setiap frame
fps_bm_list = []
fps_sgbm_list = []

# Menginisialisasi list untuk menyimpan coverage per frame
coverage_bm_list = []
coverage_sgbm_list = []

# Menginisialisasi list untuk menyimpan sample frame
sample_frames = []
sample_indices = [0, total_frames // 4, total_frames // 2, total_frames - 1]

# Menampilkan header tabel progres
print(f"\n{'Frame':<8} {'FPS BM':<12} {'FPS SGBM':<12} {'Cov BM(%)':<12} {'Cov SGBM(%)':<12}")
print("-" * 56)

# Memproses setiap frame
for frame_idx in range(total_frames):
    # Menggunakan gambar stereo real yang telah dimuat
    img_left, img_right = img_left_real.copy(), img_right_real.copy()

    # Mengkonversi gambar ke grayscale untuk stereo matching
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # --- Stereo BM ---
    # Mencatat waktu mulai BM
    t_start_bm = time.time()

    # Menghitung disparity map menggunakan StereoBM
    disp_bm = stereo_bm.compute(gray_left, gray_right)

    # Mencatat waktu selesai BM
    t_bm = time.time() - t_start_bm

    # Menghitung FPS untuk BM
    fps_bm = 1.0 / max(t_bm, 1e-6)
    fps_bm_list.append(fps_bm)

    # Mengkonversi disparity BM ke float
    disp_bm_float = disp_bm.astype(np.float32) / 16.0

    # Menghitung coverage BM
    cov_bm = np.sum(disp_bm_float > 0) / (frame_h * frame_w) * 100
    coverage_bm_list.append(cov_bm)

    # --- Stereo SGBM ---
    # Mencatat waktu mulai SGBM
    t_start_sgbm = time.time()

    # Menghitung disparity map menggunakan StereoSGBM
    disp_sgbm = stereo_sgbm.compute(gray_left, gray_right)

    # Mencatat waktu selesai SGBM
    t_sgbm = time.time() - t_start_sgbm

    # Menghitung FPS untuk SGBM
    fps_sgbm = 1.0 / max(t_sgbm, 1e-6)
    fps_sgbm_list.append(fps_sgbm)

    # Mengkonversi disparity SGBM ke float
    disp_sgbm_float = disp_sgbm.astype(np.float32) / 16.0

    # Menghitung coverage SGBM
    cov_sgbm = np.sum(disp_sgbm_float > 0) / (frame_h * frame_w) * 100
    coverage_sgbm_list.append(cov_sgbm)

    # Menampilkan progres setiap 5 frame
    if frame_idx % 5 == 0 or frame_idx == total_frames - 1:
        print(f"{frame_idx:<8} {fps_bm:<12.1f} {fps_sgbm:<12.1f} "
              f"{cov_bm:<12.1f} {cov_sgbm:<12.1f}")

    # Menyimpan sample frame untuk visualisasi
    if frame_idx in sample_indices:
        # Menormalisasi disparity BM untuk visualisasi
        disp_bm_vis = cv2.normalize(disp_bm_float, None, 0, 255, cv2.NORM_MINMAX)
        disp_bm_vis = disp_bm_vis.astype(np.uint8)
        disp_bm_color = cv2.applyColorMap(disp_bm_vis, cv2.COLORMAP_JET)

        # Menormalisasi disparity SGBM untuk visualisasi
        disp_sgbm_vis = cv2.normalize(disp_sgbm_float, None, 0, 255, cv2.NORM_MINMAX)
        disp_sgbm_vis = disp_sgbm_vis.astype(np.uint8)
        disp_sgbm_color = cv2.applyColorMap(disp_sgbm_vis, cv2.COLORMAP_JET)

        # Menyimpan data sample frame
        sample_frames.append({
            "idx": frame_idx,
            "left": img_left.copy(),
            "disp_bm": disp_bm_color.copy(),
            "disp_sgbm": disp_sgbm_color.copy(),
        })

# ============================================================
# 5. Statistik Performa
# ============================================================

# Menampilkan informasi tahap statistik
print("\n--- Statistik Performa ---")

# Menghitung rata-rata FPS untuk BM
avg_fps_bm = np.mean(fps_bm_list)

# Menghitung rata-rata FPS untuk SGBM
avg_fps_sgbm = np.mean(fps_sgbm_list)

# Menghitung rata-rata coverage untuk BM
avg_cov_bm = np.mean(coverage_bm_list)

# Menghitung rata-rata coverage untuk SGBM
avg_cov_sgbm = np.mean(coverage_sgbm_list)

# Menghitung speedup BM dibanding SGBM
speedup = avg_fps_bm / max(avg_fps_sgbm, 1e-6)

# Menampilkan statistik
print(f"\n[StereoBM]")
print(f"  FPS rata-rata: {avg_fps_bm:.1f}")
print(f"  FPS min/max  : {min(fps_bm_list):.1f} / {max(fps_bm_list):.1f}")
print(f"  Coverage avg : {avg_cov_bm:.1f}%")

# Menampilkan statistik SGBM
print(f"\n[StereoSGBM]")
print(f"  FPS rata-rata: {avg_fps_sgbm:.1f}")
print(f"  FPS min/max  : {min(fps_sgbm_list):.1f} / {max(fps_sgbm_list):.1f}")
print(f"  Coverage avg : {avg_cov_sgbm:.1f}%")

# Menampilkan perbandingan kecepatan
print(f"\n[PERBANDINGAN]")
print(f"  BM {speedup:.1f}x lebih cepat dari SGBM")
print(f"  Coverage SGBM {avg_cov_sgbm - avg_cov_bm:+.1f}% dibanding BM")

# ============================================================
# 6. Visualisasi Sample Frame
# ============================================================

# Menampilkan informasi tahap visualisasi sample
print("\n--- Menyimpan Sample Frame ---")

# Membuat figure untuk sample frame
n_samples = len(sample_frames)
fig1, axes = plt.subplots(n_samples, 3, figsize=(14, 4 * n_samples))

# Menampilkan setiap sample frame
for i, sample in enumerate(sample_frames):
    # Menampilkan gambar kiri asli
    axes[i, 0].imshow(cv2.cvtColor(sample["left"], cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"Frame {sample['idx']} - Gambar Kiri", fontsize=9)
    axes[i, 0].axis('off')

    # Menampilkan disparity BM
    axes[i, 1].imshow(cv2.cvtColor(sample["disp_bm"], cv2.COLOR_BGR2RGB))
    axes[i, 1].set_title(f"Frame {sample['idx']} - StereoBM", fontsize=9)
    axes[i, 1].axis('off')

    # Menampilkan disparity SGBM
    axes[i, 2].imshow(cv2.cvtColor(sample["disp_sgbm"], cv2.COLOR_BGR2RGB))
    axes[i, 2].set_title(f"Frame {sample['idx']} - StereoSGBM", fontsize=9)
    axes[i, 2].axis('off')

# Mengatur judul utama
plt.suptitle("Sample Frame: Stereo Matching Real-Time", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output sample frame
path_samples = os.path.join(OUTPUT_DIR, "16_sample_frames_stereo.png")

# Menyimpan figure sample frame
fig1.savefig(path_samples, dpi=150, bbox_inches='tight')
print(f"[SAVE] Sample frames: {path_samples}")

# ============================================================
# 7. Grafik Performa FPS dan Coverage
# ============================================================

# Menampilkan informasi tahap grafik performa
print("\n--- Membuat Grafik Performa ---")

# Membuat figure untuk grafik performa
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Mendefinisikan array nomor frame
frame_numbers = np.arange(total_frames)

# Menggambar grafik FPS per frame
ax1.plot(frame_numbers, fps_bm_list, 'b-o', markersize=3, linewidth=1.5, label="StereoBM")
ax1.plot(frame_numbers, fps_sgbm_list, 'r-s', markersize=3, linewidth=1.5, label="StereoSGBM")
ax1.set_xlabel("Nomor Frame")
ax1.set_ylabel("FPS")
ax1.set_title("FPS per Frame", fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Menggambar grafik coverage per frame
ax2.plot(frame_numbers, coverage_bm_list, 'b-o', markersize=3, linewidth=1.5, label="StereoBM")
ax2.plot(frame_numbers, coverage_sgbm_list, 'r-s', markersize=3, linewidth=1.5, label="StereoSGBM")
ax2.set_xlabel("Nomor Frame")
ax2.set_ylabel("Coverage (%)")
ax2.set_title("Coverage per Frame", fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Menggambar bar chart rata-rata FPS
methods = ["StereoBM", "StereoSGBM"]
avg_fps_values = [avg_fps_bm, avg_fps_sgbm]
bar_colors = ['#3498db', '#e74c3c']

# Menggambar bar chart FPS rata-rata
bars1 = ax3.bar(methods, avg_fps_values, color=bar_colors, edgecolor='black', width=0.5)
ax3.set_title("Rata-rata FPS", fontsize=11, fontweight='bold')
ax3.set_ylabel("FPS")

# Menambahkan nilai di atas bar
for bar, val in zip(bars1, avg_fps_values):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{val:.1f}", ha='center', fontsize=10, fontweight='bold')

# Menggambar bar chart rata-rata coverage
avg_cov_values = [avg_cov_bm, avg_cov_sgbm]

# Menggambar bar chart coverage rata-rata
bars2 = ax4.bar(methods, avg_cov_values, color=bar_colors, edgecolor='black', width=0.5)
ax4.set_title("Rata-rata Coverage", fontsize=11, fontweight='bold')
ax4.set_ylabel("Coverage (%)")

# Menambahkan nilai di atas bar
for bar, val in zip(bars2, avg_cov_values):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha='center', fontsize=10, fontweight='bold')

# Mengatur judul utama
plt.suptitle("Performa Stereo Matching Real-Time: BM vs SGBM",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output grafik performa
path_performance = os.path.join(OUTPUT_DIR, "16_performance_stereo_realtime.png")

# Menyimpan figure performa
fig2.savefig(path_performance, dpi=150, bbox_inches='tight')
print(f"[SAVE] Grafik performa: {path_performance}")

# Menampilkan semua figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 16: SIMULASI STEREO MATCHING REAL-TIME")
print("=" * 60)
print(f"1. Total frame diproses: {total_frames}")
print(f"2. StereoBM rata-rata FPS: {avg_fps_bm:.1f}")
print(f"3. StereoSGBM rata-rata FPS: {avg_fps_sgbm:.1f}")
print(f"4. BM {speedup:.1f}x lebih cepat dari SGBM")
print(f"5. Coverage BM: {avg_cov_bm:.1f}%, SGBM: {avg_cov_sgbm:.1f}%")
print("6. BM cocok untuk aplikasi real-time (FPS tinggi)")
print("7. SGBM memberikan disparity lebih akurat tapi lebih lambat")
print("8. Objek bergerak menyebabkan variasi coverage antar frame")
print("9. Parameter blockSize dan numDisparities mempengaruhi trade-off")
print("10. Pemilihan metode bergantung pada kebutuhan aplikasi")
print("=" * 60)
