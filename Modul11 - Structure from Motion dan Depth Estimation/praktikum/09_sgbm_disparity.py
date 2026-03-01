"""
==========================================================================
PERCOBAAN 9: SEMI-GLOBAL BLOCK MATCHING (SGBM)
==========================================================================
Program ini mempelajari algoritma StereoSGBM (Semi-Global Block Matching)
yang lebih akurat dari StereoBM untuk menghitung disparity map.
SGBM mengoptimasi biaya matching secara semi-global dari 8 arah,
menghasilkan disparity yang lebih halus dan konsisten.

Konsep utama:
- SGBM meminimalkan cost function secara semi-global (8 atau 16 arah)
- Parameter P1 mengontrol penalti perubahan disparity kecil (halus)
- Parameter P2 mengontrol penalti perubahan disparity besar (diskontinuitas)
- Mode SGBM, HH, SGBM_3WAY mempengaruhi akurasi dan kecepatan
- WLS Filter (jika tersedia) menghaluskan disparity dengan edge-aware

Fungsi utama yang dipelajari:
- cv2.StereoSGBM_create()              : Membuat objek StereoSGBM
- cv2.StereoBM_create()                : Pembanding dengan BM standar
- stereo.compute()                      : Menghitung disparity map
- cv2.ximgproc.createDisparityWLSFilter : WLS filter (jika tersedia)
- cv2.normalize()                       : Normalisasi untuk visualisasi
- cv2.applyColorMap()                   : Menerapkan colormap pada disparity

Hasil: Perbandingan BM vs SGBM, variasi parameter, dan disparity terfilter
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
print("PERCOBAAN 9: SEMI-GLOBAL BLOCK MATCHING (SGBM)")
print("=" * 60)

# ============================================================
# 1. Memuat atau membuat pasangan gambar stereo sintetis
# ============================================================

# Menampilkan informasi tahap memuat gambar
print("\n--- Memuat Pasangan Gambar Stereo ---")

# Mendefinisikan path gambar kiri
path_left = os.path.join(IMAGE_DIR, "stereo_left.png")

# Mendefinisikan path gambar kanan
path_right = os.path.join(IMAGE_DIR, "stereo_right.png")

# Membaca gambar kiri dalam format BGR
img_left = cv2.imread(path_left)

# Membaca gambar kanan dalam format BGR
img_right = cv2.imread(path_right)

# Memeriksa apakah gambar berhasil dimuat, jika tidak download otomatis
if img_left is None or img_right is None:
    print("[WARN] Gambar stereo tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    img_left  = cv2.imread(path_left)
    img_right = cv2.imread(path_right)
if img_left is None:
    raise FileNotFoundError(
        "[ERROR] stereo_left.png tidak tersedia setelah download.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )
if img_right is None:
    raise FileNotFoundError(
        "[ERROR] stereo_right.png tidak tersedia setelah download.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )
else:
    # Menampilkan informasi dimensi gambar
    print(f"[INFO] Ukuran gambar kiri : {img_left.shape}")
    print(f"[INFO] Ukuran gambar kanan: {img_right.shape}")

# ============================================================
# 2. Konversi ke grayscale
# ============================================================

# Menampilkan informasi tahap konversi
print("\n--- Konversi ke Grayscale ---")

# Mengkonversi gambar kiri ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kanan ke grayscale
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Menampilkan informasi hasil konversi
print(f"[INFO] Grayscale kiri : {gray_left.shape}")
print(f"[INFO] Grayscale kanan: {gray_right.shape}")

# Mendapatkan ukuran gambar
h, w = gray_left.shape

# ============================================================
# 3. Perbandingan BM vs SGBM
# ============================================================

# Menampilkan informasi tahap perbandingan
print("\n--- Perbandingan BM vs SGBM ---")

# Mendefinisikan parameter umum
num_disp = 64
block_size = 9

# --- StereoBM ---
# Mencatat waktu mulai untuk BM
t_start_bm = time.time()

# Membuat objek StereoBM
stereo_bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)

# Menghitung disparity map menggunakan BM
disp_bm = stereo_bm.compute(gray_left, gray_right)

# Mencatat waktu selesai BM
t_bm = time.time() - t_start_bm

# Mengkonversi disparity BM ke float
disp_bm_float = disp_bm.astype(np.float32) / 16.0

# --- StereoSGBM ---
# Mendefinisikan parameter P1 dan P2 untuk SGBM
P1 = 8 * 3 * block_size ** 2
P2 = 32 * 3 * block_size ** 2

# Mencatat waktu mulai untuk SGBM
t_start_sgbm = time.time()

# Membuat objek StereoSGBM dengan parameter standar
stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=P1,
    P2=P2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Menghitung disparity map menggunakan SGBM
disp_sgbm = stereo_sgbm.compute(gray_left, gray_right)

# Mencatat waktu selesai SGBM
t_sgbm = time.time() - t_start_sgbm

# Mengkonversi disparity SGBM ke float
disp_sgbm_float = disp_sgbm.astype(np.float32) / 16.0

# Menghitung statistik BM
valid_bm = disp_bm_float > 0
bm_mean = disp_bm_float[valid_bm].mean() if valid_bm.any() else 0
bm_coverage = valid_bm.sum() / (h * w) * 100

# Menghitung statistik SGBM
valid_sgbm = disp_sgbm_float > 0
sgbm_mean = disp_sgbm_float[valid_sgbm].mean() if valid_sgbm.any() else 0
sgbm_coverage = valid_sgbm.sum() / (h * w) * 100

# Menampilkan perbandingan
print(f"  BM   : waktu={t_bm:.4f}s, mean={bm_mean:.2f}, coverage={bm_coverage:.1f}%")
print(f"  SGBM : waktu={t_sgbm:.4f}s, mean={sgbm_mean:.2f}, coverage={sgbm_coverage:.1f}%")

# ============================================================
# 4. Variasi parameter P1 dan P2
# ============================================================

# Menampilkan informasi tahap variasi P1 P2
print("\n--- Variasi Parameter P1 dan P2 ---")

# Mendefinisikan kombinasi P1 dan P2 yang akan diuji
p_combinations = [
    (8 * 3 * block_size**2, 32 * 3 * block_size**2, "Standard"),
    (2 * 3 * block_size**2, 8 * 3 * block_size**2, "Low Penalty"),
    (16 * 3 * block_size**2, 64 * 3 * block_size**2, "High Penalty"),
    (8 * 1 * block_size**2, 32 * 1 * block_size**2, "1-Channel"),
]

# Menyiapkan list untuk menyimpan hasil variasi P1 P2
p_results = []

# Iterasi setiap kombinasi P1 P2
for p1_val, p2_val, label in p_combinations:
    # Membuat objek SGBM dengan P1 P2 tertentu
    stereo_p = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=p1_val,
        P2=p2_val,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # Menghitung disparity map
    disp_p = stereo_p.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Menghitung statistik
    valid_p = disp_p > 0
    mean_p = disp_p[valid_p].mean() if valid_p.any() else 0

    # Menampilkan informasi
    print(f"  {label:15s}: P1={p1_val:6d}, P2={p2_val:6d}, mean={mean_p:.2f}")

    # Menormalisasi untuk visualisasi
    disp_p_vis = cv2.normalize(
        (disp_p * 16).astype(np.int16), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Menerapkan colormap JET
    disp_p_color = cv2.applyColorMap(disp_p_vis, cv2.COLORMAP_JET)

    # Menyimpan ke list
    p_results.append((label, disp_p_color))

# ============================================================
# 5. Variasi mode SGBM
# ============================================================

# Menampilkan informasi tahap variasi mode
print("\n--- Variasi Mode SGBM ---")

# Mendefinisikan mode SGBM yang akan diuji
sgbm_modes = [
    (cv2.STEREO_SGBM_MODE_SGBM, "MODE_SGBM (5 arah)"),
    (cv2.STEREO_SGBM_MODE_HH, "MODE_HH (8 arah full)"),
    (cv2.STEREO_SGBM_MODE_SGBM_3WAY, "MODE_3WAY (3-pass)"),
]

# Menyiapkan list untuk menyimpan hasil mode
mode_results = []

# Iterasi setiap mode
for mode_val, mode_label in sgbm_modes:
    # Mencatat waktu mulai
    t_start = time.time()

    # Membuat objek SGBM dengan mode tertentu
    stereo_mode = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=mode_val
    )

    # Menghitung disparity map
    disp_mode = stereo_mode.compute(gray_left, gray_right)

    # Mencatat waktu selesai
    t_mode = time.time() - t_start

    # Mengkonversi ke float
    disp_mode_float = disp_mode.astype(np.float32) / 16.0

    # Menghitung statistik
    valid_mode = disp_mode_float > 0
    mean_mode = disp_mode_float[valid_mode].mean() if valid_mode.any() else 0
    coverage_mode = valid_mode.sum() / (h * w) * 100

    # Menampilkan informasi
    print(f"  {mode_label:25s}: waktu={t_mode:.4f}s, "
          f"mean={mean_mode:.2f}, coverage={coverage_mode:.1f}%")

    # Menormalisasi untuk visualisasi
    disp_mode_vis = cv2.normalize(disp_mode, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Menerapkan colormap JET
    disp_mode_color = cv2.applyColorMap(disp_mode_vis, cv2.COLORMAP_JET)

    # Menyimpan ke list
    mode_results.append((mode_label, disp_mode_color))

# ============================================================
# 6. Aplikasi WLS Filter (jika tersedia)
# ============================================================

# Menampilkan informasi tahap WLS filter
print("\n--- WLS Filter (Weighted Least Squares) ---")

# Mencoba menggunakan WLS filter dari modul ximgproc
try:
    # Membuat matcher kanan untuk WLS filter
    right_matcher = cv2.ximgproc.createRightMatcher(stereo_sgbm)

    # Menghitung disparity kanan
    disp_right = right_matcher.compute(gray_right, gray_left)

    # Membuat WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_sgbm)

    # Mengatur parameter WLS filter
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)

    # Menerapkan WLS filter
    disp_wls = wls_filter.filter(disp_sgbm, img_left, None, disp_right)

    # Mengkonversi ke float
    disp_wls_float = disp_wls.astype(np.float32) / 16.0

    # Menampilkan bahwa WLS berhasil
    print("[INFO] WLS Filter berhasil diterapkan (cv2.ximgproc tersedia)")

    # Menormalisasi untuk visualisasi
    disp_wls_vis = cv2.normalize(disp_wls, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Menerapkan colormap JET
    disp_wls_color = cv2.applyColorMap(disp_wls_vis, cv2.COLORMAP_JET)

    # Menandai bahwa WLS tersedia
    wls_available = True

except AttributeError:
    # Menampilkan pesan fallback jika ximgproc tidak tersedia
    print("[INFO] cv2.ximgproc tidak tersedia, melewati WLS filter")
    print("[INFO] Install opencv-contrib-python untuk fitur WLS filter")

    # Menandai bahwa WLS tidak tersedia
    wls_available = False

    # Menggunakan median blur sebagai alternatif sederhana
    disp_sgbm_vis_tmp = cv2.normalize(disp_sgbm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_wls_vis = cv2.medianBlur(disp_sgbm_vis_tmp, 5)
    disp_wls_color = cv2.applyColorMap(disp_wls_vis, cv2.COLORMAP_JET)
    print("[INFO] Menggunakan Median Blur sebagai alternatif sederhana")

# ============================================================
# 7. Konversi SGBM disparity ke depth
# ============================================================

# Menampilkan informasi tahap konversi depth
print("\n--- Konversi Disparity SGBM ke Depth ---")

# Mendefinisikan parameter kamera
focal_length = w * 1.0
baseline = 30.0

# Menampilkan parameter
print(f"[INFO] Focal length: {focal_length:.1f}")
print(f"[INFO] Baseline: {baseline:.1f}")

# Menyalin disparity SGBM
disp_depth = disp_sgbm_float.copy()

# Mengganti nilai <= 0 untuk menghindari division by zero
disp_depth[disp_depth <= 0] = 0.1

# Menghitung depth map: Z = f * B / d
depth_sgbm = (focal_length * baseline) / disp_depth

# Membatasi depth maksimum
depth_sgbm[depth_sgbm > 5000] = 5000

# Menampilkan statistik depth
valid_depth = depth_sgbm[disp_sgbm_float > 0]
if len(valid_depth) > 0:
    print(f"[DEPTH] Min: {valid_depth.min():.2f}")
    print(f"[DEPTH] Max: {valid_depth.max():.2f}")
    print(f"[DEPTH] Mean: {valid_depth.mean():.2f}")

# ============================================================
# 8. Tabel perbandingan
# ============================================================

# Menampilkan tabel perbandingan
print("\n--- Tabel Perbandingan BM vs SGBM ---")
print(f"{'Metode':<20} {'Waktu (s)':<12} {'Mean Disp':<12} {'Coverage %':<12}")
print("-" * 56)
print(f"{'StereoBM':<20} {t_bm:<12.4f} {bm_mean:<12.2f} {bm_coverage:<12.1f}")
print(f"{'StereoSGBM':<20} {t_sgbm:<12.4f} {sgbm_mean:<12.2f} {sgbm_coverage:<12.1f}")

# ============================================================
# 9. Visualisasi BM vs SGBM
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi BM vs SGBM ---")

# Menormalisasi disparity BM untuk visualisasi
disp_bm_vis = cv2.normalize(disp_bm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disp_bm_color = cv2.applyColorMap(disp_bm_vis, cv2.COLORMAP_JET)

# Menormalisasi disparity SGBM untuk visualisasi
disp_sgbm_vis = cv2.normalize(disp_sgbm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disp_sgbm_color = cv2.applyColorMap(disp_sgbm_vis, cv2.COLORMAP_JET)

# Membuat figure 2x2 untuk perbandingan utama
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

# Menampilkan gambar kiri original
axes1[0, 0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
axes1[0, 0].set_title("Gambar Kiri (Input)", fontsize=12)
axes1[0, 0].axis("off")

# Menampilkan gambar kanan original
axes1[0, 1].imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
axes1[0, 1].set_title("Gambar Kanan (Input)", fontsize=12)
axes1[0, 1].axis("off")

# Menampilkan disparity BM
axes1[1, 0].imshow(cv2.cvtColor(disp_bm_color, cv2.COLOR_BGR2RGB))
axes1[1, 0].set_title(f"StereoBM (waktu={t_bm:.4f}s)", fontsize=12)
axes1[1, 0].axis("off")

# Menampilkan disparity SGBM
axes1[1, 1].imshow(cv2.cvtColor(disp_sgbm_color, cv2.COLOR_BGR2RGB))
axes1[1, 1].set_title(f"StereoSGBM (waktu={t_sgbm:.4f}s)", fontsize=12)
axes1[1, 1].axis("off")

# Mengatur judul utama
plt.suptitle("Percobaan 9: Perbandingan BM vs SGBM",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure perbandingan BM vs SGBM
output_bm_sgbm = os.path.join(OUTPUT_DIR, "09_bm_vs_sgbm.png")
plt.savefig(output_bm_sgbm, dpi=150, bbox_inches='tight')
print(f"[SAVED] BM vs SGBM: {output_bm_sgbm}")

# ============================================================
# 10. Visualisasi variasi P1 P2
# ============================================================

# Menampilkan informasi tahap visualisasi P1 P2
print("\n--- Membuat Visualisasi Variasi P1 P2 ---")

# Membuat figure 2x2 untuk variasi P1 P2
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2_flat = axes2.flatten()

# Menampilkan setiap variasi P1 P2
for i, (label, disp_color) in enumerate(p_results):
    axes2_flat[i].imshow(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB))
    axes2_flat[i].set_title(f"P1/P2: {label}", fontsize=12)
    axes2_flat[i].axis("off")

# Mengatur judul utama
plt.suptitle("Percobaan 9: Variasi Parameter P1 dan P2 (SGBM)",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure variasi P1 P2
output_p1p2 = os.path.join(OUTPUT_DIR, "09_variasi_p1_p2.png")
plt.savefig(output_p1p2, dpi=150, bbox_inches='tight')
print(f"[SAVED] Variasi P1 P2: {output_p1p2}")

# ============================================================
# 11. Visualisasi variasi mode dan WLS
# ============================================================

# Menampilkan informasi tahap visualisasi mode
print("\n--- Membuat Visualisasi Mode SGBM dan WLS ---")

# Membuat figure 2x2 untuk mode dan WLS
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

# Menampilkan setiap mode SGBM
for i, (mode_label, mode_color) in enumerate(mode_results):
    axes3_flat = axes3.flatten()
    axes3_flat[i].imshow(cv2.cvtColor(mode_color, cv2.COLOR_BGR2RGB))
    axes3_flat[i].set_title(mode_label, fontsize=11)
    axes3_flat[i].axis("off")

# Menampilkan hasil WLS filter atau alternatif pada subplot keempat
wls_title = "WLS Filter" if wls_available else "Median Blur (alternatif)"
axes3.flatten()[3].imshow(cv2.cvtColor(disp_wls_color, cv2.COLOR_BGR2RGB))
axes3.flatten()[3].set_title(wls_title, fontsize=11)
axes3.flatten()[3].axis("off")

# Mengatur judul utama
plt.suptitle("Percobaan 9: Mode SGBM dan Post-Processing Filter",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure mode dan WLS
output_modes = os.path.join(OUTPUT_DIR, "09_mode_sgbm_wls.png")
plt.savefig(output_modes, dpi=150, bbox_inches='tight')
print(f"[SAVED] Mode & WLS: {output_modes}")

# Menampilkan semua figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 9: SEMI-GLOBAL BLOCK MATCHING (SGBM)")
print("=" * 60)
print(f"1. StereoSGBM mengoptimasi cost semi-global dari 8 arah")
print(f"2. SGBM menghasilkan disparity lebih halus dari BM")
print(f"3. Parameter P1: penalti perubahan kecil (smoothness)")
print(f"4. Parameter P2: penalti perubahan besar (diskontinuitas)")
print(f"5. Rekomendasi: P1 = 8*channels*blockSize^2")
print(f"6.              P2 = 32*channels*blockSize^2")
print(f"7. Mode HH (full 8-pass) paling akurat tapi lambat")
print(f"8. Mode SGBM_3WAY kompromi baik antara akurasi dan kecepatan")
print(f"9. WLS Filter menghaluskan disparity edge-aware (butuh ximgproc)")
print(f"10. SGBM cocok untuk aplikasi yang butuh akurasi tinggi")
print("=" * 60)
