"""
==========================================================================
PERCOBAAN 18: PENGARUH BASELINE TERHADAP AKURASI DEPTH
==========================================================================
Program ini mempelajari bagaimana jarak antar kamera (baseline) 
mempengaruhi akurasi dan resolusi depth estimation. Pasangan stereo
sintetis dibuat dengan berbagai baseline, kemudian error depth
dibandingkan terhadap ground truth yang diketahui.

Konsep utama:
- Baseline adalah jarak horizontal antara dua kamera stereo
- Baseline kecil → disparity kecil → depth resolusi rendah
- Baseline besar → disparity besar → depth resolusi tinggi
- Namun baseline terlalu besar menyebabkan area occlusion lebih banyak
- Hubungan depth dan disparity: Z = f * B / d
- MAE (Mean Absolute Error) dan RMSE mengukur akurasi depth

Fungsi utama yang dipelajari:
- cv2.StereoSGBM_create()  : Menghitung disparity map
- cv2.warpAffine()          : Menggeser gambar untuk simulasi baseline
- np.abs()                  : Menghitung error absolut
- np.mean()                 : Menghitung rata-rata error
- np.sqrt()                 : Menghitung akar kuadrat untuk RMSE

Hasil: Grafik error vs baseline, perbandingan depth map untuk berbagai
       baseline, dan tabel analisis akurasi near vs far object
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
print("PERCOBAAN 18: PENGARUH BASELINE TERHADAP AKURASI DEPTH")
print("=" * 60)

# ============================================================
# 1. Memuat gambar dasar stereo real
# ============================================================

# Menampilkan informasi tahap memuat gambar
print("\n--- Memuat Gambar Stereo Real ---")

# Mendefinisikan path gambar stereo kiri sebagai gambar dasar
path_base = os.path.join(IMAGE_DIR, "stereo_left.png")

# Memuat gambar
img_base = cv2.imread(path_base)

# Download otomatis jika file tidak tersedia
if img_base is None:
    print("[WARN] stereo_left.png tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    img_base = cv2.imread(path_base)
if img_base is None:
    raise FileNotFoundError("[ERROR] stereo_left.png tidak tersedia. Jalankan: python download_image.py")

# Mendapatkan dimensi gambar
h, w = img_base.shape[:2]

# Parameter kamera estimasi
focal_length = 700.0

# Membuat estimasi gt_depth dari gradien luminansi gambar
# (area terang/edges = dekat, area gelap/uniform = jauh)
_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
_lap = np.abs(cv2.Laplacian(_gray.astype(np.float32), cv2.CV_32F))
_lap_blur = cv2.GaussianBlur(_lap, (51, 51), 0)
_lap_max = _lap_blur.max() if _lap_blur.max() > 0 else 1
gt_depth = (2500.0 - (_lap_blur / _lap_max) * 2300.0).astype(np.float32)

# Menampilkan informasi gambar
print(f"[INFO] Gambar dasar dimuat: {img_base.shape}")
print(f"[INFO] Focal length (estimasi): {focal_length:.1f} px")
print(f"[INFO] Rentang gt_depth estimasi: [{gt_depth.min():.0f}, {gt_depth.max():.0f}]")

# ============================================================
# 2. Membuat pasangan stereo dengan berbagai baseline
# ============================================================

# Menampilkan informasi tahap variasi baseline
print("\n--- Membuat Pasangan Stereo dengan Berbagai Baseline ---")

# Mendefinisikan daftar baseline yang akan diuji
baselines = [5, 20, 50, 100]

# Mendefinisikan label deskriptif untuk setiap baseline
baseline_labels = ["Kecil (5px)", "Medium (20px)", "Besar (50px)", "Sangat Besar (100px)"]

# Menyiapkan dictionary untuk menyimpan hasil setiap baseline
results = {}

# Mendefinisikan parameter StereoSGBM yang konsisten
num_disp = 128

# Mendefinisikan ukuran blok matching
block_size = 7

# Menghitung parameter smoothness P1 dan P2
P1 = 8 * 3 * block_size ** 2
P2 = 32 * 3 * block_size ** 2

# Iterasi untuk setiap nilai baseline
for i, bl in enumerate(baselines):
    # Menampilkan informasi baseline yang sedang diproses
    print(f"\n[BASELINE {bl}px] Memproses pasangan stereo...")

    # Membuat gambar kiri (sama untuk semua baseline)
    img_left = img_base.copy()

    # Membuat gambar kanan sebagai salinan gambar kiri
    img_right = img_base.copy()

    # Menghitung shift background berdasarkan baseline
    bg_shift = max(1, int(focal_length * bl / 2500.0))

    # Memastikan bg_shift tidak melebihi lebar gambar
    bg_shift = min(bg_shift, w - 1)

    # Menerapkan shift pada background gambar kanan
    img_right[:, :-bg_shift] = img_left[:, bg_shift:]

    # Menggeser setiap objek sesuai baseline dan depth-nya
    for obj in objects:
        # Mengekstrak posisi dan depth objek
        x1, y1, x2, y2 = obj["pos"]
        depth = obj["depth"]
        # Menghitung disparity = f * B / Z
        disp_val = int(focal_length * bl / depth)
        # Memastikan shift tidak negatif
        disp_val = max(0, min(disp_val, w - 1))
        # Menghitung posisi baru di gambar kanan
        new_x1 = max(0, x1 - disp_val)
        new_x2 = max(0, x2 - disp_val)
        # Menggambar objek di posisi baru pada gambar kanan
        if new_x2 > new_x1:
            cv2.rectangle(img_right, (new_x1, y1), (new_x2, y2), obj["color"], -1)

    # Mengkonversi ke grayscale untuk stereo matching
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Membuat objek StereoSGBM
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Menghitung disparity map
    disp_raw = sgbm.compute(gray_left, gray_right)

    # Mengkonversi ke float (dibagi 16)
    disp_float = disp_raw.astype(np.float32) / 16.0

    # Membuat mask piksel valid
    valid_mask = disp_float > 0

    # Menghitung depth dari disparity: Z = f * B / d
    depth_est = np.zeros_like(disp_float)
    depth_est[valid_mask] = focal_length * bl / disp_float[valid_mask]

    # Menyimpan hasil ke dictionary
    results[bl] = {
        "disp": disp_float,
        "depth": depth_est,
        "valid": valid_mask,
        "img_left": img_left,
        "img_right": img_right,
        "coverage": valid_mask.sum() / (h * w) * 100
    }

    # Menampilkan informasi coverage
    print(f"  Coverage: {results[bl]['coverage']:.1f}%")

# ============================================================
# 3. Menghitung error metrics per baseline
# ============================================================

# Menampilkan informasi tahap perhitungan error
print("\n--- Menghitung Error Metrics ---")

# Menyiapkan list untuk menyimpan metrik error
mae_list = []
rmse_list = []

# Iterasi untuk setiap baseline
for bl in baselines:
    # Mengambil depth estimasi dan mask valid
    depth_est = results[bl]["depth"]
    valid = results[bl]["valid"]

    # Membuat mask gabungan: valid dan depth ground truth bukan background
    eval_mask = valid & (gt_depth < 2500.0) & (depth_est > 0) & (depth_est < 5000)

    # Menghitung error absolut pada area evaluasi
    abs_error = np.abs(depth_est[eval_mask] - gt_depth[eval_mask])

    # Menghitung MAE (Mean Absolute Error)
    mae = np.mean(abs_error) if abs_error.size > 0 else float('inf')

    # Menghitung RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(abs_error ** 2)) if abs_error.size > 0 else float('inf')

    # Menyimpan metrik ke list
    mae_list.append(mae)
    rmse_list.append(rmse)

    # Menyimpan metrik ke dictionary hasil
    results[bl]["mae"] = mae
    results[bl]["rmse"] = rmse

    # Menampilkan informasi metrik error
    print(f"[BASELINE {bl}px] MAE={mae:.2f}, RMSE={rmse:.2f}, Eval pixels={eval_mask.sum()}")

# ============================================================
# 4. Analisis akurasi Near vs Far per baseline
# ============================================================

# Menampilkan informasi tahap analisis near vs far
print("\n--- Analisis Akurasi Near vs Far ---")

# Mendefinisikan threshold untuk klasifikasi near dan far
near_threshold = 300.0
far_threshold = 600.0

# Menyiapkan dictionary untuk menyimpan analisis
near_errors = []
far_errors = []

# Iterasi untuk setiap baseline
for bl in baselines:
    # Mengambil depth estimasi dan mask valid
    depth_est = results[bl]["depth"]
    valid = results[bl]["valid"]

    # Membuat mask untuk objek dekat (near)
    near_mask = valid & (gt_depth <= near_threshold) & (depth_est > 0) & (depth_est < 5000)

    # Membuat mask untuk objek jauh (far)
    far_mask = valid & (gt_depth >= far_threshold) & (gt_depth < 2500) & (depth_est > 0) & (depth_est < 5000)

    # Menghitung MAE untuk objek dekat
    near_mae = np.mean(np.abs(depth_est[near_mask] - gt_depth[near_mask])) if near_mask.sum() > 0 else float('inf')

    # Menghitung MAE untuk objek jauh
    far_mae = np.mean(np.abs(depth_est[far_mask] - gt_depth[far_mask])) if far_mask.sum() > 0 else float('inf')

    # Menyimpan error ke list
    near_errors.append(near_mae)
    far_errors.append(far_mae)

    # Menampilkan informasi error per zona
    print(f"[BASELINE {bl}px] Near MAE={near_mae:.2f}, Far MAE={far_mae:.2f}")

# ============================================================
# 5. Membuat visualisasi perbandingan depth map
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi Perbandingan ---")

# Membuat figure untuk perbandingan depth map
fig1, axes1 = plt.subplots(2, 4, figsize=(18, 9))

# Mengatur judul utama
fig1.suptitle("Pengaruh Baseline terhadap Depth Map", fontsize=16, fontweight='bold')

# Menampilkan pasangan stereo dan depth map untuk setiap baseline
for idx, bl in enumerate(baselines):
    # Menampilkan gambar kanan (menunjukkan shift) di baris atas
    axes1[0, idx].imshow(cv2.cvtColor(results[bl]["img_right"], cv2.COLOR_BGR2RGB))
    axes1[0, idx].set_title(f"Right (B={bl}px)", fontsize=10)
    axes1[0, idx].axis('off')

    # Menormalisasi depth map untuk visualisasi
    disp = results[bl]["disp"]
    disp_vis = np.zeros_like(disp, dtype=np.uint8)
    valid = results[bl]["valid"]
    if valid.sum() > 0:
        cv2.normalize(disp, disp_vis, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=valid.astype(np.uint8))

    # Menampilkan depth map di baris bawah
    axes1[1, idx].imshow(disp_vis, cmap='jet')
    axes1[1, idx].set_title(f"Depth (MAE={results[bl]['mae']:.1f})", fontsize=10)
    axes1[1, idx].axis('off')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure perbandingan depth map
output_path1 = os.path.join(OUTPUT_DIR, "18_baseline_depth_comparison.png")
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVE] Perbandingan depth map disimpan: {output_path1}")

# Menutup figure
plt.close(fig1)

# ============================================================
# 6. Membuat grafik Error vs Baseline
# ============================================================

# Menampilkan informasi tahap grafik error
print("\n--- Membuat Grafik Error vs Baseline ---")

# Membuat figure untuk grafik error
fig2, (ax_err, ax_nf) = plt.subplots(1, 2, figsize=(14, 5))

# Mengatur judul utama
fig2.suptitle("Analisis Error terhadap Baseline", fontsize=15, fontweight='bold')

# Menampilkan grafik MAE dan RMSE vs baseline
ax_err.plot(baselines, mae_list, 'bo-', linewidth=2, markersize=8, label='MAE')
ax_err.plot(baselines, rmse_list, 'rs-', linewidth=2, markersize=8, label='RMSE')

# Mengatur judul subplot error
ax_err.set_title("Error Depth vs Baseline", fontsize=12)

# Mengatur label sumbu X
ax_err.set_xlabel("Baseline (pixel)", fontsize=11)

# Mengatur label sumbu Y
ax_err.set_ylabel("Error Depth", fontsize=11)

# Menambahkan legend
ax_err.legend(fontsize=10)

# Menambahkan grid
ax_err.grid(True, alpha=0.3)

# Menampilkan grafik Near vs Far error
ax_nf.plot(baselines, near_errors, 'go-', linewidth=2, markersize=8, label='Near Objects')
ax_nf.plot(baselines, far_errors, 'r^-', linewidth=2, markersize=8, label='Far Objects')

# Mengatur judul subplot near vs far
ax_nf.set_title("Near vs Far Object Error", fontsize=12)

# Mengatur label sumbu X
ax_nf.set_xlabel("Baseline (pixel)", fontsize=11)

# Mengatur label sumbu Y
ax_nf.set_ylabel("MAE Depth", fontsize=11)

# Menambahkan legend
ax_nf.legend(fontsize=10)

# Menambahkan grid
ax_nf.grid(True, alpha=0.3)

# Mengatur layout
plt.tight_layout()

# Menyimpan grafik error ke file
output_path2 = os.path.join(OUTPUT_DIR, "18_error_vs_baseline.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVE] Grafik error vs baseline disimpan: {output_path2}")

# Menutup figure
plt.close(fig2)

# ============================================================
# 7. Membuat tabel ringkasan error
# ============================================================

# Menampilkan informasi tahap tabel ringkasan
print("\n--- Tabel Ringkasan Error ---")

# Membuat figure untuk tabel
fig3, ax_tbl = plt.subplots(1, 1, figsize=(10, 4))

# Menonaktifkan axis pada subplot tabel
ax_tbl.axis('off')

# Mendefinisikan header kolom tabel
col_labels = ["Baseline", "Coverage", "MAE", "RMSE", "Near MAE", "Far MAE"]

# Menyiapkan data baris tabel
table_data = []

# Mengisi data tabel dari hasil analisis
for idx, bl in enumerate(baselines):
    # Membuat baris data untuk setiap baseline
    row = [
        f"{bl} px",
        f"{results[bl]['coverage']:.1f}%",
        f"{mae_list[idx]:.2f}",
        f"{rmse_list[idx]:.2f}",
        f"{near_errors[idx]:.2f}",
        f"{far_errors[idx]:.2f}"
    ]
    # Menambahkan baris ke data tabel
    table_data.append(row)

# Membuat tabel pada subplot
table = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')

# Mengatur ukuran font tabel
table.auto_set_font_size(False)
table.set_fontsize(11)

# Mengatur skala tabel agar lebih besar
table.scale(1.2, 1.6)

# Mengatur judul tabel
ax_tbl.set_title("Ringkasan Error per Baseline", fontsize=14, fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Menyimpan tabel ke file
output_path3 = os.path.join(OUTPUT_DIR, "18_error_table.png")
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"[SAVE] Tabel error disimpan: {output_path3}")

# Menutup figure
plt.close(fig3)

# Menampilkan ringkasan akhir percobaan
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 18")
print("=" * 60)
print(f"  Ukuran gambar       : {h}x{w}")
print(f"  Focal length        : {focal_length:.1f}")
print(f"  Sumber gambar       : stereo_left.png")
print(f"  Baseline diuji      : {baselines}")
print(f"  Baseline terbaik    : {baselines[np.argmin(mae_list)]}px (MAE={min(mae_list):.2f})")
print(f"  File output tersimpan: 3 file")
print("=" * 60)
