"""
==========================================================================
PERCOBAAN 12: PERBANDINGAN STEREO BM VS SGBM
==========================================================================
Program ini membandingkan StereoBM dan StereoSGBM secara menyeluruh
pada beberapa scene dengan tingkat kesulitan berbeda. Perbandingan
meliputi akurasi, kecepatan komputasi, coverage (persentase piksel
valid), dan kualitas visual disparity map.

Konsep utama:
- StereoBM: block matching sederhana, cepat tapi kurang akurat
- StereoSGBM: semi-global matching, lebih akurat tapi lebih lambat
- Coverage: persentase piksel yang memiliki disparity valid
- Noise level mempengaruhi kualitas matching kedua algoritma
- Scene dengan tekstur kaya lebih mudah untuk stereo matching

Fungsi utama yang dipelajari:
- cv2.StereoBM_create()     : Membuat objek StereoBM
- cv2.StereoSGBM_create()   : Membuat objek StereoSGBM
- stereo.compute()           : Menghitung disparity map
- time.time()                : Mengukur waktu komputasi
- cv2.normalize()            : Normalisasi untuk visualisasi
- cv2.applyColorMap()        : Menerapkan colormap pada disparity

Hasil: Grid perbandingan visual, tabel metrik, dan analisis pro/kontra
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
print("PERCOBAAN 12: PERBANDINGAN STEREO BM VS SGBM")
print("=" * 60)


# ============================================================
# 1. Memuat gambar stereo real dan membuat variasi scene
# ============================================================

# Menampilkan informasi tahap memuat gambar
print("\n--- Memuat Gambar Stereo Real ---")

# Mendefinisikan path gambar stereo
path_left  = os.path.join(IMAGE_DIR, "stereo_left.png")
path_right = os.path.join(IMAGE_DIR, "stereo_right.png")

# Memuat gambar stereo
img_left_base  = cv2.imread(path_left)
img_right_base = cv2.imread(path_right)

# Download otomatis jika file tidak tersedia
if img_left_base is None or img_right_base is None:
    print("[WARN] Gambar stereo tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    img_left_base  = cv2.imread(path_left)
    img_right_base = cv2.imread(path_right)
if img_left_base is None:
    raise FileNotFoundError("[ERROR] stereo_left.png tidak tersedia. Jalankan: python download_image.py")
if img_right_base is None:
    raise FileNotFoundError("[ERROR] stereo_right.png tidak tersedia. Jalankan: python download_image.py")

# Membuat 3 variasi scene dari gambar asli untuk perbandingan BM vs SGBM
# Easy: gambar asli langsung
# Medium: gambar asli dikonversi ke grayscale lalu kembali ke BGR
# Hard: gambar asli dengan noise tinggi (sedikit fitur)
def create_real_scene(scene_type, img_l, img_r):
    """Menyiapkan pasangan stereo real dengan variasi preprocessing."""
    h_r, w_r = img_l.shape[:2]
    target_w, target_h = 400, 300
    left  = cv2.resize(img_l, (target_w, target_h))
    right = cv2.resize(img_r, (target_w, target_h))
    if scene_type == "easy":
        # Gambar asli langsung (banyak tekstur detail tinggi)
        return left, right
    elif scene_type == "medium":
        # Soft blur untuk mengurangi noise tinggi frekuensi
        left  = cv2.GaussianBlur(left,  (3, 3), 0.5)
        right = cv2.GaussianBlur(right, (3, 3), 0.5)
        return left, right
    elif scene_type == "hard":
        # Tambahkan noise kuat untuk simulasi kondisi sulit
        noise = np.random.randint(0, 60, left.shape, dtype=np.uint8)
        left  = cv2.add(left,  noise)
        right = cv2.add(right, noise)
        return left, right
    return left, right

# Mendefinisikan konfigurasi scene
scene_configs = [
    ("Easy (Stereo Real)",    "easy"),
    ("Medium (Blur Ringan)",  "medium"),
    ("Hard (Noise Tinggi)",   "hard"),
]

# Membuat setiap variasi scene
stereo_pairs = []
for scene_name, scene_type in scene_configs:
    left, right = create_real_scene(scene_type, img_left_base, img_right_base)
    stereo_pairs.append((scene_name, left, right))
    print(f"[INFO] Scene '{scene_name}' disiapkan: {left.shape}")

# ============================================================
# 2. Menghitung disparity dengan BM dan SGBM untuk setiap scene
# ============================================================

# Menampilkan informasi tahap komputasi
print("\n--- Menghitung Disparity BM & SGBM per Scene ---")

# Mendefinisikan parameter umum
num_disp = 64
block_size = 11

# Mendefinisikan parameter SGBM
P1 = 8 * 3 * block_size ** 2
P2 = 32 * 3 * block_size ** 2

# Menyiapkan list untuk menyimpan semua hasil
all_results = []

# Iterasi setiap scene
for scene_name, img_left, img_right in stereo_pairs:
    # Menampilkan informasi scene
    print(f"\n  Scene: {scene_name}")

    # Mengkonversi ke grayscale
    gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Mendapatkan ukuran gambar
    h, w = gray_l.shape

    # --- StereoBM ---
    # Mencatat waktu mulai BM
    t_start_bm = time.time()

    # Membuat dan menjalankan StereoBM
    stereo_bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    disp_bm = stereo_bm.compute(gray_l, gray_r)

    # Mencatat waktu selesai BM
    time_bm = time.time() - t_start_bm

    # Mengkonversi disparity BM ke float
    disp_bm_float = disp_bm.astype(np.float32) / 16.0

    # --- StereoSGBM ---
    # Mencatat waktu mulai SGBM
    t_start_sgbm = time.time()

    # Membuat dan menjalankan StereoSGBM
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
    disp_sgbm = stereo_sgbm.compute(gray_l, gray_r)

    # Mencatat waktu selesai SGBM
    time_sgbm = time.time() - t_start_sgbm

    # Mengkonversi disparity SGBM ke float
    disp_sgbm_float = disp_sgbm.astype(np.float32) / 16.0

    # Menghitung metrik BM
    valid_bm = disp_bm_float > 0
    coverage_bm = valid_bm.sum() / (h * w) * 100
    mean_bm = disp_bm_float[valid_bm].mean() if valid_bm.any() else 0
    std_bm = disp_bm_float[valid_bm].std() if valid_bm.any() else 0

    # Menghitung metrik SGBM
    valid_sgbm = disp_sgbm_float > 0
    coverage_sgbm = valid_sgbm.sum() / (h * w) * 100
    mean_sgbm = disp_sgbm_float[valid_sgbm].mean() if valid_sgbm.any() else 0
    std_sgbm = disp_sgbm_float[valid_sgbm].std() if valid_sgbm.any() else 0

    # Menampilkan metrik
    print(f"    BM  : waktu={time_bm:.4f}s, coverage={coverage_bm:.1f}%, "
          f"mean={mean_bm:.2f}, std={std_bm:.2f}")
    print(f"    SGBM: waktu={time_sgbm:.4f}s, coverage={coverage_sgbm:.1f}%, "
          f"mean={mean_sgbm:.2f}, std={std_sgbm:.2f}")

    # Menormalisasi disparity BM untuk visualisasi
    disp_bm_vis = cv2.normalize(disp_bm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_bm_color = cv2.applyColorMap(disp_bm_vis, cv2.COLORMAP_JET)

    # Menormalisasi disparity SGBM untuk visualisasi
    disp_sgbm_vis = cv2.normalize(disp_sgbm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_sgbm_color = cv2.applyColorMap(disp_sgbm_vis, cv2.COLORMAP_JET)

    # Menyimpan semua hasil ke list
    all_results.append({
        'scene': scene_name,
        'img_left': img_left,
        'disp_bm_color': disp_bm_color,
        'disp_sgbm_color': disp_sgbm_color,
        'time_bm': time_bm,
        'time_sgbm': time_sgbm,
        'coverage_bm': coverage_bm,
        'coverage_sgbm': coverage_sgbm,
        'mean_bm': mean_bm,
        'mean_sgbm': mean_sgbm,
        'std_bm': std_bm,
        'std_sgbm': std_sgbm,
    })

# ============================================================
# 3. Tabel perbandingan menyeluruh
# ============================================================

# Menampilkan tabel perbandingan
print("\n" + "=" * 80)
print("TABEL PERBANDINGAN BM VS SGBM")
print("=" * 80)

# Menampilkan header tabel
print(f"{'Scene':<25} {'Metode':<8} {'Waktu(s)':<10} {'Coverage%':<10} "
      f"{'Mean':<8} {'StdDev':<8}")
print("-" * 80)

# Menampilkan data setiap scene
for res in all_results:
    # Menampilkan baris BM
    print(f"{res['scene']:<25} {'BM':<8} {res['time_bm']:<10.4f} "
          f"{res['coverage_bm']:<10.1f} {res['mean_bm']:<8.2f} {res['std_bm']:<8.2f}")

    # Menampilkan baris SGBM
    print(f"{'':<25} {'SGBM':<8} {res['time_sgbm']:<10.4f} "
          f"{res['coverage_sgbm']:<10.1f} {res['mean_sgbm']:<8.2f} {res['std_sgbm']:<8.2f}")

    # Menampilkan separator
    print("-" * 80)

# ============================================================
# 4. Menghitung rata-rata performa
# ============================================================

# Menampilkan informasi rata-rata
print("\n--- Rata-rata Performa ---")

# Menghitung rata-rata waktu BM
avg_time_bm = np.mean([r['time_bm'] for r in all_results])

# Menghitung rata-rata waktu SGBM
avg_time_sgbm = np.mean([r['time_sgbm'] for r in all_results])

# Menghitung rata-rata coverage BM
avg_cov_bm = np.mean([r['coverage_bm'] for r in all_results])

# Menghitung rata-rata coverage SGBM
avg_cov_sgbm = np.mean([r['coverage_sgbm'] for r in all_results])

# Menghitung speedup SGBM vs BM
speedup = avg_time_sgbm / (avg_time_bm + 1e-8)

# Menampilkan rata-rata
print(f"[AVG] BM  : waktu={avg_time_bm:.4f}s, coverage={avg_cov_bm:.1f}%")
print(f"[AVG] SGBM: waktu={avg_time_sgbm:.4f}s, coverage={avg_cov_sgbm:.1f}%")
print(f"[AVG] SGBM {speedup:.1f}x lebih lambat dari BM")
print(f"[AVG] SGBM coverage {avg_cov_sgbm - avg_cov_bm:+.1f}% dibanding BM")

# ============================================================
# 5. Visualisasi grid perbandingan (3 scenes × 3 kolom)
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi Grid Perbandingan ---")

# Membuat figure 3x3 (3 scenes × Input/BM/SGBM)
fig1, axes1 = plt.subplots(3, 3, figsize=(16, 14))

# Iterasi setiap scene
for i, res in enumerate(all_results):
    # Menampilkan gambar input kiri
    axes1[i, 0].imshow(cv2.cvtColor(res['img_left'], cv2.COLOR_BGR2RGB))
    axes1[i, 0].set_title(f"{res['scene']}\n(Input)", fontsize=10)
    axes1[i, 0].axis("off")

    # Menampilkan disparity BM
    axes1[i, 1].imshow(cv2.cvtColor(res['disp_bm_color'], cv2.COLOR_BGR2RGB))
    axes1[i, 1].set_title(f"BM (t={res['time_bm']:.4f}s, "
                           f"cov={res['coverage_bm']:.1f}%)", fontsize=10)
    axes1[i, 1].axis("off")

    # Menampilkan disparity SGBM
    axes1[i, 2].imshow(cv2.cvtColor(res['disp_sgbm_color'], cv2.COLOR_BGR2RGB))
    axes1[i, 2].set_title(f"SGBM (t={res['time_sgbm']:.4f}s, "
                           f"cov={res['coverage_sgbm']:.1f}%)", fontsize=10)
    axes1[i, 2].axis("off")

# Mengatur judul utama
plt.suptitle("Percobaan 12: Perbandingan BM vs SGBM pada Berbagai Scene",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure grid
output_grid = os.path.join(OUTPUT_DIR, "12_bm_vs_sgbm_grid.png")
plt.savefig(output_grid, dpi=150, bbox_inches='tight')
print(f"[SAVED] Grid perbandingan: {output_grid}")

# ============================================================
# 6. Visualisasi bar chart metrik
# ============================================================

# Menampilkan informasi tahap bar chart
print("\n--- Membuat Visualisasi Bar Chart Metrik ---")

# Menyiapkan data untuk bar chart
scene_labels = [r['scene'].split(' ')[0] for r in all_results]
x = np.arange(len(scene_labels))
bar_width = 0.35

# Membuat figure 1x3 untuk bar chart
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

# Bar chart 1: Waktu komputasi
times_bm = [r['time_bm'] for r in all_results]
times_sgbm = [r['time_sgbm'] for r in all_results]
axes2[0].bar(x - bar_width / 2, times_bm, bar_width, label='BM', color='steelblue')
axes2[0].bar(x + bar_width / 2, times_sgbm, bar_width, label='SGBM', color='coral')
axes2[0].set_title("Waktu Komputasi (detik)", fontsize=12)
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(scene_labels, fontsize=9)
axes2[0].legend()
axes2[0].set_ylabel("Detik")

# Bar chart 2: Coverage
cov_bm = [r['coverage_bm'] for r in all_results]
cov_sgbm = [r['coverage_sgbm'] for r in all_results]
axes2[1].bar(x - bar_width / 2, cov_bm, bar_width, label='BM', color='steelblue')
axes2[1].bar(x + bar_width / 2, cov_sgbm, bar_width, label='SGBM', color='coral')
axes2[1].set_title("Coverage (% Piksel Valid)", fontsize=12)
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(scene_labels, fontsize=9)
axes2[1].legend()
axes2[1].set_ylabel("Persen (%)")

# Bar chart 3: Mean disparity
mean_bm_vals = [r['mean_bm'] for r in all_results]
mean_sgbm_vals = [r['mean_sgbm'] for r in all_results]
axes2[2].bar(x - bar_width / 2, mean_bm_vals, bar_width, label='BM', color='steelblue')
axes2[2].bar(x + bar_width / 2, mean_sgbm_vals, bar_width, label='SGBM', color='coral')
axes2[2].set_title("Mean Disparity (pixel)", fontsize=12)
axes2[2].set_xticks(x)
axes2[2].set_xticklabels(scene_labels, fontsize=9)
axes2[2].legend()
axes2[2].set_ylabel("Pixel")

# Mengatur judul utama
plt.suptitle("Percobaan 12: Metrik Perbandingan BM vs SGBM",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure bar chart
output_chart = os.path.join(OUTPUT_DIR, "12_bm_vs_sgbm_chart.png")
plt.savefig(output_chart, dpi=150, bbox_inches='tight')
print(f"[SAVED] Bar chart metrik: {output_chart}")

# ============================================================
# 7. Analisis pro dan kontra
# ============================================================

# Menampilkan analisis pro/kontra
print("\n" + "=" * 60)
print("ANALISIS PRO DAN KONTRA")
print("=" * 60)

# Menampilkan kelebihan StereoBM
print("\n[StereoBM - Kelebihan]")
print("  + Komputasi sangat cepat")
print("  + Implementasi sederhana, sedikit parameter")
print("  + Cocok untuk aplikasi real-time")
print("  + Memori yang dibutuhkan lebih sedikit")

# Menampilkan kekurangan StereoBM
print("\n[StereoBM - Kekurangan]")
print("  - Hanya bekerja pada gambar grayscale")
print("  - Hasil noisy pada area dengan sedikit tekstur")
print("  - Coverage lebih rendah (banyak piksel invalid)")
print("  - Kurang akurat pada batas objek")

# Menampilkan kelebihan StereoSGBM
print("\n[StereoSGBM - Kelebihan]")
print("  + Hasil lebih halus dan konsisten")
print("  + Coverage lebih tinggi (lebih banyak piksel valid)")
print("  + Mendukung gambar berwarna (multi-channel)")
print("  + Lebih baik pada area dengan sedikit tekstur")
print("  + Akurat pada batas objek")

# Menampilkan kekurangan StereoSGBM
print("\n[StereoSGBM - Kekurangan]")
print("  - Komputasi lebih lambat dari BM")
print("  - Lebih banyak parameter yang perlu di-tune")
print("  - Membutuhkan memori lebih banyak")

# ============================================================
# 8. Rekomendasi penggunaan
# ============================================================

# Menampilkan rekomendasi
print("\n[Rekomendasi]")
print("  - Real-time + sumber daya terbatas -> StereoBM")
print("  - Akurasi tinggi + offline processing -> StereoSGBM")
print("  - Scene bertekstur kaya -> keduanya baik")
print("  - Scene homogen/minim tekstur -> StereoSGBM")

# Menampilkan semua figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 12: PERBANDINGAN STEREO BM VS SGBM")
print("=" * 60)
print(f"1. StereoBM menggunakan block matching lokal (cepat)")
print(f"2. StereoSGBM mengoptimasi cost semi-global (akurat)")
print(f"3. SGBM rata-rata {speedup:.1f}x lebih lambat dari BM")
print(f"4. SGBM memiliki coverage {avg_cov_sgbm - avg_cov_bm:+.1f}% lebih baik")
print(f"5. Scene bertekstur kaya: kedua metode bekerja baik")
print(f"6. Scene homogen: SGBM jauh lebih unggul")
print(f"7. BM cocok untuk real-time, SGBM untuk offline/akurat")
print(f"8. Parameter P1, P2 pada SGBM mengontrol smoothness")
print(f"9. Coverage menunjukkan seberapa banyak piksel ter-match")
print(f"10. Evaluasi multi-scene penting untuk pilihan algoritma")
print("=" * 60)
