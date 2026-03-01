"""
==========================================================================
PERCOBAAN 5: AKAZE DAN FAST FEATURE DETECTION
==========================================================================
Program ini mempelajari dua detektor fitur tambahan: AKAZE dan FAST.
AKAZE menggunakan nonlinear scale space untuk deteksi fitur yang lebih
robust, sedangkan FAST menggunakan segment test yang sangat cepat.

Konsep yang dipelajari:
- AKAZE: Accelerated-KAZE, menggunakan nonlinear diffusion filtering
  untuk membangun scale space (bukan Gaussian blur seperti SIFT)
- FAST: Features from Accelerated Segment Test, mendeteksi corner
  dengan membandingkan intensitas piksel pada lingkaran Bresenham
- Perbandingan performa berbagai detektor (SIFT, ORB, AKAZE, FAST)
- Benchmark kecepatan dan jumlah keypoint

Fungsi utama yang dipelajari:
- cv2.AKAZE_create()              : Membuat detektor AKAZE
- cv2.FastFeatureDetector_create() : Membuat detektor FAST
- akaze.detectAndCompute()         : Mendeteksi dan menghitung descriptor AKAZE
- fast.detect()                    : Mendeteksi keypoint FAST (tanpa descriptor)

Hasil: Visualisasi dan perbandingan AKAZE, FAST, SIFT, dan ORB
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan deteksi fitur
import cv2

# Mengimpor NumPy untuk operasi array dan statistik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu pemrosesan
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
print("PERCOBAAN 5: AKAZE DAN FAST FEATURE DETECTION")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar
# ============================================================

# Membaca gambar bangunan dari file
img_bangunan = cv2.imread(os.path.join(IMAGE_DIR, "bangunan.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_bangunan is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi ukuran gambar
print(f"[INFO] Ukuran bangunan: {img_bangunan.shape}")

# Mengkonversi gambar ke grayscale
gray_bangunan = cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Deteksi AKAZE Dasar
# ============================================================

# Membuat objek detektor AKAZE dengan parameter default
akaze = cv2.AKAZE_create()

# Mendeteksi keypoint dan menghitung descriptor AKAZE
kp_akaze, desc_akaze = akaze.detectAndCompute(gray_bangunan, None)

# Menampilkan jumlah keypoint AKAZE
print(f"\n[HASIL] Jumlah keypoint AKAZE: {len(kp_akaze)}")

# Menampilkan informasi descriptor AKAZE
if desc_akaze is not None:
    print(f"[HASIL] Shape descriptor AKAZE: {desc_akaze.shape}")
    print(f"[HASIL] Tipe descriptor AKAZE: {desc_akaze.dtype}")

# ============================================================
# 3. Deteksi FAST Dasar
# ============================================================

# Membuat objek detektor FAST dengan threshold=25
fast = cv2.FastFeatureDetector_create(threshold=25)

# Mendeteksi keypoint FAST (FAST hanya mendeteksi keypoint, bukan descriptor)
kp_fast = fast.detect(gray_bangunan, None)

# Menampilkan jumlah keypoint FAST
print(f"[HASIL] Jumlah keypoint FAST: {len(kp_fast)}")

# Menampilkan status non-max suppression
print(f"[HASIL] FAST nonmaxSuppression: {fast.getNonmaxSuppression()}")

# Menampilkan threshold FAST
print(f"[HASIL] FAST threshold: {fast.getThreshold()}")

# ============================================================
# 4. Visualisasi AKAZE dan FAST
# ============================================================

# Menggambar keypoints AKAZE dengan rich keypoints
img_akaze = cv2.drawKeypoints(img_bangunan, kp_akaze, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Menggambar keypoints FAST
img_fast = cv2.drawKeypoints(img_bangunan, kp_fast, None,
                              color=(0, 255, 0))

# Membuat figure untuk visualisasi AKAZE dan FAST
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Menampilkan gambar asli
axes[0].imshow(cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2RGB))
axes[0].set_title("Gambar Asli", fontsize=12)
axes[0].axis('off')

# Menampilkan AKAZE keypoints
axes[1].imshow(cv2.cvtColor(img_akaze, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"AKAZE ({len(kp_akaze)} keypoints)", fontsize=12)
axes[1].axis('off')

# Menampilkan FAST keypoints
axes[2].imshow(cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"FAST ({len(kp_fast)} keypoints)", fontsize=12)
axes[2].axis('off')

# Memberikan judul utama
fig.suptitle("Percobaan 5: AKAZE dan FAST Feature Detection", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi
plt.savefig(os.path.join(OUTPUT_DIR, "05_akaze_fast.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 05_akaze_fast.png")

# Menutup figure
plt.close()

# ============================================================
# 5. Variasi FAST Threshold
# ============================================================

# Mendefinisikan daftar threshold FAST yang akan diuji
fast_thresholds = [10, 25, 50, 100]

# Menampilkan header variasi threshold FAST
print(f"\n--- Variasi FAST Threshold ---")

# Membuat figure untuk variasi threshold
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Melakukan iterasi untuk setiap threshold
for i, thr in enumerate(fast_thresholds):
    # Membuat detektor FAST dengan threshold tertentu
    fast_thr = cv2.FastFeatureDetector_create(threshold=thr)

    # Mendeteksi keypoint
    kp_thr = fast_thr.detect(gray_bangunan, None)

    # Menggambar keypoints
    img_thr = cv2.drawKeypoints(img_bangunan, kp_thr, None, color=(0, 255, 0))

    # Menampilkan pada subplot
    axes[i].imshow(cv2.cvtColor(img_thr, cv2.COLOR_BGR2RGB))

    # Memberikan judul
    axes[i].set_title(f"threshold={thr}\n({len(kp_thr)} keypoints)", fontsize=11)

    # Menonaktifkan sumbu
    axes[i].axis('off')

    # Menampilkan ke konsol
    print(f"  FAST threshold={thr}: {len(kp_thr)} keypoints")

# Memberikan judul utama
fig.suptitle("Variasi FAST Threshold", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi variasi threshold FAST
plt.savefig(os.path.join(OUTPUT_DIR, "05_fast_threshold.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 05_fast_threshold.png")

# Menutup figure
plt.close()

# ============================================================
# 6. Variasi AKAZE Threshold
# ============================================================

# Mendefinisikan daftar threshold AKAZE
akaze_thresholds = [0.0005, 0.001, 0.005, 0.01]

# Menampilkan header variasi threshold AKAZE
print(f"\n--- Variasi AKAZE Threshold ---")

# Membuat figure untuk variasi AKAZE
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Melakukan iterasi untuk setiap threshold AKAZE
for i, thr in enumerate(akaze_thresholds):
    # Membuat detektor AKAZE dengan threshold tertentu
    akaze_thr = cv2.AKAZE_create(threshold=thr)

    # Mendeteksi keypoint dan descriptor
    kp_a_thr, desc_a_thr = akaze_thr.detectAndCompute(gray_bangunan, None)

    # Menggambar rich keypoints
    img_a_thr = cv2.drawKeypoints(img_bangunan, kp_a_thr, None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Menampilkan pada subplot
    axes[i].imshow(cv2.cvtColor(img_a_thr, cv2.COLOR_BGR2RGB))

    # Memberikan judul
    axes[i].set_title(f"threshold={thr}\n({len(kp_a_thr)} keypoints)", fontsize=10)

    # Menonaktifkan sumbu
    axes[i].axis('off')

    # Menampilkan ke konsol
    print(f"  AKAZE threshold={thr}: {len(kp_a_thr)} keypoints")

# Memberikan judul utama
fig.suptitle("Variasi AKAZE Threshold", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi variasi AKAZE
plt.savefig(os.path.join(OUTPUT_DIR, "05_akaze_threshold.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"[SAVED] 05_akaze_threshold.png")

# Menutup figure
plt.close()

# ============================================================
# 7. Benchmark Waktu: SIFT vs ORB vs AKAZE vs FAST
# ============================================================

# Menentukan jumlah iterasi benchmark
n_runs = 10

# Menampilkan header benchmark
print(f"\n--- Benchmark Waktu ({n_runs} runs) ---")

# Dictionary untuk menyimpan hasil benchmark
benchmark = {}

# Benchmark SIFT
sift_times = []
sift_counts = []
for run in range(n_runs):
    sift_b = cv2.SIFT_create()
    t_start = time.time()
    kp_b, _ = sift_b.detectAndCompute(gray_bangunan, None)
    sift_times.append(time.time() - t_start)
    sift_counts.append(len(kp_b))

# Menyimpan hasil SIFT
benchmark['SIFT'] = {'time': np.mean(sift_times), 'std': np.std(sift_times),
                     'count': int(np.mean(sift_counts))}

# Benchmark ORB
orb_times = []
orb_counts = []
for run in range(n_runs):
    orb_b = cv2.ORB_create(nfeatures=500)
    t_start = time.time()
    kp_b, _ = orb_b.detectAndCompute(gray_bangunan, None)
    orb_times.append(time.time() - t_start)
    orb_counts.append(len(kp_b))

# Menyimpan hasil ORB
benchmark['ORB'] = {'time': np.mean(orb_times), 'std': np.std(orb_times),
                    'count': int(np.mean(orb_counts))}

# Benchmark AKAZE
akaze_times = []
akaze_counts = []
for run in range(n_runs):
    akaze_b = cv2.AKAZE_create()
    t_start = time.time()
    kp_b, _ = akaze_b.detectAndCompute(gray_bangunan, None)
    akaze_times.append(time.time() - t_start)
    akaze_counts.append(len(kp_b))

# Menyimpan hasil AKAZE
benchmark['AKAZE'] = {'time': np.mean(akaze_times), 'std': np.std(akaze_times),
                      'count': int(np.mean(akaze_counts))}

# Benchmark FAST (hanya deteksi, tanpa descriptor)
fast_times = []
fast_counts = []
for run in range(n_runs):
    fast_b = cv2.FastFeatureDetector_create(threshold=25)
    t_start = time.time()
    kp_b = fast_b.detect(gray_bangunan, None)
    fast_times.append(time.time() - t_start)
    fast_counts.append(len(kp_b))

# Menyimpan hasil FAST
benchmark['FAST'] = {'time': np.mean(fast_times), 'std': np.std(fast_times),
                     'count': int(np.mean(fast_counts))}

# Menampilkan hasil benchmark ke konsol
for name, data in benchmark.items():
    print(f"  {name}: {data['time']*1000:.2f} ms (±{data['std']*1000:.2f}), "
          f"{data['count']} keypoints")

# ============================================================
# 8. Visualisasi Benchmark
# ============================================================

# Membuat figure untuk benchmark
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Mengekstrak data untuk plotting
names = list(benchmark.keys())
times_ms = [benchmark[n]['time']*1000 for n in names]
stds_ms = [benchmark[n]['std']*1000 for n in names]
counts = [benchmark[n]['count'] for n in names]
colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']

# Membuat bar chart waktu
bars1 = axes[0].bar(names, times_ms, yerr=stds_ms, color=colors,
                     edgecolor='black', capsize=5, alpha=0.8)

# Memberikan judul dan label
axes[0].set_title("Waktu Deteksi (ms)", fontsize=12)
axes[0].set_ylabel("Waktu (ms)", fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Menambahkan label nilai
for bar, t in zip(bars1, times_ms):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{t:.1f}', ha='center', va='bottom', fontsize=9)

# Membuat bar chart jumlah keypoint
bars2 = axes[1].bar(names, counts, color=colors, edgecolor='black', alpha=0.8)

# Memberikan judul dan label
axes[1].set_title("Jumlah Keypoint", fontsize=12)
axes[1].set_ylabel("Jumlah", fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

# Menambahkan label nilai
for bar, c in zip(bars2, counts):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                 str(c), ha='center', va='bottom', fontsize=9)

# Membuat scatter plot: waktu vs jumlah keypoint
for j, name in enumerate(names):
    axes[2].scatter(times_ms[j], counts[j], s=200, c=colors[j],
                    edgecolors='black', zorder=5, label=name)

# Memberikan judul dan label
axes[2].set_title("Waktu vs Jumlah Keypoint", fontsize=12)
axes[2].set_xlabel("Waktu (ms)", fontsize=11)
axes[2].set_ylabel("Jumlah Keypoint", fontsize=11)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

# Memberikan judul utama
fig.suptitle("Benchmark Detektor Fitur", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan benchmark
plt.savefig(os.path.join(OUTPUT_DIR, "05_detektor_benchmark.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 05_detektor_benchmark.png")

# Menutup figure
plt.close()

# ============================================================
# 9. Tabel Perbandingan Komprehensif
# ============================================================

# Membuat figure untuk tabel perbandingan
fig, ax = plt.subplots(figsize=(14, 6))

# Menyembunyikan sumbu
ax.axis('off')

# Mendefinisikan data tabel
table_data = [
    ['SIFT', 'DoG + Gradient', 'Float32 (128-D)', f"{benchmark['SIFT']['count']}",
     f"{benchmark['SIFT']['time']*1000:.2f} ms", 'Tinggi', 'Ya'],
    ['ORB', 'FAST + rBRIEF', 'Binary (256 bit)', f"{benchmark['ORB']['count']}",
     f"{benchmark['ORB']['time']*1000:.2f} ms", 'Sedang', 'Ya'],
    ['AKAZE', 'Nonlinear Diffusion', 'Binary (M-LDB)', f"{benchmark['AKAZE']['count']}",
     f"{benchmark['AKAZE']['time']*1000:.2f} ms", 'Tinggi', 'Ya'],
    ['FAST', 'Segment Test', 'Tidak ada', f"{benchmark['FAST']['count']}",
     f"{benchmark['FAST']['time']*1000:.2f} ms", 'Rendah', 'Tidak'],
]

# Mendefinisikan header kolom
col_labels = ['Detektor', 'Metode', 'Descriptor', 'Keypoints',
              'Waktu', 'Akurasi', 'Descriptor?']

# Membuat tabel
table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc='center', cellLoc='center')

# Mengatur ukuran font tabel
table.auto_set_font_size(False)
table.set_fontsize(10)

# Mengatur tinggi baris header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Mengatur warna baris data bergantian
for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[i, j].set_facecolor('#D6E4F0')
        else:
            table[i, j].set_facecolor('#EBF1F8')

# Mengatur skala tabel
table.scale(1, 1.8)

# Memberikan judul
ax.set_title("Tabel Perbandingan Detektor Fitur", fontsize=16,
             fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Menyimpan tabel perbandingan
plt.savefig(os.path.join(OUTPUT_DIR, "05_perbandingan_tabel.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"[SAVED] 05_perbandingan_tabel.png")

# Menutup figure
plt.close()

# ============================================================
# 10. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 5: AKAZE DAN FAST")

# Menampilkan garis pemisah
print("=" * 60)

# Menjelaskan AKAZE
print("1. AKAZE menggunakan nonlinear scale space (bukan Gaussian)")
print("   -> Lebih baik mempertahankan tepi dan detail halus")

# Menjelaskan FAST
print("2. FAST sangat cepat (segment test pada lingkaran Bresenham)")
print("   -> Hanya detektor keypoint, tanpa descriptor bawaan")

# Menjelaskan perbandingan
print("3. Perbandingan kecepatan:")
for name in names:
    print(f"   {name}: {benchmark[name]['time']*1000:.2f} ms, "
          f"{benchmark[name]['count']} keypoints")

# Menjelaskan penggunaan
print("4. Rekomendasi penggunaan:")
print("   - SIFT: akurasi tinggi, tidak sensitif terhadap waktu")
print("   - ORB: real-time, kebutuhan memori rendah")
print("   - AKAZE: keseimbangan akurasi dan kecepatan")
print("   - FAST: deteksi corner sangat cepat (perlu descriptor tambahan)")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 05_akaze_fast.png")
print("  - 05_fast_threshold.png")
print("  - 05_akaze_threshold.png")
print("  - 05_detektor_benchmark.png")
print("  - 05_perbandingan_tabel.png")

# Menampilkan garis penutup
print("=" * 60)
