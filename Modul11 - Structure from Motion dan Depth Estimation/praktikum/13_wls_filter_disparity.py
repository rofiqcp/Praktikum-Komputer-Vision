"""
==========================================================================
PERCOBAAN 13: WLS FILTER UNTUK POST-PROCESSING DISPARITY
==========================================================================
Program ini mempelajari cara memperhalus disparity map menggunakan
WLS (Weighted Least Squares) Filter dan teknik post-processing lainnya
seperti median blur dan bilateral filter.

Konsep utama:
- Disparity map mentah (raw) sering memiliki noise dan lubang (holes)
- Median blur efektif menghilangkan salt-and-pepper noise
- Bilateral filter memperhalus sambil menjaga tepi (edge-preserving)
- WLS filter menggunakan informasi gambar asli sebagai panduan
  untuk menghasilkan disparity yang halus dan edge-aware
- Perbandingan metrik coverage dan smoothness mengukur kualitas

Fungsi utama yang dipelajari:
- cv2.ximgproc.createDisparityWLSFilter() : Membuat WLS filter
- cv2.ximgproc.createRightMatcher()       : Membuat right matcher
- cv2.bilateralFilter()                    : Bilateral filter
- cv2.medianBlur()                         : Median blur filter
- cv2.StereoSGBM_create()                 : Menghitung raw disparity
- cv2.normalize()                          : Normalisasi visualisasi

Hasil: Perbandingan visual dan metrik antara berbagai metode filtering
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
print("PERCOBAAN 13: WLS FILTER UNTUK POST-PROCESSING DISPARITY")
print("=" * 60)

# ============================================================
# 1. Memuat pasangan gambar stereo real
# ============================================================

# Menampilkan informasi tahap memuat gambar
print("\n--- Memuat Pasangan Gambar Stereo ---")

# Mendefinisikan path gambar stereo
path_left  = os.path.join(IMAGE_DIR, "stereo_left.png")
path_right = os.path.join(IMAGE_DIR, "stereo_right.png")

# Memuat gambar stereo
img_left  = cv2.imread(path_left)
img_right = cv2.imread(path_right)

# Download otomatis jika file tidak tersedia
if img_left is None or img_right is None:
    print("[WARN] Gambar stereo tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    img_left  = cv2.imread(path_left)
    img_right = cv2.imread(path_right)
if img_left is None:
    raise FileNotFoundError("[ERROR] stereo_left.png tidak tersedia. Jalankan: python download_image.py")
if img_right is None:
    raise FileNotFoundError("[ERROR] stereo_right.png tidak tersedia. Jalankan: python download_image.py")

# Mendapatkan dimensi gambar
h, w = img_left.shape[:2]

# Parameter kamera estimasi
focal_length = 700.0
baseline     = 30.0

# Menampilkan informasi gambar
print(f"[INFO] Gambar kiri : {img_left.shape}")
print(f"[INFO] Gambar kanan: {img_right.shape}")

# ============================================================
# 2. Menghitung Raw Disparity dengan SGBM
# ============================================================

# Menampilkan informasi tahap disparity
print("\n--- Menghitung Raw Disparity dengan SGBM ---")

# Mengkonversi gambar kiri ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kanan ke grayscale
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Mendefinisikan jumlah disparity (kelipatan 16)
num_disp = 64

# Mendefinisikan ukuran blok matching
block_size = 7

# Menghitung parameter P1 untuk SGBM (penalty smoothness kecil)
P1 = 8 * 3 * block_size ** 2

# Menghitung parameter P2 untuk SGBM (penalty smoothness besar)
P2 = 32 * 3 * block_size ** 2

# Membuat objek StereoSGBM untuk matching kiri-ke-kanan
left_matcher = cv2.StereoSGBM_create(
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

# Mencatat waktu mulai komputasi disparity
t_start = time.time()

# Menghitung disparity map mentah (raw)
disp_raw = left_matcher.compute(gray_left, gray_right)

# Mencatat waktu selesai komputasi
t_raw = time.time() - t_start

# Mengkonversi disparity ke float (dibagi 16 karena format fixed-point)
disp_float = disp_raw.astype(np.float32) / 16.0

# Menghitung jumlah piksel valid (disparity > 0)
valid_mask = disp_float > 0

# Menghitung coverage piksel valid
coverage_raw = valid_mask.sum() / (h * w) * 100

# Menampilkan informasi raw disparity
print(f"[RAW] Waktu komputasi: {t_raw*1000:.1f} ms")
print(f"[RAW] Disparity range: [{disp_float.min():.2f}, {disp_float.max():.2f}]")
print(f"[RAW] Coverage: {coverage_raw:.1f}%")

# ============================================================
# 3. Post-processing dengan Median Blur
# ============================================================

# Menampilkan informasi tahap median blur
print("\n--- Post-processing: Median Blur ---")

# Mengkonversi disparity ke uint8 untuk median blur
disp_uint8 = cv2.normalize(disp_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Mencatat waktu mulai median blur
t_start = time.time()

# Menerapkan median blur dengan kernel 5x5
disp_median_5 = cv2.medianBlur(disp_uint8, 5)

# Mencatat waktu selesai
t_median5 = time.time() - t_start

# Menerapkan median blur dengan kernel 7x7 untuk smoothing lebih kuat
disp_median_7 = cv2.medianBlur(disp_uint8, 7)

# Menampilkan informasi hasil median blur
print(f"[MEDIAN 5x5] Waktu: {t_median5*1000:.2f} ms")
print(f"[MEDIAN 5x5] Range: [{disp_median_5.min()}, {disp_median_5.max()}]")
print(f"[MEDIAN 7x7] Range: [{disp_median_7.min()}, {disp_median_7.max()}]")

# ============================================================
# 4. Post-processing dengan Bilateral Filter
# ============================================================

# Menampilkan informasi tahap bilateral filter
print("\n--- Post-processing: Bilateral Filter ---")

# Mencatat waktu mulai bilateral filter
t_start = time.time()

# Menerapkan bilateral filter (d=9, sigmaColor=75, sigmaSpace=75)
disp_bilateral = cv2.bilateralFilter(disp_uint8, d=9, sigmaColor=75, sigmaSpace=75)

# Mencatat waktu selesai bilateral filter
t_bilateral = time.time() - t_start

# Menerapkan bilateral filter dengan parameter lebih kuat
disp_bilateral_strong = cv2.bilateralFilter(disp_uint8, d=15, sigmaColor=100, sigmaSpace=100)

# Menampilkan informasi hasil bilateral filter
print(f"[BILATERAL d=9] Waktu: {t_bilateral*1000:.2f} ms")
print(f"[BILATERAL d=9] Range: [{disp_bilateral.min()}, {disp_bilateral.max()}]")
print(f"[BILATERAL d=15] Range: [{disp_bilateral_strong.min()}, {disp_bilateral_strong.max()}]")

# ============================================================
# 5. Post-processing dengan WLS Filter
# ============================================================

# Menampilkan informasi tahap WLS filter
print("\n--- Post-processing: WLS Filter ---")

# Memeriksa apakah modul ximgproc tersedia
wls_available = hasattr(cv2, 'ximgproc')

# Menampilkan status ketersediaan ximgproc
print(f"[INFO] cv2.ximgproc tersedia: {wls_available}")

# Menginisialisasi variabel hasil WLS
disp_wls = None

# Menginisialisasi waktu WLS
t_wls = 0

if wls_available:
    # Membuat right matcher dari left matcher
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Menghitung disparity kanan (untuk WLS filter)
    disp_right = right_matcher.compute(gray_right, gray_left)

    # Membuat WLS filter dengan left matcher sebagai referensi
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)

    # Mengatur parameter lambda (mengontrol smoothness)
    wls_filter.setLambda(8000)

    # Mengatur parameter sigma (sensitivitas terhadap tepi)
    wls_filter.setSigmaColor(1.5)

    # Mencatat waktu mulai WLS filtering
    t_start = time.time()

    # Menerapkan WLS filter pada disparity kiri menggunakan disparity kanan
    disp_wls_raw = wls_filter.filter(disp_raw, img_left, None, disp_right)

    # Mencatat waktu selesai WLS filtering
    t_wls = time.time() - t_start

    # Mengkonversi hasil WLS ke uint8 untuk visualisasi
    disp_wls = cv2.normalize(disp_wls_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Menampilkan informasi WLS filter
    print(f"[WLS] Lambda: 8000, Sigma: 1.5")
    print(f"[WLS] Waktu: {t_wls*1000:.2f} ms")
    print(f"[WLS] Range: [{disp_wls.min()}, {disp_wls.max()}]")
else:
    # Menampilkan pesan fallback jika ximgproc tidak tersedia
    print("[WARNING] cv2.ximgproc tidak tersedia!")
    print("[INFO] Menggunakan guided bilateral filter sebagai alternatif...")

    # Membuat fallback: guided filter manual menggunakan bilateral + edge
    edges = cv2.Canny(gray_left, 50, 150)

    # Menginversi edge map sebagai bobot
    edge_weight = 255 - edges

    # Mengkonversi ke float untuk operasi
    weight_float = edge_weight.astype(np.float32) / 255.0

    # Menerapkan bilateral filter sebagai dasar
    disp_guided_base = cv2.bilateralFilter(disp_uint8, d=11, sigmaColor=80, sigmaSpace=80)

    # Menggabungkan hasil bilateral dengan edge-awareness
    disp_wls = (disp_guided_base.astype(np.float32) * weight_float +
                disp_uint8.astype(np.float32) * (1 - weight_float))

    # Mengkonversi kembali ke uint8
    disp_wls = np.clip(disp_wls, 0, 255).astype(np.uint8)

    # Menampilkan informasi fallback
    print(f"[FALLBACK] Guided bilateral filter diterapkan")
    print(f"[FALLBACK] Range: [{disp_wls.min()}, {disp_wls.max()}]")

# ============================================================
# 6. Menghitung Metrik Kualitas
# ============================================================

# Menampilkan informasi tahap metrik
print("\n--- Metrik Kualitas Filtering ---")


# Mendefinisikan fungsi untuk menghitung coverage (persentase piksel non-zero)
def calc_coverage(disp_img):
    """Menghitung persentase piksel dengan disparity valid."""
    # Menghitung jumlah piksel dengan nilai > 0
    valid = np.sum(disp_img > 0)
    # Menghitung total piksel
    total = disp_img.shape[0] * disp_img.shape[1]
    # Mengembalikan persentase coverage
    return valid / total * 100


# Mendefinisikan fungsi untuk menghitung smoothness (rata-rata gradien)
def calc_smoothness(disp_img):
    """Menghitung rata-rata gradien sebagai ukuran smoothness."""
    # Menghitung gradien horizontal menggunakan Sobel
    grad_x = cv2.Sobel(disp_img, cv2.CV_64F, 1, 0, ksize=3)
    # Menghitung gradien vertikal menggunakan Sobel
    grad_y = cv2.Sobel(disp_img, cv2.CV_64F, 0, 1, ksize=3)
    # Menghitung magnitude gradien
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Mengembalikan rata-rata magnitude (semakin kecil semakin halus)
    return magnitude.mean()


# Mengumpulkan semua metode dan hasilnya dalam dictionary
methods = {
    "Raw (SGBM)": disp_uint8,
    "Median 5x5": disp_median_5,
    "Median 7x7": disp_median_7,
    "Bilateral d=9": disp_bilateral,
    "Bilateral d=15": disp_bilateral_strong,
    "WLS Filter": disp_wls
}

# Menampilkan header tabel metrik
print(f"\n{'Metode':<20} {'Coverage(%)':<15} {'Smoothness':<15}")
print("-" * 50)

# Menginisialisasi list untuk menyimpan metrik
coverage_list = []
smoothness_list = []
method_names = []

# Menghitung dan menampilkan metrik untuk setiap metode
for name, disp_img in methods.items():
    if disp_img is not None:
        # Menghitung coverage untuk metode ini
        cov = calc_coverage(disp_img)
        # Menghitung smoothness untuk metode ini
        smooth = calc_smoothness(disp_img)
        # Menampilkan metrik
        print(f"{name:<20} {cov:<15.1f} {smooth:<15.2f}")
        # Menyimpan metrik ke list
        coverage_list.append(cov)
        smoothness_list.append(smooth)
        method_names.append(name)

# ============================================================
# 7. Visualisasi dan Perbandingan
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi Perbandingan ---")

# Membuat figure utama untuk perbandingan semua metode
fig1, axes = plt.subplots(2, 4, figsize=(18, 9))

# Menampilkan gambar kiri asli pada subplot pertama
axes[0, 0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Kiri (Asli)", fontsize=10)
axes[0, 0].axis('off')

# Menampilkan gambar kanan asli pada subplot kedua
axes[0, 1].imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Gambar Kanan (Asli)", fontsize=10)
axes[0, 1].axis('off')

# Menampilkan raw disparity dengan colormap
im_raw = axes[0, 2].imshow(disp_uint8, cmap='jet')
axes[0, 2].set_title("Raw SGBM Disparity", fontsize=10)
axes[0, 2].axis('off')

# Menampilkan hasil median blur 5x5
axes[0, 3].imshow(disp_median_5, cmap='jet')
axes[0, 3].set_title("Median Blur 5x5", fontsize=10)
axes[0, 3].axis('off')

# Menampilkan hasil median blur 7x7
axes[1, 0].imshow(disp_median_7, cmap='jet')
axes[1, 0].set_title("Median Blur 7x7", fontsize=10)
axes[1, 0].axis('off')

# Menampilkan hasil bilateral filter d=9
axes[1, 1].imshow(disp_bilateral, cmap='jet')
axes[1, 1].set_title("Bilateral d=9", fontsize=10)
axes[1, 1].axis('off')

# Menampilkan hasil bilateral filter d=15
axes[1, 2].imshow(disp_bilateral_strong, cmap='jet')
axes[1, 2].set_title("Bilateral d=15", fontsize=10)
axes[1, 2].axis('off')

# Menampilkan hasil WLS filter atau fallback
if disp_wls is not None:
    # Menampilkan hasil WLS/fallback filter
    axes[1, 3].imshow(disp_wls, cmap='jet')
    # Menentukan label berdasarkan ketersediaan ximgproc
    wls_label = "WLS Filter" if wls_available else "Guided Bilateral (Fallback)"
    axes[1, 3].set_title(wls_label, fontsize=10)
    axes[1, 3].axis('off')
else:
    # Menyembunyikan subplot jika WLS tidak tersedia
    axes[1, 3].axis('off')

# Mengatur layout agar tidak bertumpuk
plt.suptitle("Perbandingan Metode Post-Processing Disparity", fontsize=14, fontweight='bold')
plt.tight_layout()

# Mendefinisikan path output untuk figure perbandingan
path_comparison = os.path.join(OUTPUT_DIR, "13_perbandingan_filter_disparity.png")

# Menyimpan figure perbandingan
fig1.savefig(path_comparison, dpi=150, bbox_inches='tight')
print(f"[SAVE] Perbandingan filter: {path_comparison}")

# Membuat figure kedua untuk metrik kualitas (bar chart)
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Mendefinisikan warna untuk setiap metode
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']

# Membuat bar chart untuk coverage
bars1 = ax1.bar(range(len(method_names)), coverage_list,
                color=colors[:len(method_names)], edgecolor='black', linewidth=0.5)

# Mengatur label sumbu x
ax1.set_xticks(range(len(method_names)))
ax1.set_xticklabels(method_names, rotation=30, ha='right', fontsize=8)

# Mengatur judul dan label sumbu y
ax1.set_title("Coverage (% Piksel Valid)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Coverage (%)")

# Menambahkan nilai di atas setiap bar
for bar, val in zip(bars1, coverage_list):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha='center', va='bottom', fontsize=8)

# Membuat bar chart untuk smoothness
bars2 = ax2.bar(range(len(method_names)), smoothness_list,
                color=colors[:len(method_names)], edgecolor='black', linewidth=0.5)

# Mengatur label sumbu x
ax2.set_xticks(range(len(method_names)))
ax2.set_xticklabels(method_names, rotation=30, ha='right', fontsize=8)

# Mengatur judul dan label sumbu y
ax2.set_title("Smoothness (Rata-rata Gradien)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Gradien (lebih rendah = lebih halus)")

# Menambahkan nilai di atas setiap bar
for bar, val in zip(bars2, smoothness_list):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f"{val:.1f}", ha='center', va='bottom', fontsize=8)

# Mengatur layout
plt.suptitle("Metrik Kualitas Post-Processing Disparity", fontsize=14, fontweight='bold')
plt.tight_layout()

# Mendefinisikan path output untuk figure metrik
path_metrics = os.path.join(OUTPUT_DIR, "13_metrik_filter_disparity.png")

# Menyimpan figure metrik
fig2.savefig(path_metrics, dpi=150, bbox_inches='tight')
print(f"[SAVE] Metrik filter: {path_metrics}")

# Menampilkan semua figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 13: WLS FILTER POST-PROCESSING DISPARITY")
print("=" * 60)
print("1. Disparity map mentah (raw) sering memiliki noise & holes")
print("2. Median blur efektif menghilangkan salt-and-pepper noise")
print("3. Bilateral filter menghaluskan sambil menjaga tepi objek")
print("4. WLS filter memanfaatkan gambar asli sebagai panduan (guided)")
print("5. Coverage mengukur persentase piksel disparity yang valid")
print("6. Smoothness mengukur seberapa halus transisi disparity")
print("7. Filter yang terlalu kuat bisa menghilangkan detail penting")
print("8. WLS filter umumnya memberikan keseimbangan terbaik")
print("9. Kombinasi filter bisa digunakan untuk hasil optimal")
print("10. Post-processing penting sebelum konversi disparity ke depth")
print("=" * 60)
