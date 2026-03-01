"""
==========================================================================
PERCOBAAN 1: HARRIS CORNER DETECTION
==========================================================================
Program ini mempelajari cara mendeteksi sudut (corner) pada gambar
menggunakan metode Harris Corner Detection. Metode ini memanfaatkan
matriks auto-korelasi untuk mengukur perubahan intensitas di sekitar
setiap piksel.

Konsep yang dipelajari:
- Auto-correlation matrix (matriks kedua turunan intensitas)
- Eigenvalue dari matriks auto-korelasi (lambda1, lambda2)
- Response function R = det(M) - k * trace(M)^2
- Pengaruh parameter blockSize, ksize, k, dan threshold
- Heatmap fungsi respons Harris

Fungsi utama yang dipelajari:
- cv2.cornerHarris()   : Menghitung respons Harris corner di setiap piksel
- cv2.dilate()          : Memperbesar area corner agar lebih terlihat
- np.float32()          : Mengkonversi citra ke tipe float32 (syarat cornerHarris)
- cv2.cvtColor()        : Mengkonversi ruang warna citra

Hasil: Visualisasi deteksi sudut Harris dengan variasi parameter
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan deteksi sudut
import cv2

# Mengimpor NumPy untuk operasi array dan matriks numerik
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
print("PERCOBAAN 1: HARRIS CORNER DETECTION")
print("=" * 60)

# ============================================================
# 1. Memuat dan Mempersiapkan Gambar
# ============================================================

# Membaca gambar checkerboard dari file
img_checker = cv2.imread(os.path.join(IMAGE_DIR, "checkerboard.jpg"))

# Membaca gambar bangunan dari file
img_bangunan = cv2.imread(os.path.join(IMAGE_DIR, "bangunan.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_checker is None or img_bangunan is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi ukuran gambar checkerboard
print(f"[INFO] Ukuran checkerboard: {img_checker.shape}")

# Menampilkan informasi ukuran gambar bangunan
print(f"[INFO] Ukuran bangunan: {img_bangunan.shape}")

# Mengkonversi gambar checkerboard ke grayscale
gray_checker = cv2.cvtColor(img_checker, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar bangunan ke grayscale
gray_bangunan = cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2GRAY)

# Mengkonversi grayscale checkerboard ke float32 (syarat cornerHarris)
float_checker = np.float32(gray_checker)

# Mengkonversi grayscale bangunan ke float32 (syarat cornerHarris)
float_bangunan = np.float32(gray_bangunan)

# Menampilkan tipe data hasil konversi
print(f"[INFO] Tipe data setelah konversi: {float_checker.dtype}")

# ============================================================
# 2. Menerapkan Harris Corner Detection Dasar
# ============================================================

# Menerapkan cornerHarris pada checkerboard dengan parameter default
# blockSize=2: ukuran neighbourhood, ksize=3: parameter Sobel, k=0.04: konstanta Harris
harris_checker = cv2.cornerHarris(float_checker, blockSize=2, ksize=3, k=0.04)

# Menerapkan cornerHarris pada bangunan dengan parameter yang sama
harris_bangunan = cv2.cornerHarris(float_bangunan, blockSize=2, ksize=3, k=0.04)

# Menampilkan informasi nilai respons Harris pada checkerboard
print(f"\n[HASIL] Respons Harris checkerboard - min: {harris_checker.min():.6f}, max: {harris_checker.max():.6f}")

# Menampilkan informasi nilai respons Harris pada bangunan
print(f"[HASIL] Respons Harris bangunan - min: {harris_bangunan.min():.6f}, max: {harris_bangunan.max():.6f}")

# Mendilasi hasil Harris agar titik corner lebih terlihat jelas
harris_checker_dilated = cv2.dilate(harris_checker, None)

# Mendilasi hasil Harris bangunan agar titik corner lebih terlihat jelas
harris_bangunan_dilated = cv2.dilate(harris_bangunan, None)

# Membuat salinan gambar checkerboard untuk ditandai
result_checker = img_checker.copy()

# Membuat salinan gambar bangunan untuk ditandai
result_bangunan = img_bangunan.copy()

# Menentukan threshold: piksel dengan R > 0.01 * R_max dianggap corner
threshold_checker = 0.01 * harris_checker_dilated.max()

# Menentukan threshold untuk bangunan
threshold_bangunan = 0.01 * harris_bangunan_dilated.max()

# Menandai corner pada checkerboard dengan warna merah (BGR: 0,0,255)
result_checker[harris_checker_dilated > threshold_checker] = [0, 0, 255]

# Menandai corner pada bangunan dengan warna merah
result_bangunan[harris_bangunan_dilated > threshold_bangunan] = [0, 0, 255]

# Menghitung jumlah corner yang terdeteksi pada checkerboard
corner_count_checker = np.sum(harris_checker_dilated > threshold_checker)

# Menghitung jumlah corner yang terdeteksi pada bangunan
corner_count_bangunan = np.sum(harris_bangunan_dilated > threshold_bangunan)

# Menampilkan jumlah corner
print(f"[HASIL] Jumlah piksel corner checkerboard: {corner_count_checker}")
print(f"[HASIL] Jumlah piksel corner bangunan: {corner_count_bangunan}")

# ============================================================
# 3. Visualisasi Hasil Harris Corner Detection Dasar
# ============================================================

# Membuat figure dengan 2x2 subplot untuk visualisasi
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Menampilkan gambar checkerboard asli pada subplot pertama
axes[0, 0].imshow(cv2.cvtColor(img_checker, cv2.COLOR_BGR2RGB))

# Memberikan judul pada subplot pertama
axes[0, 0].set_title("Checkerboard - Asli", fontsize=12)

# Menonaktifkan sumbu pada subplot pertama
axes[0, 0].axis('off')

# Menampilkan hasil deteksi Harris pada checkerboard
axes[0, 1].imshow(cv2.cvtColor(result_checker, cv2.COLOR_BGR2RGB))

# Memberikan judul pada subplot kedua
axes[0, 1].set_title(f"Harris Corner (corners: {corner_count_checker})", fontsize=12)

# Menonaktifkan sumbu pada subplot kedua
axes[0, 1].axis('off')

# Menampilkan gambar bangunan asli pada subplot ketiga
axes[1, 0].imshow(cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2RGB))

# Memberikan judul pada subplot ketiga
axes[1, 0].set_title("Bangunan - Asli", fontsize=12)

# Menonaktifkan sumbu pada subplot ketiga
axes[1, 0].axis('off')

# Menampilkan hasil deteksi Harris pada bangunan
axes[1, 1].imshow(cv2.cvtColor(result_bangunan, cv2.COLOR_BGR2RGB))

# Memberikan judul pada subplot keempat
axes[1, 1].set_title(f"Harris Corner (corners: {corner_count_bangunan})", fontsize=12)

# Menonaktifkan sumbu pada subplot keempat
axes[1, 1].axis('off')

# Memberikan judul utama pada figure
fig.suptitle("Percobaan 1: Harris Corner Detection", fontsize=16, fontweight='bold')

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan hasil visualisasi ke file
plt.savefig(os.path.join(OUTPUT_DIR, "01_harris_corner.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan bahwa file telah disimpan
print(f"\n[SAVED] 01_harris_corner.png")

# Menutup figure untuk menghemat memori
plt.close()

# ============================================================
# 4. Variasi Parameter blockSize
# ============================================================

# Mendefinisikan daftar nilai blockSize yang akan diuji
block_sizes = [2, 3, 5, 7]

# Membuat figure 2x4 untuk menampilkan variasi blockSize pada dua gambar
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Melakukan iterasi untuk setiap nilai blockSize
for i, bs in enumerate(block_sizes):
    # Menerapkan Harris pada checkerboard dengan blockSize tertentu
    harris_bs_checker = cv2.cornerHarris(float_checker, blockSize=bs, ksize=3, k=0.04)

    # Mendilasi hasil Harris
    harris_bs_checker = cv2.dilate(harris_bs_checker, None)

    # Membuat salinan gambar untuk ditandai
    result_bs_checker = img_checker.copy()

    # Menentukan threshold
    thresh_bs = 0.01 * harris_bs_checker.max()

    # Menandai corner dengan warna merah
    result_bs_checker[harris_bs_checker > thresh_bs] = [0, 0, 255]

    # Menghitung jumlah piksel corner
    count_bs_checker = np.sum(harris_bs_checker > thresh_bs)

    # Menampilkan hasil pada subplot baris pertama
    axes[0, i].imshow(cv2.cvtColor(result_bs_checker, cv2.COLOR_BGR2RGB))

    # Memberikan judul dengan informasi blockSize dan jumlah corner
    axes[0, i].set_title(f"blockSize={bs}\n({count_bs_checker} px)", fontsize=10)

    # Menonaktifkan sumbu
    axes[0, i].axis('off')

    # Menerapkan Harris pada bangunan dengan blockSize tertentu
    harris_bs_bangunan = cv2.cornerHarris(float_bangunan, blockSize=bs, ksize=3, k=0.04)

    # Mendilasi hasil Harris pada bangunan
    harris_bs_bangunan = cv2.dilate(harris_bs_bangunan, None)

    # Membuat salinan gambar bangunan untuk ditandai
    result_bs_bangunan = img_bangunan.copy()

    # Menentukan threshold untuk bangunan
    thresh_bs_b = 0.01 * harris_bs_bangunan.max()

    # Menandai corner bangunan dengan warna merah
    result_bs_bangunan[harris_bs_bangunan > thresh_bs_b] = [0, 0, 255]

    # Menghitung jumlah piksel corner bangunan
    count_bs_bangunan = np.sum(harris_bs_bangunan > thresh_bs_b)

    # Menampilkan hasil pada subplot baris kedua
    axes[1, i].imshow(cv2.cvtColor(result_bs_bangunan, cv2.COLOR_BGR2RGB))

    # Memberikan judul dengan informasi blockSize dan jumlah corner
    axes[1, i].set_title(f"blockSize={bs}\n({count_bs_bangunan} px)", fontsize=10)

    # Menonaktifkan sumbu
    axes[1, i].axis('off')

# Memberikan label baris pertama
axes[0, 0].set_ylabel("Checkerboard", fontsize=12)

# Memberikan label baris kedua
axes[1, 0].set_ylabel("Bangunan", fontsize=12)

# Memberikan judul utama
fig.suptitle("Variasi Parameter blockSize pada Harris Corner", fontsize=16, fontweight='bold')

# Mengatur layout agar rapi
plt.tight_layout()

# Menyimpan hasil visualisasi variasi blockSize
plt.savefig(os.path.join(OUTPUT_DIR, "01_harris_variasi_blocksize.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print("[SAVED] 01_harris_variasi_blocksize.png")

# Menutup figure
plt.close()

# ============================================================
# 5. Variasi Parameter k (Konstanta Harris)
# ============================================================

# Mendefinisikan daftar nilai k yang akan diuji
k_values = [0.01, 0.04, 0.06, 0.1]

# Menampilkan header variasi k
print(f"\n--- Variasi Parameter k ---")

# Membuat figure untuk menampilkan variasi k
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Melakukan iterasi untuk setiap nilai k
for i, k_val in enumerate(k_values):
    # Menerapkan Harris pada checkerboard dengan k tertentu
    harris_k_checker = cv2.cornerHarris(float_checker, blockSize=2, ksize=3, k=k_val)

    # Mendilasi hasil
    harris_k_checker = cv2.dilate(harris_k_checker, None)

    # Membuat salinan gambar
    result_k_checker = img_checker.copy()

    # Menentukan threshold
    thresh_k = 0.01 * harris_k_checker.max()

    # Menandai corner merah
    result_k_checker[harris_k_checker > thresh_k] = [0, 0, 255]

    # Menghitung jumlah corner
    count_k = np.sum(harris_k_checker > thresh_k)

    # Menampilkan hasil pada subplot baris pertama
    axes[0, i].imshow(cv2.cvtColor(result_k_checker, cv2.COLOR_BGR2RGB))

    # Memberikan judul
    axes[0, i].set_title(f"k={k_val}\n({count_k} px)", fontsize=10)

    # Menonaktifkan sumbu
    axes[0, i].axis('off')

    # Menerapkan Harris pada bangunan dengan k tertentu
    harris_k_bang = cv2.cornerHarris(float_bangunan, blockSize=2, ksize=3, k=k_val)

    # Mendilasi hasil bangunan
    harris_k_bang = cv2.dilate(harris_k_bang, None)

    # Membuat salinan gambar bangunan
    result_k_bang = img_bangunan.copy()

    # Menentukan threshold bangunan
    thresh_k_b = 0.01 * harris_k_bang.max()

    # Menandai corner bangunan
    result_k_bang[harris_k_bang > thresh_k_b] = [0, 0, 255]

    # Menghitung corner bangunan
    count_k_b = np.sum(harris_k_bang > thresh_k_b)

    # Menampilkan hasil bangunan
    axes[1, i].imshow(cv2.cvtColor(result_k_bang, cv2.COLOR_BGR2RGB))

    # Memberikan judul bangunan
    axes[1, i].set_title(f"k={k_val}\n({count_k_b} px)", fontsize=10)

    # Menonaktifkan sumbu
    axes[1, i].axis('off')

    # Menampilkan informasi ke konsol
    print(f"  k={k_val}: checker={count_k} px, bangunan={count_k_b} px")

# Memberikan judul utama figure
fig.suptitle("Variasi Parameter k pada Harris Corner", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi variasi k
plt.savefig(os.path.join(OUTPUT_DIR, "01_harris_variasi_k.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print("[SAVED] 01_harris_variasi_k.png")

# Menutup figure
plt.close()

# ============================================================
# 6. Variasi Threshold dan Penghitungan Corner
# ============================================================

# Mendefinisikan daftar nilai threshold yang akan diuji
thresholds = [0.001, 0.01, 0.05, 0.1]

# Menampilkan header variasi threshold
print(f"\n--- Variasi Threshold ---")

# Menghitung Harris response sekali untuk digunakan berulang
harris_for_thresh = cv2.cornerHarris(float_bangunan, blockSize=2, ksize=3, k=0.04)

# Mendilasi hasil
harris_for_thresh = cv2.dilate(harris_for_thresh, None)

# Mendapatkan nilai maksimum respons
max_response = harris_for_thresh.max()

# Membuat figure untuk variasi threshold
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Menyimpan jumlah corner untuk setiap threshold
corner_counts = []

# Melakukan iterasi untuk setiap threshold
for i, thr in enumerate(thresholds):
    # Menghitung threshold absolut dari nilai relatif
    abs_thresh = thr * max_response

    # Membuat salinan gambar bangunan
    result_thr = img_bangunan.copy()

    # Menandai corner di atas threshold dengan warna merah
    result_thr[harris_for_thresh > abs_thresh] = [0, 0, 255]

    # Menghitung jumlah piksel corner
    count_thr = np.sum(harris_for_thresh > abs_thresh)

    # Menyimpan hitungan corner
    corner_counts.append(count_thr)

    # Menampilkan hasil pada subplot
    axes[i].imshow(cv2.cvtColor(result_thr, cv2.COLOR_BGR2RGB))

    # Memberikan judul dengan info threshold dan jumlah corner
    axes[i].set_title(f"threshold={thr}\n({count_thr} corners)", fontsize=11)

    # Menonaktifkan sumbu
    axes[i].axis('off')

    # Menampilkan informasi ke konsol
    print(f"  threshold={thr}: {count_thr} piksel corner")

# Memberikan judul utama
fig.suptitle("Pengaruh Threshold terhadap Jumlah Corner (Bangunan)", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi variasi threshold
plt.savefig(os.path.join(OUTPUT_DIR, "01_harris_variasi_threshold.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print("[SAVED] 01_harris_variasi_threshold.png")

# Menutup figure
plt.close()

# ============================================================
# 7. Heatmap Fungsi Respons Harris
# ============================================================

# Menghitung respons Harris untuk heatmap (tanpa dilasi)
harris_heatmap_checker = cv2.cornerHarris(float_checker, blockSize=2, ksize=3, k=0.04)

# Menghitung respons Harris untuk heatmap bangunan
harris_heatmap_bangunan = cv2.cornerHarris(float_bangunan, blockSize=2, ksize=3, k=0.04)

# Membuat figure untuk heatmap
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Menampilkan gambar checkerboard asli
axes[0, 0].imshow(cv2.cvtColor(img_checker, cv2.COLOR_BGR2RGB))

# Memberikan judul
axes[0, 0].set_title("Checkerboard - Asli", fontsize=12)

# Menonaktifkan sumbu
axes[0, 0].axis('off')

# Menampilkan heatmap Harris response checkerboard dengan colormap 'jet'
im1 = axes[0, 1].imshow(harris_heatmap_checker, cmap='jet')

# Memberikan judul heatmap
axes[0, 1].set_title("Harris Response (Heatmap)", fontsize=12)

# Menonaktifkan sumbu
axes[0, 1].axis('off')

# Menambahkan colorbar untuk heatmap checkerboard
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Menampilkan heatmap dengan threshold overlay pada checkerboard
axes[0, 2].imshow(cv2.cvtColor(img_checker, cv2.COLOR_BGR2RGB))

# Membuat mask dari piksel yang terdeteksi sebagai corner
mask_checker = harris_heatmap_checker > (0.01 * harris_heatmap_checker.max())

# Membuat overlay merah transparan
overlay_checker = np.zeros_like(img_checker, dtype=np.uint8)

# Mengisi area corner dengan warna merah
overlay_checker[mask_checker] = [255, 0, 0]

# Menampilkan overlay
axes[0, 2].imshow(overlay_checker, alpha=0.5)

# Memberikan judul
axes[0, 2].set_title("Corner Overlay", fontsize=12)

# Menonaktifkan sumbu
axes[0, 2].axis('off')

# Menampilkan gambar bangunan asli
axes[1, 0].imshow(cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2RGB))

# Memberikan judul
axes[1, 0].set_title("Bangunan - Asli", fontsize=12)

# Menonaktifkan sumbu
axes[1, 0].axis('off')

# Menampilkan heatmap Harris response bangunan
im2 = axes[1, 1].imshow(harris_heatmap_bangunan, cmap='jet')

# Memberikan judul heatmap bangunan
axes[1, 1].set_title("Harris Response (Heatmap)", fontsize=12)

# Menonaktifkan sumbu
axes[1, 1].axis('off')

# Menambahkan colorbar untuk heatmap bangunan
plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

# Menampilkan overlay corner pada bangunan
axes[1, 2].imshow(cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2RGB))

# Membuat mask corner bangunan
mask_bangunan = harris_heatmap_bangunan > (0.01 * harris_heatmap_bangunan.max())

# Membuat overlay merah
overlay_bangunan = np.zeros_like(img_bangunan, dtype=np.uint8)

# Mengisi area corner
overlay_bangunan[mask_bangunan] = [255, 0, 0]

# Menampilkan overlay pada bangunan
axes[1, 2].imshow(overlay_bangunan, alpha=0.5)

# Memberikan judul
axes[1, 2].set_title("Corner Overlay", fontsize=12)

# Menonaktifkan sumbu
axes[1, 2].axis('off')

# Memberikan judul utama
fig.suptitle("Heatmap Fungsi Respons Harris Corner", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan heatmap ke file
plt.savefig(os.path.join(OUTPUT_DIR, "01_harris_heatmap.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 01_harris_heatmap.png")

# Menutup figure
plt.close()

# ============================================================
# 8. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 1: HARRIS CORNER DETECTION")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan Harris corner detector
print("1. Harris Corner Detection menggunakan auto-correlation matrix")
print("   M = [[Ix^2, Ix*Iy], [Ix*Iy, Iy^2]] untuk menghitung")
print("   respons R = det(M) - k * trace(M)^2")

# Menampilkan pengaruh parameter blockSize
print("2. Parameter blockSize mengontrol ukuran neighbourhood:")
print("   - blockSize kecil: lebih sensitif, lebih banyak corner")
print("   - blockSize besar: lebih selektif, corner lebih sedikit")

# Menampilkan pengaruh parameter k
print("3. Parameter k mengontrol sensitivitas:")
print("   - k kecil (0.01): lebih banyak corner terdeteksi")
print("   - k besar (0.1): hanya corner yang sangat kuat terdeteksi")

# Menampilkan pengaruh threshold
print("4. Threshold mengontrol selektivitas:")
print("   - Threshold rendah: banyak corner (termasuk noise)")
print("   - Threshold tinggi: hanya corner terkuat")

# Menampilkan daftar file yang disimpan
print("\nFile output yang dihasilkan:")
print("  - 01_harris_corner.png")
print("  - 01_harris_variasi_blocksize.png")
print("  - 01_harris_variasi_k.png")
print("  - 01_harris_variasi_threshold.png")
print("  - 01_harris_heatmap.png")

# Menampilkan garis penutup
print("=" * 60)
