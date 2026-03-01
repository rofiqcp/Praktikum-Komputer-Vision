"""
==========================================================================
PERCOBAAN 2: SHI-TOMASI CORNER DETECTION (GOOD FEATURES TO TRACK)
==========================================================================
Program ini mempelajari cara mendeteksi sudut menggunakan metode
Shi-Tomasi yang merupakan penyempurnaan dari Harris Corner Detection.
Shi-Tomasi menggunakan min(lambda1, lambda2) sebagai fungsi respons
alih-alih R = det(M) - k * trace(M)^2 seperti Harris.

Konsep yang dipelajari:
- Perbedaan respons Shi-Tomasi: min(lambda1, lambda2) > threshold
- Parameter maxCorners: jumlah maksimum corner yang diambil
- Parameter qualityLevel: threshold relatif terhadap respons terkuat
- Parameter minDistance: jarak minimum antar corner
- Perbandingan Harris vs Shi-Tomasi

Fungsi utama yang dipelajari:
- cv2.goodFeaturesToTrack()  : Mendeteksi corner terbaik (Shi-Tomasi)
- cv2.cornerHarris()         : Untuk perbandingan dengan Harris
- cv2.circle()               : Menggambar lingkaran pada lokasi corner

Hasil: Visualisasi corner Shi-Tomasi dengan variasi parameter dan
       perbandingan dengan Harris Corner Detection
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan deteksi fitur
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
print("PERCOBAAN 2: SHI-TOMASI CORNER DETECTION")
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

# Menampilkan informasi ukuran gambar
print(f"[INFO] Ukuran checkerboard: {img_checker.shape}")
print(f"[INFO] Ukuran bangunan: {img_bangunan.shape}")

# Mengkonversi checkerboard ke grayscale
gray_checker = cv2.cvtColor(img_checker, cv2.COLOR_BGR2GRAY)

# Mengkonversi bangunan ke grayscale
gray_bangunan = cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Deteksi Shi-Tomasi Corner Dasar
# ============================================================

# Mendeteksi corner pada checkerboard dengan goodFeaturesToTrack
# maxCorners=100: maksimal 100 corner, qualityLevel=0.01, minDistance=10
corners_checker = cv2.goodFeaturesToTrack(gray_checker, maxCorners=100,
                                          qualityLevel=0.01, minDistance=10)

# Mendeteksi corner pada bangunan dengan parameter yang sama
corners_bangunan = cv2.goodFeaturesToTrack(gray_bangunan, maxCorners=100,
                                           qualityLevel=0.01, minDistance=10)

# Menampilkan jumlah corner yang terdeteksi
print(f"\n[HASIL] Corner checkerboard (Shi-Tomasi): {len(corners_checker)}")
print(f"[HASIL] Corner bangunan (Shi-Tomasi): {len(corners_bangunan)}")

# Membuat salinan gambar checkerboard untuk digambar corner
result_checker = img_checker.copy()

# Membuat salinan gambar bangunan untuk digambar corner
result_bangunan = img_bangunan.copy()

# Menggambar corner pada checkerboard sebagai lingkaran berwarna
for i, corner in enumerate(corners_checker):
    # Mengambil koordinat x, y dari corner
    x, y = corner.ravel().astype(int)

    # Menentukan warna berdasarkan indeks (variasi warna)
    color = (0, 255, 0)  # Hijau

    # Menggambar lingkaran pada posisi corner dengan radius 5
    cv2.circle(result_checker, (x, y), 5, color, -1)

# Menggambar corner pada bangunan sebagai lingkaran berwarna
for i, corner in enumerate(corners_bangunan):
    # Mengambil koordinat x, y dari corner
    x, y = corner.ravel().astype(int)

    # Menggambar lingkaran hijau pada posisi corner
    cv2.circle(result_bangunan, (x, y), 5, (0, 255, 0), -1)

# ============================================================
# 3. Visualisasi Hasil Deteksi Dasar
# ============================================================

# Membuat figure dengan 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Menampilkan gambar checkerboard asli
axes[0, 0].imshow(cv2.cvtColor(img_checker, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Checkerboard - Asli", fontsize=12)
axes[0, 0].axis('off')

# Menampilkan hasil Shi-Tomasi pada checkerboard
axes[0, 1].imshow(cv2.cvtColor(result_checker, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"Shi-Tomasi ({len(corners_checker)} corners)", fontsize=12)
axes[0, 1].axis('off')

# Menampilkan gambar bangunan asli
axes[1, 0].imshow(cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Bangunan - Asli", fontsize=12)
axes[1, 0].axis('off')

# Menampilkan hasil Shi-Tomasi pada bangunan
axes[1, 1].imshow(cv2.cvtColor(result_bangunan, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"Shi-Tomasi ({len(corners_bangunan)} corners)", fontsize=12)
axes[1, 1].axis('off')

# Memberikan judul utama
fig.suptitle("Percobaan 2: Shi-Tomasi Corner Detection", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi
plt.savefig(os.path.join(OUTPUT_DIR, "02_shi_tomasi_corners.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 02_shi_tomasi_corners.png")

# Menutup figure
plt.close()

# ============================================================
# 4. Variasi Parameter maxCorners
# ============================================================

# Mendefinisikan daftar nilai maxCorners untuk pengujian
max_corners_list = [25, 50, 100, 500]

# Menampilkan header variasi maxCorners
print(f"\n--- Variasi maxCorners ---")

# Membuat figure untuk variasi maxCorners
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Melakukan iterasi untuk setiap nilai maxCorners
for i, mc in enumerate(max_corners_list):
    # Mendeteksi corner pada checkerboard dengan maxCorners tertentu
    corners_mc = cv2.goodFeaturesToTrack(gray_checker, maxCorners=mc,
                                         qualityLevel=0.01, minDistance=10)

    # Membuat salinan gambar untuk ditandai
    result_mc = img_checker.copy()

    # Menentukan jumlah corner yang terdeteksi
    n_corners = len(corners_mc) if corners_mc is not None else 0

    # Menggambar setiap corner sebagai lingkaran hijau
    if corners_mc is not None:
        for corner in corners_mc:
            # Mengambil koordinat corner
            x, y = corner.ravel().astype(int)
            # Menggambar lingkaran
            cv2.circle(result_mc, (x, y), 5, (0, 255, 0), -1)

    # Menampilkan pada subplot baris pertama
    axes[0, i].imshow(cv2.cvtColor(result_mc, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"maxCorners={mc}\n({n_corners} detected)", fontsize=10)
    axes[0, i].axis('off')

    # Mendeteksi corner pada bangunan
    corners_mc_b = cv2.goodFeaturesToTrack(gray_bangunan, maxCorners=mc,
                                            qualityLevel=0.01, minDistance=10)

    # Membuat salinan gambar bangunan
    result_mc_b = img_bangunan.copy()

    # Menentukan jumlah corner bangunan
    n_corners_b = len(corners_mc_b) if corners_mc_b is not None else 0

    # Menggambar corner pada bangunan
    if corners_mc_b is not None:
        for corner in corners_mc_b:
            x, y = corner.ravel().astype(int)
            cv2.circle(result_mc_b, (x, y), 5, (0, 255, 0), -1)

    # Menampilkan pada subplot baris kedua
    axes[1, i].imshow(cv2.cvtColor(result_mc_b, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"maxCorners={mc}\n({n_corners_b} detected)", fontsize=10)
    axes[1, i].axis('off')

    # Menampilkan informasi ke konsol
    print(f"  maxCorners={mc}: checker={n_corners}, bangunan={n_corners_b}")

# Memberikan judul utama
fig.suptitle("Variasi Parameter maxCorners", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# ============================================================
# 5. Variasi Parameter qualityLevel
# ============================================================

# Mendefinisikan daftar nilai qualityLevel
quality_levels = [0.001, 0.01, 0.05, 0.1]

# Menampilkan header variasi qualityLevel
print(f"\n--- Variasi qualityLevel ---")

# Membuat figure baru untuk variasi qualityLevel
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))

# Melakukan iterasi untuk setiap nilai qualityLevel
for i, ql in enumerate(quality_levels):
    # Mendeteksi corner pada checkerboard dengan qualityLevel tertentu
    corners_ql = cv2.goodFeaturesToTrack(gray_checker, maxCorners=100,
                                         qualityLevel=ql, minDistance=10)

    # Membuat salinan gambar
    result_ql = img_checker.copy()

    # Menghitung jumlah corner
    n_ql = len(corners_ql) if corners_ql is not None else 0

    # Menggambar corner
    if corners_ql is not None:
        for corner in corners_ql:
            x, y = corner.ravel().astype(int)
            cv2.circle(result_ql, (x, y), 5, (0, 255, 0), -1)

    # Menampilkan pada subplot
    axes2[0, i].imshow(cv2.cvtColor(result_ql, cv2.COLOR_BGR2RGB))
    axes2[0, i].set_title(f"qualityLevel={ql}\n({n_ql} detected)", fontsize=10)
    axes2[0, i].axis('off')

    # Mendeteksi pada bangunan
    corners_ql_b = cv2.goodFeaturesToTrack(gray_bangunan, maxCorners=100,
                                            qualityLevel=ql, minDistance=10)

    # Membuat salinan dan gambar corner bangunan
    result_ql_b = img_bangunan.copy()
    n_ql_b = len(corners_ql_b) if corners_ql_b is not None else 0

    if corners_ql_b is not None:
        for corner in corners_ql_b:
            x, y = corner.ravel().astype(int)
            cv2.circle(result_ql_b, (x, y), 5, (0, 255, 0), -1)

    # Menampilkan hasil bangunan
    axes2[1, i].imshow(cv2.cvtColor(result_ql_b, cv2.COLOR_BGR2RGB))
    axes2[1, i].set_title(f"qualityLevel={ql}\n({n_ql_b} detected)", fontsize=10)
    axes2[1, i].axis('off')

    # Menampilkan ke konsol
    print(f"  qualityLevel={ql}: checker={n_ql}, bangunan={n_ql_b}")

# Memberikan judul utama
fig2.suptitle("Variasi Parameter qualityLevel", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# ============================================================
# 6. Variasi Parameter minDistance
# ============================================================

# Mendefinisikan daftar nilai minDistance
min_distances = [5, 10, 20, 50]

# Menampilkan header variasi minDistance
print(f"\n--- Variasi minDistance ---")

# Membuat figure baru untuk variasi minDistance
fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))

# Melakukan iterasi untuk setiap nilai minDistance
for i, md in enumerate(min_distances):
    # Mendeteksi corner pada checkerboard
    corners_md = cv2.goodFeaturesToTrack(gray_checker, maxCorners=100,
                                         qualityLevel=0.01, minDistance=md)

    # Membuat salinan dan gambar
    result_md = img_checker.copy()
    n_md = len(corners_md) if corners_md is not None else 0

    if corners_md is not None:
        for corner in corners_md:
            x, y = corner.ravel().astype(int)
            cv2.circle(result_md, (x, y), 5, (0, 255, 0), -1)

    # Menampilkan pada subplot
    axes3[0, i].imshow(cv2.cvtColor(result_md, cv2.COLOR_BGR2RGB))
    axes3[0, i].set_title(f"minDistance={md}\n({n_md} detected)", fontsize=10)
    axes3[0, i].axis('off')

    # Mendeteksi pada bangunan
    corners_md_b = cv2.goodFeaturesToTrack(gray_bangunan, maxCorners=100,
                                            qualityLevel=0.01, minDistance=md)

    result_md_b = img_bangunan.copy()
    n_md_b = len(corners_md_b) if corners_md_b is not None else 0

    if corners_md_b is not None:
        for corner in corners_md_b:
            x, y = corner.ravel().astype(int)
            cv2.circle(result_md_b, (x, y), 5, (0, 255, 0), -1)

    # Menampilkan hasil bangunan
    axes3[1, i].imshow(cv2.cvtColor(result_md_b, cv2.COLOR_BGR2RGB))
    axes3[1, i].set_title(f"minDistance={md}\n({n_md_b} detected)", fontsize=10)
    axes3[1, i].axis('off')

    # Menampilkan ke konsol
    print(f"  minDistance={md}: checker={n_md}, bangunan={n_md_b}")

# Memberikan judul utama
fig3.suptitle("Variasi Parameter minDistance", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan semua variasi parameter dalam satu gambar gabungan
fig_all, axes_all = plt.subplots(3, 4, figsize=(20, 15))

# Menampilkan variasi maxCorners pada baris pertama
for i, mc in enumerate(max_corners_list):
    corners_v = cv2.goodFeaturesToTrack(gray_bangunan, maxCorners=mc,
                                        qualityLevel=0.01, minDistance=10)
    result_v = img_bangunan.copy()
    n_v = len(corners_v) if corners_v is not None else 0
    if corners_v is not None:
        for corner in corners_v:
            x, y = corner.ravel().astype(int)
            cv2.circle(result_v, (x, y), 5, (0, 255, 0), -1)
    axes_all[0, i].imshow(cv2.cvtColor(result_v, cv2.COLOR_BGR2RGB))
    axes_all[0, i].set_title(f"maxCorners={mc}\n({n_v})", fontsize=10)
    axes_all[0, i].axis('off')

# Memberikan label baris pertama
axes_all[0, 0].set_ylabel("maxCorners", fontsize=12)

# Menampilkan variasi qualityLevel pada baris kedua
for i, ql in enumerate(quality_levels):
    corners_v = cv2.goodFeaturesToTrack(gray_bangunan, maxCorners=100,
                                        qualityLevel=ql, minDistance=10)
    result_v = img_bangunan.copy()
    n_v = len(corners_v) if corners_v is not None else 0
    if corners_v is not None:
        for corner in corners_v:
            x, y = corner.ravel().astype(int)
            cv2.circle(result_v, (x, y), 5, (0, 255, 0), -1)
    axes_all[1, i].imshow(cv2.cvtColor(result_v, cv2.COLOR_BGR2RGB))
    axes_all[1, i].set_title(f"qualityLevel={ql}\n({n_v})", fontsize=10)
    axes_all[1, i].axis('off')

# Memberikan label baris kedua
axes_all[1, 0].set_ylabel("qualityLevel", fontsize=12)

# Menampilkan variasi minDistance pada baris ketiga
for i, md in enumerate(min_distances):
    corners_v = cv2.goodFeaturesToTrack(gray_bangunan, maxCorners=100,
                                        qualityLevel=0.01, minDistance=md)
    result_v = img_bangunan.copy()
    n_v = len(corners_v) if corners_v is not None else 0
    if corners_v is not None:
        for corner in corners_v:
            x, y = corner.ravel().astype(int)
            cv2.circle(result_v, (x, y), 5, (0, 255, 0), -1)
    axes_all[2, i].imshow(cv2.cvtColor(result_v, cv2.COLOR_BGR2RGB))
    axes_all[2, i].set_title(f"minDistance={md}\n({n_v})", fontsize=10)
    axes_all[2, i].axis('off')

# Memberikan label baris ketiga
axes_all[2, 0].set_ylabel("minDistance", fontsize=12)

# Memberikan judul utama
fig_all.suptitle("Variasi Parameter Shi-Tomasi (Bangunan)", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi variasi parameter
plt.savefig(os.path.join(OUTPUT_DIR, "02_variasi_parameter.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 02_variasi_parameter.png")

# Menutup semua figure
plt.close('all')

# ============================================================
# 7. Perbandingan Harris vs Shi-Tomasi
# ============================================================

# Menampilkan header perbandingan
print(f"\n--- Perbandingan Harris vs Shi-Tomasi ---")

# Mengkonversi ke float32 untuk Harris
float_checker = np.float32(gray_checker)
float_bangunan = np.float32(gray_bangunan)

# Menghitung respons Harris pada bangunan
harris_resp = cv2.cornerHarris(float_bangunan, blockSize=2, ksize=3, k=0.04)

# Mendilasi respons Harris
harris_resp = cv2.dilate(harris_resp, None)

# Menentukan threshold Harris
harris_thresh = 0.01 * harris_resp.max()

# Mendapatkan posisi corner Harris sebagai koordinat (y, x)
harris_yx = np.argwhere(harris_resp > harris_thresh)

# Menampilkan jumlah piksel corner Harris
print(f"  Harris corners (piksel): {len(harris_yx)}")

# Mendeteksi corner Shi-Tomasi pada bangunan
shi_corners = cv2.goodFeaturesToTrack(gray_bangunan, maxCorners=500,
                                      qualityLevel=0.01, minDistance=10)

# Menghitung jumlah corner Shi-Tomasi
n_shi = len(shi_corners) if shi_corners is not None else 0

# Menampilkan jumlah corner Shi-Tomasi
print(f"  Shi-Tomasi corners: {n_shi}")

# Membuat gambar overlay untuk perbandingan
overlay_both = img_bangunan.copy()

# Menandai Harris corners dengan warna merah
overlay_both[harris_resp > harris_thresh] = [0, 0, 255]

# Menandai Shi-Tomasi corners dengan warna hijau (lingkaran)
if shi_corners is not None:
    for corner in shi_corners:
        # Mengambil koordinat
        x, y = corner.ravel().astype(int)
        # Menggambar lingkaran hijau
        cv2.circle(overlay_both, (x, y), 6, (0, 255, 0), 2)

# Membuat gambar terpisah untuk Harris saja
harris_only = img_bangunan.copy()

# Menandai Harris corners dengan merah
harris_only[harris_resp > harris_thresh] = [0, 0, 255]

# Membuat gambar terpisah untuk Shi-Tomasi saja
shi_only = img_bangunan.copy()

# Menggambar corner Shi-Tomasi
if shi_corners is not None:
    for corner in shi_corners:
        x, y = corner.ravel().astype(int)
        cv2.circle(shi_only, (x, y), 5, (0, 255, 0), -1)

# Menghitung overlap: corner yang terdeteksi oleh kedua metode
overlap_count = 0

# Memeriksa setiap corner Shi-Tomasi apakah juga ada Harris corner di dekatnya
if shi_corners is not None:
    for corner in shi_corners:
        # Mengambil koordinat Shi-Tomasi
        cx, cy = corner.ravel().astype(int)

        # Memeriksa apakah area sekitar corner ini juga memiliki respons Harris tinggi
        # Menggunakan radius 5 piksel untuk toleransi
        y_min = max(0, cy - 5)
        y_max = min(harris_resp.shape[0], cy + 5)
        x_min = max(0, cx - 5)
        x_max = min(harris_resp.shape[1], cx + 5)

        # Mengecek apakah ada piksel Harris di area tersebut
        if np.any(harris_resp[y_min:y_max, x_min:x_max] > harris_thresh):
            overlap_count += 1

# Menampilkan jumlah overlap
print(f"  Corner overlap (radius 5px): {overlap_count} dari {n_shi} Shi-Tomasi")

# Menghitung persentase overlap
if n_shi > 0:
    overlap_pct = (overlap_count / n_shi) * 100
else:
    overlap_pct = 0

# Menampilkan persentase overlap
print(f"  Persentase overlap: {overlap_pct:.1f}%")

# Membuat figure untuk perbandingan
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Menampilkan Harris corners saja (merah)
axes[0].imshow(cv2.cvtColor(harris_only, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Harris Corner\n({len(harris_yx)} piksel)", fontsize=12)
axes[0].axis('off')

# Menampilkan Shi-Tomasi corners saja (hijau)
axes[1].imshow(cv2.cvtColor(shi_only, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Shi-Tomasi\n({n_shi} corners)", fontsize=12)
axes[1].axis('off')

# Menampilkan overlay kedua metode
axes[2].imshow(cv2.cvtColor(overlay_both, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Overlay: Harris(merah) + Shi-Tomasi(hijau)\nOverlap: {overlap_count} ({overlap_pct:.1f}%)", fontsize=11)
axes[2].axis('off')

# Memberikan judul utama
fig.suptitle("Perbandingan Harris vs Shi-Tomasi", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi perbandingan
plt.savefig(os.path.join(OUTPUT_DIR, "02_harris_vs_shi_tomasi.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 02_harris_vs_shi_tomasi.png")

# Menutup figure
plt.close()

# ============================================================
# 8. Tabel Perbandingan
# ============================================================

# Menampilkan tabel perbandingan
print(f"\n{'='*65}")
print(f"{'TABEL PERBANDINGAN':^65}")
print(f"{'='*65}")
print(f"{'Kriteria':<25} {'Harris':<20} {'Shi-Tomasi':<20}")
print(f"{'-'*65}")
print(f"{'Fungsi Respons':<25} {'det-k*trace^2':<20} {'min(l1,l2)':<20}")
print(f"{'Corner Count':<25} {len(harris_yx):<20} {n_shi:<20}")
print(f"{'Output Type':<25} {'Pixel map':<20} {'Point list':<20}")
print(f"{'Overlap':<25} {f'{overlap_count} ({overlap_pct:.1f}%)':<20} {'':<20}")
print(f"{'='*65}")

# ============================================================
# 9. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 2: SHI-TOMASI CORNER DETECTION")

# Menampilkan garis pemisah
print("=" * 60)

# Menjelaskan perbedaan respons Shi-Tomasi vs Harris
print("1. Shi-Tomasi menggunakan min(lambda1, lambda2) sebagai respons")
print("   sedangkan Harris menggunakan R = det(M) - k * trace(M)^2")

# Menjelaskan parameter maxCorners
print("2. maxCorners membatasi jumlah corner yang diambil")
print("   (hanya corner terkuat yang dipertahankan)")

# Menjelaskan parameter qualityLevel
print("3. qualityLevel menentukan threshold relatif terhadap corner terkuat")
print("   Semakin kecil -> semakin banyak corner")

# Menjelaskan parameter minDistance
print("4. minDistance mencegah corner terlalu berdekatan")
print("   Semakin besar -> corner lebih tersebar merata")

# Menjelaskan perbandingan
print("5. Shi-Tomasi umumnya memberikan corner lebih stabil dan")
print("   lebih cocok untuk tracking dibanding Harris")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 02_shi_tomasi_corners.png")
print("  - 02_variasi_parameter.png")
print("  - 02_harris_vs_shi_tomasi.png")

# Menampilkan garis penutup
print("=" * 60)
