"""
==========================================================================
PERCOBAAN 8: TRANSFER LEARNING - KONSEP DAN DEMONSTRASI
==========================================================================
Program ini mempelajari konsep transfer learning, yaitu teknik menggunakan
pengetahuan (fitur) yang telah dipelajari dari satu tugas untuk diterapkan
pada tugas lain. Dua strategi utama dibahas: feature extraction dan
fine-tuning. Program mendemonstrasikan ekstraksi fitur multi-level
(histogram warna, histogram tepi, tekstur) dan membangun klasifier
sederhana menggunakan KNN untuk mengenali bentuk-bentuk geometris.

Fungsi utama yang dipelajari:
- cv2.calcHist()                        : Menghitung histogram warna
- cv2.Canny()                           : Deteksi tepi untuk fitur edge
- np.histogram()                        : Histogram orientasi/tekstur
- sklearn.neighbors.KNeighborsClassifier: Klasifikasi KNN (dengan fallback manual)
- cv2.resize()                          : Menyesuaikan ukuran gambar
- sklearn.decomposition.PCA             : Reduksi dimensi (dengan fallback manual)

Konsep yang dipelajari:
- Transfer learning: feature extraction vs fine-tuning
- Ekstraksi fitur multi-level: warna, tepi, tekstur
- Klasifikasi dengan K-Nearest Neighbors (KNN)
- Reduksi dimensi dengan PCA untuk visualisasi 2D
- Perbandingan performa dengan jumlah sampel berbeda
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan komputasi numerik
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor glob untuk pencarian file berdasarkan pola
import glob

# Mengimpor matplotlib untuk visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor random untuk pengacakan data
import random

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Mendefinisikan path folder dataset bentuk geometris
DATASET_DIR = os.path.join(IMAGE_DIR, "dataset")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan header percobaan
print("=" * 60)
print("PERCOBAAN 8: TRANSFER LEARNING - KONSEP DAN DEMONSTRASI")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep transfer learning
# ============================================================
print("\n--- 1. Konsep Transfer Learning ---")

# Menjelaskan konsep transfer learning
print("""
  Transfer Learning adalah teknik memanfaatkan pengetahuan (fitur)
  yang telah dipelajari dari satu tugas untuk tugas lain.

  Dua strategi utama:
  1. Feature Extraction:
     - Menggunakan fitur yang sudah diekstrak dari model/metode yang ada
     - Hanya melatih classifier di atasnya
     - Cocok untuk dataset kecil

  2. Fine-Tuning:
     - Menggunakan model pre-trained sebagai titik awal
     - Melatih ulang sebagian/seluruh model pada data baru
     - Cocok untuk dataset menengah-besar

  Dalam percobaan ini, kita demonstrasikan konsep feature extraction
  dengan mengekstrak fitur multi-level dari gambar dan melatih KNN.
""")

# ============================================================
# 2. Memuat dataset bentuk geometris
# ============================================================
print("\n--- 2. Memuat Dataset Bentuk Geometris ---")

# Mendefinisikan daftar kategori bentuk
kategori_list = ["lingkaran", "persegi", "segitiga", "bintang", "elips"]

# Menyiapkan list untuk menyimpan gambar dan label
semua_gambar = []
semua_label = []
semua_nama_kategori = []

# Melakukan iterasi untuk setiap kategori
for idx_kat, kategori in enumerate(kategori_list):
    # Mendefinisikan path folder kategori
    folder_kategori = os.path.join(DATASET_DIR, kategori)

    # Memeriksa apakah folder kategori ada
    if not os.path.exists(folder_kategori):
        # Menampilkan error jika folder tidak ditemukan
        print(f"  [ERROR] Folder dataset '{kategori}' tidak ditemukan!")
        print(f"          Jalankan download_image.py terlebih dahulu.")
        exit()

    # Mencari semua gambar dalam folder kategori
    pola_gambar = os.path.join(folder_kategori, "*.*")
    daftar_file = glob.glob(pola_gambar)

    # Memfilter hanya file gambar
    ekstensi_valid = ('.png', '.jpg', '.jpeg', '.bmp')
    daftar_file = [f for f in daftar_file if f.lower().endswith(ekstensi_valid)]

    # Menampilkan jumlah gambar ditemukan
    print(f"  Kategori '{kategori}': {len(daftar_file)} gambar ditemukan")

    # Memuat setiap gambar dalam kategori
    for path_file in daftar_file:
        # Membaca gambar dari file
        gambar = cv2.imread(path_file)

        # Memeriksa apakah gambar berhasil dimuat
        if gambar is not None:
            # Meresize gambar ke ukuran standar 64x64
            gambar = cv2.resize(gambar, (64, 64))

            # Menambahkan gambar ke list
            semua_gambar.append(gambar)

            # Menambahkan label numerik ke list
            semua_label.append(idx_kat)

            # Menambahkan nama kategori ke list
            semua_nama_kategori.append(kategori)

# Menampilkan total data yang dimuat
print(f"\n  Total gambar dimuat: {len(semua_gambar)}")
print(f"  Jumlah kategori   : {len(kategori_list)}")

# ============================================================
# 3. Ekstraksi fitur multi-level
# ============================================================
print("\n--- 3. Ekstraksi Fitur Multi-Level ---")

def ekstrak_fitur_warna(gambar):
    """
    Mengekstrak fitur histogram warna dari gambar.
    Menghitung histogram untuk setiap channel BGR.
    """
    # Menyiapkan list untuk menyimpan histogram
    fitur = []

    # Menghitung histogram untuk setiap channel (B, G, R)
    for channel in range(3):
        # Menghitung histogram dengan 16 bin untuk channel ini
        hist = cv2.calcHist([gambar], [channel], None, [16], [0, 256])

        # Menormalisasi histogram agar total = 1
        hist = hist.flatten() / (hist.sum() + 1e-7)

        # Menambahkan histogram ke fitur
        fitur.extend(hist)

    # Mengembalikan array fitur warna (48 dimensi = 3 channel x 16 bin)
    return np.array(fitur)


def ekstrak_fitur_tepi(gambar):
    """
    Mengekstrak fitur histogram tepi dari gambar.
    Menggunakan Canny edge detection dan histogram orientasi.
    """
    # Mengkonversi gambar ke grayscale
    gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # Mendeteksi tepi menggunakan Canny
    edges = cv2.Canny(gray, 50, 150)

    # Menghitung gradien arah X menggunakan Sobel
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # Menghitung gradien arah Y menggunakan Sobel
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Menghitung magnitude gradien
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Menghitung orientasi gradien (dalam derajat 0-180)
    orientasi = np.arctan2(grad_y, grad_x) * 180 / np.pi
    orientasi = orientasi % 180

    # Menghitung histogram orientasi (9 bin, seperti HOG)
    hist_orientasi, _ = np.histogram(orientasi[edges > 0], bins=9,
                                     range=(0, 180),
                                     weights=magnitude[edges > 0])

    # Menormalisasi histogram orientasi
    hist_orientasi = hist_orientasi / (hist_orientasi.sum() + 1e-7)

    # Menghitung rasio piksel tepi terhadap total piksel
    rasio_tepi = np.sum(edges > 0) / edges.size

    # Menggabungkan fitur tepi (10 dimensi)
    fitur_tepi = np.concatenate([hist_orientasi, [rasio_tepi]])

    # Mengembalikan array fitur tepi
    return fitur_tepi


def ekstrak_fitur_tekstur(gambar):
    """
    Mengekstrak fitur tekstur sederhana dari gambar.
    Menggunakan statistik intensitas dan frekuensi spasial.
    """
    # Mengkonversi gambar ke grayscale
    gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Menghitung rata-rata intensitas
    mean_val = np.mean(gray) / 255.0

    # Menghitung standar deviasi intensitas
    std_val = np.std(gray) / 255.0

    # Menghitung energi (rata-rata kuadrat intensitas)
    energi = np.mean(gray**2) / (255.0**2)

    # Menghitung respons Laplacian untuk mengukur detail/tekstur
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Menghitung rata-rata absolut Laplacian
    laplacian_mean = np.mean(np.abs(laplacian)) / 255.0

    # Menghitung standar deviasi Laplacian
    laplacian_std = np.std(laplacian) / 255.0

    # Membagi gambar menjadi 4 kuadran dan hitung rata-rata masing-masing
    h, w = gray.shape
    kuadran_means = []

    # Menghitung rata-rata kuadran kiri atas
    kuadran_means.append(np.mean(gray[:h//2, :w//2]) / 255.0)

    # Menghitung rata-rata kuadran kanan atas
    kuadran_means.append(np.mean(gray[:h//2, w//2:]) / 255.0)

    # Menghitung rata-rata kuadran kiri bawah
    kuadran_means.append(np.mean(gray[h//2:, :w//2]) / 255.0)

    # Menghitung rata-rata kuadran kanan bawah
    kuadran_means.append(np.mean(gray[h//2:, w//2:]) / 255.0)

    # Menggabungkan semua fitur tekstur (9 dimensi)
    fitur_tekstur = np.array([mean_val, std_val, energi,
                              laplacian_mean, laplacian_std] + kuadran_means)

    # Mengembalikan array fitur tekstur
    return fitur_tekstur


def ekstrak_semua_fitur(gambar):
    """
    Menggabungkan semua fitur dari ketiga level menjadi satu vektor.
    Total dimensi: 48 (warna) + 10 (tepi) + 9 (tekstur) = 67
    """
    # Mengekstrak fitur warna
    fitur_warna = ekstrak_fitur_warna(gambar)

    # Mengekstrak fitur tepi
    fitur_tepi = ekstrak_fitur_tepi(gambar)

    # Mengekstrak fitur tekstur
    fitur_tekstur = ekstrak_fitur_tekstur(gambar)

    # Menggabungkan semua fitur menjadi satu vektor
    fitur_gabungan = np.concatenate([fitur_warna, fitur_tepi, fitur_tekstur])

    # Mengembalikan vektor fitur gabungan
    return fitur_gabungan


# Menampilkan informasi dimensi fitur
print("  Dimensi fitur per level:")
print("    - Histogram warna : 48 (3 channel x 16 bin)")
print("    - Histogram tepi  : 10 (9 orientasi + 1 rasio)")
print("    - Fitur tekstur   :  9 (5 statistik + 4 kuadran)")
print("    - Total           : 67 dimensi")

# ============================================================
# 4. Mengekstrak fitur dari seluruh dataset
# ============================================================
print("\n--- 4. Mengekstrak Fitur dari Seluruh Dataset ---")

# Menyiapkan array untuk menyimpan semua vektor fitur
semua_fitur = []

# Melakukan iterasi untuk setiap gambar dan mengekstrak fiturnya
for i, gambar in enumerate(semua_gambar):
    # Mengekstrak fitur gabungan dari gambar
    fitur = ekstrak_semua_fitur(gambar)

    # Menambahkan fitur ke list
    semua_fitur.append(fitur)

# Mengkonversi list ke array NumPy
semua_fitur = np.array(semua_fitur)

# Mengkonversi label ke array NumPy
semua_label = np.array(semua_label)

# Menampilkan informasi dataset fitur
print(f"  Shape fitur: {semua_fitur.shape}")
print(f"  Shape label: {semua_label.shape}")

# ============================================================
# 5. Visualisasi fitur yang diekstrak
# ============================================================
print("\n--- 5. Visualisasi Fitur yang Diekstrak ---")

# Membuat figure untuk visualisasi fitur
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Mengambil satu sampel per kategori untuk visualisasi
for idx_kat, kategori in enumerate(kategori_list):
    # Mencari indeks gambar pertama dari kategori ini
    indeks_sampel = np.where(semua_label == idx_kat)[0]

    # Memeriksa apakah ada sampel untuk kategori ini
    if len(indeks_sampel) == 0:
        continue

    # Mengambil fitur sampel pertama
    fitur_sampel = semua_fitur[indeks_sampel[0]]

    # Memplot fitur warna (48 dimensi pertama) di subplot pertama
    if idx_kat < 5:
        # Menentukan posisi subplot
        baris = idx_kat // 3
        kolom = idx_kat % 3

        # Memplot vektor fitur lengkap sebagai bar chart
        axes[baris, kolom].bar(range(len(fitur_sampel)), fitur_sampel,
                               color='steelblue', alpha=0.7, width=1.0)

        # Menambahkan garis pemisah antar level fitur
        axes[baris, kolom].axvline(x=47.5, color='red', linestyle='--',
                                   alpha=0.5, label='Warna|Tepi')
        axes[baris, kolom].axvline(x=57.5, color='green', linestyle='--',
                                   alpha=0.5, label='Tepi|Tekstur')

        # Mengatur judul subplot
        axes[baris, kolom].set_title(f"Fitur: {kategori}", fontsize=11,
                                     fontweight='bold')

        # Mengatur label sumbu
        axes[baris, kolom].set_xlabel("Indeks Fitur", fontsize=9)
        axes[baris, kolom].set_ylabel("Nilai", fontsize=9)

        # Menambahkan legend
        axes[baris, kolom].legend(fontsize=7)

# Menambahkan penjelasan di subplot terakhir
axes[1, 2].text(0.5, 0.5, "Fitur Multi-Level:\n\n"
                "BIRU (0-47): Histogram Warna\n"
                "  3 channel x 16 bin\n\n"
                "MERAH (48-57): Histogram Tepi\n"
                "  9 orientasi + 1 rasio\n\n"
                "HIJAU (58-66): Fitur Tekstur\n"
                "  5 statistik + 4 kuadran",
                ha='center', va='center', fontsize=11,
                transform=axes[1, 2].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
axes[1, 2].axis('off')

# Menambahkan judul utama figure
plt.suptitle("Percobaan 8: Visualisasi Fitur Multi-Level per Kategori",
             fontsize=14, fontweight='bold')

# Mengatur layout agar tidak bertumpuk
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "08_transfer_learning_fitur.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure untuk menghemat memori
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/08_transfer_learning_fitur.png")

# ============================================================
# 6. Membagi data menjadi training dan testing
# ============================================================
print("\n--- 6. Membagi Data Training dan Testing ---")

# Mengacak urutan data dengan seed tetap untuk reproduksibilitas
np.random.seed(42)
indeks_acak = np.random.permutation(len(semua_fitur))

# Mengacak fitur dan label sesuai indeks acak
fitur_acak = semua_fitur[indeks_acak]
label_acak = semua_label[indeks_acak]

# Menentukan rasio pembagian data (70% training, 30% testing)
rasio_train = 0.7
jumlah_train = int(len(fitur_acak) * rasio_train)

# Membagi data menjadi set training
fitur_train = fitur_acak[:jumlah_train]
label_train = label_acak[:jumlah_train]

# Membagi data menjadi set testing
fitur_test = fitur_acak[jumlah_train:]
label_test = label_acak[jumlah_train:]

# Menampilkan informasi pembagian data
print(f"  Data training: {len(fitur_train)} sampel")
print(f"  Data testing : {len(fitur_test)} sampel")

# ============================================================
# 7. Klasifikasi dengan KNN
# ============================================================
print("\n--- 7. Klasifikasi dengan K-Nearest Neighbors ---")

# Mencoba mengimpor KNN dari scikit-learn
try:
    # Mengimpor KNeighborsClassifier dari sklearn
    from sklearn.neighbors import KNeighborsClassifier

    # Menampilkan pesan bahwa sklearn tersedia
    print("  Menggunakan sklearn.neighbors.KNeighborsClassifier")

    # Membuat model KNN dengan k=3
    knn = KNeighborsClassifier(n_neighbors=3)

    # Melatih model KNN pada data training
    knn.fit(fitur_train, label_train)

    # Memprediksi label data testing
    prediksi = knn.predict(fitur_test)

    # Flag untuk menandai penggunaan sklearn
    sklearn_tersedia = True

except ImportError:
    # Menampilkan pesan bahwa sklearn tidak tersedia
    print("  sklearn tidak tersedia, menggunakan implementasi KNN manual")

    # Flag untuk menandai penggunaan manual
    sklearn_tersedia = False

    def knn_manual(fitur_train, label_train, fitur_test, k=3):
        """
        Implementasi KNN manual tanpa sklearn.
        Menggunakan jarak Euclidean untuk mencari k tetangga terdekat.
        """
        # Menyiapkan list untuk prediksi
        prediksi = []

        # Melakukan iterasi untuk setiap data test
        for fitur_q in fitur_test:
            # Menghitung jarak Euclidean ke semua data training
            jarak = np.sqrt(np.sum((fitur_train - fitur_q)**2, axis=1))

            # Mengurutkan indeks berdasarkan jarak terdekat
            indeks_terurut = np.argsort(jarak)

            # Mengambil k label tetangga terdekat
            label_terdekat = label_train[indeks_terurut[:k]]

            # Menentukan label mayoritas (voting)
            label_unik, jumlah = np.unique(label_terdekat, return_counts=True)
            label_prediksi = label_unik[np.argmax(jumlah)]

            # Menambahkan prediksi ke list
            prediksi.append(label_prediksi)

        # Mengembalikan array prediksi
        return np.array(prediksi)

    # Memprediksi label data testing dengan KNN manual
    prediksi = knn_manual(fitur_train, label_train, fitur_test, k=3)

# Menghitung akurasi klasifikasi
akurasi = np.mean(prediksi == label_test) * 100

# Menampilkan hasil akurasi
print(f"\n  Akurasi klasifikasi: {akurasi:.1f}%")
print(f"  Benar: {np.sum(prediksi == label_test)}/{len(label_test)}")

# Menghitung akurasi per kategori
print("\n  Akurasi per kategori:")
for idx_kat, kategori in enumerate(kategori_list):
    # Mencari indeks data test untuk kategori ini
    mask = label_test == idx_kat

    # Memeriksa apakah ada data untuk kategori ini
    if np.sum(mask) > 0:
        # Menghitung akurasi kategori ini
        akurasi_kat = np.mean(prediksi[mask] == label_test[mask]) * 100
        print(f"    {kategori:12s}: {akurasi_kat:.1f}% "
              f"({np.sum(prediksi[mask] == label_test[mask])}/{np.sum(mask)})")

# ============================================================
# 8. Reduksi dimensi untuk visualisasi 2D (konsep PCA)
# ============================================================
print("\n--- 8. Reduksi Dimensi (PCA) untuk Visualisasi ---")

# Mencoba mengimpor PCA dari scikit-learn
try:
    # Mengimpor PCA dari sklearn
    from sklearn.decomposition import PCA

    # Menampilkan pesan bahwa sklearn PCA tersedia
    print("  Menggunakan sklearn.decomposition.PCA")

    # Membuat objek PCA untuk reduksi ke 2 dimensi
    pca = PCA(n_components=2)

    # Menerapkan PCA pada semua fitur
    fitur_2d = pca.fit_transform(semua_fitur)

    # Menampilkan variance yang dijelaskan
    print(f"  Variance explained: {pca.explained_variance_ratio_}")

except ImportError:
    # Menampilkan pesan bahwa sklearn tidak tersedia
    print("  sklearn tidak tersedia, menggunakan PCA manual sederhana")

    def pca_manual(data, n_komponen=2):
        """
        Implementasi PCA manual sederhana.
        Menggunakan SVD dari numpy untuk dekomposisi.
        """
        # Menghitung rata-rata setiap fitur
        rata_rata = np.mean(data, axis=0)

        # Mengurangi data dengan rata-ratanya (centering)
        data_centered = data - rata_rata

        # Menghitung matriks kovarians
        cov_matrix = np.cov(data_centered, rowvar=False)

        # Menghitung eigenvalue dan eigenvector
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Mengurutkan dari eigenvalue terbesar
        indeks_urut = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, indeks_urut]

        # Mengambil n komponen utama
        komponen_utama = eigenvectors[:, :n_komponen]

        # Memproyeksikan data ke ruang berdimensi rendah
        data_proyeksi = data_centered @ komponen_utama

        # Mengembalikan data yang sudah direduksi
        return data_proyeksi

    # Menerapkan PCA manual pada semua fitur
    fitur_2d = pca_manual(semua_fitur, n_komponen=2)

# Menampilkan informasi hasil reduksi
print(f"  Shape sebelum PCA: {semua_fitur.shape}")
print(f"  Shape setelah PCA: {fitur_2d.shape}")

# ============================================================
# 9. Visualisasi klasifikasi dan reduksi dimensi
# ============================================================
print("\n--- 9. Visualisasi Hasil Klasifikasi ---")

# Membuat figure dengan 2 subplot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Subplot 1: Scatter plot PCA 2D ---
# Mendefinisikan warna untuk setiap kategori
warna_kategori = ['red', 'blue', 'green', 'orange', 'purple']

# Mendefinisikan marker untuk setiap kategori
marker_kategori = ['o', 's', '^', '*', 'D']

# Memplot setiap kategori dengan warna dan marker berbeda
for idx_kat, kategori in enumerate(kategori_list):
    # Mencari indeks data untuk kategori ini
    mask = semua_label == idx_kat

    # Memplot scatter untuk kategori ini
    axes[0].scatter(fitur_2d[mask, 0], fitur_2d[mask, 1],
                    c=warna_kategori[idx_kat],
                    marker=marker_kategori[idx_kat],
                    label=kategori, alpha=0.7, s=60, edgecolors='black',
                    linewidths=0.5)

# Mengatur judul subplot PCA
axes[0].set_title("Visualisasi Fitur 2D (PCA)", fontsize=12, fontweight='bold')

# Mengatur label sumbu X
axes[0].set_xlabel("Komponen Utama 1", fontsize=10)

# Mengatur label sumbu Y
axes[0].set_ylabel("Komponen Utama 2", fontsize=10)

# Menambahkan legend
axes[0].legend(fontsize=9)

# Menambahkan grid
axes[0].grid(True, alpha=0.3)

# --- Subplot 2: Bar chart akurasi per kategori ---
# Menyiapkan data akurasi per kategori
akurasi_per_kat = []
for idx_kat in range(len(kategori_list)):
    # Mencari data test untuk kategori ini
    mask = label_test == idx_kat

    # Menghitung akurasi jika ada data
    if np.sum(mask) > 0:
        akurasi_per_kat.append(np.mean(prediksi[mask] == label_test[mask]) * 100)
    else:
        akurasi_per_kat.append(0)

# Memplot bar chart akurasi
bars = axes[1].bar(kategori_list, akurasi_per_kat,
                   color=warna_kategori, alpha=0.7, edgecolor='black')

# Menambahkan nilai akurasi di atas setiap bar
for bar, akur in zip(bars, akurasi_per_kat):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{akur:.1f}%", ha='center', va='bottom', fontsize=10,
                 fontweight='bold')

# Mengatur judul subplot akurasi
axes[1].set_title(f"Akurasi per Kategori (Total: {akurasi:.1f}%)",
                  fontsize=12, fontweight='bold')

# Mengatur label sumbu Y
axes[1].set_ylabel("Akurasi (%)", fontsize=10)

# Mengatur batas sumbu Y
axes[1].set_ylim(0, 110)

# Menambahkan grid horizontal
axes[1].grid(True, axis='y', alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 8: Transfer Learning - Klasifikasi Bentuk Geometris",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "08_klasifikasi_akurasi.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/08_klasifikasi_akurasi.png")

# ============================================================
# 10. Perbandingan jumlah sampel training
# ============================================================
print("\n--- 10. Perbandingan Few vs Many Samples ---")

# Mendefinisikan variasi jumlah sampel per kategori yang akan diuji
variasi_sampel = [2, 3, 5, 7, 10]

# Menyiapkan list untuk menyimpan hasil akurasi
hasil_akurasi = []

# Melakukan eksperimen untuk setiap variasi jumlah sampel
for n_sampel in variasi_sampel:
    # Menyiapkan data training terbatas
    fitur_train_terbatas = []
    label_train_terbatas = []

    # Mengambil n sampel dari setiap kategori
    for idx_kat in range(len(kategori_list)):
        # Mencari indeks data untuk kategori ini
        indeks_kat = np.where(semua_label == idx_kat)[0]

        # Membatasi jumlah sampel
        n_ambil = min(n_sampel, len(indeks_kat))

        # Mengambil n sampel pertama
        for idx in indeks_kat[:n_ambil]:
            fitur_train_terbatas.append(semua_fitur[idx])
            label_train_terbatas.append(semua_label[idx])

    # Mengkonversi ke array NumPy
    fitur_train_terbatas = np.array(fitur_train_terbatas)
    label_train_terbatas = np.array(label_train_terbatas)

    # Mengklasifikasikan menggunakan KNN
    if sklearn_tersedia:
        # Membuat dan melatih model KNN baru
        knn_temp = KNeighborsClassifier(n_neighbors=min(3, len(fitur_train_terbatas)))
        knn_temp.fit(fitur_train_terbatas, label_train_terbatas)
        pred_temp = knn_temp.predict(fitur_test)
    else:
        # Menggunakan KNN manual
        k_temp = min(3, len(fitur_train_terbatas))
        pred_temp = knn_manual(fitur_train_terbatas, label_train_terbatas,
                               fitur_test, k=k_temp)

    # Menghitung akurasi
    akurasi_temp = np.mean(pred_temp == label_test) * 100

    # Menyimpan hasil
    hasil_akurasi.append(akurasi_temp)

    # Menampilkan hasil
    print(f"  {n_sampel} sampel/kategori -> Akurasi: {akurasi_temp:.1f}%")

# Menampilkan kesimpulan
print(f"\n  Kesimpulan: Lebih banyak sampel umumnya meningkatkan akurasi")
print(f"  Few-shot ({variasi_sampel[0]} sampel): {hasil_akurasi[0]:.1f}%")
print(f"  Many-shot ({variasi_sampel[-1]} sampel): {hasil_akurasi[-1]:.1f}%")

# ============================================================
# 11. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 8")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. Transfer learning memanfaatkan fitur dari satu tugas untuk tugas lain
2. Feature extraction menggunakan fitur multi-level (warna, tepi, tekstur)
3. cv2.calcHist() untuk histogram warna, cv2.Canny() untuk fitur tepi
4. KNN (K-Nearest Neighbors) sebagai classifier sederhana
5. PCA mereduksi dimensi fitur untuk visualisasi 2D
6. Jumlah sampel training mempengaruhi akurasi klasifikasi
7. Fitur yang baik membuat kategori terpisah di ruang fitur 2D
8. Konsep ini menjadi dasar transfer learning pada deep learning

Output disimpan di folder: output/
- 08_transfer_learning_fitur.png : Visualisasi fitur multi-level
- 08_klasifikasi_akurasi.png     : Hasil klasifikasi dan PCA
""")
