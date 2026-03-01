"""
==========================================================================
PERCOBAAN 20: PROYEK KLASIFIKASI BENTUK - PIPELINE LENGKAP
==========================================================================
Program ini membangun pipeline klasifikasi bentuk yang lengkap,
menggabungkan semua konsep dari Modul 05 (Deep Learning). Pipeline
mencakup: pemuatan dataset, ekstraksi fitur multi-skala, training
classifier sederhana, evaluasi, augmentasi data, dan visualisasi
hasil secara komprehensif.

Fungsi utama yang dipelajari:
- cv2.calcHist()        : Menghitung histogram warna
- cv2.Canny()           : Deteksi tepi
- cv2.resize()          : Mengubah ukuran gambar
- cv2.flip()            : Membalik gambar (augmentasi)
- cv2.GaussianBlur()    : Menghaluskan gambar (augmentasi)
- KNN / centroid-based  : Klasifikasi sederhana
- Confusion matrix      : Evaluasi per-kelas

Konsep yang dipelajari:
- Pipeline klasifikasi end-to-end
- Ekstraksi fitur: histogram warna, fitur tepi, HOG-like
- Train/test split dan evaluasi model
- Augmentasi data untuk memperbesar dataset
- Perbandingan performa dengan/tanpa augmentasi
- Visualisasi prediksi dan laporan evaluasi
==========================================================================
"""

# Mengimpor OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan komputasi
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor glob untuk pencarian file berdasarkan pola
import glob

# Mengimpor matplotlib untuk visualisasi hasil
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Mendefinisikan path folder dataset
DATASET_DIR = os.path.join(IMAGE_DIR, "dataset")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan header percobaan
print("=" * 60)
print("PERCOBAAN 20: PROYEK KLASIFIKASI BENTUK - PIPELINE LENGKAP")
print("=" * 60)

# ============================================================
# 1. Penjelasan pipeline klasifikasi
# ============================================================
print("\n--- 1. Pipeline Klasifikasi Bentuk ---")

# Menjelaskan pipeline
print("""
  Pipeline Klasifikasi Bentuk Lengkap:

  1. LOAD DATA    : Memuat gambar dari 5 kategori di folder dataset/
  2. PREPROCESS   : Resize, normalisasi, grayscale
  3. FEATURE EXT. : Histogram warna, fitur tepi, HOG-like
  4. SPLIT DATA   : 70% training, 30% testing
  5. TRAIN        : KNN / Centroid-based classifier
  6. EVALUATE     : Accuracy, confusion matrix, per-class metrics
  7. AUGMENT      : Flip, rotate, blur, brightness
  8. RETRAIN      : Training ulang dengan data augmented
  9. COMPARE      : Bandingkan dengan/tanpa augmentasi
  10. REPORT      : Visualisasi dan laporan lengkap

  Kategori bentuk (5 kelas):
  - lingkaran, persegi, segitiga, bintang, segi_enam
""")

# ============================================================
# 2. Memuat dataset
# ============================================================
print("\n--- 2. Memuat Dataset ---")

# Mendefinisikan nama kategori bentuk
kategori_bentuk = ['lingkaran', 'persegi', 'segitiga', 'bintang', 'segi_enam']

# Mendefinisikan ukuran standar gambar
IMG_SIZE = 64

# Menyiapkan list untuk menyimpan data
semua_gambar = []
semua_label = []
semua_nama_file = []


def muat_gambar(path, ukuran=(IMG_SIZE, IMG_SIZE)):
    """
    Memuat dan meresize gambar.
    """
    # Membaca gambar
    img = cv2.imread(path)

    if img is None:
        # Mengembalikan None jika gagal membaca
        return None

    # Meresize gambar ke ukuran standar
    img = cv2.resize(img, ukuran)

    # Mengembalikan gambar
    return img


# Mengecek apakah folder dataset ada
dataset_ada = os.path.exists(DATASET_DIR)

if dataset_ada:
    # Memuat gambar dari setiap kategori
    for idx_kat, kat in enumerate(kategori_bentuk):
        # Mendefinisikan path folder kategori
        folder_kat = os.path.join(DATASET_DIR, kat)

        if os.path.exists(folder_kat):
            # Mencari semua file gambar dalam folder
            pola_file = os.path.join(folder_kat, "*.*")
            daftar_file = glob.glob(pola_file)

            # Menyaring hanya file gambar
            daftar_file = [f for f in daftar_file
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            # Menampilkan jumlah gambar ditemukan
            print(f"  Kategori '{kat}': {len(daftar_file)} gambar ditemukan")

            # Memuat setiap gambar
            for file_path in daftar_file:
                img = muat_gambar(file_path)
                if img is not None:
                    semua_gambar.append(img)
                    semua_label.append(idx_kat)
                    semua_nama_file.append(os.path.basename(file_path))
        else:
            # Folder kategori tidak ditemukan
            print(f"  Kategori '{kat}': folder tidak ditemukan")

# Jika dataset kosong atau tidak ditemukan, tampilkan error
if len(semua_gambar) < 10:
    print("\n[ERROR] Dataset tidak cukup (kurang dari 10 gambar)!")
    print("        Jalankan download_image.py terlebih dahulu untuk mengunduh")
    print("        gambar asli ke folder image/dataset/{kategori}/.")
    exit()

# Mengkonversi list ke array NumPy
semua_gambar = np.array(semua_gambar)
semua_label = np.array(semua_label)

# Menampilkan informasi dataset
print(f"\n  Total gambar : {len(semua_gambar)}")
print(f"  Ukuran gambar: {semua_gambar[0].shape}")
print(f"  Distribusi label:")
for idx_kat, kat in enumerate(kategori_bentuk):
    n = np.sum(semua_label == idx_kat)
    print(f"    {kat:<12}: {n} gambar")

# ============================================================
# 3. Ekstraksi fitur multi-skala
# ============================================================
print("\n--- 3. Ekstraksi Fitur ---")


def ekstrak_histogram_warna(img, bins=16):
    """
    Menghitung histogram warna untuk setiap channel.
    Mengembalikan histogram gabungan sebagai fitur.
    """
    # Menyiapkan list fitur
    fitur = []

    # Menghitung histogram untuk setiap channel BGR
    for ch in range(3):
        # Menghitung histogram channel
        hist = cv2.calcHist([img], [ch], None, [bins], [0, 256])

        # Menormalisasi histogram
        hist = hist.flatten() / (img.shape[0] * img.shape[1])

        # Menambahkan ke list fitur
        fitur.extend(hist)

    # Mengembalikan array fitur
    return np.array(fitur, dtype=np.float32)


def ekstrak_fitur_tepi(img, bins=16):
    """
    Menghitung fitur berbasis deteksi tepi (Canny).
    """
    # Mengkonversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Menerapkan deteksi tepi Canny
    edges = cv2.Canny(gray, 50, 150)

    # Menghitung histogram tepi (distribusi spasial)
    hist = cv2.calcHist([edges], [0], None, [bins], [0, 256])

    # Menormalisasi histogram
    hist = hist.flatten() / (edges.shape[0] * edges.shape[1])

    # Menghitung rata-rata dan std tepi
    mean_edge = np.mean(edges) / 255.0
    std_edge = np.std(edges) / 255.0

    # Menghitung rasio piksel tepi
    rasio_tepi = np.sum(edges > 0) / edges.size

    # Menggabungkan semua fitur tepi
    fitur_tepi = np.concatenate([hist, [mean_edge, std_edge, rasio_tepi]])

    # Mengembalikan fitur tepi
    return fitur_tepi.astype(np.float32)


def ekstrak_hog_sederhana(img, cell_size=8):
    """
    Menghitung fitur HOG-like sederhana menggunakan gradient.
    """
    # Mengkonversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Menghitung gradien x dan y menggunakan Sobel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Menghitung magnitude dan orientasi gradien
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) * 180.0 / np.pi

    # Memastikan orientasi positif (0-360)
    orientation[orientation < 0] += 360

    # Membagi gambar menjadi cells dan menghitung histogram orientasi
    n_bins = 9
    h, w = gray.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size

    # Menyiapkan array fitur HOG
    hog_fitur = []

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            # Mendapatkan region cell
            y_start = cy * cell_size
            y_end = y_start + cell_size
            x_start = cx * cell_size
            x_end = x_start + cell_size

            # Mengambil magnitude dan orientasi untuk cell ini
            mag_cell = magnitude[y_start:y_end, x_start:x_end]
            ori_cell = orientation[y_start:y_end, x_start:x_end]

            # Menghitung histogram orientasi weighted by magnitude
            hist_cell = np.zeros(n_bins)
            for b in range(n_bins):
                # Menghitung range orientasi untuk bin ini
                low = b * (360.0 / n_bins)
                high = (b + 1) * (360.0 / n_bins)
                mask = (ori_cell >= low) & (ori_cell < high)
                hist_cell[b] = np.sum(mag_cell[mask])

            # Menormalisasi histogram cell
            norm = np.sqrt(np.sum(hist_cell ** 2) + 1e-6)
            hist_cell = hist_cell / norm

            # Menambahkan ke fitur
            hog_fitur.extend(hist_cell)

    # Mengembalikan fitur HOG
    return np.array(hog_fitur, dtype=np.float32)


def ekstrak_semua_fitur(img):
    """
    Menggabungkan semua fitur: histogram warna + fitur tepi + HOG.
    """
    # Mengekstrak histogram warna (16 bins * 3 channels = 48)
    f_warna = ekstrak_histogram_warna(img, bins=16)

    # Mengekstrak fitur tepi (16 + 3 = 19)
    f_tepi = ekstrak_fitur_tepi(img, bins=16)

    # Mengekstrak fitur HOG-like
    f_hog = ekstrak_hog_sederhana(img, cell_size=8)

    # Menggabungkan semua fitur
    fitur_gabungan = np.concatenate([f_warna, f_tepi, f_hog])

    # Mengembalikan fitur gabungan
    return fitur_gabungan


# Mengekstrak fitur untuk semua gambar
print("  Mengekstrak fitur untuk semua gambar...")

# Menyiapkan list fitur
semua_fitur = []

for i in range(len(semua_gambar)):
    # Mengekstrak fitur untuk gambar ke-i
    fitur = ekstrak_semua_fitur(semua_gambar[i])

    # Menyimpan fitur
    semua_fitur.append(fitur)

# Mengkonversi list ke array
semua_fitur = np.array(semua_fitur)

# Menampilkan informasi fitur
print(f"  Dimensi fitur per gambar: {semua_fitur.shape[1]}")
print(f"  Total fitur array      : {semua_fitur.shape}")

# ============================================================
# 4. Split data: training dan testing
# ============================================================
print("\n--- 4. Split Data Training/Testing ---")

# Mengatur random seed
np.random.seed(42)

# Mendefinisikan rasio split
rasio_train = 0.7

# Membuat indeks acak
indeks = np.arange(len(semua_fitur))
np.random.shuffle(indeks)

# Menghitung titik split
titik_split = int(rasio_train * len(indeks))

# Memisahkan indeks training dan testing
idx_train = indeks[:titik_split]
idx_test = indeks[titik_split:]

# Memisahkan fitur dan label
X_train = semua_fitur[idx_train]
y_train = semua_label[idx_train]
X_test = semua_fitur[idx_test]
y_test = semua_label[idx_test]
img_test = semua_gambar[idx_test]

# Menampilkan informasi split
print(f"  Training : {len(X_train)} sampel")
print(f"  Testing  : {len(X_test)} sampel")
print(f"  Distribusi training:")
for idx_kat, kat in enumerate(kategori_bentuk):
    n = np.sum(y_train == idx_kat)
    print(f"    {kat:<12}: {n}")

# ============================================================
# 5. Implementasi KNN Classifier
# ============================================================
print("\n--- 5. Implementasi KNN Classifier ---")


def knn_predict(X_train, y_train, X_test, k=5):
    """
    K-Nearest Neighbors classifier sederhana.
    """
    # Menyiapkan array prediksi
    prediksi = []
    probabilitas = []

    for i in range(len(X_test)):
        # Menghitung jarak Euclidean ke semua data training
        jarak = np.sqrt(np.sum((X_train - X_test[i]) ** 2, axis=1))

        # Mendapatkan indeks K tetangga terdekat
        idx_terdekat = np.argsort(jarak)[:k]

        # Mendapatkan label K tetangga terdekat
        label_terdekat = y_train[idx_terdekat]

        # Menghitung voting (majority voting)
        counts = np.bincount(label_terdekat, minlength=len(kategori_bentuk))

        # Mendapatkan kelas dengan vote terbanyak
        kelas_prediksi = np.argmax(counts)

        # Menghitung probabilitas (proporsi vote)
        prob = counts / k

        # Menyimpan prediksi dan probabilitas
        prediksi.append(kelas_prediksi)
        probabilitas.append(prob)

    # Mengembalikan prediksi dan probabilitas
    return np.array(prediksi), np.array(probabilitas)


# Mendefinisikan nilai K untuk KNN
K = 5

# Melakukan prediksi pada data test
y_pred, y_prob = knn_predict(X_train, y_train, X_test, k=K)

# Menghitung akurasi
akurasi = np.mean(y_pred == y_test) * 100

# Menampilkan hasil
print(f"  KNN (K={K}) Akurasi: {akurasi:.1f}%")

# ============================================================
# 6. Evaluasi detail
# ============================================================
print("\n--- 6. Evaluasi Detail ---")


def hitung_confusion_matrix(y_true, y_pred, n_classes):
    """
    Menghitung confusion matrix.
    """
    # Menginisialisasi matriks
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Mengisi matriks
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Mengembalikan confusion matrix
    return cm


def hitung_metrik_per_kelas(cm):
    """
    Menghitung precision, recall, F1 per kelas.
    """
    # Mendapatkan jumlah kelas
    n_cls = cm.shape[0]

    # Menyiapkan list metrik
    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(n_cls):
        # Menghitung TP, FP, FN
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        # Menghitung precision
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Menghitung recall
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Menghitung F1
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # Menyimpan metrik
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    # Mengembalikan metrik
    return precision_list, recall_list, f1_list


# Menghitung confusion matrix
cm_tanpa_aug = hitung_confusion_matrix(y_test, y_pred, len(kategori_bentuk))

# Menghitung metrik per kelas
prec_list, rec_list, f1_list = hitung_metrik_per_kelas(cm_tanpa_aug)

# Menampilkan tabel metrik
print(f"\n  {'Kelas':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print(f"  {'-'*48}")

for c in range(len(kategori_bentuk)):
    print(f"  {kategori_bentuk[c]:<12} {prec_list[c]:<12.4f} "
          f"{rec_list[c]:<12.4f} {f1_list[c]:<12.4f}")

# Menghitung macro average
macro_prec = np.mean(prec_list)
macro_rec = np.mean(rec_list)
macro_f1 = np.mean(f1_list)
print(f"  {'Macro Avg':<12} {macro_prec:<12.4f} {macro_rec:<12.4f} {macro_f1:<12.4f}")

# ============================================================
# 7. Augmentasi data
# ============================================================
print("\n--- 7. Augmentasi Data ---")


def augmentasi_gambar(img):
    """
    Menerapkan augmentasi acak pada gambar.
    Mengembalikan list gambar augmented.
    """
    # Menyiapkan list hasil augmentasi
    augmented = []

    # Augmentasi 1: Flip horizontal
    img_flip_h = cv2.flip(img, 1)
    augmented.append(img_flip_h)

    # Augmentasi 2: Flip vertikal
    img_flip_v = cv2.flip(img, 0)
    augmented.append(img_flip_v)

    # Augmentasi 3: Rotasi 90 derajat
    img_rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    augmented.append(img_rot90)

    # Augmentasi 4: Gaussian blur
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    augmented.append(img_blur)

    # Augmentasi 5: Brightness adjustment
    brightness = np.random.randint(-30, 30)
    img_bright = np.clip(img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    augmented.append(img_bright)

    # Mengembalikan list gambar augmented
    return augmented


# Melakukan augmentasi pada data training
print("  Melakukan augmentasi pada data training...")

# Menyiapkan list untuk data augmented
gambar_aug = list(semua_gambar[idx_train])
label_aug = list(y_train)

# Menghitung jumlah gambar asli
n_asli = len(gambar_aug)

# Melakukan augmentasi untuk setiap gambar training
for i in range(len(idx_train)):
    # Mengambil gambar training
    img_ori = semua_gambar[idx_train[i]]

    # Menerapkan augmentasi
    imgs_augmented = augmentasi_gambar(img_ori)

    # Menambahkan gambar augmented ke dataset
    for img_aug in imgs_augmented:
        gambar_aug.append(img_aug)
        label_aug.append(y_train[i])

# Mengkonversi ke array
gambar_aug = np.array(gambar_aug)
label_aug = np.array(label_aug)

# Menampilkan informasi augmentasi
print(f"  Gambar asli training  : {n_asli}")
print(f"  Gambar setelah augment: {len(gambar_aug)}")
print(f"  Peningkatan           : {len(gambar_aug)/n_asli:.1f}x")

# Mengekstrak fitur untuk data augmented
print("  Mengekstrak fitur data augmented...")

# Menyiapkan list fitur
fitur_aug = []

for i in range(len(gambar_aug)):
    # Mengekstrak fitur
    f = ekstrak_semua_fitur(gambar_aug[i])
    fitur_aug.append(f)

# Mengkonversi ke array
X_train_aug = np.array(fitur_aug)
y_train_aug = label_aug

# ============================================================
# 8. Training ulang dengan data augmented
# ============================================================
print("\n--- 8. Training dengan Data Augmented ---")

# Melakukan prediksi dengan KNN menggunakan data augmented
y_pred_aug, y_prob_aug = knn_predict(X_train_aug, y_train_aug, X_test, k=K)

# Menghitung akurasi dengan augmentasi
akurasi_aug = np.mean(y_pred_aug == y_test) * 100

# Menghitung confusion matrix dengan augmentasi
cm_dengan_aug = hitung_confusion_matrix(y_test, y_pred_aug, len(kategori_bentuk))

# Menghitung metrik per kelas dengan augmentasi
prec_aug, rec_aug, f1_aug = hitung_metrik_per_kelas(cm_dengan_aug)

# Menampilkan perbandingan
print(f"\n  Perbandingan:")
print(f"  {'Metrik':<20} {'Tanpa Aug':<15} {'Dengan Aug':<15}")
print(f"  {'-'*50}")
print(f"  {'Akurasi':<20} {akurasi:<15.1f} {akurasi_aug:<15.1f}")
print(f"  {'Macro Precision':<20} {macro_prec:<15.4f} {np.mean(prec_aug):<15.4f}")
print(f"  {'Macro Recall':<20} {macro_rec:<15.4f} {np.mean(rec_aug):<15.4f}")
print(f"  {'Macro F1':<20} {macro_f1:<15.4f} {np.mean(f1_aug):<15.4f}")

# ============================================================
# 9. Visualisasi pipeline lengkap (Gambar 1)
# ============================================================
print("\n--- 9. Visualisasi Pipeline Lengkap ---")

# Membuat figure untuk pipeline
fig, axes = plt.subplots(3, 5, figsize=(18, 10))

# Memberikan judul utama
fig.suptitle("Pipeline Klasifikasi Bentuk - Contoh per Kategori",
             fontsize=16, fontweight='bold')

# --- Baris 1: Contoh gambar per kategori ---
for idx_kat in range(len(kategori_bentuk)):
    ax = axes[0, idx_kat]

    # Mencari gambar pertama dari kategori ini
    idx_gambar = np.where(semua_label == idx_kat)[0]
    if len(idx_gambar) > 0:
        img_contoh = semua_gambar[idx_gambar[0]]
        img_rgb = cv2.cvtColor(img_contoh, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')

    # Mengatur judul
    ax.set_title(f"{kategori_bentuk[idx_kat]}", fontsize=11, fontweight='bold')
    ax.axis('off')

# --- Baris 2: Fitur tepi (Canny edge) per kategori ---
for idx_kat in range(len(kategori_bentuk)):
    ax = axes[1, idx_kat]

    # Mencari gambar dari kategori ini
    idx_gambar = np.where(semua_label == idx_kat)[0]
    if len(idx_gambar) > 0:
        img_contoh = semua_gambar[idx_gambar[0]]
        gray = cv2.cvtColor(img_contoh, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        ax.imshow(edges, cmap='gray')
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')

    # Mengatur judul
    ax.set_title(f"Edge: {kategori_bentuk[idx_kat]}", fontsize=10)
    ax.axis('off')

# --- Baris 3: Histogram warna per kategori ---
for idx_kat in range(len(kategori_bentuk)):
    ax = axes[2, idx_kat]

    # Mencari gambar dari kategori ini
    idx_gambar = np.where(semua_label == idx_kat)[0]
    if len(idx_gambar) > 0:
        img_contoh = semua_gambar[idx_gambar[0]]
        # Menghitung histogram per channel
        for ch, warna in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([img_contoh], [ch], None, [32], [0, 256])
            ax.plot(hist, color=warna, linewidth=1.5)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')

    # Mengatur judul
    ax.set_title(f"Hist: {kategori_bentuk[idx_kat]}", fontsize=10)
    ax.set_xlim(0, 32)

    # Mengatur label
    if idx_kat == 0:
        ax.set_ylabel("Count")

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_1 = os.path.join(OUTPUT_DIR, "20_pipeline_lengkap.png")

# Menyimpan figure
plt.savefig(output_path_1, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_1}")

# ============================================================
# 10. Visualisasi confusion matrix (Gambar 2)
# ============================================================
print("\n--- 10. Visualisasi Confusion Matrix ---")

# Membuat figure untuk confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Memberikan judul utama
fig.suptitle("Confusion Matrix: Tanpa vs Dengan Augmentasi",
             fontsize=16, fontweight='bold')

# --- Subplot 1: Tanpa augmentasi ---
ax1 = axes[0]

# Menampilkan heatmap
im1 = ax1.imshow(cm_tanpa_aug, interpolation='nearest', cmap='Blues')
plt.colorbar(im1, ax=ax1)

# Menambahkan angka di setiap sel
for i in range(len(kategori_bentuk)):
    for j in range(len(kategori_bentuk)):
        warna = 'white' if cm_tanpa_aug[i, j] > cm_tanpa_aug.max() / 2 else 'black'
        ax1.text(j, i, str(cm_tanpa_aug[i, j]), ha='center', va='center',
                 color=warna, fontsize=12, fontweight='bold')

# Mengatur label
ax1.set_xticks(range(len(kategori_bentuk)))
ax1.set_yticks(range(len(kategori_bentuk)))
ax1.set_xticklabels(kategori_bentuk, rotation=45, ha='right', fontsize=9)
ax1.set_yticklabels(kategori_bentuk, fontsize=9)
ax1.set_title(f"Tanpa Augmentasi (Acc={akurasi:.1f}%)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Prediksi")
ax1.set_ylabel("Aktual")

# --- Subplot 2: Dengan augmentasi ---
ax2 = axes[1]

# Menampilkan heatmap
im2 = ax2.imshow(cm_dengan_aug, interpolation='nearest', cmap='Greens')
plt.colorbar(im2, ax=ax2)

# Menambahkan angka
for i in range(len(kategori_bentuk)):
    for j in range(len(kategori_bentuk)):
        warna = 'white' if cm_dengan_aug[i, j] > cm_dengan_aug.max() / 2 else 'black'
        ax2.text(j, i, str(cm_dengan_aug[i, j]), ha='center', va='center',
                 color=warna, fontsize=12, fontweight='bold')

# Mengatur label
ax2.set_xticks(range(len(kategori_bentuk)))
ax2.set_yticks(range(len(kategori_bentuk)))
ax2.set_xticklabels(kategori_bentuk, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(kategori_bentuk, fontsize=9)
ax2.set_title(f"Dengan Augmentasi (Acc={akurasi_aug:.1f}%)", fontsize=12,
              fontweight='bold')
ax2.set_xlabel("Prediksi")
ax2.set_ylabel("Aktual")

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_2 = os.path.join(OUTPUT_DIR, "20_confusion_matrix.png")

# Menyimpan figure
plt.savefig(output_path_2, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_2}")

# ============================================================
# 11. Visualisasi prediksi contoh (Gambar 3)
# ============================================================
print("\n--- 11. Visualisasi Prediksi Contoh ---")

# Membuat figure untuk prediksi contoh
n_contoh = min(10, len(X_test))
fig, axes = plt.subplots(2, 5, figsize=(16, 6))

# Memberikan judul utama
fig.suptitle("Contoh Prediksi pada Data Test",
             fontsize=16, fontweight='bold')

# Menampilkan gambar test dengan prediksi
for i in range(n_contoh):
    row = i // 5
    col = i % 5
    ax = axes[row, col]

    # Mengambil gambar test
    img_show = cv2.cvtColor(img_test[i], cv2.COLOR_BGR2RGB)
    ax.imshow(img_show)

    # Mendapatkan label aktual dan prediksi
    aktual = kategori_bentuk[y_test[i]]
    prediksi_label = kategori_bentuk[y_pred_aug[i]]
    prob = y_prob_aug[i, y_pred_aug[i]] if y_prob_aug[i].size > y_pred_aug[i] else 0

    # Menentukan warna border berdasarkan kebenaran prediksi
    benar = y_test[i] == y_pred_aug[i]
    warna_border = 'green' if benar else 'red'
    status = "✓" if benar else "✗"

    # Mengatur judul dengan warna
    ax.set_title(f"{status} Pred: {prediksi_label}\n(Aktual: {aktual})",
                 fontsize=9, fontweight='bold',
                 color='green' if benar else 'red')
    ax.axis('off')

    # Menambahkan border berwarna
    for spine in ax.spines.values():
        spine.set_edgecolor(warna_border)
        spine.set_linewidth(3)
        spine.set_visible(True)

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_3 = os.path.join(OUTPUT_DIR, "20_prediksi_contoh.png")

# Menyimpan figure
plt.savefig(output_path_3, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_3}")

# ============================================================
# 12. Laporan final (Gambar 4)
# ============================================================
print("\n--- 12. Laporan Final ---")

# Membuat figure untuk laporan
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Memberikan judul utama
fig.suptitle("Laporan Final: Proyek Klasifikasi Bentuk",
             fontsize=16, fontweight='bold')

# --- Subplot 1: Perbandingan akurasi per kelas ---
ax1 = axes[0, 0]

# Menghitung akurasi per kelas tanpa augmentasi
akurasi_per_kelas = []
akurasi_per_kelas_aug = []

for c in range(len(kategori_bentuk)):
    # Menghitung akurasi kelas tanpa augmentasi
    mask_kelas = y_test == c
    if np.sum(mask_kelas) > 0:
        acc_c = np.mean(y_pred[mask_kelas] == c) * 100
        acc_c_aug = np.mean(y_pred_aug[mask_kelas] == c) * 100
    else:
        acc_c = 0
        acc_c_aug = 0
    akurasi_per_kelas.append(acc_c)
    akurasi_per_kelas_aug.append(acc_c_aug)

# Mendefinisikan posisi bar
x_pos = np.arange(len(kategori_bentuk))
bar_width = 0.35

# Menggambar bar tanpa augmentasi
ax1.bar(x_pos - bar_width / 2, akurasi_per_kelas, bar_width,
        label='Tanpa Aug', color='steelblue')

# Menggambar bar dengan augmentasi
ax1.bar(x_pos + bar_width / 2, akurasi_per_kelas_aug, bar_width,
        label='Dengan Aug', color='#4CAF50')

# Mengatur label
ax1.set_xticks(x_pos)
ax1.set_xticklabels(kategori_bentuk, rotation=45, ha='right', fontsize=9)
ax1.set_title("Akurasi per Kelas", fontsize=11, fontweight='bold')
ax1.set_ylabel("Akurasi (%)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 110)

# --- Subplot 2: Perbandingan F1-Score ---
ax2 = axes[0, 1]

# Menggambar bar F1 tanpa augmentasi
ax2.bar(x_pos - bar_width / 2, f1_list, bar_width,
        label='Tanpa Aug', color='steelblue')

# Menggambar bar F1 dengan augmentasi
ax2.bar(x_pos + bar_width / 2, f1_aug, bar_width,
        label='Dengan Aug', color='#4CAF50')

# Mengatur label
ax2.set_xticks(x_pos)
ax2.set_xticklabels(kategori_bentuk, rotation=45, ha='right', fontsize=9)
ax2.set_title("F1-Score per Kelas", fontsize=11, fontweight='bold')
ax2.set_ylabel("F1-Score")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.1)

# --- Subplot 3: Distribusi jumlah data ---
ax3 = axes[1, 0]

# Menghitung distribusi data training dan augmented
dist_train = [np.sum(y_train == c) for c in range(len(kategori_bentuk))]
dist_aug = [np.sum(y_train_aug == c) for c in range(len(kategori_bentuk))]
dist_test = [np.sum(y_test == c) for c in range(len(kategori_bentuk))]

# Menggambar grouped bar chart
bar_width_3 = 0.25
ax3.bar(x_pos - bar_width_3, dist_train, bar_width_3,
        label='Train (asli)', color='steelblue')
ax3.bar(x_pos, dist_aug, bar_width_3,
        label='Train (augmented)', color='#4CAF50')
ax3.bar(x_pos + bar_width_3, dist_test, bar_width_3,
        label='Test', color='#FF9800')

# Mengatur label
ax3.set_xticks(x_pos)
ax3.set_xticklabels(kategori_bentuk, rotation=45, ha='right', fontsize=9)
ax3.set_title("Distribusi Data", fontsize=11, fontweight='bold')
ax3.set_ylabel("Jumlah Gambar")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

# --- Subplot 4: Tabel laporan ringkasan ---
ax4 = axes[1, 1]

# Menyembunyikan axes
ax4.axis('off')

# Membuat data tabel laporan
tabel_data = [
    ['Total Gambar', str(len(semua_gambar)), '-'],
    ['Data Training', str(len(X_train)), str(len(X_train_aug))],
    ['Data Testing', str(len(X_test)), str(len(X_test))],
    ['Dimensi Fitur', str(X_train.shape[1]), str(X_train_aug.shape[1])],
    ['Classifier', f'KNN (K={K})', f'KNN (K={K})'],
    ['Akurasi', f'{akurasi:.1f}%', f'{akurasi_aug:.1f}%'],
    ['Macro F1', f'{macro_f1:.4f}', f'{np.mean(f1_aug):.4f}'],
]

# Mendefinisikan header
kolom_header = ['Aspek', 'Tanpa Aug', 'Dengan Aug']

# Menggambar tabel
tabel = ax4.table(cellText=tabel_data, colLabels=kolom_header,
                  loc='center', cellLoc='center')

# Mengatur font
tabel.auto_set_font_size(False)
tabel.set_fontsize(10)

# Mengatur skala
tabel.scale(1.0, 1.8)

# Mewarnai header
for j in range(len(kolom_header)):
    tabel[0, j].set_facecolor('#673AB7')
    tabel[0, j].set_text_props(color='white', fontweight='bold')

# Mengatur judul
ax4.set_title("Laporan Ringkasan", fontsize=11, fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_4 = os.path.join(OUTPUT_DIR, "20_laporan_final.png")

# Menyimpan figure
plt.savefig(output_path_4, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_4}")

# ============================================================
# 13. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 20")
print("=" * 60)
print(f"""
Konsep yang telah dipelajari:
1. Pipeline klasifikasi end-to-end: load -> preprocess -> fitur -> train -> eval
2. Ekstraksi fitur multi-skala: histogram warna, fitur tepi, HOG-like
3. KNN classifier sederhana untuk klasifikasi bentuk
4. Train/test split untuk evaluasi yang fair
5. Confusion matrix dan metrik per-kelas
6. Augmentasi data: flip, rotasi, blur, brightness
7. Augmentasi meningkatkan jumlah data training sebesar {len(gambar_aug)/n_asli:.1f}x
8. Perbandingan performa dengan/tanpa augmentasi

Hasil Klasifikasi:
- Tanpa augmentasi : Akurasi = {akurasi:.1f}%, F1 = {macro_f1:.4f}
- Dengan augmentasi: Akurasi = {akurasi_aug:.1f}%, F1 = {np.mean(f1_aug):.4f}
- Jumlah kelas     : {len(kategori_bentuk)} ({', '.join(kategori_bentuk)})
- Dimensi fitur    : {X_train.shape[1]}

Output disimpan di folder: output/
- 20_pipeline_lengkap.png  : Pipeline dengan contoh per kategori
- 20_confusion_matrix.png  : Confusion matrix tanpa/dengan augmentasi
- 20_prediksi_contoh.png   : Contoh prediksi pada data test
- 20_laporan_final.png     : Laporan ringkasan lengkap
""")
