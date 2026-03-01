"""
==========================================================================
PERCOBAAN 3: FACE RECOGNITION DENGAN LBPH
(Local Binary Pattern Histogram)
==========================================================================
Program ini mempelajari cara mengenali wajah menggunakan metode LBPH
(Local Binary Pattern Histogram). Metode ini bekerja dengan menghitung
pola biner lokal di setiap piksel, lalu membuat histogram sebagai
fitur deskriptif untuk setiap wajah.

Konsep yang dipelajari:
- LBP (Local Binary Pattern): pola biner dari perbandingan piksel
  tengah dengan 8 piksel tetangganya
- Histogram LBP sebagai fitur pembeda antar individu
- Training dan prediksi menggunakan LBPHFaceRecognizer
- Evaluasi akurasi pengenalan wajah

Fungsi utama yang dipelajari:
- Implementasi manual LBP pattern
- cv2.face.LBPHFaceRecognizer_create()  : Membuat recognizer LBPH
- recognizer.train()                     : Melatih model dengan data wajah
- recognizer.predict()                   : Memprediksi identitas wajah
- plt.bar() / plt.hist()                : Visualisasi histogram

Hasil: Visualisasi LBP pattern, histogram, dan hasil recognition
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor glob untuk mencari file dengan pola tertentu
import glob

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 3: FACE RECOGNITION DENGAN LBPH")
print("=" * 60)

# ============================================================
# 1. Implementasi Manual LBP (Local Binary Pattern)
# ============================================================

print("\n[INFO] Menghitung LBP Pattern secara manual...")
print("-" * 50)

# Membaca gambar wajah pertama untuk demonstrasi LBP
img_demo = cv2.imread(os.path.join(IMAGE_DIR, "wajah_single.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_demo is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Mengkonversi gambar ke grayscale (LBP bekerja pada gambar grayscale)
gray_demo = cv2.cvtColor(img_demo, cv2.COLOR_BGR2GRAY)

# Mendefinisikan fungsi untuk menghitung LBP secara manual
def hitung_lbp_manual(gambar):
    """
    Menghitung Local Binary Pattern (LBP) secara manual.
    Untuk setiap piksel, bandingkan dengan 8 tetangga.
    Jika tetangga >= pusat, bit = 1, selain itu bit = 0.
    Hasilnya adalah nilai desimal 0-255 dari 8-bit biner.
    """
    # Mendapatkan tinggi dan lebar gambar
    tinggi, lebar = gambar.shape

    # Membuat array kosong untuk menyimpan hasil LBP
    lbp_image = np.zeros((tinggi - 2, lebar - 2), dtype=np.uint8)

    # Mendefinisikan offset 8 tetangga (searah jarum jam)
    # Urutan: kanan, kanan-bawah, bawah, kiri-bawah, kiri, kiri-atas, atas, kanan-atas
    tetangga = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                (1, 1), (1, 0), (1, -1), (0, -1)]

    # Iterasi setiap piksel (kecuali border)
    for y in range(1, tinggi - 1):
        for x in range(1, lebar - 1):
            # Mengambil nilai piksel pusat
            pusat = gambar[y, x]

            # Menghitung kode biner LBP
            kode_biner = 0
            for bit, (dy, dx) in enumerate(tetangga):
                # Membandingkan tetangga dengan piksel pusat
                if gambar[y + dy, x + dx] >= pusat:
                    # Jika tetangga >= pusat, set bit ke 1
                    kode_biner |= (1 << bit)

            # Menyimpan nilai LBP (0-255) ke dalam gambar hasil
            lbp_image[y - 1, x - 1] = kode_biner

    # Mengembalikan gambar LBP
    return lbp_image

# Menghitung LBP pada gambar demonstrasi
lbp_result = hitung_lbp_manual(gray_demo)

# Menampilkan informasi LBP
print(f"  Ukuran gambar asli: {gray_demo.shape}")
print(f"  Ukuran LBP result: {lbp_result.shape}")
print(f"  Nilai LBP min: {lbp_result.min()}, max: {lbp_result.max()}")
print(f"  Nilai LBP rata-rata: {lbp_result.mean():.2f}")

# ============================================================
# 2. Visualisasi LBP Pattern dengan Detail
# ============================================================

# Mengambil potongan kecil gambar untuk menunjukkan proses LBP
# Mengambil area 5x5 piksel di sekitar tengah gambar
cy, cx = gray_demo.shape[0] // 2, gray_demo.shape[1] // 2
patch = gray_demo[cy-2:cy+3, cx-2:cx+3]

# Membuat figure untuk visualisasi LBP pattern
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Menampilkan gambar asli grayscale
axes[0, 0].imshow(gray_demo, cmap='gray')
axes[0, 0].set_title("Gambar Asli (Grayscale)", fontsize=11)
axes[0, 0].axis("off")

# Menampilkan gambar LBP hasil komputasi manual
axes[0, 1].imshow(lbp_result, cmap='gray')
axes[0, 1].set_title("Gambar LBP (Manual)", fontsize=11)
axes[0, 1].axis("off")

# Menampilkan patch 5x5 dengan nilai piksel untuk menunjukkan proses LBP
axes[0, 2].imshow(patch, cmap='gray', interpolation='nearest')
axes[0, 2].set_title("Patch 5x5 (Demonstrasi LBP)", fontsize=11)
# Menambahkan nilai piksel pada setiap sel
for i in range(5):
    for j in range(5):
        axes[0, 2].text(j, i, str(patch[i, j]), ha='center', va='center',
                        color='red' if (i == 2 and j == 2) else 'yellow', fontsize=10)

# Menghitung histogram LBP
hist_lbp, bins = np.histogram(lbp_result.ravel(), bins=256, range=(0, 256))

# Menampilkan histogram LBP
axes[1, 0].bar(range(256), hist_lbp, width=1, color='steelblue', alpha=0.7)
axes[1, 0].set_title("Histogram LBP (256 bins)", fontsize=11)
axes[1, 0].set_xlabel("Nilai LBP")
axes[1, 0].set_ylabel("Frekuensi")

# Menampilkan histogram LBP yang dinormalisasi
hist_norm = hist_lbp.astype(np.float64) / hist_lbp.sum()
axes[1, 1].bar(range(256), hist_norm, width=1, color='coral', alpha=0.7)
axes[1, 1].set_title("Histogram LBP (Normalized)", fontsize=11)
axes[1, 1].set_xlabel("Nilai LBP")
axes[1, 1].set_ylabel("Probabilitas")

# Menampilkan proses biner LBP untuk satu piksel pusat
# Mengambil piksel pusat patch dan menghitung kode binernya
pusat_val = patch[2, 2]
tetangga_vals = [patch[1, 1], patch[1, 2], patch[1, 3], patch[2, 3],
                 patch[3, 3], patch[3, 2], patch[3, 1], patch[2, 1]]
bit_str = ""
for v in tetangga_vals:
    bit_str += "1" if v >= pusat_val else "0"

# Menghitung nilai desimal dari kode biner
lbp_value = int(bit_str, 2)

# Menampilkan penjelasan proses LBP
info_text = f"Piksel pusat = {pusat_val}\n\nTetangga:\n"
for i, v in enumerate(tetangga_vals):
    info_text += f"  T{i}: {v} {'≥' if v >= pusat_val else '<'} {pusat_val} → {'1' if v >= pusat_val else '0'}\n"
info_text += f"\nBiner: {bit_str}\nDesimal: {lbp_value}"
axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 2].set_title("Proses Komputasi LBP", fontsize=11)
axes[1, 2].axis("off")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 3: Local Binary Pattern (LBP) - Manual Computation",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi LBP pattern
output_path_1 = os.path.join(OUTPUT_DIR, "03_lbp_pattern.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 3. Memuat Dataset Wajah dan Menghitung Histogram LBPH per Orang
# ============================================================

print("\n[INFO] Memuat dataset wajah untuk LBPH Recognition...")
print("-" * 50)

# Mendefinisikan path folder faces
faces_dir = os.path.join(IMAGE_DIR, "faces")

# Mendefinisikan nama-nama orang dan label numerik
nama_orang = ["andi", "budi", "citra"]
label_map = {0: "Andi", 1: "Budi", 2: "Citra"}

# Menyiapkan list untuk menyimpan gambar training dan label
train_images = []
train_labels = []
test_images = []
test_labels = []
test_names = []

# Memuat gambar wajah untuk setiap orang
for label, nama in enumerate(nama_orang):
    # Mendapatkan path folder orang ini
    person_dir = os.path.join(faces_dir, nama)

    # Mencari semua file gambar jpg dalam folder
    foto_list = sorted(glob.glob(os.path.join(person_dir, "*.jpg")))

    # Menampilkan jumlah foto yang ditemukan
    print(f"  {nama}: {len(foto_list)} foto ditemukan")

    # Membagi data: 7 foto untuk training, 2 untuk testing
    for i, foto_path in enumerate(foto_list):
        # Membaca gambar dalam format grayscale
        img_face = cv2.imread(foto_path, cv2.IMREAD_GRAYSCALE)

        # Memeriksa apakah gambar berhasil dimuat
        if img_face is not None:
            # Meresize gambar ke ukuran standar 100x100
            img_resized = cv2.resize(img_face, (100, 100))

            # Membagi ke training atau testing
            if i < 7:
                # Menyimpan ke data training
                train_images.append(img_resized)
                train_labels.append(label)
            else:
                # Menyimpan ke data testing
                test_images.append(img_resized)
                test_labels.append(label)
                test_names.append(nama)

# Menampilkan jumlah data training dan testing
print(f"\n  Total training: {len(train_images)} gambar")
print(f"  Total testing : {len(test_images)} gambar")

# ============================================================
# 4. Menghitung Histogram LBP per Orang dan Visualisasi
# ============================================================

print("\n[INFO] Menghitung histogram LBP per orang...")

# Menyiapkan dictionary untuk menyimpan histogram per orang
histograms_per_person = {nama: [] for nama in nama_orang}

# Menghitung LBP dan histogram untuk setiap gambar training
for img, label in zip(train_images, train_labels):
    # Menghitung LBP pada gambar
    lbp_img = hitung_lbp_manual(img)

    # Menghitung histogram LBP
    hist, _ = np.histogram(lbp_img.ravel(), bins=256, range=(0, 256))

    # Menormalisasi histogram
    hist_normalized = hist.astype(np.float64) / hist.sum()

    # Menyimpan histogram ke dictionary
    histograms_per_person[nama_orang[label]].append(hist_normalized)

# Membuat figure untuk visualisasi histogram per orang
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Mendefinisikan warna untuk setiap orang
warna_orang = ['steelblue', 'coral', 'mediumseagreen']

# Baris 1: Menampilkan contoh gambar wajah dan LBP per orang
for col, nama in enumerate(nama_orang):
    # Mengambil gambar pertama dari training set
    idx = col * 7
    if idx < len(train_images):
        # Menghitung LBP untuk gambar contoh
        lbp_contoh = hitung_lbp_manual(train_images[idx])

        # Membuat gambar gabungan asli dan LBP
        combined = np.hstack([
            cv2.resize(train_images[idx], (100, 100)),
            cv2.resize(lbp_contoh, (100, 100))
        ])

        # Menampilkan gambar gabungan
        axes[0, col].imshow(combined, cmap='gray')
        axes[0, col].set_title(f"{label_map[col]}\nAsli | LBP", fontsize=11)
        axes[0, col].axis("off")

# Baris 2: Menampilkan rata-rata histogram per orang
for col, nama in enumerate(nama_orang):
    # Menghitung rata-rata histogram dari semua gambar orang ini
    hists = histograms_per_person[nama]
    if len(hists) > 0:
        avg_hist = np.mean(hists, axis=0)

        # Menampilkan histogram rata-rata
        axes[1, col].bar(range(256), avg_hist, width=1, color=warna_orang[col], alpha=0.7)
        axes[1, col].set_title(f"Histogram LBP: {label_map[col]}", fontsize=11)
        axes[1, col].set_xlabel("Nilai LBP")
        axes[1, col].set_ylabel("Probabilitas")
        axes[1, col].set_xlim([0, 256])

# Menambahkan judul utama
plt.suptitle("Percobaan 3: Histogram LBPH per Orang\n"
             "Setiap orang memiliki distribusi LBP yang berbeda",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi histogram LBPH
output_path_2 = os.path.join(OUTPUT_DIR, "03_lbph_histogram.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 5. Training dan Prediksi LBPH Face Recognizer
# ============================================================

print("\n[INFO] Training LBPH Face Recognizer...")
print("-" * 50)

# Mengkonversi list label ke numpy array
train_labels_np = np.array(train_labels)

# Mencoba menggunakan cv2.face.LBPHFaceRecognizer (opencv-contrib)
try:
    # Membuat objek LBPH Face Recognizer
    # radius=1: radius lingkaran LBP
    # neighbors=8: jumlah titik sampling pada lingkaran
    # grid_x=8, grid_y=8: ukuran grid untuk pembagian gambar
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )

    # Melatih recognizer dengan data training
    recognizer.train(train_images, train_labels_np)

    print("[INFO] LBPH Recognizer berhasil di-train dengan opencv-contrib!")

    # Menyiapkan list untuk menyimpan hasil prediksi
    prediksi_list = []
    confidence_list = []
    benar = 0

    # Memprediksi setiap gambar testing
    for img_test, label_asli, nama_asli in zip(test_images, test_labels, test_names):
        # Melakukan prediksi menggunakan recognizer
        label_pred, confidence = recognizer.predict(img_test)

        # Menyimpan hasil prediksi
        prediksi_list.append(label_pred)
        confidence_list.append(confidence)

        # Memeriksa apakah prediksi benar
        if label_pred == label_asli:
            benar += 1

        # Menampilkan hasil prediksi
        status = "✓" if label_pred == label_asli else "✗"
        print(f"  {status} Asli: {label_map[label_asli]}, "
              f"Prediksi: {label_map.get(label_pred, '?')}, "
              f"Confidence: {confidence:.2f}")

    # Menghitung akurasi keseluruhan
    akurasi = benar / len(test_labels) * 100 if len(test_labels) > 0 else 0
    print(f"\n  Akurasi: {benar}/{len(test_labels)} = {akurasi:.1f}%")

    # Menandai bahwa opencv-contrib tersedia
    opencv_contrib_available = True

except AttributeError:
    # Jika opencv-contrib tidak terinstal
    print("[WARNING] opencv-contrib-python tidak tersedia!")
    print("[INFO] Menggunakan implementasi manual LBPH recognition...")

    # Menghitung rata-rata histogram sebagai template untuk setiap orang
    templates = {}
    for nama in nama_orang:
        hists = histograms_per_person[nama]
        if len(hists) > 0:
            templates[nama] = np.mean(hists, axis=0)

    # Fungsi untuk menghitung jarak Chi-Square antar histogram
    def chi_square_distance(hist1, hist2):
        """Menghitung jarak Chi-Square antara dua histogram."""
        eps = 1e-10
        return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + eps))

    # Menyiapkan list untuk menyimpan hasil prediksi manual
    prediksi_list = []
    confidence_list = []
    benar = 0

    # Memprediksi setiap gambar testing menggunakan histogram matching
    for img_test, label_asli, nama_asli in zip(test_images, test_labels, test_names):
        # Menghitung LBP pada gambar test
        lbp_test = hitung_lbp_manual(img_test)

        # Menghitung histogram LBP test
        hist_test, _ = np.histogram(lbp_test.ravel(), bins=256, range=(0, 256))
        hist_test = hist_test.astype(np.float64) / hist_test.sum()

        # Mencari jarak minimum ke setiap template
        min_dist = float('inf')
        pred_nama = ""
        for nama, template_hist in templates.items():
            dist = chi_square_distance(hist_test, template_hist)
            if dist < min_dist:
                min_dist = dist
                pred_nama = nama

        # Mengkonversi nama ke label
        label_pred = nama_orang.index(pred_nama)
        prediksi_list.append(label_pred)
        confidence_list.append(min_dist)

        # Memeriksa prediksi
        if label_pred == label_asli:
            benar += 1

        # Menampilkan hasil
        status = "V" if label_pred == label_asli else "X"
        print(f"  {status} Asli: {label_map[label_asli]}, "
              f"Prediksi: {label_map.get(label_pred, '?')}, "
              f"Distance: {min_dist:.4f}")

    # Menghitung akurasi
    akurasi = benar / len(test_labels) * 100 if len(test_labels) > 0 else 0
    print(f"\n  Akurasi: {benar}/{len(test_labels)} = {akurasi:.1f}%")

    # Menandai bahwa opencv-contrib tidak tersedia
    opencv_contrib_available = False

# ============================================================
# 6. Visualisasi Hasil Recognition
# ============================================================

# Menentukan jumlah gambar test yang akan ditampilkan
n_test = len(test_images)
n_cols = min(n_test, 6)
n_rows = max(1, (n_test + n_cols - 1) // n_cols)

# Membuat figure untuk hasil recognition
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows))

# Memastikan axes selalu 2D
if n_test == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes.reshape(1, -1)

# Menampilkan setiap gambar test dengan hasil prediksi
for i in range(n_test):
    # Menentukan posisi subplot
    r = i // n_cols
    c = i % n_cols

    # Menampilkan gambar test
    axes[r, c].imshow(test_images[i], cmap='gray')

    # Menentukan warna border berdasarkan kebenaran prediksi
    benar_pred = prediksi_list[i] == test_labels[i]
    warna_border = 'green' if benar_pred else 'red'

    # Menambahkan judul dengan hasil prediksi
    axes[r, c].set_title(
        f"Asli: {label_map[test_labels[i]]}\n"
        f"Pred: {label_map.get(prediksi_list[i], '?')}\n"
        f"Conf: {confidence_list[i]:.2f}",
        fontsize=9, color=warna_border
    )
    axes[r, c].axis("off")

# Menyembunyikan subplot kosong jika ada
for i in range(n_test, n_rows * n_cols):
    r = i // n_cols
    c = i % n_cols
    axes[r, c].axis("off")

# Menambahkan judul utama dengan akurasi
metode_str = "cv2.face.LBPHFaceRecognizer" if opencv_contrib_available else "Manual LBPH (Chi-Square)"
plt.suptitle(f"Percobaan 3: Hasil Recognition ({metode_str})\n"
             f"Akurasi: {akurasi:.1f}% ({benar}/{len(test_labels)})",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi recognition
output_path_3 = os.path.join(OUTPUT_DIR, "03_recognition_hasil.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 3")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. LBP (Local Binary Pattern)")
print("     - Bandingkan piksel pusat dengan 8 tetangga")
print("     - Jika tetangga >= pusat → bit 1, selain itu → bit 0")
print("     - Hasilnya: nilai 0-255 (8-bit biner)")
print("  2. LBPH (LBP Histogram)")
print("     - Bagi gambar wajah menjadi grid (misalnya 8x8)")
print("     - Hitung histogram LBP per cell")
print("     - Gabungkan semua histogram → feature vector")
print("  3. cv2.face.LBPHFaceRecognizer_create()")
print("     - radius: radius lingkaran sampling (default: 1)")
print("     - neighbors: jumlah titik sampling (default: 8)")
print("     - grid_x, grid_y: ukuran pembagian grid")
print("  4. recognizer.train(images, labels)")
print("     - Melatih model dengan gambar dan label")
print("  5. recognizer.predict(image)")
print("     - Return: (label, confidence)")
print("     - Confidence kecil = lebih mirip (Chi-Square distance)")
print(f"\nHasil Recognition:")
print(f"  Metode: {metode_str}")
print(f"  Akurasi: {akurasi:.1f}%")
print("=" * 60)
