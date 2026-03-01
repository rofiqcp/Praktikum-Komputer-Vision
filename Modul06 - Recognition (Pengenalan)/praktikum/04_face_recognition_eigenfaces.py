"""
==========================================================================
PERCOBAAN 4: FACE RECOGNITION DENGAN EIGENFACES (PCA)
==========================================================================
Program ini mempelajari cara mengenali wajah menggunakan metode Eigenfaces
yang berbasis Principal Component Analysis (PCA). PCA mereduksi dimensi
data wajah ke ruang fitur yang lebih kecil (eigenspace) sambil
mempertahankan variasi informasi yang paling penting.

Konsep yang dipelajari:
- PCA (Principal Component Analysis): reduksi dimensionalitas
- Mean face: rata-rata semua wajah dalam dataset
- Eigenfaces: eigenvector dari matriks kovarians yang merepresentasikan
  variasi utama pada wajah
- Proyeksi ke eigenspace dan rekonstruksi dari eigenspace
- Nearest-neighbor classification di eigenspace

Fungsi utama yang dipelajari:
- np.mean()                      : Menghitung rata-rata (mean face)
- np.cov()                       : Menghitung matriks kovarians
- np.linalg.eigh()              : Menghitung eigenvalue dan eigenvector
- np.dot()                       : Proyeksi ke eigenspace

Hasil: Visualisasi mean face, eigenfaces, rekonstruksi, dan recognition
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan aljabar linear
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
print("PERCOBAAN 4: FACE RECOGNITION DENGAN EIGENFACES (PCA)")
print("=" * 60)

# ============================================================
# 1. Memuat Dataset Wajah
# ============================================================

print("\n[INFO] Memuat dataset wajah...")
print("-" * 50)

# Mendefinisikan path folder faces
faces_dir = os.path.join(IMAGE_DIR, "faces")

# Mendefinisikan nama-nama orang dan label
nama_orang = ["andi", "budi", "citra"]
label_map = {0: "Andi", 1: "Budi", 2: "Citra"}

# Mendefinisikan ukuran standar gambar wajah
FACE_SIZE = (80, 80)

# Mendefinisikan jumlah piksel per gambar setelah di-flatten
n_pixels = FACE_SIZE[0] * FACE_SIZE[1]

# Menyiapkan list untuk menyimpan data wajah
all_images = []
all_labels = []
all_names = []

# Memuat gambar wajah dari setiap orang
for label, nama in enumerate(nama_orang):
    # Mendapatkan path folder orang ini
    person_dir = os.path.join(faces_dir, nama)

    # Mencari semua file jpg
    foto_list = sorted(glob.glob(os.path.join(person_dir, "*.jpg")))

    # Menampilkan jumlah foto
    print(f"  {nama}: {len(foto_list)} foto")

    # Memuat setiap foto
    for foto_path in foto_list:
        # Membaca gambar dalam format grayscale
        img = cv2.imread(foto_path, cv2.IMREAD_GRAYSCALE)

        # Memeriksa apakah gambar berhasil dimuat
        if img is not None:
            # Meresize gambar ke ukuran standar
            img_resized = cv2.resize(img, FACE_SIZE)

            # Menyimpan gambar, label, dan nama
            all_images.append(img_resized)
            all_labels.append(label)
            all_names.append(nama)

# Memeriksa apakah ada gambar yang dimuat
if len(all_images) == 0:
    print("[ERROR] Tidak ada gambar wajah! Jalankan download_image.py.")
    exit()

# Menampilkan total dataset
print(f"\n  Total gambar: {len(all_images)}")
print(f"  Ukuran per gambar: {FACE_SIZE}")
print(f"  Dimensi vektor: {n_pixels}")

# Membagi data menjadi training dan testing (7:2 per orang)
train_images = []
train_labels = []
test_images = []
test_labels = []

# Memisahkan data training dan testing per orang
for label in range(len(nama_orang)):
    # Mengambil indeks semua gambar orang ini
    indices = [i for i, l in enumerate(all_labels) if l == label]

    # 7 gambar pertama untuk training, sisanya untuk testing
    for i, idx in enumerate(indices):
        if i < 7:
            train_images.append(all_images[idx])
            train_labels.append(all_labels[idx])
        else:
            test_images.append(all_images[idx])
            test_labels.append(all_labels[idx])

# Menampilkan jumlah data training dan testing
print(f"  Training: {len(train_images)} gambar")
print(f"  Testing : {len(test_images)} gambar")

# ============================================================
# 2. Menghitung Mean Face
# ============================================================

print("\n[INFO] Menghitung Mean Face...")

# Mengkonversi list gambar training ke matrix (setiap baris = 1 gambar yang di-flatten)
# Bentuk: (n_samples, n_pixels)
train_matrix = np.array([img.flatten().astype(np.float64) for img in train_images])

# Menghitung mean face (rata-rata semua vektor wajah)
mean_face = np.mean(train_matrix, axis=0)

# Menampilkan informasi mean face
print(f"  Shape matrix training: {train_matrix.shape}")
print(f"  Shape mean face: {mean_face.shape}")
print(f"  Nilai mean face: [{mean_face.min():.1f}, {mean_face.max():.1f}]")

# Mengurangi mean face dari setiap gambar (centering / zero-mean)
# Ini penting karena PCA mencari variasi, bukan nilai absolut
centered_matrix = train_matrix - mean_face

# Menampilkan informasi centered matrix
print(f"  Shape centered matrix: {centered_matrix.shape}")

# ============================================================
# 3. Menghitung PCA (Eigendecomposition)
# ============================================================

print("\n[INFO] Menghitung PCA (eigendecomposition)...")

# Menghitung matriks kovarians
# Trick: jika n_pixels >> n_samples, hitung C = X * X.T (ukuran n_samples x n_samples)
# daripada X.T * X (ukuran n_pixels x n_pixels) yang sangat besar
n_samples = centered_matrix.shape[0]

# Menghitung matriks kovarians kecil (n_samples x n_samples)
cov_small = np.dot(centered_matrix, centered_matrix.T) / (n_samples - 1)

# Menampilkan ukuran matriks kovarians
print(f"  Ukuran matriks kovarians: {cov_small.shape}")

# Menghitung eigenvalue dan eigenvector dari matriks kovarians kecil
eigenvalues_small, eigenvectors_small = np.linalg.eigh(cov_small)

# Mengurutkan eigenvalue dari besar ke kecil
sorted_indices = np.argsort(eigenvalues_small)[::-1]
eigenvalues = eigenvalues_small[sorted_indices]
eigenvectors_small_sorted = eigenvectors_small[:, sorted_indices]

# Mengkonversi eigenvector kecil ke eigenfaces (eigenvector di ruang piksel)
# eigenface_i = X.T * v_i (lalu dinormalisasi)
eigenfaces = np.dot(centered_matrix.T, eigenvectors_small_sorted)

# Menormalisasi setiap eigenface agar memiliki panjang unit
for i in range(eigenfaces.shape[1]):
    norm = np.linalg.norm(eigenfaces[:, i])
    if norm > 0:
        eigenfaces[:, i] /= norm

# Menampilkan informasi eigenvalues
print(f"  Jumlah eigenvalues: {len(eigenvalues)}")
print(f"  Top 5 eigenvalues: {eigenvalues[:5].astype(int)}")

# Menghitung variance explained oleh setiap komponen
total_variance = np.sum(eigenvalues)
variance_explained = eigenvalues / total_variance * 100 if total_variance > 0 else eigenvalues * 0

# Menampilkan variance explained
print(f"  Variance explained (top 5): {variance_explained[:5].round(1)}%")
print(f"  Kumulatif (top 5): {np.cumsum(variance_explained[:5]).round(1)}%")

# ============================================================
# 4. Visualisasi Mean Face dan Eigenfaces
# ============================================================

# Membuat figure untuk menampilkan mean face
fig, axes = plt.subplots(1, 1, figsize=(5, 5))

# Mereshape mean face kembali ke bentuk 2D untuk ditampilkan
mean_face_img = mean_face.reshape(FACE_SIZE)

# Menampilkan mean face
axes.imshow(mean_face_img, cmap='gray')
axes.set_title("Mean Face (Rata-rata Semua Wajah)", fontsize=12)
axes.axis("off")

# Mengatur layout
plt.tight_layout()

# Menyimpan mean face
output_path_1 = os.path.join(OUTPUT_DIR, "04_mean_face.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# Mendefinisikan jumlah eigenfaces yang akan ditampilkan
n_eigenfaces_show = min(8, eigenfaces.shape[1])

# Membuat figure untuk menampilkan eigenfaces
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Menampilkan setiap eigenface
for i in range(n_eigenfaces_show):
    # Mereshape eigenface ke bentuk 2D
    ef = eigenfaces[:, i].reshape(FACE_SIZE)

    # Menormalisasi ke range 0-255 untuk visualisasi
    ef_norm = ((ef - ef.min()) / (ef.max() - ef.min() + 1e-8) * 255).astype(np.uint8)

    # Menampilkan eigenface
    axes[i].imshow(ef_norm, cmap='gray')
    axes[i].set_title(f"Eigenface #{i+1}\nVar: {variance_explained[i]:.1f}%", fontsize=10)
    axes[i].axis("off")

# Menyembunyikan subplot kosong
for i in range(n_eigenfaces_show, 8):
    axes[i].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 4: Top Eigenfaces (Variasi Utama Wajah)\n"
             "Setiap eigenface merepresentasikan satu dimensi variasi",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan eigenfaces
output_path_2 = os.path.join(OUTPUT_DIR, "04_eigenfaces.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 5. Rekonstruksi Wajah dengan Berbagai Jumlah Komponen
# ============================================================

print("\n[INFO] Merekonstruksi wajah dari eigenspace...")

# Memilih gambar pertama dari dataset sebagai contoh
contoh_idx = 0
contoh_vec = centered_matrix[contoh_idx]

# Mendefinisikan jumlah komponen untuk rekonstruksi
komponen_list = [1, 2, 3, 5, 8, 10, 15, n_samples - 1]
# Memfilter komponen yang valid
komponen_list = [k for k in komponen_list if k <= eigenfaces.shape[1]]

# Membuat figure untuk visualisasi rekonstruksi
n_cols_rek = len(komponen_list) + 1
fig, axes = plt.subplots(2, max(n_cols_rek, 4), figsize=(3 * n_cols_rek, 7))

# Menampilkan gambar asli di baris atas kolom pertama
axes[0, 0].imshow(train_images[contoh_idx], cmap='gray')
axes[0, 0].set_title("Asli", fontsize=10)
axes[0, 0].axis("off")

# Merekonstruksi wajah dengan berbagai jumlah komponen
for i, n_comp in enumerate(komponen_list):
    # Mengambil n_comp eigenfaces pertama
    basis = eigenfaces[:, :n_comp]

    # Memproyeksikan wajah ke eigenspace (mendapatkan koefisien)
    koefisien = np.dot(contoh_vec, basis)

    # Merekonstruksi wajah dari eigenspace
    rekonstruksi_vec = np.dot(koefisien, basis.T)

    # Menambahkan mean face kembali
    rekonstruksi_vec += mean_face

    # Mereshape ke bentuk 2D
    rekonstruksi_img = rekonstruksi_vec.reshape(FACE_SIZE)

    # Menclip nilai ke range valid
    rekonstruksi_img = np.clip(rekonstruksi_img, 0, 255).astype(np.uint8)

    # Menghitung error rekonstruksi (MSE)
    mse = np.mean((train_images[contoh_idx].astype(np.float64) - rekonstruksi_img.astype(np.float64)) ** 2)

    # Menampilkan rekonstruksi di baris atas
    col_idx = i + 1
    if col_idx < axes.shape[1]:
        axes[0, col_idx].imshow(rekonstruksi_img, cmap='gray')
        axes[0, col_idx].set_title(f"K={n_comp}\nMSE={mse:.1f}", fontsize=9)
        axes[0, col_idx].axis("off")

# Menyembunyikan subplot kosong baris atas
for i in range(len(komponen_list) + 1, axes.shape[1]):
    axes[0, i].axis("off")

# Baris bawah: menampilkan variance explained kumulatif
# Menghitung kumulatif variance
cumulative_var = np.cumsum(variance_explained[:min(20, len(variance_explained))])

# Menampilkan grafik variance explained
axes[1, 0].bar(range(1, len(cumulative_var) + 1), variance_explained[:len(cumulative_var)],
               color='steelblue', alpha=0.7, label='Individual')
axes[1, 0].set_xlabel("Komponen")
axes[1, 0].set_ylabel("Variance (%)")
axes[1, 0].set_title("Variance per Komponen", fontsize=10)
axes[1, 0].legend(fontsize=8)

# Menampilkan grafik kumulatif variance
axes[1, 1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-', markersize=4)
axes[1, 1].set_xlabel("Jumlah Komponen")
axes[1, 1].set_ylabel("Kumulatif Variance (%)")
axes[1, 1].set_title("Kumulatif Variance Explained", fontsize=10)
axes[1, 1].axhline(y=95, color='g', linestyle='--', alpha=0.5, label='95%')
axes[1, 1].legend(fontsize=8)

# Menampilkan tabel rekonstruksi error
if len(komponen_list) > 0:
    info_text = "Komponen vs MSE:\n"
    for n_comp in komponen_list:
        basis = eigenfaces[:, :n_comp]
        koefisien = np.dot(contoh_vec, basis)
        rekon = np.dot(koefisien, basis.T) + mean_face
        rekon_img = np.clip(rekon.reshape(FACE_SIZE), 0, 255)
        mse = np.mean((train_images[contoh_idx].astype(np.float64) - rekon_img) ** 2)
        info_text += f"  K={n_comp:2d} → MSE={mse:7.1f}\n"

    axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes,
                    fontsize=9, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 2].set_title("Tabel Error Rekonstruksi", fontsize=10)
    axes[1, 2].axis("off")

# Menyembunyikan subplot kosong baris bawah
for i in range(3, axes.shape[1]):
    axes[1, i].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 4: Rekonstruksi Wajah dari Eigenspace\n"
             "Semakin banyak komponen, semakin mirip dengan aslinya",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil rekonstruksi
output_path_3 = os.path.join(OUTPUT_DIR, "04_rekonstruksi.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# 6. Recognition dengan Nearest-Neighbor di Eigenspace
# ============================================================

print("\n[INFO] Face Recognition menggunakan Eigenfaces + Nearest Neighbor...")
print("-" * 50)

# Menentukan jumlah komponen optimal untuk recognition
n_components = min(10, eigenfaces.shape[1])

# Mengambil eigenfaces yang akan digunakan
basis_recognition = eigenfaces[:, :n_components]

# Memproyeksikan semua gambar training ke eigenspace
train_projected = np.dot(centered_matrix, basis_recognition)

# Menampilkan informasi proyeksi
print(f"  Jumlah komponen: {n_components}")
print(f"  Dimensi proyeksi: {train_projected.shape}")

# Menyiapkan list untuk menyimpan hasil prediksi
prediksi_list = []
jarak_list = []
benar = 0

# Memprediksi setiap gambar testing
for img_test, label_asli in zip(test_images, test_labels):
    # Flatten dan centering gambar test
    test_vec = img_test.flatten().astype(np.float64) - mean_face

    # Memproyeksikan gambar test ke eigenspace
    test_projected = np.dot(test_vec, basis_recognition)

    # Menghitung jarak Euclidean ke setiap gambar training di eigenspace
    distances = np.linalg.norm(train_projected - test_projected, axis=1)

    # Mencari indeks gambar training terdekat
    nearest_idx = np.argmin(distances)

    # Mendapatkan label prediksi dari gambar terdekat
    label_pred = train_labels[nearest_idx]

    # Menyimpan hasil
    prediksi_list.append(label_pred)
    jarak_list.append(distances[nearest_idx])

    # Memeriksa kebenaran prediksi
    if label_pred == label_asli:
        benar += 1

    # Menampilkan hasil
    status = "V" if label_pred == label_asli else "X"
    print(f"  {status} Asli: {label_map[label_asli]}, "
          f"Prediksi: {label_map[label_pred]}, "
          f"Jarak: {distances[nearest_idx]:.2f}")

# Menghitung akurasi
akurasi = benar / len(test_labels) * 100 if len(test_labels) > 0 else 0
print(f"\n  Akurasi: {benar}/{len(test_labels)} = {akurasi:.1f}%")

# ============================================================
# 7. Visualisasi Hasil Eigenface Recognition
# ============================================================

# Menentukan jumlah test yang ditampilkan
n_show = len(test_images)
n_cols_show = min(n_show, 6)

# Membuat figure untuk hasil recognition
fig, axes = plt.subplots(1, max(n_cols_show, 1), figsize=(4 * max(n_cols_show, 1), 5))

# Memastikan axes selalu iterable
if n_show == 1:
    axes = [axes]

# Menampilkan setiap gambar test dengan hasil prediksi
for i in range(n_show):
    # Menampilkan gambar test
    axes[i].imshow(test_images[i], cmap='gray')

    # Menentukan warna berdasarkan kebenaran prediksi
    benar_pred = prediksi_list[i] == test_labels[i]
    warna = 'green' if benar_pred else 'red'

    # Menambahkan judul dengan hasil prediksi
    axes[i].set_title(
        f"Asli: {label_map[test_labels[i]]}\n"
        f"Pred: {label_map[prediksi_list[i]]}\n"
        f"Jarak: {jarak_list[i]:.1f}",
        fontsize=10, color=warna
    )
    axes[i].axis("off")

# Menyembunyikan subplot kosong
for i in range(n_show, len(axes)):
    axes[i].axis("off")

# Menambahkan judul utama
plt.suptitle(f"Percobaan 4: Eigenface Recognition (K={n_components} komponen)\n"
             f"Akurasi: {akurasi:.1f}% | Nearest Neighbor di Eigenspace",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil recognition
output_path_4 = os.path.join(OUTPUT_DIR, "04_eigenface_recognition.png")
plt.savefig(output_path_4, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_4}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 4")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. PCA (Principal Component Analysis)")
print("     - Reduksi dimensi dari n_pixels ke n_komponen")
print("     - Mempertahankan variasi data yang paling penting")
print("  2. Langkah-langkah Eigenfaces:")
print(f"     a. Flatten gambar: {FACE_SIZE} → vektor {n_pixels}D")
print(f"     b. Hitung mean face (rata-rata {len(train_images)} gambar)")
print(f"     c. Centering: kurangi mean face dari setiap gambar")
print(f"     d. Hitung kovarians dan eigendecomposition")
print(f"     e. Ambil top-K eigenfaces sebagai basis")
print(f"     f. Proyeksikan ke eigenspace ({n_pixels}D → {n_components}D)")
print("  3. Recognition: Nearest Neighbor di eigenspace")
print("     - Hitung jarak Euclidean antar proyeksi")
print("     - Prediksi = label gambar training terdekat")
print(f"\nHasil Recognition:")
print(f"  Komponen PCA: {n_components}")
print(f"  Akurasi: {akurasi:.1f}%")
print("=" * 60)
