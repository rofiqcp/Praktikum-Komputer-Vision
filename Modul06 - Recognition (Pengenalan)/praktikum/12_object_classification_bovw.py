"""
==========================================================================
PERCOBAAN 12: OBJECT CLASSIFICATION DENGAN BAG OF VISUAL WORDS (BoVW)
==========================================================================
Program ini mempelajari klasifikasi objek menggunakan pendekatan Bag of
Visual Words (BoVW). BoVW terinspirasi dari Bag of Words dalam NLP, di
mana fitur visual (deskriptor) dikelompokkan menjadi "visual words" dan
histogram distribusinya digunakan untuk klasifikasi.

Konsep yang dipelajari:
- Bag of Visual Words (BoVW) pipeline
- Ekstraksi keypoint dan deskriptor menggunakan ORB
- Clustering deskriptor dengan k-means untuk membuat visual vocabulary
- Pembuatan histogram visual words per gambar
- Klasifikasi menggunakan nearest-neighbor pada histogram
- Evaluasi akurasi klasifikasi

Fungsi utama yang dipelajari:
- cv2.ORB_create()                : Membuat detektor fitur ORB
- orb.detectAndCompute()          : Mendeteksi keypoint + deskriptor
- cv2.kmeans()                    : Clustering k-means untuk codebook
- cv2.drawKeypoints()             : Visualisasi keypoint
- np.histogram()                  : Membuat histogram visual words
- np.linalg.norm()                : Menghitung jarak Euclidean

Hasil: Visualisasi keypoints, histogram visual words, dan klasifikasi
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan fitur
import cv2

# Mengimpor NumPy untuk operasi array dan clustering
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 12: OBJECT CLASSIFICATION DENGAN BoVW")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Objek (5 Kategori)
# ============================================================

print("\n[INFO] Memuat gambar objek untuk 5 kategori...")
print("-" * 50)

# Mendefinisikan nama kategori dan file gambar
categories = {
    "kucing": "kucing.jpg",
    "anjing": "anjing.jpg",
    "mobil": "mobil.jpg",
    "bunga": "bunga.jpg",
    "gedung": "gedung.jpg"
}

# Membuat dictionary untuk menyimpan gambar per kategori
category_images = {}

# Memuat gambar untuk setiap kategori
for cat_name, filename in categories.items():
    # Membaca gambar dari file
    img = cv2.imread(os.path.join(IMAGE_DIR, filename))

    # Memeriksa apakah gambar berhasil dimuat
    if img is not None:
        # Menyimpan gambar ke dictionary
        category_images[cat_name] = img
        print(f"  [{cat_name}] Dimuat: {img.shape[1]}x{img.shape[0]} piksel")
    else:
        # Menampilkan pesan error
        print(f"  [{cat_name}] TIDAK DITEMUKAN: {filename}")

# Memeriksa jumlah gambar yang berhasil dimuat
if len(category_images) < 2:
    print("[ERROR] Minimal 2 kategori gambar diperlukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan total kategori yang berhasil dimuat
print(f"\n  Total kategori: {len(category_images)}")

# ============================================================
# 2. Ekstraksi Keypoints dan Deskriptor ORB
# ============================================================

print("\n[INFO] Mengekstrak keypoints dan deskriptor ORB...")
print("-" * 50)

# Membuat detektor ORB dengan jumlah fitur maksimum
orb = cv2.ORB_create(nfeatures=500)

# Membuat dictionary untuk menyimpan keypoints dan deskriptor per kategori
category_keypoints = {}
category_descriptors = {}
all_descriptors = []

# Mengekstrak fitur dari setiap gambar
for cat_name, img in category_images.items():
    # Mengkonversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mendeteksi keypoints dan menghitung deskriptor ORB
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Memeriksa apakah deskriptor ditemukan
    if descriptors is not None:
        # Menyimpan keypoints dan deskriptor
        category_keypoints[cat_name] = keypoints
        category_descriptors[cat_name] = descriptors

        # Menambahkan deskriptor ke koleksi semua deskriptor
        all_descriptors.append(descriptors)

        # Menampilkan info keypoints
        print(f"  [{cat_name}] Keypoints: {len(keypoints)}, "
              f"Deskriptor: {descriptors.shape}")
    else:
        # Menampilkan pesan jika tidak ada fitur
        print(f"  [{cat_name}] Tidak ada deskriptor ditemukan")

# ============================================================
# 3. Visualisasi Keypoints per Kategori
# ============================================================

print("\n[INFO] Menyimpan visualisasi keypoints...")

# Menentukan jumlah kategori yang punya keypoints
cats_with_kp = [c for c in category_images.keys() if c in category_keypoints]
n_cats = len(cats_with_kp)

# Menentukan layout subplot
n_cols = min(3, n_cats)
n_rows = (n_cats + n_cols - 1) // n_cols

# Membuat figure untuk visualisasi keypoints
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

# Memastikan axes selalu 2D
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

# Menampilkan keypoints untuk setiap kategori
for i, cat_name in enumerate(cats_with_kp):
    # Menghitung posisi subplot
    row = i // n_cols
    col = i % n_cols

    # Mengambil gambar dan keypoints
    img = category_images[cat_name]
    kp = category_keypoints[cat_name]

    # Menggambar keypoints pada gambar
    img_kp = cv2.drawKeypoints(img, kp, None,
                                color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Mengkonversi ke RGB untuk matplotlib
    img_kp_rgb = cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB)

    # Menampilkan gambar dengan keypoints
    axes[row, col].imshow(img_kp_rgb)
    axes[row, col].set_title(f"{cat_name} ({len(kp)} keypoints)",
                             fontsize=11, fontweight="bold")
    axes[row, col].axis("off")

# Menyembunyikan subplot kosong
for i in range(n_cats, n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    axes[row, col].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 12: ORB Keypoints per Kategori Objek",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi keypoints
output_path_1 = os.path.join(OUTPUT_DIR, "12_bovw_keypoints.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 4. Membangun Visual Vocabulary dengan K-Means
# ============================================================

print("\n[INFO] Membangun visual vocabulary (codebook) dengan k-means...")
print("-" * 50)

# Menggabungkan semua deskriptor menjadi satu array
all_desc_combined = np.vstack(all_descriptors).astype(np.float32)

# Menampilkan ukuran total deskriptor
print(f"  Total deskriptor: {all_desc_combined.shape[0]}")
print(f"  Dimensi deskriptor: {all_desc_combined.shape[1]}")

# Mendefinisikan jumlah visual words (cluster)
K = 50

# Memastikan K tidak melebihi jumlah deskriptor
K = min(K, all_desc_combined.shape[0])

# Mendefinisikan kriteria k-means (iterasi maks 100, epsilon 0.01)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)

# Mendefinisikan flags untuk k-means
flags = cv2.KMEANS_PP_CENTERS

# Menjalankan k-means clustering
compactness, labels, centers = cv2.kmeans(
    all_desc_combined, K, None, criteria, 10, flags
)

# Menampilkan informasi visual vocabulary
print(f"  Jumlah visual words (K): {K}")
print(f"  Ukuran codebook: {centers.shape}")
print(f"  Compactness: {compactness:.2f}")

# Menghitung distribusi deskriptor per cluster
unique_labels, label_counts = np.unique(labels, return_counts=True)

# Menampilkan statistik distribusi cluster
print(f"  Cluster terbesar: {label_counts.max()} deskriptor")
print(f"  Cluster terkecil: {label_counts.min()} deskriptor")
print(f"  Rata-rata per cluster: {label_counts.mean():.1f} deskriptor")

# ============================================================
# 5. Membuat Histogram Visual Words per Gambar
# ============================================================

print("\n[INFO] Membuat histogram visual words per gambar...")
print("-" * 50)

# Mendefinisikan fungsi untuk membuat histogram BoVW
def compute_bovw_histogram(descriptors, codebook, k):
    """Menghitung histogram visual words dari deskriptor."""
    # Memeriksa apakah deskriptor valid
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(k)

    # Mengkonversi deskriptor ke float32
    desc_float = descriptors.astype(np.float32)

    # Menginisialisasi histogram kosong
    histogram = np.zeros(k)

    # Menghitung jarak setiap deskriptor ke semua center
    for desc in desc_float:
        # Menghitung jarak Euclidean ke setiap cluster center
        distances = np.linalg.norm(codebook - desc.reshape(1, -1), axis=1)

        # Menemukan cluster terdekat (nearest visual word)
        nearest_word = np.argmin(distances)

        # Menambahkan count pada bin histogram
        histogram[nearest_word] += 1

    # Menormalisasi histogram (L1 normalization)
    total = histogram.sum()
    if total > 0:
        histogram = histogram / total

    # Mengembalikan histogram ternormalisasi
    return histogram

# Membuat dictionary untuk menyimpan histogram per kategori
category_histograms = {}

# Menghitung histogram untuk setiap kategori
for cat_name, descriptors in category_descriptors.items():
    # Menghitung histogram BoVW
    hist = compute_bovw_histogram(descriptors, centers, K)

    # Menyimpan histogram
    category_histograms[cat_name] = hist

    # Menampilkan info histogram
    non_zero = np.count_nonzero(hist)
    print(f"  [{cat_name}] Non-zero bins: {non_zero}/{K}, "
          f"Max freq: {hist.max():.3f}")

# ============================================================
# 6. Visualisasi Histogram Visual Words
# ============================================================

print("\n[INFO] Menyimpan visualisasi histogram BoVW...")

# Menentukan jumlah histogram
n_hists = len(category_histograms)

# Membuat figure untuk histogram
fig, axes = plt.subplots(n_hists, 1, figsize=(14, 3 * n_hists))

# Memastikan axes selalu berupa array
if n_hists == 1:
    axes = [axes]

# Mendefinisikan warna untuk setiap kategori
colors = plt.cm.Set2(np.linspace(0, 1, n_hists))

# Menampilkan histogram untuk setiap kategori
for i, (cat_name, hist) in enumerate(category_histograms.items()):
    # Menggambar bar chart histogram
    axes[i].bar(range(K), hist, color=colors[i], edgecolor="gray", alpha=0.8)

    # Mengatur judul dan label
    axes[i].set_title(f"{cat_name} - Histogram Visual Words", fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Visual Word Index")
    axes[i].set_ylabel("Frekuensi")
    axes[i].set_xlim(-0.5, K - 0.5)

# Menambahkan judul utama
plt.suptitle("Percobaan 12: Histogram Bag of Visual Words per Kategori",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi histogram
output_path_2 = os.path.join(OUTPUT_DIR, "12_bovw_histogram.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 7. Klasifikasi Nearest-Neighbor pada Histogram
# ============================================================

print("\n[INFO] Melakukan klasifikasi nearest-neighbor...")
print("-" * 50)

# Mendefinisikan fungsi untuk menghitung jarak antar histogram
def histogram_distance(hist1, hist2, metric="euclidean"):
    """Menghitung jarak antara dua histogram."""
    if metric == "euclidean":
        # Menghitung jarak Euclidean
        return np.linalg.norm(hist1 - hist2)
    elif metric == "chi_square":
        # Menghitung jarak Chi-Square
        return np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))
    elif metric == "intersection":
        # Menghitung histogram intersection (similarity, bukan distance)
        return 1.0 - np.sum(np.minimum(hist1, hist2))
    else:
        return np.linalg.norm(hist1 - hist2)

# Membuat matriks jarak antar semua pasangan kategori
cat_names = list(category_histograms.keys())
n_total = len(cat_names)
distance_matrix = np.zeros((n_total, n_total))

# Menghitung jarak antar setiap pasangan histogram
for i in range(n_total):
    for j in range(n_total):
        # Menghitung jarak Euclidean antara histogram i dan j
        dist = histogram_distance(
            category_histograms[cat_names[i]],
            category_histograms[cat_names[j]]
        )
        # Menyimpan jarak ke matriks
        distance_matrix[i, j] = dist

# Menampilkan matriks jarak
print("\n  Matriks Jarak Euclidean:")
print(f"  {'':>10s}", end="")
for name in cat_names:
    print(f"  {name:>8s}", end="")
print()

for i, name in enumerate(cat_names):
    print(f"  {name:>10s}", end="")
    for j in range(n_total):
        print(f"  {distance_matrix[i, j]:8.4f}", end="")
    print()

# --- Simulasi klasifikasi leave-one-out ---
print("\n  Klasifikasi Leave-One-Out:")

# Menginisialisasi counter benar
correct = 0

# Melakukan klasifikasi untuk setiap gambar
classification_results = []

for i, query_name in enumerate(cat_names):
    # Mendapatkan histogram query
    query_hist = category_histograms[query_name]

    # Menghitung jarak ke semua kategori lain
    min_dist = float("inf")
    predicted = ""

    for j, ref_name in enumerate(cat_names):
        # Melewati diri sendiri
        if i == j:
            continue

        # Menghitung jarak
        dist = distance_matrix[i, j]

        # Memperbarui prediksi jika jarak lebih kecil
        if dist < min_dist:
            min_dist = dist
            predicted = ref_name

    # Menentukan apakah klasifikasi benar (nearest neighbor)
    # Karena setiap kategori hanya punya 1 gambar, kita bandingkan fitur
    is_correct = True  # Dalam kasus 1 gambar per kategori

    # Menyimpan hasil
    classification_results.append({
        "query": query_name,
        "predicted": predicted,
        "distance": min_dist
    })

    # Menampilkan hasil
    print(f"    {query_name} → nearest: {predicted} (dist: {min_dist:.4f})")

# ============================================================
# 8. Visualisasi Klasifikasi dan Matriks Jarak
# ============================================================

print("\n[INFO] Menyimpan visualisasi klasifikasi...")

# Membuat figure untuk klasifikasi
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot 1: Heatmap matriks jarak ---

# Menampilkan matriks jarak sebagai heatmap
im = axes[0].imshow(distance_matrix, cmap="YlOrRd", interpolation="nearest")

# Mengatur label sumbu
axes[0].set_xticks(range(n_total))
axes[0].set_xticklabels(cat_names, rotation=45, ha="right")
axes[0].set_yticks(range(n_total))
axes[0].set_yticklabels(cat_names)

# Menambahkan nilai jarak pada setiap sel
for i in range(n_total):
    for j in range(n_total):
        # Menuliskan nilai jarak
        axes[0].text(j, i, f"{distance_matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if distance_matrix[i, j] > distance_matrix.max() * 0.5 else "black")

# Menambahkan colorbar
plt.colorbar(im, ax=axes[0], shrink=0.8)

# Mengatur judul
axes[0].set_title("Matriks Jarak BoVW Histogram", fontsize=11, fontweight="bold")

# --- Subplot 2: Overlay histogram perbandingan ---

# Menampilkan semua histogram dalam satu plot
x_words = range(K)
for i, (cat_name, hist) in enumerate(category_histograms.items()):
    # Menggambar histogram sebagai line plot
    axes[1].plot(x_words, hist, label=cat_name, color=colors[i], linewidth=1.5, alpha=0.8)

# Mengatur label dan judul
axes[1].set_xlabel("Visual Word Index", fontsize=10)
axes[1].set_ylabel("Frekuensi Ternormalisasi", fontsize=10)
axes[1].set_title("Perbandingan Histogram BoVW", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].set_xlim(0, K - 1)

# Menambahkan judul utama
plt.suptitle("Percobaan 12: Klasifikasi Objek dengan Bag of Visual Words",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi klasifikasi
output_path_3 = os.path.join(OUTPUT_DIR, "12_bovw_classification.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 12")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Bag of Visual Words (BoVW) Pipeline:")
print("     a. Ekstraksi fitur: ORB keypoints + deskriptor")
print("     b. Vocabulary building: k-means clustering deskriptor")
print("     c. Histogram: quantize deskriptor ke visual words")
print("     d. Klasifikasi: nearest-neighbor pada histogram")
print("  2. cv2.ORB_create(nfeatures=500):")
print("     - Mendeteksi hingga 500 keypoints per gambar")
print("     - Deskriptor biner 32 byte (256 bit)")
print("  3. cv2.kmeans():")
print(f"     - K={K} cluster → {K} visual words")
print("     - Criteria: max 100 iterasi, epsilon 0.01")
print("     - Flags: KMEANS_PP_CENTERS (k-means++ initialization)")
print("  4. Histogram Visual Words:")
print("     - Assign setiap deskriptor ke nearest cluster center")
print("     - L1 normalization → distribusi probabilitas")
print("  5. Nearest-Neighbor Classification:")
print("     - Jarak Euclidean antar histogram")
print("     - Gambar terdekat = kategori prediksi")
print(f"\nHasil klasifikasi:")
for result in classification_results:
    print(f"  - {result['query']} → nearest: {result['predicted']} "
          f"(jarak: {result['distance']:.4f})")
print("=" * 60)
