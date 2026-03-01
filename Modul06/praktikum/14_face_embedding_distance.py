"""
==========================================================================
PERCOBAAN 14: FACE EMBEDDING DAN DISTANCE METRICS
==========================================================================
Program ini mempelajari konsep face embedding, yaitu merepresentasikan
wajah sebagai vektor fitur berdimensi tinggi, lalu mengukur kemiripan
antar wajah menggunakan berbagai metrik jarak. Program mengekstrak fitur
multi-skala dan mengimplementasikan verifikasi wajah sederhana.

Konsep yang dipelajari:
- Face embedding: representasi wajah sebagai vektor fitur
- Multi-scale features: color histogram, LBP histogram, edge histogram
- Distance metrics: Euclidean, Cosine similarity, Manhattan
- Face verification: menentukan same/different person berdasarkan threshold
- FAR (False Acceptance Rate) dan FRR (False Rejection Rate)
- Visualisasi embedding 2D menggunakan PCA sederhana

Fungsi utama yang dipelajari:
- cv2.calcHist()                  : Histogram warna untuk face embedding
- cv2.Sobel()                     : Gradient untuk edge features
- np.linalg.norm()                : Menghitung jarak Euclidean/Manhattan
- np.dot() / cosine similarity    : Menghitung kesamaan kosinus
- PCA manual (eigendecomposition) : Reduksi dimensi untuk visualisasi

Hasil: Visualisasi embedding, distribusi jarak, dan kurva ROC
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan fitur wajah
import cv2

# Mengimpor NumPy untuk operasi array, linear algebra, dan statistik
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
print("PERCOBAAN 14: FACE EMBEDDING DAN DISTANCE METRICS")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Wajah dari Beberapa Orang
# ============================================================

print("\n[INFO] Memuat gambar wajah dari folder per orang...")
print("-" * 50)

# Mendefinisikan folder wajah per orang
face_folders = {
    "andi": os.path.join(IMAGE_DIR, "faces", "andi"),
    "budi": os.path.join(IMAGE_DIR, "faces", "budi"),
    "citra": os.path.join(IMAGE_DIR, "faces", "citra")
}

# Membuat dictionary untuk menyimpan gambar wajah per orang
face_images = {}

# Memuat gambar dari setiap folder
for person_name, folder_path in face_folders.items():
    # Memeriksa apakah folder ada
    if not os.path.exists(folder_path):
        print(f"  [{person_name}] Folder tidak ditemukan: {folder_path}")
        continue

    # Mendapatkan daftar file gambar dalam folder
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Mengurutkan file gambar secara alfabetis
    image_files.sort()

    # Memuat setiap gambar wajah
    person_images = []
    for filename in image_files:
        # Membaca gambar wajah
        img = cv2.imread(os.path.join(folder_path, filename))

        # Menambahkan gambar ke list jika berhasil dimuat
        if img is not None:
            person_images.append((filename, img))

    # Menyimpan gambar per orang jika ada
    if len(person_images) > 0:
        face_images[person_name] = person_images
        print(f"  [{person_name}] {len(person_images)} gambar dimuat")
    else:
        print(f"  [{person_name}] Tidak ada gambar yang ditemukan")

# Memeriksa apakah ada gambar yang dimuat
if len(face_images) < 2:
    print("[ERROR] Minimal 2 orang dengan gambar wajah diperlukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()

# ============================================================
# 2. Ekstraksi Multi-Scale Face Embedding
# ============================================================

print("\n[INFO] Mengekstrak face embedding multi-scale...")
print("-" * 50)

# Mendefinisikan ukuran standar untuk semua wajah
FACE_SIZE = (128, 128)

# Mendefinisikan jumlah bin untuk setiap jenis histogram
COLOR_BINS = 32
LBP_BINS = 26
EDGE_BINS = 18

# Mendefinisikan fungsi untuk menghitung LBP (Local Binary Pattern)
def compute_lbp(gray_image):
    """Menghitung Local Binary Pattern pada gambar grayscale."""
    # Mendapatkan dimensi gambar
    h, w = gray_image.shape

    # Menginisialisasi output LBP
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    # Mendefinisikan offset tetangga 8-connected (searah jarum jam)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                 (1, 1), (1, 0), (1, -1), (0, -1)]

    # Menghitung nilai LBP untuk setiap piksel
    for idx, (dy, dx) in enumerate(neighbors):
        # Mengambil nilai tetangga
        neighbor_vals = gray_image[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]

        # Mengambil nilai pusat
        center_vals = gray_image[1:h - 1, 1:w - 1]

        # Membandingkan tetangga dengan pusat (1 jika >= center, 0 jika <)
        lbp += (neighbor_vals >= center_vals).astype(np.uint8) * (1 << idx)

    # Mengembalikan LBP image
    return lbp

# Mendefinisikan fungsi untuk mengekstrak face embedding
def extract_face_embedding(image, face_size=(128, 128)):
    """Mengekstrak face embedding multi-scale dari gambar wajah."""
    # Meresize gambar ke ukuran standar
    resized = cv2.resize(image, face_size)

    # Mengkonversi ke grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Mengkonversi ke HSV untuk color features
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # --- Fitur 1: Color Histogram (HSV) ---

    # Menghitung histogram Hue
    hist_h = cv2.calcHist([hsv], [0], None, [COLOR_BINS], [0, 180]).flatten()
    hist_h = hist_h / (hist_h.sum() + 1e-7)

    # Menghitung histogram Saturation
    hist_s = cv2.calcHist([hsv], [1], None, [COLOR_BINS], [0, 256]).flatten()
    hist_s = hist_s / (hist_s.sum() + 1e-7)

    # Menghitung histogram Value
    hist_v = cv2.calcHist([hsv], [2], None, [COLOR_BINS], [0, 256]).flatten()
    hist_v = hist_v / (hist_v.sum() + 1e-7)

    # Menggabungkan color histogram
    color_feat = np.concatenate([hist_h, hist_s, hist_v])

    # --- Fitur 2: LBP Histogram (Texture) ---

    # Menghitung LBP
    lbp = compute_lbp(gray)

    # Menghitung histogram LBP (uniform patterns = simplified)
    hist_lbp, _ = np.histogram(lbp.flatten(), bins=LBP_BINS, range=(0, 256))
    hist_lbp = hist_lbp / (hist_lbp.sum() + 1e-7)

    # --- Fitur 3: Edge Histogram (Shape) ---

    # Menghitung gradient menggunakan Sobel
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Menghitung magnitude dan orientasi
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi % 360

    # Menghitung histogram orientasi tertimbang magnitude
    hist_edge, _ = np.histogram(orientation.flatten(), bins=EDGE_BINS,
                                range=(0, 360), weights=magnitude.flatten())
    hist_edge = hist_edge / (hist_edge.sum() + 1e-7)

    # --- Fitur 4: Spatial features (atas/tengah/bawah wajah) ---

    # Membagi wajah menjadi 3 bagian horizontal
    h_third = face_size[1] // 3

    # Menghitung rata-rata intensitas per bagian
    spatial_features = []

    for part in range(3):
        # Menghitung koordinat bagian
        y1 = part * h_third
        y2 = (part + 1) * h_third

        # Menghitung statistik pada bagian ini
        part_region = gray[y1:y2, :]
        spatial_features.append(np.mean(part_region) / 255.0)
        spatial_features.append(np.std(part_region) / 255.0)

    # Mengkonversi spatial features ke numpy array
    spatial_feat = np.array(spatial_features)

    # --- Menggabungkan semua fitur menjadi embedding ---

    # Menggabungkan semua komponen fitur
    embedding = np.concatenate([color_feat, hist_lbp, hist_edge, spatial_feat])

    # Mengembalikan embedding dan komponen-komponen fiturnya
    return embedding, {
        "color": color_feat,
        "lbp": hist_lbp,
        "edge": hist_edge,
        "spatial": spatial_feat
    }

# Mengekstrak embedding untuk setiap wajah
all_embeddings = {}
all_labels = []
all_vectors = []

for person_name, images in face_images.items():
    # Menginisialisasi list embedding per orang
    person_embeddings = []

    for filename, img in images:
        # Mengekstrak face embedding
        embedding, components = extract_face_embedding(img, FACE_SIZE)

        # Menyimpan embedding
        person_embeddings.append((filename, embedding, components))

        # Menambahkan ke list global
        all_labels.append(person_name)
        all_vectors.append(embedding)

    # Menyimpan embeddings per orang
    all_embeddings[person_name] = person_embeddings

    # Menampilkan info embedding
    print(f"  [{person_name}] {len(person_embeddings)} embedding, "
          f"dimensi: {len(embedding)}")
    print(f"    Color: {len(components['color'])}, LBP: {len(components['lbp'])}, "
          f"Edge: {len(components['edge'])}, Spatial: {len(components['spatial'])}")

# Mengkonversi ke numpy arrays
all_vectors = np.array(all_vectors)
all_labels = np.array(all_labels)

# ============================================================
# 3. Visualisasi Face Embedding
# ============================================================

print("\n[INFO] Menyimpan visualisasi face embedding...")

# Menentukan jumlah orang dan total gambar
n_persons = len(all_embeddings)
max_images = max(len(imgs) for imgs in all_embeddings.values())

# Membuat figure untuk menampilkan embedding
fig, axes = plt.subplots(n_persons, max_images + 1, figsize=(4 * (max_images + 1), 4 * n_persons))

# Memastikan axes 2D
if n_persons == 1:
    axes = axes.reshape(1, -1)

# Menampilkan setiap orang dan embedding-nya
for row, (person_name, embeddings) in enumerate(all_embeddings.items()):
    for col, (filename, embedding, components) in enumerate(embeddings):
        # Mengambil gambar asli
        img = [img for fn, img in face_images[person_name] if fn == filename][0]

        # Mengkonversi ke RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Menampilkan gambar
        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(f"{person_name}\n{filename}", fontsize=8)
        axes[row, col].axis("off")

    # Menampilkan embedding sebagai heatmap di kolom terakhir
    # Mengambil semua embedding orang ini
    person_embs = np.array([emb for _, emb, _ in embeddings])

    # Menampilkan embedding sebagai gambar
    axes[row, max_images].imshow(person_embs, aspect="auto", cmap="viridis")
    axes[row, max_images].set_title(f"Embedding {person_name}\n({person_embs.shape[0]}x{person_embs.shape[1]})",
                                     fontsize=8)
    axes[row, max_images].set_xlabel("Feature dim", fontsize=7)
    axes[row, max_images].set_ylabel("Image", fontsize=7)

    # Menyembunyikan subplot kosong
    for col in range(len(embeddings), max_images):
        axes[row, col].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 14: Face Embedding - Wajah dan Vektor Fitur",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi embedding
output_path_1 = os.path.join(OUTPUT_DIR, "14_face_embedding.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 4. Menghitung Jarak Antar Embedding
# ============================================================

print("\n[INFO] Menghitung jarak antar embedding...")
print("-" * 50)

# Mendefinisikan fungsi untuk menghitung berbagai metrik jarak
def compute_distances(vec1, vec2):
    """Menghitung berbagai metrik jarak antara dua vektor."""
    # Menghitung jarak Euclidean
    euclidean = np.linalg.norm(vec1 - vec2)

    # Menghitung cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cosine_sim = dot_product / (norm1 * norm2 + 1e-10)

    # Mengkonversi cosine similarity ke cosine distance
    cosine_dist = 1.0 - cosine_sim

    # Menghitung jarak Manhattan (L1)
    manhattan = np.sum(np.abs(vec1 - vec2))

    # Mengembalikan semua jarak
    return {
        "euclidean": euclidean,
        "cosine_sim": cosine_sim,
        "cosine_dist": cosine_dist,
        "manhattan": manhattan
    }

# Menghitung jarak untuk semua pasangan: same-person dan different-person
same_person_distances = []
diff_person_distances = []

# Mendapatkan nama-nama orang
person_names = list(all_embeddings.keys())

# Menghitung jarak same-person (intra-class)
print("\n  Jarak Intra-class (Same Person):")
for person_name, embeddings in all_embeddings.items():
    # Membandingkan semua pasangan gambar orang yang sama
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            # Menghitung jarak antara embedding i dan j
            dists = compute_distances(embeddings[i][1], embeddings[j][1])

            # Menyimpan jarak
            same_person_distances.append(dists)

            # Menampilkan info jarak
            print(f"    {person_name}: {embeddings[i][0]} vs {embeddings[j][0]} "
                  f"→ Eucl: {dists['euclidean']:.4f}, "
                  f"Cos: {dists['cosine_sim']:.4f}")

# Menghitung jarak different-person (inter-class)
print("\n  Jarak Inter-class (Different Person):")
for i in range(len(person_names)):
    for j in range(i + 1, len(person_names)):
        # Mendapatkan nama orang
        person_a = person_names[i]
        person_b = person_names[j]

        # Membandingkan semua pasangan antar kedua orang
        embs_a = all_embeddings[person_a]
        embs_b = all_embeddings[person_b]

        for ea in embs_a:
            for eb in embs_b:
                # Menghitung jarak
                dists = compute_distances(ea[1], eb[1])

                # Menyimpan jarak
                diff_person_distances.append(dists)

        # Menampilkan rata-rata jarak antar orang
        avg_eucl = np.mean([d["euclidean"] for d in diff_person_distances[-len(embs_a)*len(embs_b):]])
        avg_cos = np.mean([d["cosine_sim"] for d in diff_person_distances[-len(embs_a)*len(embs_b):]])
        print(f"    {person_a} vs {person_b} → Avg Eucl: {avg_eucl:.4f}, "
              f"Avg Cos: {avg_cos:.4f}")

# Menampilkan statistik jarak
print(f"\n  Statistik Jarak Euclidean:")
if same_person_distances:
    same_eucl = [d["euclidean"] for d in same_person_distances]
    print(f"    Same person   : mean={np.mean(same_eucl):.4f}, "
          f"std={np.std(same_eucl):.4f}")
if diff_person_distances:
    diff_eucl = [d["euclidean"] for d in diff_person_distances]
    print(f"    Diff person   : mean={np.mean(diff_eucl):.4f}, "
          f"std={np.std(diff_eucl):.4f}")

# ============================================================
# 5. Visualisasi Distribusi Jarak
# ============================================================

print("\n[INFO] Menyimpan visualisasi distribusi jarak...")

# Membuat figure untuk distribusi jarak
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mengumpulkan data jarak untuk plotting
metrics = ["euclidean", "cosine_dist", "manhattan"]
metric_labels = ["Euclidean Distance", "Cosine Distance", "Manhattan Distance"]

# Memplot distribusi untuk 3 metrik
for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    # Menentukan posisi subplot
    ax = axes[idx // 2, idx % 2]

    # Mengambil data jarak per kategori
    same_vals = [d[metric] for d in same_person_distances] if same_person_distances else [0]
    diff_vals = [d[metric] for d in diff_person_distances] if diff_person_distances else [0]

    # Mendefinisikan range histogram
    all_vals = same_vals + diff_vals
    hist_min = min(all_vals)
    hist_max = max(all_vals)
    bins = np.linspace(hist_min, hist_max, 20)

    # Menggambar histogram same-person (hijau)
    ax.hist(same_vals, bins=bins, alpha=0.6, color="green",
            label=f"Same Person (n={len(same_vals)})", edgecolor="black")

    # Menggambar histogram different-person (merah)
    ax.hist(diff_vals, bins=bins, alpha=0.6, color="red",
            label=f"Diff Person (n={len(diff_vals)})", edgecolor="black")

    # Mengatur judul dan label
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Distance", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.legend(fontsize=8)

# --- Subplot 4: Cosine Similarity distribution ---

# Mengambil data cosine similarity
same_cos = [d["cosine_sim"] for d in same_person_distances] if same_person_distances else [0]
diff_cos = [d["cosine_sim"] for d in diff_person_distances] if diff_person_distances else [0]

# Menggambar histogram cosine similarity
all_cos = same_cos + diff_cos
cos_bins = np.linspace(min(all_cos), max(all_cos), 20)

axes[1, 1].hist(same_cos, bins=cos_bins, alpha=0.6, color="green",
                label=f"Same Person", edgecolor="black")
axes[1, 1].hist(diff_cos, bins=cos_bins, alpha=0.6, color="red",
                label=f"Diff Person", edgecolor="black")

axes[1, 1].set_title("Cosine Similarity Distribution", fontsize=11, fontweight="bold")
axes[1, 1].set_xlabel("Similarity (1=identik, 0=berbeda)", fontsize=9)
axes[1, 1].set_ylabel("Frequency", fontsize=9)
axes[1, 1].legend(fontsize=8)

# Menambahkan judul utama
plt.suptitle("Percobaan 14: Distribusi Jarak Same-Person vs Different-Person",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi distribusi jarak
output_path_2 = os.path.join(OUTPUT_DIR, "14_distance_distribution.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 6. Face Verification dan Kurva ROC (FAR/FRR)
# ============================================================

print("\n[INFO] Menghitung FAR/FRR dan membuat kurva ROC...")
print("-" * 50)

# Mengumpulkan semua jarak Euclidean dengan label
all_pairs_distances = []
all_pairs_is_same = []

# Menambahkan pasangan same-person
for d in same_person_distances:
    all_pairs_distances.append(d["euclidean"])
    all_pairs_is_same.append(True)

# Menambahkan pasangan different-person
for d in diff_person_distances:
    all_pairs_distances.append(d["euclidean"])
    all_pairs_is_same.append(False)

# Mengkonversi ke numpy arrays
all_pairs_distances = np.array(all_pairs_distances)
all_pairs_is_same = np.array(all_pairs_is_same)

# Mendefinisikan range threshold untuk evaluasi  
if len(all_pairs_distances) > 0:
    # Menentukan range threshold dari min ke max jarak
    thresholds = np.linspace(all_pairs_distances.min(), all_pairs_distances.max(), 100)
else:
    # Menggunakan threshold default jika tidak ada data
    thresholds = np.linspace(0, 1, 100)

# Menghitung jumlah pasangan same dan different
n_same = np.sum(all_pairs_is_same)
n_diff = np.sum(~all_pairs_is_same)

# Menginisialisasi array untuk FAR dan FRR
far_values = []
frr_values = []

# Menghitung FAR dan FRR untuk setiap threshold
for threshold in thresholds:
    # Mengklasifikasikan setiap pasangan (distance < threshold → same person)
    predictions = all_pairs_distances < threshold

    # Menghitung False Acceptance (diff predicted as same)
    false_accept = np.sum(predictions & ~all_pairs_is_same)

    # Menghitung False Rejection (same predicted as diff)
    false_reject = np.sum(~predictions & all_pairs_is_same)

    # Menghitung FAR dan FRR
    far = false_accept / max(n_diff, 1)
    frr = false_reject / max(n_same, 1)

    # Menyimpan FAR dan FRR
    far_values.append(far)
    frr_values.append(frr)

# Mengkonversi ke numpy arrays
far_values = np.array(far_values)
frr_values = np.array(frr_values)

# Mencari EER (Equal Error Rate) - titik di mana FAR ≈ FRR
eer_idx = np.argmin(np.abs(far_values - frr_values))
eer_threshold = thresholds[eer_idx]
eer_value = (far_values[eer_idx] + frr_values[eer_idx]) / 2

# Menampilkan hasil EER
print(f"  EER (Equal Error Rate): {eer_value:.4f}")
print(f"  EER Threshold: {eer_threshold:.4f}")
print(f"  Total pasangan same: {n_same}")
print(f"  Total pasangan diff: {n_diff}")

# ============================================================
# 7. Visualisasi PCA 2D dan Kurva ROC
# ============================================================

print("\n[INFO] Membuat visualisasi PCA dan kurva ROC...")

# --- Melakukan PCA sederhana untuk visualisasi 2D ---

# Menghitung mean embedding
mean_vec = np.mean(all_vectors, axis=0)

# Mengurangi mean dari semua embedding (centering)
centered = all_vectors - mean_vec

# Menghitung matriks kovarian
cov_matrix = np.cov(centered, rowvar=False)

# Menghitung eigenvalues dan eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Mengurutkan eigenvalues dan eigenvectors (descending)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Mengambil 2 eigenvectors pertama (komponen utama)
pca_components = eigenvectors[:, :2]

# Memproyeksikan embedding ke 2D
projected = centered @ pca_components

# Menghitung variance explained
total_variance = np.sum(eigenvalues)
var_explained = eigenvalues[:2] / (total_variance + 1e-10) * 100

# Menampilkan info PCA
print(f"  PCA Variance explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%")

# Membuat figure untuk PCA dan kurva ROC
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Subplot 1: PCA 2D scatter plot ---

# Mendefinisikan warna per orang
color_map = {"andi": "red", "budi": "blue", "citra": "green"}
marker_map = {"andi": "o", "budi": "s", "citra": "^"}

# Menampilkan setiap titik pada scatter plot
for person_name in person_names:
    # Mendapatkan indeks gambar orang ini
    indices = np.where(all_labels == person_name)[0]

    # Menampilkan titik-titik embedding
    axes[0].scatter(projected[indices, 0], projected[indices, 1],
                   c=color_map.get(person_name, "gray"),
                   marker=marker_map.get(person_name, "o"),
                   s=100, label=person_name, edgecolors="black", alpha=0.8)

    # Menambahkan label text pada setiap titik
    for idx in indices:
        axes[0].annotate(f"{all_labels[idx]}",
                        (projected[idx, 0], projected[idx, 1]),
                        fontsize=7, ha="center", va="bottom")

# Mengatur judul dan label
axes[0].set_title(f"PCA 2D Embedding\n(Var: {var_explained[0]:.1f}%, {var_explained[1]:.1f}%)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("PC1", fontsize=10)
axes[0].set_ylabel("PC2", fontsize=10)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# --- Subplot 2: FAR/FRR vs Threshold ---

# Menggambar kurva FAR
axes[1].plot(thresholds, far_values, color="red", label="FAR (False Accept)", linewidth=2)

# Menggambar kurva FRR
axes[1].plot(thresholds, frr_values, color="blue", label="FRR (False Reject)", linewidth=2)

# Menandai titik EER
axes[1].plot(eer_threshold, eer_value, "ko", markersize=10, label=f"EER={eer_value:.3f}")

# Menggambar garis vertikal pada EER threshold
axes[1].axvline(x=eer_threshold, color="gray", linestyle="--", alpha=0.5)

# Mengatur judul dan label
axes[1].set_title(f"FAR/FRR vs Threshold\n(EER={eer_value:.3f} at T={eer_threshold:.3f})",
                  fontsize=11, fontweight="bold")
axes[1].set_xlabel("Threshold (Euclidean Distance)", fontsize=10)
axes[1].set_ylabel("Error Rate", fontsize=10)
axes[1].legend(fontsize=9)
axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(True, alpha=0.3)

# --- Subplot 3: ROC Curve (FAR vs 1-FRR) ---

# Menghitung True Acceptance Rate (TAR = 1 - FRR)
tar_values = 1.0 - frr_values

# Menggambar kurva ROC
axes[2].plot(far_values, tar_values, color="green", linewidth=2, label="ROC Curve")

# Menggambar garis diagonal (random classifier)
axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

# Menandai titik EER pada kurva ROC
eer_tar = 1.0 - eer_value
axes[2].plot(eer_value, eer_tar, "ro", markersize=10, label=f"EER ({eer_value:.3f})")

# Menghitung AUC (Area Under Curve) menggunakan trapezoidal rule
auc = np.trapz(tar_values, far_values)

# Mengatur judul dan label
axes[2].set_title(f"ROC Curve\n(AUC={abs(auc):.3f})", fontsize=11, fontweight="bold")
axes[2].set_xlabel("FAR (False Acceptance Rate)", fontsize=10)
axes[2].set_ylabel("TAR (True Acceptance Rate)", fontsize=10)
axes[2].legend(fontsize=9)
axes[2].set_xlim(-0.05, 1.05)
axes[2].set_ylim(-0.05, 1.05)
axes[2].grid(True, alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 14: Face Verification - PCA Embedding dan ROC",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ROC
output_path_3 = os.path.join(OUTPUT_DIR, "14_verifikasi_roc.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 14")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Face Embedding (Wajah → Vektor):")
print(f"     - Color histogram (HSV): {COLOR_BINS}*3 = {COLOR_BINS*3} dim")
print(f"     - LBP histogram (texture): {LBP_BINS} dim")
print(f"     - Edge histogram (shape): {EDGE_BINS} dim")
print("     - Spatial features (atas/tengah/bawah): 6 dim")
print(f"     - Total embedding: {COLOR_BINS*3 + LBP_BINS + EDGE_BINS + 6} dim")
print("  2. Distance Metrics:")
print("     - Euclidean: ||v1 - v2||₂")
print("     - Cosine: 1 - (v1·v2)/(||v1||·||v2||)")
print("     - Manhattan: Σ|v1ᵢ - v2ᵢ|")
print("  3. Face Verification:")
print("     - Same person: jarak kecil (intra-class)")
print("     - Diff person: jarak besar (inter-class)")
print("     - Threshold → accept/reject decision")
print("  4. Evaluasi:")
print("     - FAR: proportion diff pairs wrongly accepted")
print("     - FRR: proportion same pairs wrongly rejected")
print(f"     - EER: {eer_value:.4f} at threshold {eer_threshold:.4f}")
print("  5. PCA (Principal Component Analysis):")
print("     - Reduksi dimensi untuk visualisasi 2D")
print("     - Eigendecomposition pada matriks kovarian")
print(f"     - Variance explained: {var_explained[0]:.1f}% + {var_explained[1]:.1f}%")
print(f"\nTotal orang: {len(face_images)}")
for person_name, images in face_images.items():
    print(f"  - {person_name}: {len(images)} gambar")
print(f"Pasangan same: {n_same}, Pasangan diff: {n_diff}")
print("=" * 60)
