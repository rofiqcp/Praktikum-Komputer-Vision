"""
==========================================================
PERCOBAAN 3: ESTIMASI NORMAL PADA POINT CLOUD
Mempelajari cara menghitung surface normal pada setiap
titik menggunakan PCA lokal (eigenvector analisis).

Fungsi utama:
- open3d estimate_normals() (atau manual PCA)
- numpy linalg.eig(), linalg.svd()
- scipy.spatial.KDTree
==========================================================
"""

# Mengimpor library numpy untuk operasi numerik
import numpy as np

# Mengimpor library os untuk operasi sistem file
import os

# Mengimpor library matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mengimpor Axes3D untuk plot 3D
from mpl_toolkits.mplot3d import Axes3D

# Mengimpor KDTree dari scipy untuk pencarian tetangga
from scipy.spatial import KDTree

# Mencoba mengimpor Open3D, jika gagal gunakan fallback
try:
    # Mengimpor library Open3D
    import open3d as o3d
    # Menandai ketersediaan Open3D
    HAS_OPEN3D = True
    print("[INFO] Open3D berhasil diimpor")
except ImportError:
    # Menandai bahwa Open3D tidak tersedia
    HAS_OPEN3D = False
    print("[INFO] Open3D tidak tersedia, menggunakan fallback NumPy+SciPy")

# ========================================================
# KONFIGURASI DIREKTORI
# ========================================================

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Menentukan direktori untuk gambar/data input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Menentukan direktori untuk menyimpan hasil output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat direktori output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mencetak header utama percobaan
print("=" * 60)
print("PERCOBAAN 3: ESTIMASI NORMAL PADA POINT CLOUD")
print("=" * 60)
print()


def load_ply_manual(filepath):
    """
    Memuat file PLY secara manual tanpa Open3D.
    """
    # Menginisialisasi list untuk titik dan warna
    points = []
    colors = []
    has_colors = False
    num_vertices = 0

    # Membuka file PLY
    with open(filepath, 'r') as f:
        in_header = True
        properties = []

        # Mengiterasi setiap baris
        for line in f:
            line = line.strip()

            if in_header:
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                elif line.startswith("property"):
                    parts = line.split()
                    properties.append(parts[-1])
                    if parts[-1] in ['red', 'green', 'blue']:
                        has_colors = True
                elif line == "end_header":
                    in_header = False
                continue

            values = line.split()
            if len(points) >= num_vertices:
                break

            x, y, z = float(values[0]), float(values[1]), float(values[2])
            points.append([x, y, z])

            if has_colors:
                r, g, b = int(values[3]), int(values[4]), int(values[5])
                colors.append([r, g, b])

    # Mengkonversi ke numpy array
    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8) if colors else None

    return points, colors


def estimate_normals_pca(points, k_neighbors=20):
    """
    Menghitung surface normal pada setiap titik menggunakan PCA lokal.
    Untuk setiap titik, ambil k tetangga terdekat, hitung matriks kovarian,
    dan eigenvector dengan eigenvalue terkecil adalah normal.
    """
    # Membangun KDTree untuk pencarian tetangga
    kdtree = KDTree(points)

    # Menginisialisasi array untuk menyimpan normal
    normals = np.zeros_like(points)

    # Menginisialisasi array untuk menyimpan eigenvalue (untuk curvature)
    eigenvalues_all = np.zeros((len(points), 3))

    # Mengiterasi setiap titik
    for i in range(len(points)):
        # Mencari k tetangga terdekat
        distances, indices = kdtree.query(points[i], k=k_neighbors)

        # Mengambil titik-titik tetangga
        neighbors = points[indices]

        # Menghitung centroid lokal dari tetangga
        centroid = np.mean(neighbors, axis=0)

        # Menghitung matriks kovarian dari titik-titik tetangga
        centered = neighbors - centroid

        # Menghitung matriks kovarian
        cov_matrix = (centered.T @ centered) / len(neighbors)

        # Menghitung eigenvalue dan eigenvector menggunakan SVD
        U, S, Vt = np.linalg.svd(cov_matrix)

        # Eigenvector dengan eigenvalue terkecil adalah normal permukaan
        normal = Vt[2]  # Baris terakhir dari Vt (eigenvalue terkecil)

        # Menyimpan normal
        normals[i] = normal

        # Menyimpan eigenvalue (diurutkan dari besar ke kecil)
        eigenvalues_all[i] = S

    # Mengembalikan normal dan eigenvalue
    return normals, eigenvalues_all


def orient_normals_consistently(points, normals, viewpoint=None):
    """
    Mengorientasikan normal secara konsisten agar mengarah ke luar objek.
    Menggunakan viewpoint (titik pandang) atau centroid sebagai referensi.
    """
    # Menentukan viewpoint default jika tidak diberikan
    if viewpoint is None:
        # Menggunakan titik jauh di atas objek sebagai viewpoint
        centroid = np.mean(points, axis=0)
        viewpoint = centroid + np.array([0, 0, 5])

    # Mengorientasikan setiap normal
    for i in range(len(normals)):
        # Menghitung vektor dari titik ke viewpoint
        to_viewpoint = viewpoint - points[i]

        # Menghitung dot product antara normal dan vektor ke viewpoint
        dot = np.dot(normals[i], to_viewpoint)

        # Jika dot product negatif, balikkan arah normal
        if dot < 0:
            normals[i] = -normals[i]

    # Mengembalikan normal yang sudah terorienta si
    return normals


# ========================================================
# 1. MEMUAT POINT CLOUD
# ========================================================

# Mencetak header bagian memuat data
print("=" * 60)
print("1. MEMUAT POINT CLOUD")
print("=" * 60)

# Menentukan path file PLY
ply_path = os.path.join(IMAGE_DIR, "bunny_point_cloud.ply")

# Mengecek file ada
if not os.path.exists(ply_path):
    print(f"[ERROR] File tidak ditemukan: {ply_path}")
    print("Jalankan download_image.py terlebih dahulu!")
    exit(1)

# Memuat point cloud
if HAS_OPEN3D:
    # Memuat dengan Open3D
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    has_col = pcd.has_colors()
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if has_col else None
    print(f"  [Open3D] Point cloud dimuat: {len(points)} titik")
else:
    # Memuat dengan parser manual
    points, colors = load_ply_manual(ply_path)
    print(f"  [Manual] Point cloud dimuat: {len(points)} titik")

# Menggunakan subsample untuk mempercepat perhitungan
max_points_for_normal = 2000
if len(points) > max_points_for_normal:
    # Memilih subset acak untuk estimasi normal
    sample_indices = np.random.choice(len(points), max_points_for_normal, replace=False)
    points_sub = points[sample_indices]
    colors_sub = colors[sample_indices] if colors is not None else None
    print(f"  Subsample untuk estimasi normal: {max_points_for_normal} titik")
else:
    # Menggunakan semua titik
    points_sub = points.copy()
    colors_sub = colors.copy() if colors is not None else None

print()


# ========================================================
# 2. ESTIMASI NORMAL MENGGUNAKAN PCA (MANUAL)
# ========================================================

# Mencetak header bagian estimasi normal PCA
print("=" * 60)
print("2. ESTIMASI NORMAL MENGGUNAKAN PCA (MANUAL)")
print("=" * 60)

# Menentukan jumlah tetangga untuk PCA lokal
k_neighbors = 20

# Mencetak parameter estimasi
print(f"  Metode             : PCA lokal (SVD)")
print(f"  K tetangga         : {k_neighbors}")
print(f"  Jumlah titik       : {len(points_sub)}")

# Menghitung normal menggunakan PCA manual
print("  Menghitung normal... (mungkin memerlukan waktu)")
normals_manual, eigenvalues_all = estimate_normals_pca(points_sub, k_neighbors=k_neighbors)

# Mencetak statistik normal sebelum orientasi
print(f"  Normal dihitung    : {len(normals_manual)}")

# Menghitung magnitude normal (seharusnya ~1 karena SVD menghasilkan unit vector)
normal_magnitudes = np.linalg.norm(normals_manual, axis=1)
print(f"  Rata-rata magnitude: {np.mean(normal_magnitudes):.6f}")
print(f"  Min magnitude      : {np.min(normal_magnitudes):.6f}")
print(f"  Max magnitude      : {np.max(normal_magnitudes):.6f}")
print()


# ========================================================
# 3. PERBANDINGAN DENGAN OPEN3D (JIKA TERSEDIA)
# ========================================================

# Mencetak header bagian perbandingan
print("=" * 60)
print("3. PERBANDINGAN DENGAN OPEN3D")
print("=" * 60)

# Menginisialisasi variabel untuk normal Open3D
normals_o3d = None

if HAS_OPEN3D:
    # Membuat point cloud Open3D dari subsample
    pcd_sub = o3d.geometry.PointCloud()
    pcd_sub.points = o3d.utility.Vector3dVector(points_sub)

    # Menghitung normal menggunakan Open3D
    pcd_sub.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))

    # Mengambil normal hasil Open3D
    normals_o3d = np.asarray(pcd_sub.normals)

    # Menghitung perbedaan sudut antara normal manual dan Open3D
    dot_products = np.sum(normals_manual * normals_o3d, axis=1)

    # Mengklip dot product ke range [-1, 1] untuk menghindari error arccos
    dot_products = np.clip(dot_products, -1, 1)

    # Menghitung sudut perbedaan dalam derajat
    angle_diffs = np.degrees(np.arccos(np.abs(dot_products)))

    # Mencetak statistik perbandingan
    print(f"  [Open3D] Normal dihitung: {len(normals_o3d)}")
    print(f"  Perbedaan sudut rata-rata     : {np.mean(angle_diffs):.2f} derajat")
    print(f"  Perbedaan sudut median         : {np.median(angle_diffs):.2f} derajat")
    print(f"  Perbedaan sudut max            : {np.max(angle_diffs):.2f} derajat")
    print(f"  Titik dengan perbedaan < 5 deg : {np.sum(angle_diffs < 5) / len(angle_diffs) * 100:.1f}%")
else:
    # Mencetak info bahwa Open3D tidak tersedia
    print("  [INFO] Open3D tidak tersedia, perbandingan dilewati")
    print("  Menggunakan hasil PCA manual saja")

print()


# ========================================================
# 4. ORIENTASI NORMAL SECARA KONSISTEN
# ========================================================

# Mencetak header bagian orientasi normal
print("=" * 60)
print("4. ORIENTASI NORMAL SECARA KONSISTEN")
print("=" * 60)

# Menghitung centroid point cloud
centroid = np.mean(points_sub, axis=0)

# Menentukan viewpoint di atas objek
viewpoint = centroid + np.array([0, 0, 5])
print(f"  Centroid           : ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
print(f"  Viewpoint          : ({viewpoint[0]:.4f}, {viewpoint[1]:.4f}, {viewpoint[2]:.4f})")

# Menghitung jumlah normal yang perlu dibalik sebelum orientasi
# Untuk mengecek berapa banyak yang akan diflip
to_viewpoint_vecs = viewpoint - points_sub
dots_before = np.sum(normals_manual * to_viewpoint_vecs, axis=1)
flipped_before = np.sum(dots_before < 0)
print(f"  Normal mengarah salah sebelum orientasi: {flipped_before}/{len(normals_manual)}")

# Mengorientasikan normal secara konsisten
normals_oriented = orient_normals_consistently(points_sub, normals_manual.copy(), viewpoint)

# Mengecek setelah orientasi
dots_after = np.sum(normals_oriented * to_viewpoint_vecs, axis=1)
flipped_after = np.sum(dots_after < 0)
print(f"  Normal mengarah salah sesudah orientasi: {flipped_after}/{len(normals_oriented)}")
print()


# ========================================================
# 5. VISUALISASI NORMAL SEBAGAI PANAH (QUIVER 3D)
# ========================================================

# Mencetak header bagian visualisasi quiver
print("=" * 60)
print("5. VISUALISASI NORMAL SEBAGAI PANAH (QUIVER 3D)")
print("=" * 60)

# Membuat figure untuk visualisasi normal
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 7),
                            subplot_kw={'projection': '3d'})

# Menentukan jumlah titik untuk ditampilkan (subsample)
n_display = min(500, len(points_sub))
display_idx = np.random.choice(len(points_sub), n_display, replace=False)

# Mengambil titik dan normal untuk ditampilkan
pts_display = points_sub[display_idx]
nrm_display = normals_oriented[display_idx]

# Menentukan panjang panah normal
arrow_length = 0.08

# --- Subplot kiri: Normal sebelum orientasi ---
nrm_before = normals_manual[display_idx]

# Menggambar titik-titik
axes1[0].scatter(pts_display[:, 0], pts_display[:, 1], pts_display[:, 2],
                 c='blue', s=2, alpha=0.5)

# Menggambar panah normal menggunakan quiver
axes1[0].quiver(pts_display[:, 0], pts_display[:, 1], pts_display[:, 2],
                nrm_before[:, 0] * arrow_length,
                nrm_before[:, 1] * arrow_length,
                nrm_before[:, 2] * arrow_length,
                color='red', alpha=0.4, linewidth=0.5,
                arrow_length_ratio=0.3)

# Mengatur judul subplot
axes1[0].set_title("Normal Sebelum Orientasi", fontsize=11)

# Mengatur label sumbu
axes1[0].set_xlabel('X')
axes1[0].set_ylabel('Y')
axes1[0].set_zlabel('Z')

# --- Subplot kanan: Normal sesudah orientasi ---
# Menggambar titik-titik
axes1[1].scatter(pts_display[:, 0], pts_display[:, 1], pts_display[:, 2],
                 c='blue', s=2, alpha=0.5)

# Menggambar panah normal yang sudah terorientasi
axes1[1].quiver(pts_display[:, 0], pts_display[:, 1], pts_display[:, 2],
                nrm_display[:, 0] * arrow_length,
                nrm_display[:, 1] * arrow_length,
                nrm_display[:, 2] * arrow_length,
                color='green', alpha=0.4, linewidth=0.5,
                arrow_length_ratio=0.3)

# Mengatur judul subplot
axes1[1].set_title("Normal Sesudah Orientasi", fontsize=11)

# Mengatur label sumbu
axes1[1].set_xlabel('X')
axes1[1].set_ylabel('Y')
axes1[1].set_zlabel('Z')

# Mengatur judul utama
fig1.suptitle("Estimasi Normal - Sebelum vs Sesudah Orientasi",
              fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi
output_path1 = os.path.join(OUTPUT_DIR, "03_normals_quiver.png")
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"  Tersimpan: {os.path.basename(output_path1)}")

# Menutup figure
plt.close()


# ========================================================
# 6. PEWARNAAN BERDASARKAN ARAH NORMAL (RGB = XYZ)
# ========================================================

# Mencetak header bagian pewarnaan normal
print()
print("=" * 60)
print("6. PEWARNAAN BERDASARKAN ARAH NORMAL (RGB = XYZ)")
print("=" * 60)

# Mengkonversi normal menjadi warna RGB
# Memetakan komponen normal dari [-1, 1] ke [0, 1]
normal_colors = (normals_oriented + 1.0) / 2.0

# Mengklip nilai ke range [0, 1]
normal_colors = np.clip(normal_colors, 0, 1)

# Mencetak informasi pewarnaan
print(f"  Pemetaan: Normal XYZ -> RGB")
print(f"    X normal (-1..+1) -> Red   (0..1)")
print(f"    Y normal (-1..+1) -> Green (0..1)")
print(f"    Z normal (-1..+1) -> Blue  (0..1)")

# Membuat figure untuk visualisasi warna normal
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6),
                            subplot_kw={'projection': '3d'})

# Subsample untuk performa
n_color_display = min(2000, len(points_sub))
color_idx = np.random.choice(len(points_sub), n_color_display, replace=False)

# --- Subplot kiri: Pewarnaan berdasarkan normal ---
axes2[0].scatter(points_sub[color_idx, 0],
                 points_sub[color_idx, 1],
                 points_sub[color_idx, 2],
                 c=normal_colors[color_idx], s=3, alpha=0.7)

# Mengatur judul
axes2[0].set_title("Warna = Arah Normal (RGB=XYZ)", fontsize=11)
axes2[0].set_xlabel('X')
axes2[0].set_ylabel('Y')
axes2[0].set_zlabel('Z')

# --- Subplot kanan: Pewarnaan berdasarkan Z normal saja ---
# Menormalisasi komponen Z normal untuk colormap
z_normal = normals_oriented[color_idx, 2]
z_norm_mapped = (z_normal + 1.0) / 2.0

# Menggambar dengan colormap
scatter2 = axes2[1].scatter(points_sub[color_idx, 0],
                             points_sub[color_idx, 1],
                             points_sub[color_idx, 2],
                             c=z_norm_mapped, cmap='coolwarm', s=3, alpha=0.7)

# Menambahkan colorbar
plt.colorbar(scatter2, ax=axes2[1], shrink=0.6, label='Z Normal Component')

# Mengatur judul
axes2[1].set_title("Warna = Komponen Z Normal", fontsize=11)
axes2[1].set_xlabel('X')
axes2[1].set_ylabel('Y')
axes2[1].set_zlabel('Z')

# Mengatur judul utama
fig2.suptitle("Pewarnaan Point Cloud Berdasarkan Arah Normal",
              fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi
output_path2 = os.path.join(OUTPUT_DIR, "03_normal_coloring.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"  Tersimpan: {os.path.basename(output_path2)}")

# Menutup figure
plt.close()


# ========================================================
# 7. ANALISIS CURVATURE DARI EIGENVALUE
# ========================================================

# Mencetak header bagian curvature
print()
print("=" * 60)
print("7. ANALISIS CURVATURE DARI EIGENVALUE")
print("=" * 60)

# Menghitung curvature dari eigenvalue
# Curvature = lambda_min / (lambda_1 + lambda_2 + lambda_3)
eigenvalue_sum = np.sum(eigenvalues_all, axis=1)

# Menghindari pembagian dengan nol
eigenvalue_sum = np.maximum(eigenvalue_sum, 1e-10)

# Menghitung curvature (eigenvalue terkecil / jumlah semua eigenvalue)
curvature = eigenvalues_all[:, 2] / eigenvalue_sum

# Mencetak statistik curvature
print(f"  Curvature minimum    : {np.min(curvature):.6f}")
print(f"  Curvature maksimum   : {np.max(curvature):.6f}")
print(f"  Curvature rata-rata  : {np.mean(curvature):.6f}")
print(f"  Curvature median     : {np.median(curvature):.6f}")
print(f"  Curvature std dev    : {np.std(curvature):.6f}")

# Mengklasifikasikan titik berdasarkan curvature
# Titik dengan curvature rendah = datar, tinggi = berbelok
threshold_flat = np.percentile(curvature, 25)
threshold_curved = np.percentile(curvature, 75)
n_flat = np.sum(curvature <= threshold_flat)
n_medium = np.sum((curvature > threshold_flat) & (curvature <= threshold_curved))
n_curved = np.sum(curvature > threshold_curved)
print(f"\n  Klasifikasi curvature:")
print(f"    Datar (< {threshold_flat:.4f})      : {n_flat} titik ({n_flat / len(curvature) * 100:.1f}%)")
print(f"    Sedang                     : {n_medium} titik ({n_medium / len(curvature) * 100:.1f}%)")
print(f"    Melengkung (> {threshold_curved:.4f}) : {n_curved} titik ({n_curved / len(curvature) * 100:.1f}%)")

# Membuat visualisasi curvature
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot kiri: Histogram curvature ---
axes3[0].hist(curvature, bins=50, color='steelblue', edgecolor='black', alpha=0.7)

# Menambahkan garis threshold
axes3[0].axvline(x=threshold_flat, color='green', linestyle='--', linewidth=1.5,
                 label=f'Flat (P25={threshold_flat:.4f})')
axes3[0].axvline(x=threshold_curved, color='red', linestyle='--', linewidth=1.5,
                 label=f'Curved (P75={threshold_curved:.4f})')

# Mengatur judul
axes3[0].set_title("Distribusi Curvature", fontsize=12)
axes3[0].set_xlabel("Curvature")
axes3[0].set_ylabel("Frekuensi")
axes3[0].legend(fontsize=9)

# --- Subplot kanan: Eigenvalue scatter ---
axes3[1].scatter(eigenvalues_all[:, 0], eigenvalues_all[:, 2],
                 c=curvature, cmap='hot', s=3, alpha=0.6)

# Mengatur judul
axes3[1].set_title("Eigenvalue: Terbesar vs Terkecil", fontsize=12)
axes3[1].set_xlabel("Eigenvalue Terbesar (λ₁)")
axes3[1].set_ylabel("Eigenvalue Terkecil (λ₃)")

# Menambahkan colorbar
scatter_cb = axes3[1].scatter([], [], c=[], cmap='hot')
plt.colorbar(plt.cm.ScalarMappable(cmap='hot',
             norm=plt.Normalize(vmin=curvature.min(), vmax=curvature.max())),
             ax=axes3[1], shrink=0.8, label='Curvature')

# Mengatur judul utama
fig3.suptitle("Analisis Curvature dari Eigenvalue PCA",
              fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi
output_path3 = os.path.join(OUTPUT_DIR, "03_curvature_analysis.png")
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"\n  Tersimpan: {os.path.basename(output_path3)}")

# Menutup figure
plt.close()


# --- Visualisasi curvature pada point cloud 3D ---
# Membuat figure 3D untuk visualisasi curvature
fig4 = plt.figure(figsize=(10, 8))

# Membuat subplot 3D
ax4 = fig4.add_subplot(111, projection='3d')

# Subsample untuk performa
n_curv_display = min(2000, len(points_sub))
curv_idx = np.random.choice(len(points_sub), n_curv_display, replace=False)

# Menggambar point cloud dengan warna berdasarkan curvature
scatter4 = ax4.scatter(points_sub[curv_idx, 0],
                        points_sub[curv_idx, 1],
                        points_sub[curv_idx, 2],
                        c=curvature[curv_idx], cmap='hot', s=4, alpha=0.7)

# Menambahkan colorbar
plt.colorbar(scatter4, ax=ax4, shrink=0.6, label='Curvature')

# Mengatur label sumbu
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')

# Mengatur judul
ax4.set_title("Curvature pada Point Cloud\n(Merah = Melengkung, Gelap = Datar)",
              fontsize=13, fontweight='bold')

# Menyimpan visualisasi 3D curvature
output_path4 = os.path.join(OUTPUT_DIR, "03_curvature_3d.png")
plt.savefig(output_path4, dpi=150, bbox_inches='tight')
print(f"  Tersimpan: {os.path.basename(output_path4)}")

# Menutup figure
plt.close()


# ========================================================
# 8. RINGKASAN
# ========================================================

# Mencetak header ringkasan
print()
print("=" * 60)
print("RINGKASAN PERCOBAAN 3")
print("=" * 60)
print(f"  Point cloud dimuat   : {len(points)} titik")
print(f"  Subsample digunakan  : {len(points_sub)} titik")
print(f"  K tetangga PCA       : {k_neighbors}")
print(f"  Normal dihitung      : {len(normals_oriented)}")
print(f"  Normal diflip        : {flipped_before} dari {len(normals_manual)}")
print(f"  Curvature rata-rata  : {np.mean(curvature):.6f}")
if HAS_OPEN3D and normals_o3d is not None:
    print(f"  Perbedaan dgn Open3D : {np.mean(angle_diffs):.2f} derajat (rata-rata)")
else:
    print(f"  Open3D comparison    : Tidak tersedia")
print(f"  Visualisasi disimpan : 4 file di output/")
print("=" * 60)
