"""
==========================================================
PERCOBAAN 2: POINT CLOUD FILTERING DAN DOWNSAMPLING
Menerapkan teknik preprocessing: voxel downsampling,
statistical outlier removal, radius outlier removal.

Fungsi utama:
- open3d voxel_down_sample() (atau manual implementation)
- KDTree / scipy.spatial
- numpy operations
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
print("PERCOBAAN 2: POINT CLOUD FILTERING DAN DOWNSAMPLING")
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


def save_ply_simple(filepath, points, colors=None):
    """
    Menyimpan point cloud ke format PLY ASCII.
    """
    # Membuka file untuk menulis
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if colors is not None:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
            f.write(line + "\n")


def voxel_downsample_manual(points, voxel_size, colors=None):
    """
    Melakukan voxel downsampling secara manual menggunakan grid.
    Setiap voxel diwakili oleh rata-rata titik di dalamnya.
    """
    # Menghitung index voxel untuk setiap titik
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Membuat dictionary untuk mengelompokkan titik per voxel
    voxel_dict = {}

    # Mengiterasi setiap titik
    for i in range(len(points)):
        # Membuat key dari index voxel (sebagai tuple)
        key = tuple(voxel_indices[i])

        # Menambahkan titik ke voxel yang sesuai
        if key not in voxel_dict:
            voxel_dict[key] = {'points': [], 'colors': []}

        # Menyimpan titik
        voxel_dict[key]['points'].append(points[i])

        # Menyimpan warna jika tersedia
        if colors is not None:
            voxel_dict[key]['colors'].append(colors[i])

    # Menghitung rata-rata titik untuk setiap voxel
    new_points = []
    new_colors = []

    # Mengiterasi setiap voxel
    for key, val in voxel_dict.items():
        # Menghitung centroid voxel
        centroid = np.mean(val['points'], axis=0)
        new_points.append(centroid)

        # Menghitung rata-rata warna voxel
        if colors is not None and val['colors']:
            avg_color = np.mean(val['colors'], axis=0).astype(np.uint8)
            new_colors.append(avg_color)

    # Mengkonversi ke numpy array
    new_points = np.array(new_points)
    new_colors = np.array(new_colors) if new_colors else None

    # Mengembalikan hasil downsampling
    return new_points, new_colors


def statistical_outlier_removal_manual(points, nb_neighbors=20, std_ratio=2.0, colors=None):
    """
    Menghapus outlier berdasarkan statistik jarak rata-rata ke tetangga terdekat.
    Titik dianggap outlier jika jarak rata-ratanya melebihi threshold.
    """
    # Membangun KDTree untuk pencarian tetangga
    kdtree = KDTree(points)

    # Mencari k tetangga terdekat untuk setiap titik
    distances, _ = kdtree.query(points, k=nb_neighbors + 1)

    # Menghitung jarak rata-rata ke tetangga (mengecualikan diri sendiri, index 0)
    mean_distances = np.mean(distances[:, 1:], axis=1)

    # Menghitung statistik global jarak rata-rata
    global_mean = np.mean(mean_distances)

    # Menghitung standar deviasi global
    global_std = np.std(mean_distances)

    # Menentukan threshold untuk inlier
    threshold = global_mean + std_ratio * global_std

    # Membuat mask untuk titik-titik inlier
    inlier_mask = mean_distances < threshold

    # Menerapkan mask untuk mendapatkan inlier
    inlier_points = points[inlier_mask]

    # Menerapkan mask pada warna jika tersedia
    inlier_colors = colors[inlier_mask] if colors is not None else None

    # Mengembalikan titik inlier, warna, dan mask
    return inlier_points, inlier_colors, inlier_mask


def radius_outlier_removal_manual(points, radius=0.1, min_neighbors=5, colors=None):
    """
    Menghapus outlier berdasarkan jumlah tetangga dalam radius tertentu.
    Titik dianggap outlier jika jumlah tetangganya kurang dari threshold.
    """
    # Membangun KDTree untuk pencarian radius
    kdtree = KDTree(points)

    # Mencari semua tetangga dalam radius untuk setiap titik
    neighbor_counts = np.zeros(len(points), dtype=int)

    # Mengiterasi setiap titik
    for i in range(len(points)):
        # Menghitung jumlah titik dalam radius
        neighbors = kdtree.query_ball_point(points[i], radius)
        # Menyimpan jumlah tetangga (dikurangi 1 untuk mengecualikan diri sendiri)
        neighbor_counts[i] = len(neighbors) - 1

    # Membuat mask untuk titik-titik inlier
    inlier_mask = neighbor_counts >= min_neighbors

    # Menerapkan mask
    inlier_points = points[inlier_mask]

    # Menerapkan mask pada warna
    inlier_colors = colors[inlier_mask] if colors is not None else None

    # Mengembalikan hasil
    return inlier_points, inlier_colors, inlier_mask


# ========================================================
# 1. MEMUAT POINT CLOUD DAN MENAMBAHKAN NOISE
# ========================================================

# Mencetak header bagian memuat data
print("=" * 60)
print("1. MEMUAT POINT CLOUD DAN MENAMBAHKAN NOISE")
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
    pcd_orig = o3d.io.read_point_cloud(ply_path)
    points_orig = np.asarray(pcd_orig.points)
    has_col = pcd_orig.has_colors()
    colors_orig = (np.asarray(pcd_orig.colors) * 255).astype(np.uint8) if has_col else None
else:
    # Memuat dengan parser manual
    points_orig, colors_orig = load_ply_manual(ply_path)

# Mencetak jumlah titik asli
print(f"  Titik asli: {len(points_orig)}")

# Menambahkan noise Gaussian pada point cloud
noise_std = 0.02
noise = np.random.normal(0, noise_std, points_orig.shape)

# Menerapkan noise pada titik
points_noisy = points_orig.copy() + noise

# Menambahkan outlier acak (titik-titik jauh dari objek)
n_outliers = int(len(points_orig) * 0.05)

# Menghitung batas bounding box
bb_min = np.min(points_orig, axis=0)
bb_max = np.max(points_orig, axis=0)
bb_range = bb_max - bb_min

# Membuat titik outlier di luar bounding box
outlier_points = bb_min - bb_range * 0.5 + np.random.rand(n_outliers, 3) * bb_range * 3

# Menggabungkan titik noisy dengan outlier
points_noisy_with_outliers = np.vstack([points_noisy, outlier_points])

# Membuat warna untuk outlier (merah)
if colors_orig is not None:
    outlier_colors = np.full((n_outliers, 3), [255, 0, 0], dtype=np.uint8)
    colors_noisy = np.vstack([colors_orig, outlier_colors])
else:
    colors_noisy = None

# Mencetak informasi noise
print(f"  Noise ditambahkan: Gaussian sigma={noise_std}")
print(f"  Outlier ditambahkan: {n_outliers} titik")
print(f"  Total titik (noisy + outlier): {len(points_noisy_with_outliers)}")
print()


# ========================================================
# 2. VOXEL DOWNSAMPLING
# ========================================================

# Mencetak header bagian voxel downsampling
print("=" * 60)
print("2. VOXEL DOWNSAMPLING")
print("=" * 60)

# Menentukan ukuran voxel
voxel_size = 0.08

# Melakukan voxel downsampling
if HAS_OPEN3D:
    # Membuat point cloud Open3D dari data noisy
    pcd_noisy = o3d.geometry.PointCloud()
    pcd_noisy.points = o3d.utility.Vector3dVector(points_noisy_with_outliers)
    if colors_noisy is not None:
        pcd_noisy.colors = o3d.utility.Vector3dVector(colors_noisy.astype(np.float64) / 255.0)

    # Menerapkan voxel downsampling Open3D
    pcd_voxel = pcd_noisy.voxel_down_sample(voxel_size=voxel_size)
    points_voxel = np.asarray(pcd_voxel.points)
    colors_voxel = (np.asarray(pcd_voxel.colors) * 255).astype(np.uint8) if pcd_voxel.has_colors() else None
    print(f"  [Open3D] Voxel downsampling selesai")
else:
    # Menerapkan voxel downsampling manual
    points_voxel, colors_voxel = voxel_downsample_manual(
        points_noisy_with_outliers, voxel_size, colors_noisy)
    print(f"  [Manual] Voxel downsampling selesai")

# Menghitung persentase reduksi
reduction_voxel = (1 - len(points_voxel) / len(points_noisy_with_outliers)) * 100

# Mencetak hasil voxel downsampling
print(f"  Ukuran voxel         : {voxel_size}")
print(f"  Titik sebelum        : {len(points_noisy_with_outliers)}")
print(f"  Titik sesudah        : {len(points_voxel)}")
print(f"  Reduksi              : {reduction_voxel:.1f}%")
print()


# ========================================================
# 3. STATISTICAL OUTLIER REMOVAL
# ========================================================

# Mencetak header bagian statistical outlier removal
print("=" * 60)
print("3. STATISTICAL OUTLIER REMOVAL")
print("=" * 60)

# Menentukan parameter untuk statistical outlier removal
nb_neighbors_stat = 20
std_ratio_stat = 2.0

# Melakukan statistical outlier removal
if HAS_OPEN3D:
    # Membuat point cloud Open3D dari data voxel
    pcd_for_stat = o3d.geometry.PointCloud()
    pcd_for_stat.points = o3d.utility.Vector3dVector(points_voxel)
    if colors_voxel is not None:
        pcd_for_stat.colors = o3d.utility.Vector3dVector(colors_voxel.astype(np.float64) / 255.0)

    # Menerapkan statistical outlier removal Open3D
    pcd_stat, ind_stat = pcd_for_stat.remove_statistical_outlier(
        nb_neighbors=nb_neighbors_stat, std_ratio=std_ratio_stat)
    points_stat = np.asarray(pcd_stat.points)
    colors_stat = (np.asarray(pcd_stat.colors) * 255).astype(np.uint8) if pcd_stat.has_colors() else None

    # Membuat mask inlier
    stat_mask = np.zeros(len(points_voxel), dtype=bool)
    stat_mask[ind_stat] = True
    print(f"  [Open3D] Statistical outlier removal selesai")
else:
    # Menerapkan statistical outlier removal manual
    points_stat, colors_stat, stat_mask = statistical_outlier_removal_manual(
        points_voxel, nb_neighbors=nb_neighbors_stat,
        std_ratio=std_ratio_stat, colors=colors_voxel)
    print(f"  [Manual] Statistical outlier removal selesai")

# Menghitung persentase yang dihapus
outliers_stat = len(points_voxel) - len(points_stat)
reduction_stat = (outliers_stat / len(points_voxel)) * 100

# Mencetak hasil
print(f"  Nb neighbors         : {nb_neighbors_stat}")
print(f"  Std ratio            : {std_ratio_stat}")
print(f"  Titik sebelum        : {len(points_voxel)}")
print(f"  Titik sesudah        : {len(points_stat)}")
print(f"  Outlier dihapus      : {outliers_stat} ({reduction_stat:.1f}%)")
print()


# ========================================================
# 4. RADIUS OUTLIER REMOVAL
# ========================================================

# Mencetak header bagian radius outlier removal
print("=" * 60)
print("4. RADIUS OUTLIER REMOVAL")
print("=" * 60)

# Menentukan parameter radius outlier removal
search_radius = 0.15
min_neighbors_rad = 5

# Melakukan radius outlier removal pada data setelah voxel downsample
if HAS_OPEN3D:
    # Menerapkan radius outlier removal Open3D
    pcd_for_rad = o3d.geometry.PointCloud()
    pcd_for_rad.points = o3d.utility.Vector3dVector(points_voxel)
    if colors_voxel is not None:
        pcd_for_rad.colors = o3d.utility.Vector3dVector(colors_voxel.astype(np.float64) / 255.0)

    pcd_rad, ind_rad = pcd_for_rad.remove_radius_outlier(
        nb_points=min_neighbors_rad, radius=search_radius)
    points_rad = np.asarray(pcd_rad.points)
    colors_rad = (np.asarray(pcd_rad.colors) * 255).astype(np.uint8) if pcd_rad.has_colors() else None

    rad_mask = np.zeros(len(points_voxel), dtype=bool)
    rad_mask[ind_rad] = True
    print(f"  [Open3D] Radius outlier removal selesai")
else:
    # Menerapkan radius outlier removal manual
    points_rad, colors_rad, rad_mask = radius_outlier_removal_manual(
        points_voxel, radius=search_radius,
        min_neighbors=min_neighbors_rad, colors=colors_voxel)
    print(f"  [Manual] Radius outlier removal selesai")

# Menghitung persentase yang dihapus
outliers_rad = len(points_voxel) - len(points_rad)
reduction_rad = (outliers_rad / len(points_voxel)) * 100

# Mencetak hasil
print(f"  Radius pencarian     : {search_radius}")
print(f"  Min neighbors        : {min_neighbors_rad}")
print(f"  Titik sebelum        : {len(points_voxel)}")
print(f"  Titik sesudah        : {len(points_rad)}")
print(f"  Outlier dihapus      : {outliers_rad} ({reduction_rad:.1f}%)")
print()


# ========================================================
# 5. PERBANDINGAN SEBELUM DAN SESUDAH FILTERING
# ========================================================

# Mencetak header bagian perbandingan
print("=" * 60)
print("5. PERBANDINGAN HASIL FILTERING")
print("=" * 60)

# Mencetak tabel perbandingan
print(f"  {'Tahap':<30s} {'Jumlah Titik':>15s} {'Reduksi':>10s}")
print(f"  {'-' * 55}")
print(f"  {'Original':<30s} {len(points_orig):>15d} {'—':>10s}")
print(f"  {'+ Noise + Outlier':<30s} {len(points_noisy_with_outliers):>15d} {'—':>10s}")
print(f"  {'Voxel Downsample':<30s} {len(points_voxel):>15d} {reduction_voxel:>9.1f}%")
print(f"  {'Statistical Outlier Removal':<30s} {len(points_stat):>15d} {reduction_stat:>9.1f}%")
print(f"  {'Radius Outlier Removal':<30s} {len(points_rad):>15d} {reduction_rad:>9.1f}%")
print()


# ========================================================
# 6. VISUALISASI HASIL FILTERING
# ========================================================

# Mencetak header bagian visualisasi
print("=" * 60)
print("6. VISUALISASI SEMUA HASIL")
print("=" * 60)

# --- Visualisasi 1: Grid 2x3 semua tahapan ---
# Membuat figure dengan 2x3 subplot
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12),
                            subplot_kw={'projection': '3d'})

# Menentukan data dan judul untuk setiap subplot
datasets = [
    ("Original", points_orig),
    ("+ Noise + Outlier", points_noisy_with_outliers),
    ("Voxel Downsample", points_voxel),
    ("Stat. Outlier Removal", points_stat),
    ("Radius Outlier Removal", points_rad),
]

# Menyiapkan subplot keenam untuk perbandingan jumlah
plot_idx = 0

# Mengiterasi setiap dataset
for idx, (title, pts) in enumerate(datasets):
    # Menentukan posisi subplot
    row, col = divmod(idx, 3)
    ax = axes1[row, col]

    # Subsample untuk performa
    max_display = 2000
    if len(pts) > max_display:
        sample_idx = np.random.choice(len(pts), max_display, replace=False)
        pts_show = pts[sample_idx]
    else:
        pts_show = pts

    # Menormalisasi Z untuk colormap
    z_n = (pts_show[:, 2] - pts_show[:, 2].min()) / (pts_show[:, 2].max() - pts_show[:, 2].min() + 1e-8)

    # Menggambar scatter plot
    ax.scatter(pts_show[:, 0], pts_show[:, 1], pts_show[:, 2],
               c=z_n, cmap='viridis', s=1, alpha=0.6)

    # Mengatur judul
    ax.set_title(f"{title}\n({len(pts)} titik)", fontsize=10)

    # Mengatur label
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)

# Menggunakan subplot terakhir untuk grafik batang perbandingan
ax_bar = fig1.add_subplot(2, 3, 6)

# Menentukan nama tahapan
stage_names = ["Original", "Noisy", "Voxel", "Stat.", "Radius"]

# Menentukan jumlah titik per tahapan
stage_counts = [len(points_orig), len(points_noisy_with_outliers),
                len(points_voxel), len(points_stat), len(points_rad)]

# Menentukan warna batang
bar_colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

# Menggambar grafik batang
bars = ax_bar.bar(stage_names, stage_counts, color=bar_colors, edgecolor='black')

# Menambahkan label nilai di atas setiap batang
for bar, count in zip(bars, stage_counts):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha='center', va='bottom', fontsize=8)

# Mengatur judul grafik batang
ax_bar.set_title("Perbandingan Jumlah Titik", fontsize=10)

# Mengatur label sumbu Y
ax_bar.set_ylabel("Jumlah Titik")

# Menghilangkan subplot 3D yang tidak terpakai (posisi 2,2)
axes1[1, 2].remove()

# Mengatur judul utama
fig1.suptitle("Point Cloud Filtering dan Downsampling - Semua Tahapan",
              fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi
output_path1 = os.path.join(OUTPUT_DIR, "02_filtering_all_stages.png")
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"  Tersimpan: {os.path.basename(output_path1)}")

# Menutup figure
plt.close()


# --- Visualisasi 2: Perbandingan inlier vs outlier ---
# Membuat figure untuk menunjukkan inlier dan outlier
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6),
                            subplot_kw={'projection': '3d'})

# Menggambar inlier dan outlier dari statistical removal
# Mengambil outlier (titik yang dihapus)
stat_outlier_pts = points_voxel[~stat_mask]
stat_inlier_pts = points_voxel[stat_mask]

# Subsample untuk performa
max_show = 2000

# Menampilkan inlier (hijau) dan outlier (merah) untuk stat removal
if len(stat_inlier_pts) > max_show:
    idx_in = np.random.choice(len(stat_inlier_pts), max_show, replace=False)
else:
    idx_in = np.arange(len(stat_inlier_pts))

# Menggambar inlier
axes2[0].scatter(stat_inlier_pts[idx_in, 0], stat_inlier_pts[idx_in, 1],
                 stat_inlier_pts[idx_in, 2], c='green', s=1, alpha=0.5, label='Inlier')

# Menggambar outlier
if len(stat_outlier_pts) > 0:
    axes2[0].scatter(stat_outlier_pts[:, 0], stat_outlier_pts[:, 1],
                     stat_outlier_pts[:, 2], c='red', s=5, alpha=0.8, label='Outlier')

# Mengatur judul
axes2[0].set_title(f"Statistical Outlier Removal\nInlier: {len(stat_inlier_pts)}, "
                   f"Outlier: {len(stat_outlier_pts)}", fontsize=10)
axes2[0].legend(fontsize=8)
axes2[0].set_xlabel('X')
axes2[0].set_ylabel('Y')
axes2[0].set_zlabel('Z')

# Menampilkan inlier dan outlier untuk radius removal
rad_outlier_pts = points_voxel[~rad_mask]
rad_inlier_pts = points_voxel[rad_mask]

if len(rad_inlier_pts) > max_show:
    idx_in_r = np.random.choice(len(rad_inlier_pts), max_show, replace=False)
else:
    idx_in_r = np.arange(len(rad_inlier_pts))

# Menggambar inlier
axes2[1].scatter(rad_inlier_pts[idx_in_r, 0], rad_inlier_pts[idx_in_r, 1],
                 rad_inlier_pts[idx_in_r, 2], c='green', s=1, alpha=0.5, label='Inlier')

# Menggambar outlier
if len(rad_outlier_pts) > 0:
    axes2[1].scatter(rad_outlier_pts[:, 0], rad_outlier_pts[:, 1],
                     rad_outlier_pts[:, 2], c='red', s=5, alpha=0.8, label='Outlier')

# Mengatur judul
axes2[1].set_title(f"Radius Outlier Removal\nInlier: {len(rad_inlier_pts)}, "
                   f"Outlier: {len(rad_outlier_pts)}", fontsize=10)
axes2[1].legend(fontsize=8)
axes2[1].set_xlabel('X')
axes2[1].set_ylabel('Y')
axes2[1].set_zlabel('Z')

# Mengatur judul utama
fig2.suptitle("Inlier vs Outlier - Perbandingan Metode Removal",
              fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi
output_path2 = os.path.join(OUTPUT_DIR, "02_inlier_vs_outlier.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"  Tersimpan: {os.path.basename(output_path2)}")

# Menutup figure
plt.close()


# ========================================================
# 7. MENYIMPAN POINT CLOUD HASIL FILTERING
# ========================================================

# Mencetak header bagian penyimpanan
print()
print("=" * 60)
print("7. MENYIMPAN POINT CLOUD HASIL FILTERING")
print("=" * 60)

# Menyimpan point cloud setelah voxel downsampling
voxel_output_path = os.path.join(OUTPUT_DIR, "02_voxel_downsampled.ply")
save_ply_simple(voxel_output_path, points_voxel, colors_voxel)
print(f"  Tersimpan: {os.path.basename(voxel_output_path)} ({len(points_voxel)} titik)")

# Menyimpan point cloud setelah statistical outlier removal
stat_output_path = os.path.join(OUTPUT_DIR, "02_stat_filtered.ply")
save_ply_simple(stat_output_path, points_stat, colors_stat)
print(f"  Tersimpan: {os.path.basename(stat_output_path)} ({len(points_stat)} titik)")

# Menyimpan point cloud setelah radius outlier removal
rad_output_path = os.path.join(OUTPUT_DIR, "02_radius_filtered.ply")
save_ply_simple(rad_output_path, points_rad, colors_rad)
print(f"  Tersimpan: {os.path.basename(rad_output_path)} ({len(points_rad)} titik)")
print()


# ========================================================
# 8. RINGKASAN
# ========================================================

# Mencetak header ringkasan
print("=" * 60)
print("RINGKASAN PERCOBAAN 2")
print("=" * 60)
print(f"  Point cloud original : {len(points_orig)} titik")
print(f"  Setelah noise+outlier: {len(points_noisy_with_outliers)} titik")
print(f"  Voxel downsample     : {len(points_voxel)} titik (reduksi {reduction_voxel:.1f}%)")
print(f"  Stat outlier removal : {len(points_stat)} titik (dihapus {outliers_stat})")
print(f"  Radius outlier removal: {len(points_rad)} titik (dihapus {outliers_rad})")
print(f"  Visualisasi disimpan : 2 file gambar + 3 file PLY di output/")
print("=" * 60)
