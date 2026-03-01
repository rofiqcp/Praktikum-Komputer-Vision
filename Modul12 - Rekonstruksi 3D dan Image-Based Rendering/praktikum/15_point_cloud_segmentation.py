"""
==========================================================
PERCOBAAN 15: SEGMENTASI POINT CLOUD
Mempelajari teknik segmentasi point cloud: plane fitting
(RANSAC), clustering (DBSCAN-like), dan region-based
segmentation.

Fungsi utama:
- open3d segment_plane() (atau manual RANSAC)
- scipy.spatial KDTree
- numpy random sampling
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

# Mengimpor KDTree dari scipy untuk pencarian tetangga terdekat
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
    print("[INFO] Open3D tidak tersedia, menggunakan fallback numpy/scipy")

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
print("PERCOBAAN 15: SEGMENTASI POINT CLOUD")
print("=" * 60)
print()


# ========================================================
# FUNGSI UTILITAS
# ========================================================

def generate_scene_point_cloud():
    """
    Membuat point cloud sintetis berisi lantai datar dan beberapa objek di atasnya.
    """
    # Mencetak informasi pembuatan scene
    print("  Membuat scene sintetis dengan objek di atas lantai...")

    # --- Membuat lantai datar (plane) ---
    num_floor = 2000
    # Membuat titik acak pada bidang XY
    floor_x = np.random.uniform(-3, 3, num_floor)
    floor_y = np.random.uniform(-3, 3, num_floor)
    # Lantai pada Z = 0 dengan sedikit noise
    floor_z = np.random.normal(0, 0.02, num_floor)
    # Menggabungkan menjadi array lantai
    floor_points = np.column_stack([floor_x, floor_y, floor_z])

    # --- Membuat objek 1: Bola ---
    num_sphere = 500
    theta = np.random.uniform(0, 2 * np.pi, num_sphere)
    phi = np.random.uniform(0, np.pi, num_sphere)
    r_sphere = 0.5
    # Menghitung koordinat bola
    sx = r_sphere * np.sin(phi) * np.cos(theta) + 1.0
    sy = r_sphere * np.sin(phi) * np.sin(theta) + 1.0
    sz = r_sphere * np.cos(phi) + r_sphere + 0.02
    # Menambahkan noise kecil
    sphere_points = np.column_stack([sx, sy, sz]) + np.random.normal(0, 0.01, (num_sphere, 3))

    # --- Membuat objek 2: Kubus kecil ---
    num_cube = 400
    # Membuat titik acak di permukaan kubus
    cube_points_list = []
    cube_center = np.array([-1.5, -1.0, 0.35])
    cube_size = 0.7
    for _ in range(num_cube):
        # Memilih face secara acak
        face = np.random.randint(0, 6)
        p = np.random.uniform(-cube_size/2, cube_size/2, 3)
        if face == 0:
            p[0] = cube_size / 2
        elif face == 1:
            p[0] = -cube_size / 2
        elif face == 2:
            p[1] = cube_size / 2
        elif face == 3:
            p[1] = -cube_size / 2
        elif face == 4:
            p[2] = cube_size / 2
        else:
            p[2] = -cube_size / 2
        cube_points_list.append(p + cube_center)
    # Mengkonversi ke numpy array
    cube_points = np.array(cube_points_list) + np.random.normal(0, 0.01, (num_cube, 3))

    # --- Membuat objek 3: Silinder ---
    num_cyl = 400
    cyl_theta = np.random.uniform(0, 2 * np.pi, num_cyl)
    cyl_h = np.random.uniform(0, 1.0, num_cyl)
    cyl_r = 0.3
    # Menghitung koordinat silinder
    cx = cyl_r * np.cos(cyl_theta) - 0.5
    cy = cyl_r * np.sin(cyl_theta) + 2.0
    cz = cyl_h + 0.02
    # Menggabungkan dengan noise
    cyl_points = np.column_stack([cx, cy, cz]) + np.random.normal(0, 0.01, (num_cyl, 3))

    # Menggabungkan semua titik
    all_points = np.vstack([floor_points, sphere_points, cube_points, cyl_points])

    # Membuat label ground truth untuk evaluasi
    labels_gt = np.concatenate([
        np.zeros(num_floor),
        np.ones(num_sphere),
        np.full(num_cube, 2),
        np.full(num_cyl, 3)
    ]).astype(int)

    # Mencetak informasi scene
    print(f"  Lantai: {num_floor} titik")
    print(f"  Bola: {num_sphere} titik")
    print(f"  Kubus: {num_cube} titik")
    print(f"  Silinder: {num_cyl} titik")
    print(f"  Total: {len(all_points)} titik")

    # Mengembalikan point cloud dan label
    return all_points, labels_gt


def ransac_plane_fitting(points, num_iterations=1000, distance_threshold=0.05):
    """
    Mendeteksi plane menggunakan RANSAC secara manual.
    """
    # Menghitung jumlah titik
    n = len(points)

    # Menginisialisasi variabel untuk model terbaik
    best_inliers = np.array([], dtype=int)
    best_plane = None

    # Mengiterasi RANSAC
    for iteration in range(num_iterations):
        # Memilih 3 titik secara acak
        sample_idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = points[sample_idx]

        # Menghitung normal plane dari 3 titik
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        # Memeriksa apakah normal valid (bukan nol)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue

        # Menormalisasi normal
        normal = normal / norm_len

        # Menghitung parameter d dari persamaan plane ax + by + cz + d = 0
        d = -np.dot(normal, p1)

        # Menghitung jarak setiap titik ke plane
        distances = np.abs(np.dot(points, normal) + d)

        # Menentukan inliers (titik yang dekat dengan plane)
        inlier_mask = distances < distance_threshold
        inlier_idx = np.where(inlier_mask)[0]

        # Memeriksa apakah ini model terbaik
        if len(inlier_idx) > len(best_inliers):
            best_inliers = inlier_idx
            best_plane = (normal[0], normal[1], normal[2], d)

    # Mengembalikan plane terbaik dan inliers
    return best_plane, best_inliers


def euclidean_clustering(points, eps=0.3, min_points=10):
    """
    Melakukan clustering menggunakan metode Euclidean berbasis KDTree
    (mirip DBSCAN).
    """
    # Membangun KDTree untuk pencarian tetangga efisien
    tree = KDTree(points)

    # Menginisialisasi label cluster (-1 = belum dikluster)
    labels = np.full(len(points), -1, dtype=int)

    # Menginisialisasi ID cluster
    cluster_id = 0

    # Mengiterasi setiap titik
    for i in range(len(points)):
        # Melewati titik yang sudah dikluster
        if labels[i] != -1:
            continue

        # Mencari tetangga dalam radius eps
        neighbors = tree.query_ball_point(points[i], eps)

        # Memeriksa apakah cukup titik untuk membentuk cluster
        if len(neighbors) < min_points:
            continue

        # Menginisialisasi cluster baru
        labels[i] = cluster_id

        # Menggunakan BFS untuk memperluas cluster
        queue = list(neighbors)
        visited = set(neighbors)

        # Memproses antrian BFS
        while queue:
            # Mengambil titik dari antrian
            current = queue.pop(0)

            # Melewati titik yang sudah dilabel cluster lain
            if labels[current] != -1 and labels[current] != cluster_id:
                continue

            # Melabel titik dengan cluster saat ini
            labels[current] = cluster_id

            # Mencari tetangga dari titik saat ini
            current_neighbors = tree.query_ball_point(points[current], eps)

            # Menambahkan tetangga baru ke antrian jika cukup padat
            if len(current_neighbors) >= min_points:
                for nb in current_neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

        # Menaikkan ID cluster
        cluster_id += 1

    # Mengembalikan label cluster
    return labels, cluster_id


# ========================================================
# BAGIAN 1: MEMBUAT POINT CLOUD DENGAN MULTIPLE OBJECTS
# ========================================================
print("=" * 60)
print("BAGIAN 1: MEMBUAT POINT CLOUD SCENE")
print("=" * 60)
print()

# Memuat point cloud dari file PLY (bunny_point_cloud.ply)
_ply_path = os.path.join(IMAGE_DIR, "bunny_point_cloud.ply")
if not os.path.exists(_ply_path):
    print("  [WARN] bunny_point_cloud.ply tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
if not os.path.exists(_ply_path):
    raise FileNotFoundError(
        "[ERROR] bunny_point_cloud.ply tidak tersedia.\n"
        "  Jalankan: python download_image.py"
    )

# Memuat titik-titik dari file PLY
_ply_pts, _ply_colors = [], []
with open(_ply_path, 'r', errors='replace') as _f:
    _in_data = False
    for _line in _f:
        _line = _line.strip()
        if _line == 'end_header':
            _in_data = True
            continue
        if _in_data and _line:
            _parts = _line.split()
            try:
                _ply_pts.append([float(_parts[0]), float(_parts[1]), float(_parts[2])])
                if len(_parts) >= 6:
                    _ply_colors.append([int(_parts[3]), int(_parts[4]), int(_parts[5])])
            except (ValueError, IndexError):
                pass
points = np.array(_ply_pts, dtype=np.float64)
labels_gt = None  # Tidak ada label ground truth untuk data PLY nyata
print(f"  PLY dimuat: {len(points)} titik")

# Mencetak informasi point cloud
print(f"\n  Total titik: {len(points)}")
print(f"  Rentang X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
print(f"  Rentang Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
print(f"  Rentang Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
print()


# ========================================================
# BAGIAN 2: RANSAC PLANE FITTING
# ========================================================
print("=" * 60)
print("BAGIAN 2: RANSAC PLANE FITTING")
print("=" * 60)
print()

# Memeriksa apakah Open3D tersedia untuk segmentasi plane
if HAS_OPEN3D:
    # Membuat objek point cloud Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Melakukan segmentasi plane dengan Open3D
    try:
        plane_model, plane_inliers = pcd.segment_plane(
            distance_threshold=0.05,
            ransac_n=3,
            num_iterations=1000
        )
        # Mengekstrak parameter plane
        a, b, c, d = plane_model
        print(f"  [Open3D] Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        # Mengkonversi indeks ke numpy array
        plane_inliers = np.array(plane_inliers)
    except Exception as e:
        # Fallback ke RANSAC manual jika Open3D gagal
        print(f"  [Open3D error: {e}] Menggunakan RANSAC manual...")
        plane_params, plane_inliers = ransac_plane_fitting(points)
        a, b, c, d = plane_params
        print(f"  Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
else:
    # Menggunakan RANSAC manual
    print("  Menggunakan RANSAC manual untuk plane fitting...")
    plane_params, plane_inliers = ransac_plane_fitting(points)
    # Mengekstrak parameter plane
    a, b, c, d = plane_params
    print(f"  Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

# Mencetak statistik plane fitting
print(f"  Jumlah inlier plane: {len(plane_inliers)}")
print(f"  Persentase plane: {len(plane_inliers)/len(points)*100:.1f}%")
print()


# ========================================================
# BAGIAN 3: MEMISAHKAN PLANE DAN OBJEK
# ========================================================
print("=" * 60)
print("BAGIAN 3: MEMISAHKAN PLANE DAN OBJEK")
print("=" * 60)
print()

# Membuat mask untuk inlier plane
plane_mask = np.zeros(len(points), dtype=bool)
plane_mask[plane_inliers] = True

# Mengekstrak titik plane
plane_points = points[plane_mask]

# Mengekstrak titik objek (outlier dari plane)
object_points = points[~plane_mask]

# Menyimpan indeks asli objek
object_indices = np.where(~plane_mask)[0]

# Mencetak informasi pemisahan
print(f"  Titik plane  : {len(plane_points)}")
print(f"  Titik objek  : {len(object_points)}")
print()


# ========================================================
# BAGIAN 4: CLUSTERING TITIK OBJEK
# ========================================================
print("=" * 60)
print("BAGIAN 4: CLUSTERING TITIK OBJEK")
print("=" * 60)
print()

# Memeriksa apakah ada titik objek untuk dikluster
if len(object_points) > 0:
    # Melakukan euclidean clustering pada titik objek
    print("  Menjalankan Euclidean clustering (eps=0.3, min_points=10)...")
    cluster_labels, num_clusters = euclidean_clustering(object_points, eps=0.3, min_points=10)

    # Mencetak jumlah cluster yang ditemukan
    print(f"  Jumlah cluster ditemukan: {num_clusters}")

    # Menghitung titik noise (tidak terkluster)
    noise_count = np.sum(cluster_labels == -1)
    print(f"  Titik noise (tidak terkluster): {noise_count}")
else:
    # Tidak ada titik objek
    cluster_labels = np.array([])
    num_clusters = 0
    print("  Tidak ada titik objek untuk dikluster")

print()


# ========================================================
# BAGIAN 5: STATISTIK CLUSTER
# ========================================================
print("=" * 60)
print("BAGIAN 5: STATISTIK CLUSTER")
print("=" * 60)
print()

# Mencetak header tabel statistik cluster
print(f"  {'Cluster':<10} {'Ukuran':>10} {'Centroid X':>12} {'Centroid Y':>12} {'Centroid Z':>12}")
print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

# Mengiterasi setiap cluster untuk mencetak statistik
for cid in range(num_clusters):
    # Mendapatkan titik dalam cluster ini
    cluster_mask = cluster_labels == cid
    cluster_pts = object_points[cluster_mask]

    # Menghitung jumlah titik
    size = len(cluster_pts)

    # Menghitung centroid cluster
    centroid = np.mean(cluster_pts, axis=0)

    # Menghitung bounding box
    bbox_min = cluster_pts.min(axis=0)
    bbox_max = cluster_pts.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # Mencetak baris statistik
    print(f"  {'#' + str(cid):<10} {size:>10} {centroid[0]:>12.4f} {centroid[1]:>12.4f} {centroid[2]:>12.4f}")

    # Mencetak bounding box
    print(f"  {'':10} BBox: [{bbox_min[0]:.2f},{bbox_min[1]:.2f},{bbox_min[2]:.2f}] → "
          f"[{bbox_max[0]:.2f},{bbox_max[1]:.2f},{bbox_max[2]:.2f}] "
          f"Size: [{bbox_size[0]:.2f},{bbox_size[1]:.2f},{bbox_size[2]:.2f}]")

print()


# ========================================================
# BAGIAN 6: VISUALISASI HASIL SEGMENTASI
# ========================================================
print("=" * 60)
print("BAGIAN 6: VISUALISASI HASIL SEGMENTASI")
print("=" * 60)
print()

# Membuat figure 2x2 untuk visualisasi
fig = plt.figure(figsize=(16, 14))

# Menambahkan judul utama
fig.suptitle("Segmentasi Point Cloud", fontsize=15, fontweight='bold')

# --- Subplot 1: Point cloud asli ---
ax1 = fig.add_subplot(221, projection='3d')
# Menampilkan point cloud asli dengan warna uniform
ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
            c='steelblue', s=1, alpha=0.3)
# Mengatur judul
ax1.set_title(f"Point Cloud Asli\n({len(points)} titik)", fontsize=11, fontweight='bold')
# Mengatur label sumbu
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# --- Subplot 2: Plane saja ---
ax2 = fig.add_subplot(222, projection='3d')
# Menampilkan titik plane dengan warna hijau
ax2.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2],
            c='limegreen', s=1, alpha=0.3)
# Mengatur judul
ax2.set_title(f"Plane (Lantai)\n({len(plane_points)} titik)", fontsize=11, fontweight='bold')
# Mengatur label sumbu
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

# --- Subplot 3: Objek saja ---
ax3 = fig.add_subplot(223, projection='3d')
# Menampilkan titik objek (tanpa plane) dengan warna merah
if len(object_points) > 0:
    ax3.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2],
                c='tomato', s=2, alpha=0.5)
# Mengatur judul
ax3.set_title(f"Objek (Tanpa Plane)\n({len(object_points)} titik)", fontsize=11, fontweight='bold')
# Mengatur label sumbu
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")

# --- Subplot 4: Cluster berwarna ---
ax4 = fig.add_subplot(224, projection='3d')

# Menampilkan plane dengan warna abu-abu di belakang
ax4.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2],
            c='lightgray', s=0.5, alpha=0.2, label='Plane')

# Menentukan daftar warna untuk cluster
cluster_colors_list = ['red', 'blue', 'green', 'orange', 'purple',
                       'cyan', 'magenta', 'yellow', 'brown', 'pink']

# Menampilkan setiap cluster dengan warna berbeda
for cid in range(num_clusters):
    # Mendapatkan mask cluster
    cmask = cluster_labels == cid
    # Memilih warna
    color = cluster_colors_list[cid % len(cluster_colors_list)]
    # Menampilkan titik cluster
    ax4.scatter(object_points[cmask, 0], object_points[cmask, 1], object_points[cmask, 2],
                c=color, s=3, alpha=0.7, label=f'Cluster #{cid}')

# Menampilkan noise dengan warna hitam
if np.any(cluster_labels == -1):
    # Mendapatkan mask noise
    noise_mask = cluster_labels == -1
    ax4.scatter(object_points[noise_mask, 0], object_points[noise_mask, 1],
                object_points[noise_mask, 2],
                c='black', s=1, alpha=0.3, label='Noise')

# Mengatur judul
ax4.set_title(f"Cluster Berwarna\n({num_clusters} cluster)", fontsize=11, fontweight='bold')
# Mengatur label sumbu
ax4.set_xlabel("X")
ax4.set_ylabel("Y")
ax4.set_zlabel("Z")
# Menambahkan legend
ax4.legend(fontsize=7, loc='upper left')

# Mengatur layout figure
plt.tight_layout()

# Menyimpan visualisasi segmentasi
output_path = os.path.join(OUTPUT_DIR, "15_segmentation_results.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Visualisasi segmentasi disimpan: {output_path}")

# Menutup figure
plt.close()
print()


# ========================================================
# BAGIAN 7: MENYIMPAN DETAIL SEGMENTASI
# ========================================================
print("=" * 60)
print("BAGIAN 7: MENYIMPAN DETAIL SEGMENTASI")
print("=" * 60)
print()

# Membuat gambar ringkasan statistik
fig_stats, ax_stats = plt.subplots(1, 1, figsize=(10, 6))

# Menyiapkan data untuk bar chart cluster
if num_clusters > 0:
    # Menghitung ukuran setiap cluster
    cluster_sizes = [np.sum(cluster_labels == cid) for cid in range(num_clusters)]
    # Menambahkan ukuran plane dan noise
    all_sizes = [len(plane_points)] + cluster_sizes
    all_labels_chart = ["Plane"] + [f"Cluster #{i}" for i in range(num_clusters)]

    # Menentukan warna untuk setiap bar
    bar_colors = ['limegreen'] + [cluster_colors_list[i % len(cluster_colors_list)]
                                  for i in range(num_clusters)]

    # Menampilkan bar chart
    bars = ax_stats.bar(all_labels_chart, all_sizes, color=bar_colors, edgecolor='black')

    # Menambahkan label jumlah pada setiap bar
    for bar, size in zip(bars, all_sizes):
        ax_stats.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                      str(size), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Mengatur judul
    ax_stats.set_title("Distribusi Ukuran Segmen", fontsize=13, fontweight='bold')
    # Mengatur label sumbu
    ax_stats.set_xlabel("Segmen", fontsize=11)
    ax_stats.set_ylabel("Jumlah Titik", fontsize=11)
    # Menambahkan grid
    ax_stats.grid(axis='y', alpha=0.3)
    # Merotasi label sumbu X
    plt.xticks(rotation=15)
else:
    # Menampilkan teks jika tidak ada cluster
    ax_stats.text(0.5, 0.5, "Tidak ada cluster ditemukan",
                  transform=ax_stats.transAxes, ha='center', fontsize=14)

# Mengatur layout
plt.tight_layout()

# Menyimpan statistik segmentasi
output_stats = os.path.join(OUTPUT_DIR, "15_segmentation_statistics.png")
plt.savefig(output_stats, dpi=150, bbox_inches='tight')
print(f"  Statistik segmentasi disimpan: {output_stats}")

# Menutup figure
plt.close()
print()


# ========================================================
# SELESAI
# ========================================================
print("=" * 60)
print("PERCOBAAN 15 SELESAI")
print("=" * 60)
print(f"Semua output disimpan di: {OUTPUT_DIR}")
