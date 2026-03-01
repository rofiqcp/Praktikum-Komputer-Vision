"""
==========================================================
PERCOBAAN 4: REGISTRASI POINT CLOUD DENGAN ICP
Mempelajari algoritma Iterative Closest Point (ICP) untuk
menyelaraskan dua point cloud. Membandingkan Point-to-Point
ICP dan Point-to-Plane ICP.

Fungsi utama:
- open3d registration_icp() (atau manual implementation)
- scipy.spatial.KDTree
- numpy linalg operations
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
    # Mengimpor library Open3D untuk pemrosesan point cloud
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
print("PERCOBAAN 4: REGISTRASI POINT CLOUD DENGAN ICP")
print("=" * 60)
print()


def load_ply_manual(filepath):
    """
    Memuat file PLY secara manual tanpa Open3D.
    """
    # Menginisialisasi list untuk menyimpan titik
    points = []
    # Menginisialisasi list untuk menyimpan warna
    colors = []
    # Menginisialisasi list untuk menyimpan normal
    normals = []
    # Menandai ketersediaan warna dan normal
    has_colors = False
    has_normals = False
    # Menentukan jumlah vertex
    num_vertices = 0

    # Membuka file PLY
    with open(filepath, 'r') as f:
        # Menandai bagian header
        in_header = True
        # Menyimpan daftar properti
        properties = []

        # Mengiterasi setiap baris file
        for line in f:
            # Menghapus whitespace di awal dan akhir
            line = line.strip()

            # Memproses header
            if in_header:
                # Membaca jumlah vertex
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                # Membaca properti
                elif line.startswith("property"):
                    properties.append(line.split()[-1])
                    # Mengecek apakah ada warna
                    if "red" in line:
                        has_colors = True
                    # Mengecek apakah ada normal
                    if "nx" in line:
                        has_normals = True
                # Menandai akhir header
                elif line == "end_header":
                    in_header = False
                # Melanjutkan ke baris berikutnya selama masih di header
                continue

            # Memproses data vertex
            if num_vertices > 0:
                # Memecah baris menjadi komponen
                vals = line.split()
                # Mengambil koordinat XYZ
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
                # Mengambil warna jika tersedia
                idx = 3
                if has_colors:
                    colors.append([int(vals[idx]), int(vals[idx+1]), int(vals[idx+2])])
                    idx += 3
                # Mengambil normal jika tersedia
                if has_normals:
                    normals.append([float(vals[idx]), float(vals[idx+1]), float(vals[idx+2])])
                # Mengurangi counter vertex
                num_vertices -= 1

    # Mengkonversi list ke numpy array
    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8) if colors else None
    normals = np.array(normals, dtype=np.float64) if normals else None

    # Mengembalikan titik, warna, dan normal
    return points, colors, normals


def best_fit_transform(source, target):
    """
    Menghitung transformasi rigid (R, t) terbaik menggunakan SVD.
    Meminimalkan ||target - (R @ source + t)||.
    """
    # Menghitung centroid dari source
    centroid_src = np.mean(source, axis=0)

    # Menghitung centroid dari target
    centroid_tgt = np.mean(target, axis=0)

    # Mengurangi centroid dari source (centering)
    src_centered = source - centroid_src

    # Mengurangi centroid dari target (centering)
    tgt_centered = target - centroid_tgt

    # Menghitung matriks kovariansi silang H
    H = src_centered.T @ tgt_centered

    # Melakukan Singular Value Decomposition pada H
    U, S, Vt = np.linalg.svd(H)

    # Menghitung matriks rotasi R
    R = Vt.T @ U.T

    # Memastikan rotasi proper (determinan = 1)
    if np.linalg.det(R) < 0:
        # Mengoreksi refleksi jika determinan negatif
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Menghitung vektor translasi t
    t = centroid_tgt - R @ centroid_src

    # Mengembalikan matriks rotasi dan vektor translasi
    return R, t


def icp_point_to_point(source, target, max_iterations=50, tolerance=1e-6):
    """
    Implementasi manual ICP Point-to-Point.
    """
    # Menyalin source agar tidak mengubah data asli
    src = source.copy()

    # Menginisialisasi list untuk menyimpan error setiap iterasi
    errors = []

    # Menginisialisasi matriks transformasi kumulatif
    R_total = np.eye(3)
    t_total = np.zeros(3)

    # Membangun KD-Tree dari target untuk pencarian efisien
    tree = KDTree(target)

    # Mencetak informasi awal ICP
    print(f"  Max iterasi: {max_iterations}, Toleransi: {tolerance}")

    # Melakukan iterasi ICP
    for i in range(max_iterations):
        # Mencari titik terdekat di target untuk setiap titik source
        distances, indices = tree.query(src)

        # Menghitung mean squared error
        mean_error = np.mean(distances ** 2)

        # Menyimpan error iterasi saat ini
        errors.append(mean_error)

        # Mengecek konvergensi
        if i > 0 and abs(errors[-2] - errors[-1]) < tolerance:
            # Mencetak informasi konvergensi
            print(f"  Konvergen pada iterasi {i+1}, error: {mean_error:.8f}")
            break

        # Mengambil titik target yang berkoresponden
        closest_points = target[indices]

        # Menghitung transformasi terbaik untuk iterasi ini
        R, t = best_fit_transform(src, closest_points)

        # Mengaplikasikan transformasi pada source
        src = (R @ src.T).T + t

        # Mengakumulasi transformasi total
        R_total = R @ R_total
        t_total = R @ t_total + t

    # Mencetak error akhir
    print(f"  Error akhir: {errors[-1]:.8f} (setelah {len(errors)} iterasi)")

    # Mengembalikan hasil transformasi
    return R_total, t_total, errors, src


def icp_point_to_plane(source, target, target_normals, max_iterations=50, tolerance=1e-6):
    """
    Implementasi manual ICP Point-to-Plane (linearized).
    """
    # Menyalin source agar tidak mengubah data asli
    src = source.copy()

    # Menginisialisasi list untuk menyimpan error
    errors = []

    # Menginisialisasi transformasi kumulatif
    R_total = np.eye(3)
    t_total = np.zeros(3)

    # Membangun KD-Tree dari target
    tree = KDTree(target)

    # Mencetak informasi awal
    print(f"  Max iterasi: {max_iterations}, Toleransi: {tolerance}")

    # Melakukan iterasi ICP Point-to-Plane
    for i in range(max_iterations):
        # Mencari titik terdekat
        distances, indices = tree.query(src)

        # Mengambil titik target yang berkoresponden
        closest_pts = target[indices]

        # Mengambil normal yang berkoresponden
        closest_norms = target_normals[indices]

        # Menghitung error point-to-plane
        diff = src - closest_pts

        # Menghitung jarak point-to-plane (dot product dengan normal)
        plane_dist = np.sum(diff * closest_norms, axis=1)

        # Menghitung mean squared error
        mean_error = np.mean(plane_dist ** 2)

        # Menyimpan error
        errors.append(mean_error)

        # Mengecek konvergensi
        if i > 0 and abs(errors[-2] - errors[-1]) < tolerance:
            # Mencetak informasi konvergensi
            print(f"  Konvergen pada iterasi {i+1}, error: {mean_error:.8f}")
            break

        # Membangun sistem linear untuk point-to-plane
        # Menggunakan linearisasi: [n x (s x n)] @ [alpha, beta, gamma, tx, ty, tz]^T = -n.(s-t)
        n_pts = len(src)
        # Menginisialisasi matriks A dan vektor b
        A = np.zeros((n_pts, 6))
        b = -plane_dist

        # Mengisi matriks A dengan cross product dan normal
        A[:, 0] = closest_norms[:, 2] * src[:, 1] - closest_norms[:, 1] * src[:, 2]
        A[:, 1] = closest_norms[:, 0] * src[:, 2] - closest_norms[:, 2] * src[:, 0]
        A[:, 2] = closest_norms[:, 1] * src[:, 0] - closest_norms[:, 0] * src[:, 1]
        A[:, 3] = closest_norms[:, 0]
        A[:, 4] = closest_norms[:, 1]
        A[:, 5] = closest_norms[:, 2]

        # Menyelesaikan sistem linear dengan least squares
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Mengekstrak sudut rotasi kecil
        alpha, beta, gamma = result[0], result[1], result[2]

        # Mengekstrak translasi
        tx, ty, tz = result[3], result[4], result[5]

        # Membangun matriks rotasi dari sudut kecil
        R = np.array([
            [1, -gamma, beta],
            [gamma, 1, -alpha],
            [-beta, alpha, 1]
        ])

        # Membuat vektor translasi
        t = np.array([tx, ty, tz])

        # Mengaplikasikan transformasi pada source
        src = (R @ src.T).T + t

        # Mengakumulasi transformasi total
        R_total = R @ R_total
        t_total = R @ t_total + t

    # Mencetak error akhir
    print(f"  Error akhir: {errors[-1]:.8f} (setelah {len(errors)} iterasi)")

    # Mengembalikan hasil transformasi
    return R_total, t_total, errors, src


# ========================================================
# 1. MEMUAT DUA POINT CLOUD (SOURCE DAN TARGET)
# ========================================================
print("=" * 60)
print("1. MEMUAT DUA POINT CLOUD")
print("=" * 60)

# Menentukan path file point cloud asli (target)
target_path = os.path.join(IMAGE_DIR, "bunny_point_cloud.ply")

# Menentukan path file point cloud yang ditransformasi (source)
source_path = os.path.join(IMAGE_DIR, "bunny_transformed.ply")

# Memuat point cloud target
target_pts, target_colors, target_normals = load_ply_manual(target_path)
print(f"  Target: {len(target_pts)} titik dimuat dari bunny_point_cloud.ply")

# Memuat point cloud source
source_pts, source_colors, _ = load_ply_manual(source_path)
print(f"  Source:  {len(source_pts)} titik dimuat dari bunny_transformed.ply")

# Menghitung jarak awal antara centroid kedua point cloud
centroid_dist = np.linalg.norm(np.mean(source_pts, axis=0) - np.mean(target_pts, axis=0))
print(f"  Jarak centroid awal: {centroid_dist:.4f}")
print()

# ========================================================
# 2. VISUALISASI SEBELUM ALIGNMENT
# ========================================================
print("=" * 60)
print("2. VISUALISASI SEBELUM ALIGNMENT")
print("=" * 60)

# Membuat figure untuk visualisasi sebelum alignment
fig = plt.figure(figsize=(14, 5))

# Membuat subplot untuk tampilan sebelum alignment
ax1 = fig.add_subplot(131, projection='3d')

# Menampilkan point cloud target (biru)
ax1.scatter(target_pts[::3, 0], target_pts[::3, 1], target_pts[::3, 2],
            c='blue', s=1, alpha=0.5, label='Target')

# Menampilkan point cloud source (merah)
ax1.scatter(source_pts[::3, 0], source_pts[::3, 1], source_pts[::3, 2],
            c='red', s=1, alpha=0.5, label='Source')

# Mengatur judul subplot
ax1.set_title('Sebelum Alignment')

# Menambahkan legenda
ax1.legend(fontsize=8)

# Mengatur label sumbu
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Mencetak informasi selesai
print("  Visualisasi sebelum alignment disiapkan")
print()

# ========================================================
# 3. ICP POINT-TO-POINT (MANUAL)
# ========================================================
print("=" * 60)
print("3. ICP POINT-TO-POINT (MANUAL)")
print("=" * 60)

# Menjalankan ICP Point-to-Point manual
R_p2p, t_p2p, errors_p2p, aligned_p2p = icp_point_to_point(
    source_pts, target_pts, max_iterations=50, tolerance=1e-8
)

# Mencetak matriks transformasi hasil ICP
print(f"\n  Matriks Rotasi:")
for row in R_p2p:
    print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
print(f"  Vektor Translasi: [{t_p2p[0]:.4f}, {t_p2p[1]:.4f}, {t_p2p[2]:.4f}]")

# Menghitung fitness score (persentase titik yang dekat)
tree_target = KDTree(target_pts)
dists_after, _ = tree_target.query(aligned_p2p)
# Menentukan threshold fitness (2x rata-rata jarak terdekat)
fitness_threshold = 0.05
# Menghitung rasio titik yang lebih dekat dari threshold
fitness_p2p = np.mean(dists_after < fitness_threshold)
print(f"  Fitness Score (threshold={fitness_threshold}): {fitness_p2p:.4f}")
print()

# Menampilkan hasil P2P pada subplot kedua
ax2 = fig.add_subplot(132, projection='3d')

# Menampilkan target
ax2.scatter(target_pts[::3, 0], target_pts[::3, 1], target_pts[::3, 2],
            c='blue', s=1, alpha=0.5, label='Target')

# Menampilkan source setelah alignment P2P
ax2.scatter(aligned_p2p[::3, 0], aligned_p2p[::3, 1], aligned_p2p[::3, 2],
            c='green', s=1, alpha=0.5, label='P2P Aligned')

# Mengatur judul
ax2.set_title('Setelah ICP Point-to-Point')

# Menambahkan legenda
ax2.legend(fontsize=8)

# Mengatur label sumbu
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# ========================================================
# 4. ICP POINT-TO-PLANE (MANUAL)
# ========================================================
print("=" * 60)
print("4. ICP POINT-TO-PLANE (MANUAL)")
print("=" * 60)

# Mengecek apakah target memiliki normal
if target_normals is not None and len(target_normals) > 0:
    # Menjalankan ICP Point-to-Plane
    R_p2pl, t_p2pl, errors_p2pl, aligned_p2pl = icp_point_to_plane(
        source_pts, target_pts, target_normals, max_iterations=50, tolerance=1e-8
    )

    # Mencetak matriks transformasi
    print(f"\n  Matriks Rotasi:")
    for row in R_p2pl:
        print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
    print(f"  Vektor Translasi: [{t_p2pl[0]:.4f}, {t_p2pl[1]:.4f}, {t_p2pl[2]:.4f}]")

    # Menghitung fitness score Point-to-Plane
    dists_p2pl, _ = tree_target.query(aligned_p2pl)
    fitness_p2pl = np.mean(dists_p2pl < fitness_threshold)
    print(f"  Fitness Score (threshold={fitness_threshold}): {fitness_p2pl:.4f}")
else:
    # Mengestimasi normal jika tidak tersedia
    print("  [INFO] Normal tidak tersedia, mengestimasi dari KDTree...")
    tree_est = KDTree(target_pts)
    normals_est = np.zeros_like(target_pts)
    # Mengestimasi normal menggunakan PCA lokal
    for idx in range(len(target_pts)):
        # Mencari k tetangga terdekat
        _, nn_idx = tree_est.query(target_pts[idx], k=15)
        # Mengambil posisi tetangga
        neighbors = target_pts[nn_idx]
        # Menghitung centroid lokal
        centroid = np.mean(neighbors, axis=0)
        # Menghitung matriks kovarian
        cov = (neighbors - centroid).T @ (neighbors - centroid)
        # Menghitung eigenvector untuk normal
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Normal adalah eigenvector dengan eigenvalue terkecil
        normals_est[idx] = eigvecs[:, 0]

    # Menjalankan ICP Point-to-Plane dengan normal estimasi
    R_p2pl, t_p2pl, errors_p2pl, aligned_p2pl = icp_point_to_plane(
        source_pts, target_pts, normals_est, max_iterations=50, tolerance=1e-8
    )
    # Menghitung fitness score
    dists_p2pl, _ = tree_target.query(aligned_p2pl)
    fitness_p2pl = np.mean(dists_p2pl < fitness_threshold)
    print(f"  Fitness Score: {fitness_p2pl:.4f}")

print()

# Menampilkan hasil P2Plane pada subplot ketiga
ax3 = fig.add_subplot(133, projection='3d')

# Menampilkan target
ax3.scatter(target_pts[::3, 0], target_pts[::3, 1], target_pts[::3, 2],
            c='blue', s=1, alpha=0.5, label='Target')

# Menampilkan source setelah alignment Point-to-Plane
ax3.scatter(aligned_p2pl[::3, 0], aligned_p2pl[::3, 1], aligned_p2pl[::3, 2],
            c='orange', s=1, alpha=0.5, label='P2Plane Aligned')

# Mengatur judul
ax3.set_title('Setelah ICP Point-to-Plane')

# Menambahkan legenda
ax3.legend(fontsize=8)

# Mengatur label sumbu
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

# Mengatur layout keseluruhan
plt.suptitle('Perbandingan ICP Registration', fontsize=14)
plt.tight_layout()

# Menyimpan visualisasi alignment
alignment_path = os.path.join(OUTPUT_DIR, "04_icp_alignment_comparison.png")
plt.savefig(alignment_path, dpi=150, bbox_inches='tight')
print(f"  Disimpan: {alignment_path}")
plt.close()

# ========================================================
# 5. PLOT KONVERGENSI (ERROR VS ITERASI)
# ========================================================
print("=" * 60)
print("5. PLOT KONVERGENSI ERROR")
print("=" * 60)

# Membuat figure untuk plot konvergensi
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot konvergensi Point-to-Point
axes[0].plot(range(1, len(errors_p2p) + 1), errors_p2p, 'g-o', markersize=3, label='P2P Error')
axes[0].set_xlabel('Iterasi')
axes[0].set_ylabel('Mean Squared Error')
axes[0].set_title('Konvergensi ICP Point-to-Point')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot konvergensi Point-to-Plane
axes[1].plot(range(1, len(errors_p2pl) + 1), errors_p2pl, 'r-o', markersize=3, label='P2Plane Error')
axes[1].set_xlabel('Iterasi')
axes[1].set_ylabel('Mean Squared Error')
axes[1].set_title('Konvergensi ICP Point-to-Plane')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Mengatur layout
plt.tight_layout()

# Menyimpan plot konvergensi
convergence_path = os.path.join(OUTPUT_DIR, "04_icp_convergence.png")
plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
print(f"  Disimpan: {convergence_path}")
plt.close()
print()

# ========================================================
# 6. PERBANDINGAN DENGAN OPEN3D ICP (JIKA TERSEDIA)
# ========================================================
print("=" * 60)
print("6. PERBANDINGAN DENGAN OPEN3D ICP")
print("=" * 60)

# Mengecek ketersediaan Open3D
if HAS_OPEN3D:
    # Membuat point cloud Open3D untuk source
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source_pts)

    # Membuat point cloud Open3D untuk target
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target_pts)

    # Mengestimasi normal untuk kedua point cloud
    pcd_source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Menentukan threshold untuk ICP
    threshold_o3d = 0.05

    # Membuat matriks transformasi awal (identitas)
    init_transform = np.eye(4)

    # Menjalankan ICP Point-to-Point Open3D
    reg_p2p_o3d = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, threshold_o3d, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Mencetak hasil ICP Open3D Point-to-Point
    print(f"  [Open3D P2P] Fitness: {reg_p2p_o3d.fitness:.4f}")
    print(f"  [Open3D P2P] RMSE: {reg_p2p_o3d.inlier_rmse:.6f}")

    # Menjalankan ICP Point-to-Plane Open3D
    reg_p2pl_o3d = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, threshold_o3d, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Mencetak hasil ICP Open3D Point-to-Plane
    print(f"  [Open3D P2Plane] Fitness: {reg_p2pl_o3d.fitness:.4f}")
    print(f"  [Open3D P2Plane] RMSE: {reg_p2pl_o3d.inlier_rmse:.6f}")

    # Mencetak matriks transformasi hasil Open3D
    print(f"\n  Matriks Transformasi Open3D P2P:")
    for row in reg_p2p_o3d.transformation:
        print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")
else:
    # Mencetak pesan jika Open3D tidak tersedia
    print("  Open3D tidak tersedia, melewatkan perbandingan Open3D")
print()

# ========================================================
# 7. TABEL PERBANDINGAN HASIL
# ========================================================
print("=" * 60)
print("7. TABEL PERBANDINGAN HASIL ICP")
print("=" * 60)

# Mencetak header tabel
print(f"\n  {'Metode':<25} {'Iterasi':<10} {'Error Akhir':<15} {'Fitness':<10}")
print(f"  {'-'*60}")

# Mencetak hasil Point-to-Point manual
print(f"  {'Manual P2P':<25} {len(errors_p2p):<10} {errors_p2p[-1]:<15.8f} {fitness_p2p:<10.4f}")

# Mencetak hasil Point-to-Plane manual
print(f"  {'Manual P2Plane':<25} {len(errors_p2pl):<10} {errors_p2pl[-1]:<15.8f} {fitness_p2pl:<10.4f}")

# Mencetak hasil Open3D jika tersedia
if HAS_OPEN3D:
    print(f"  {'Open3D P2P':<25} {'N/A':<10} {reg_p2p_o3d.inlier_rmse:<15.8f} {reg_p2p_o3d.fitness:<10.4f}")
    print(f"  {'Open3D P2Plane':<25} {'N/A':<10} {reg_p2pl_o3d.inlier_rmse:<15.8f} {reg_p2pl_o3d.fitness:<10.4f}")
print()

# ========================================================
# 8. VISUALISASI LANGKAH INTERMEDIATE
# ========================================================
print("=" * 60)
print("8. VISUALISASI LANGKAH INTERMEDIATE ICP")
print("=" * 60)

# Menjalankan ulang ICP dengan menyimpan langkah intermediate
src_intermediate = source_pts.copy()
tree_tgt = KDTree(target_pts)
# Menentukan langkah yang akan divisualisasikan
steps_to_save = [0, 1, 3, 10, 25]
# Menyimpan state intermediate
intermediate_states = {0: src_intermediate.copy()}

# Menjalankan ICP dan menyimpan langkah tertentu
for i in range(max(steps_to_save) + 1):
    # Mencari titik terdekat
    dists_i, indices_i = tree_tgt.query(src_intermediate)
    # Mengambil titik yang berkoresponden
    closest_i = target_pts[indices_i]
    # Menghitung transformasi
    R_i, t_i = best_fit_transform(src_intermediate, closest_i)
    # Mengaplikasikan transformasi
    src_intermediate = (R_i @ src_intermediate.T).T + t_i
    # Menyimpan state jika pada langkah yang ditentukan
    if (i + 1) in steps_to_save:
        intermediate_states[i + 1] = src_intermediate.copy()

# Membuat figure untuk langkah intermediate
fig3, axes3 = plt.subplots(1, len(intermediate_states), figsize=(4 * len(intermediate_states), 4))

# Mengiterasi setiap state intermediate
for ax_idx, (step, state) in enumerate(sorted(intermediate_states.items())):
    # Membuat subplot 3D
    ax = fig3.add_subplot(1, len(intermediate_states), ax_idx + 1, projection='3d')
    # Menampilkan target
    ax.scatter(target_pts[::5, 0], target_pts[::5, 1], target_pts[::5, 2],
               c='blue', s=0.5, alpha=0.3)
    # Menampilkan source pada langkah ini
    ax.scatter(state[::5, 0], state[::5, 1], state[::5, 2],
               c='red', s=0.5, alpha=0.3)
    # Mengatur judul
    ax.set_title(f'Iterasi {step}', fontsize=9)
    # Mengatur ukuran label
    ax.tick_params(labelsize=6)

# Mengatur layout
plt.suptitle('Langkah Intermediate ICP', fontsize=13)
plt.tight_layout()

# Menyimpan visualisasi intermediate
intermediate_path = os.path.join(OUTPUT_DIR, "04_icp_intermediate_steps.png")
plt.savefig(intermediate_path, dpi=150, bbox_inches='tight')
print(f"  Disimpan: {intermediate_path}")
plt.close()
print()

# Mencetak ringkasan akhir
print("=" * 60)
print("PERCOBAAN 4 SELESAI")
print("=" * 60)
print(f"  Output tersimpan di: {OUTPUT_DIR}")
print(f"  File yang dihasilkan:")
print(f"    - 04_icp_alignment_comparison.png")
print(f"    - 04_icp_convergence.png")
print(f"    - 04_icp_intermediate_steps.png")
