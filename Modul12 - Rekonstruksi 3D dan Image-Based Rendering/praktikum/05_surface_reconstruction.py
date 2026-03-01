"""
==========================================================
PERCOBAAN 5: SURFACE RECONSTRUCTION (POISSON DAN BPA)
Mempelajari rekonstruksi permukaan (mesh) dari point cloud
menggunakan Poisson Reconstruction dan Ball Pivoting
Algorithm.

Fungsi utama:
- open3d create_from_point_cloud_poisson()
- open3d create_from_point_cloud_ball_pivoting()
- Delaunay triangulation (scipy) sebagai fallback
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

# Mengimpor art3d untuk menampilkan polygon 3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Mengimpor Delaunay dari scipy sebagai fallback
from scipy.spatial import Delaunay, KDTree

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
    print("[INFO] Open3D tidak tersedia, menggunakan fallback SciPy Delaunay")

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
print("PERCOBAAN 5: SURFACE RECONSTRUCTION (POISSON DAN BPA)")
print("=" * 60)
print()


def load_ply_manual(filepath):
    """
    Memuat file PLY secara manual tanpa Open3D.
    """
    # Menginisialisasi list untuk titik, warna, dan normal
    points = []
    colors = []
    normals = []
    has_colors = False
    has_normals = False
    num_vertices = 0

    # Membuka file PLY
    with open(filepath, 'r') as f:
        in_header = True
        properties = []

        # Mengiterasi setiap baris
        for line in f:
            line = line.strip()

            # Memproses header
            if in_header:
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                elif line.startswith("property"):
                    properties.append(line.split()[-1])
                    if "red" in line:
                        has_colors = True
                    if "nx" in line:
                        has_normals = True
                elif line == "end_header":
                    in_header = False
                continue

            # Memproses data vertex
            if num_vertices > 0:
                vals = line.split()
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
                idx = 3
                if has_colors:
                    colors.append([int(vals[idx]), int(vals[idx+1]), int(vals[idx+2])])
                    idx += 3
                if has_normals:
                    normals.append([float(vals[idx]), float(vals[idx+1]), float(vals[idx+2])])
                num_vertices -= 1

    # Mengkonversi ke numpy array
    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8) if colors else None
    normals = np.array(normals, dtype=np.float64) if normals else None

    # Mengembalikan data
    return points, colors, normals


def estimate_normals_manual(points, k=15):
    """
    Mengestimasi normal menggunakan PCA lokal.
    """
    # Membangun KD-Tree untuk pencarian tetangga
    tree = KDTree(points)

    # Menginisialisasi array untuk menyimpan normal
    normals = np.zeros_like(points)

    # Menghitung centroid global untuk orientasi normal
    centroid = np.mean(points, axis=0)

    # Mengiterasi setiap titik
    for i in range(len(points)):
        # Mencari k tetangga terdekat
        _, nn_idx = tree.query(points[i], k=k)

        # Mengambil posisi tetangga
        neighbors = points[nn_idx]

        # Menghitung centroid lokal
        local_center = np.mean(neighbors, axis=0)

        # Menghitung matriks kovarian
        cov = (neighbors - local_center).T @ (neighbors - local_center) / k

        # Menghitung eigenvector
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Normal adalah eigenvector dengan eigenvalue terkecil
        normal = eigvecs[:, 0]

        # Mengorientasikan normal agar menjauh dari centroid
        if np.dot(normal, points[i] - centroid) < 0:
            normal = -normal

        # Menyimpan normal
        normals[i] = normal

    # Mengembalikan array normal
    return normals


def delaunay_surface_reconstruction(points, normals=None):
    """
    Rekonstruksi permukaan menggunakan Delaunay triangulation (fallback).
    Memproyeksikan ke 2D untuk triangulasi kemudian mapping ke 3D.
    """
    # Mencetak info metode
    print("  Menggunakan Delaunay Triangulation sebagai fallback...")

    # Menghitung PCA untuk memproyeksikan ke bidang 2D terbaik
    centroid = np.mean(points, axis=0)

    # Menghitung matriks kovarian
    cov = (points - centroid).T @ (points - centroid)

    # Menghitung eigenvector untuk menemukan bidang utama
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Memproyeksikan titik ke dua sumbu utama (eigenvalue terbesar)
    proj_2d = (points - centroid) @ eigvecs[:, 1:]

    # Melakukan Delaunay triangulation pada proyeksi 2D
    tri = Delaunay(proj_2d)

    # Mendapatkan simpleks (segitiga)
    triangles = tri.simplices

    # Mencetak jumlah segitiga yang dihasilkan
    print(f"  Dihasilkan {len(triangles)} segitiga dari {len(points)} titik")

    # Mengembalikan array segitiga
    return triangles


def filter_long_edges(points, triangles, max_edge_factor=3.0):
    """
    Memfilter segitiga dengan sisi yang terlalu panjang.
    """
    # Menghitung rata-rata jarak antar titik
    tree = KDTree(points)
    dists, _ = tree.query(points, k=2)
    avg_dist = np.mean(dists[:, 1])

    # Menentukan threshold panjang sisi maksimum
    max_edge = avg_dist * max_edge_factor

    # Memfilter segitiga
    filtered = []
    for tri in triangles:
        # Menghitung panjang ketiga sisi segitiga
        e1 = np.linalg.norm(points[tri[0]] - points[tri[1]])
        e2 = np.linalg.norm(points[tri[1]] - points[tri[2]])
        e3 = np.linalg.norm(points[tri[2]] - points[tri[0]])
        # Memeriksa apakah semua sisi di bawah threshold
        if e1 < max_edge and e2 < max_edge and e3 < max_edge:
            filtered.append(tri)

    # Mengkonversi ke numpy array
    filtered = np.array(filtered)
    print(f"  Difilter: {len(triangles)} -> {len(filtered)} segitiga (max edge={max_edge:.4f})")

    # Mengembalikan segitiga yang sudah difilter
    return filtered


def visualize_mesh(points, triangles, colors=None, title="Mesh", ax=None):
    """
    Memvisualisasikan mesh 3D menggunakan matplotlib.
    """
    # Membuat axes jika belum tersedia
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    # Membuat koleksi polygon dari segitiga
    verts = points[triangles]

    # Menentukan warna face
    if colors is not None:
        # Menghitung warna rata-rata per segitiga
        face_colors = np.mean(colors[triangles] / 255.0, axis=1)
        face_colors = np.clip(face_colors, 0, 1)
        # Menambahkan alpha channel
        face_colors_rgba = np.column_stack([face_colors, np.full(len(face_colors), 0.7)])
    else:
        # Menggunakan warna default
        face_colors_rgba = (0.6, 0.8, 1.0, 0.7)

    # Membuat koleksi Poly3D
    mesh_collection = Poly3DCollection(verts, alpha=0.7)

    # Mengatur warna face
    mesh_collection.set_facecolor(face_colors_rgba)

    # Mengatur warna edge
    mesh_collection.set_edgecolor('gray')

    # Mengatur ketebalan edge
    mesh_collection.set_linewidth(0.1)

    # Menambahkan koleksi ke axes
    ax.add_collection3d(mesh_collection)

    # Mengatur batas axes berdasarkan data
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_zlim(points[:, 2].min(), points[:, 2].max())

    # Mengatur judul
    ax.set_title(title, fontsize=10)

    # Mengatur label sumbu
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)

    # Mengembalikan axes
    return ax


# ========================================================
# 1. MEMUAT POINT CLOUD DENGAN NORMAL
# ========================================================
print("=" * 60)
print("1. MEMUAT POINT CLOUD DENGAN NORMAL")
print("=" * 60)

# Menentukan path file point cloud
ply_path = os.path.join(IMAGE_DIR, "bunny_point_cloud.ply")

# Memuat point cloud secara manual
points, colors, normals = load_ply_manual(ply_path)
print(f"  Dimuat: {len(points)} titik dari bunny_point_cloud.ply")
print(f"  Warna: {'Ya' if colors is not None else 'Tidak'}")
print(f"  Normal: {'Ya' if normals is not None else 'Tidak'}")

# Mengestimasi normal jika tidak tersedia
if normals is None:
    print("  Mengestimasi normal secara manual...")
    normals = estimate_normals_manual(points, k=15)
    print(f"  Normal diestimasi untuk {len(normals)} titik")
print()

# ========================================================
# 2. POISSON RECONSTRUCTION
# ========================================================
print("=" * 60)
print("2. POISSON RECONSTRUCTION")
print("=" * 60)

# Variabel untuk menyimpan hasil Poisson
poisson_triangles = None
poisson_vertices = None
poisson_densities = None

# Mengecek ketersediaan Open3D untuk Poisson
if HAS_OPEN3D:
    # Membuat point cloud Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Menambahkan warna jika tersedia
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Menjalankan Poisson reconstruction
    print("  Menjalankan Poisson Surface Reconstruction (Open3D)...")
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, width=0, scale=1.1, linear_fit=False
    )

    # Mengekstrak data mesh Poisson
    poisson_vertices = np.asarray(mesh_poisson.vertices)
    poisson_triangles = np.asarray(mesh_poisson.triangles)
    poisson_densities = np.asarray(densities)

    # Mencetak statistik mesh Poisson
    print(f"  Vertices: {len(poisson_vertices)}")
    print(f"  Triangles: {len(poisson_triangles)}")
    print(f"  Density range: [{poisson_densities.min():.4f}, {poisson_densities.max():.4f}]")
else:
    # Menggunakan Delaunay sebagai fallback untuk Poisson
    print("  Open3D tidak tersedia, menggunakan Delaunay sebagai pendekatan Poisson...")
    poisson_triangles = delaunay_surface_reconstruction(points, normals)
    poisson_vertices = points.copy()

    # Memfilter segitiga dengan sisi panjang
    poisson_triangles = filter_long_edges(poisson_vertices, poisson_triangles, max_edge_factor=2.5)
    print(f"  Vertices: {len(poisson_vertices)}")
    print(f"  Triangles: {len(poisson_triangles)}")
print()

# ========================================================
# 3. BALL PIVOTING ALGORITHM (BPA)
# ========================================================
print("=" * 60)
print("3. BALL PIVOTING ALGORITHM (BPA)")
print("=" * 60)

# Variabel untuk menyimpan hasil BPA
bpa_triangles = None
bpa_vertices = None

# Mengecek ketersediaan Open3D untuk BPA
if HAS_OPEN3D:
    # Menghitung jarak rata-rata antar titik untuk radius
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    # Menentukan radius ball untuk BPA
    radii = [avg_dist * 1.0, avg_dist * 2.0, avg_dist * 4.0]
    radii_vector = o3d.utility.DoubleVector(radii)

    # Menjalankan Ball Pivoting Algorithm
    print(f"  Menjalankan BPA (radii: {[f'{r:.4f}' for r in radii]})...")
    mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii_vector
    )

    # Mengekstrak data mesh BPA
    bpa_vertices = np.asarray(mesh_bpa.vertices)
    bpa_triangles = np.asarray(mesh_bpa.triangles)

    # Mencetak statistik mesh BPA
    print(f"  Vertices: {len(bpa_vertices)}")
    print(f"  Triangles: {len(bpa_triangles)}")
else:
    # Menggunakan Delaunay dengan parameter berbeda sebagai fallback BPA
    print("  Open3D tidak tersedia, menggunakan Delaunay sebagai pendekatan BPA...")
    bpa_triangles = delaunay_surface_reconstruction(points, normals)
    bpa_vertices = points.copy()

    # Memfilter segitiga dengan threshold berbeda
    bpa_triangles = filter_long_edges(bpa_vertices, bpa_triangles, max_edge_factor=3.5)
    print(f"  Vertices: {len(bpa_vertices)}")
    print(f"  Triangles: {len(bpa_triangles)}")
print()

# ========================================================
# 4. PERBANDINGAN HASIL
# ========================================================
print("=" * 60)
print("4. PERBANDINGAN HASIL REKONSTRUKSI")
print("=" * 60)

# Mencetak header tabel perbandingan
print(f"\n  {'Metode':<25} {'Vertices':<12} {'Triangles':<12}")
print(f"  {'-'*50}")

# Mencetak statistik Poisson
print(f"  {'Poisson Recon':<25} {len(poisson_vertices):<12} {len(poisson_triangles):<12}")

# Mencetak statistik BPA
print(f"  {'Ball Pivoting (BPA)':<25} {len(bpa_vertices):<12} {len(bpa_triangles):<12}")

# Menghitung rasio segitiga
ratio = len(poisson_triangles) / max(len(bpa_triangles), 1)
print(f"\n  Rasio Poisson/BPA triangles: {ratio:.2f}x")
print()

# ========================================================
# 5. POISSON CLEANUP (MENGHAPUS LOW-DENSITY VERTICES)
# ========================================================
print("=" * 60)
print("5. POISSON CLEANUP (MENGHAPUS LOW-DENSITY VERTICES)")
print("=" * 60)

# Variabel untuk menyimpan mesh yang dibersihkan
cleaned_vertices = poisson_vertices.copy()
cleaned_triangles = poisson_triangles.copy()

# Mengecek apakah densitas tersedia (dari Open3D Poisson)
if poisson_densities is not None:
    # Menghitung threshold berdasarkan quantile
    density_threshold = np.quantile(poisson_densities, 0.1)
    print(f"  Density threshold (quantile 10%): {density_threshold:.4f}")

    # Membuat mask untuk vertex dengan densitas tinggi
    density_mask = poisson_densities > density_threshold

    # Menghitung jumlah vertex yang dihapus
    removed_count = np.sum(~density_mask)
    print(f"  Vertices dihapus: {removed_count} dari {len(poisson_vertices)}")

    # Membuat mapping index lama ke baru
    new_indices = np.full(len(poisson_vertices), -1, dtype=int)
    new_indices[density_mask] = np.arange(np.sum(density_mask))

    # Memfilter vertices
    cleaned_vertices = poisson_vertices[density_mask]

    # Memfilter triangles: hanya yang semua vertex-nya valid
    valid_triangles = []
    for tri in poisson_triangles:
        # Mengecek apakah semua vertex segitiga ada di mask
        if density_mask[tri[0]] and density_mask[tri[1]] and density_mask[tri[2]]:
            # Memetakan ke index baru
            new_tri = [new_indices[tri[0]], new_indices[tri[1]], new_indices[tri[2]]]
            valid_triangles.append(new_tri)

    # Mengkonversi ke numpy array
    cleaned_triangles = np.array(valid_triangles) if valid_triangles else np.array([]).reshape(0, 3)
    print(f"  Hasil cleanup: {len(cleaned_vertices)} vertices, {len(cleaned_triangles)} triangles")
else:
    # Menggunakan filtering jarak sebagai alternatif cleanup
    print("  Densitas tidak tersedia, menggunakan distance-based filtering...")

    # Menghitung jarak ke centroid
    centroid = np.mean(cleaned_vertices, axis=0)
    dists_to_center = np.linalg.norm(cleaned_vertices - centroid, axis=1)

    # Menentukan threshold jarak
    dist_threshold = np.percentile(dists_to_center, 95)

    # Membuat mask vertex dalam jangkauan
    dist_mask = dists_to_center < dist_threshold

    # Menghitung vertex yang dihapus
    removed_count = np.sum(~dist_mask)
    print(f"  Vertices dihapus (outlier): {removed_count}")
print()

# ========================================================
# 6. VISUALISASI KEDUA MESH
# ========================================================
print("=" * 60)
print("6. VISUALISASI MESH HASIL REKONSTRUKSI")
print("=" * 60)

# Membuat figure utama untuk perbandingan
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

# Mensubsampling untuk visualisasi yang lebih cepat
max_display_tris = 5000

# Visualisasi Poisson reconstruction
poisson_display = poisson_triangles[:max_display_tris] if len(poisson_triangles) > max_display_tris else poisson_triangles
visualize_mesh(poisson_vertices, poisson_display, colors=None,
               title=f'Poisson ({len(poisson_triangles)} tris)', ax=axes[0])

# Visualisasi BPA reconstruction
bpa_display = bpa_triangles[:max_display_tris] if len(bpa_triangles) > max_display_tris else bpa_triangles
visualize_mesh(bpa_vertices, bpa_display, colors=None,
               title=f'BPA ({len(bpa_triangles)} tris)', ax=axes[1])

# Visualisasi Poisson cleanup
if len(cleaned_triangles) > 0:
    cleaned_display = cleaned_triangles[:max_display_tris] if len(cleaned_triangles) > max_display_tris else cleaned_triangles
    visualize_mesh(cleaned_vertices, cleaned_display.astype(int), colors=None,
                   title=f'Poisson Cleaned ({len(cleaned_triangles)} tris)', ax=axes[2])
else:
    # Menampilkan point cloud jika tidak ada segitiga
    axes[2].scatter(cleaned_vertices[::3, 0], cleaned_vertices[::3, 1],
                    cleaned_vertices[::3, 2], c='blue', s=1, alpha=0.5)
    axes[2].set_title('Poisson Cleaned (no triangles)')

# Mengatur judul keseluruhan
plt.suptitle('Perbandingan Surface Reconstruction', fontsize=14)

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi perbandingan
comparison_path = os.path.join(OUTPUT_DIR, "05_surface_reconstruction_comparison.png")
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f"  Disimpan: {comparison_path}")
plt.close()

# ========================================================
# 7. VISUALISASI DETAIL SETIAP METODE
# ========================================================
print()
print("=" * 60)
print("7. VISUALISASI DETAIL SETIAP METODE")
print("=" * 60)

# Membuat figure untuk detail Poisson
fig_detail = plt.figure(figsize=(14, 5))

# Subplot 1: Point cloud original
ax_pc = fig_detail.add_subplot(131, projection='3d')
if colors is not None:
    ax_pc.scatter(points[::3, 0], points[::3, 1], points[::3, 2],
                  c=colors[::3] / 255.0, s=1, alpha=0.6)
else:
    ax_pc.scatter(points[::3, 0], points[::3, 1], points[::3, 2],
                  c='steelblue', s=1, alpha=0.6)
ax_pc.set_title('Point Cloud Original', fontsize=10)

# Subplot 2: Wireframe mesh Poisson (sample)
ax_wire = fig_detail.add_subplot(132, projection='3d')
# Menampilkan wireframe dari beberapa segitiga
sample_tris = poisson_triangles[:2000] if len(poisson_triangles) > 2000 else poisson_triangles
for tri in sample_tris[::5]:
    # Mengambil koordinat tiga titik segitiga
    pts_tri = poisson_vertices[tri]
    # Menghubungkan titik-titik segitiga (closed polygon)
    pts_closed = np.vstack([pts_tri, pts_tri[0]])
    ax_wire.plot(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2],
                 'b-', linewidth=0.2, alpha=0.3)
ax_wire.set_title('Wireframe (Poisson sample)', fontsize=10)

# Subplot 3: Histogram distribusi ukuran segitiga
ax_hist = fig_detail.add_subplot(133)

# Menghitung area setiap segitiga Poisson
areas_poisson = []
for tri in poisson_triangles[:5000]:
    # Mengambil tiga titik
    v0, v1, v2 = poisson_vertices[tri[0]], poisson_vertices[tri[1]], poisson_vertices[tri[2]]
    # Menghitung area menggunakan cross product
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    areas_poisson.append(area)

# Menampilkan histogram area segitiga
ax_hist.hist(areas_poisson, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax_hist.set_xlabel('Area Segitiga')
ax_hist.set_ylabel('Frekuensi')
ax_hist.set_title('Distribusi Area Segitiga (Poisson)')
ax_hist.grid(True, alpha=0.3)

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi detail
detail_path = os.path.join(OUTPUT_DIR, "05_surface_reconstruction_detail.png")
plt.savefig(detail_path, dpi=150, bbox_inches='tight')
print(f"  Disimpan: {detail_path}")
plt.close()
print()

# ========================================================
# 8. STATISTIK MESH
# ========================================================
print("=" * 60)
print("8. STATISTIK MESH FINAL")
print("=" * 60)

# Menghitung area total mesh Poisson
total_area_poisson = sum(areas_poisson)
print(f"\n  [Poisson Reconstruction]")
print(f"    Jumlah vertices:  {len(poisson_vertices)}")
print(f"    Jumlah triangles: {len(poisson_triangles)}")
print(f"    Total area:       {total_area_poisson:.4f}")
print(f"    Rata-rata area:   {np.mean(areas_poisson):.6f}")

# Menghitung area total mesh BPA
areas_bpa = []
for tri in bpa_triangles[:5000]:
    v0, v1, v2 = bpa_vertices[tri[0]], bpa_vertices[tri[1]], bpa_vertices[tri[2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    areas_bpa.append(area)

# Mencetak statistik BPA
total_area_bpa = sum(areas_bpa)
print(f"\n  [Ball Pivoting Algorithm]")
print(f"    Jumlah vertices:  {len(bpa_vertices)}")
print(f"    Jumlah triangles: {len(bpa_triangles)}")
print(f"    Total area:       {total_area_bpa:.4f}")
print(f"    Rata-rata area:   {np.mean(areas_bpa):.6f}")

# Mencetak statistik cleanup
print(f"\n  [Poisson Cleaned]")
print(f"    Jumlah vertices:  {len(cleaned_vertices)}")
print(f"    Jumlah triangles: {len(cleaned_triangles)}")
print()

# Mencetak ringkasan akhir
print("=" * 60)
print("PERCOBAAN 5 SELESAI")
print("=" * 60)
print(f"  Output tersimpan di: {OUTPUT_DIR}")
print(f"  File yang dihasilkan:")
print(f"    - 05_surface_reconstruction_comparison.png")
print(f"    - 05_surface_reconstruction_detail.png")
