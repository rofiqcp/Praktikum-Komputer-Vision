"""
==========================================================
PERCOBAAN 11: BALL PIVOTING ALGORITHM (BPA)
Mempelajari rekonstruksi surface menggunakan Ball Pivoting
Algorithm dengan berbagai radius bola dan menganalisis
pengaruhnya.

Fungsi utama:
- open3d create_from_point_cloud_ball_pivoting() (atau manual Delaunay)
- scipy.spatial.Delaunay()
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

# Mengimpor art3d untuk menampilkan polygon 3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Mengimpor Delaunay dari scipy sebagai fallback
from scipy.spatial import Delaunay

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
print("PERCOBAAN 11: BALL PIVOTING ALGORITHM (BPA)")
print("=" * 60)
print()


# ========================================================
# FUNGSI UTILITAS
# ========================================================

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

    # Membuka file PLY untuk dibaca
    with open(filepath, 'r') as f:
        # Menandai bahwa kita masih di header
        in_header = True

        # Membaca setiap baris file
        for line in f:
            # Menghapus whitespace di awal/akhir baris
            line = line.strip()

            # Memproses header PLY
            if in_header:
                # Memeriksa jumlah vertex
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                # Memeriksa apakah ada properti warna merah
                elif line.startswith("property") and "red" in line:
                    has_colors = True
                # Memeriksa apakah ada properti normal nx
                elif line.startswith("property") and "nx" in line:
                    has_normals = True
                # Mendeteksi akhir header
                elif line == "end_header":
                    in_header = False
                # Melanjutkan ke baris selanjutnya jika masih header
                continue

            # Memproses data vertex
            parts = line.split()

            # Memeriksa apakah data valid
            if len(parts) < 3:
                continue

            # Membaca koordinat XYZ
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])

            # Menentukan index mulai untuk data tambahan
            idx = 3

            # Membaca normal jika tersedia
            if has_normals and len(parts) >= idx + 3:
                nx, ny, nz = float(parts[idx]), float(parts[idx+1]), float(parts[idx+2])
                normals.append([nx, ny, nz])
                idx += 3

            # Membaca warna jika tersedia
            if has_colors and len(parts) >= idx + 3:
                r, g, b = int(parts[idx]), int(parts[idx+1]), int(parts[idx+2])
                colors.append([r, g, b])

    # Mengkonversi list ke numpy array
    points = np.array(points, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64) if normals else None
    colors = np.array(colors, dtype=np.uint8) if colors else None

    # Mencetak informasi file yang dimuat
    print(f"  Dimuat: {filepath}")
    print(f"  Jumlah titik: {len(points)}")
    print(f"  Memiliki normal: {normals is not None}")
    print(f"  Memiliki warna: {colors is not None}")

    # Mengembalikan data point cloud
    return points, normals, colors


def estimate_normals_simple(points, k=10):
    """
    Mengestimasi normal secara sederhana menggunakan PCA pada tetangga terdekat.
    """
    # Menghitung jumlah titik
    n = len(points)

    # Menginisialisasi array normal
    normals = np.zeros_like(points)

    # Mengiterasi setiap titik
    for i in range(n):
        # Menghitung jarak ke semua titik lain
        dists = np.linalg.norm(points - points[i], axis=1)

        # Mendapatkan index k tetangga terdekat
        nn_idx = np.argsort(dists)[1:k+1]

        # Mendapatkan titik-titik tetangga
        neighbors = points[nn_idx]

        # Menghitung centroid tetangga
        centroid = np.mean(neighbors, axis=0)

        # Menghitung matriks kovarians
        cov = (neighbors - centroid).T @ (neighbors - centroid)

        # Menghitung eigenvalues dan eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Normal adalah eigenvector dengan eigenvalue terkecil
        normal = eigenvectors[:, 0]

        # Memastikan normal mengarah keluar (dari centroid)
        center = np.mean(points, axis=0)
        if np.dot(normal, points[i] - center) < 0:
            normal = -normal

        # Menyimpan normal
        normals[i] = normal

    # Mengembalikan array normal
    return normals


# ========================================================
# 1. MEMUAT POINT CLOUD DENGAN NORMAL
# ========================================================
print("=" * 60)
print("1. MEMUAT POINT CLOUD DENGAN NORMAL")
print("=" * 60)

# Menentukan path file point cloud
ply_path = os.path.join(IMAGE_DIR, "bunny_point_cloud.ply")

# Memeriksa apakah file point cloud tersedia dan download jika perlu
if not os.path.exists(ply_path):
    print("  [WARN] bunny_point_cloud.ply tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
if not os.path.exists(ply_path):
    raise FileNotFoundError(
        "[ERROR] bunny_point_cloud.ply tidak tersedia.\n"
        "  Jalankan: python download_image.py"
    )

# Memuat point cloud dari file PLY
points, normals, colors = load_ply_manual(ply_path)

# Mengestimasi normal jika belum tersedia
if normals is None or len(normals) == 0:
    print("  Mengestimasi normal...")
    # Mengestimasi normal menggunakan metode sederhana
    normals = estimate_normals_simple(points, k=10)
    print(f"  Normal diestimasi: {normals.shape}")

print()


# ========================================================
# 2. BPA DENGAN OPEN3D (RADIUS KECIL, SEDANG, BESAR)
# ========================================================
print("=" * 60)
print("2. BPA DENGAN BERBAGAI RADIUS BOLA")
print("=" * 60)

# Menghitung rata-rata jarak antar titik untuk menentukan radius
dists_sample = []
# Mengambil sampel untuk estimasi jarak
sample_idx = np.random.choice(len(points), min(200, len(points)), replace=False)
for idx in sample_idx:
    # Menghitung jarak terdekat dari titik ini
    d = np.linalg.norm(points - points[idx], axis=1)
    d_sorted = np.sort(d)
    # Mengambil jarak ke tetangga terdekat (bukan diri sendiri)
    if len(d_sorted) > 1:
        dists_sample.append(d_sorted[1])

# Menghitung estimasi jarak rata-rata
avg_dist = np.mean(dists_sample)
print(f"  Estimasi jarak rata-rata antar titik: {avg_dist:.4f}")

# Mendefinisikan tiga radius bola untuk BPA
radii_config = {
    "kecil": avg_dist * 1.0,
    "sedang": avg_dist * 2.0,
    "besar": avg_dist * 4.0,
}

# Menyiapkan dictionary untuk menyimpan hasil BPA
bpa_results = {}

# Memeriksa ketersediaan Open3D
if HAS_OPEN3D:
    # Membuat point cloud Open3D
    pcd = o3d.geometry.PointCloud()

    # Mengatur titik-titik point cloud
    pcd.points = o3d.utility.Vector3dVector(points)

    # Mengatur normal point cloud
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Mengiterasi setiap konfigurasi radius
    for nama, radius in radii_config.items():
        # Mencetak informasi radius
        print(f"\n  BPA radius {nama} = {radius:.4f}")

        # Membuat array radius untuk multi-scale BPA
        radii_arr = o3d.utility.DoubleVector([radius, radius * 1.5])

        # Menjalankan Ball Pivoting Algorithm
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, radii_arr
        )

        # Mendapatkan jumlah triangle
        n_triangles = len(mesh.triangles)

        # Mendapatkan jumlah vertices pada mesh
        n_verts = len(mesh.vertices)

        # Menyimpan hasil BPA
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        bpa_results[nama] = {
            "vertices": vertices,
            "triangles": triangles,
            "n_triangles": n_triangles,
            "n_vertices": n_verts,
            "radius": radius,
        }

        # Mencetak statistik
        print(f"    Vertices: {n_verts}, Triangles: {n_triangles}")

else:
    # Fallback menggunakan Delaunay triangulation
    print("  [FALLBACK] Menggunakan SciPy Delaunay triangulation")
    print()

    # Mengiterasi setiap konfigurasi radius (simulasi efek radius)
    for nama, radius in radii_config.items():
        # Mencetak informasi radius simulasi
        print(f"  Delaunay dengan filter jarak '{nama}' (threshold={radius:.4f})")

        # Memproyeksikan titik 3D ke 2D untuk Delaunay (menggunakan XY)
        points_2d = points[:, :2]

        # Melakukan Delaunay triangulation
        tri = Delaunay(points_2d)

        # Mendapatkan simplices (triangles)
        simplices = tri.simplices

        # Memfilter triangle berdasarkan panjang sisi (simulasi radius)
        filtered_triangles = []
        for simplex in simplices:
            # Mendapatkan titik-titik triangle
            p0 = points[simplex[0]]
            p1 = points[simplex[1]]
            p2 = points[simplex[2]]

            # Menghitung panjang setiap sisi
            d01 = np.linalg.norm(p1 - p0)
            d12 = np.linalg.norm(p2 - p1)
            d20 = np.linalg.norm(p0 - p2)

            # Memfilter berdasarkan threshold (sisi maks < 2*radius)
            if max(d01, d12, d20) < 2 * radius:
                filtered_triangles.append(simplex)

        # Mengkonversi ke array numpy
        filtered_triangles = np.array(filtered_triangles) if filtered_triangles else np.array([]).reshape(0, 3)

        # Menyimpan hasil
        bpa_results[nama] = {
            "vertices": points,
            "triangles": filtered_triangles,
            "n_triangles": len(filtered_triangles),
            "n_vertices": len(points),
            "radius": radius,
        }

        # Mencetak statistik
        print(f"    Vertices: {len(points)}, Triangles: {len(filtered_triangles)}")

print()


# ========================================================
# 3. MANUAL FALLBACK: DELAUNAY TRIANGULATION
# ========================================================
print("=" * 60)
print("3. DELAUNAY TRIANGULATION (REFERENSI)")
print("=" * 60)

# Memproyeksikan titik 3D ke bidang XY untuk Delaunay
points_2d_ref = points[:, :2]

# Melakukan Delaunay triangulation tanpa filter
tri_ref = Delaunay(points_2d_ref)

# Mendapatkan jumlah triangle Delaunay
n_tri_ref = len(tri_ref.simplices)

# Mencetak informasi Delaunay referensi
print(f"  Delaunay (tanpa filter): {n_tri_ref} triangles")
print(f"  Ini sebagai baseline perbandingan dengan BPA")
print()


# ========================================================
# 4. PERBANDINGAN EFEK BERBAGAI RADIUS BOLA
# ========================================================
print("=" * 60)
print("4. PERBANDINGAN EFEK BERBAGAI RADIUS BOLA")
print("=" * 60)

# Mencetak tabel perbandingan
print(f"  {'Radius':<12} {'Nilai':<10} {'Triangles':<12} {'Vertices':<10}")
print(f"  {'-'*44}")

# Mengiterasi setiap hasil
for nama, result in bpa_results.items():
    # Mencetak baris tabel
    print(f"  {nama:<12} {result['radius']:<10.4f} {result['n_triangles']:<12} {result['n_vertices']:<10}")

print()


# ========================================================
# 5. ANALISIS: JUMLAH TRIANGLE, COVERAGE, HOLES
# ========================================================
print("=" * 60)
print("5. ANALISIS: JUMLAH TRIANGLE, COVERAGE, HOLES")
print("=" * 60)

# Mengiterasi setiap hasil untuk analisis
for nama, result in bpa_results.items():
    # Mencetak header analisis untuk radius ini
    print(f"\n  [{nama.upper()}] Radius = {result['radius']:.4f}")

    # Menghitung jumlah triangle
    n_tri = result["n_triangles"]
    print(f"    Jumlah triangle: {n_tri}")

    # Menghitung vertex yang terpakai dalam triangle
    if n_tri > 0 and len(result["triangles"]) > 0:
        # Mendapatkan unique vertices yang digunakan
        used_verts = np.unique(result["triangles"].flatten())
        coverage = len(used_verts) / result["n_vertices"] * 100
    else:
        used_verts = np.array([])
        coverage = 0.0

    # Mencetak coverage
    print(f"    Coverage vertices: {len(used_verts)}/{result['n_vertices']} ({coverage:.1f}%)")

    # Menghitung estimasi holes (vertices yang tidak tercover)
    n_holes_est = result["n_vertices"] - len(used_verts)
    print(f"    Estimasi vertices tanpa triangle: {n_holes_est}")

    # Menganalisis ukuran triangle
    if n_tri > 0 and len(result["triangles"]) > 0:
        # Menghitung luas setiap triangle
        areas = []
        verts = result["vertices"]
        for tri_idx in result["triangles"][:min(1000, n_tri)]:
            # Mendapatkan titik-titik triangle
            p0 = verts[tri_idx[0]]
            p1 = verts[tri_idx[1]]
            p2 = verts[tri_idx[2]]
            # Menghitung luas menggunakan cross product
            area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            areas.append(area)
        # Mencetak statistik luas
        areas = np.array(areas)
        print(f"    Rata-rata luas triangle: {np.mean(areas):.6f}")
        print(f"    Min/Max luas: {np.min(areas):.6f} / {np.max(areas):.6f}")

print()


# ========================================================
# 6. VISUALISASI SETIAP HASIL
# ========================================================
print("=" * 60)
print("6. VISUALISASI SETIAP HASIL")
print("=" * 60)

# Mengiterasi setiap hasil untuk visualisasi individual
for nama, result in bpa_results.items():
    # Membuat figure 3D untuk hasil ini
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Mendapatkan data
    verts = result["vertices"]
    tris = result["triangles"]

    # Menampilkan titik-titik point cloud sebagai scatter
    subsample = np.random.choice(len(verts), min(500, len(verts)), replace=False)
    ax.scatter(verts[subsample, 0], verts[subsample, 1], verts[subsample, 2],
               s=1, alpha=0.3, color="gray", label="Points")

    # Menampilkan mesh (triangles) jika tersedia
    if len(tris) > 0:
        # Memilih subset triangle untuk visualisasi
        max_show = min(2000, len(tris))
        tri_subset = tris[:max_show]

        # Membuat polygon collection
        polygons = []
        for tri_idx in tri_subset:
            polygon = [verts[tri_idx[0]], verts[tri_idx[1]], verts[tri_idx[2]]]
            polygons.append(polygon)

        # Membuat dan menambahkan Poly3DCollection
        poly_coll = Poly3DCollection(polygons, alpha=0.4, facecolor="skyblue",
                                     edgecolor="navy", linewidth=0.3)
        ax.add_collection3d(poly_coll)

    # Mengatur judul
    ax.set_title(f"BPA Radius {nama} (r={result['radius']:.4f}, "
                 f"tri={result['n_triangles']})", fontsize=11)

    # Mengatur label sumbu
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan visualisasi individual
    ind_path = os.path.join(OUTPUT_DIR, f"11_bpa_{nama}.png")
    plt.savefig(ind_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Mencetak konfirmasi
    print(f"  Disimpan: {ind_path}")

print()


# ========================================================
# 7. MEMBUAT COMPARISON GRID
# ========================================================
print("=" * 60)
print("7. MEMBUAT COMPARISON GRID")
print("=" * 60)

# Membuat figure grid perbandingan (1x3)
fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                         subplot_kw={"projection": "3d"})

# Mengiterasi setiap hasil untuk grid
for i, (nama, result) in enumerate(bpa_results.items()):
    # Mendapatkan sumbu saat ini
    ax = axes[i]

    # Mendapatkan data
    verts = result["vertices"]
    tris = result["triangles"]

    # Menampilkan point cloud
    sub = np.random.choice(len(verts), min(400, len(verts)), replace=False)
    ax.scatter(verts[sub, 0], verts[sub, 1], verts[sub, 2],
               s=1, alpha=0.2, color="gray")

    # Menampilkan mesh
    if len(tris) > 0:
        # Memilih subset triangle
        max_show = min(1500, len(tris))
        tri_sub = tris[:max_show]

        # Membuat polygon
        polys = []
        for tri_idx in tri_sub:
            poly = [verts[tri_idx[0]], verts[tri_idx[1]], verts[tri_idx[2]]]
            polys.append(poly)

        # Menambahkan polygon collection
        pc = Poly3DCollection(polys, alpha=0.4, facecolor="skyblue",
                              edgecolor="navy", linewidth=0.2)
        ax.add_collection3d(pc)

    # Mengatur judul subplot
    ax.set_title(f"Radius {nama}\nr={result['radius']:.3f}, "
                 f"tri={result['n_triangles']}", fontsize=10)

    # Mengatur label sumbu
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)

# Mengatur judul utama
plt.suptitle("Perbandingan BPA dengan Berbagai Radius Bola",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan comparison grid
grid_path = os.path.join(OUTPUT_DIR, "11_bpa_comparison_grid.png")
plt.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close()

# Mencetak konfirmasi
print(f"  Disimpan: {grid_path}")
print()


# ========================================================
# 8. TABEL ANALISIS
# ========================================================
print("=" * 60)
print("8. TABEL ANALISIS HASIL BPA")
print("=" * 60)

# Mencetak header tabel
print()
print(f"  +{'='*64}+")
print(f"  | {'Metrik':<20} | {'Kecil':<12} | {'Sedang':<12} | {'Besar':<12} |")
print(f"  +{'-'*64}+")

# Menyiapkan data untuk setiap baris
metrik_rows = [
    ("Radius", [f"{bpa_results[n]['radius']:.4f}" for n in ["kecil", "sedang", "besar"]]),
    ("Jumlah Triangle", [str(bpa_results[n]["n_triangles"]) for n in ["kecil", "sedang", "besar"]]),
    ("Jumlah Vertices", [str(bpa_results[n]["n_vertices"]) for n in ["kecil", "sedang", "besar"]]),
]

# Menghitung coverage untuk setiap radius
coverages = []
for nama in ["kecil", "sedang", "besar"]:
    r = bpa_results[nama]
    if r["n_triangles"] > 0 and len(r["triangles"]) > 0:
        uv = len(np.unique(r["triangles"].flatten()))
        cov = uv / r["n_vertices"] * 100
    else:
        cov = 0.0
    coverages.append(f"{cov:.1f}%")

# Menambahkan baris coverage
metrik_rows.append(("Coverage", coverages))

# Mencetak setiap baris tabel
for metrik, values in metrik_rows:
    print(f"  | {metrik:<20} | {values[0]:<12} | {values[1]:<12} | {values[2]:<12} |")

# Mencetak footer tabel
print(f"  +{'='*64}+")
print()

# Mencetak interpretasi hasil
print("  Interpretasi:")
print("  - Radius kecil: lebih detail, lebih banyak holes (gap)")
print("  - Radius sedang: keseimbangan detail dan coverage")
print("  - Radius besar: coverage tinggi, detail berkurang")
print()


# ========================================================
# 9. MENYIMPAN SEMUA HASIL
# ========================================================
print("=" * 60)
print("9. MENYIMPAN SEMUA HASIL")
print("=" * 60)

# Menyimpan mesh hasil BPA terbaik sebagai PLY
best_nama = max(bpa_results.keys(), key=lambda k: bpa_results[k]["n_triangles"])
best_result = bpa_results[best_nama]

# Menentukan path output PLY
ply_output_path = os.path.join(OUTPUT_DIR, "11_bpa_best_mesh.ply")

# Memeriksa apakah ada triangle untuk disimpan
if best_result["n_triangles"] > 0:
    # Menulis file PLY mesh
    with open(ply_output_path, 'w') as f:
        # Menulis header PLY
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {best_result['n_vertices']}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {best_result['n_triangles']}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Menulis vertices
        for pt in best_result["vertices"]:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        # Menulis faces (triangles)
        for tri in best_result["triangles"]:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

    # Mencetak konfirmasi
    print(f"  Mesh terbaik ({best_nama}): {ply_output_path}")
    print(f"    Vertices: {best_result['n_vertices']}")
    print(f"    Triangles: {best_result['n_triangles']}")
else:
    # Mencetak pesan jika tidak ada mesh untuk disimpan
    print("  [INFO] Tidak ada mesh dengan triangle untuk disimpan")

print()

# Mencetak ringkasan akhir
print("=" * 60)
print("PERCOBAAN 11 SELESAI")
print("=" * 60)
print(f"  Output tersimpan di: {OUTPUT_DIR}")
print(f"  File yang dihasilkan:")
print(f"    - 11_bpa_kecil.png")
print(f"    - 11_bpa_sedang.png")
print(f"    - 11_bpa_besar.png")
print(f"    - 11_bpa_comparison_grid.png")
print(f"    - 11_bpa_best_mesh.ply")
