"""
==========================================================
PERCOBAAN 12: ALPHA SHAPES
Mempelajari rekonstruksi surface menggunakan Alpha Shapes,
generalisasi dari convex hull yang memungkinkan bentuk
concave.

Fungsi utama:
- open3d create_from_point_cloud_alpha_shape()
- scipy.spatial.ConvexHull(), Delaunay()
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

# Mengimpor ConvexHull dan Delaunay dari scipy
from scipy.spatial import ConvexHull, Delaunay

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
    print("[INFO] Open3D tidak tersedia, menggunakan fallback SciPy")

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
print("PERCOBAAN 12: ALPHA SHAPES")
print("=" * 60)
print()


# ========================================================
# FUNGSI UTILITAS
# ========================================================

def load_ply_manual(filepath):
    """
    Memuat file PLY secara manual tanpa Open3D.
    """
    # Menginisialisasi list untuk titik dan warna
    points = []
    colors = []

    # Membuka file PLY untuk dibaca
    with open(filepath, 'r') as f:
        # Menandai bahwa kita masih di header
        in_header = True
        # Menandai keberadaan warna
        has_colors = False

        # Membaca setiap baris file
        for line in f:
            # Menghapus whitespace di awal/akhir baris
            line = line.strip()

            # Memproses header PLY
            if in_header:
                # Memeriksa apakah ada properti warna
                if line.startswith("property") and "red" in line:
                    has_colors = True
                # Mendeteksi akhir header
                elif line == "end_header":
                    in_header = False
                continue

            # Memproses data vertex
            parts = line.split()
            # Memeriksa apakah data valid
            if len(parts) < 3:
                continue
            # Membaca koordinat XYZ
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            # Membaca warna jika tersedia
            if has_colors and len(parts) >= 6:
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])

    # Mengkonversi list ke numpy array
    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8) if colors else None

    # Mengembalikan data point cloud
    return points, colors


def generate_sample_point_cloud(num_points=2000):
    """
    Membuat point cloud sintetis dengan bentuk concave (torus-like).
    """
    # Mencetak informasi pembuatan data sintetis
    print("  Membuat point cloud sintetis berbentuk torus...")

    # Membuat parameter sudut untuk torus
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    # Menentukan radius mayor dan minor torus
    R = 1.0
    r = 0.4

    # Menghitung koordinat X torus
    x = (R + r * np.cos(phi)) * np.cos(theta)
    # Menghitung koordinat Y torus
    y = (R + r * np.cos(phi)) * np.sin(theta)
    # Menghitung koordinat Z torus
    z = r * np.sin(phi)

    # Menambahkan sedikit noise untuk realisme
    noise = np.random.normal(0, 0.02, (num_points, 3))

    # Menggabungkan koordinat menjadi array titik
    points = np.column_stack([x, y, z]) + noise

    # Mengembalikan point cloud sintetis
    return points


def compute_circumradius(p1, p2, p3):
    """
    Menghitung circumradius dari segitiga yang dibentuk oleh 3 titik.
    """
    # Menghitung vektor sisi segitiga
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)

    # Menghitung semi-perimeter
    s = (a + b + c) / 2.0

    # Menghitung luas segitiga dengan formula Heron
    area_sq = s * (s - a) * (s - b) * (s - c)
    area_sq = max(area_sq, 1e-15)
    area = np.sqrt(area_sq)

    # Menghitung circumradius = abc / (4 * area)
    circumradius = (a * b * c) / (4.0 * area + 1e-15)

    # Mengembalikan circumradius
    return circumradius


def alpha_shape_delaunay(points, alpha):
    """
    Menghitung alpha shape menggunakan Delaunay triangulation.
    Simplex dihapus jika circumradius > 1/alpha.
    """
    # Memeriksa apakah alpha bernilai nol (convex hull)
    if alpha <= 0:
        # Menghitung convex hull sebagai pengganti
        hull = ConvexHull(points)
        return hull.simplices

    # Menghitung Delaunay triangulation
    tri = Delaunay(points)

    # Menghitung batas radius = 1/alpha
    radius_limit = 1.0 / alpha

    # Menginisialisasi list untuk simplex yang lolos filter
    valid_simplices = []

    # Mengiterasi setiap simplex dalam triangulasi
    for simplex in tri.simplices:
        # Mendapatkan titik-titik simplex
        pts = points[simplex]

        # Menghitung circumradius untuk setiap kombinasi 3 titik
        if len(pts) == 4:
            # Untuk 3D tetrahedra, periksa setiap face
            faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            # Menandai apakah simplex valid
            valid = True
            # Memeriksa setiap face
            for face in faces:
                cr = compute_circumradius(pts[face[0]], pts[face[1]], pts[face[2]])
                if cr > radius_limit:
                    valid = False
                    break
            # Menambahkan simplex jika valid
            if valid:
                for face in faces:
                    valid_simplices.append(simplex[face])
        else:
            # Untuk 2D, langsung hitung circumradius segitiga
            cr = compute_circumradius(pts[0], pts[1], pts[2])
            if cr < radius_limit:
                valid_simplices.append(simplex)

    # Mengkonversi ke numpy array
    if len(valid_simplices) == 0:
        return np.array([]).reshape(0, 3).astype(int)

    # Mengembalikan simplex yang valid
    return np.array(valid_simplices, dtype=int)


# ========================================================
# BAGIAN 1: MEMUAT POINT CLOUD
# ========================================================
print("=" * 60)
print("BAGIAN 1: MEMUAT POINT CLOUD")
print("=" * 60)
print()

# Menentukan path file PLY
ply_path = os.path.join(IMAGE_DIR, "bunny_point_cloud.ply")

# Download otomatis jika file PLY tidak tersedia
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
print(f"  Memuat file: {ply_path}")
points, colors = load_ply_manual(ply_path)

# Mencetak informasi point cloud yang dimuat
print(f"  Jumlah titik: {len(points)}")
print(f"  Rentang X: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]")
print(f"  Rentang Y: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
print(f"  Rentang Z: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]")
print()


# ========================================================
# BAGIAN 2: CONVEX HULL (ALPHA = 0)
# ========================================================
print("=" * 60)
print("BAGIAN 2: CONVEX HULL (ALPHA = 0)")
print("=" * 60)
print()

# Menghitung convex hull dari point cloud
hull = ConvexHull(points)

# Mencetak statistik convex hull
print(f"  Jumlah face convex hull  : {len(hull.simplices)}")
print(f"  Volume convex hull       : {hull.volume:.6f}")
print(f"  Luas permukaan           : {hull.area:.6f}")
print(f"  Jumlah vertex pada hull  : {len(hull.vertices)}")
print()


# ========================================================
# BAGIAN 3: ALPHA SHAPES DENGAN BERBAGAI NILAI ALPHA
# ========================================================
print("=" * 60)
print("BAGIAN 3: ALPHA SHAPES DENGAN BERBAGAI NILAI ALPHA")
print("=" * 60)
print()

# Menentukan nilai-nilai alpha yang akan dicoba
alpha_values = [0.01, 0.05, 0.1, 0.5]

# Menginisialisasi dictionary untuk menyimpan hasil
results = {}

# Menyimpan hasil convex hull sebagai referensi
results["Convex Hull"] = hull.simplices

# Mengiterasi setiap nilai alpha
for alpha in alpha_values:
    # Mencetak informasi alpha yang sedang diproses
    print(f"  Memproses alpha = {alpha}...")

    # Memeriksa apakah Open3D tersedia
    if HAS_OPEN3D:
        # Membuat objek point cloud Open3D
        pcd = o3d.geometry.PointCloud()
        # Mengisi titik-titik ke point cloud
        pcd.points = o3d.utility.Vector3dVector(points)
        # Mengestimasi normal untuk alpha shape
        pcd.estimate_normals()

        # Menghitung alpha shape menggunakan Open3D
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            # Mengambil triangles sebagai numpy array
            triangles = np.asarray(mesh.triangles)
            # Menyimpan hasil
            results[f"Alpha={alpha}"] = triangles
            # Mencetak jumlah face
            print(f"    Jumlah face: {len(triangles)}")
        except Exception as e:
            # Menangani error dan menggunakan fallback
            print(f"    Open3D error: {e}, menggunakan fallback...")
            triangles = alpha_shape_delaunay(points, alpha)
            results[f"Alpha={alpha}"] = triangles
            print(f"    Jumlah face (fallback): {len(triangles)}")
    else:
        # Menggunakan fallback SciPy Delaunay
        triangles = alpha_shape_delaunay(points, alpha)
        # Menyimpan hasil
        results[f"Alpha={alpha}"] = triangles
        # Mencetak jumlah face
        print(f"    Jumlah face: {len(triangles)}")

    print()


# ========================================================
# BAGIAN 4: STATISTIK PER ALPHA VALUE
# ========================================================
print("=" * 60)
print("BAGIAN 4: STATISTIK PER ALPHA VALUE")
print("=" * 60)
print()

# Mencetak header tabel statistik
print(f"  {'Metode':<20} {'Jumlah Face':>15} {'Unik Vertex':>15}")
print(f"  {'-'*20} {'-'*15} {'-'*15}")

# Mengiterasi setiap hasil untuk mencetak statistik
for name, faces in results.items():
    # Menghitung jumlah face
    num_faces = len(faces)
    # Menghitung jumlah vertex unik
    if num_faces > 0:
        unique_verts = len(np.unique(faces.flatten()))
    else:
        unique_verts = 0
    # Mencetak baris statistik
    print(f"  {name:<20} {num_faces:>15} {unique_verts:>15}")

print()


# ========================================================
# BAGIAN 5: VISUALISASI SEMUA HASIL DALAM GRID
# ========================================================
print("=" * 60)
print("BAGIAN 5: VISUALISASI SEMUA HASIL")
print("=" * 60)
print()

# Menentukan jumlah subplot yang dibutuhkan
num_plots = len(results) + 1
# Menghitung jumlah kolom dan baris grid
ncols = 3
nrows = int(np.ceil(num_plots / ncols))

# Membuat figure dan axes untuk grid visualisasi
fig = plt.figure(figsize=(6 * ncols, 5 * nrows))

# Menambahkan judul utama figure
fig.suptitle("Perbandingan Alpha Shapes dengan Berbagai Nilai Alpha",
             fontsize=14, fontweight='bold')

# --- Subplot 1: Point cloud asli ---
ax0 = fig.add_subplot(nrows, ncols, 1, projection='3d')
# Mengambil sampel titik untuk visualisasi (agar tidak terlalu berat)
sample_idx = np.random.choice(len(points), min(1000, len(points)), replace=False)
# Menampilkan point cloud asli
ax0.scatter(points[sample_idx, 0], points[sample_idx, 1], points[sample_idx, 2],
            c='steelblue', s=1, alpha=0.5)
# Mengatur judul subplot
ax0.set_title("Point Cloud Asli", fontsize=10)
# Mengatur label sumbu
ax0.set_xlabel("X", fontsize=8)
ax0.set_ylabel("Y", fontsize=8)
ax0.set_zlabel("Z", fontsize=8)

# --- Subplot 2-N: Hasil alpha shapes ---
for idx, (name, faces) in enumerate(results.items()):
    # Membuat subplot 3D
    ax = fig.add_subplot(nrows, ncols, idx + 2, projection='3d')

    # Memeriksa apakah ada face yang ditampilkan
    if len(faces) > 0:
        # Membatasi jumlah face untuk performa visualisasi
        max_faces = min(len(faces), 2000)
        face_idx = np.random.choice(len(faces), max_faces, replace=False) if len(faces) > max_faces else np.arange(len(faces))

        # Membuat koleksi polygon 3D
        polygons = []
        for fi in face_idx:
            # Mendapatkan vertex dari setiap face
            verts = points[faces[fi]]
            polygons.append(verts)

        # Membuat objek Poly3DCollection
        poly_collection = Poly3DCollection(polygons, alpha=0.4,
                                           edgecolor='darkblue', linewidth=0.3)
        # Mengatur warna face
        poly_collection.set_facecolor('skyblue')
        # Menambahkan koleksi ke axes
        ax.add_collection3d(poly_collection)

        # Mengatur batas sumbu berdasarkan data
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        ax.set_zlim(points[:, 2].min(), points[:, 2].max())
    else:
        # Menampilkan teks jika tidak ada face
        ax.text2D(0.5, 0.5, "Tidak ada face", transform=ax.transAxes,
                  ha='center', va='center', fontsize=12, color='red')

    # Mengatur judul subplot
    ax.set_title(f"{name}\n({len(faces)} faces)", fontsize=10)
    # Mengatur label sumbu
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)

# Mengatur layout figure
plt.tight_layout()

# Menyimpan visualisasi perbandingan
output_path = os.path.join(OUTPUT_DIR, "12_alpha_shapes_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Visualisasi disimpan: {output_path}")

# Menutup figure untuk membebaskan memori
plt.close()
print()


# ========================================================
# BAGIAN 6: ANALISIS DETAIL VS LUBANG
# ========================================================
print("=" * 60)
print("BAGIAN 6: ANALISIS DETAIL VS LUBANG")
print("=" * 60)
print()

# Menginisialisasi list untuk data grafik
alpha_list = []
face_counts = []

# Mengumpulkan data dari hasil alpha shapes
for name, faces in results.items():
    # Mengekstrak nilai alpha dari nama
    if "Alpha=" in name:
        alpha_val = float(name.split("=")[1])
        alpha_list.append(alpha_val)
        face_counts.append(len(faces))

# Membuat grafik hubungan alpha vs jumlah face
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))

# Menampilkan grafik garis dan scatter
ax2.plot(alpha_list, face_counts, 'bo-', markersize=8, linewidth=2)

# Mengatur judul grafik
ax2.set_title("Pengaruh Nilai Alpha terhadap Jumlah Face", fontsize=13, fontweight='bold')

# Mengatur label sumbu X
ax2.set_xlabel("Nilai Alpha", fontsize=11)

# Mengatur label sumbu Y
ax2.set_ylabel("Jumlah Face", fontsize=11)

# Menambahkan grid pada grafik
ax2.grid(True, alpha=0.3)

# Menambahkan anotasi pada setiap titik data
for a, f in zip(alpha_list, face_counts):
    ax2.annotate(f"  α={a}\n  {f} faces", (a, f), fontsize=9)

# Mengatur layout figure
plt.tight_layout()

# Menyimpan grafik analisis
output_path2 = os.path.join(OUTPUT_DIR, "12_alpha_vs_faces.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"  Grafik analisis disimpan: {output_path2}")

# Menutup figure
plt.close()

# Mencetak kesimpulan analisis
print()
print("  Kesimpulan:")
print("  - Alpha kecil → banyak detail, banyak lubang")
print("  - Alpha besar → lebih smooth, kurang detail")
print("  - Alpha = 0 → convex hull (tidak ada lubang)")
print()


# ========================================================
# SELESAI
# ========================================================
print("=" * 60)
print("PERCOBAAN 12 SELESAI")
print("=" * 60)
print(f"Semua output disimpan di: {OUTPUT_DIR}")
