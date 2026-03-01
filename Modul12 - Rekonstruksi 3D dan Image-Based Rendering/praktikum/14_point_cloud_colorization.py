"""
==========================================================
PERCOBAAN 14: COLORIZATION POINT CLOUD
Mempelajari berbagai cara mewarnai point cloud: dari gambar
(projecting), dari height (z-coordinate), dari normal
direction, dan dari curvature.

Fungsi utama:
- numpy array operations
- matplotlib colormaps
- cv2.projectPoints()
==========================================================
"""

# Mengimpor library numpy untuk operasi numerik
import numpy as np

# Mengimpor library os untuk operasi sistem file
import os

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor library matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mengimpor Axes3D untuk plot 3D
from mpl_toolkits.mplot3d import Axes3D

# Mengimpor colormap dari matplotlib
from matplotlib import cm

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
    print("[INFO] Open3D tidak tersedia, menggunakan fallback numpy/matplotlib")

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
print("PERCOBAAN 14: COLORIZATION POINT CLOUD")
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

    # Membuka file PLY untuk dibaca
    with open(filepath, 'r') as f:
        # Menandai bahwa kita masih di header
        in_header = True

        # Membaca setiap baris file
        for line in f:
            # Menghapus whitespace
            line = line.strip()

            # Memproses header PLY
            if in_header:
                if line.startswith("property") and "red" in line:
                    has_colors = True
                elif line.startswith("property") and "nx" in line:
                    has_normals = True
                elif line == "end_header":
                    in_header = False
                continue

            # Memproses data vertex
            parts = line.split()
            if len(parts) < 3:
                continue

            # Membaca koordinat XYZ
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])

            # Menentukan index saat ini
            idx = 3

            # Membaca normal jika tersedia
            if has_normals and len(parts) >= idx + 3:
                normals.append([float(parts[idx]), float(parts[idx+1]), float(parts[idx+2])])
                idx += 3

            # Membaca warna jika tersedia
            if has_colors and len(parts) >= idx + 3:
                colors.append([int(parts[idx]), int(parts[idx+1]), int(parts[idx+2])])

    # Mengkonversi list ke numpy array
    points = np.array(points, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64) if normals else None
    colors = np.array(colors, dtype=np.uint8) if colors else None

    # Mengembalikan data point cloud
    return points, normals, colors


def generate_sample_point_cloud(num_points=3000):
    """
    Membuat point cloud sintetis berbentuk bola dengan tonjolan.
    """
    # Mencetak informasi pembuatan data sintetis
    print("  Membuat point cloud sintetis...")

    # Membuat parameter sudut
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)

    # Menghitung radius dengan variasi untuk membentuk objek menarik
    r = 1.0 + 0.3 * np.sin(3 * theta) * np.sin(2 * phi)

    # Menghitung koordinat kartesian
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Menambahkan sedikit noise
    noise = np.random.normal(0, 0.02, (num_points, 3))

    # Menggabungkan koordinat
    points = np.column_stack([x, y, z]) + noise

    # Mengembalikan point cloud sintetis
    return points


def estimate_normals_pca(points, k=15):
    """
    Mengestimasi normal menggunakan PCA pada tetangga terdekat.
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

        # Mengorientasikan normal menjauh dari centroid keseluruhan
        center = np.mean(points, axis=0)
        if np.dot(normal, points[i] - center) < 0:
            normal = -normal

        # Menyimpan normal
        normals[i] = normal

    # Mengembalikan array normal yang sudah dinormalisasi
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return normals / norms


def estimate_curvature(points, normals, k=15):
    """
    Mengestimasi curvature dari variasi normal pada tetangga.
    """
    # Menghitung jumlah titik
    n = len(points)

    # Menginisialisasi array curvature
    curvature = np.zeros(n)

    # Mengiterasi setiap titik
    for i in range(n):
        # Menghitung jarak ke semua titik lain
        dists = np.linalg.norm(points - points[i], axis=1)

        # Mendapatkan index k tetangga terdekat
        nn_idx = np.argsort(dists)[1:k+1]

        # Menghitung rata-rata perbedaan normal dengan tetangga
        normal_diffs = 1.0 - np.abs(np.dot(normals[nn_idx], normals[i]))

        # Curvature adalah rata-rata perbedaan normal
        curvature[i] = np.mean(normal_diffs)

    # Mengembalikan array curvature
    return curvature


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
points, normals_loaded, colors_loaded = load_ply_manual(ply_path)

# Mencetak informasi point cloud
print(f"  Jumlah titik: {len(points)}")
print(f"  Memiliki normal: {normals_loaded is not None}")
print()


# ========================================================
# BAGIAN 2: COLOR BY Z-HEIGHT (RAINBOW COLORMAP)
# ========================================================
print("=" * 60)
print("BAGIAN 2: COLOR BY Z-HEIGHT (RAINBOW)")
print("=" * 60)
print()

# Mengekstrak komponen Z dari semua titik
z_values = points[:, 2]

# Menormalisasi Z ke rentang [0, 1]
z_min = z_values.min()
z_max = z_values.max()
z_normalized = (z_values - z_min) / (z_max - z_min + 1e-10)

# Menerapkan colormap rainbow/jet
color_height = cm.jet(z_normalized)[:, :3]

# Mencetak statistik
print(f"  Rentang Z: [{z_min:.4f}, {z_max:.4f}]")
print(f"  Warna ditentukan oleh posisi vertikal (Z)")
print()


# ========================================================
# BAGIAN 3: COLOR BY DISTANCE FROM CENTROID
# ========================================================
print("=" * 60)
print("BAGIAN 3: COLOR BY DISTANCE FROM CENTROID")
print("=" * 60)
print()

# Menghitung centroid dari point cloud
centroid = np.mean(points, axis=0)

# Mencetak posisi centroid
print(f"  Centroid: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")

# Menghitung jarak setiap titik dari centroid
distances = np.linalg.norm(points - centroid, axis=1)

# Menormalisasi jarak ke rentang [0, 1]
dist_normalized = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)

# Menerapkan colormap viridis
color_distance = cm.viridis(dist_normalized)[:, :3]

# Mencetak statistik jarak
print(f"  Jarak minimum dari centroid: {distances.min():.4f}")
print(f"  Jarak maksimum dari centroid: {distances.max():.4f}")
print(f"  Jarak rata-rata: {distances.mean():.4f}")
print()


# ========================================================
# BAGIAN 4: COLOR BY NORMAL DIRECTION (XYZ → RGB)
# ========================================================
print("=" * 60)
print("BAGIAN 4: COLOR BY NORMAL DIRECTION")
print("=" * 60)
print()

# Memeriksa apakah normal tersedia atau perlu dihitung
if normals_loaded is not None:
    # Menggunakan normal yang sudah ada
    normals = normals_loaded
    print("  Menggunakan normal dari file PLY")
else:
    # Mengestimasi normal secara manual
    print("  Menghitung normal dengan PCA (k=15)...")
    normals = estimate_normals_pca(points, k=15)
    print("  Normal berhasil dihitung")

# Mengkonversi normal direction ke warna RGB
# Normal X,Y,Z dalam rentang [-1, 1] → dimapping ke [0, 1] untuk RGB
color_normal = (normals + 1.0) / 2.0

# Meng-clip nilai untuk memastikan rentang valid
color_normal = np.clip(color_normal, 0, 1)

# Mencetak statistik normal
print(f"  Rata-rata normal X: {normals[:, 0].mean():.4f}")
print(f"  Rata-rata normal Y: {normals[:, 1].mean():.4f}")
print(f"  Rata-rata normal Z: {normals[:, 2].mean():.4f}")
print()


# ========================================================
# BAGIAN 5: COLOR FROM PROJECTED IMAGE
# ========================================================
print("=" * 60)
print("BAGIAN 5: COLOR FROM PROJECTED IMAGE")
print("=" * 60)
print()

# Memuat gambar tekstur dari file; jika tidak ada, download otomatis
texture_path = os.path.join(IMAGE_DIR, "texture_sample.jpg")

# Menjalankan download_image.py jika tekstur belum ada
if not os.path.exists(texture_path):
    print("[WARN] texture_sample.jpg tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)

texture_img = cv2.imread(texture_path)
if texture_img is None:
    raise FileNotFoundError(
        f"[ERROR] {texture_path} tidak tersedia.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )
print(f"  Tekstur dimuat: {texture_path}")

# Membuat parameter kamera untuk proyeksi
fx, fy = 500.0, 500.0
cx, cy = 256.0, 256.0
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
# Membuat vektor rotasi dan translasi
rvec = np.array([0.5, 0.0, 0.0], dtype=np.float64)
tvec = np.array([0.0, 0.0, 3.0], dtype=np.float64)
dist_coeffs = np.zeros(5, dtype=np.float64)

# Memproyeksikan titik-titik 3D ke bidang gambar
proj_pts, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
proj_pts = proj_pts.reshape(-1, 2)

# Mengambil warna dari gambar pada posisi proyeksi
h_tex, w_tex = texture_img.shape[:2]
color_projected = np.full((len(points), 3), 0.5)
valid_count = 0

# Mengiterasi setiap titik yang diproyeksikan
for i in range(len(proj_pts)):
    # Membulatkan koordinat
    px = int(round(proj_pts[i, 0]))
    py = int(round(proj_pts[i, 1]))
    # Memeriksa validitas koordinat
    if 0 <= px < w_tex and 0 <= py < h_tex:
        bgr = texture_img[py, px]
        color_projected[i] = [bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0]
        valid_count += 1

# Mencetak statistik proyeksi
print(f"  Titik dengan warna valid: {valid_count}/{len(points)}")
print()


# ========================================================
# BAGIAN 6: COLOR BY CURVATURE
# ========================================================
print("=" * 60)
print("BAGIAN 6: COLOR BY CURVATURE")
print("=" * 60)
print()

# Mengestimasi curvature dari point cloud
print("  Menghitung curvature (flat=biru, curved=merah)...")
curvature = estimate_curvature(points, normals, k=15)

# Menormalisasi curvature ke rentang [0, 1]
curv_min = curvature.min()
curv_max = curvature.max()
curv_normalized = (curvature - curv_min) / (curv_max - curv_min + 1e-10)

# Menerapkan colormap coolwarm (biru → merah)
color_curvature = cm.coolwarm(curv_normalized)[:, :3]

# Mencetak statistik curvature
print(f"  Curvature minimum: {curv_min:.6f}")
print(f"  Curvature maksimum: {curv_max:.6f}")
print(f"  Curvature rata-rata: {curvature.mean():.6f}")
print()


# ========================================================
# BAGIAN 7: VISUALISASI 6-PANEL
# ========================================================
print("=" * 60)
print("BAGIAN 7: VISUALISASI 6-PANEL")
print("=" * 60)
print()

# Mengambil sampel titik untuk visualisasi (agar tidak terlalu berat)
max_vis = min(2000, len(points))
vis_idx = np.random.choice(len(points), max_vis, replace=False)

# Menyiapkan data dan konfigurasi untuk setiap panel
panels = [
    ("Z-Height (Jet)", color_height, "jet"),
    ("Jarak dari Centroid", color_distance, "viridis"),
    ("Arah Normal (RGB)", color_normal, None),
    ("Proyeksi Gambar", color_projected, None),
    ("Curvature", color_curvature, "coolwarm"),
    ("Asli (Uniform)", np.full((len(points), 3), [0.3, 0.6, 0.9]), None),
]

# Membuat figure 2x3 untuk 6 panel
fig = plt.figure(figsize=(18, 11))

# Menambahkan judul utama
fig.suptitle("Metode Colorization Point Cloud", fontsize=15, fontweight='bold')

# Mengiterasi setiap panel
for idx, (title, colors_arr, cmap_name) in enumerate(panels):
    # Membuat subplot 3D
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')

    # Mengambil warna untuk titik yang divisualisasikan
    vis_colors = colors_arr[vis_idx]

    # Menampilkan scatter plot 3D
    ax.scatter(points[vis_idx, 0], points[vis_idx, 1], points[vis_idx, 2],
               c=vis_colors, s=2, alpha=0.7)

    # Mengatur judul panel
    ax.set_title(title, fontsize=10, fontweight='bold')

    # Mengatur label sumbu
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)

    # Mengecilkan ukuran tick label
    ax.tick_params(labelsize=6)

# Mengatur layout figure
plt.tight_layout()

# Menyimpan visualisasi 6-panel
output_path = os.path.join(OUTPUT_DIR, "14_colorization_6panel.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Visualisasi 6-panel disimpan: {output_path}")

# Menutup figure
plt.close()
print()


# ========================================================
# BAGIAN 8: MENYIMPAN VISUALISASI INDIVIDUAL
# ========================================================
print("=" * 60)
print("BAGIAN 8: MENYIMPAN VISUALISASI INDIVIDUAL")
print("=" * 60)
print()

# Menyimpan setiap metode sebagai gambar terpisah
for idx, (title, colors_arr, cmap_name) in enumerate(panels):
    # Membuat figure individual
    fig_single = plt.figure(figsize=(8, 6))
    ax_single = fig_single.add_subplot(111, projection='3d')

    # Mengambil warna untuk titik yang divisualisasikan
    vis_colors = colors_arr[vis_idx]

    # Menampilkan scatter plot
    sc = ax_single.scatter(points[vis_idx, 0], points[vis_idx, 1], points[vis_idx, 2],
                           c=vis_colors, s=3, alpha=0.8)

    # Mengatur judul
    ax_single.set_title(f"Colorization: {title}", fontsize=12, fontweight='bold')

    # Mengatur label sumbu
    ax_single.set_xlabel("X")
    ax_single.set_ylabel("Y")
    ax_single.set_zlabel("Z")

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan gambar individual
    save_name = f"14_colorization_{idx+1}_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Mencetak informasi penyimpanan
    print(f"  Disimpan: {save_name}")

    # Menutup figure
    plt.close()

print()


# ========================================================
# SELESAI
# ========================================================
print("=" * 60)
print("PERCOBAAN 14 SELESAI")
print("=" * 60)
print(f"Semua output disimpan di: {OUTPUT_DIR}")
