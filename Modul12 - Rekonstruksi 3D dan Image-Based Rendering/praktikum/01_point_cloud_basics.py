"""
==========================================================
PERCOBAAN 1: POINT CLOUD BASICS
Memuat, memvisualisasikan, dan memahami struktur data
point cloud. Mempelajari format PLY, properti titik,
bounding box, centroid.

Fungsi utama:
- open3d.io.read_point_cloud() (atau numpy manual)
- matplotlib Axes3D
- numpy statistical operations
==========================================================
"""

# Mengimpor library numpy untuk operasi numerik
import numpy as np

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor library os untuk operasi sistem file
import os

# Mengimpor library matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mengimpor Axes3D untuk plot 3D
from mpl_toolkits.mplot3d import Axes3D

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
    print("[INFO] Open3D tidak tersedia, menggunakan fallback NumPy+Matplotlib")

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
print("PERCOBAAN 1: POINT CLOUD BASICS")
print("=" * 60)
print()


def load_ply_manual(filepath):
    """
    Memuat file PLY secara manual tanpa Open3D.
    Mendukung format ASCII PLY.
    """
    # Menginisialisasi list untuk menyimpan titik
    points = []

    # Menginisialisasi list untuk menyimpan warna
    colors = []

    # Menginisialisasi list untuk menyimpan normal
    normals = []

    # Menandai apakah file memiliki warna
    has_colors = False

    # Menandai apakah file memiliki normal
    has_normals = False

    # Menginisialisasi jumlah vertex
    num_vertices = 0

    # Membuka file PLY untuk dibaca
    with open(filepath, 'r') as f:
        # Membaca header
        in_header = True

        # Menginisialisasi daftar properti
        properties = []

        # Mengiterasi setiap baris file
        for line in f:
            # Menghilangkan whitespace di awal dan akhir
            line = line.strip()

            # Memproses header
            if in_header:
                # Mengecek jumlah vertex
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                # Mengecek properti
                elif line.startswith("property"):
                    parts = line.split()
                    properties.append(parts[-1])
                    # Mengecek apakah ada warna
                    if parts[-1] in ['red', 'green', 'blue']:
                        has_colors = True
                    # Mengecek apakah ada normal
                    if parts[-1] in ['nx', 'ny', 'nz']:
                        has_normals = True
                # Mengecek akhir header
                elif line == "end_header":
                    in_header = False
                # Melanjutkan ke baris berikutnya jika masih header
                continue

            # Memproses data vertex
            values = line.split()

            # Mengecek apakah masih ada vertex yang perlu dibaca
            if len(points) >= num_vertices:
                break

            # Mengekstrak koordinat XYZ
            x, y, z = float(values[0]), float(values[1]), float(values[2])
            points.append([x, y, z])

            # Mencari index properti warna dan normal
            idx = 3

            # Mengekstrak warna jika tersedia
            if has_colors:
                r, g, b = int(values[idx]), int(values[idx + 1]), int(values[idx + 2])
                colors.append([r, g, b])
                idx += 3

            # Mengekstrak normal jika tersedia
            if has_normals:
                nx, ny, nz = float(values[idx]), float(values[idx + 1]), float(values[idx + 2])
                normals.append([nx, ny, nz])

    # Mengkonversi ke numpy array
    points = np.array(points, dtype=np.float64)

    # Mengkonversi warna jika tersedia
    colors = np.array(colors, dtype=np.uint8) if colors else None

    # Mengkonversi normal jika tersedia
    normals = np.array(normals, dtype=np.float64) if normals else None

    # Mengembalikan data point cloud
    return points, colors, normals, has_colors, has_normals


# ========================================================
# 1. MEMUAT POINT CLOUD DARI FILE PLY
# ========================================================

# Mencetak header bagian memuat point cloud
print("=" * 60)
print("1. MEMUAT POINT CLOUD DARI FILE PLY")
print("=" * 60)

# Menentukan path file PLY
ply_path = os.path.join(IMAGE_DIR, "bunny_point_cloud.ply")

# Mengecek apakah file PLY ada
if not os.path.exists(ply_path):
    print(f"[ERROR] File tidak ditemukan: {ply_path}")
    print("Jalankan download_image.py terlebih dahulu!")
    exit(1)

# Memuat point cloud menggunakan Open3D atau manual parser
if HAS_OPEN3D:
    # Memuat dengan Open3D
    pcd = o3d.io.read_point_cloud(ply_path)
    # Mengkonversi ke numpy array
    points = np.asarray(pcd.points)
    # Mengecek apakah memiliki warna
    has_colors = pcd.has_colors()
    colors_float = np.asarray(pcd.colors) if has_colors else None
    colors = (colors_float * 255).astype(np.uint8) if colors_float is not None else None
    # Mengecek apakah memiliki normal
    has_normals = pcd.has_normals()
    normals = np.asarray(pcd.normals) if has_normals else None
    print(f"  [Open3D] Point cloud dimuat dari: {os.path.basename(ply_path)}")
else:
    # Memuat dengan parser manual
    points, colors, normals, has_colors, has_normals = load_ply_manual(ply_path)
    print(f"  [Manual] Point cloud dimuat dari: {os.path.basename(ply_path)}")

# Mencetak jumlah titik yang dimuat
print(f"  Jumlah titik: {len(points)}")
print()


# ========================================================
# 2. MENAMPILKAN PROPERTI POINT CLOUD
# ========================================================

# Mencetak header bagian properti
print("=" * 60)
print("2. PROPERTI POINT CLOUD")
print("=" * 60)

# Mencetak jumlah total titik
print(f"  Jumlah titik         : {len(points)}")

# Mencetak tipe data
print(f"  Tipe data            : {points.dtype}")

# Mencetak dimensi array
print(f"  Dimensi array        : {points.shape}")

# Mencetak ketersediaan warna
print(f"  Memiliki warna       : {has_colors}")

# Mencetak ketersediaan normal
print(f"  Memiliki normal      : {has_normals}")

# Mencetak ukuran file
file_size = os.path.getsize(ply_path)
print(f"  Ukuran file          : {file_size / 1024:.1f} KB")
print()


# ========================================================
# 3. STATISTIK POINT CLOUD (NumPy)
# ========================================================

# Mencetak header bagian statistik
print("=" * 60)
print("3. STATISTIK POINT CLOUD")
print("=" * 60)

# Menghitung nilai minimum per sumbu
min_vals = np.min(points, axis=0)
print(f"  Minimum (X, Y, Z)   : ({min_vals[0]:.4f}, {min_vals[1]:.4f}, {min_vals[2]:.4f})")

# Menghitung nilai maksimum per sumbu
max_vals = np.max(points, axis=0)
print(f"  Maksimum (X, Y, Z)  : ({max_vals[0]:.4f}, {max_vals[1]:.4f}, {max_vals[2]:.4f})")

# Menghitung centroid (rata-rata)
centroid = np.mean(points, axis=0)
print(f"  Centroid (X, Y, Z)  : ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")

# Menghitung standar deviasi per sumbu
std_vals = np.std(points, axis=0)
print(f"  Std Dev (X, Y, Z)   : ({std_vals[0]:.4f}, {std_vals[1]:.4f}, {std_vals[2]:.4f})")

# Menghitung median per sumbu
median_vals = np.median(points, axis=0)
print(f"  Median (X, Y, Z)    : ({median_vals[0]:.4f}, {median_vals[1]:.4f}, {median_vals[2]:.4f})")
print()


# ========================================================
# 4. BOUNDING BOX
# ========================================================

# Mencetak header bagian bounding box
print("=" * 60)
print("4. BOUNDING BOX")
print("=" * 60)

# Menghitung dimensi bounding box
bbox_size = max_vals - min_vals
print(f"  Minimum corner       : ({min_vals[0]:.4f}, {min_vals[1]:.4f}, {min_vals[2]:.4f})")
print(f"  Maximum corner       : ({max_vals[0]:.4f}, {max_vals[1]:.4f}, {max_vals[2]:.4f})")
print(f"  Dimensi (W x H x D) : ({bbox_size[0]:.4f} x {bbox_size[1]:.4f} x {bbox_size[2]:.4f})")

# Menghitung volume bounding box
volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
print(f"  Volume bounding box  : {volume:.4f}")

# Menghitung diagonal bounding box
diagonal = np.linalg.norm(bbox_size)
print(f"  Diagonal             : {diagonal:.4f}")
print()


# ========================================================
# 5. MEMBUAT POINT CLOUD DARI SCRATCH (SPHERE)
# ========================================================

# Mencetak header bagian pembuatan sphere
print("=" * 60)
print("5. MEMBUAT POINT CLOUD SPHERE DARI SCRATCH")
print("=" * 60)

# Menentukan jumlah titik untuk sphere
n_sphere = 2000

# Membuat parameter u acak (azimuthal angle)
u = np.random.uniform(0, 2 * np.pi, n_sphere)

# Membuat parameter v acak (polar angle)
v = np.random.uniform(0, np.pi, n_sphere)

# Menentukan radius sphere
r_sphere = 1.0

# Menghitung koordinat X sphere
x_sphere = r_sphere * np.sin(v) * np.cos(u)

# Menghitung koordinat Y sphere
y_sphere = r_sphere * np.sin(v) * np.sin(u)

# Menghitung koordinat Z sphere
z_sphere = r_sphere * np.cos(v)

# Menggabungkan koordinat menjadi array Nx3
sphere_points = np.column_stack([x_sphere, y_sphere, z_sphere])

# Mencetak informasi sphere yang dibuat
print(f"  Sphere dibuat: {n_sphere} titik, radius={r_sphere}")
print(f"  Centroid sphere: ({np.mean(sphere_points, axis=0)})")
print()


# ========================================================
# 6. PEWARNAAN BERDASARKAN KETINGGIAN (Z-HEIGHT COLORMAP)
# ========================================================

# Mencetak header bagian pewarnaan
print("=" * 60)
print("6. PEWARNAAN BERDASARKAN KETINGGIAN (Z-HEIGHT)")
print("=" * 60)

# Menormalisasi nilai Z ke rentang [0, 1]
z_values = points[:, 2]
z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-8)

# Membuat colormap berdasarkan ketinggian (biru=rendah, merah=tinggi)
height_colors = np.zeros((len(points), 3))

# Menghitung komponen merah (tinggi = merah)
height_colors[:, 0] = z_norm

# Menghitung komponen hijau (tengah = hijau)
height_colors[:, 1] = 1.0 - np.abs(z_norm - 0.5) * 2

# Menghitung komponen biru (rendah = biru)
height_colors[:, 2] = 1.0 - z_norm

# Mencetak informasi pewarnaan
print(f"  Rentang Z: {z_values.min():.4f} - {z_values.max():.4f}")
print(f"  Colormap: Biru (rendah) -> Hijau (tengah) -> Merah (tinggi)")
print()


# ========================================================
# 7. CROPPING POINT CLOUD KE REGION
# ========================================================

# Mencetak header bagian cropping
print("=" * 60)
print("7. CROPPING POINT CLOUD KE REGION")
print("=" * 60)

# Menentukan batas region crop (hanya bagian atas)
z_threshold = centroid[2] + 0.3

# Membuat mask untuk titik-titik di atas threshold
crop_mask = points[:, 2] > z_threshold

# Menerapkan mask untuk mendapatkan titik yang di-crop
cropped_points = points[crop_mask]

# Menerapkan mask pada warna
cropped_colors = height_colors[crop_mask]

# Mencetak informasi cropping
print(f"  Threshold Z          : > {z_threshold:.4f}")
print(f"  Titik sebelum crop   : {len(points)}")
print(f"  Titik setelah crop   : {len(cropped_points)}")
print(f"  Persentase tersisa   : {len(cropped_points) / len(points) * 100:.1f}%")
print()


# ========================================================
# 8. VISUALISASI DENGAN MATPLOTLIB 3D (MULTIPLE ANGLES)
# ========================================================

# Mencetak header bagian visualisasi
print("=" * 60)
print("8. VISUALISASI POINT CLOUD (MULTIPLE ANGLES)")
print("=" * 60)

# --- Visualisasi 1: Point cloud original dari 4 sudut pandang ---
# Membuat figure dengan 2x2 subplot
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12),
                            subplot_kw={'projection': '3d'})

# Menentukan sudut pandang yang berbeda-beda
view_angles = [(30, 45), (60, 135), (15, 225), (75, 315)]

# Menentukan judul untuk setiap sudut pandang
view_titles = ["Sudut 1 (elev=30, azim=45)",
               "Sudut 2 (elev=60, azim=135)",
               "Sudut 3 (elev=15, azim=225)",
               "Sudut 4 (elev=75, azim=315)"]

# Mengiterasi setiap subplot
for idx, ax in enumerate(axes1.flat):
    # Menentukan jumlah titik yang ditampilkan (subsample untuk performa)
    sample_size = min(3000, len(points))

    # Membuat index acak untuk subsample
    sample_idx = np.random.choice(len(points), sample_size, replace=False)

    # Mengambil titik-titik subsample
    sample_pts = points[sample_idx]

    # Mengambil warna subsample
    sample_cols = height_colors[sample_idx]

    # Menggambar scatter plot 3D
    ax.scatter(sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2],
               c=sample_cols, s=1, alpha=0.6)

    # Mengatur sudut pandang
    ax.view_init(elev=view_angles[idx][0], azim=view_angles[idx][1])

    # Mengatur label sumbu
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Mengatur judul subplot
    ax.set_title(view_titles[idx], fontsize=10)

# Mengatur judul utama figure
fig1.suptitle("Point Cloud Original - Berbagai Sudut Pandang", fontsize=14, fontweight='bold')

# Mengatur layout agar rapi
plt.tight_layout()

# Menyimpan visualisasi pertama
output_path1 = os.path.join(OUTPUT_DIR, "01_point_cloud_multi_angle.png")
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"  Tersimpan: {os.path.basename(output_path1)}")

# Menutup figure
plt.close()


# --- Visualisasi 2: Perbandingan original vs cropped ---
# Membuat figure dengan 1x2 subplot
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6),
                            subplot_kw={'projection': '3d'})

# Menggambar point cloud original
sample_size_orig = min(3000, len(points))
sample_idx_orig = np.random.choice(len(points), sample_size_orig, replace=False)
axes2[0].scatter(points[sample_idx_orig, 0],
                 points[sample_idx_orig, 1],
                 points[sample_idx_orig, 2],
                 c=height_colors[sample_idx_orig], s=1, alpha=0.6)
axes2[0].set_title("Original Point Cloud", fontsize=12)
axes2[0].set_xlabel('X')
axes2[0].set_ylabel('Y')
axes2[0].set_zlabel('Z')

# Menggambar point cloud yang sudah di-crop
sample_size_crop = min(3000, len(cropped_points))
if sample_size_crop > 0:
    sample_idx_crop = np.random.choice(len(cropped_points), sample_size_crop, replace=False)
    axes2[1].scatter(cropped_points[sample_idx_crop, 0],
                     cropped_points[sample_idx_crop, 1],
                     cropped_points[sample_idx_crop, 2],
                     c=cropped_colors[sample_idx_crop], s=2, alpha=0.7)
axes2[1].set_title(f"Cropped (Z > {z_threshold:.2f})", fontsize=12)
axes2[1].set_xlabel('X')
axes2[1].set_ylabel('Y')
axes2[1].set_zlabel('Z')

# Mengatur judul utama
fig2.suptitle("Perbandingan Original vs Cropped Point Cloud", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi kedua
output_path2 = os.path.join(OUTPUT_DIR, "01_original_vs_cropped.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"  Tersimpan: {os.path.basename(output_path2)}")

# Menutup figure
plt.close()


# --- Visualisasi 3: Sphere dari scratch ---
# Membuat figure untuk sphere
fig3 = plt.figure(figsize=(8, 8))

# Membuat subplot 3D
ax3 = fig3.add_subplot(111, projection='3d')

# Menormalisasi Z sphere untuk colormap
z_sphere_norm = (z_sphere - z_sphere.min()) / (z_sphere.max() - z_sphere.min() + 1e-8)

# Menggambar scatter plot sphere
scatter = ax3.scatter(x_sphere, y_sphere, z_sphere,
                      c=z_sphere_norm, cmap='viridis', s=3, alpha=0.7)

# Menambahkan colorbar
plt.colorbar(scatter, ax=ax3, shrink=0.6, label='Ketinggian (Z)')

# Mengatur label sumbu
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

# Mengatur judul
ax3.set_title("Sphere Point Cloud (Dibuat dari Scratch)", fontsize=13, fontweight='bold')

# Mengatur aspek rasio yang sama
ax3.set_box_aspect([1, 1, 1])

# Menyimpan visualisasi sphere
output_path3 = os.path.join(OUTPUT_DIR, "01_sphere_from_scratch.png")
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"  Tersimpan: {os.path.basename(output_path3)}")

# Menutup figure
plt.close()


# --- Visualisasi 4: Bounding box dan statistik ---
# Membuat figure dengan informasi bounding box
fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8),
                          subplot_kw={'projection': '3d'})

# Menggambar titik-titik dengan subsample
sample_size_bb = min(2000, len(points))
sample_idx_bb = np.random.choice(len(points), sample_size_bb, replace=False)
ax4.scatter(points[sample_idx_bb, 0], points[sample_idx_bb, 1],
            points[sample_idx_bb, 2], c=height_colors[sample_idx_bb],
            s=1, alpha=0.4, label='Point Cloud')

# Menggambar bounding box menggunakan garis
# Menentukan 8 sudut bounding box
corners = np.array([
    [min_vals[0], min_vals[1], min_vals[2]],
    [max_vals[0], min_vals[1], min_vals[2]],
    [max_vals[0], max_vals[1], min_vals[2]],
    [min_vals[0], max_vals[1], min_vals[2]],
    [min_vals[0], min_vals[1], max_vals[2]],
    [max_vals[0], min_vals[1], max_vals[2]],
    [max_vals[0], max_vals[1], max_vals[2]],
    [min_vals[0], max_vals[1], max_vals[2]]
])

# Menentukan pasangan sudut untuk sisi-sisi bounding box
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# Menggambar setiap sisi bounding box
for edge in edges:
    # Mengambil titik-titik sisi
    p1, p2 = corners[edge[0]], corners[edge[1]]
    # Menggambar garis antara dua sudut
    ax4.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
               'r-', linewidth=1.5, alpha=0.7)

# Menandai centroid
ax4.scatter([centroid[0]], [centroid[1]], [centroid[2]],
            c='yellow', s=100, marker='*', edgecolors='black',
            linewidth=1, label='Centroid', zorder=5)

# Mengatur label sumbu
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')

# Mengatur judul
ax4.set_title("Point Cloud dengan Bounding Box dan Centroid",
              fontsize=13, fontweight='bold')

# Menambahkan legenda
ax4.legend(loc='upper right')

# Menyimpan visualisasi bounding box
output_path4 = os.path.join(OUTPUT_DIR, "01_bounding_box_centroid.png")
plt.savefig(output_path4, dpi=150, bbox_inches='tight')
print(f"  Tersimpan: {os.path.basename(output_path4)}")

# Menutup figure
plt.close()


# ========================================================
# 9. RINGKASAN
# ========================================================

# Mencetak header ringkasan
print()
print("=" * 60)
print("RINGKASAN PERCOBAAN 1")
print("=" * 60)
print(f"  File PLY dimuat      : {os.path.basename(ply_path)}")
print(f"  Total titik          : {len(points)}")
print(f"  Centroid             : ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
print(f"  Bounding box size    : ({bbox_size[0]:.4f} x {bbox_size[1]:.4f} x {bbox_size[2]:.4f})")
print(f"  Sphere (dari scratch): {n_sphere} titik")
print(f"  Cropped points       : {len(cropped_points)} titik")
print(f"  Visualisasi disimpan : 4 file di output/")
print("=" * 60)
