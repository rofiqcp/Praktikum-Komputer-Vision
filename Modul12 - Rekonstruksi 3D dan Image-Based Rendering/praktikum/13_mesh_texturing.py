"""
==========================================================
PERCOBAAN 13: TEXTURE MAPPING PADA MESH
Mempelajari cara memproyeksikan warna/texture dari gambar
2D ke permukaan mesh 3D.

Fungsi utama:
- cv2.projectPoints()
- numpy interpolation
- matplotlib 3D plot_trisurf()
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

# Mengimpor art3d untuk polygon 3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Mengimpor Triangulation dari matplotlib
from matplotlib.tri import Triangulation

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
    print("[INFO] Open3D tidak tersedia, menggunakan fallback matplotlib/cv2")

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
print("PERCOBAAN 13: TEXTURE MAPPING PADA MESH")
print("=" * 60)
print()


# ========================================================
# FUNGSI UTILITAS
# ========================================================

def create_terrain_mesh(grid_size=30):
    """
    Membuat mesh terrain (dataran berbukit) sederhana.
    """
    # Membuat grid koordinat X dan Y
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    # Membuat meshgrid 2D
    X, Y = np.meshgrid(x, y)

    # Menghitung ketinggian Z menggunakan fungsi sinusoidal
    Z = 0.5 * np.sin(X * 1.5) * np.cos(Y * 1.5) + 0.3 * np.sin(X * 3)

    # Membuat vertices dari grid
    vertices = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])

    # Membuat triangulasi dari grid
    triangles = []
    # Mengiterasi setiap sel grid untuk membuat dua segitiga
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # Menghitung index vertex dari sudut-sudut sel
            v00 = i * grid_size + j
            v01 = i * grid_size + j + 1
            v10 = (i + 1) * grid_size + j
            v11 = (i + 1) * grid_size + j + 1
            # Menambahkan segitiga pertama
            triangles.append([v00, v01, v10])
            # Menambahkan segitiga kedua
            triangles.append([v01, v11, v10])

    # Mengkonversi ke numpy array
    triangles = np.array(triangles, dtype=int)

    # Mengembalikan vertices dan triangles
    return vertices, triangles, X, Y, Z


def create_camera_parameters(image_width=512, image_height=512):
    """
    Membuat parameter kamera sintetis untuk proyeksi.
    """
    # Menentukan focal length kamera
    fx = 500.0
    fy = 500.0

    # Menentukan titik pusat optik
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Membuat matriks intrinsik kamera
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Membuat vektor rotasi (kamera melihat dari atas-depan)
    rvec = np.array([0.8, 0.0, 0.0], dtype=np.float64)

    # Membuat vektor translasi
    tvec = np.array([0.0, -1.0, 5.0], dtype=np.float64)

    # Membuat koefisien distorsi (tanpa distorsi)
    dist_coeffs = np.zeros(5, dtype=np.float64)

    # Mengembalikan parameter kamera
    return camera_matrix, rvec, tvec, dist_coeffs


def create_texture_image(width=512, height=512):
    """
    Membuat gambar tekstur sintetis untuk mapping.
    """
    # Membuat gambar dengan pola checkerboard berwarna
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Menentukan ukuran kotak checkerboard
    block_size = 64

    # Menentukan warna-warna untuk checkerboard
    color_pairs = [
        ([50, 120, 200], [200, 180, 50]),
        ([180, 60, 60], [60, 180, 60]),
    ]

    # Mengisi setiap blok dengan warna
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Menentukan pasangan warna berdasarkan posisi
            ci = (i // block_size) % 2
            cj = (j // block_size) % 2
            pair_idx = ((i // block_size) + (j // block_size)) % 2
            # Memilih warna dari pasangan
            color = color_pairs[pair_idx % len(color_pairs)][cj]
            # Mengisi blok dengan warna yang dipilih
            img[i:i+block_size, j:j+block_size] = color

    # Menambahkan gradien halus sebagai overlay
    grad_y = np.linspace(0, 50, height).reshape(-1, 1, 1)
    grad_x = np.linspace(0, 30, width).reshape(1, -1, 1)
    # Menambahkan gradien dan clipping
    img = np.clip(img.astype(np.float32) + grad_y + grad_x, 0, 255).astype(np.uint8)

    # Mengembalikan gambar tekstur
    return img


def project_vertices_to_image(vertices, camera_matrix, rvec, tvec, dist_coeffs):
    """
    Memproyeksikan vertex 3D ke bidang gambar 2D.
    """
    # Menggunakan cv2.projectPoints untuk proyeksi perspektif
    image_points, _ = cv2.projectPoints(
        vertices.astype(np.float64),
        rvec, tvec, camera_matrix, dist_coeffs
    )

    # Mengubah bentuk array hasil proyeksi
    image_points = image_points.reshape(-1, 2)

    # Mengembalikan titik-titik 2D hasil proyeksi
    return image_points


def sample_colors_from_image(image, image_points):
    """
    Mengambil sampel warna dari gambar pada posisi yang diproyeksikan.
    """
    # Mendapatkan dimensi gambar
    h, w = image.shape[:2]

    # Menginisialisasi array warna
    colors = np.zeros((len(image_points), 3), dtype=np.float64)

    # Menginisialisasi mask untuk titik yang valid
    valid_mask = np.ones(len(image_points), dtype=bool)

    # Mengiterasi setiap titik proyeksi
    for i, (px, py) in enumerate(image_points):
        # Membulatkan koordinat ke integer
        ix = int(round(px))
        iy = int(round(py))

        # Memeriksa apakah koordinat berada di dalam gambar
        if 0 <= ix < w and 0 <= iy < h:
            # Mengambil warna BGR dari gambar
            bgr = image[iy, ix]
            # Mengkonversi BGR ke RGB dan normalisasi ke [0, 1]
            colors[i] = [bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0]
        else:
            # Menandai titik sebagai tidak valid
            valid_mask[i] = False
            # Memberikan warna default abu-abu
            colors[i] = [0.5, 0.5, 0.5]

    # Mengembalikan warna dan mask validitas
    return colors, valid_mask


# ========================================================
# BAGIAN 1: MEMBUAT MESH 3D
# ========================================================
print("=" * 60)
print("BAGIAN 1: MEMBUAT MESH TERRAIN 3D")
print("=" * 60)
print()

# Membuat mesh terrain
vertices, triangles, X, Y, Z = create_terrain_mesh(grid_size=30)

# Mencetak informasi mesh
print(f"  Jumlah vertex   : {len(vertices)}")
print(f"  Jumlah triangles: {len(triangles)}")
print(f"  Rentang X: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]")
print(f"  Rentang Y: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]")
print(f"  Rentang Z: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]")
print()


# ========================================================
# BAGIAN 2: MEMBUAT PARAMETER KAMERA DAN TEKSTUR
# ========================================================
print("=" * 60)
print("BAGIAN 2: PARAMETER KAMERA DAN TEKSTUR")
print("=" * 60)
print()

# Membuat parameter kamera
camera_matrix, rvec, tvec, dist_coeffs = create_camera_parameters()

# Mencetak matriks kamera
print("  Matriks Kamera Intrinsik:")
print(f"    fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
print(f"    cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
print()

# Memuat tekstur dari file; jika tidak ada, download otomatis
texture_path = os.path.join(IMAGE_DIR, "texture_sample.jpg")
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
print(f"  Tekstur dimuat dari: {texture_path}")

# Menyimpan tekstur yang digunakan
tex_save = os.path.join(OUTPUT_DIR, "13_texture_used.png")
cv2.imwrite(tex_save, texture_img)
print(f"  Tekstur disimpan: {tex_save}")
print()


# ========================================================
# BAGIAN 3: MEMPROYEKSIKAN VERTEX KE GAMBAR
# ========================================================
print("=" * 60)
print("BAGIAN 3: MEMPROYEKSIKAN VERTEX KE GAMBAR")
print("=" * 60)
print()

# Memproyeksikan semua vertex mesh ke bidang gambar
image_points = project_vertices_to_image(vertices, camera_matrix, rvec, tvec, dist_coeffs)

# Mencetak statistik proyeksi
print(f"  Jumlah titik diproyeksikan: {len(image_points)}")
print(f"  Rentang X proyeksi: [{image_points[:, 0].min():.1f}, {image_points[:, 0].max():.1f}]")
print(f"  Rentang Y proyeksi: [{image_points[:, 1].min():.1f}, {image_points[:, 1].max():.1f}]")
print()


# ========================================================
# BAGIAN 4: SAMPLING WARNA DARI TEKSTUR
# ========================================================
print("=" * 60)
print("BAGIAN 4: SAMPLING WARNA DARI TEKSTUR")
print("=" * 60)
print()

# Mengambil sampel warna dari gambar tekstur
vertex_colors, valid_mask = sample_colors_from_image(texture_img, image_points)

# Menghitung jumlah titik valid dan tidak valid
num_valid = np.sum(valid_mask)
num_invalid = np.sum(~valid_mask)

# Mencetak statistik warna
print(f"  Titik dengan warna valid  : {num_valid}")
print(f"  Titik di luar gambar      : {num_invalid}")
print(f"  Rata-rata warna R: {vertex_colors[:, 0].mean():.3f}")
print(f"  Rata-rata warna G: {vertex_colors[:, 1].mean():.3f}")
print(f"  Rata-rata warna B: {vertex_colors[:, 2].mean():.3f}")
print()


# ========================================================
# BAGIAN 5: VISUALISASI MESH BERTEKSTUR
# ========================================================
print("=" * 60)
print("BAGIAN 5: VISUALISASI MESH BERTEKSTUR")
print("=" * 60)
print()

# Menghitung warna rata-rata per face untuk trisurf
face_colors = np.zeros((len(triangles), 4))
for i, tri in enumerate(triangles):
    # Menghitung rata-rata warna dari 3 vertex tiap face
    avg_color = vertex_colors[tri].mean(axis=0)
    # Menyimpan warna face dengan alpha
    face_colors[i] = [avg_color[0], avg_color[1], avg_color[2], 0.9]

# Membuat figure untuk visualisasi
fig = plt.figure(figsize=(16, 6))

# --- Subplot 1: Mesh tanpa tekstur ---
ax1 = fig.add_subplot(131, projection='3d')
# Menampilkan mesh tanpa tekstur menggunakan plot_trisurf
ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                 triangles=triangles, color='lightgray', edgecolor='gray',
                 linewidth=0.2, alpha=0.7)
# Mengatur judul subplot
ax1.set_title("Mesh Tanpa Tekstur", fontsize=11, fontweight='bold')
# Mengatur label sumbu
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# --- Subplot 2: Mesh bertekstur ---
ax2 = fig.add_subplot(132, projection='3d')
# Membuat koleksi polygon 3D dengan warna tekstur
polygons = []
for tri in triangles:
    polygons.append(vertices[tri])
# Membuat objek Poly3DCollection
poly_col = Poly3DCollection(polygons, linewidth=0.1, edgecolor='gray')
# Mengatur warna face dari warna yang di-sample
poly_col.set_facecolors(face_colors)
# Menambahkan koleksi polygon ke axes
ax2.add_collection3d(poly_col)
# Mengatur batas sumbu
ax2.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
ax2.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
ax2.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
# Mengatur judul subplot
ax2.set_title("Mesh dengan Tekstur", fontsize=11, fontweight='bold')
# Mengatur label sumbu
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

# --- Subplot 3: Tekstur asli ---
ax3 = fig.add_subplot(133)
# Menampilkan gambar tekstur (konversi BGR ke RGB)
ax3.imshow(cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB))
# Mengatur judul subplot
ax3.set_title("Gambar Tekstur", fontsize=11, fontweight='bold')
# Mematikan sumbu
ax3.axis('off')

# Mengatur layout figure
plt.tight_layout()

# Menyimpan visualisasi perbandingan
output_path = os.path.join(OUTPUT_DIR, "13_textured_mesh_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Visualisasi disimpan: {output_path}")

# Menutup figure
plt.close()
print()


# ========================================================
# BAGIAN 6: PROYEKSI DARI SUDUT KAMERA BERBEDA
# ========================================================
print("=" * 60)
print("BAGIAN 6: PROYEKSI DARI SUDUT KAMERA BERBEDA")
print("=" * 60)
print()

# Menentukan beberapa sudut kamera untuk proyeksi
camera_angles = [
    ("Depan", np.array([0.8, 0.0, 0.0]), np.array([0.0, -1.0, 5.0])),
    ("Samping", np.array([0.6, 0.5, 0.0]), np.array([1.0, -0.5, 5.0])),
    ("Atas", np.array([1.2, 0.0, 0.0]), np.array([0.0, -2.0, 4.0])),
]

# Membuat figure untuk perbandingan sudut kamera
fig2, axes = plt.subplots(1, len(camera_angles), figsize=(6 * len(camera_angles), 5),
                          subplot_kw={'projection': '3d'})

# Mengiterasi setiap sudut kamera
for idx, (name, rv, tv) in enumerate(camera_angles):
    # Mencetak informasi sudut kamera
    print(f"  Memproses sudut kamera: {name}")

    # Memproyeksikan vertex dari sudut kamera ini
    pts_2d = project_vertices_to_image(vertices, camera_matrix, rv, tv, dist_coeffs)

    # Mengambil warna dari tekstur
    cam_colors, cam_valid = sample_colors_from_image(texture_img, pts_2d)

    # Menghitung warna per face
    cam_face_colors = np.zeros((len(triangles), 4))
    for i, tri in enumerate(triangles):
        avg = cam_colors[tri].mean(axis=0)
        cam_face_colors[i] = [avg[0], avg[1], avg[2], 0.9]

    # Membuat koleksi polygon
    polys = [vertices[tri] for tri in triangles]
    pc = Poly3DCollection(polys, linewidth=0.1, edgecolor='gray')
    pc.set_facecolors(cam_face_colors)

    # Menambahkan ke subplot
    ax = axes[idx]
    ax.add_collection3d(pc)
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    # Mengatur judul
    ax.set_title(f"Kamera: {name}\n(Valid: {np.sum(cam_valid)}/{len(cam_valid)})",
                 fontsize=10, fontweight='bold')
    # Mengatur label
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Mencetak jumlah titik valid
    print(f"    Titik valid: {np.sum(cam_valid)}")

# Mengatur layout figure
plt.tight_layout()

# Menyimpan perbandingan sudut kamera
output_path2 = os.path.join(OUTPUT_DIR, "13_multi_camera_texturing.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"\n  Perbandingan sudut kamera disimpan: {output_path2}")

# Menutup figure
plt.close()
print()


# ========================================================
# SELESAI
# ========================================================
print("=" * 60)
print("PERCOBAAN 13 SELESAI")
print("=" * 60)
print(f"Semua output disimpan di: {OUTPUT_DIR}")
