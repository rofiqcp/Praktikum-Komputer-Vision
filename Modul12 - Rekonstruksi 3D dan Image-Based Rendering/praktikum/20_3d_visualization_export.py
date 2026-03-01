"""
==========================================================
PERCOBAAN 20: VISUALISASI 3D DAN EXPORT
Mempelajari berbagai teknik visualisasi dan format export
untuk data 3D: PLY, OBJ, rendering dari berbagai sudut,
dan pembuatan turntable video.

Fungsi utama:
- open3d visualization (atau matplotlib 3D)
- numpy operations
- cv2.VideoWriter()
==========================================================
"""

# Mengimpor library numpy untuk operasi numerik
import numpy as np

# Mengimpor library os untuk operasi sistem file
import os

# Mengimpor library OpenCV untuk pemrosesan citra dan video
import cv2

# Mengimpor library matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mengimpor Axes3D untuk plot 3D
from mpl_toolkits.mplot3d import Axes3D

# Mengimpor art3d untuk polygon 3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Mengimpor io untuk buffer gambar
from io import BytesIO

# Mencoba mengimpor Open3D untuk visualisasi lanjut
try:
    # Mengimpor library Open3D
    import open3d as o3d
    # Menandai ketersediaan Open3D
    HAS_OPEN3D = True
    print("[INFO] Open3D berhasil diimpor")
except ImportError:
    # Menandai bahwa Open3D tidak tersedia
    HAS_OPEN3D = False
    print("[INFO] Open3D tidak tersedia, menggunakan matplotlib 3D")

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
print("PERCOBAAN 20: VISUALISASI 3D DAN EXPORT")
print("=" * 60)

# ========================================================
# BAGIAN 1: MEMBUAT POINT CLOUD DAN MESH
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 1: Membuat Point Cloud dan Mesh")
print("=" * 60)

# Membuat point cloud berbentuk bunny sederhana (dari parametrik)
jumlah_titik = 3000

# Membuat bola sebagai dasar bentuk
phi = np.random.uniform(0, 2 * np.pi, jumlah_titik)
theta = np.random.uniform(0, np.pi, jumlah_titik)
r_base = 1.0

# Menghitung koordinat kartesian dari koordinat bola
x_ball = r_base * np.sin(theta) * np.cos(phi)
y_ball = r_base * np.sin(theta) * np.sin(phi)
z_ball = r_base * np.cos(theta)

# Menambahkan deformasi untuk membuat bentuk lebih menarik
# Menambahkan tonjolan (bump) di sisi atas
mask_atas = z_ball > 0.5
x_ball[mask_atas] *= 0.8
y_ball[mask_atas] *= 0.8
z_ball[mask_atas] *= 1.3

# Menambahkan dua "telinga" kecil
jumlah_telinga = 200
for ear_offset in [-0.3, 0.3]:
    # Membuat titik untuk telinga
    phi_ear = np.random.uniform(0, 2 * np.pi, jumlah_telinga)
    r_ear = np.random.uniform(0, 0.25, jumlah_telinga)
    x_ear = r_ear * np.cos(phi_ear) + ear_offset
    y_ear = r_ear * np.sin(phi_ear)
    z_ear = np.random.uniform(1.2, 1.8, jumlah_telinga)

    # Menggabungkan telinga ke point cloud utama
    x_ball = np.concatenate([x_ball, x_ear])
    y_ball = np.concatenate([y_ball, y_ear])
    z_ball = np.concatenate([z_ball, z_ear])

# Menggabungkan koordinat menjadi array point cloud
point_cloud = np.column_stack([x_ball, y_ball, z_ball])

# Menghitung warna berdasarkan posisi (colormap)
# Normalisasi z untuk warna
z_norm = (point_cloud[:, 2] - point_cloud[:, 2].min()) / \
         (point_cloud[:, 2].max() - point_cloud[:, 2].min() + 1e-8)

# Membuat warna RGB dari colormap (biru ke merah)
warna_r = (z_norm * 255).astype(np.uint8)
warna_g = ((1 - np.abs(z_norm - 0.5) * 2) * 200).astype(np.uint8)
warna_b = ((1 - z_norm) * 255).astype(np.uint8)
warna_pc = np.column_stack([warna_r, warna_g, warna_b])

# Mencetak informasi point cloud
print(f"Jumlah titik point cloud: {len(point_cloud)}")
print(f"Bounding box: x=[{point_cloud[:,0].min():.2f}, {point_cloud[:,0].max():.2f}]")
print(f"              y=[{point_cloud[:,1].min():.2f}, {point_cloud[:,1].max():.2f}]")
print(f"              z=[{point_cloud[:,2].min():.2f}, {point_cloud[:,2].max():.2f}]")

# Membuat mesh sederhana (icosphere approx) menggunakan ConvexHull
from scipy.spatial import ConvexHull

# Menggunakan subset titik untuk mesh agar lebih bersih
idx_mesh = np.random.choice(len(point_cloud), min(500, len(point_cloud)), replace=False)
titik_mesh = point_cloud[idx_mesh]

# Menghitung ConvexHull untuk membuat mesh
try:
    hull = ConvexHull(titik_mesh)
    faces_mesh = hull.simplices
    vertices_mesh = titik_mesh
    print(f"Mesh vertices: {len(vertices_mesh)}, faces: {len(faces_mesh)}")
except Exception as e:
    print(f"ConvexHull gagal: {e}")
    faces_mesh = np.zeros((0, 3), dtype=int)
    vertices_mesh = titik_mesh

# ========================================================
# BAGIAN 2: RENDER DARI 36 SUDUT PANDANG (TURNTABLE)
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 2: Render dari 36 Sudut Pandang")
print("=" * 60)

# Menentukan jumlah sudut pandang (setiap 10 derajat)
jumlah_sudut = 36

# Menentukan ukuran frame rendering
frame_width = 400
frame_height = 400

# Menyimpan semua frame rendering
frames = []

# Fungsi untuk merender point cloud dari sudut tertentu
def render_sudut(points, colors, elev, azim, frame_w, frame_h, title=""):
    """Merender point cloud dari sudut pandang tertentu."""
    # Membuat figure matplotlib
    fig = plt.figure(figsize=(frame_w/100, frame_h/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Menampilkan point cloud dengan warna
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors / 255.0, s=1, alpha=0.6
    )

    # Mengatur sudut pandang
    ax.view_init(elev=elev, azim=azim)

    # Mengatur label sumbu
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Mengatur judul jika ada
    if title:
        ax.set_title(title, fontsize=10)

    # Mengatur batas sumbu agar konsisten
    max_range = 2.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # Menyimpan figure ke buffer
    fig.canvas.draw()

    # Mengkonversi canvas ke numpy array
    buf = fig.canvas.buffer_rgba()
    gambar = np.asarray(buf)

    # Mengubah RGBA ke BGR untuk OpenCV
    gambar_bgr = cv2.cvtColor(gambar, cv2.COLOR_RGBA2BGR)

    # Menutup figure untuk menghemat memori
    plt.close(fig)

    # Mengubah ukuran ke dimensi yang tepat
    gambar_bgr = cv2.resize(gambar_bgr, (frame_w, frame_h))

    return gambar_bgr

# Merender dari setiap sudut pandang
for i in range(jumlah_sudut):
    # Menghitung sudut azimuth (0-360 derajat)
    azimuth = i * (360 / jumlah_sudut)

    # Menentukan elevasi tetap
    elevasi = 25

    # Merender frame
    frame = render_sudut(
        point_cloud, warna_pc,
        elevasi, azimuth,
        frame_width, frame_height,
        title=f"Azimuth: {azimuth:.0f}°"
    )

    # Menambahkan frame ke daftar
    frames.append(frame)

    # Mencetak progress setiap 6 frame
    if (i + 1) % 6 == 0:
        print(f"  Rendered {i+1}/{jumlah_sudut} frames (azimuth={azimuth:.0f}°)")

# ========================================================
# BAGIAN 3: MEMBUAT VIDEO TURNTABLE
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 3: Membuat Video Turntable")
print("=" * 60)

# Menentukan path file video output
path_video = os.path.join(OUTPUT_DIR, "20_turntable_animation.avi")

# Menentukan FPS video
fps_video = 10

# Membuat VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter(
    path_video, fourcc, fps_video,
    (frame_width, frame_height)
)

# Menulis setiap frame ke video (2 loop untuk animasi lebih smooth)
for loop in range(2):
    for frame in frames:
        # Menulis frame ke video
        video_writer.write(frame)

# Melepas VideoWriter
video_writer.release()

# Mencetak informasi video
total_frames = len(frames) * 2
durasi = total_frames / fps_video
print(f"Video disimpan: {path_video}")
print(f"Total frames: {total_frames}")
print(f"FPS: {fps_video}")
print(f"Durasi: {durasi:.1f} detik")

# ========================================================
# BAGIAN 4: EXPORT POINT CLOUD SEBAGAI PLY
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 4: Export Point Cloud sebagai PLY")
print("=" * 60)

# Menentukan path file PLY
path_ply = os.path.join(OUTPUT_DIR, "20_point_cloud.ply")

# Menulis file PLY secara manual
with open(path_ply, 'w') as f:
    # Menulis header PLY
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(point_cloud)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")

    # Menulis data vertex
    for i in range(len(point_cloud)):
        # Menulis koordinat dan warna setiap titik
        f.write(f"{point_cloud[i,0]:.6f} {point_cloud[i,1]:.6f} {point_cloud[i,2]:.6f} "
                f"{warna_pc[i,0]} {warna_pc[i,1]} {warna_pc[i,2]}\n")

# Mencetak informasi file PLY
ukuran_ply = os.path.getsize(path_ply)
print(f"PLY disimpan: {path_ply}")
print(f"Ukuran file: {ukuran_ply / 1024:.1f} KB")

# ========================================================
# BAGIAN 5: EXPORT POINT CLOUD SEBAGAI XYZ DAN CSV
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 5: Export Point Cloud sebagai XYZ dan CSV")
print("=" * 60)

# Export sebagai format XYZ (x y z per baris)
path_xyz = os.path.join(OUTPUT_DIR, "20_point_cloud.xyz")
with open(path_xyz, 'w') as f:
    for i in range(len(point_cloud)):
        # Menulis koordinat x, y, z dipisahkan spasi
        f.write(f"{point_cloud[i,0]:.6f} {point_cloud[i,1]:.6f} {point_cloud[i,2]:.6f}\n")

# Mencetak informasi file XYZ
ukuran_xyz = os.path.getsize(path_xyz)
print(f"XYZ disimpan: {path_xyz} ({ukuran_xyz/1024:.1f} KB)")

# Export sebagai format CSV (dengan header)
path_csv = os.path.join(OUTPUT_DIR, "20_point_cloud.csv")
with open(path_csv, 'w') as f:
    # Menulis header CSV
    f.write("x,y,z,r,g,b\n")

    for i in range(len(point_cloud)):
        # Menulis data per baris dengan format CSV
        f.write(f"{point_cloud[i,0]:.6f},{point_cloud[i,1]:.6f},{point_cloud[i,2]:.6f},"
                f"{warna_pc[i,0]},{warna_pc[i,1]},{warna_pc[i,2]}\n")

# Mencetak informasi file CSV
ukuran_csv = os.path.getsize(path_csv)
print(f"CSV disimpan: {path_csv} ({ukuran_csv/1024:.1f} KB)")

# ========================================================
# BAGIAN 6: EXPORT MESH SEBAGAI OBJ
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 6: Export Mesh sebagai OBJ")
print("=" * 60)

# Menentukan path file OBJ
path_obj = os.path.join(OUTPUT_DIR, "20_mesh.obj")

# Menulis file OBJ secara manual
with open(path_obj, 'w') as f:
    # Menulis komentar header
    f.write("# OBJ file - Percobaan 20\n")
    f.write(f"# Vertices: {len(vertices_mesh)}\n")
    f.write(f"# Faces: {len(faces_mesh)}\n")
    f.write("# Generated by Modul 12 Praktikum\n\n")

    # Menulis vertices (format: v x y z)
    for v in vertices_mesh:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

    # Menambahkan baris kosong antara vertices dan faces
    f.write("\n")

    # Menulis faces (format: f v1 v2 v3, indeks dimulai dari 1)
    for face in faces_mesh:
        # OBJ menggunakan indeks berbasis 1
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

# Mencetak informasi file OBJ
ukuran_obj = os.path.getsize(path_obj)
print(f"OBJ disimpan: {path_obj} ({ukuran_obj/1024:.1f} KB)")

# ========================================================
# BAGIAN 7: VISUALISASI DENGAN COORDINATE AXES
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 7: Visualisasi dengan Coordinate Axes")
print("=" * 60)

# Membuat figure untuk visualisasi dengan axes
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': '3d'})

# Menampilkan point cloud
ax2.scatter(
    point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
    c=warna_pc / 255.0, s=2, alpha=0.5
)

# Menggambar sumbu koordinat (X=merah, Y=hijau, Z=biru)
panjang_sumbu = 2.5

# Menggambar sumbu X (merah)
ax2.quiver(0, 0, 0, panjang_sumbu, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
ax2.text(panjang_sumbu + 0.2, 0, 0, "X", color='red', fontsize=12, fontweight='bold')

# Menggambar sumbu Y (hijau)
ax2.quiver(0, 0, 0, 0, panjang_sumbu, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
ax2.text(0, panjang_sumbu + 0.2, 0, "Y", color='green', fontsize=12, fontweight='bold')

# Menggambar sumbu Z (biru)
ax2.quiver(0, 0, 0, 0, 0, panjang_sumbu, color='blue', arrow_length_ratio=0.1, linewidth=2)
ax2.text(0, 0, panjang_sumbu + 0.2, "Z", color='blue', fontsize=12, fontweight='bold')

# Mengatur judul
ax2.set_title("Point Cloud dengan Coordinate Axes", fontsize=14, fontweight='bold')

# Mengatur label sumbu
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan visualisasi dengan axes
path_axes = os.path.join(OUTPUT_DIR, "20_visualisasi_axes.png")
plt.savefig(path_axes, dpi=150, bbox_inches='tight')
print(f"Visualisasi axes disimpan: {path_axes}")

# Menutup figure
plt.close(fig2)

# ========================================================
# BAGIAN 8: VISUALISASI DENGAN BOUNDING BOX
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 8: Visualisasi dengan Bounding Box")
print("=" * 60)

# Membuat figure untuk bounding box
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': '3d'})

# Menampilkan point cloud
ax3.scatter(
    point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
    c=warna_pc / 255.0, s=2, alpha=0.5
)

# Menghitung bounding box
bbox_min = point_cloud.min(axis=0)
bbox_max = point_cloud.max(axis=0)

# Mendefinisikan 8 sudut bounding box
sudut_bbox = np.array([
    [bbox_min[0], bbox_min[1], bbox_min[2]],
    [bbox_max[0], bbox_min[1], bbox_min[2]],
    [bbox_max[0], bbox_max[1], bbox_min[2]],
    [bbox_min[0], bbox_max[1], bbox_min[2]],
    [bbox_min[0], bbox_min[1], bbox_max[2]],
    [bbox_max[0], bbox_min[1], bbox_max[2]],
    [bbox_max[0], bbox_max[1], bbox_max[2]],
    [bbox_min[0], bbox_max[1], bbox_max[2]]
])

# Mendefinisikan 12 tepi bounding box (pasangan indeks sudut)
tepi_bbox = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bawah
    (4, 5), (5, 6), (6, 7), (7, 4),  # Atas
    (0, 4), (1, 5), (2, 6), (3, 7)   # Vertikal
]

# Menggambar setiap tepi bounding box
for i, j in tepi_bbox:
    # Menggambar garis tepi
    ax3.plot3D(
        [sudut_bbox[i, 0], sudut_bbox[j, 0]],
        [sudut_bbox[i, 1], sudut_bbox[j, 1]],
        [sudut_bbox[i, 2], sudut_bbox[j, 2]],
        'r-', linewidth=1.5, alpha=0.7
    )

# Mengatur judul
ax3.set_title("Point Cloud dengan Bounding Box", fontsize=14, fontweight='bold')

# Mengatur label sumbu
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")

# Mencetak informasi bounding box
print(f"Bounding box min: ({bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f})")
print(f"Bounding box max: ({bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f})")
print(f"Ukuran: ({bbox_max[0]-bbox_min[0]:.3f}, {bbox_max[1]-bbox_min[1]:.3f}, {bbox_max[2]-bbox_min[2]:.3f})")

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan visualisasi bounding box
path_bbox = os.path.join(OUTPUT_DIR, "20_visualisasi_bbox.png")
plt.savefig(path_bbox, dpi=150, bbox_inches='tight')
print(f"Visualisasi bounding box disimpan: {path_bbox}")

# Menutup figure
plt.close(fig3)

# ========================================================
# BAGIAN 9: SHOWCASE MULTI-ANGLE GRID
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 9: Showcase Multi-Angle Grid")
print("=" * 60)

# Membuat grid 3x3 dari sudut pandang terpilih
sudut_terpilih = [0, 4, 8, 12, 16, 20, 24, 28, 32]

# Membuat figure grid
fig4, axes4 = plt.subplots(3, 3, figsize=(15, 15))

# Mengatur judul figure
fig4.suptitle("Multi-Angle Showcase (Turntable Views)",
              fontsize=16, fontweight='bold')

# Menampilkan setiap sudut pandang terpilih
for idx, frame_idx in enumerate(sudut_terpilih):
    # Menghitung posisi grid
    baris = idx // 3
    kolom = idx % 3

    # Mengambil frame yang sudah dirender
    frame = frames[frame_idx]

    # Mengkonversi BGR ke RGB untuk matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Menampilkan frame
    axes4[baris, kolom].imshow(frame_rgb)

    # Menghitung sudut azimuth
    azim_deg = frame_idx * (360 / jumlah_sudut)

    # Mengatur judul subplot
    axes4[baris, kolom].set_title(f"Azimuth: {azim_deg:.0f}°")
    axes4[baris, kolom].axis('off')

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan showcase grid
path_showcase = os.path.join(OUTPUT_DIR, "20_showcase_multi_angle.png")
plt.savefig(path_showcase, dpi=150, bbox_inches='tight')
print(f"Showcase grid disimpan: {path_showcase}")

# Menutup figure
plt.close(fig4)

# ========================================================
# BAGIAN 10: RINGKASAN EXPORT
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 10: Ringkasan Export")
print("=" * 60)

# Membuat daftar semua file yang diekspor
exported_files = [
    ("PLY (Point Cloud)", path_ply),
    ("XYZ (Point Cloud)", path_xyz),
    ("CSV (Point Cloud)", path_csv),
    ("OBJ (Mesh)", path_obj),
    ("AVI (Turntable Video)", path_video),
    ("PNG (Coordinate Axes)", path_axes),
    ("PNG (Bounding Box)", path_bbox),
    ("PNG (Multi-Angle Grid)", path_showcase),
]

# Mencetak tabel ringkasan
print(f"\n{'Format':<25s} {'Ukuran':>10s}  Path")
print("-" * 80)

# Menghitung total ukuran
total_ukuran = 0

# Iterasi setiap file yang diekspor
for label, path in exported_files:
    if os.path.exists(path):
        # Mendapatkan ukuran file
        ukuran = os.path.getsize(path)
        total_ukuran += ukuran

        # Menentukan satuan ukuran
        if ukuran > 1024 * 1024:
            ukuran_str = f"{ukuran / 1024 / 1024:.2f} MB"
        else:
            ukuran_str = f"{ukuran / 1024:.1f} KB"

        # Mencetak informasi file
        print(f"{label:<25s} {ukuran_str:>10s}  {os.path.basename(path)}")
    else:
        print(f"{label:<25s} {'N/A':>10s}  (tidak ditemukan)")

# Mencetak total ukuran
print("-" * 80)
if total_ukuran > 1024 * 1024:
    print(f"{'TOTAL':<25s} {total_ukuran/1024/1024:.2f} MB")
else:
    print(f"{'TOTAL':<25s} {total_ukuran/1024:.1f} KB")

# ========================================================
# RINGKASAN
# ========================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 20: VISUALISASI 3D DAN EXPORT")
print("=" * 60)
print(f"Jumlah titik point cloud : {len(point_cloud)}")
print(f"Jumlah vertices mesh     : {len(vertices_mesh)}")
print(f"Jumlah faces mesh        : {len(faces_mesh)}")
print(f"Sudut turntable          : {jumlah_sudut} views (setiap 10°)")
print(f"Format export            : PLY, XYZ, CSV, OBJ, AVI, PNG")
print(f"\nSemua output disimpan di: {OUTPUT_DIR}")
print("=" * 60)
