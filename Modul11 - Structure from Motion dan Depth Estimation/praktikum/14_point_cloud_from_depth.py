"""
==========================================================================
PERCOBAAN 14: MEMBUAT POINT CLOUD DARI DEPTH MAP
==========================================================================
Program ini mempelajari cara mengkonversi depth map menjadi point cloud
3D dan memvisualisasikannya menggunakan Matplotlib dari berbagai sudut
pandang. Dua metode konversi dibandingkan: manual (Z=f*B/d) dan
menggunakan cv2.reprojectImageTo3D().

Konsep utama:
- Point cloud adalah kumpulan titik 3D (X, Y, Z) yang merepresentasikan
  permukaan objek dalam ruang tiga dimensi
- Konversi manual: Z = f*B/d, X = (x - cx)*Z/f, Y = (y - cy)*Z/f
- cv2.reprojectImageTo3D() menggunakan matriks Q dari stereo calibration
- Titik invalid (disparity <= 0) harus difilter sebelum visualisasi
- Warna titik diambil dari gambar asli untuk visualisasi realistis

Fungsi utama yang dipelajari:
- cv2.reprojectImageTo3D()  : Konversi disparity ke koordinat 3D
- cv2.StereoSGBM_create()   : Menghitung disparity map
- np.meshgrid()              : Membuat grid koordinat piksel
- matplotlib Axes3D scatter  : Visualisasi scatter plot 3D
- cv2.normalize()            : Normalisasi untuk visualisasi

Hasil: Point cloud 3D dari berbagai sudut pandang dengan warna dari
       gambar asli, perbandingan metode konversi manual vs reprojectImageTo3D
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk membuat dan menyimpan visualisasi
import matplotlib.pyplot as plt

# Mengimpor Axes3D untuk plot 3D (diperlukan meskipun tidak dipanggil langsung)
from mpl_toolkits.mplot3d import Axes3D

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan judul percobaan
print("=" * 60)
print("PERCOBAAN 14: MEMBUAT POINT CLOUD DARI DEPTH MAP")
print("=" * 60)

# ============================================================
# 1. Memuat pasangan gambar stereo real
# ============================================================

# Menampilkan informasi tahap memuat gambar
print("\n--- Memuat Pasangan Gambar Stereo ---")

# Mendefinisikan path gambar stereo
path_left  = os.path.join(IMAGE_DIR, "stereo_left.png")
path_right = os.path.join(IMAGE_DIR, "stereo_right.png")

# Memuat gambar stereo
img_left  = cv2.imread(path_left)
img_right = cv2.imread(path_right)

# Download otomatis jika file tidak tersedia
if img_left is None or img_right is None:
    print("[WARN] Gambar stereo tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    img_left  = cv2.imread(path_left)
    img_right = cv2.imread(path_right)
if img_left is None:
    raise FileNotFoundError("[ERROR] stereo_left.png tidak tersedia. Jalankan: python download_image.py")
if img_right is None:
    raise FileNotFoundError("[ERROR] stereo_right.png tidak tersedia. Jalankan: python download_image.py")

# Mendapatkan dimensi gambar
h, w = img_left.shape[:2]

# Parameter kamera estimasi
focal_length = 700.0
baseline     = 30.0
cx = w / 2.0
cy = h / 2.0

# Menampilkan informasi gambar
print(f"[INFO] Gambar kiri : {img_left.shape}")
print(f"[INFO] Gambar kanan: {img_right.shape}")
print(f"[INFO] Focal length (estimasi): {focal_length:.1f} px")
print(f"[INFO] Principal point: ({cx:.1f}, {cy:.1f})")

# ============================================================
# 2. Menghitung Disparity Map
# ============================================================

# Menampilkan informasi tahap disparity
print("\n--- Menghitung Disparity Map ---")

# Mengkonversi gambar kiri ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kanan ke grayscale
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Mendefinisikan parameter SGBM
num_disp = 64
block_size = 7

# Membuat objek StereoSGBM
sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Menghitung disparity map
disp_raw = sgbm.compute(gray_left, gray_right)

# Mengkonversi ke float (dibagi 16)
disp_float = disp_raw.astype(np.float32) / 16.0

# Membuat mask piksel valid
valid_mask = disp_float > 0

# Menampilkan informasi disparity
print(f"[INFO] Disparity range: [{disp_float.min():.2f}, {disp_float.max():.2f}]")
print(f"[INFO] Piksel valid: {valid_mask.sum()} ({valid_mask.sum()/(h*w)*100:.1f}%)")

# ============================================================
# 3. Konversi Manual ke Point Cloud (Z=f*B/d)
# ============================================================

# Menampilkan informasi tahap konversi manual
print("\n--- Konversi Manual: Z = f*B/d ---")

# Membuat grid koordinat piksel menggunakan meshgrid
u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

# Menyalin disparity dan mengganti nilai invalid
disp_safe = disp_float.copy()

# Mengganti disparity <= 0 dengan nilai kecil untuk menghindari division by zero
disp_safe[disp_safe <= 0] = 0.01

# Menghitung depth (Z) menggunakan rumus Z = f * B / d
Z_manual = (focal_length * baseline) / disp_safe

# Menghitung koordinat X menggunakan rumus X = (u - cx) * Z / f
X_manual = (u_coords.astype(np.float32) - cx) * Z_manual / focal_length

# Menghitung koordinat Y menggunakan rumus Y = (v - cy) * Z / f
Y_manual = (v_coords.astype(np.float32) - cy) * Z_manual / focal_length

# Menampilkan statistik point cloud manual
print(f"[MANUAL] X range: [{X_manual[valid_mask].min():.1f}, {X_manual[valid_mask].max():.1f}]")
print(f"[MANUAL] Y range: [{Y_manual[valid_mask].min():.1f}, {Y_manual[valid_mask].max():.1f}]")
print(f"[MANUAL] Z range: [{Z_manual[valid_mask].min():.1f}, {Z_manual[valid_mask].max():.1f}]")

# ============================================================
# 4. Konversi dengan cv2.reprojectImageTo3D()
# ============================================================

# Menampilkan informasi tahap reprojectImageTo3D
print("\n--- Konversi dengan cv2.reprojectImageTo3D() ---")

# Membuat matriks Q (perspektif reprojection matrix)
Q = np.float64([
    [1, 0, 0, -cx],
    [0, 1, 0, -cy],
    [0, 0, 0, focal_length],
    [0, 0, -1.0 / baseline, 0]
])

# Menampilkan matriks Q
print("[INFO] Matriks Q:")
for row in Q:
    print(f"  [{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f} {row[3]:8.2f}]")

# Menggunakan reprojectImageTo3D untuk konversi
points_3d = cv2.reprojectImageTo3D(disp_float, Q, handleMissingValues=True)

# Mengekstrak komponen X, Y, Z dari point cloud
X_reproj = points_3d[:, :, 0]
Y_reproj = points_3d[:, :, 1]
Z_reproj = points_3d[:, :, 2]

# Menampilkan statistik point cloud reprojectImageTo3D
mask_reproj = (Z_reproj > 0) & (Z_reproj < 10000) & valid_mask
print(f"[REPROJ] X range: [{X_reproj[mask_reproj].min():.1f}, {X_reproj[mask_reproj].max():.1f}]")
print(f"[REPROJ] Y range: [{Y_reproj[mask_reproj].min():.1f}, {Y_reproj[mask_reproj].max():.1f}]")
print(f"[REPROJ] Z range: [{Z_reproj[mask_reproj].min():.1f}, {Z_reproj[mask_reproj].max():.1f}]")

# ============================================================
# 5. Filter Titik Invalid dan Siapkan Data Visualisasi
# ============================================================

# Menampilkan informasi tahap filtering
print("\n--- Filtering Titik Invalid ---")

# Membatasi depth maksimum untuk visualisasi
max_depth = 2000.0

# Membuat mask titik valid (disparity > 0 dan depth dalam batas wajar)
point_mask = valid_mask & (Z_manual > 0) & (Z_manual < max_depth)

# Mengekstrak koordinat titik valid dari konversi manual
x_pts = X_manual[point_mask]
y_pts = Y_manual[point_mask]
z_pts = Z_manual[point_mask]

# Menampilkan jumlah titik valid
print(f"[INFO] Total piksel: {h * w}")
print(f"[INFO] Titik valid: {len(x_pts)}")
print(f"[INFO] Persentase: {len(x_pts) / (h * w) * 100:.1f}%")

# Mengambil warna dari gambar kiri asli (konversi BGR ke RGB, normalisasi 0-1)
img_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

# Mengekstrak warna untuk setiap titik valid
colors = img_rgb[point_mask].astype(np.float32) / 255.0

# Menampilkan informasi warna
print(f"[INFO] Jumlah warna: {colors.shape}")

# Melakukan subsampling jika terlalu banyak titik (untuk performa plot)
max_points = 8000

# Memeriksa apakah subsampling diperlukan
if len(x_pts) > max_points:
    # Membuat indeks acak untuk subsampling
    indices = np.random.choice(len(x_pts), max_points, replace=False)
    # Mengambil subset titik
    x_sub = x_pts[indices]
    y_sub = y_pts[indices]
    z_sub = z_pts[indices]
    colors_sub = colors[indices]
    print(f"[INFO] Subsampling: {len(x_pts)} -> {max_points} titik")
else:
    # Menggunakan semua titik tanpa subsampling
    x_sub = x_pts
    y_sub = y_pts
    z_sub = z_pts
    colors_sub = colors
    print(f"[INFO] Menggunakan semua {len(x_pts)} titik")

# ============================================================
# 6. Visualisasi Point Cloud 3D dari Berbagai Sudut
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Visualisasi Point Cloud 3D ---")

# Mendefinisikan sudut pandang untuk visualisasi (elevation, azimuth)
viewpoints = [
    (30, -60, "Tampak Depan-Kiri"),
    (30, 60, "Tampak Depan-Kanan"),
    (80, -90, "Tampak Atas"),
    (10, 0, "Tampak Samping"),
]

# Membuat figure dengan 4 subplot untuk berbagai sudut pandang
fig1, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={'projection': '3d'})

# Meratakan array axes menjadi 1D untuk iterasi mudah
axes_flat = axes.flatten()

# Membuat visualisasi untuk setiap sudut pandang
for idx, (elev, azim, title) in enumerate(viewpoints):
    # Mengambil subplot saat ini
    ax = axes_flat[idx]
    # Membuat scatter plot 3D dengan warna dari gambar asli
    ax.scatter(x_sub, z_sub, -y_sub, c=colors_sub, s=1, alpha=0.6)
    # Mengatur sudut pandang
    ax.view_init(elev=elev, azim=azim)
    # Mengatur judul subplot
    ax.set_title(title, fontsize=11, fontweight='bold')
    # Mengatur label sumbu
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Z (Depth)", fontsize=8)
    ax.set_zlabel("-Y", fontsize=8)

# Mengatur judul utama figure
plt.suptitle("Point Cloud 3D dari Berbagai Sudut Pandang\n(Konversi Manual: Z=f*B/d)",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
path_multi_view = os.path.join(OUTPUT_DIR, "14_point_cloud_multi_view.png")

# Menyimpan figure multi-view
fig1.savefig(path_multi_view, dpi=150, bbox_inches='tight')
print(f"[SAVE] Multi-view point cloud: {path_multi_view}")

# ============================================================
# 7. Perbandingan Metode Manual vs reprojectImageTo3D
# ============================================================

# Menampilkan informasi tahap perbandingan
print("\n--- Perbandingan Manual vs reprojectImageTo3D ---")

# Mengekstrak titik valid dari reprojectImageTo3D
x_rep = X_reproj[point_mask]
y_rep = Y_reproj[point_mask]
z_rep = Z_reproj[point_mask]

# Melakukan subsampling pada data reprojectImageTo3D
if len(x_rep) > max_points:
    # Menggunakan indeks yang sama untuk konsistensi
    x_rep_sub = x_rep[indices]
    y_rep_sub = y_rep[indices]
    z_rep_sub = z_rep[indices]
else:
    x_rep_sub = x_rep
    y_rep_sub = y_rep
    z_rep_sub = z_rep

# Membuat figure perbandingan dua metode
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': '3d'})

# Menampilkan point cloud dari konversi manual
ax1.scatter(x_sub, z_sub, -y_sub, c=colors_sub, s=1, alpha=0.6)
ax1.view_init(elev=30, azim=-60)
ax1.set_title("Metode Manual\n(Z=f*B/d, X=(u-cx)*Z/f)", fontsize=11, fontweight='bold')
ax1.set_xlabel("X")
ax1.set_ylabel("Z (Depth)")
ax1.set_zlabel("-Y")

# Menampilkan point cloud dari reprojectImageTo3D
ax2.scatter(x_rep_sub, z_rep_sub, -y_rep_sub, c=colors_sub, s=1, alpha=0.6)
ax2.view_init(elev=30, azim=-60)
ax2.set_title("reprojectImageTo3D()\n(Menggunakan Matriks Q)", fontsize=11, fontweight='bold')
ax2.set_xlabel("X")
ax2.set_ylabel("Z (Depth)")
ax2.set_zlabel("-Y")

# Mengatur judul utama
plt.suptitle("Perbandingan Metode Konversi Disparity ke Point Cloud",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output perbandingan
path_comparison = os.path.join(OUTPUT_DIR, "14_point_cloud_perbandingan.png")

# Menyimpan figure perbandingan
fig2.savefig(path_comparison, dpi=150, bbox_inches='tight')
print(f"[SAVE] Perbandingan metode: {path_comparison}")

# ============================================================
# 8. Visualisasi Disparity dan Depth Map
# ============================================================

# Menampilkan informasi tahap visualisasi 2D
print("\n--- Visualisasi Disparity dan Depth Map ---")

# Membuat figure untuk visualisasi 2D
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 9))

# Menampilkan gambar kiri asli
axes3[0, 0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
axes3[0, 0].set_title("Gambar Kiri (Asli)", fontsize=11)
axes3[0, 0].axis('off')

# Menormalisasi disparity untuk visualisasi
disp_vis = cv2.normalize(disp_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Menampilkan disparity map dengan colormap
axes3[0, 1].imshow(disp_vis, cmap='jet')
axes3[0, 1].set_title("Disparity Map (SGBM)", fontsize=11)
axes3[0, 1].axis('off')

# Menampilkan depth map manual
depth_vis = np.clip(Z_manual, 0, max_depth)
im_depth = axes3[1, 0].imshow(depth_vis, cmap='plasma')
axes3[1, 0].set_title("Depth Map (Manual)", fontsize=11)
axes3[1, 0].axis('off')
plt.colorbar(im_depth, ax=axes3[1, 0], fraction=0.046)

# Menampilkan depth map dari reprojectImageTo3D
depth_reproj_vis = np.clip(Z_reproj, 0, max_depth)
im_reproj = axes3[1, 1].imshow(depth_reproj_vis, cmap='plasma')
axes3[1, 1].set_title("Depth Map (reprojectImageTo3D)", fontsize=11)
axes3[1, 1].axis('off')
plt.colorbar(im_reproj, ax=axes3[1, 1], fraction=0.046)

# Mengatur layout
plt.suptitle("Disparity dan Depth Map", fontsize=14, fontweight='bold')
plt.tight_layout()

# Mendefinisikan path output 2D
path_2d = os.path.join(OUTPUT_DIR, "14_depth_dan_disparity.png")

# Menyimpan figure 2D
fig3.savefig(path_2d, dpi=150, bbox_inches='tight')
print(f"[SAVE] Depth dan disparity: {path_2d}")

# Menampilkan semua figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 14: POINT CLOUD DARI DEPTH MAP")
print("=" * 60)
print(f"1. Point cloud berisi {len(x_pts)} titik 3D valid")
print("2. Konversi manual: Z=f*B/d, X=(u-cx)*Z/f, Y=(v-cy)*Z/f")
print("3. reprojectImageTo3D() menggunakan matriks Q 4x4")
print("4. Kedua metode menghasilkan point cloud yang serupa")
print("5. Titik invalid (disparity<=0) harus difilter")
print("6. Warna titik diambil dari gambar asli (RGB)")
print("7. Subsampling diperlukan untuk visualisasi yang efisien")
print("8. Berbagai sudut pandang membantu memahami struktur 3D")
print("9. Depth map plasma colormap menunjukkan jarak objek")
print("10. Point cloud adalah dasar untuk rekonstruksi 3D")
print("=" * 60)
