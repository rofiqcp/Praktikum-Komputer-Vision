"""
==========================================================================
PERCOBAAN 11: KONVERSI DISPARITY KE DEPTH MAP
==========================================================================
Program ini mempelajari hubungan matematis antara disparity dan depth
menggunakan rumus Z = f * B / d, serta teknik visualisasi dan analisis
depth map. Juga membandingkan konversi manual dengan fungsi
cv2.reprojectImageTo3D() menggunakan matriks Q.

Konsep utama:
- Depth (Z) = focal_length * baseline / disparity
- Matriks Q (4x4) perspektif reprojection dari stereo rectification
- cv2.reprojectImageTo3D() mengkonversi disparity ke point cloud 3D
- Depth zones (near/mid/far) membantu analisis scene
- Histogram depth menunjukkan distribusi kedalaman objek dalam scene

Fungsi utama yang dipelajari:
- cv2.reprojectImageTo3D()  : Konversi disparity ke koordinat 3D
- cv2.StereoBM_create()     : Menghitung disparity map
- cv2.normalize()           : Normalisasi untuk visualisasi
- np.where(), np.clip()     : Operasi array untuk analisis depth
- cv2.applyColorMap()       : Visualisasi depth dengan colormap

Hasil: Depth map dari konversi disparity, depth zones, dan point cloud
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
print("PERCOBAAN 11: KONVERSI DISPARITY KE DEPTH MAP")
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

# Parameter kamera estimasi (untuk pasangan stereo nyata yang terektifikasi)
focal_length = 700.0  # estimasi focal length dalam piksel
baseline     = 30.0   # estimasi baseline

# Menampilkan informasi gambar
print(f"[INFO] Gambar kiri : {img_left.shape}")
print(f"[INFO] Gambar kanan: {img_right.shape}")
print(f"[INFO] Focal length (estimasi): {focal_length:.1f} px")
print(f"[INFO] Baseline    (estimasi): {baseline:.1f} mm-equiv")

# ============================================================
# 2. Konversi ke grayscale dan hitung disparity
# ============================================================

# Menampilkan informasi tahap disparity
print("\n--- Menghitung Disparity Map ---")

# Mengkonversi gambar ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Membuat objek StereoBM
num_disp = 64
block_size = 15
stereo_bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)

# Menghitung disparity map menggunakan StereoBM
disp_bm = stereo_bm.compute(gray_left, gray_right)

# Mengkonversi disparity ke float (dibagi 16)
disp_float = disp_bm.astype(np.float32) / 16.0

# Menampilkan informasi disparity
print(f"[INFO] numDisparities: {num_disp}, blockSize: {block_size}")
print(f"[INFO] Disparity range: [{disp_float.min():.2f}, {disp_float.max():.2f}]")

# Menghitung jumlah piksel valid
valid_mask = disp_float > 0
print(f"[INFO] Piksel valid: {valid_mask.sum()} / {h * w} "
      f"({valid_mask.sum() / (h * w) * 100:.1f}%)")

# ============================================================
# 3. Konversi disparity ke depth secara manual (Z = f*B/d)
# ============================================================

# Menampilkan informasi tahap konversi manual
print("\n--- Konversi Manual: Z = f * B / d ---")

# Menyalin disparity untuk konversi
disp_safe = disp_float.copy()

# Mengganti nilai <= 0 agar tidak terjadi division by zero
disp_safe[disp_safe <= 0] = 0.1

# Menghitung depth map manual menggunakan rumus Z = f * B / d
depth_manual = (focal_length * baseline) / disp_safe

# Membatasi depth maksimum agar visualisasi lebih baik
max_depth = 3000.0
depth_manual_clipped = np.clip(depth_manual, 0, max_depth)

# Menampilkan statistik depth manual
valid_depth = depth_manual[valid_mask]
if len(valid_depth) > 0:
    print(f"[DEPTH] Min depth: {valid_depth.min():.2f}")
    print(f"[DEPTH] Max depth: {valid_depth.max():.2f}")
    print(f"[DEPTH] Mean depth: {valid_depth.mean():.2f}")
    print(f"[DEPTH] Median depth: {np.median(valid_depth):.2f}")

# ============================================================
# 4. Konversi menggunakan cv2.reprojectImageTo3D() dengan matriks Q
# ============================================================

# Menampilkan informasi tahap reprojectImageTo3D
print("\n--- Konversi dengan cv2.reprojectImageTo3D() ---")

# Mendefinisikan matriks Q (perspektif reprojection matrix)
# Q biasanya dihasilkan oleh cv2.stereoRectify()
# Format standar Q berukuran 4x4:
cx = w / 2.0
cy = h / 2.0

# Membuat matriks Q secara manual
Q = np.float64([
    [1, 0, 0, -cx],
    [0, 1, 0, -cy],
    [0, 0, 0, focal_length],
    [0, 0, -1.0 / baseline, 0]
])

# Menampilkan matriks Q
print("[INFO] Matriks Q (4x4):")
for row in Q:
    print(f"  [{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f} {row[3]:8.2f}]")

# Menggunakan cv2.reprojectImageTo3D() untuk konversi
points_3d = cv2.reprojectImageTo3D(disp_float, Q, handleMissingValues=True)

# Mengekstrak komponen Z (depth) dari point cloud 3D
depth_reproject = points_3d[:, :, 2]

# Mengganti nilai depth tidak valid (sangat besar atau negatif)
depth_reproject[depth_reproject > 10000] = max_depth
depth_reproject[depth_reproject < 0] = max_depth

# Membatasi depth agar visualisasi lebih baik
depth_reproject_clipped = np.clip(depth_reproject, 0, max_depth)

# Menampilkan statistik depth reprojection
valid_reproject = depth_reproject[valid_mask]
valid_reproject = valid_reproject[(valid_reproject > 0) & (valid_reproject < 10000)]
if len(valid_reproject) > 0:
    print(f"[REPROJECT] Min depth: {valid_reproject.min():.2f}")
    print(f"[REPROJECT] Max depth: {valid_reproject.max():.2f}")
    print(f"[REPROJECT] Mean depth: {valid_reproject.mean():.2f}")

# Menampilkan informasi point cloud
print(f"[INFO] Ukuran point cloud: {points_3d.shape}")
print(f"[INFO] Koordinat X range: [{points_3d[:,:,0].min():.1f}, {points_3d[:,:,0].max():.1f}]")
print(f"[INFO] Koordinat Y range: [{points_3d[:,:,1].min():.1f}, {points_3d[:,:,1].max():.1f}]")

# ============================================================
# 5. Analisis depth histogram
# ============================================================

# Menampilkan informasi tahap histogram
print("\n--- Analisis Depth Histogram ---")

# Mengambil depth valid untuk histogram
depth_valid = depth_manual[valid_mask]
depth_valid = depth_valid[depth_valid < max_depth]

# Menampilkan statistik
if len(depth_valid) > 0:
    print(f"[HIST] Jumlah piksel valid: {len(depth_valid)}")
    print(f"[HIST] Depth range: [{depth_valid.min():.1f}, {depth_valid.max():.1f}]")
    print(f"[HIST] Std dev: {depth_valid.std():.2f}")

# ============================================================
# 6. Membuat depth zones (near, mid, far)
# ============================================================

# Menampilkan informasi tahap depth zones
print("\n--- Membuat Depth Zones ---")

# Mendefinisikan batas depth zones
zone_near = 400.0
zone_mid = 800.0

# Menampilkan batas zones
print(f"[INFO] Near zone: depth < {zone_near:.0f}")
print(f"[INFO] Mid zone : {zone_near:.0f} <= depth < {zone_mid:.0f}")
print(f"[INFO] Far zone : depth >= {zone_mid:.0f}")

# Membuat mask untuk setiap zone
mask_near = np.where((depth_manual < zone_near) & valid_mask, 1, 0).astype(np.uint8)
mask_mid = np.where((depth_manual >= zone_near) & (depth_manual < zone_mid) & valid_mask,
                     1, 0).astype(np.uint8)
mask_far = np.where((depth_manual >= zone_mid) & valid_mask, 1, 0).astype(np.uint8)

# Menghitung jumlah piksel per zone
n_near = mask_near.sum()
n_mid = mask_mid.sum()
n_far = mask_far.sum()
n_total = n_near + n_mid + n_far

# Menampilkan statistik zones
print(f"[ZONE] Near: {n_near} piksel ({n_near/(n_total+1e-8)*100:.1f}%)")
print(f"[ZONE] Mid : {n_mid} piksel ({n_mid/(n_total+1e-8)*100:.1f}%)")
print(f"[ZONE] Far : {n_far} piksel ({n_far/(n_total+1e-8)*100:.1f}%)")

# Membuat visualisasi depth zones dengan warna berbeda
zone_vis = np.zeros((h, w, 3), dtype=np.uint8)
zone_vis[mask_near == 1] = [0, 0, 255]    # Merah untuk dekat
zone_vis[mask_mid == 1] = [0, 255, 0]     # Hijau untuk menengah
zone_vis[mask_far == 1] = [255, 0, 0]     # Biru untuk jauh

# Menyimpan visualisasi depth zones
path_zones = os.path.join(OUTPUT_DIR, "11_depth_zones.png")
cv2.imwrite(path_zones, zone_vis)
print(f"[SAVED] Depth zones: {path_zones}")

# ============================================================
# 7. Perbandingan manual vs reprojectImageTo3D
# ============================================================

# Menampilkan informasi tahap perbandingan
print("\n--- Perbandingan Manual vs reprojectImageTo3D ---")

# Menghitung perbedaan absolut antara kedua metode (hanya pada piksel valid)
diff = np.abs(depth_manual_clipped - depth_reproject_clipped)
diff_valid = diff[valid_mask]

# Menampilkan statistik perbedaan
if len(diff_valid) > 0:
    print(f"[DIFF] Mean absolute error: {diff_valid.mean():.4f}")
    print(f"[DIFF] Max absolute error : {diff_valid.max():.4f}")
    print(f"[DIFF] Std dev error      : {diff_valid.std():.4f}")

# ============================================================
# 8. Visualisasi utama
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi Utama ---")

# Membuat figure 2x3
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))

# Menampilkan gambar kiri
axes1[0, 0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
axes1[0, 0].set_title("Gambar Kiri (Input)", fontsize=12)
axes1[0, 0].axis("off")

# Menampilkan gambar kanan
axes1[0, 1].imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
axes1[0, 1].set_title("Gambar Kanan (Input)", fontsize=12)
axes1[0, 1].axis("off")

# Menampilkan disparity map
im_disp = axes1[0, 2].imshow(disp_float, cmap='jet')
axes1[0, 2].set_title("Disparity Map", fontsize=12)
axes1[0, 2].axis("off")
plt.colorbar(im_disp, ax=axes1[0, 2], fraction=0.046, label='Disparity (px)')

# Menampilkan depth manual
im_depth_m = axes1[1, 0].imshow(depth_manual_clipped, cmap='plasma')
axes1[1, 0].set_title("Depth Manual (Z=f*B/d)", fontsize=12)
axes1[1, 0].axis("off")
plt.colorbar(im_depth_m, ax=axes1[1, 0], fraction=0.046, label='Depth')

# Menampilkan depth reprojectImageTo3D
im_depth_r = axes1[1, 1].imshow(depth_reproject_clipped, cmap='plasma')
axes1[1, 1].set_title("reprojectImageTo3D", fontsize=12)
axes1[1, 1].axis("off")
plt.colorbar(im_depth_r, ax=axes1[1, 1], fraction=0.046, label='Depth')

# Menampilkan depth zones
axes1[1, 2].imshow(cv2.cvtColor(zone_vis, cv2.COLOR_BGR2RGB))
axes1[1, 2].set_title("Depth Zones (R=Near, G=Mid, B=Far)", fontsize=11)
axes1[1, 2].axis("off")

# Mengatur judul utama
plt.suptitle("Percobaan 11: Konversi Disparity ke Depth Map",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure utama
output_main = os.path.join(OUTPUT_DIR, "11_disparity_to_depth.png")
plt.savefig(output_main, dpi=150, bbox_inches='tight')
print(f"[SAVED] Visualisasi utama: {output_main}")

# ============================================================
# 9. Visualisasi depth histogram
# ============================================================

# Menampilkan informasi tahap histogram
print("\n--- Membuat Visualisasi Depth Histogram ---")

# Membuat figure untuk histogram
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Menampilkan histogram depth manual
if len(depth_valid) > 0:
    axes2[0].hist(depth_valid, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes2[0].axvline(x=zone_near, color='red', linestyle='--', label=f'Near ({zone_near:.0f})')
    axes2[0].axvline(x=zone_mid, color='green', linestyle='--', label=f'Mid ({zone_mid:.0f})')
    axes2[0].legend()
axes2[0].set_title("Histogram Depth (Manual)", fontsize=12)
axes2[0].set_xlabel("Depth")
axes2[0].set_ylabel("Jumlah Piksel")

# Menampilkan histogram perbandingan error
if len(diff_valid) > 0:
    axes2[1].hist(diff_valid, bins=50, color='coral', alpha=0.7, edgecolor='black')
axes2[1].set_title("Histogram Error (Manual vs Reproject)", fontsize=12)
axes2[1].set_xlabel("Absolute Error")
axes2[1].set_ylabel("Jumlah Piksel")

# Mengatur judul utama
plt.suptitle("Percobaan 11: Analisis Depth Histogram",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure histogram
output_hist = os.path.join(OUTPUT_DIR, "11_depth_histogram.png")
plt.savefig(output_hist, dpi=150, bbox_inches='tight')
print(f"[SAVED] Depth histogram: {output_hist}")

# Menampilkan semua figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 11: KONVERSI DISPARITY KE DEPTH MAP")
print("=" * 60)
print(f"1. Depth dihitung dari disparity: Z = f * B / d")
print(f"2. f = focal length, B = baseline, d = disparity")
print(f"3. Disparity besar -> depth kecil (objek dekat)")
print(f"4. Disparity kecil -> depth besar (objek jauh)")
print(f"5. Matriks Q mengkodekan parameter stereo untuk reprojection")
print(f"6. cv2.reprojectImageTo3D() menghasilkan point cloud 3D")
print(f"7. Depth zones membantu segmentasi scene berdasarkan jarak")
print(f"8. Histogram depth menunjukkan distribusi kedalaman")
print(f"9. Konversi manual dan reprojectImageTo3D harus memberikan")
print(f"   hasil yang serupa jika parameter Q benar")
print("=" * 60)
