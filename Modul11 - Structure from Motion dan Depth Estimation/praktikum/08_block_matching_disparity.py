"""
==========================================================================
PERCOBAAN 8: BLOCK MATCHING DISPARITY
==========================================================================
Program ini mempelajari cara menghitung disparity map menggunakan
algoritma StereoBM (Block Matching). Disparity map menunjukkan
perbedaan posisi horizontal piksel antara gambar kiri dan kanan,
yang berbanding terbalik dengan kedalaman objek.

Konsep utama:
- Disparity (d) = x_left - x_right, semakin besar d semakin dekat objek
- Depth (Z) = focal_length * baseline / disparity
- StereoBM mencari blok piksel yang cocok secara horizontal
- numDisparities menentukan rentang pencarian (kelipatan 16)
- blockSize menentukan ukuran jendela matching (harus ganjil)

Fungsi utama yang dipelajari:
- cv2.StereoBM_create()     : Membuat objek StereoBM
- stereo.compute()           : Menghitung disparity map
- cv2.normalize()            : Normalisasi disparity untuk visualisasi
- cv2.applyColorMap()        : Menerapkan colormap (JET) pada disparity

Hasil: Disparity map dan estimasi depth map dari pasangan stereo
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan aljabar linier
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
print("PERCOBAAN 8: BLOCK MATCHING DISPARITY")
print("=" * 60)

# ============================================================
# 1. Memuat pasangan gambar stereo
# ============================================================

# Menampilkan informasi tahap memuat gambar
print("\n--- Memuat Pasangan Gambar Stereo ---")

# Mendefinisikan path gambar kiri
path_left = os.path.join(IMAGE_DIR, "stereo_left.png")

# Mendefinisikan path gambar kanan
path_right = os.path.join(IMAGE_DIR, "stereo_right.png")

# Membaca gambar kiri dalam format BGR
img_left = cv2.imread(path_left)

# Membaca gambar kanan dalam format BGR
img_right = cv2.imread(path_right)

# Memeriksa apakah gambar berhasil dimuat
if img_left is None or img_right is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi dimensi gambar
print(f"[INFO] Ukuran gambar kiri : {img_left.shape}")
print(f"[INFO] Ukuran gambar kanan: {img_right.shape}")

# ============================================================
# 2. Konversi ke grayscale
# ============================================================

# Menampilkan informasi tahap konversi
print("\n--- Konversi ke Grayscale ---")

# Mengkonversi gambar kiri ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kanan ke grayscale
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Menampilkan informasi hasil konversi
print(f"[INFO] Gambar kiri grayscale : {gray_left.shape}")
print(f"[INFO] Gambar kanan grayscale: {gray_right.shape}")

# Mendapatkan ukuran gambar
h, w = gray_left.shape

# ============================================================
# 3. Menghitung disparity dengan parameter default
# ============================================================

# Menampilkan informasi tahap disparity default
print("\n--- Disparity dengan Parameter Default ---")

# Mendefinisikan parameter default
default_num_disp = 64
default_block_size = 15

# Membuat objek StereoBM dengan parameter default
stereo_default = cv2.StereoBM_create(
    numDisparities=default_num_disp,
    blockSize=default_block_size
)

# Menghitung disparity map menggunakan StereoBM
disparity_default = stereo_default.compute(gray_left, gray_right)

# Mengkonversi disparity dari fixed-point ke float (dibagi 16)
disparity_float = disparity_default.astype(np.float32) / 16.0

# Menampilkan informasi disparity
print(f"[INFO] numDisparities: {default_num_disp}")
print(f"[INFO] blockSize: {default_block_size}")
print(f"[INFO] Disparity min: {disparity_float.min():.2f}")
print(f"[INFO] Disparity max: {disparity_float.max():.2f}")
print(f"[INFO] Disparity mean: {disparity_float[disparity_float > 0].mean():.2f}")

# Menormalisasi disparity ke range 0-255 untuk visualisasi
disparity_vis = cv2.normalize(disparity_default, None, 0, 255, cv2.NORM_MINMAX)

# Mengkonversi ke uint8
disparity_vis = disparity_vis.astype(np.uint8)

# Menerapkan colormap JET pada disparity
disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)

# Menyimpan disparity map default
path_disp_default = os.path.join(OUTPUT_DIR, "08_disparity_default.png")
cv2.imwrite(path_disp_default, disparity_color)
print(f"[SAVED] Disparity default: {path_disp_default}")

# ============================================================
# 4. Variasi numDisparities
# ============================================================

# Menampilkan informasi tahap variasi numDisparities
print("\n--- Variasi numDisparities ---")

# Mendefinisikan nilai numDisparities yang akan diuji
num_disp_values = [16, 32, 64, 128]

# Menyiapkan list untuk menyimpan hasil setiap variasi
disp_results_nd = []

# Iterasi setiap nilai numDisparities
for nd in num_disp_values:
    # Membuat objek StereoBM dengan numDisparities tertentu
    stereo_nd = cv2.StereoBM_create(numDisparities=nd, blockSize=default_block_size)

    # Menghitung disparity map
    disp_nd = stereo_nd.compute(gray_left, gray_right)

    # Mengkonversi ke float
    disp_nd_float = disp_nd.astype(np.float32) / 16.0

    # Menghitung statistik disparity (hanya yang valid > 0)
    valid_mask = disp_nd_float > 0
    if valid_mask.any():
        mean_val = disp_nd_float[valid_mask].mean()
        max_val = disp_nd_float[valid_mask].max()
    else:
        mean_val = 0
        max_val = 0

    # Menampilkan informasi untuk setiap variasi
    print(f"  numDisp={nd:3d}: mean={mean_val:.2f}, max={max_val:.2f}, "
          f"valid pixels={valid_mask.sum()}")

    # Menormalisasi untuk visualisasi
    disp_vis = cv2.normalize(disp_nd, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Menerapkan colormap JET
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # Menyimpan ke list
    disp_results_nd.append((nd, disp_color, disp_nd_float))

# ============================================================
# 5. Variasi blockSize
# ============================================================

# Menampilkan informasi tahap variasi blockSize
print("\n--- Variasi blockSize ---")

# Mendefinisikan nilai blockSize yang akan diuji (harus ganjil)
block_size_values = [5, 9, 15, 21]

# Menyiapkan list untuk menyimpan hasil setiap variasi
disp_results_bs = []

# Iterasi setiap nilai blockSize
for bs in block_size_values:
    # Membuat objek StereoBM dengan blockSize tertentu
    stereo_bs = cv2.StereoBM_create(numDisparities=default_num_disp, blockSize=bs)

    # Menghitung disparity map
    disp_bs = stereo_bs.compute(gray_left, gray_right)

    # Mengkonversi ke float
    disp_bs_float = disp_bs.astype(np.float32) / 16.0

    # Menghitung statistik disparity (hanya yang valid > 0)
    valid_mask = disp_bs_float > 0
    if valid_mask.any():
        mean_val = disp_bs_float[valid_mask].mean()
        max_val = disp_bs_float[valid_mask].max()
    else:
        mean_val = 0
        max_val = 0

    # Menampilkan informasi untuk setiap variasi
    print(f"  blockSize={bs:2d}: mean={mean_val:.2f}, max={max_val:.2f}, "
          f"valid pixels={valid_mask.sum()}")

    # Menormalisasi untuk visualisasi
    disp_vis = cv2.normalize(disp_bs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Menerapkan colormap JET
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # Menyimpan ke list
    disp_results_bs.append((bs, disp_color, disp_bs_float))

# ============================================================
# 6. Konversi disparity ke depth map
# ============================================================

# Menampilkan informasi tahap konversi depth
print("\n--- Konversi Disparity ke Depth ---")

# Mendefinisikan focal length (estimasi berdasarkan lebar gambar)
focal_length = w * 1.0

# Mendefinisikan baseline (jarak antar kamera, dalam satuan piksel-ekuivalen)
baseline = 30.0

# Menampilkan parameter untuk konversi
print(f"[INFO] Focal length: {focal_length:.1f}")
print(f"[INFO] Baseline: {baseline:.1f}")
print(f"[INFO] Rumus: Z = f * b / d")

# Menggunakan disparity default untuk konversi ke depth
disp_for_depth = disparity_float.copy()

# Mengganti nilai 0 dan negatif dengan nilai kecil agar tidak terjadi division by zero
disp_for_depth[disp_for_depth <= 0] = 0.1

# Menghitung depth map: Z = focal_length * baseline / disparity
depth_map = (focal_length * baseline) / disp_for_depth

# Membatasi depth maksimum agar visualisasi lebih baik
max_depth = 5000.0
depth_map[depth_map > max_depth] = max_depth

# Menampilkan statistik depth
valid_depth = depth_map[disparity_float > 0]
if len(valid_depth) > 0:
    print(f"[DEPTH] Min depth: {valid_depth.min():.2f}")
    print(f"[DEPTH] Max depth: {valid_depth.max():.2f}")
    print(f"[DEPTH] Mean depth: {valid_depth.mean():.2f}")

# Menormalisasi depth map untuk visualisasi
depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Menerapkan colormap pada depth map (invert agar dekat=merah, jauh=biru)
depth_color = cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_JET)

# Menyimpan depth map
path_depth = os.path.join(OUTPUT_DIR, "08_depth_map.png")
cv2.imwrite(path_depth, depth_color)
print(f"[SAVED] Depth map: {path_depth}")

# ============================================================
# 7. Visualisasi perbandingan numDisparities
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi Perbandingan numDisparities ---")

# Membuat figure 2x2 untuk perbandingan numDisparities
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

# Meratakan axes untuk iterasi mudah
axes1_flat = axes1.flatten()

# Menampilkan setiap variasi numDisparities
for i, (nd, disp_color, disp_float) in enumerate(disp_results_nd):
    # Mengkonversi dari BGR ke RGB
    disp_rgb = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)

    # Menampilkan pada subplot
    axes1_flat[i].imshow(disp_rgb)
    axes1_flat[i].set_title(f"numDisparities = {nd}", fontsize=12)
    axes1_flat[i].axis("off")

# Mengatur judul utama
plt.suptitle("Percobaan 8: Variasi numDisparities (blockSize=15)",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure perbandingan numDisparities
output_nd = os.path.join(OUTPUT_DIR, "08_variasi_numDisparities.png")
plt.savefig(output_nd, dpi=150, bbox_inches='tight')
print(f"[SAVED] Perbandingan numDisparities: {output_nd}")

# ============================================================
# 8. Visualisasi perbandingan blockSize
# ============================================================

# Menampilkan informasi tahap visualisasi blockSize
print("\n--- Membuat Visualisasi Perbandingan blockSize ---")

# Membuat figure 2x2 untuk perbandingan blockSize
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Meratakan axes untuk iterasi mudah
axes2_flat = axes2.flatten()

# Menampilkan setiap variasi blockSize
for i, (bs, disp_color, disp_float) in enumerate(disp_results_bs):
    # Mengkonversi dari BGR ke RGB
    disp_rgb = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)

    # Menampilkan pada subplot
    axes2_flat[i].imshow(disp_rgb)
    axes2_flat[i].set_title(f"blockSize = {bs}", fontsize=12)
    axes2_flat[i].axis("off")

# Mengatur judul utama
plt.suptitle("Percobaan 8: Variasi blockSize (numDisparities=64)",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure perbandingan blockSize
output_bs = os.path.join(OUTPUT_DIR, "08_variasi_blockSize.png")
plt.savefig(output_bs, dpi=150, bbox_inches='tight')
print(f"[SAVED] Perbandingan blockSize: {output_bs}")

# ============================================================
# 9. Visualisasi gabungan: input, disparity, dan depth
# ============================================================

# Menampilkan informasi tahap visualisasi gabungan
print("\n--- Membuat Visualisasi Gabungan ---")

# Mengkonversi gambar input ke RGB
img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

# Mengkonversi disparity dan depth colormap ke RGB
disp_default_rgb = cv2.cvtColor(
    cv2.applyColorMap(
        cv2.normalize(disparity_default, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET
    ),
    cv2.COLOR_BGR2RGB
)
depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

# Membuat figure dengan 2 baris dan 2 kolom
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

# Menampilkan gambar kiri original
axes3[0, 0].imshow(img_left_rgb)
axes3[0, 0].set_title("Gambar Kiri (Input)", fontsize=12)
axes3[0, 0].axis("off")

# Menampilkan gambar kanan original
axes3[0, 1].imshow(img_right_rgb)
axes3[0, 1].set_title("Gambar Kanan (Input)", fontsize=12)
axes3[0, 1].axis("off")

# Menampilkan disparity map dengan colormap JET
im_disp = axes3[1, 0].imshow(disparity_float, cmap='jet')
axes3[1, 0].set_title(f"Disparity Map (numDisp={default_num_disp}, block={default_block_size})",
                       fontsize=11)
axes3[1, 0].axis("off")
plt.colorbar(im_disp, ax=axes3[1, 0], fraction=0.046, label='Disparity (pixel)')

# Menampilkan depth map
im_depth = axes3[1, 1].imshow(depth_map, cmap='plasma')
axes3[1, 1].set_title("Depth Map (Z = f*b/d)", fontsize=12)
axes3[1, 1].axis("off")
plt.colorbar(im_depth, ax=axes3[1, 1], fraction=0.046, label='Depth')

# Mengatur judul utama
plt.suptitle("Percobaan 8: Block Matching Disparity dan Depth Estimation",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure gabungan
output_combined = os.path.join(OUTPUT_DIR, "08_block_matching_disparity.png")
plt.savefig(output_combined, dpi=150, bbox_inches='tight')
print(f"\n[SAVED] Hasil gabungan: {output_combined}")

# Menampilkan semua figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 8: BLOCK MATCHING DISPARITY")
print("=" * 60)
print(f"1. StereoBM menghitung disparity dengan block matching")
print(f"2. cv2.StereoBM_create(numDisparities, blockSize)")
print(f"3. Disparity output dalam fixed-point (dibagi 16 untuk float)")
print(f"4. numDisparities: rentang pencarian (kelipatan 16)")
print(f"   - Nilai kecil (16): cepat tapi rentang terbatas")
print(f"   - Nilai besar (128): lambat tapi menangkap disparitas besar")
print(f"5. blockSize: ukuran jendela matching (harus ganjil)")
print(f"   - Nilai kecil (5): detail tinggi tapi lebih noisy")
print(f"   - Nilai besar (21): lebih halus tapi kehilangan detail")
print(f"6. Depth dihitung dari: Z = focal_length * baseline / disparity")
print(f"7. Disparity besar = objek dekat, disparity kecil = objek jauh")
print(f"8. Colormap JET membantu visualisasi distribusi kedalaman")
print("=" * 60)
