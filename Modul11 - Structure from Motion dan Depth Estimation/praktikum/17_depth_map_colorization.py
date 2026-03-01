"""
==========================================================================
PERCOBAAN 17: DEPTH MAP COLORIZATION DAN VISUALISASI
==========================================================================
Program ini mempelajari berbagai teknik visualisasi depth map menggunakan
colormap, kontur kedalaman, overlay semi-transparan, dan segmentasi
berbasis depth. Berbagai colormap OpenCV diterapkan untuk menghasilkan
representasi visual depth yang informatif dan mudah diinterpretasi.

Konsep utama:
- Depth map grayscale sulit diinterpretasi tanpa pewarnaan
- Colormap memetakan nilai skalar ke warna untuk visualisasi intuitif
- Kontur kedalaman menunjukkan batas-batas perubahan depth
- Overlay depth pada gambar asli membantu korelasi spasial
- Histogram depth menunjukkan distribusi jarak objek dalam scene
- Segmentasi zona depth membagi scene menjadi near/mid/far

Fungsi utama yang dipelajari:
- cv2.applyColorMap()       : Menerapkan colormap pada gambar grayscale
- cv2.addWeighted()         : Menggabungkan dua gambar dengan bobot
- cv2.threshold()           : Segmentasi berdasarkan nilai threshold
- cv2.StereoSGBM_create()  : Menghitung disparity/depth map
- cv2.normalize()           : Normalisasi nilai gambar ke rentang tertentu
- np.histogram()            : Menghitung distribusi histogram

Hasil: Grid visualisasi depth map dengan berbagai colormap, kontur,
       overlay, histogram distribusi, dan segmentasi zona kedalaman
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
print("PERCOBAAN 17: DEPTH MAP COLORIZATION DAN VISUALISASI")
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

# Menampilkan informasi gambar
print(f"[INFO] Gambar kiri : {img_left.shape}")
print(f"[INFO] Gambar kanan: {img_right.shape}")

# ============================================================
# 2. Menghitung Depth Map menggunakan StereoSGBM
# ============================================================

# Menampilkan informasi tahap komputasi depth
print("\n--- Menghitung Depth Map dengan StereoSGBM ---")

# Mengkonversi gambar kiri ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kanan ke grayscale
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Mendefinisikan parameter stereo matching
num_disp = 80

# Mendefinisikan ukuran blok matching
block_size = 7

# Membuat objek StereoSGBM dengan parameter optimal
sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Menghitung disparity map mentah
disp_raw = sgbm.compute(gray_left, gray_right)

# Mengkonversi disparity ke float (dibagi 16 karena format fixed-point)
disp_float = disp_raw.astype(np.float32) / 16.0

# Membuat mask piksel valid (disparity > 0)
valid_mask = disp_float > 0

# Menormalisasi depth map ke rentang 0-255 untuk visualisasi
depth_norm = np.zeros_like(disp_float, dtype=np.uint8)

# Menerapkan normalisasi hanya pada piksel valid
cv2.normalize(disp_float, depth_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=valid_mask.astype(np.uint8))

# Menampilkan informasi depth map
print(f"[INFO] Disparity range: [{disp_float[valid_mask].min():.2f}, {disp_float[valid_mask].max():.2f}]")
print(f"[INFO] Piksel valid: {valid_mask.sum()} ({valid_mask.sum()/(h*w)*100:.1f}%)")

# ============================================================
# 3. Menerapkan berbagai colormap pada depth map
# ============================================================

# Menampilkan informasi tahap colormap
print("\n--- Menerapkan Berbagai Colormap ---")

# Mendefinisikan daftar colormap beserta nama dan kode OpenCV
colormaps = [
    ("JET", cv2.COLORMAP_JET),
    ("INFERNO", cv2.COLORMAP_INFERNO),
    ("VIRIDIS", cv2.COLORMAP_VIRIDIS),
    ("PLASMA", cv2.COLORMAP_PLASMA),
    ("HOT", cv2.COLORMAP_HOT),
    ("BONE", cv2.COLORMAP_BONE),
]

# Menyiapkan dictionary untuk menyimpan hasil colormap
colormap_results = {}

# Menerapkan setiap colormap pada depth map yang sudah dinormalisasi
for name, cmap_code in colormaps:
    # Menerapkan colormap pada depth map
    colored = cv2.applyColorMap(depth_norm, cmap_code)
    # Mengatur area invalid menjadi hitam
    colored[~valid_mask] = [0, 0, 0]
    # Menyimpan hasil colormap ke dictionary
    colormap_results[name] = colored
    # Menampilkan informasi colormap yang diterapkan
    print(f"[INFO] Colormap {name} diterapkan")

# Membuat figure untuk visualisasi grid colormap
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))

# Mengatur judul utama figure
fig1.suptitle("Depth Map dengan Berbagai Colormap", fontsize=16, fontweight='bold')

# Menampilkan setiap hasil colormap pada subplot
for idx, (name, _) in enumerate(colormaps):
    # Menentukan posisi baris dan kolom pada grid
    row = idx // 3
    col = idx % 3
    # Mengkonversi BGR ke RGB untuk matplotlib
    rgb_img = cv2.cvtColor(colormap_results[name], cv2.COLOR_BGR2RGB)
    # Menampilkan gambar pada subplot yang sesuai
    axes1[row, col].imshow(rgb_img)
    # Mengatur judul subplot
    axes1[row, col].set_title(f"Colormap: {name}", fontsize=12)
    # Menonaktifkan axis
    axes1[row, col].axis('off')

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan figure colormap ke file
output_path1 = os.path.join(OUTPUT_DIR, "17_colormaps_comparison.png")
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"\n[SAVE] Grid colormap disimpan: {output_path1}")

# Menutup figure untuk membebaskan memori
plt.close(fig1)

# ============================================================
# 4. Membuat Visualisasi Kontur Kedalaman
# ============================================================

# Menampilkan informasi tahap kontur kedalaman
print("\n--- Membuat Kontur Kedalaman ---")

# Mendefinisikan jumlah level kontur
num_contour_levels = 8

# Menghitung interval antar level kontur
contour_interval = 256 // num_contour_levels

# Membuat gambar kontur dengan kuantisasi depth
depth_quantized = (depth_norm // contour_interval) * contour_interval

# Mendeteksi tepi kontur menggunakan Canny edge detection
contour_edges = cv2.Canny(depth_quantized, 30, 100)

# Membuat gambar kontur berwarna (garis kontur putih di atas colormap)
contour_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

# Mengatur area invalid menjadi hitam
contour_vis[~valid_mask] = [0, 0, 0]

# Menambahkan garis kontur putih pada visualisasi
contour_vis[contour_edges > 0] = [255, 255, 255]

# Menampilkan informasi kontur
print(f"[INFO] Jumlah level kontur: {num_contour_levels}")
print(f"[INFO] Interval kontur: {contour_interval}")

# ============================================================
# 5. Membuat Overlay Depth pada Gambar Asli
# ============================================================

# Menampilkan informasi tahap overlay
print("\n--- Membuat Overlay Depth pada Gambar Asli ---")

# Menerapkan colormap JET pada depth map
depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

# Mengatur area invalid menjadi hitam pada depth berwarna
depth_colored[~valid_mask] = [0, 0, 0]

# Mendefinisikan bobot transparansi untuk overlay
alpha = 0.6

# Mendefinisikan bobot gambar asli
beta = 0.4

# Membuat overlay semi-transparan depth pada gambar asli
overlay = cv2.addWeighted(img_left, beta, depth_colored, alpha, 0)

# Menampilkan informasi overlay
print(f"[INFO] Alpha (depth): {alpha}, Beta (original): {beta}")

# ============================================================
# 6. Membuat Histogram Distribusi Depth
# ============================================================

# Menampilkan informasi tahap histogram
print("\n--- Membuat Histogram Distribusi Depth ---")

# Mengekstrak nilai depth yang valid saja
valid_depths = disp_float[valid_mask]

# Menghitung histogram distribusi depth
hist_values, bin_edges = np.histogram(valid_depths, bins=50)

# Menghitung bin centers untuk plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Menampilkan statistik distribusi depth
print(f"[INFO] Mean disparity: {valid_depths.mean():.2f}")
print(f"[INFO] Std disparity: {valid_depths.std():.2f}")
print(f"[INFO] Median disparity: {np.median(valid_depths):.2f}")

# ============================================================
# 7. Segmentasi Zona Depth (Near/Mid/Far)
# ============================================================

# Menampilkan informasi tahap segmentasi
print("\n--- Segmentasi Zona Depth ---")

# Menghitung batas-batas zona berdasarkan percentile
p33 = np.percentile(valid_depths, 33)
p66 = np.percentile(valid_depths, 66)

# Menampilkan batas zona
print(f"[INFO] Batas zona: Near > {p66:.2f}, Mid [{p33:.2f}-{p66:.2f}], Far < {p33:.2f}")

# Membuat mask untuk zona dekat (near) - disparity tinggi berarti dekat
near_mask = (disp_float > p66) & valid_mask

# Membuat mask untuk zona tengah (mid)
mid_mask = (disp_float >= p33) & (disp_float <= p66) & valid_mask

# Membuat mask untuk zona jauh (far) - disparity rendah berarti jauh
far_mask = (disp_float < p33) & valid_mask

# Membuat gambar segmentasi zona depth berwarna
zone_vis = np.zeros((h, w, 3), dtype=np.uint8)

# Mewarnai zona dekat dengan warna merah (near = danger/close)
zone_vis[near_mask] = [0, 0, 255]

# Mewarnai zona tengah dengan warna kuning (mid = caution)
zone_vis[mid_mask] = [0, 255, 255]

# Mewarnai zona jauh dengan warna hijau (far = safe/distant)
zone_vis[far_mask] = [0, 255, 0]

# Menghitung persentase area masing-masing zona
near_pct = near_mask.sum() / valid_mask.sum() * 100
mid_pct = mid_mask.sum() / valid_mask.sum() * 100
far_pct = far_mask.sum() / valid_mask.sum() * 100

# Menampilkan statistik zona
print(f"[INFO] Zona Near (merah): {near_pct:.1f}%")
print(f"[INFO] Zona Mid (kuning): {mid_pct:.1f}%")
print(f"[INFO] Zona Far (hijau): {far_pct:.1f}%")

# ============================================================
# 8. Menyimpan semua visualisasi dalam grid gabungan
# ============================================================

# Menampilkan informasi tahap penyimpanan
print("\n--- Menyimpan Visualisasi Gabungan ---")

# Membuat figure besar untuk semua visualisasi
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))

# Mengatur judul utama figure
fig2.suptitle("Depth Map Colorization dan Visualisasi Lengkap", fontsize=16, fontweight='bold')

# Menampilkan gambar asli di subplot pertama
axes2[0, 0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title("Gambar Asli (Kiri)", fontsize=11)
axes2[0, 0].axis('off')

# Menampilkan depth map grayscale di subplot kedua
axes2[0, 1].imshow(depth_norm, cmap='gray')
axes2[0, 1].set_title("Depth Map (Grayscale)", fontsize=11)
axes2[0, 1].axis('off')

# Menampilkan depth map dengan colormap JET
axes2[0, 2].imshow(cv2.cvtColor(colormap_results["JET"], cv2.COLOR_BGR2RGB))
axes2[0, 2].set_title("Depth Map (Colormap JET)", fontsize=11)
axes2[0, 2].axis('off')

# Menampilkan kontur kedalaman
axes2[1, 0].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
axes2[1, 0].set_title("Kontur Kedalaman", fontsize=11)
axes2[1, 0].axis('off')

# Menampilkan overlay depth pada gambar asli
axes2[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes2[1, 1].set_title(f"Overlay Depth (α={alpha})", fontsize=11)
axes2[1, 1].axis('off')

# Menampilkan segmentasi zona depth
axes2[1, 2].imshow(cv2.cvtColor(zone_vis, cv2.COLOR_BGR2RGB))
axes2[1, 2].set_title("Segmentasi Zona (R=Near,Y=Mid,G=Far)", fontsize=11)
axes2[1, 2].axis('off')

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan figure gabungan ke file
output_path2 = os.path.join(OUTPUT_DIR, "17_depth_visualization_all.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVE] Visualisasi gabungan disimpan: {output_path2}")

# Menutup figure untuk membebaskan memori
plt.close(fig2)

# Membuat figure histogram terpisah
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))

# Menampilkan histogram distribusi depth sebagai bar plot
ax3.bar(bin_centers, hist_values, width=(bin_edges[1] - bin_edges[0]) * 0.9,
        color='steelblue', edgecolor='black', alpha=0.8)

# Menambahkan garis vertikal untuk batas zona
ax3.axvline(x=p33, color='green', linestyle='--', linewidth=2, label=f'Far/Mid ({p33:.1f})')
ax3.axvline(x=p66, color='red', linestyle='--', linewidth=2, label=f'Mid/Near ({p66:.1f})')

# Mengatur judul histogram
ax3.set_title("Histogram Distribusi Disparity", fontsize=14, fontweight='bold')

# Mengatur label sumbu X
ax3.set_xlabel("Nilai Disparity", fontsize=12)

# Mengatur label sumbu Y
ax3.set_ylabel("Jumlah Piksel", fontsize=12)

# Menambahkan legend
ax3.legend(fontsize=11)

# Menambahkan grid
ax3.grid(True, alpha=0.3)

# Mengatur layout
plt.tight_layout()

# Menyimpan histogram ke file
output_path3 = os.path.join(OUTPUT_DIR, "17_depth_histogram.png")
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"[SAVE] Histogram depth disimpan: {output_path3}")

# Menutup figure histogram
plt.close(fig3)

# Menyimpan gambar individual menggunakan cv2.imwrite
output_overlay = os.path.join(OUTPUT_DIR, "17_depth_overlay.png")
cv2.imwrite(output_overlay, overlay)
print(f"[SAVE] Overlay disimpan: {output_overlay}")

# Menyimpan segmentasi zona menggunakan cv2.imwrite
output_zone = os.path.join(OUTPUT_DIR, "17_depth_zones.png")
cv2.imwrite(output_zone, zone_vis)
print(f"[SAVE] Segmentasi zona disimpan: {output_zone}")

# Menampilkan ringkasan akhir percobaan
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 17")
print("=" * 60)
print(f"  Ukuran gambar        : {h}x{w}")
print(f"  Sumber gambar        : stereo_left.png")
print(f"  Colormap diterapkan  : {len(colormaps)}")
print(f"  Level kontur         : {num_contour_levels}")
print(f"  Zona Near            : {near_pct:.1f}%")
print(f"  Zona Mid             : {mid_pct:.1f}%")
print(f"  Zona Far             : {far_pct:.1f}%")
print(f"  Mean disparity       : {valid_depths.mean():.2f}")
print(f"  File output tersimpan: 5 file")
print("=" * 60)
