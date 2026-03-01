"""
==========================================================================
PERCOBAAN 19: SEGMENTASI OBJEK BERBASIS DEPTH
==========================================================================
Program ini mempelajari cara menggunakan informasi depth untuk melakukan
segmentasi objek berdasarkan jaraknya dari kamera. Teknik yang digunakan
meliputi thresholding, connected components, morphological operations,
dan pewarnaan objek berdasarkan zona kedalaman.

Konsep utama:
- Depth map dapat digunakan sebagai dasar segmentasi objek
- Thresholding pada depth memisahkan foreground dan background
- Connected components mengidentifikasi objek individual
- Morphological operations membersihkan noise pada mask
- Informasi depth memberikan konteks jarak setiap objek
- Kombinasi depth + spatial memberi segmentasi lebih robust

Fungsi utama yang dipelajari:
- cv2.threshold()           : Segmentasi biner berdasarkan threshold
- cv2.inRange()             : Membuat mask untuk rentang nilai tertentu
- cv2.findContours()        : Menemukan kontur objek pada mask
- cv2.connectedComponents() : Mengidentifikasi komponen terhubung
- cv2.morphologyEx()        : Operasi morfologi (opening, closing)
- cv2.drawContours()        : Menggambar kontur pada gambar

Hasil: Visualisasi segmentasi objek berbasis depth, mask per zona,
       statistik area dan rata-rata depth per objek
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
print("PERCOBAAN 19: SEGMENTASI OBJEK BERBASIS DEPTH")
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
print(f"[INFO] Focal length: {focal_length:.1f}, Baseline: {baseline:.1f}")

# ============================================================
# 2. Menghitung Depth Map
# ============================================================

# Menampilkan informasi tahap komputasi depth
print("\n--- Menghitung Depth Map ---")

# Mengkonversi gambar kiri ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kanan ke grayscale
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Mendefinisikan parameter SGBM
num_disp = 80
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
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Menghitung disparity map mentah
disp_raw = sgbm.compute(gray_left, gray_right)

# Mengkonversi ke float (dibagi 16)
disp_float = disp_raw.astype(np.float32) / 16.0

# Membuat mask piksel valid
valid_mask = disp_float > 0

# Menormalisasi disparity ke rentang 0-255
disp_norm = np.zeros_like(disp_float, dtype=np.uint8)
cv2.normalize(disp_float, disp_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=valid_mask.astype(np.uint8))

# Menampilkan informasi depth map
print(f"[INFO] Disparity range: [{disp_float[valid_mask].min():.2f}, {disp_float[valid_mask].max():.2f}]")
print(f"[INFO] Piksel valid: {valid_mask.sum()} ({valid_mask.sum()/(h*w)*100:.1f}%)")

# ============================================================
# 3. Segmentasi Foreground vs Background menggunakan threshold
# ============================================================

# Menampilkan informasi tahap segmentasi FG/BG
print("\n--- Segmentasi Foreground vs Background ---")

# Menghitung threshold otomatis menggunakan Otsu
thresh_val, fg_mask = cv2.threshold(disp_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Menampilkan nilai threshold Otsu
print(f"[INFO] Threshold Otsu: {thresh_val:.1f}")

# Menghitung jumlah piksel foreground
fg_count = np.count_nonzero(fg_mask)

# Menghitung jumlah piksel background
bg_count = h * w - fg_count

# Menampilkan statistik FG/BG
print(f"[INFO] Foreground: {fg_count} piksel ({fg_count/(h*w)*100:.1f}%)")
print(f"[INFO] Background: {bg_count} piksel ({bg_count/(h*w)*100:.1f}%)")

# ============================================================
# 4. Membersihkan mask dengan operasi morfologi
# ============================================================

# Menampilkan informasi tahap morfologi
print("\n--- Membersihkan Mask dengan Morfologi ---")

# Membuat kernel untuk operasi morfologi
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Membuat kernel besar untuk operasi closing
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Menerapkan opening untuk menghilangkan noise kecil
fg_opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)

# Menerapkan closing untuk mengisi lubang kecil
fg_cleaned = cv2.morphologyEx(fg_opened, cv2.MORPH_CLOSE, kernel_large, iterations=2)

# Menghitung perubahan setelah morphology
changed_pixels = np.count_nonzero(fg_mask != fg_cleaned)
print(f"[INFO] Piksel berubah setelah morfologi: {changed_pixels}")

# ============================================================
# 5. Identifikasi objek individual dengan Connected Components
# ============================================================

# Menampilkan informasi tahap connected components
print("\n--- Identifikasi Objek dengan Connected Components ---")

# Menerapkan connected components labeling
num_labels, labels = cv2.connectedComponents(fg_cleaned)

# Menghitung jumlah objek (dikurangi 1 untuk background label=0)
num_objects = num_labels - 1

# Menampilkan jumlah objek yang ditemukan
print(f"[INFO] Jumlah komponen ditemukan: {num_objects}")

# Membuat gambar berwarna untuk setiap komponen
component_vis = np.zeros((h, w, 3), dtype=np.uint8)

# Mendefinisikan palet warna untuk komponen
palette = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 255), (255, 128, 0), (0, 128, 255),
    (128, 255, 0), (255, 0, 128), (0, 255, 128),
]

# Mewarnai setiap komponen dengan warna berbeda
for label_id in range(1, num_labels):
    # Memilih warna dari palet (berulang jika perlu)
    color = palette[(label_id - 1) % len(palette)]
    # Mewarnai piksel komponen
    component_vis[labels == label_id] = color

# ============================================================
# 6. Membuat mask berdasarkan rentang depth (inRange)
# ============================================================

# Menampilkan informasi tahap depth range mask
print("\n--- Membuat Mask Berdasarkan Rentang Depth ---")

# Mendefinisikan batas zona kedalaman pada disparity yang dinormalisasi
near_low, near_high = 160, 255
mid_low, mid_high = 80, 159
far_low, far_high = 1, 79

# Membuat mask zona near menggunakan cv2.inRange
near_mask = cv2.inRange(disp_norm, near_low, near_high)

# Membuat mask zona mid menggunakan cv2.inRange
mid_mask = cv2.inRange(disp_norm, mid_low, mid_high)

# Membuat mask zona far menggunakan cv2.inRange
far_mask = cv2.inRange(disp_norm, far_low, far_high)

# Membersihkan mask zona dengan morfologi opening
near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_OPEN, kernel_small)
mid_mask = cv2.morphologyEx(mid_mask, cv2.MORPH_OPEN, kernel_small)
far_mask = cv2.morphologyEx(far_mask, cv2.MORPH_OPEN, kernel_small)

# Menampilkan statistik zona
print(f"[NEAR] Range [{near_low}-{near_high}]: {np.count_nonzero(near_mask)} piksel")
print(f"[MID]  Range [{mid_low}-{mid_high}]: {np.count_nonzero(mid_mask)} piksel")
print(f"[FAR]  Range [{far_low}-{far_high}]: {np.count_nonzero(far_mask)} piksel")

# Membuat visualisasi zona depth gabungan
zone_vis = np.zeros((h, w, 3), dtype=np.uint8)

# Mewarnai zona near dengan merah
zone_vis[near_mask > 0] = [0, 0, 255]

# Mewarnai zona mid dengan kuning
zone_vis[mid_mask > 0] = [0, 255, 255]

# Mewarnai zona far dengan hijau
zone_vis[far_mask > 0] = [0, 255, 0]

# ============================================================
# 7. Menemukan kontur dan menghitung statistik per objek
# ============================================================

# Menampilkan informasi tahap analisis kontur
print("\n--- Analisis Kontur dan Statistik Per Objek ---")

# Menemukan kontur pada mask foreground yang sudah dibersihkan
contours, _ = cv2.findContours(fg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Menampilkan jumlah kontur ditemukan
print(f"[INFO] Jumlah kontur ditemukan: {len(contours)}")

# Membuat salinan gambar asli untuk menggambar kontur
contour_vis = img_left.copy()

# Menyiapkan list untuk statistik objek
obj_stats = []

# Mendefinisikan area minimum untuk menyaring noise
min_area = 200

# Iterasi untuk setiap kontur
obj_id = 0
for cnt in contours:
    # Menghitung luas area kontur
    area = cv2.contourArea(cnt)

    # Memfilter kontur yang terlalu kecil
    if area < min_area:
        continue

    # Increment ID objek
    obj_id += 1

    # Menghitung bounding rectangle
    x, y, bw, bh = cv2.boundingRect(cnt)

    # Membuat mask untuk kontur ini saja
    cnt_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

    # Menghitung rata-rata disparity di area kontur
    avg_disp = np.mean(disp_float[(cnt_mask > 0) & valid_mask]) if np.any((cnt_mask > 0) & valid_mask) else 0

    # Menentukan zona berdasarkan rata-rata disparity dinormalisasi
    avg_norm = np.mean(disp_norm[(cnt_mask > 0) & valid_mask]) if np.any((cnt_mask > 0) & valid_mask) else 0

    # Mengklasifikasikan zona
    if avg_norm >= near_low:
        zone = "Near"
    elif avg_norm >= mid_low:
        zone = "Mid"
    else:
        zone = "Far"

    # Memilih warna berdasarkan zona
    color = palette[(obj_id - 1) % len(palette)]

    # Menggambar kontur pada gambar
    cv2.drawContours(contour_vis, [cnt], -1, color, 2)

    # Menggambar bounding box
    cv2.rectangle(contour_vis, (x, y), (x + bw, y + bh), color, 1)

    # Menulis label ID objek
    cv2.putText(contour_vis, f"#{obj_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Menyimpan statistik objek
    obj_stats.append({
        "id": obj_id,
        "area": area,
        "avg_disp": avg_disp,
        "zone": zone,
        "bbox": (x, y, bw, bh)
    })

    # Menampilkan statistik objek
    print(f"  Objek #{obj_id}: area={area:.0f}px, avg_disp={avg_disp:.2f}, zona={zone}")

# ============================================================
# 8. Menyimpan semua visualisasi dalam grid
# ============================================================

# Menampilkan informasi tahap penyimpanan
print("\n--- Menyimpan Visualisasi ---")

# Membuat figure besar untuk semua hasil
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Mengatur judul utama
fig.suptitle("Segmentasi Objek Berbasis Depth", fontsize=16, fontweight='bold')

# Menampilkan gambar asli
axes[0, 0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli", fontsize=10)
axes[0, 0].axis('off')

# Menampilkan depth map dengan colormap
depth_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
depth_colored[~valid_mask] = [0, 0, 0]
axes[0, 1].imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Depth Map (JET)", fontsize=10)
axes[0, 1].axis('off')

# Menampilkan mask foreground sebelum morfologi
axes[0, 2].imshow(fg_mask, cmap='gray')
axes[0, 2].set_title(f"FG Mask (Otsu={thresh_val:.0f})", fontsize=10)
axes[0, 2].axis('off')

# Menampilkan mask foreground setelah morfologi
axes[0, 3].imshow(fg_cleaned, cmap='gray')
axes[0, 3].set_title("FG Mask (Cleaned)", fontsize=10)
axes[0, 3].axis('off')

# Menampilkan connected components
axes[1, 0].imshow(cv2.cvtColor(component_vis, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f"Connected Components ({num_objects})", fontsize=10)
axes[1, 0].axis('off')

# Menampilkan zona depth
axes[1, 1].imshow(cv2.cvtColor(zone_vis, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Zona Depth (R=Near,Y=Mid,G=Far)", fontsize=10)
axes[1, 1].axis('off')

# Menampilkan kontur objek pada gambar asli
axes[1, 2].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f"Kontur Objek ({len(obj_stats)})", fontsize=10)
axes[1, 2].axis('off')

# Membuat subplot untuk tabel statistik objek
axes[1, 3].axis('off')
if obj_stats:
    # Mendefinisikan header kolom tabel
    col_labels = ["ID", "Area", "Avg Disp", "Zona"]
    # Menyiapkan data baris tabel
    table_data = [[s["id"], f"{s['area']:.0f}", f"{s['avg_disp']:.1f}", s["zone"]] for s in obj_stats]
    # Membuat tabel pada subplot
    tbl = axes[1, 3].table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    # Mengatur ukuran font tabel
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    # Mengatur skala tabel
    tbl.scale(1.0, 1.4)
# Mengatur judul subplot tabel
axes[1, 3].set_title("Statistik Objek", fontsize=10)

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan figure utama ke file
output_path = os.path.join(OUTPUT_DIR, "19_depth_segmentation_all.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[SAVE] Visualisasi gabungan disimpan: {output_path}")

# Menutup figure
plt.close(fig)

# Menyimpan kontur objek menggunakan cv2.imwrite
output_contour = os.path.join(OUTPUT_DIR, "19_object_contours.png")
cv2.imwrite(output_contour, contour_vis)
print(f"[SAVE] Kontur objek disimpan: {output_contour}")

# Menyimpan zona depth menggunakan cv2.imwrite
output_zone = os.path.join(OUTPUT_DIR, "19_depth_zones.png")
cv2.imwrite(output_zone, zone_vis)
print(f"[SAVE] Zona depth disimpan: {output_zone}")

# Menampilkan ringkasan akhir percobaan
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 19")
print("=" * 60)
print(f"  Ukuran gambar         : {h}x{w}")
print(f"  Sumber gambar         : stereo_left.png")
print(f"  Objek terdeteksi      : {len(obj_stats)}")
print(f"  Connected components  : {num_objects}")
print(f"  Threshold Otsu        : {thresh_val:.1f}")
print(f"  File output tersimpan : 3 file")
print("=" * 60)
