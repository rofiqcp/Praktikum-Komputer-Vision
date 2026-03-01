"""
==========================================================
PERCOBAAN 18: FORWARD VS INVERSE WARPING DETAIL
Mempelajari perbedaan forward warping dan inverse warping
secara mendalam, termasuk masalah holes, splatting, dan
interpolasi.

Fungsi utama:
- cv2.remap()
- numpy array indexing
- cv2.inpaint()
==========================================================
"""

# Mengimpor library numpy untuk operasi numerik
import numpy as np

# Mengimpor library os untuk operasi sistem file
import os

# Mengimpor library OpenCV untuk pemrosesan citra
import cv2

# Mengimpor library matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mencoba mengimpor Open3D untuk visualisasi tambahan
try:
    # Mengimpor library Open3D
    import open3d as o3d
    # Menandai ketersediaan Open3D
    HAS_OPEN3D = True
    print("[INFO] Open3D berhasil diimpor")
except ImportError:
    # Menandai bahwa Open3D tidak tersedia
    HAS_OPEN3D = False
    print("[INFO] Open3D tidak tersedia, menggunakan cv2/numpy")

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
print("PERCOBAAN 18: FORWARD VS INVERSE WARPING DETAIL")
print("=" * 60)

# ========================================================
# BAGIAN 1: MEMBUAT TEST IMAGE DENGAN DEPTH MAP
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 1: Membuat Test Image dengan Depth Map")
print("=" * 60)

# Menentukan ukuran gambar
tinggi, lebar = 256, 256

# Mencoba memuat gambar dari direktori atau membuat checkerboard
path_gambar = os.path.join(IMAGE_DIR, "chess.png")

# Download otomatis jika chess.png tidak tersedia
if not os.path.exists(path_gambar):
    print(f"  [WARN] chess.png tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
if not os.path.exists(path_gambar):
    raise FileNotFoundError(
        "[ERROR] chess.png tidak tersedia.\n"
        "  Jalankan: python download_image.py"
    )

# Memuat gambar dari file
gambar_src = cv2.imread(path_gambar)
gambar_src = cv2.resize(gambar_src, (lebar, tinggi))
print(f"Gambar dimuat dari: {path_gambar}")

# Mengestimasi depth map dari gambar asli menggunakan edge strength
# (area dengan banyak detail/edges = dekat, area uniform = jauh)
_gray_src = cv2.cvtColor(gambar_src, cv2.COLOR_BGR2GRAY)
_lap_src  = np.abs(cv2.Laplacian(_gray_src.astype(np.float32), cv2.CV_32F))
_lap_blur = cv2.GaussianBlur(_lap_src, (31, 31), 0)
_lap_max  = _lap_blur.max() if _lap_blur.max() > 0 else 1
depth_map = (1.0 - (_lap_blur / _lap_max) * 0.8).astype(np.float32)  # 0=dekat, 1=jauh

# Mencetak informasi gambar dan depth map
print(f"Ukuran gambar: {gambar_src.shape}")
print(f"Rentang depth (estimasi): [{depth_map.min():.2f}, {depth_map.max():.2f}]")

# ========================================================
# BAGIAN 2: DEFINISI TRANSFORMASI KAMERA
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 2: Definisi Transformasi Kamera")
print("=" * 60)

# Mendefinisikan translasi kamera (baseline) dalam piksel
translasi_x = 15.0  # Pergeseran horizontal

# Menentukan focal length virtual
focal_length = 200.0

# Menghitung disparitas dari depth map
# Disparitas = baseline * focal_length / depth
def hitung_disparitas(depth, baseline, focal):
    """Menghitung disparitas dari depth map."""
    # Menghindari pembagian dengan nol
    depth_safe = np.maximum(depth, 0.01)
    # Menghitung disparitas
    disparitas = baseline * focal / (depth_safe * lebar)
    return disparitas

# Menghitung disparitas untuk translasi yang diberikan
disparitas = hitung_disparitas(depth_map, translasi_x, focal_length)

# Mencetak informasi transformasi
print(f"Translasi kamera  : {translasi_x} piksel")
print(f"Focal length      : {focal_length}")
print(f"Rentang disparitas: [{disparitas.min():.2f}, {disparitas.max():.2f}]")

# ========================================================
# BAGIAN 3: FORWARD WARPING
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 3: Forward Warping")
print("=" * 60)

def forward_warp(gambar, disparitas):
    """Melakukan forward warping: map setiap piksel sumber ke target."""
    # Mendapatkan dimensi gambar
    h, w = gambar.shape[:2]

    # Membuat gambar target kosong
    target = np.zeros_like(gambar)

    # Membuat mask untuk mendeteksi holes
    mask_filled = np.zeros((h, w), dtype=np.uint8)

    # Membuat z-buffer untuk menangani oklusi
    z_buffer = np.full((h, w), np.inf)

    # Iterasi setiap piksel sumber
    for y in range(h):
        for x in range(w):
            # Menghitung posisi target berdasarkan disparitas
            x_target = int(round(x + disparitas[y, x]))

            # Memeriksa apakah target valid
            if 0 <= x_target < w:
                # Mengambil nilai depth sebagai z
                z_val = 1.0 / max(disparitas[y, x], 0.001)

                # Memperbarui jika lebih dekat (z-buffer test)
                if z_val < z_buffer[y, x_target]:
                    target[y, x_target] = gambar[y, x]
                    z_buffer[y, x_target] = z_val
                    mask_filled[y, x_target] = 255

    return target, mask_filled

# Melakukan forward warping
gambar_forward, mask_forward = forward_warp(gambar_src, disparitas)

# Menghitung jumlah holes (piksel kosong)
total_piksel = tinggi * lebar
piksel_terisi = np.count_nonzero(mask_forward)
piksel_kosong = total_piksel - piksel_terisi

# Mencetak statistik forward warping
print(f"Piksel terisi : {piksel_terisi} ({100*piksel_terisi/total_piksel:.1f}%)")
print(f"Piksel holes  : {piksel_kosong} ({100*piksel_kosong/total_piksel:.1f}%)")

# ========================================================
# BAGIAN 4: MASALAH HOLES PADA FORWARD WARPING
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 4: Masalah Holes pada Forward Warping")
print("=" * 60)

# Membuat visualisasi holes
gambar_holes = gambar_forward.copy()

# Menandai holes dengan warna merah terang
mask_holes = mask_forward == 0
gambar_holes[mask_holes] = [0, 0, 255]  # BGR: Merah

# Mencetak jumlah piksel holes
print(f"Jumlah piksel holes: {np.count_nonzero(mask_holes)}")
print("Holes ditandai dengan warna merah pada visualisasi")

# ========================================================
# BAGIAN 5: SPLATTING (MENGATASI HOLES)
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 5: Splatting untuk Mengatasi Holes")
print("=" * 60)

def forward_warp_splatting(gambar, disparitas, splat_size=2):
    """Forward warping dengan splatting: mengisi area lebih besar."""
    # Mendapatkan dimensi gambar
    h, w = gambar.shape[:2]

    # Membuat gambar target kosong
    target = np.zeros_like(gambar, dtype=np.float64)

    # Membuat weight map untuk normalisasi
    weight_map = np.zeros((h, w), dtype=np.float64)

    # Iterasi setiap piksel sumber
    for y in range(h):
        for x in range(w):
            # Menghitung posisi target
            x_target = x + disparitas[y, x]

            # Menghitung batas splat
            x_min = int(max(0, np.floor(x_target) - splat_size // 2))
            x_max = int(min(w, np.ceil(x_target) + splat_size // 2 + 1))

            # Menyebarkan warna piksel ke area splat
            for xt in range(x_min, x_max):
                # Menghitung weight berdasarkan jarak
                dist = abs(xt - x_target)
                weight = max(0, 1.0 - dist / (splat_size / 2 + 0.5))

                # Menambahkan kontribusi piksel dengan weight
                target[y, xt] += gambar[y, x].astype(np.float64) * weight
                weight_map[y, xt] += weight

    # Menormalisasi dengan weight map
    mask_valid = weight_map > 0
    for c in range(3):
        target[:, :, c][mask_valid] /= weight_map[mask_valid]

    # Mengkonversi ke uint8
    target = np.clip(target, 0, 255).astype(np.uint8)

    # Membuat mask terisi
    mask_filled = (weight_map > 0).astype(np.uint8) * 255

    return target, mask_filled

# Melakukan forward warping dengan splatting
gambar_splatting, mask_splatting = forward_warp_splatting(
    gambar_src, disparitas, splat_size=3
)

# Menghitung piksel terisi setelah splatting
piksel_splatting = np.count_nonzero(mask_splatting)
print(f"Piksel terisi (splatting): {piksel_splatting} ({100*piksel_splatting/total_piksel:.1f}%)")

# ========================================================
# BAGIAN 6: INVERSE WARPING
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 6: Inverse Warping")
print("=" * 60)

def inverse_warp(gambar, disparitas):
    """Inverse warping: untuk setiap piksel target, cari sumber."""
    # Mendapatkan dimensi gambar
    h, w = gambar.shape[:2]

    # Membuat map koordinat untuk cv2.remap
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    # Iterasi setiap piksel target
    for y in range(h):
        for x in range(w):
            # Menghitung posisi sumber: x_src = x_target - disparitas
            x_src = x - disparitas[y, x]

            # Menyimpan koordinat sumber
            map_x[y, x] = x_src
            map_y[y, x] = y  # y tidak berubah untuk translasi horizontal

    # Menggunakan cv2.remap untuk interpolasi bilinear
    gambar_warp = cv2.remap(
        gambar, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return gambar_warp, map_x, map_y

# Melakukan inverse warping
gambar_inverse, map_x, map_y = inverse_warp(gambar_src, disparitas)

# Mencetak informasi inverse warping
print(f"Rentang map_x: [{map_x.min():.2f}, {map_x.max():.2f}]")
print(f"Inverse warping selesai tanpa holes")

# ========================================================
# BAGIAN 7: INTERPOLASI BILINEAR DETAIL
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 7: Interpolasi Bilinear dalam Inverse Warping")
print("=" * 60)

def bilinear_interpolasi_manual(gambar, x_float, y_float):
    """Melakukan interpolasi bilinear secara manual."""
    # Mendapatkan dimensi gambar
    h, w = gambar.shape[:2]

    # Menghitung koordinat integer di sekitar titik
    x0 = int(np.floor(x_float))
    x1 = x0 + 1
    y0 = int(np.floor(y_float))
    y1 = y0 + 1

    # Menghitung weight fraksi
    wx = x_float - x0
    wy = y_float - y0

    # Memastikan koordinat valid
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    # Mengambil 4 piksel tetangga
    p00 = gambar[y0, x0].astype(np.float64)
    p10 = gambar[y0, x1].astype(np.float64)
    p01 = gambar[y1, x0].astype(np.float64)
    p11 = gambar[y1, x1].astype(np.float64)

    # Menghitung interpolasi bilinear
    hasil = (p00 * (1 - wx) * (1 - wy) +
             p10 * wx * (1 - wy) +
             p01 * (1 - wx) * wy +
             p11 * wx * wy)

    return np.clip(hasil, 0, 255).astype(np.uint8)

# Mendemonstrasikan interpolasi bilinear pada beberapa titik
titik_uji = [(100.5, 100.3), (150.7, 80.2), (200.1, 200.9)]
print("\nDemonstrasi interpolasi bilinear:")

for x_f, y_f in titik_uji:
    # Menghitung nilai interpolasi manual
    val_manual = bilinear_interpolasi_manual(gambar_src, x_f, y_f)

    # Menghitung nilai nearest neighbor sebagai perbandingan
    val_nn = gambar_src[int(round(y_f)), int(round(x_f))]

    # Mencetak perbandingan
    print(f"  ({x_f:.1f}, {y_f:.1f}): Bilinear={val_manual}, NN={val_nn}")

# ========================================================
# BAGIAN 8: PERBANDINGAN SEMUA METODE
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 8: Perbandingan Forward vs Splatting vs Inverse")
print("=" * 60)

# Membuat figure perbandingan besar
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))

# Mengatur judul figure
fig1.suptitle("Perbandingan Forward Warping vs Inverse Warping",
              fontsize=14, fontweight='bold')

# Menampilkan gambar sumber
axes1[0, 0].imshow(cv2.cvtColor(gambar_src, cv2.COLOR_BGR2RGB))
axes1[0, 0].set_title("Gambar Sumber")
axes1[0, 0].axis('off')

# Menampilkan depth map
im_depth = axes1[0, 1].imshow(depth_map, cmap='viridis')
axes1[0, 1].set_title("Depth Map")
axes1[0, 1].axis('off')
plt.colorbar(im_depth, ax=axes1[0, 1], fraction=0.046)

# Menampilkan disparitas
im_disp = axes1[0, 2].imshow(disparitas, cmap='plasma')
axes1[0, 2].set_title("Disparitas")
axes1[0, 2].axis('off')
plt.colorbar(im_disp, ax=axes1[0, 2], fraction=0.046)

# Menampilkan forward warping dengan holes
axes1[1, 0].imshow(cv2.cvtColor(gambar_holes, cv2.COLOR_BGR2RGB))
axes1[1, 0].set_title(f"Forward Warp\n(Holes: merah, {piksel_kosong} px)")
axes1[1, 0].axis('off')

# Menampilkan forward warping dengan splatting
axes1[1, 1].imshow(cv2.cvtColor(gambar_splatting, cv2.COLOR_BGR2RGB))
axes1[1, 1].set_title(f"Forward + Splatting\n(Terisi: {piksel_splatting} px)")
axes1[1, 1].axis('off')

# Menampilkan inverse warping
axes1[1, 2].imshow(cv2.cvtColor(gambar_inverse, cv2.COLOR_BGR2RGB))
axes1[1, 2].set_title("Inverse Warp\n(Tanpa holes)")
axes1[1, 2].axis('off')

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan perbandingan
path_perbandingan = os.path.join(OUTPUT_DIR, "18_warping_perbandingan.png")
plt.savefig(path_perbandingan, dpi=150, bbox_inches='tight')
print(f"Perbandingan disimpan: {path_perbandingan}")

# Menutup figure
plt.close(fig1)

# ========================================================
# BAGIAN 9: ANALISIS ARTEFAK TEPI
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 9: Analisis Artefak Tepi (Edge Artifacts)")
print("=" * 60)

# Menghitung gradien depth untuk mendeteksi tepi depth
grad_depth_x = np.abs(np.diff(depth_map, axis=1))
grad_depth_y = np.abs(np.diff(depth_map, axis=0))

# Membuat mask tepi depth (diskontinuitas depth)
threshold_tepi = 0.05
mask_tepi_x = np.zeros((tinggi, lebar), dtype=np.uint8)
mask_tepi_x[:, :-1] = (grad_depth_x > threshold_tepi).astype(np.uint8) * 255
mask_tepi_y = np.zeros((tinggi, lebar), dtype=np.uint8)
mask_tepi_y[:-1, :] = (grad_depth_y > threshold_tepi).astype(np.uint8) * 255

# Menggabungkan mask tepi
mask_tepi = cv2.bitwise_or(mask_tepi_x, mask_tepi_y)

# Menghitung error antara forward dan inverse warping di area tepi
diff_forward_inverse = cv2.absdiff(gambar_forward, gambar_inverse)
diff_gray = cv2.cvtColor(diff_forward_inverse, cv2.COLOR_BGR2GRAY)

# Mencetak statistik artefak tepi
print(f"Piksel tepi depth   : {np.count_nonzero(mask_tepi)}")
print(f"Error rata-rata     : {diff_gray.mean():.2f}")
print(f"Error max           : {diff_gray.max()}")

# Membuat figure untuk artefak tepi
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

# Mengatur judul figure
fig2.suptitle("Analisis Artefak Tepi pada Warping", fontsize=14, fontweight='bold')

# Menampilkan tepi depth
axes2[0].imshow(mask_tepi, cmap='gray')
axes2[0].set_title("Tepi Depth (Diskontinuitas)")
axes2[0].axis('off')

# Menampilkan perbedaan forward vs inverse
axes2[1].imshow(diff_gray, cmap='hot')
axes2[1].set_title("Perbedaan Forward vs Inverse")
axes2[1].axis('off')

# Menampilkan overlay artefak
overlay = gambar_inverse.copy()
overlay[mask_tepi > 0] = [0, 255, 255]  # Kuning pada tepi
axes2[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes2[2].set_title("Overlay Tepi pada Inverse Warp")
axes2[2].axis('off')

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan analisis artefak
path_artefak = os.path.join(OUTPUT_DIR, "18_warping_artefak_tepi.png")
plt.savefig(path_artefak, dpi=150, bbox_inches='tight')
print(f"Analisis artefak disimpan: {path_artefak}")

# Menutup figure
plt.close(fig2)

# ========================================================
# BAGIAN 10: INPAINTING HOLES
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 10: Inpainting Holes pada Forward Warping")
print("=" * 60)

# Membuat mask holes (inversi dari mask_forward)
mask_inpaint = cv2.bitwise_not(mask_forward)

# Melakukan inpainting menggunakan cv2.inpaint
gambar_inpainted = cv2.inpaint(
    gambar_forward, mask_inpaint,
    inpaintRadius=3, flags=cv2.INPAINT_TELEA
)

# Mencetak informasi inpainting
print(f"Piksel yang di-inpaint: {np.count_nonzero(mask_inpaint)}")

# Membuat figure perbandingan inpainting
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

# Mengatur judul figure
fig3.suptitle("Inpainting Holes pada Forward Warping", fontsize=14, fontweight='bold')

# Menampilkan forward warp dengan holes
axes3[0].imshow(cv2.cvtColor(gambar_forward, cv2.COLOR_BGR2RGB))
axes3[0].set_title("Forward Warp (dengan holes)")
axes3[0].axis('off')

# Menampilkan mask holes
axes3[1].imshow(mask_inpaint, cmap='gray')
axes3[1].set_title("Mask Holes")
axes3[1].axis('off')

# Menampilkan hasil inpainting
axes3[2].imshow(cv2.cvtColor(gambar_inpainted, cv2.COLOR_BGR2RGB))
axes3[2].set_title("Hasil Inpainting")
axes3[2].axis('off')

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan hasil inpainting
path_inpaint = os.path.join(OUTPUT_DIR, "18_warping_inpainting.png")
plt.savefig(path_inpaint, dpi=150, bbox_inches='tight')
print(f"Inpainting disimpan: {path_inpaint}")

# Menutup figure
plt.close(fig3)

# ========================================================
# RINGKASAN
# ========================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 18: FORWARD VS INVERSE WARPING")
print("=" * 60)
print(f"Ukuran gambar    : {tinggi}x{lebar}")
print(f"Translasi kamera : {translasi_x} px")
print(f"Forward holes    : {piksel_kosong} piksel")
print(f"Splatting terisi : {piksel_splatting} piksel")
print(f"Inverse warping  : 0 holes (smooth)")
print(f"\nSemua output disimpan di: {OUTPUT_DIR}")
print("=" * 60)
