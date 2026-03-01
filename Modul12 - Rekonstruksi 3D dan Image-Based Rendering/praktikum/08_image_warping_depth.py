"""
==========================================================
PERCOBAAN 8: IMAGE WARPING BERBASIS DEPTH
Mempelajari cara melakukan forward dan inverse warping pada
gambar menggunakan informasi depth untuk menghasilkan novel
view.

Fungsi utama:
- cv2.remap()
- cv2.inpaint()
- numpy meshgrid operations
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
print("PERCOBAAN 8: IMAGE WARPING BERBASIS DEPTH")
print("=" * 60)
print()


# ========================================================
# 1. MEMUAT GAMBAR DAN DEPTH MAP
# ========================================================
print("=" * 60)
print("1. MEMUAT GAMBAR DAN DEPTH MAP")
print("=" * 60)

# Menentukan path gambar warna input
color_path = os.path.join(IMAGE_DIR, "rgbd_color_00.png")

# Menentukan path depth map input
depth_path = os.path.join(IMAGE_DIR, "rgbd_depth_00.png")

# Menjalankan download_image.py otomatis jika file RGB-D tidak tersedia
if not os.path.exists(color_path) or not os.path.exists(depth_path):
    print("[WARN] File RGB-D tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)

# Memuat gambar warna
img_color = cv2.imread(color_path)
if img_color is None:
    raise FileNotFoundError(
        f"[ERROR] {color_path} tidak tersedia.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )
print(f"  Gambar warna dimuat: {color_path}")

# Memuat depth map (16-bit)
depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
if depth_raw is None:
    raise FileNotFoundError(
        f"[ERROR] {depth_path} tidak tersedia.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )
print(f"  Depth map dimuat: {depth_path}")

# Mengkonversi depth ke float32 dan normalisasi ke [0,1]
depth_float = depth_raw.astype(np.float32) / 65535.0

# Mendapatkan dimensi gambar
h, w = img_color.shape[:2]

# Mencetak informasi gambar yang dimuat
print(f"  Ukuran gambar: {w}x{h}")
print(f"  Rentang depth: [{depth_float.min():.4f}, {depth_float.max():.4f}]")
print()


# ========================================================
# 2. MENDEFINISIKAN GERAKAN KAMERA VIRTUAL
# ========================================================
print("=" * 60)
print("2. MENDEFINISIKAN GERAKAN KAMERA VIRTUAL")
print("=" * 60)

# Menentukan parameter kamera (focal length dalam piksel)
focal_length = 500.0

# Menentukan pusat optik kamera
cx_cam = w / 2.0

# Menentukan pusat optik vertikal
cy_cam = h / 2.0

# Mendefinisikan translasi kamera virtual (geser kiri, kanan, atas)
translations = {
    "kiri": (-30.0, 0.0),
    "kanan": (30.0, 0.0),
    "atas": (0.0, -20.0),
}

# Mencetak translasi kamera virtual yang didefinisikan
for nama, (tx, ty) in translations.items():
    print(f"  Translasi {nama}: tx={tx:.1f}, ty={ty:.1f}")
print()


# ========================================================
# 3. FORWARD WARPING: PROYEKSI PIKSEL KE POSISI BARU
# ========================================================
print("=" * 60)
print("3. FORWARD WARPING: PROYEKSI PIKSEL KE POSISI BARU")
print("=" * 60)


def forward_warp(image, depth, tx, ty, focal, cx_o, cy_o):
    """
    Melakukan forward warping: memproyeksikan setiap piksel ke posisi baru
    berdasarkan informasi depth dan translasi kamera.
    """
    # Mendapatkan dimensi gambar
    h_img, w_img = depth.shape[:2]

    # Membuat output gambar kosong (hitam)
    warped = np.zeros_like(image)

    # Membuat z-buffer untuk menangani oklusi
    z_buffer = np.full((h_img, w_img), np.inf, dtype=np.float32)

    # Mengiterasi setiap piksel
    for y in range(h_img):
        for x in range(w_img):
            # Mendapatkan kedalaman piksel saat ini
            z = depth[y, x]

            # Melewatkan piksel dengan depth sangat kecil (tidak valid)
            if z < 0.01:
                continue

            # Menghitung posisi 3D piksel dalam ruang kamera
            x3d = (x - cx_o) * z / focal

            # Menghitung posisi 3D vertikal
            y3d = (y - cy_o) * z / focal

            # Menetapkan z3d sama dengan depth
            z3d = z

            # Menerapkan translasi kamera ke posisi 3D
            x3d_new = x3d - tx / focal

            # Menerapkan translasi vertikal
            y3d_new = y3d - ty / focal

            # Memproyeksikan kembali ke koordinat piksel baru
            x_new = int(round(x3d_new * focal / z3d + cx_o))

            # Menghitung koordinat piksel baru vertikal
            y_new = int(round(y3d_new * focal / z3d + cy_o))

            # Memeriksa apakah posisi baru masih dalam batas gambar
            if 0 <= x_new < w_img and 0 <= y_new < h_img:
                # Memeriksa z-buffer (hanya tulis jika lebih dekat)
                if z3d < z_buffer[y_new, x_new]:
                    # Memperbarui z-buffer
                    z_buffer[y_new, x_new] = z3d
                    # Menulis warna piksel ke posisi baru
                    warped[y_new, x_new] = image[y, x]

    # Mengembalikan gambar hasil forward warping
    return warped


# Melakukan forward warp dengan translasi ke kanan
print("  Melakukan forward warping (translasi kanan)...")

# Menggunakan versi downscale untuk mempercepat proses
scale = 0.25

# Meresize gambar untuk forward warp
img_small = cv2.resize(img_color, (int(w * scale), int(h * scale)))

# Meresize depth map
depth_small = cv2.resize(depth_float, (int(w * scale), int(h * scale)))

# Menghitung forward warp pada gambar kecil
forward_result = forward_warp(
    img_small, depth_small, 30.0 * scale, 0.0,
    focal_length * scale, cx_cam * scale, cy_cam * scale
)

# Meresize kembali hasil ke ukuran asli
forward_result = cv2.resize(forward_result, (w, h))

# Mencetak konfirmasi
print(f"  Forward warping selesai, ukuran output: {forward_result.shape}")
print()


# ========================================================
# 4. MENDETEKSI HOLES (DISOCCLUSION)
# ========================================================
print("=" * 60)
print("4. MENDETEKSI HOLES (DISOCCLUSION)")
print("=" * 60)

# Mengkonversi hasil forward warp ke grayscale untuk mendeteksi hole
gray_forward = cv2.cvtColor(forward_result, cv2.COLOR_BGR2GRAY)

# Membuat mask holes (piksel yang masih hitam/kosong)
hole_mask = (gray_forward == 0).astype(np.uint8) * 255

# Menghitung jumlah piksel hole
num_holes = np.count_nonzero(hole_mask)

# Menghitung persentase hole
pct_holes = num_holes / (h * w) * 100

# Mencetak informasi tentang holes
print(f"  Jumlah piksel hole: {num_holes}")
print(f"  Persentase hole: {pct_holes:.2f}%")
print()


# ========================================================
# 5. INVERSE WARPING
# ========================================================
print("=" * 60)
print("5. INVERSE WARPING")
print("=" * 60)


def inverse_warp(image, depth, tx, ty, focal, cx_o, cy_o):
    """
    Melakukan inverse warping menggunakan cv2.remap().
    Untuk setiap piksel di gambar tujuan, cari piksel sumbernya.
    """
    # Mendapatkan dimensi gambar
    h_img, w_img = depth.shape[:2]

    # Membuat meshgrid koordinat piksel untuk gambar tujuan
    x_coords = np.arange(w_img, dtype=np.float32)

    # Membuat array koordinat vertikal
    y_coords = np.arange(h_img, dtype=np.float32)

    # Membuat meshgrid dari kedua array koordinat
    map_x, map_y = np.meshgrid(x_coords, y_coords)

    # Menghitung posisi 3D untuk setiap piksel tujuan
    z = depth.astype(np.float32)

    # Menghindari pembagian dengan nol
    z_safe = np.maximum(z, 0.01)

    # Menghitung peta sumber X: membalik translasi kamera
    src_x = map_x + tx / z_safe

    # Menghitung peta sumber Y: membalik translasi vertikal
    src_y = map_y + ty / z_safe

    # Memastikan tipe data float32 untuk cv2.remap
    src_x = src_x.astype(np.float32)

    # Memastikan tipe data float32 untuk koordinat Y
    src_y = src_y.astype(np.float32)

    # Melakukan remapping menggunakan interpolasi bilinear
    warped = cv2.remap(image, src_x, src_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Mengembalikan gambar hasil inverse warping
    return warped


# Melakukan inverse warp dengan translasi ke kanan
print("  Melakukan inverse warping (translasi kanan)...")

# Menggunakan inverse warp pada gambar asli
inverse_result = inverse_warp(img_color, depth_float, 30.0, 0.0,
                              focal_length, cx_cam, cy_cam)

# Mencetak konfirmasi
print(f"  Inverse warping selesai, ukuran output: {inverse_result.shape}")
print()


# ========================================================
# 6. INPAINTING UNTUK MENGISI HOLES
# ========================================================
print("=" * 60)
print("6. INPAINTING UNTUK MENGISI HOLES")
print("=" * 60)

# Mendeteksi hole pada hasil inverse warping
gray_inverse = cv2.cvtColor(inverse_result, cv2.COLOR_BGR2GRAY)

# Membuat mask hole untuk inpainting
inpaint_mask = (gray_inverse == 0).astype(np.uint8) * 255

# Melakukan morphological dilation pada mask untuk memperbesar area inpaint
kernel = np.ones((3, 3), np.uint8)

# Mendilasi mask agar inpainting lebih halus
inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel, iterations=1)

# Melakukan inpainting menggunakan metode Telea
inpainted_telea = cv2.inpaint(inverse_result, inpaint_mask_dilated, 5, cv2.INPAINT_TELEA)

# Melakukan inpainting menggunakan metode Navier-Stokes
inpainted_ns = cv2.inpaint(inverse_result, inpaint_mask_dilated, 5, cv2.INPAINT_NS)

# Mencetak perbandingan metode inpainting
print(f"  Inpainting Telea selesai")
print(f"  Inpainting Navier-Stokes selesai")
print(f"  Jumlah piksel diisi: {np.count_nonzero(inpaint_mask)}")
print()


# ========================================================
# 7. PERBANDINGAN FORWARD VS INVERSE WARPING
# ========================================================
print("=" * 60)
print("7. PERBANDINGAN FORWARD VS INVERSE WARPING")
print("=" * 60)

# Membuat figure perbandingan forward vs inverse warping
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Menampilkan gambar asli
axes[0, 0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli", fontsize=12)
axes[0, 0].axis("off")

# Menampilkan depth map
axes[0, 1].imshow(depth_float, cmap="plasma")
axes[0, 1].set_title("Depth Map", fontsize=12)
axes[0, 1].axis("off")

# Menampilkan mask hole (forward)
axes[0, 2].imshow(hole_mask, cmap="gray")
axes[0, 2].set_title(f"Hole Mask Forward ({pct_holes:.1f}%)", fontsize=12)
axes[0, 2].axis("off")

# Menampilkan hasil forward warping
axes[1, 0].imshow(cv2.cvtColor(forward_result, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Forward Warping", fontsize=12)
axes[1, 0].axis("off")

# Menampilkan hasil inverse warping
axes[1, 1].imshow(cv2.cvtColor(inverse_result, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Inverse Warping", fontsize=12)
axes[1, 1].axis("off")

# Menampilkan hasil inpainting
axes[1, 2].imshow(cv2.cvtColor(inpainted_telea, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Inverse + Inpainting", fontsize=12)
axes[1, 2].axis("off")

# Mengatur layout figure agar rapi
plt.suptitle("Perbandingan Forward vs Inverse Warping", fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan figure perbandingan
comparison_path = os.path.join(OUTPUT_DIR, "08_forward_vs_inverse_warping.png")
plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close()

# Mencetak konfirmasi penyimpanan
print(f"  Disimpan: {comparison_path}")
print()


# ========================================================
# 8. MEMBUAT MULTIPLE NOVEL VIEWS (5 POSISI BERBEDA)
# ========================================================
print("=" * 60)
print("8. MEMBUAT MULTIPLE NOVEL VIEWS (5 POSISI BERBEDA)")
print("=" * 60)

# Mendefinisikan 5 posisi translasi kamera yang berbeda
novel_positions = [
    ("kiri_jauh", -50.0, 0.0),
    ("kiri_dekat", -25.0, 0.0),
    ("tengah", 0.0, 0.0),
    ("kanan_dekat", 25.0, 0.0),
    ("kanan_jauh", 50.0, 0.0),
]

# Menyiapkan list untuk menyimpan novel views
novel_views = []

# Mengiterasi setiap posisi novel view
for nama, tx, ty in novel_positions:
    # Melakukan inverse warping untuk posisi ini
    view = inverse_warp(img_color, depth_float, tx, ty,
                        focal_length, cx_cam, cy_cam)

    # Mendeteksi hole pada view
    gray_v = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    mask_v = (gray_v == 0).astype(np.uint8) * 255

    # Melakukan inpainting untuk mengisi hole
    view_clean = cv2.inpaint(view, mask_v, 5, cv2.INPAINT_TELEA)

    # Menyimpan view ke list
    novel_views.append(view_clean)

    # Mencetak informasi novel view
    print(f"  Novel view '{nama}': tx={tx:.1f}, ty={ty:.1f}")

# Menyimpan setiap novel view secara individual
for i, (nama, tx, ty) in enumerate(novel_positions):
    # Menentukan path output untuk setiap view
    view_path = os.path.join(OUTPUT_DIR, f"08_novel_view_{nama}.png")
    # Menyimpan gambar novel view
    cv2.imwrite(view_path, novel_views[i])
    # Mencetak konfirmasi
    print(f"  Disimpan: {view_path}")

print()


# ========================================================
# 9. MEMBUAT ANIMATION STRIP PERUBAHAN VIEW
# ========================================================
print("=" * 60)
print("9. MEMBUAT ANIMATION STRIP PERUBAHAN VIEW")
print("=" * 60)

# Menentukan ukuran thumbnail untuk setiap frame
thumb_h = 160

# Menghitung rasio aspek
aspect = w / h

# Menghitung lebar thumbnail
thumb_w = int(thumb_h * aspect)

# Menyiapkan list untuk thumbnail
thumbnails = []

# Meresize setiap novel view menjadi thumbnail
for view in novel_views:
    # Meresize gambar ke ukuran thumbnail
    thumb = cv2.resize(view, (thumb_w, thumb_h))
    # Menambahkan thumbnail ke list
    thumbnails.append(thumb)

# Menggabungkan semua thumbnail secara horizontal
strip = np.hstack(thumbnails)

# Menambahkan label posisi pada strip
for i, (nama, tx, ty) in enumerate(novel_positions):
    # Menghitung posisi teks label
    x_pos = i * thumb_w + 5
    # Menulis label pada strip
    cv2.putText(strip, f"tx={tx:.0f}", (x_pos, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Menyimpan animation strip
strip_path = os.path.join(OUTPUT_DIR, "08_animation_strip.png")
cv2.imwrite(strip_path, strip)

# Mencetak konfirmasi penyimpanan
print(f"  Animation strip dibuat: {len(novel_views)} frame")
print(f"  Ukuran strip: {strip.shape[1]}x{strip.shape[0]}")
print(f"  Disimpan: {strip_path}")
print()


# ========================================================
# 10. MENYIMPAN SEMUA HASIL DAN RINGKASAN
# ========================================================
print("=" * 60)
print("10. MENYIMPAN SEMUA HASIL DAN RINGKASAN")
print("=" * 60)

# Menyimpan hasil forward warping
fwd_path = os.path.join(OUTPUT_DIR, "08_forward_warped.png")
cv2.imwrite(fwd_path, forward_result)
print(f"  Disimpan: {fwd_path}")

# Menyimpan hasil inverse warping
inv_path = os.path.join(OUTPUT_DIR, "08_inverse_warped.png")
cv2.imwrite(inv_path, inverse_result)
print(f"  Disimpan: {inv_path}")

# Menyimpan hasil inpainting Telea
inp_telea_path = os.path.join(OUTPUT_DIR, "08_inpainted_telea.png")
cv2.imwrite(inp_telea_path, inpainted_telea)
print(f"  Disimpan: {inp_telea_path}")

# Menyimpan hasil inpainting Navier-Stokes
inp_ns_path = os.path.join(OUTPUT_DIR, "08_inpainted_ns.png")
cv2.imwrite(inp_ns_path, inpainted_ns)
print(f"  Disimpan: {inp_ns_path}")

# Mencetak ringkasan akhir
print()
print("=" * 60)
print("PERCOBAAN 8 SELESAI")
print("=" * 60)
print(f"  Output tersimpan di: {OUTPUT_DIR}")
print(f"  File yang dihasilkan:")
print(f"    - 08_forward_vs_inverse_warping.png")
print(f"    - 08_forward_warped.png")
print(f"    - 08_inverse_warped.png")
print(f"    - 08_inpainted_telea.png")
print(f"    - 08_inpainted_ns.png")
print(f"    - 08_novel_view_*.png (5 file)")
print(f"    - 08_animation_strip.png")
