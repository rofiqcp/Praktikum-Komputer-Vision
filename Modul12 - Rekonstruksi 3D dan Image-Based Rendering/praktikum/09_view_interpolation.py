"""
==========================================================
PERCOBAAN 9: VIEW INTERPOLATION DAN BLENDING
Mempelajari cara menghasilkan tampilan antara (intermediate
view) dari dua gambar menggunakan interpolasi pose dan
depth-based blending.

Fungsi utama:
- cv2.remap()
- cv2.addWeighted()
- numpy interpolation
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
print("PERCOBAAN 9: VIEW INTERPOLATION DAN BLENDING")
print("=" * 60)
print()


# ========================================================
# FUNGSI UTILITAS
# ========================================================

def load_or_generate_view(index, height=480, width=640):
    """
    Memuat gambar multi-view dari file. Jika tidak ada, jalankan download_image.py otomatis.
    """
    # Menentukan path gambar multi-view
    color_path = os.path.join(IMAGE_DIR, f"multiview_{index:02d}.png")

    # Menentukan path depth map terkait
    depth_path = os.path.join(IMAGE_DIR, f"rgbd_depth_{index:02d}.png")

    # Download otomatis jika file tidak tersedia
    if not os.path.exists(color_path) or not os.path.exists(depth_path):
        print(f"[WARN] File view {index:02d} tidak ditemukan. Menjalankan download_image.py...")
        import subprocess as _subp, sys as _sys
        _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
        _subp.run([_sys.executable, _dl], check=False)

    # Memuat gambar warna
    img = cv2.imread(color_path)
    if img is None:
        raise FileNotFoundError(
            f"[ERROR] {color_path} tidak tersedia.\n"
            "  Jalankan terlebih dahulu: python download_image.py"
        )
    img = cv2.resize(img, (width, height))

    # Memuat depth map
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(
            f"[ERROR] {depth_path} tidak tersedia.\n"
            "  Jalankan terlebih dahulu: python download_image.py"
        )
    depth = cv2.resize(depth, (width, height))
    depth = depth.astype(np.float32) / 65535.0

    # Mengembalikan gambar dan depth map
    return img, depth


def depth_based_warp(image, depth, t_shift):
    """
    Melakukan warping berbasis depth menggunakan cv2.remap().
    """
    # Mendapatkan dimensi gambar
    h_img, w_img = image.shape[:2]

    # Membuat meshgrid koordinat piksel
    x_coords = np.arange(w_img, dtype=np.float32)
    y_coords = np.arange(h_img, dtype=np.float32)
    map_x, map_y = np.meshgrid(x_coords, y_coords)

    # Menghitung displacement berbasis depth (semakin dekat semakin banyak bergeser)
    z_safe = np.maximum(depth, 0.01)

    # Menghitung peta sumber X dengan displacement
    src_x = (map_x + t_shift / z_safe).astype(np.float32)

    # Peta sumber Y tidak berubah (hanya translasi horizontal)
    src_y = map_y.astype(np.float32)

    # Melakukan remapping
    warped = cv2.remap(image, src_x, src_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    # Mengembalikan gambar yang sudah di-warp
    return warped


# ========================================================
# 1. MEMUAT DUA VIEW DARI SCENE YANG SAMA
# ========================================================
print("=" * 60)
print("1. MEMUAT DUA VIEW DARI SCENE YANG SAMA")
print("=" * 60)

# Memuat view pertama (view kiri)
view_left, depth_left = load_or_generate_view(0)
print(f"  View kiri dimuat: {view_left.shape}")

# Memuat view kedua (view kanan)
view_right, depth_right = load_or_generate_view(1)
print(f"  View kanan dimuat: {view_right.shape}")

# Mendapatkan dimensi gambar
h, w = view_left.shape[:2]
print(f"  Resolusi: {w}x{h}")
print()


# ========================================================
# 2. SIMPLE LINEAR BLENDING ANTAR VIEW
# ========================================================
print("=" * 60)
print("2. SIMPLE LINEAR BLENDING ANTAR VIEW")
print("=" * 60)

# Mendefinisikan parameter interpolasi t (0.0 = view kiri, 1.0 = view kanan)
t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

# Menyiapkan list untuk hasil simple blend
simple_blends = []

# Mengiterasi setiap nilai t
for t in t_values:
    # Melakukan blending linear sederhana
    blended = cv2.addWeighted(view_left, 1.0 - t, view_right, t, 0)
    # Menambahkan hasil ke list
    simple_blends.append(blended)
    # Mencetak informasi blending
    print(f"  Simple blend t={t:.2f}: alpha_kiri={1-t:.2f}, alpha_kanan={t:.2f}")

print()


# ========================================================
# 3. DEPTH-BASED WARPING UNTUK SETIAP VIEW
# ========================================================
print("=" * 60)
print("3. DEPTH-BASED WARPING UNTUK SETIAP VIEW")
print("=" * 60)

# Menentukan total displacement antara dua view (dalam piksel)
total_shift = 30.0

# Menyiapkan list untuk view yang sudah di-warp
warped_left_views = []

# Menyiapkan list untuk view kanan yang di-warp
warped_right_views = []

# Mengiterasi setiap nilai t untuk warping
for t in t_values:
    # Menghitung shift untuk view kiri (warp ke kanan sesuai t)
    shift_left = -total_shift * t

    # Menghitung shift untuk view kanan (warp ke kiri sesuai 1-t)
    shift_right = total_shift * (1.0 - t)

    # Melakukan warping pada view kiri
    warp_l = depth_based_warp(view_left, depth_left, shift_left)

    # Melakukan warping pada view kanan
    warp_r = depth_based_warp(view_right, depth_right, shift_right)

    # Menyimpan hasil warping
    warped_left_views.append(warp_l)
    warped_right_views.append(warp_r)

    # Mencetak informasi warping
    print(f"  t={t:.2f}: shift_kiri={shift_left:.1f}px, shift_kanan={shift_right:.1f}px")

print()


# ========================================================
# 4. CROSS-DISSOLVE BLENDING PADA POSISI INTERMEDIATE
# ========================================================
print("=" * 60)
print("4. CROSS-DISSOLVE BLENDING PADA POSISI INTERMEDIATE")
print("=" * 60)

# Menyiapkan list untuk hasil depth-aware blend
depth_blends = []

# Mengiterasi setiap nilai t
for i, t in enumerate(t_values):
    # Melakukan cross-dissolve blending antara warped views
    blended = cv2.addWeighted(warped_left_views[i], 1.0 - t,
                              warped_right_views[i], t, 0)

    # Menambahkan hasil ke list
    depth_blends.append(blended)

    # Mencetak informasi blending
    print(f"  Depth-aware blend t={t:.2f} selesai")

print()


# ========================================================
# 5. PERBANDINGAN SIMPLE BLEND VS DEPTH-AWARE BLEND
# ========================================================
print("=" * 60)
print("5. PERBANDINGAN SIMPLE BLEND VS DEPTH-AWARE BLEND")
print("=" * 60)

# Membuat figure perbandingan
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

# Mengiterasi setiap nilai t untuk menampilkan perbandingan
for i, t in enumerate(t_values):
    # Menampilkan simple blend pada baris pertama
    axes[0, i].imshow(cv2.cvtColor(simple_blends[i], cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"Simple t={t:.2f}", fontsize=11)
    axes[0, i].axis("off")

    # Menampilkan depth-aware blend pada baris kedua
    axes[1, i].imshow(cv2.cvtColor(depth_blends[i], cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"Depth-Aware t={t:.2f}", fontsize=11)
    axes[1, i].axis("off")

# Menambahkan label baris
axes[0, 0].set_ylabel("Simple Blend", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Depth-Aware", fontsize=12, fontweight="bold")

# Mengatur layout figure
plt.suptitle("Perbandingan Simple Blend vs Depth-Aware Blend",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan figure perbandingan
comparison_path = os.path.join(OUTPUT_DIR, "09_blend_comparison.png")
plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close()

# Mencetak konfirmasi penyimpanan
print(f"  Disimpan: {comparison_path}")

# Menghitung perbedaan antara dua metode pada t=0.5
diff = cv2.absdiff(simple_blends[2], depth_blends[2])

# Menghitung rata-rata perbedaan
mean_diff = np.mean(diff)
print(f"  Rata-rata perbedaan pada t=0.5: {mean_diff:.2f}")
print()


# ========================================================
# 6. SMOOTH INTERPOLATION SEQUENCE (VIDEO DOLLY)
# ========================================================
print("=" * 60)
print("6. SMOOTH INTERPOLATION SEQUENCE (VIDEO DOLLY)")
print("=" * 60)

# Menentukan jumlah frame untuk interpolasi halus
num_frames = 15

# Membuat array t dari 0 ke 1 secara halus
t_smooth = np.linspace(0.0, 1.0, num_frames)

# Menyiapkan list untuk frame interpolasi
smooth_frames = []

# Mengiterasi setiap frame
for idx, t in enumerate(t_smooth):
    # Menghitung shift untuk view kiri
    shift_l = -total_shift * t

    # Menghitung shift untuk view kanan
    shift_r = total_shift * (1.0 - t)

    # Melakukan depth-based warping pada view kiri
    wl = depth_based_warp(view_left, depth_left, shift_l)

    # Melakukan depth-based warping pada view kanan
    wr = depth_based_warp(view_right, depth_right, shift_r)

    # Melakukan cross-dissolve blending
    frame = cv2.addWeighted(wl, 1.0 - t, wr, t, 0)

    # Menyimpan frame ke list
    smooth_frames.append(frame)

    # Mencetak progress setiap 5 frame
    if idx % 5 == 0 or idx == num_frames - 1:
        print(f"  Frame {idx + 1}/{num_frames}: t={t:.3f}")

# Mencetak informasi sequence
print(f"  Total frame: {num_frames}")
print()


# ========================================================
# 7. MENYIMPAN INTERPOLATION STRIP
# ========================================================
print("=" * 60)
print("7. MENYIMPAN INTERPOLATION STRIP")
print("=" * 60)

# Menentukan ukuran thumbnail untuk strip
thumb_h = 120

# Menghitung rasio aspek
aspect = w / h

# Menghitung lebar thumbnail
thumb_w = int(thumb_h * aspect)

# Memilih subset frame untuk strip (setiap frame ke-2)
step = max(1, num_frames // 8)

# Menyiapkan list untuk thumbnails
strip_thumbs = []

# Mengiterasi frame yang dipilih
for idx in range(0, num_frames, step):
    # Meresize frame menjadi thumbnail
    thumb = cv2.resize(smooth_frames[idx], (thumb_w, thumb_h))

    # Menghitung nilai t untuk label
    t_val = t_smooth[idx]

    # Menambahkan label t pada thumbnail
    cv2.putText(thumb, f"t={t_val:.2f}", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Menambahkan thumbnail ke list
    strip_thumbs.append(thumb)

# Menggabungkan thumbnail secara horizontal
interp_strip = np.hstack(strip_thumbs)

# Menyimpan interpolation strip
strip_path = os.path.join(OUTPUT_DIR, "09_interpolation_strip.png")
cv2.imwrite(strip_path, interp_strip)

# Mencetak informasi strip
print(f"  Strip dibuat: {len(strip_thumbs)} thumbnail")
print(f"  Ukuran strip: {interp_strip.shape[1]}x{interp_strip.shape[0]}")
print(f"  Disimpan: {strip_path}")
print()


# ========================================================
# 8. MENYIMPAN INDIVIDUAL INTERPOLATED VIEWS
# ========================================================
print("=" * 60)
print("8. MENYIMPAN INDIVIDUAL INTERPOLATED VIEWS")
print("=" * 60)

# Menyimpan setiap frame interpolasi individual
for idx, frame in enumerate(smooth_frames):
    # Menentukan path output untuk setiap frame
    frame_path = os.path.join(OUTPUT_DIR, f"09_interp_frame_{idx:02d}.png")

    # Menyimpan frame
    cv2.imwrite(frame_path, frame)

# Mencetak konfirmasi penyimpanan
print(f"  Disimpan: {num_frames} frame interpolasi")
print(f"  Format: 09_interp_frame_XX.png")

# Menyimpan juga view awal dan akhir
left_path = os.path.join(OUTPUT_DIR, "09_view_left.png")
cv2.imwrite(left_path, view_left)
print(f"  Disimpan: {left_path}")

# Menyimpan view kanan
right_path = os.path.join(OUTPUT_DIR, "09_view_right.png")
cv2.imwrite(right_path, view_right)
print(f"  Disimpan: {right_path}")

# Menyimpan depth maps untuk referensi
fig_depth, ax_depth = plt.subplots(1, 2, figsize=(12, 5))

# Menampilkan depth map kiri
ax_depth[0].imshow(depth_left, cmap="plasma")
ax_depth[0].set_title("Depth Map - View Kiri", fontsize=12)
ax_depth[0].axis("off")

# Menampilkan depth map kanan
ax_depth[1].imshow(depth_right, cmap="plasma")
ax_depth[1].set_title("Depth Map - View Kanan", fontsize=12)
ax_depth[1].axis("off")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi depth
depth_vis_path = os.path.join(OUTPUT_DIR, "09_depth_maps.png")
plt.savefig(depth_vis_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Disimpan: {depth_vis_path}")

print()

# Mencetak ringkasan akhir
print("=" * 60)
print("PERCOBAAN 9 SELESAI")
print("=" * 60)
print(f"  Output tersimpan di: {OUTPUT_DIR}")
print(f"  File yang dihasilkan:")
print(f"    - 09_blend_comparison.png")
print(f"    - 09_interpolation_strip.png")
print(f"    - 09_interp_frame_XX.png ({num_frames} file)")
print(f"    - 09_view_left.png")
print(f"    - 09_view_right.png")
print(f"    - 09_depth_maps.png")
