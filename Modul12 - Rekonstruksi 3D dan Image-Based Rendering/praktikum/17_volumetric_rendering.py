"""
==========================================================
PERCOBAAN 17: VOLUMETRIC RENDERING SEDERHANA
Mempelajari konsep dasar volumetric rendering: ray casting
melalui volume data untuk menghasilkan gambar 2D dari
model 3D.

Fungsi utama:
- numpy array operations (ray-volume intersection)
- matplotlib imshow
- cv2.normalize()
==========================================================
"""

# Mengimpor library numpy untuk operasi numerik
import numpy as np

# Mengimpor library os untuk operasi sistem file
import os

# Mengimpor library matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mengimpor library OpenCV untuk pemrosesan citra
import cv2

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
    print("[INFO] Open3D tidak tersedia, menggunakan numpy/matplotlib")

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
print("PERCOBAAN 17: VOLUMETRIC RENDERING SEDERHANA")
print("=" * 60)

# ========================================================
# BAGIAN 1: MEMBUAT VOLUME 3D (VOXEL GRID)
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 1: Membuat Volume 3D (Voxel Grid)")
print("=" * 60)

# Menentukan ukuran volume
ukuran_vol = 64

# Membuat koordinat grid 3D
lin = np.linspace(-1.5, 1.5, ukuran_vol)

# Membuat meshgrid 3D
X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')

# Membuat volume dengan bola padat (density = 1 di dalam, 0 di luar)
radius_bola = 0.8
jarak_bola = np.sqrt(X**2 + Y**2 + Z**2)

# Menghitung density: smooth falloff dari pusat bola
density_bola = np.clip(1.0 - jarak_bola / radius_bola, 0, 1)

# Menambahkan bola kedua yang lebih kecil di offset
radius_kecil = 0.4
jarak_kecil = np.sqrt((X - 0.6)**2 + (Y - 0.4)**2 + Z**2)

# Menghitung density bola kecil
density_kecil = np.clip(1.0 - jarak_kecil / radius_kecil, 0, 1)

# Menggabungkan dua bola menjadi satu volume
volume = np.maximum(density_bola, density_kecil)

# Menambahkan kubus kecil sebagai objek tambahan
mask_kubus = (np.abs(X + 0.5) < 0.3) & (np.abs(Y - 0.5) < 0.3) & (np.abs(Z) < 0.3)
volume[mask_kubus] = np.maximum(volume[mask_kubus], 0.7)

# Mencetak informasi volume
print(f"Ukuran volume: {volume.shape}")
print(f"Rentang density: [{volume.min():.3f}, {volume.max():.3f}]")
print(f"Voxel non-zero: {np.count_nonzero(volume > 0.01)}")

# ========================================================
# BAGIAN 2: DEFINISI KAMERA VIRTUAL DAN RAY GENERATION
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 2: Definisi Kamera Virtual dan Ray Generation")
print("=" * 60)

# Menentukan resolusi gambar output
res_gambar = 128

# Mendefinisikan fungsi untuk membuat ray dari kamera
def buat_rays(pos_kamera, arah_pandang, up_vector, fov, resolusi):
    """Membuat ray dari posisi kamera ke setiap piksel."""
    # Menghitung vektor kanan (right)
    right = np.cross(arah_pandang, up_vector)
    right = right / np.linalg.norm(right)

    # Menghitung ulang vektor atas yang ortogonal
    up = np.cross(right, arah_pandang)
    up = up / np.linalg.norm(up)

    # Menghitung field of view dalam radian
    fov_rad = np.radians(fov)

    # Menghitung setengah lebar frustum
    half_width = np.tan(fov_rad / 2)

    # Membuat grid piksel dari -1 sampai 1
    u = np.linspace(-half_width, half_width, resolusi)
    v = np.linspace(-half_width, half_width, resolusi)

    # Membuat meshgrid 2D untuk piksel
    U, V = np.meshgrid(u, v)

    # Menghitung arah ray untuk setiap piksel
    ray_dirs = (arah_pandang[np.newaxis, np.newaxis, :]
                + U[:, :, np.newaxis] * right[np.newaxis, np.newaxis, :]
                + V[:, :, np.newaxis] * up[np.newaxis, np.newaxis, :])

    # Menormalisasi arah ray
    norms = np.linalg.norm(ray_dirs, axis=2, keepdims=True)
    ray_dirs = ray_dirs / norms

    return ray_dirs

# Mencetak informasi kamera
print(f"Resolusi gambar output: {res_gambar}x{res_gambar}")
print(f"Total rays per view: {res_gambar * res_gambar}")

# ========================================================
# BAGIAN 3: RAY CASTING - MAXIMUM INTENSITY PROJECTION
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 3: Maximum Intensity Projection (MIP)")
print("=" * 60)

def ray_cast_volume(volume, pos_kamera, ray_dirs, mode='mip',
                    n_samples=100, threshold=0.3):
    """Cast rays melalui volume dan akumulasi nilai."""
    # Mendapatkan ukuran volume
    vol_size = volume.shape[0]

    # Mendapatkan resolusi gambar
    h, w = ray_dirs.shape[:2]

    # Membuat gambar output
    gambar = np.zeros((h, w))

    # Menentukan jarak sampling sepanjang ray
    t_vals = np.linspace(0.01, 4.0, n_samples)

    # Iterasi setiap sample sepanjang ray
    for t in t_vals:
        # Menghitung posisi sampel di world space
        pos_sample = pos_kamera[np.newaxis, np.newaxis, :] + t * ray_dirs

        # Mengkonversi dari world space ke indeks volume
        idx = ((pos_sample + 1.5) / 3.0 * vol_size).astype(int)

        # Membuat mask untuk sampel yang valid (di dalam volume)
        valid = ((idx[:, :, 0] >= 0) & (idx[:, :, 0] < vol_size) &
                 (idx[:, :, 1] >= 0) & (idx[:, :, 1] < vol_size) &
                 (idx[:, :, 2] >= 0) & (idx[:, :, 2] < vol_size))

        # Mengambil nilai density dari volume untuk sampel yang valid
        density = np.zeros((h, w))
        density[valid] = volume[
            idx[valid, 0], idx[valid, 1], idx[valid, 2]
        ]

        if mode == 'mip':
            # Mode MIP: simpan nilai maksimum
            gambar = np.maximum(gambar, density)
        elif mode == 'average':
            # Mode rata-rata: akumulasi semua nilai
            gambar += density / n_samples
        elif mode == 'first_surface':
            # Mode first surface: simpan nilai pertama di atas threshold
            mask_hit = (density > threshold) & (gambar == 0)
            gambar[mask_hit] = density[mask_hit]

    return gambar

# Mendefinisikan posisi kamera depan
pos_kamera_depan = np.array([0.0, 0.0, 3.5])

# Mendefinisikan arah pandang (ke arah origin)
arah_pandang_depan = np.array([0.0, 0.0, -1.0])

# Mendefinisikan vektor atas
up_vector = np.array([0.0, 1.0, 0.0])

# Membuat rays dari kamera depan
rays_depan = buat_rays(pos_kamera_depan, arah_pandang_depan, up_vector, 60, res_gambar)

# Melakukan MIP rendering
gambar_mip = ray_cast_volume(volume, pos_kamera_depan, rays_depan, mode='mip')

# Mencetak statistik MIP
print(f"MIP - Rentang nilai: [{gambar_mip.min():.3f}, {gambar_mip.max():.3f}]")

# ========================================================
# BAGIAN 4: AVERAGE PROJECTION
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 4: Average Projection")
print("=" * 60)

# Melakukan average projection rendering
gambar_avg = ray_cast_volume(volume, pos_kamera_depan, rays_depan, mode='average')

# Mencetak statistik average projection
print(f"Average - Rentang nilai: [{gambar_avg.min():.3f}, {gambar_avg.max():.3f}]")

# ========================================================
# BAGIAN 5: FIRST SURFACE RENDERING
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 5: First Surface Rendering")
print("=" * 60)

# Melakukan first surface rendering
gambar_surface = ray_cast_volume(
    volume, pos_kamera_depan, rays_depan,
    mode='first_surface', threshold=0.3
)

# Mencetak statistik first surface rendering
print(f"Surface - Rentang nilai: [{gambar_surface.min():.3f}, {gambar_surface.max():.3f}]")

# ========================================================
# BAGIAN 6: PERBANDINGAN MODE RENDERING
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 6: Perbandingan Mode Rendering")
print("=" * 60)

# Membuat figure untuk perbandingan 3 mode rendering
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

# Mengatur judul figure
fig1.suptitle("Perbandingan Mode Volumetric Rendering", fontsize=14, fontweight='bold')

# Menampilkan MIP
axes1[0].imshow(gambar_mip, cmap='hot', origin='lower')
axes1[0].set_title("Maximum Intensity\nProjection (MIP)")
axes1[0].axis('off')

# Menampilkan Average Projection
axes1[1].imshow(gambar_avg, cmap='hot', origin='lower')
axes1[1].set_title("Average Projection")
axes1[1].axis('off')

# Menampilkan First Surface Rendering
axes1[2].imshow(gambar_surface, cmap='hot', origin='lower')
axes1[2].set_title("First Surface\nRendering")
axes1[2].axis('off')

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan perbandingan mode rendering
path_mode = os.path.join(OUTPUT_DIR, "17_volumetric_mode_perbandingan.png")
plt.savefig(path_mode, dpi=150, bbox_inches='tight')
print(f"Perbandingan mode disimpan: {path_mode}")

# Menutup figure
plt.close(fig1)

# ========================================================
# BAGIAN 7: RENDERING DARI BERBAGAI SUDUT PANDANG
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 7: Rendering dari Berbagai Sudut Pandang")
print("=" * 60)

# Mendefinisikan posisi kamera untuk 3 sudut pandang
viewpoints = {
    'Depan (Front)': {
        'pos': np.array([0.0, 0.0, 3.5]),
        'dir': np.array([0.0, 0.0, -1.0]),
        'up': np.array([0.0, 1.0, 0.0])
    },
    'Samping (Side)': {
        'pos': np.array([3.5, 0.0, 0.0]),
        'dir': np.array([-1.0, 0.0, 0.0]),
        'up': np.array([0.0, 1.0, 0.0])
    },
    'Atas (Top)': {
        'pos': np.array([0.0, 3.5, 0.0]),
        'dir': np.array([0.0, -1.0, 0.0]),
        'up': np.array([0.0, 0.0, -1.0])
    }
}

# Membuat figure untuk berbagai sudut pandang
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

# Mengatur judul figure
fig2.suptitle("MIP dari Berbagai Sudut Pandang", fontsize=14, fontweight='bold')

# Iterasi setiap sudut pandang
for idx, (nama_view, params) in enumerate(viewpoints.items()):
    # Membuat rays untuk sudut pandang ini
    rays_view = buat_rays(params['pos'], params['dir'], params['up'], 60, res_gambar)

    # Melakukan MIP rendering
    gambar_view = ray_cast_volume(volume, params['pos'], rays_view, mode='mip')

    # Menampilkan hasil rendering
    axes2[idx].imshow(gambar_view, cmap='inferno', origin='lower')
    axes2[idx].set_title(nama_view)
    axes2[idx].axis('off')

    # Mencetak informasi sudut pandang
    print(f"{nama_view}: max={gambar_view.max():.3f}, mean={gambar_view.mean():.3f}")

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan visualisasi sudut pandang
path_views = os.path.join(OUTPUT_DIR, "17_volumetric_multi_view.png")
plt.savefig(path_views, dpi=150, bbox_inches='tight')
print(f"Multi-view rendering disimpan: {path_views}")

# Menutup figure
plt.close(fig2)

# ========================================================
# BAGIAN 8: NORMALISASI DAN EXPORT DENGAN OPENCV
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 8: Normalisasi dan Export dengan OpenCV")
print("=" * 60)

# Menormalisasi gambar MIP ke range 0-255
gambar_mip_norm = cv2.normalize(
    gambar_mip, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Mengaplikasikan colormap pada gambar MIP
gambar_mip_color = cv2.applyColorMap(gambar_mip_norm, cv2.COLORMAP_HOT)

# Menyimpan gambar MIP sebagai file PNG
path_mip_cv = os.path.join(OUTPUT_DIR, "17_volumetric_mip_opencv.png")
cv2.imwrite(path_mip_cv, gambar_mip_color)
print(f"MIP (OpenCV) disimpan: {path_mip_cv}")

# Menormalisasi gambar average projection
gambar_avg_norm = cv2.normalize(
    gambar_avg, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Mengaplikasikan colormap pada gambar average
gambar_avg_color = cv2.applyColorMap(gambar_avg_norm, cv2.COLORMAP_JET)

# Menyimpan gambar average projection
path_avg_cv = os.path.join(OUTPUT_DIR, "17_volumetric_avg_opencv.png")
cv2.imwrite(path_avg_cv, gambar_avg_color)
print(f"Average (OpenCV) disimpan: {path_avg_cv}")

# ========================================================
# BAGIAN 9: VISUALISASI SLICE VOLUME
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 9: Visualisasi Slice Volume")
print("=" * 60)

# Membuat figure untuk menampilkan slice dari 3 sumbu
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

# Mengatur judul figure
fig3.suptitle("Slice dari Volume 3D", fontsize=14, fontweight='bold')

# Mengambil slice di tengah volume (sumbu X)
slice_x = volume[ukuran_vol // 2, :, :]
axes3[0].imshow(slice_x, cmap='viridis', origin='lower')
axes3[0].set_title(f"Slice X (idx={ukuran_vol//2})")
axes3[0].set_xlabel("Z")
axes3[0].set_ylabel("Y")

# Mengambil slice di tengah volume (sumbu Y)
slice_y = volume[:, ukuran_vol // 2, :]
axes3[1].imshow(slice_y, cmap='viridis', origin='lower')
axes3[1].set_title(f"Slice Y (idx={ukuran_vol//2})")
axes3[1].set_xlabel("Z")
axes3[1].set_ylabel("X")

# Mengambil slice di tengah volume (sumbu Z)
slice_z = volume[:, :, ukuran_vol // 2]
axes3[2].imshow(slice_z, cmap='viridis', origin='lower')
axes3[2].set_title(f"Slice Z (idx={ukuran_vol//2})")
axes3[2].set_xlabel("Y")
axes3[2].set_ylabel("X")

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan visualisasi slice
path_slice = os.path.join(OUTPUT_DIR, "17_volumetric_slices.png")
plt.savefig(path_slice, dpi=150, bbox_inches='tight')
print(f"Slice volume disimpan: {path_slice}")

# Menutup figure
plt.close(fig3)

# ========================================================
# RINGKASAN
# ========================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 17: VOLUMETRIC RENDERING SEDERHANA")
print("=" * 60)
print(f"Ukuran volume       : {ukuran_vol}³ voxels")
print(f"Resolusi rendering  : {res_gambar}x{res_gambar}")
print(f"Mode rendering      : MIP, Average, First Surface")
print(f"Sudut pandang       : {list(viewpoints.keys())}")
print(f"\nSemua output disimpan di: {OUTPUT_DIR}")
print("=" * 60)
