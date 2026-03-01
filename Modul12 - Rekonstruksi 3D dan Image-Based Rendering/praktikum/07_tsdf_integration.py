"""
==========================================================
PERCOBAAN 7: INTEGRASI VOLUME TSDF
Mempelajari Truncated Signed Distance Function (TSDF) untuk
merekonstruksi surface 3D secara inkremental dari depth maps
berurutan.

Fungsi utama:
- open3d ScalableTSDFVolume (atau manual voxel grid)
- cv2.imread() untuk depth maps
- numpy 3D array operations
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

# Mengimpor Axes3D untuk plot 3D
from mpl_toolkits.mplot3d import Axes3D

# Mengimpor marching cubes dari skimage jika tersedia
try:
    # Mengimpor marching cubes untuk ekstraksi permukaan
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
    print("[INFO] scikit-image berhasil diimpor (marching cubes)")
except ImportError:
    HAS_SKIMAGE = False
    print("[INFO] scikit-image tidak tersedia, menggunakan threshold sederhana")

# Mencoba mengimpor Open3D, jika gagal gunakan fallback
try:
    # Mengimpor library Open3D
    import open3d as o3d
    # Menandai ketersediaan Open3D
    HAS_OPEN3D = True
    print("[INFO] Open3D berhasil diimpor")
except ImportError:
    # Menandai bahwa Open3D tidak tersedia
    HAS_OPEN3D = False
    print("[INFO] Open3D tidak tersedia, menggunakan fallback manual TSDF")

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
print("PERCOBAAN 7: INTEGRASI VOLUME TSDF")
print("=" * 60)
print()


# ========================================================
# 1. MEMUAT SEQUENCE DEPTH MAPS
# ========================================================
print("=" * 60)
print("1. MEMUAT SEQUENCE DEPTH MAPS")
print("=" * 60)

# Menentukan jumlah frame yang akan dimuat
num_frames = 8

# Menginisialisasi list untuk depth maps dan color frames
depth_maps = []
color_frames = []

# Download otomatis jika file frame tidak tersedia
_check_path = os.path.join(IMAGE_DIR, "depth_frame_000.png")
if not os.path.exists(_check_path):
    print("  [WARN] Frame tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)

# Mengiterasi setiap frame
for i in range(num_frames):
    # Membuat path file depth map
    depth_path = os.path.join(IMAGE_DIR, f"depth_frame_{i:03d}.png")

    # Membuat path file color frame
    color_path = os.path.join(IMAGE_DIR, f"color_frame_{i:03d}.png")

    # Memuat depth map (16-bit grayscale)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Memuat color frame
    color = cv2.imread(color_path, cv2.IMREAD_COLOR)

    # Mengecek apakah file berhasil dimuat
    if depth is None:
        raise FileNotFoundError(
            f"[ERROR] depth_frame_{i:03d}.png tidak tersedia.\n"
            "  Jalankan: python download_image.py"
        )
    if color is None:
        raise FileNotFoundError(
            f"[ERROR] color_frame_{i:03d}.png tidak tersedia.\n"
            "  Jalankan: python download_image.py"
        )

    depth_maps.append(depth)
    color_frames.append(color)
    print(f"  Frame {i}: depth {depth.shape}, dtype={depth.dtype}, color {color.shape}")

# Mencetak ringkasan
print(f"\n  Total frame dimuat: {len(depth_maps)}")
print(f"  Resolusi: {depth_maps[0].shape[1]}x{depth_maps[0].shape[0]}")
print()


# ========================================================
# 2. MENDEFINISIKAN CAMERA INTRINSICS
# ========================================================
print("=" * 60)
print("2. CAMERA INTRINSICS")
print("=" * 60)

# Menentukan dimensi gambar
height, width = depth_maps[0].shape[:2]

# Menentukan focal length (asumsi field of view ~60 derajat)
fx = width / (2.0 * np.tan(np.radians(30)))

# Menggunakan focal length yang sama untuk y
fy = fx

# Menentukan principal point di tengah gambar
cx = width / 2.0

# Menentukan principal point y di tengah gambar
cy = height / 2.0

# Membuat matriks intrinsik kamera
K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1 ]
], dtype=np.float64)

# Mencetak matriks intrinsik
print(f"  Focal length (fx, fy): ({fx:.2f}, {fy:.2f})")
print(f"  Principal point (cx, cy): ({cx:.2f}, {cy:.2f})")
print(f"  Matriks Intrinsik K:")
for row in K:
    print(f"    [{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f}]")
print()


# ========================================================
# 3. MEMBUAT POSE KAMERA UNTUK SETIAP FRAME
# ========================================================
print("=" * 60)
print("3. POSE KAMERA UNTUK SETIAP FRAME")
print("=" * 60)

# Menginisialisasi list untuk pose kamera (4x4 extrinsic)
camera_poses = []

# Mengiterasi setiap frame
for i in range(num_frames):
    # Menghitung sudut rotasi untuk setiap frame (kamera berputar mengelilingi objek)
    angle = i * (360.0 / num_frames)
    rad = np.radians(angle)

    # Membuat matriks rotasi pada sumbu Y
    R = np.array([
        [np.cos(rad),  0, np.sin(rad)],
        [0,            1, 0           ],
        [-np.sin(rad), 0, np.cos(rad) ]
    ])

    # Menentukan posisi kamera pada orbit
    radius_orbit = 3.0
    t = np.array([radius_orbit * np.sin(rad), 0.0, radius_orbit * np.cos(rad)])

    # Membuat matriks extrinsik 4x4
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    # Menambahkan pose ke list
    camera_poses.append(extrinsic)

    # Mencetak informasi pose
    print(f"  Frame {i}: angle={angle:.0f}°, t=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")

print()


def depth_to_point_cloud(depth, K, extrinsic, depth_scale=65535.0, depth_max=3.0):
    """
    Mengkonversi depth map ke point cloud dalam koordinat dunia.
    """
    # Mendapatkan dimensi depth map
    h, w = depth.shape

    # Membuat grid koordinat piksel
    u_coords = np.arange(w)
    v_coords = np.arange(h)
    uu, vv = np.meshgrid(u_coords, v_coords)

    # Menormalisasi depth ke meter
    z = depth.astype(np.float64) / depth_scale * depth_max

    # Membuat mask untuk depth yang valid
    valid = z > 0.01

    # Menghitung koordinat 3D dalam frame kamera
    x = (uu - K[0, 2]) * z / K[0, 0]
    y = (vv - K[1, 2]) * z / K[1, 1]

    # Menyusun point cloud dalam frame kamera
    pts_cam = np.stack([x[valid], y[valid], z[valid]], axis=-1)

    # Mentransformasi ke koordinat dunia
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    # Mengembalikan point cloud
    return pts_world


# ========================================================
# 4. INTEGRASI TSDF
# ========================================================
print("=" * 60)
print("4. INTEGRASI TSDF")
print("=" * 60)

# Menyimpan hasil rekonstruksi per tahap
reconstruction_stages = {}

# Mengecek ketersediaan Open3D untuk TSDF
if HAS_OPEN3D:
    # Membuat parameter intrinsik Open3D
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Membuat volume TSDF
    voxel_length = 4.0 / 512.0
    sdf_trunc = 0.04
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Menentukan tahap rekonstruksi yang akan divisualisasikan
    vis_stages = [1, 3, 5, num_frames]

    # Mengintegrasikan setiap frame ke volume TSDF
    for i in range(num_frames):
        # Membuat RGBD image dari depth dan color
        depth_o3d = o3d.geometry.Image(depth_maps[i].astype(np.float32))
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_frames[i], cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=65535.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )

        # Mengintegrasikan frame ke volume
        volume.integrate(rgbd, intrinsic, np.linalg.inv(camera_poses[i]))

        # Mencetak progres
        print(f"  Frame {i+1}/{num_frames} diintegrasikan")

        # Menyimpan tahap rekonstruksi
        if (i + 1) in vis_stages:
            # Mengekstrak mesh dari volume saat ini
            mesh_stage = volume.extract_triangle_mesh()
            mesh_stage.compute_vertex_normals()
            stage_verts = np.asarray(mesh_stage.vertices)
            stage_tris = np.asarray(mesh_stage.triangles)
            reconstruction_stages[i + 1] = (stage_verts, stage_tris)
            print(f"    -> Tahap {i+1}: {len(stage_verts)} vertices, {len(stage_tris)} triangles")

    # Mengekstrak point cloud dan mesh akhir
    final_pcd = volume.extract_point_cloud()
    final_mesh = volume.extract_triangle_mesh()
    final_mesh.compute_vertex_normals()

    # Mendapatkan data akhir
    final_vertices = np.asarray(final_mesh.vertices)
    final_triangles = np.asarray(final_mesh.triangles)
    final_points = np.asarray(final_pcd.points)
    final_colors = np.asarray(final_pcd.colors) if final_pcd.has_colors() else None

    print(f"\n  Mesh akhir: {len(final_vertices)} vertices, {len(final_triangles)} triangles")
    print(f"  Point cloud akhir: {len(final_points)} titik")

else:
    # ========================================================
    # FALLBACK: IMPLEMENTASI MANUAL TSDF VOXEL GRID
    # ========================================================
    print("  Menggunakan implementasi TSDF manual...")

    # Menentukan parameter voxel grid
    vol_resolution = 64
    vol_size = 6.0
    voxel_size = vol_size / vol_resolution

    # Menentukan batas volume
    vol_origin = np.array([-vol_size/2, -vol_size/2, -vol_size/2])

    # Menginisialisasi array TSDF dan weight
    tsdf_vol = np.ones((vol_resolution, vol_resolution, vol_resolution), dtype=np.float32)
    weight_vol = np.zeros((vol_resolution, vol_resolution, vol_resolution), dtype=np.float32)
    color_vol = np.zeros((vol_resolution, vol_resolution, vol_resolution, 3), dtype=np.float32)

    # Menentukan truncation distance
    trunc_margin = 5.0 * voxel_size

    # Membuat koordinat pusat setiap voxel
    x_range = np.arange(vol_resolution) * voxel_size + vol_origin[0] + voxel_size / 2
    y_range = np.arange(vol_resolution) * voxel_size + vol_origin[1] + voxel_size / 2
    z_range = np.arange(vol_resolution) * voxel_size + vol_origin[2] + voxel_size / 2

    # Menentukan tahap visualisasi
    vis_stages = [1, 3, 5, num_frames]

    # Mengintegrasikan setiap frame
    for frame_idx in range(num_frames):
        # Mendapatkan depth map saat ini
        depth_im = depth_maps[frame_idx].astype(np.float64) / 65535.0 * 3.0

        # Mendapatkan color frame saat ini
        color_im = color_frames[frame_idx].astype(np.float32) / 255.0

        # Mendapatkan pose kamera
        cam_pose = camera_poses[frame_idx]
        cam_pose_inv = np.linalg.inv(cam_pose)

        # Mengiterasi voxel (subsampled untuk kecepatan)
        step = 2
        for xi in range(0, vol_resolution, step):
            for yi in range(0, vol_resolution, step):
                for zi in range(0, vol_resolution, step):
                    # Menghitung posisi dunia dari voxel
                    pt_world = np.array([x_range[xi], y_range[yi], z_range[zi], 1.0])

                    # Mentransformasi ke koordinat kamera
                    pt_cam = cam_pose_inv @ pt_world
                    pt_cam_3d = pt_cam[:3]

                    # Mengecek apakah titik di depan kamera
                    if pt_cam_3d[2] <= 0:
                        continue

                    # Memproyeksikan ke piksel
                    u = int(fx * pt_cam_3d[0] / pt_cam_3d[2] + cx)
                    v = int(fy * pt_cam_3d[1] / pt_cam_3d[2] + cy)

                    # Mengecek apakah piksel dalam batas gambar
                    if u < 0 or u >= width or v < 0 or v >= height:
                        continue

                    # Mendapatkan depth pada piksel tersebut
                    depth_val = depth_im[v, u]

                    # Mengecek depth yang valid
                    if depth_val <= 0.01:
                        continue

                    # Menghitung signed distance
                    sdf = depth_val - pt_cam_3d[2]

                    # Menerapkan truncation
                    if sdf < -trunc_margin:
                        continue

                    # Menormalisasi SDF
                    sdf = min(sdf / trunc_margin, 1.0)

                    # Mengupdate TSDF dengan weighted average
                    w_old = weight_vol[xi, yi, zi]
                    w_new = 1.0
                    tsdf_vol[xi, yi, zi] = (w_old * tsdf_vol[xi, yi, zi] + w_new * sdf) / (w_old + w_new)
                    weight_vol[xi, yi, zi] = min(w_old + w_new, 100.0)

                    # Mengupdate warna
                    if u < color_im.shape[1] and v < color_im.shape[0]:
                        color_vol[xi, yi, zi] = (
                            w_old * color_vol[xi, yi, zi] + w_new * color_im[v, u]
                        ) / (w_old + w_new)

        # Mencetak progres
        print(f"  Frame {frame_idx+1}/{num_frames} diintegrasikan")

        # Menyimpan tahap rekonstruksi
        if (frame_idx + 1) in vis_stages:
            # Mengekstrak titik pada zero-crossing TSDF
            mask = (np.abs(tsdf_vol) < 0.5) & (weight_vol > 0)
            coords = np.argwhere(mask)
            if len(coords) > 0:
                stage_pts = coords * voxel_size + vol_origin + voxel_size / 2
                reconstruction_stages[frame_idx + 1] = (stage_pts, None)
                print(f"    -> Tahap {frame_idx+1}: {len(stage_pts)} titik surface")

    # Mengekstrak surface points dari TSDF
    if HAS_SKIMAGE:
        # Menggunakan marching cubes untuk ekstraksi permukaan
        print("\n  Mengekstrak mesh dengan marching cubes...")
        try:
            verts_mc, faces_mc, normals_mc, values_mc = marching_cubes(tsdf_vol, level=0.0)
            # Menskala dan menggeser vertices ke koordinat dunia
            final_vertices = verts_mc * voxel_size + vol_origin
            final_triangles = faces_mc
            final_points = final_vertices.copy()
            final_colors = None
            print(f"  Marching cubes: {len(final_vertices)} vertices, {len(final_triangles)} triangles")
        except Exception as e:
            print(f"  Marching cubes gagal: {e}")
            # Fallback ke point cloud
            mask = (np.abs(tsdf_vol) < 0.3) & (weight_vol > 0)
            coords = np.argwhere(mask)
            final_points = coords * voxel_size + vol_origin + voxel_size / 2
            final_vertices = final_points
            final_triangles = np.zeros((0, 3), dtype=int)
            final_colors = color_vol[mask]
    else:
        # Mengekstrak titik pada zero-crossing
        print("\n  Mengekstrak surface points (threshold)...")
        mask = (np.abs(tsdf_vol) < 0.3) & (weight_vol > 0)
        coords = np.argwhere(mask)
        final_points = coords * voxel_size + vol_origin + voxel_size / 2
        final_vertices = final_points
        final_triangles = np.zeros((0, 3), dtype=int)
        final_colors = color_vol[mask]
        print(f"  Diekstrak: {len(final_points)} surface points")

print()


# ========================================================
# 5. VISUALISASI REKONSTRUKSI PROGRESIF
# ========================================================
print("=" * 60)
print("5. REKONSTRUKSI PROGRESIF")
print("=" * 60)

# Membuat figure untuk tahap progresif
n_stages = len(reconstruction_stages)
if n_stages > 0:
    fig_prog = plt.figure(figsize=(5 * n_stages, 5))

    # Mengiterasi setiap tahap
    for idx, (stage_num, stage_data) in enumerate(sorted(reconstruction_stages.items())):
        # Membuat subplot 3D
        ax = fig_prog.add_subplot(1, n_stages, idx + 1, projection='3d')

        # Mengecek apakah data berisi mesh atau point cloud
        if stage_data[1] is not None and len(stage_data[1]) > 0:
            # Menampilkan sebagai wireframe mesh (sample)
            verts_s, tris_s = stage_data
            sample = tris_s[::max(len(tris_s)//1000, 1)]
            for tri in sample:
                pts_tri = verts_s[tri]
                pts_c = np.vstack([pts_tri, pts_tri[0]])
                ax.plot(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], 'b-', linewidth=0.3, alpha=0.5)
        else:
            # Menampilkan sebagai point cloud
            pts_s = stage_data[0]
            ax.scatter(pts_s[::3, 0], pts_s[::3, 1], pts_s[::3, 2],
                       c='steelblue', s=1, alpha=0.5)

        # Mengatur judul
        ax.set_title(f'Setelah {stage_num} frame', fontsize=10)
        ax.set_xlabel('X', fontsize=7)
        ax.set_ylabel('Y', fontsize=7)
        ax.set_zlabel('Z', fontsize=7)

    # Mengatur layout
    plt.suptitle('Rekonstruksi TSDF Progresif', fontsize=13)
    plt.tight_layout()

    # Menyimpan visualisasi progresif
    progressive_path = os.path.join(OUTPUT_DIR, "07_tsdf_progressive.png")
    plt.savefig(progressive_path, dpi=150, bbox_inches='tight')
    print(f"  Disimpan: {progressive_path}")
    plt.close()
else:
    print("  Tidak ada tahap rekonstruksi yang tersedia")
print()


# ========================================================
# 6. VISUALISASI DEPTH MAPS INPUT
# ========================================================
print("=" * 60)
print("6. VISUALISASI DEPTH MAPS INPUT")
print("=" * 60)

# Membuat figure untuk menampilkan depth maps
n_show = min(num_frames, 8)
cols = 4
rows = (n_show + cols - 1) // cols
fig_depth, axes_d = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes_d = axes_d.flatten() if n_show > 1 else [axes_d]

# Mengiterasi setiap depth map
for i in range(n_show):
    # Menampilkan depth map
    axes_d[i].imshow(depth_maps[i], cmap='viridis')
    axes_d[i].set_title(f'Depth Frame {i}', fontsize=9)
    axes_d[i].axis('off')

# Menyembunyikan subplot kosong
for i in range(n_show, len(axes_d)):
    axes_d[i].axis('off')

# Mengatur layout
plt.suptitle('Input Depth Maps', fontsize=13)
plt.tight_layout()

# Menyimpan visualisasi depth maps
depth_vis_path = os.path.join(OUTPUT_DIR, "07_tsdf_depth_maps.png")
plt.savefig(depth_vis_path, dpi=150, bbox_inches='tight')
print(f"  Disimpan: {depth_vis_path}")
plt.close()
print()


# ========================================================
# 7. VISUALISASI MESH/POINT CLOUD AKHIR
# ========================================================
print("=" * 60)
print("7. VISUALISASI REKONSTRUKSI AKHIR")
print("=" * 60)

# Membuat figure untuk hasil akhir
fig_final = plt.figure(figsize=(14, 6))

# Subplot 1: Point cloud akhir
ax_pc = fig_final.add_subplot(121, projection='3d')

# Mengecek apakah ada titik yang bisa ditampilkan
if len(final_points) > 0:
    # Mensubsampling untuk kecepatan
    step = max(len(final_points) // 5000, 1)
    pts_display = final_points[::step]

    # Menentukan warna
    if final_colors is not None and len(final_colors) >= len(final_points):
        colors_display = final_colors[::step]
        # Memastikan warna dalam range 0-1
        if colors_display.max() > 1.0:
            colors_display = colors_display / 255.0
        ax_pc.scatter(pts_display[:, 0], pts_display[:, 1], pts_display[:, 2],
                      c=colors_display, s=1, alpha=0.6)
    else:
        # Menggunakan warna berdasarkan koordinat Z
        ax_pc.scatter(pts_display[:, 0], pts_display[:, 1], pts_display[:, 2],
                      c=pts_display[:, 2], cmap='viridis', s=1, alpha=0.6)

    # Mengatur judul
    ax_pc.set_title(f'Point Cloud ({len(final_points)} titik)', fontsize=10)
else:
    ax_pc.set_title('Point Cloud (kosong)', fontsize=10)

# Mengatur label
ax_pc.set_xlabel('X')
ax_pc.set_ylabel('Y')
ax_pc.set_zlabel('Z')

# Subplot 2: Mesh akhir atau tampilan alternatif
ax_mesh = fig_final.add_subplot(122, projection='3d')

# Mengecek apakah mesh tersedia
if len(final_triangles) > 0:
    # Menampilkan wireframe mesh
    sample_tris = final_triangles[::max(len(final_triangles)//2000, 1)]
    for tri in sample_tris:
        pts_tri = final_vertices[tri]
        pts_c = np.vstack([pts_tri, pts_tri[0]])
        ax_mesh.plot(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], 'b-', linewidth=0.3, alpha=0.4)
    ax_mesh.set_title(f'Mesh ({len(final_triangles)} tris)', fontsize=10)
else:
    # Menampilkan point cloud sebagai alternatif
    if len(final_points) > 0:
        ax_mesh.scatter(pts_display[:, 0], pts_display[:, 1], pts_display[:, 2],
                        c='coral', s=2, alpha=0.5)
    ax_mesh.set_title('Surface Points', fontsize=10)

# Mengatur label
ax_mesh.set_xlabel('X')
ax_mesh.set_ylabel('Y')
ax_mesh.set_zlabel('Z')

# Mengatur layout
plt.suptitle('Hasil Rekonstruksi TSDF', fontsize=14)
plt.tight_layout()

# Menyimpan visualisasi akhir
final_path = os.path.join(OUTPUT_DIR, "07_tsdf_final_reconstruction.png")
plt.savefig(final_path, dpi=150, bbox_inches='tight')
print(f"  Disimpan: {final_path}")
plt.close()
print()


# ========================================================
# 8. STATISTIK VOLUME
# ========================================================
print("=" * 60)
print("8. STATISTIK VOLUME TSDF")
print("=" * 60)

# Mencetak statistik volume
print(f"\n  [Parameter Volume]")
if HAS_OPEN3D:
    print(f"    Metode: Open3D ScalableTSDFVolume")
    print(f"    Voxel length: {voxel_length:.6f}")
    print(f"    SDF truncation: {sdf_trunc:.4f}")
else:
    print(f"    Metode: Manual TSDF Voxel Grid")
    print(f"    Resolusi grid: {vol_resolution}x{vol_resolution}x{vol_resolution}")
    print(f"    Ukuran voxel: {voxel_size:.4f}")
    print(f"    Volume size: {vol_size:.1f}")
    print(f"    Truncation margin: {trunc_margin:.4f}")

    # Menghitung statistik TSDF manual
    occupied = np.sum(weight_vol > 0)
    total_voxels = vol_resolution ** 3
    print(f"    Voxel terisi: {occupied} / {total_voxels} ({occupied/total_voxels*100:.1f}%)")

# Mencetak statistik hasil
print(f"\n  [Hasil Rekonstruksi]")
print(f"    Frame diintegrasikan: {num_frames}")
print(f"    Point cloud: {len(final_points)} titik")
if len(final_triangles) > 0:
    print(f"    Mesh: {len(final_vertices)} vertices, {len(final_triangles)} triangles")

# Menghitung bounding box hasil rekonstruksi
if len(final_points) > 0:
    bb_min = final_points.min(axis=0)
    bb_max = final_points.max(axis=0)
    bb_size = bb_max - bb_min
    print(f"    Bounding box: [{bb_size[0]:.3f} x {bb_size[1]:.3f} x {bb_size[2]:.3f}]")
print()


# ========================================================
# 9. MENYIMPAN HASIL REKONSTRUKSI
# ========================================================
print("=" * 60)
print("9. MENYIMPAN HASIL REKONSTRUKSI")
print("=" * 60)

# Menyimpan point cloud akhir sebagai PLY
if len(final_points) > 0:
    # Menentukan path file PLY output
    ply_output_path = os.path.join(OUTPUT_DIR, "07_tsdf_reconstructed.ply")

    # Membuka file untuk menulis
    with open(ply_output_path, 'w') as f:
        # Menulis header PLY
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(final_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Menulis setiap titik
        for pt in final_points:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

    # Mencetak konfirmasi
    print(f"  Disimpan: {ply_output_path} ({len(final_points)} titik)")
print()

# Mencetak ringkasan akhir
print("=" * 60)
print("PERCOBAAN 7 SELESAI")
print("=" * 60)
print(f"  Output tersimpan di: {OUTPUT_DIR}")
print(f"  File yang dihasilkan:")
print(f"    - 07_tsdf_progressive.png")
print(f"    - 07_tsdf_depth_maps.png")
print(f"    - 07_tsdf_final_reconstruction.png")
print(f"    - 07_tsdf_reconstructed.ply")
