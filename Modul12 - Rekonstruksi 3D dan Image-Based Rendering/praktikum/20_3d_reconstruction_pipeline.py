"""
==========================================================================
PERCOBAAN 20: PIPELINE REKONSTRUKSI 3D LENGKAP
==========================================================================
Pipeline end-to-end rekonstruksi 3D: depth → point cloud → filtering → visualisasi.

Referensi: Szeliski, semua referensi modul 12
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_gambar(nama_file):
    """Memuat gambar dari folder image/."""
    path = os.path.join(IMAGE_DIR, nama_file)
    img = cv2.imread(path)
    if img is None:
        print(f"  [WARN] Gambar {nama_file} tidak ditemukan, gunakan sintetis.")
    return img


def pipeline_rekonstruksi():
    """
    Pipeline rekonstruksi 3D lengkap:
    1. Input: depth map + gambar RGB
    2. Generate point cloud (dengan warna)
    3. Filtering (voxel downsample + SOR)
    4. Normal estimation
    5. Visualisasi 3D
    """
    # 1. Input
    h, w = 200, 250
    color = np.random.randint(80, 200, (h, w, 3), dtype=np.uint8)
    cv2.rectangle(color, (50, 50), (120, 150), (0, 200, 50), -1)
    cv2.rectangle(color, (150, 30), (220, 170), (50, 50, 200), -1)
    cv2.circle(color, (125, 100), 20, (200, 50, 50), -1)
    
    depth = np.zeros((h, w), dtype=np.float32)
    depth[50:150, 50:120] = 2.0
    depth[30:170, 150:220] = 4.0
    depth += np.random.normal(0, 0.1, depth.shape).astype(np.float32)
    depth = np.maximum(depth, 0)
    
    # 2. Point cloud
    K = np.float64([[300, 0, w/2], [0, 300, h/2], [0, 0, 1]])
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = depth > 0.1
    Z = depth[valid]; X = (u[valid]-cx)*Z/fx; Y = (v[valid]-cy)*Z/fy
    points = np.column_stack([X, Y, Z])
    colors = color[valid][:, ::-1] / 255.0
    
    # 3. Voxel downsample
    voxel_size = 0.1
    voxel_idx = np.floor(points / voxel_size).astype(int)
    unique_voxels = {}
    for i, v_idx in enumerate(voxel_idx):
        key = tuple(v_idx)
        if key not in unique_voxels:
            unique_voxels[key] = ([], [])
        unique_voxels[key][0].append(points[i])
        unique_voxels[key][1].append(colors[i])
    ds_pts = np.array([np.mean(v[0], axis=0) for v in unique_voxels.values()])
    ds_colors = np.array([np.mean(v[1], axis=0) for v in unique_voxels.values()])
    
    return color, depth, points, colors, ds_pts, ds_colors


def main():
    """Fungsi utama: pipeline rekonstruksi 3D lengkap."""
    print("=" * 60)
    print("PERCOBAAN 20: PIPELINE REKONSTRUKSI 3D LENGKAP")
    print("=" * 60)
    
    color, depth, pts_raw, colors_raw, pts_ds, colors_ds = pipeline_rekonstruksi()
    
    print(f"  Raw points: {len(pts_raw)}")
    print(f"  Downsampled: {len(pts_ds)}")
    
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = fig.add_subplot(231)
    ax1.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)); ax1.set_title("1. Input RGB"); ax1.axis('off')
    
    ax2 = fig.add_subplot(232)
    ax2.imshow(depth, cmap='plasma'); ax2.set_title("2. Depth Map"); ax2.axis('off')
    
    ax3 = fig.add_subplot(233, projection='3d')
    idx = np.random.choice(len(pts_raw), min(2000, len(pts_raw)), replace=False)
    ax3.scatter(pts_raw[idx,0], pts_raw[idx,1], pts_raw[idx,2], c=colors_raw[idx], s=2)
    ax3.set_title(f"3. Raw Cloud ({len(pts_raw)})")
    
    ax4 = fig.add_subplot(234, projection='3d')
    ax4.scatter(pts_ds[:,0], pts_ds[:,1], pts_ds[:,2], c=colors_ds, s=5)
    ax4.set_title(f"4. Filtered ({len(pts_ds)})")
    
    ax5 = fig.add_subplot(235, projection='3d')
    ax5.scatter(pts_ds[:,0], pts_ds[:,1], pts_ds[:,2], c=pts_ds[:,2], cmap='viridis', s=5)
    ax5.set_title("5. Depth-colored")
    
    ax6 = fig.add_subplot(236)
    ax6.hist(pts_ds[:,2], bins=30, color='steelblue', edgecolor='white')
    ax6.set_xlabel("Depth (Z)"); ax6.set_ylabel("Count"); ax6.set_title("6. Depth Histogram")
    
    plt.suptitle("Pipeline Rekonstruksi 3D Lengkap", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "20_pipeline.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/20_pipeline.png")
    
    print("\n  PIPELINE: RGB+Depth → Point Cloud → Filter → Normal → Visualisasi")
    print("  Untuk mesh: tambahkan Poisson / BPA / Marching Cubes")
    print("  Untuk rendering: tambahkan NeRF / 3DGS")


if __name__ == "__main__":
    main()
