"""
==========================================================================
PERCOBAAN 7: TSDF INTEGRATION
==========================================================================
Truncated Signed Distance Function untuk fusi depth map ke volume 3D.

Referensi: Szeliski, KinectFusion
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


def buat_tsdf_volume(resolution=50, truncation=0.1):
    """Buat TSDF volume kosong."""
    volume = np.ones((resolution, resolution, resolution)) * truncation
    weights = np.zeros((resolution, resolution, resolution))
    return volume, weights


def integrate_depth(volume, weights, depth_map, K, pose, resolution=50,
                    vol_bounds=(-1, 1), truncation=0.1):
    """
    Integrasi satu depth map ke TSDF volume.
    Untuk setiap voxel: proyeksi ke kamera → baca depth → hitung SDF → update.
    """
    vol_range = np.linspace(vol_bounds[0], vol_bounds[1], resolution)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    h, w = depth_map.shape
    
    R, t = pose[:3,:3], pose[:3,3]
    
    for i, x in enumerate(vol_range):
        for j, y in enumerate(vol_range):
            for k, z in enumerate(vol_range):
                pt = R @ np.array([x, y, z]) + t
                if pt[2] <= 0: continue
                u = int(fx * pt[0] / pt[2] + cx)
                v = int(fy * pt[1] / pt[2] + cy)
                if 0 <= u < w and 0 <= v < h:
                    d = depth_map[v, u]
                    if d <= 0: continue
                    sdf = d - pt[2]
                    sdf = max(-truncation, min(truncation, sdf))
                    w_new = 1.0
                    volume[i,j,k] = (volume[i,j,k]*weights[i,j,k] + sdf*w_new) / (weights[i,j,k] + w_new)
                    weights[i,j,k] += w_new
    return volume, weights


def main():
    """Fungsi utama: TSDF integration."""
    print("=" * 60)
    print("PERCOBAAN 7: TSDF INTEGRATION")
    print("=" * 60)
    
    res = 30  # resolusi rendah agar cepat
    volume, weights = buat_tsdf_volume(res)
    
    # Depth map sintetis
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[20:80, 20:80] = 1.5
    depth += np.random.normal(0, 0.02, depth.shape).astype(np.float32)
    depth[depth < 0] = 0
    
    K = np.float64([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    pose = np.eye(4)
    
    print("  Mengintegrasi depth map ke TSDF...")
    volume, weights = integrate_depth(volume, weights, depth, K, pose, res)
    print(f"  Volume: {res}x{res}x{res}")
    print(f"  Voxels terupdate: {int((weights > 0).sum())}")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(depth, cmap='plasma'); axes[0].set_title("Input Depth Map"); axes[0].axis('off')
    
    mid = res // 2
    axes[1].imshow(volume[:, :, mid], cmap='RdBu', vmin=-0.1, vmax=0.1)
    axes[1].set_title(f"TSDF Slice Z={mid}"); axes[1].axis('off')
    
    axes[2].imshow(weights[:, :, mid], cmap='hot')
    axes[2].set_title(f"Weight Slice Z={mid}"); axes[2].axis('off')
    
    plt.suptitle("TSDF Volume Integration", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_tsdf.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/07_tsdf.png")


if __name__ == "__main__":
    main()
