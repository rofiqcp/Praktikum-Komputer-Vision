"""
==========================================================================
PERCOBAAN 2: POINT CLOUD FILTERING
==========================================================================
Filtering point cloud: voxel downsampling, statistical outlier removal, radius filter.

Referensi: Open3D docs, Szeliski
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


def voxel_downsample(points, voxel_size=0.1):
    """
    Voxel downsampling: bagi ruang 3D menjadi voxel grid dan
    ambil centroid setiap voxel yang berisi titik.
    """
    voxel_idx = np.floor(points / voxel_size).astype(int)
    unique_voxels = {}
    for i, v in enumerate(voxel_idx):
        key = tuple(v)
        if key not in unique_voxels:
            unique_voxels[key] = []
        unique_voxels[key].append(points[i])
    result = np.array([np.mean(pts, axis=0) for pts in unique_voxels.values()])
    return result


def statistical_outlier_removal(points, k=20, std_ratio=2.0):
    """
    Statistical Outlier Removal: hitung rata-rata jarak ke k tetangga,
    hapus titik yang jaraknya > mean + std_ratio * std.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k+1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    mask = mean_dists < threshold
    return points[mask], mask


def radius_outlier_removal(points, radius=0.3, min_neighbors=5):
    """Hapus titik yang memiliki < min_neighbors dalam radius tertentu."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    counts = np.array([len(tree.query_ball_point(p, radius)) - 1 for p in points])
    mask = counts >= min_neighbors
    return points[mask], mask


def main():
    """Fungsi utama: point cloud filtering."""
    print("=" * 60)
    print("PERCOBAAN 2: POINT CLOUD FILTERING")
    print("=" * 60)
    
    np.random.seed(42)
    points = np.random.randn(1000, 3) * 0.5 + np.array([0, 0, 2])
    outliers = np.random.uniform(-3, 3, (50, 3))
    noisy = np.vstack([points, outliers])
    print(f"  Asli: {len(noisy)} titik ({len(points)} + {len(outliers)} outlier)")
    
    downsampled = voxel_downsample(noisy, 0.15)
    print(f"  Setelah voxel downsample (0.15): {len(downsampled)} titik")
    
    try:
        clean, mask = statistical_outlier_removal(noisy, k=15, std_ratio=1.5)
        print(f"  Setelah SOR (k=15, std=1.5): {len(clean)} titik")
    except ImportError:
        clean = noisy; print("  [WARN] scipy tidak tersedia, skip SOR")
    
    fig = plt.figure(figsize=(16, 5))
    datasets = [("Noisy", noisy), ("Voxel Downsample", downsampled), ("SOR Cleaned", clean)]
    for i, (title, pts) in enumerate(datasets):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=pts[:,2], cmap='viridis', s=2)
        ax.set_title(f"{title} ({len(pts)} pts)"); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.suptitle("Point Cloud Filtering", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_filtering.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/02_filtering.png")


if __name__ == "__main__":
    main()
