"""
==========================================================================
PERCOBAAN 15: POINT CLOUD SEGMENTATION
==========================================================================
Segmentasi point cloud berdasarkan posisi, normal, dan warna.

Referensi: Szeliski, Open3D
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


def segmentasi_euclidean(points, threshold=0.3):
    """
    Euclidean clustering: group titik berdasarkan jarak.
    Implementasi sederhana menggunakan flood fill.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    labels = -np.ones(len(points), dtype=int)
    current_label = 0
    
    for i in range(len(points)):
        if labels[i] >= 0: continue
        queue = [i]; labels[i] = current_label
        while queue:
            idx = queue.pop(0)
            neighbors = tree.query_ball_point(points[idx], threshold)
            for n in neighbors:
                if labels[n] < 0:
                    labels[n] = current_label
                    queue.append(n)
        current_label += 1
    return labels


def segmentasi_plane(points, threshold=0.1, n_iter=100):
    """RANSAC plane segmentation: pisahkan plane dari objek."""
    best_inliers = np.array([])
    n = len(points)
    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1-p0, p2-p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-10: continue
        normal /= norm
        d = -np.dot(normal, p0)
        dists = np.abs(points @ normal + d)
        inliers = np.where(dists < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    mask = np.zeros(n, dtype=bool)
    mask[best_inliers] = True
    return mask


def main():
    """Fungsi utama: point cloud segmentation."""
    print("=" * 60)
    print("PERCOBAAN 15: POINT CLOUD SEGMENTATION")
    print("=" * 60)
    
    np.random.seed(42)
    # Plane (lantai)
    floor = np.column_stack([np.random.uniform(-2,2,200), np.random.uniform(-2,2,200), np.random.normal(0,0.02,200)])
    # Object 1
    obj1 = np.random.randn(80,3)*0.2 + np.array([1, 0, 0.5])
    # Object 2
    obj2 = np.random.randn(60,3)*0.15 + np.array([-0.5, 1, 0.3])
    
    points = np.vstack([floor, obj1, obj2])
    print(f"  Total titik: {len(points)}")
    
    plane_mask = segmentasi_plane(points)
    print(f"  Plane: {plane_mask.sum()} titik")
    
    try:
        obj_points = points[~plane_mask]
        labels = segmentasi_euclidean(obj_points, 0.5)
        n_clusters = len(np.unique(labels[labels >= 0]))
        print(f"  Clusters: {n_clusters}")
    except ImportError:
        labels = np.zeros(sum(~plane_mask)); n_clusters = 1
    
    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:,0], points[:,1], points[:,2], c='gray', s=3)
    ax1.set_title("Input")
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(points[plane_mask,0], points[plane_mask,1], points[plane_mask,2], c='green', s=3, label='Plane')
    ax2.scatter(points[~plane_mask,0], points[~plane_mask,1], points[~plane_mask,2], c='red', s=3, label='Object')
    ax2.set_title("Plane Segmentation"); ax2.legend()
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(points[plane_mask,0], points[plane_mask,1], points[plane_mask,2], c='lightgray', s=1)
    if n_clusters > 0:
        cmap = plt.cm.Set1(np.linspace(0,1,max(n_clusters,1)))
        for l in range(n_clusters):
            mask = labels == l
            ax3.scatter(obj_points[mask,0], obj_points[mask,1], obj_points[mask,2], c=[cmap[l%len(cmap)]], s=5)
    ax3.set_title(f"Clustering ({n_clusters} obj)")
    
    plt.suptitle("Point Cloud Segmentation", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_segmentation.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/15_segmentation.png")


if __name__ == "__main__":
    main()
