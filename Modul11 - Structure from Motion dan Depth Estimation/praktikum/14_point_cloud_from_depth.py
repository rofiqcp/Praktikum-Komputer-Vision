"""
==========================================================================
PERCOBAAN 14: POINT CLOUD FROM DEPTH
==========================================================================
Membuat point cloud 3D dari depth map dan parameter kamera.

Referensi: Szeliski
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


def depth_to_point_cloud(depth, K):
    """
    Konversi depth map menjadi point cloud 3D.
    X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
    """
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = depth > 0
    Z = depth[valid]
    X = (u[valid] - cx) * Z / fx
    Y = (v[valid] - cy) * Z / fy
    return np.column_stack([X, Y, Z])


def main():
    """Fungsi utama: point cloud from depth."""
    print("=" * 60)
    print("PERCOBAAN 14: POINT CLOUD FROM DEPTH")
    print("=" * 60)
    
    K = np.float64([[500, 0, 250], [0, 500, 200], [0, 0, 1]])
    
    # Depth map sintetis
    depth = np.zeros((400, 500), dtype=np.float32)
    depth[50:350, 50:200] = 3.0  # objek dekat
    depth[100:300, 250:450] = 5.0  # objek jauh
    depth += np.random.normal(0, 0.1, depth.shape).astype(np.float32)
    depth = np.maximum(depth, 0)
    
    pts = depth_to_point_cloud(depth, K)
    print(f"  Points: {len(pts)}")
    print(f"  X range: [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]")
    print(f"  Y range: [{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]")
    print(f"  Z range: [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")
    
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(121)
    ax1.imshow(depth, cmap='plasma'); ax1.set_title("Depth Map"); ax1.axis('off')
    
    ax2 = fig.add_subplot(122, projection='3d')
    idx = np.random.choice(len(pts), min(5000, len(pts)), replace=False)
    ax2.scatter(pts[idx,0], pts[idx,1], pts[idx,2], c=pts[idx,2], cmap='plasma', s=1)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title("Point Cloud 3D")
    
    plt.suptitle("Depth Map → Point Cloud", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "14_point_cloud.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/14_point_cloud.png")


if __name__ == "__main__":
    main()
