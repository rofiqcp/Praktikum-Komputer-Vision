"""
==========================================================================
PERCOBAAN 3: NORMAL ESTIMATION
==========================================================================
Estimasi vektor normal pada point cloud menggunakan PCA lokal.

Referensi: Szeliski, Open3D docs
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


def estimasi_normal_pca(points, k=15):
    """
    Estimasi normal menggunakan PCA lokal.
    Untuk setiap titik, ambil k tetangga, hitung covariance, 
    eigenvector terkecil = normal.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    normals = np.zeros_like(points)
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=k)
        neighbors = points[idx]
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # eigenvector terkecil
    # Orientasi konsisten (arahkan ke kamera di origin)
    for i in range(len(normals)):
        if np.dot(normals[i], -points[i]) < 0:
            normals[i] = -normals[i]
    return normals


def main():
    """Fungsi utama: normal estimation."""
    print("=" * 60)
    print("PERCOBAAN 3: NORMAL ESTIMATION")
    print("=" * 60)
    
    np.random.seed(42)
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.linspace(0, np.pi, 100)
    T, P = np.meshgrid(theta, phi)
    x = np.sin(P.ravel()) * np.cos(T.ravel())
    y = np.sin(P.ravel()) * np.sin(T.ravel())
    z = np.cos(P.ravel())
    points = np.column_stack([x, y, z])
    idx = np.random.choice(len(points), 500, replace=False)
    points = points[idx]
    
    gt_normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    try:
        est_normals = estimasi_normal_pca(points, k=15)
        dot_products = np.abs(np.sum(gt_normals * est_normals, axis=1))
        accuracy = np.mean(dot_products)
        print(f"  Normal accuracy (|dot|): {accuracy:.4f}")
    except ImportError:
        est_normals = gt_normals
        print("  [WARN] scipy tidak tersedia, gunakan GT normals")
    
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap='viridis', s=3)
    ax1.set_title("Point Cloud Sphere"); ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    ax2 = fig.add_subplot(122, projection='3d')
    step = 5
    ax2.scatter(points[:,0], points[:,1], points[:,2], c='lightgray', s=2)
    ax2.quiver(points[::step,0], points[::step,1], points[::step,2],
               est_normals[::step,0], est_normals[::step,1], est_normals[::step,2],
               length=0.15, color='red', alpha=0.6)
    ax2.set_title("Estimated Normals"); ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    plt.suptitle("Normal Estimation via PCA", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_normals.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/03_normals.png")


if __name__ == "__main__":
    main()
