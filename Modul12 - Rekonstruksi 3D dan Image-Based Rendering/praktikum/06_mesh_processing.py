"""
==========================================================================
PERCOBAAN 6: MESH PROCESSING
==========================================================================
Operasi pada mesh: decimation, smoothing, analisis topologi.

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


def buat_mesh_grid(rows=20, cols=20):
    """Buat mesh grid sederhana untuk demonstrasi."""
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(3*X) * np.cos(3*Y) * 0.3  # terrain sinusoidal
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return vertices, X, Y, Z


def decimate_mesh(vertices, ratio=0.5):
    """Simulasi mesh decimation: subsample vertices."""
    n = len(vertices)
    n_keep = max(int(n * ratio), 4)
    idx = np.random.choice(n, n_keep, replace=False)
    return vertices[idx]


def smooth_mesh(Z, iterations=3):
    """Laplacian smoothing pada grid."""
    Z_smooth = Z.copy()
    for _ in range(iterations):
        Z_new = Z_smooth.copy()
        for i in range(1, Z_smooth.shape[0]-1):
            for j in range(1, Z_smooth.shape[1]-1):
                Z_new[i,j] = (Z_smooth[i-1,j] + Z_smooth[i+1,j] + Z_smooth[i,j-1] + Z_smooth[i,j+1]) / 4
        Z_smooth = Z_new
    return Z_smooth


def main():
    """Fungsi utama: mesh processing."""
    print("=" * 60)
    print("PERCOBAAN 6: MESH PROCESSING")
    print("=" * 60)
    
    vertices, X, Y, Z = buat_mesh_grid(30, 30)
    print(f"  Vertices: {len(vertices)}")
    
    decimated = decimate_mesh(vertices, 0.3)
    print(f"  Decimated (30%): {len(decimated)}")
    
    Z_smooth = smooth_mesh(Z, iterations=5)
    
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8)
    ax1.set_title(f"Original ({len(vertices)} verts)")
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(decimated[:,0], decimated[:,1], decimated[:,2], c=decimated[:,2], cmap='terrain', s=5)
    ax2.set_title(f"Decimated ({len(decimated)} verts)")
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, Z_smooth, cmap='terrain', alpha=0.8)
    ax3.set_title("Laplacian Smoothed")
    
    plt.suptitle("Mesh Processing", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_mesh.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/06_mesh.png")


if __name__ == "__main__":
    main()
