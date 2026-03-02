"""
==========================================================================
PERCOBAAN 5: SURFACE RECONSTRUCTION (POISSON)
==========================================================================
Rekonstruksi permukaan dari point cloud menggunakan Poisson reconstruction (konsep).

Referensi: Szeliski Ch.13
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


def poisson_reconstruction_konsep():
    """
    Poisson Surface Reconstruction (konsep):
    1. Estimasi normal pada setiap titik
    2. Definisikan indicator function χ dari normal (∇χ = V)
    3. Solve Poisson equation: ∆χ = ∇·V
    4. Extract isosurface (marching cubes)
    
    Implementasi penuh memerlukan Open3D, di sini kita visualisasikan konsep.
    """
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(0, np.pi, 25)
    T, P = np.meshgrid(theta, phi)
    x = np.sin(P) * np.cos(T)
    y = np.sin(P) * np.sin(T)
    z = np.cos(P)
    return x, y, z


def buat_mesh_sederhana():
    """Buat mesh sederhana dari grid 3D (simulasi output Poisson)."""
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    U, V = np.meshgrid(u, v)
    x = np.sin(V) * np.cos(U)
    y = np.sin(V) * np.sin(U)
    z = np.cos(V)
    return x, y, z


def main():
    """Fungsi utama: Poisson surface reconstruction konsep."""
    print("=" * 60)
    print("PERCOBAAN 5: SURFACE RECONSTRUCTION (POISSON)")
    print("=" * 60)
    
    fig = plt.figure(figsize=(16, 5))
    
    # Point cloud
    np.random.seed(42)
    n = 500
    theta = np.random.uniform(0, 2*np.pi, n)
    phi = np.random.uniform(0, np.pi, n)
    pts = np.column_stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(pts[:,0], pts[:,1], pts[:,2], c=pts[:,2], cmap='viridis', s=3)
    ax1.set_title("1. Input Point Cloud")
    
    # Normals
    ax2 = fig.add_subplot(132, projection='3d')
    step = 10
    ax2.scatter(pts[:,0], pts[:,1], pts[:,2], c='lightgray', s=1)
    normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    ax2.quiver(pts[::step,0], pts[::step,1], pts[::step,2],
               normals[::step,0], normals[::step,1], normals[::step,2],
               length=0.15, color='red', alpha=0.5)
    ax2.set_title("2. Estimated Normals")
    
    # Reconstructed surface
    x, y, z = buat_mesh_sederhana()
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(x, y, z, cmap='viridis', alpha=0.7)
    ax3.set_title("3. Reconstructed Surface")
    
    plt.suptitle("Poisson Surface Reconstruction Pipeline", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_poisson.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/05_poisson.png")
    print("\n  Pipeline: Point Cloud → Normal Estimation → Poisson Solve → Mesh")


if __name__ == "__main__":
    main()
