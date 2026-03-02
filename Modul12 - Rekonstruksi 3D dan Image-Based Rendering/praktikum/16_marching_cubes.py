"""
==========================================================================
PERCOBAAN 16: MARCHING CUBES
==========================================================================
Algoritma Marching Cubes untuk mengekstrak isosurface dari volume 3D.

Referensi: Lorensen & Cline 1987, Szeliski
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


def buat_volume_sdf(res=30):
    """Buat SDF volume: sphere."""
    x = np.linspace(-2, 2, res)
    X, Y, Z = np.meshgrid(x, x, x)
    sdf = np.sqrt(X**2 + Y**2 + Z**2) - 1.0  # sphere r=1
    return sdf, x


def marching_cubes_sederhana(volume, threshold=0.0, x_range=None):
    """
    Marching cubes sederhana: cari voxel yang dilewati isosurface.
    Voxel di-cross jika ada perubahan tanda antara corner.
    """
    if x_range is None:
        x_range = np.arange(volume.shape[0])
    vertices = []
    nx, ny, nz = volume.shape
    for i in range(nx-1):
        for j in range(ny-1):
            for k in range(nz-1):
                vals = [volume[i,j,k], volume[i+1,j,k], volume[i,j+1,k], volume[i,j,k+1],
                        volume[i+1,j+1,k], volume[i+1,j,k+1], volume[i,j+1,k+1], volume[i+1,j+1,k+1]]
                if min(vals) <= threshold <= max(vals):
                    cx = x_range[i] + (x_range[i+1]-x_range[i])/2
                    cy = x_range[j] + (x_range[j+1]-x_range[j])/2
                    cz = x_range[k] + (x_range[k+1]-x_range[k])/2
                    vertices.append([cx, cy, cz])
    return np.array(vertices) if vertices else np.zeros((0, 3))


def main():
    """Fungsi utama: marching cubes."""
    print("=" * 60)
    print("PERCOBAAN 16: MARCHING CUBES")
    print("=" * 60)
    
    sdf, x = buat_volume_sdf(30)
    print(f"  Volume: {sdf.shape}")
    
    surface = marching_cubes_sederhana(sdf, 0.0, x)
    print(f"  Surface voxels: {len(surface)}")
    
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(131)
    mid = sdf.shape[0] // 2
    ax1.imshow(sdf[mid], cmap='RdBu', vmin=-2, vmax=2)
    ax1.contour(sdf[mid], levels=[0], colors='black', linewidths=2)
    ax1.set_title("SDF Slice (Y=midplane)")
    
    ax2 = fig.add_subplot(132, projection='3d')
    if len(surface) > 0:
        ax2.scatter(surface[:,0], surface[:,1], surface[:,2], c=surface[:,2], cmap='viridis', s=3)
    ax2.set_title(f"Isosurface ({len(surface)} pts)")
    
    # Sphere GT
    ax3 = fig.add_subplot(133, projection='3d')
    u = np.linspace(0, 2*np.pi, 30); v = np.linspace(0, np.pi, 15)
    U, V = np.meshgrid(u, v)
    ax3.plot_surface(np.sin(V)*np.cos(U), np.sin(V)*np.sin(U), np.cos(V), alpha=0.5, cmap='viridis')
    ax3.set_title("Ground Truth Sphere")
    
    plt.suptitle("Marching Cubes Algorithm", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_marching_cubes.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/16_marching_cubes.png")


if __name__ == "__main__":
    main()
