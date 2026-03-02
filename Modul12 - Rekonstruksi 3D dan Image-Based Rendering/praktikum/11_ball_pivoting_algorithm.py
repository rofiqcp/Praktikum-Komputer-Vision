"""
==========================================================================
PERCOBAAN 11: BALL PIVOTING ALGORITHM
==========================================================================
Rekonstruksi mesh dari point cloud menggunakan Ball Pivoting Algorithm (konsep).

Referensi: Szeliski, Bernardini et al.
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


def bpa_konsep_2d():
    """
    Demonstrasi konsep BPA dalam 2D:
    Bola (lingkaran) berputar dan membuat segitiga dari titik yang disentuh.
    """
    np.random.seed(42)
    theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.normal(0, 0.05, (20, 2))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Step 1: Points
    axes[0].scatter(points[:,0], points[:,1], c='blue', s=50, zorder=5)
    axes[0].set_title("1. Input Points (2D)"); axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    
    # Step 2: Ball rolling
    axes[1].scatter(points[:,0], points[:,1], c='blue', s=50, zorder=5)
    r = 0.4
    for i in range(0, len(points)-1, 2):
        mid = (points[i] + points[i+1]) / 2
        circle = plt.Circle(mid, r, color='red', fill=False, linestyle='--', lw=1.5)
        axes[1].add_patch(circle)
        axes[1].plot([points[i,0], points[i+1,0]], [points[i,1], points[i+1,1]], 'g-', lw=1)
    axes[1].set_title("2. Ball Pivoting (r=0.4)"); axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)
    
    # Step 3: Mesh result
    from matplotlib.tri import Triangulation
    tri = Triangulation(points[:,0], points[:,1])
    axes[2].triplot(tri, 'g-', lw=1)
    axes[2].scatter(points[:,0], points[:,1], c='blue', s=50, zorder=5)
    axes[2].set_title("3. Result Mesh"); axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)
    
    plt.suptitle("Ball Pivoting Algorithm (Konsep 2D)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_bpa.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/11_bpa.png")


def main():
    """Fungsi utama: Ball Pivoting Algorithm."""
    print("=" * 60)
    print("PERCOBAAN 11: BALL PIVOTING ALGORITHM")
    print("=" * 60)
    bpa_konsep_2d()
    print("\n  BPA: bola menggelinding di permukaan point cloud,")
    print("  membuat segitiga dari tiga titik yang disentuh.")
    print("  Cocok untuk point cloud dengan density merata.")


if __name__ == "__main__":
    main()
