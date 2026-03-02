"""
==========================================================================
PERCOBAAN 12: ALPHA SHAPES
==========================================================================
Rekonstruksi boundary menggunakan alpha shapes.

Referensi: Edelsbrunner & Mücke
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


def alpha_shape_2d(points, alpha=1.0):
    """
    Alpha shape: Delaunay triangulation yang dihapus simplex
    dengan circumradius > 1/alpha.
    """
    from scipy.spatial import Delaunay
    tri = Delaunay(points)
    
    triangles = []
    for simplex in tri.simplices:
        pts = points[simplex]
        a = np.linalg.norm(pts[0] - pts[1])
        b = np.linalg.norm(pts[1] - pts[2])
        c = np.linalg.norm(pts[2] - pts[0])
        s = (a + b + c) / 2
        area = np.sqrt(max(s*(s-a)*(s-b)*(s-c), 0))
        circumradius = (a * b * c) / (4 * max(area, 1e-10))
        if circumradius < 1.0 / alpha:
            triangles.append(simplex)
    return np.array(triangles) if triangles else np.array([])


def main():
    """Fungsi utama: alpha shapes."""
    print("=" * 60)
    print("PERCOBAAN 12: ALPHA SHAPES")
    print("=" * 60)
    
    np.random.seed(42)
    theta = np.linspace(0, 2*np.pi, 40, endpoint=False)
    outer = np.column_stack([2*np.cos(theta), 2*np.sin(theta)])
    inner = np.column_stack([0.8*np.cos(theta[:20]), 0.8*np.sin(theta[:20])])
    points = np.vstack([outer, inner]) + np.random.normal(0, 0.05, (60, 2))
    
    alphas = [0.5, 1.0, 2.0, 5.0]
    fig, axes = plt.subplots(1, len(alphas), figsize=(4*len(alphas), 4))
    
    for ax, alpha in zip(axes, alphas):
        ax.scatter(points[:,0], points[:,1], c='blue', s=10, zorder=5)
        try:
            tris = alpha_shape_2d(points, alpha)
            if len(tris) > 0:
                for tri in tris:
                    triangle = plt.Polygon(points[tri], fill=True, facecolor='lightgreen', edgecolor='green', alpha=0.5)
                    ax.add_patch(triangle)
            ax.set_title(f"α = {alpha} ({len(tris)} tri)")
        except ImportError:
            ax.set_title(f"α = {alpha} (scipy needed)")
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    plt.suptitle("Alpha Shapes (2D)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_alpha_shapes.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/12_alpha_shapes.png")


if __name__ == "__main__":
    main()
