"""
==========================================================================
PERCOBAAN 1: EPIPOLAR GEOMETRY VISUALISASI
==========================================================================
Visualisasi konsep epipolar geometry: epipole, epipolar lines, epipolar plane.

Referensi: Szeliski Ch.11
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


def buat_diagram_epipolar():
    """Membuat diagram konsep epipolar geometry."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, 13); ax.set_ylim(-1, 8); ax.set_aspect('equal')
    
    # Kamera kiri dan kanan
    cam_l, cam_r = np.array([2, 1]), np.array([10, 1])
    point_3d = np.array([6, 7])
    
    # Garis dari kamera ke titik 3D
    ax.plot([cam_l[0], point_3d[0]], [cam_l[1], point_3d[1]], 'b-', lw=2, label='Ray Kiri')
    ax.plot([cam_r[0], point_3d[0]], [cam_r[1], point_3d[1]], 'r-', lw=2, label='Ray Kanan')
    
    # Baseline
    ax.plot([cam_l[0], cam_r[0]], [cam_l[1], cam_r[1]], 'g--', lw=2, label='Baseline')
    
    # Image planes
    for cx, color, label in [(cam_l[0], 'blue', 'Image L'), (cam_r[0], 'red', 'Image R')]:
        ax.plot([cx-0.8, cx+0.8], [3, 3], color=color, lw=3)
        ax.text(cx, 3.3, label, ha='center', fontsize=9, color=color)
    
    # Titik-titik
    ax.plot(*cam_l, 'bs', ms=12); ax.text(cam_l[0], 0.3, 'O_L', ha='center', fontsize=10)
    ax.plot(*cam_r, 'rs', ms=12); ax.text(cam_r[0], 0.3, 'O_R', ha='center', fontsize=10)
    ax.plot(*point_3d, 'ko', ms=10); ax.text(point_3d[0]+0.3, point_3d[1]+0.3, 'P (3D)', fontsize=11)
    
    # Epipolar lines (proyeksi baseline pada image plane)
    ax.annotate('Epipolar Line', xy=(3, 3), xytext=(4.5, 4.5),
               arrowprops=dict(arrowstyle='->', color='purple'), fontsize=10, color='purple')
    
    ax.set_title('Epipolar Geometry: Titik 3D P, Dua Kamera, Epipolar Plane', fontsize=13)
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3); ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_epipolar_diagram.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/01_epipolar_diagram.png")


def visualisasi_constraint():
    """Visualisasi epipolar constraint: x'^T F x = 0."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # Diagram matematis
    text = (
        "EPIPOLAR CONSTRAINT\n\n"
        "Diberikan:\n"
        "  x  = titik pada gambar kiri\n"
        "  x' = titik pada gambar kanan\n"
        "  F  = Fundamental Matrix (3×3, rank 2)\n\n"
        "Maka:\n"
        "  x'ᵀ F x = 0\n\n"
        "Artinya:\n"
        "  x' terletak pada garis epipolar l' = Fx\n"
        "  x  terletak pada garis epipolar l  = F'x'\n\n"
        "Sifat F:\n"
        "  - Rank 2 (det(F) = 0)\n"
        "  - 7 DOF (3×3 - 1 scale - 1 rank constraint)\n"
        "  - Fe = 0, F'e' = 0 (epipoles)"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_epipolar_constraint.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/01_epipolar_constraint.png")


def main():
    """Fungsi utama: visualisasi epipolar geometry."""
    print("=" * 60)
    print("PERCOBAAN 1: EPIPOLAR GEOMETRY VISUALISASI")
    print("=" * 60)
    
    print("\n--- 1. Diagram Epipolar ---")
    buat_diagram_epipolar()
    
    print("\n--- 2. Epipolar Constraint ---")
    visualisasi_constraint()
    
    print("\nRINGKASAN: Epipolar geometry menghubungkan dua pandangan")
    print("dari scene yang sama melalui fundamental matrix.")


if __name__ == "__main__":
    main()
