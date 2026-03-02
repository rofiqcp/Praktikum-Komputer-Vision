"""
==========================================================================
PERCOBAAN 19: 3D GAUSSIAN SPLATTING (KONSEP)
==========================================================================
Konsep 3D Gaussian Splatting untuk real-time novel view synthesis.

Referensi: Kerbl et al. 2023
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


def visualisasi_gaussian_splatting():
    """Visualisasi konsep 3D Gaussian Splatting."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. 3D Gaussians
    np.random.seed(42)
    centers = np.random.randn(30, 2) * 1.5
    sizes = np.random.uniform(0.1, 0.5, 30)
    colors_rgb = np.random.rand(30, 3)
    
    ax = axes[0]
    for c, s, col in zip(centers, sizes, colors_rgb):
        circle = plt.Circle(c, s, color=col, alpha=0.4)
        ax.add_patch(circle)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_title("1. 3D Gaussians (2D proj)"); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    # 2. Splatting process
    ax2 = axes[1]
    img_splat = np.ones((100, 100, 3))
    for c, s, col in zip(centers, sizes, colors_rgb):
        iy, ix = np.mgrid[0:100, 0:100]
        cx = int((c[0] + 4) / 8 * 100)
        cy = int((c[1] + 4) / 8 * 100)
        r = max(int(s / 8 * 100), 1)
        dist = np.sqrt((ix - cx)**2 + (iy - cy)**2)
        weight = np.exp(-dist**2 / (2 * r**2))
        for ch in range(3):
            img_splat[:, :, ch] -= weight * 0.3 * (1 - col[ch])
    img_splat = np.clip(img_splat, 0, 1)
    ax2.imshow(img_splat)
    ax2.set_title("2. Splatted Image"); ax2.axis('off')
    
    # 3. Pipeline diagram
    ax3 = axes[2]
    steps = [
        "SfM Points\n(COLMAP)", "Initialize\nGaussians", "Differentiable\nSplatting",
        "Compare with\nGT Image", "Optimize\n(μ,Σ,α,c)", "Novel Views\n(Real-time!)"]
    for i, step in enumerate(steps):
        y = 5 - i * 0.9
        color = ['lightblue', 'lightyellow', 'lightcoral', 'pink', 'lightgreen', 'gold'][i]
        rect = plt.Rectangle((0.5, y-0.35), 3, 0.6, facecolor=color, edgecolor='black')
        ax3.add_patch(rect)
        ax3.text(2, y, step, ha='center', va='center', fontsize=8, fontweight='bold')
        if i < len(steps)-1:
            ax3.annotate('', xy=(2, y-0.35), xytext=(2, y-0.55), arrowprops=dict(arrowstyle='->'))
    ax3.set_xlim(0, 4); ax3.set_ylim(-0.5, 5.5)
    ax3.set_title("3. 3DGS Pipeline"); ax3.axis('off')
    
    plt.suptitle("3D Gaussian Splatting (Kerbl et al., 2023)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "19_3dgs_konsep.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/19_3dgs_konsep.png")


def main():
    """Fungsi utama: 3D Gaussian Splatting konsep."""
    print("=" * 60)
    print("PERCOBAAN 19: 3D GAUSSIAN SPLATTING (KONSEP)")
    print("=" * 60)
    visualisasi_gaussian_splatting()
    print("\n  3DGS: Representasi scene sebagai kumpulan 3D Gaussian.")
    print("  Setiap Gaussian punya: posisi (μ), covariance (Σ), opacity (α), warna (c).")
    print("  Rendering: differentiable splatting → real-time (~100 FPS).")
    print("  Keunggulan dibanding NeRF: training + rendering jauh lebih cepat.")


if __name__ == "__main__":
    main()
