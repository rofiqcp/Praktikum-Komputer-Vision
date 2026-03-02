"""
==========================================================================
PERCOBAAN 17: DEPTH MAP COLORIZATION
==========================================================================
Memvisualisasikan depth map dengan berbagai colormap.

Referensi: OpenCV docs
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


def colorize_depth(depth, colormaps=None):
    """Menerapkan berbagai colormap pada depth map."""
    if colormaps is None:
        colormaps = [
            ("JET", cv2.COLORMAP_JET), ("PLASMA", cv2.COLORMAP_PLASMA),
            ("INFERNO", cv2.COLORMAP_INFERNO), ("TURBO", cv2.COLORMAP_TURBO),
            ("HOT", cv2.COLORMAP_HOT), ("BONE", cv2.COLORMAP_BONE)]
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results = [(name, cv2.applyColorMap(depth_norm, cmap)) for name, cmap in colormaps]
    return results


def main():
    """Fungsi utama: depth map colorization."""
    print("=" * 60)
    print("PERCOBAAN 17: DEPTH MAP COLORIZATION")
    print("=" * 60)
    
    depth = np.zeros((400, 500), dtype=np.float32)
    for i in range(5):
        depth[50+i*60:100+i*60, 50:450] = (i+1) * 50
    depth = cv2.GaussianBlur(depth, (21, 21), 0)
    
    results = colorize_depth(depth)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, colored) in zip(axes.flat, results):
        ax.imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        ax.set_title(name); ax.axis('off')
    plt.suptitle("Depth Map Colorization", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_colorization.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/17_colorization.png")
    
    cv2.imshow("Depth JET", results[0][1])
    cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
