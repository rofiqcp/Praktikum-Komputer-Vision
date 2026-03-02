"""
==========================================================================
PERCOBAAN 19: DEPTH OBJECT SEGMENTATION
==========================================================================
Segmentasi objek berdasarkan kedalaman (depth-based segmentation).

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


def segmentasi_depth(depth, n_layers=4):
    """Segmentasi objek berdasarkan range kedalaman."""
    valid = depth > 0
    if not valid.any(): return np.zeros_like(depth, dtype=np.uint8), []
    
    d_min, d_max = depth[valid].min(), depth[valid].max()
    thresholds = np.linspace(d_min, d_max, n_layers+1)
    
    segmap = np.zeros_like(depth, dtype=np.uint8)
    labels = []
    for i in range(n_layers):
        mask = (depth >= thresholds[i]) & (depth < thresholds[i+1]) & valid
        segmap[mask] = (i+1) * (255 // n_layers)
        labels.append(f"{thresholds[i]:.1f}-{thresholds[i+1]:.1f}m")
    return segmap, labels


def main():
    """Fungsi utama: depth object segmentation."""
    print("=" * 60)
    print("PERCOBAAN 19: DEPTH OBJECT SEGMENTATION")
    print("=" * 60)
    
    depth = np.zeros((400, 500), dtype=np.float32)
    cv2.circle(depth, (150, 200), 60, 2.0, -1)
    cv2.rectangle(depth, (300, 100), (450, 350), 5.0, -1)
    cv2.circle(depth, (250, 300), 40, 8.0, -1)
    depth = cv2.GaussianBlur(depth, (11, 11), 0)
    
    segmap, labels = segmentasi_depth(depth)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(depth, cmap='plasma'); axes[0].set_title("Depth Map"); axes[0].axis('off')
    axes[1].imshow(segmap, cmap='tab10'); axes[1].set_title("Segmentasi Depth"); axes[1].axis('off')
    
    colors = ['blue', 'green', 'orange', 'red']
    for i, (label, color) in enumerate(zip(labels, colors)):
        mask = segmap == (i+1) * (255 // len(labels))
        area = np.sum(mask)
        axes[2].barh(i, area, color=color)
        axes[2].text(area+100, i, label, va='center', fontsize=9)
    axes[2].set_title("Area per Depth Layer"); axes[2].set_xlabel("Piksel")
    
    plt.suptitle("Depth-based Object Segmentation", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "19_depth_segmentation.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/19_depth_segmentation.png")


if __name__ == "__main__":
    main()
