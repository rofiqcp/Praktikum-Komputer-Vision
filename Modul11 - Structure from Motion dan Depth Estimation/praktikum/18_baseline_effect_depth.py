"""
==========================================================================
PERCOBAAN 18: EFEK BASELINE TERHADAP DEPTH
==========================================================================
Menganalisis pengaruh baseline stereo terhadap akurasi depth.

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


def simulasi_baseline(focal=500, baselines=[0.05, 0.1, 0.2, 0.5]):
    """Simulasi efek baseline terhadap resolusi depth."""
    disparities = np.linspace(1, 100, 200)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for B in baselines:
        depths = focal * B / disparities
        axes[0].plot(disparities, depths, label=f'B={B}m')
    axes[0].set_xlabel("Disparity (piksel)"); axes[0].set_ylabel("Depth (m)")
    axes[0].set_title("Depth vs Disparity"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    depth_values = np.linspace(0.5, 10, 100)
    for B in baselines:
        resolution = focal * B / depth_values**2  # dZ/dd
        axes[1].plot(depth_values, resolution, label=f'B={B}m')
    axes[1].set_xlabel("Depth (m)"); axes[1].set_ylabel("Resolusi Depth (m/piksel)")
    axes[1].set_title("Resolusi Depth vs Jarak"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Efek Baseline pada Depth (f={focal}px)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_baseline_effect.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/18_baseline_effect.png")


def main():
    """Fungsi utama: efek baseline terhadap depth."""
    print("=" * 60)
    print("PERCOBAAN 18: EFEK BASELINE TERHADAP DEPTH")
    print("=" * 60)
    simulasi_baseline()
    print("\nRINGKASAN: Baseline besar → depth lebih akurat di jarak jauh,")
    print("tapi sulit matching di jarak dekat. Trade-off!")


if __name__ == "__main__":
    main()
