"""
==========================================================================
PERCOBAAN 11: DISPARITY TO DEPTH
==========================================================================
Konversi disparity map menjadi depth map menggunakan formula Z = f*B/d.

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


def disparity_ke_depth(disparity, focal=500, baseline=0.1):
    """
    Konversi disparity ke depth: Z = f * B / d
    f = focal length (piksel), B = baseline (meter), d = disparity (piksel).
    """
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (focal * baseline) / disparity[valid]
    return depth


def main():
    """Fungsi utama: disparity to depth."""
    print("=" * 60)
    print("PERCOBAAN 11: DISPARITY TO DEPTH")
    print("=" * 60)
    
    # Buat disparity sintetis
    h, w = 400, 500
    disp = np.zeros((h, w), dtype=np.float32)
    for i in range(5):
        x1, x2 = i*100, (i+1)*100
        disp[:, x1:x2] = 10 + i * 15  # semakin kanan semakin besar disparity
    disp += np.random.normal(0, 1, disp.shape).astype(np.float32)
    disp = np.maximum(disp, 0.1)
    
    focal, baseline = 500, 0.12
    depth = disparity_ke_depth(disp, focal, baseline)
    
    print(f"  Focal: {focal} px, Baseline: {baseline} m")
    print(f"  Disparity range: [{disp.min():.1f}, {disp.max():.1f}]")
    print(f"  Depth range: [{depth[depth>0].min():.2f}, {depth[depth>0].max():.2f}] m")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(disp, cmap='jet'); axes[0].set_title("Disparity Map"); plt.colorbar(axes[0].images[0], ax=axes[0])
    axes[1].imshow(depth, cmap='plasma'); axes[1].set_title("Depth Map (m)"); plt.colorbar(axes[1].images[0], ax=axes[1])
    axes[2].plot(disp[h//2, :], label='Disparity'); axes[2].plot(depth[h//2, :]*10, label='Depth×10')
    axes[2].set_title("Profile (baris tengah)"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.suptitle(f"Disparity → Depth (Z = f·B/d, f={focal}, B={baseline}m)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_disp_to_depth.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/11_disp_to_depth.png")


if __name__ == "__main__":
    main()
