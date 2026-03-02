"""
==========================================================================
PERCOBAAN 17: VOLUMETRIC RENDERING
==========================================================================
Rendering volume 3D: ray casting melalui density field.

Referensi: Szeliski, NeRF
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


def ray_casting_volume(volume, n_rays_x=100, n_rays_y=80, n_samples=50):
    """
    Ray casting sederhana: tembak ray dari kamera melalui volume.
    Akumulasi warna dan opacity sepanjang ray.
    """
    h, w, d = volume.shape
    image = np.zeros((n_rays_y, n_rays_x, 3))
    
    for iy in range(n_rays_y):
        for ix in range(n_rays_x):
            # Ray direction (orthographic)
            rx = int(ix * w / n_rays_x)
            ry = int(iy * h / n_rays_y)
            
            # Accumulate along z
            T = 1.0; color = np.zeros(3)
            for iz in range(d):
                sigma = volume[min(ry,h-1), min(rx,w-1), iz]
                if sigma > 0.01:
                    alpha = 1 - np.exp(-sigma * 0.5)
                    c = np.array([sigma, 0.5*sigma, 1-sigma])  # warna berdasarkan density
                    color += T * alpha * c
                    T *= (1 - alpha)
                    if T < 0.01: break
            image[iy, ix] = np.clip(color, 0, 1)
    return image


def main():
    """Fungsi utama: volumetric rendering."""
    print("=" * 60)
    print("PERCOBAAN 17: VOLUMETRIC RENDERING")
    print("=" * 60)
    
    # Volume sintetis: dua bola
    res = 40
    x = np.linspace(-2, 2, res)
    X, Y, Z = np.meshgrid(x, x, x)
    sphere1 = np.exp(-((X-0.5)**2 + Y**2 + Z**2)) * 3
    sphere2 = np.exp(-((X+0.5)**2 + (Y-0.3)**2 + (Z-0.5)**2) * 2) * 2
    volume = sphere1 + sphere2
    
    print(f"  Volume: {volume.shape}")
    print("  Ray casting...")
    
    rendered = ray_casting_volume(volume, 80, 60, res)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(volume[:,:,res//2], cmap='hot'); axes[0].set_title("Volume Slice (Z=mid)"); axes[0].axis('off')
    axes[1].imshow(volume[:,res//2,:], cmap='hot'); axes[1].set_title("Volume Slice (Y=mid)"); axes[1].axis('off')
    axes[2].imshow(rendered); axes[2].set_title("Volumetric Rendered"); axes[2].axis('off')
    
    plt.suptitle("Volumetric Rendering (Ray Casting)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_volume_render.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/17_volume_render.png")


if __name__ == "__main__":
    main()
