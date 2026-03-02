"""
==========================================================================
PERCOBAAN 10: NEURAL RENDERING (KONSEP NERF)
==========================================================================
Konsep Neural Radiance Fields dan representasi implisit untuk novel view synthesis.

Referensi: Mildenhall et al. 2020, Szeliski
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


def visualisasi_nerf_pipeline():
    """Visualisasi pipeline NeRF sebagai diagram konsep."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    
    boxes = [
        (1, 6, "Input:\nMulti-view\nImages + Poses", "lightblue"),
        (4, 6, "Ray\nSampling\n(r = o + td)", "lightyellow"),
        (7, 6, "MLP Network\nF(x,d)→(c,σ)", "lightcoral"),
        (10, 6, "Volume\nRendering\nC(r)=Σ T·α·c", "lightgreen"),
        (10, 3, "Rendered\nImage", "lightyellow"),
        (7, 3, "Loss:\nMSE with\nGT Image", "pink"),
    ]
    for bx, by, text, color in boxes:
        rect = plt.Rectangle((bx-1, by-1), 2.2, 1.8, facecolor=color, edgecolor='black', lw=1.5)
        ax.add_patch(rect)
        ax.text(bx+0.1, by, text, fontsize=8, ha='center', va='center', fontweight='bold')
    
    arrows = [(3.2, 6.9, 0.6, 0), (6.2, 6.9, 0.6, 0), (9.2, 6.9, 0.6, 0),
              (11.1, 5.8, 0, -0.8), (9, 3.9, -0.6, 0)]
    for ax_x, ay_y, dx, dy in arrows:
        ax.annotate('', xy=(ax_x+dx, ay_y+dy), xytext=(ax_x, ay_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.text(7, 1.5, "NeRF: Neural Radiance Fields (Mildenhall et al., 2020)\n"
            "Input: Gambar multi-view + pose kamera\n"
            "Representasi: MLP yang memetakan (x,y,z,θ,φ) → (R,G,B,σ)\n"
            "Training: Minimize ||rendered_pixel - gt_pixel||²\n"
            "Output: Novel view dari sudut pandang manapun",
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_title("Neural Radiance Field (NeRF) Pipeline", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_nerf_pipeline.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/10_nerf_pipeline.png")


def volume_rendering_demo():
    """Demonstrasi konsep volume rendering: C(r) = Σ T_i · α_i · c_i."""
    n_samples = 64
    t = np.linspace(0.5, 5.0, n_samples)
    sigma = np.zeros(n_samples)
    sigma[20:30] = 2.0  # objek pertama
    sigma[45:55] = 3.0  # objek kedua
    
    colors = np.zeros((n_samples, 3))
    colors[20:30] = [1. ,0.2, 0.2]  # merah
    colors[45:55] = [0.2, 0.2, 1.0]  # biru
    
    dt = np.diff(t, append=t[-1]+0.1)
    alpha = 1 - np.exp(-sigma * dt)
    T = np.cumprod(1 - alpha + 1e-10)
    T = np.concatenate([[1.0], T[:-1]])
    weights = T * alpha
    
    C = np.sum(weights[:, None] * colors, axis=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(t, sigma, width=dt[0]*0.8, color='steelblue'); axes[0].set_title("σ(t) Density"); axes[0].set_xlabel("t (depth)")
    axes[1].bar(t, weights, width=dt[0]*0.8, color='coral'); axes[1].set_title("Weight w(t)=T·α"); axes[1].set_xlabel("t")
    axes[2].imshow([[C]], aspect='auto'); axes[2].set_title(f"Rendered Color: RGB={C.round(2)}"); axes[2].axis('off')
    
    plt.suptitle("Volume Rendering: C(r) = Σ T_i · α_i · c_i", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_volume_rendering.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/10_volume_rendering.png")


def main():
    """Fungsi utama: neural rendering (konsep NeRF)."""
    print("=" * 60)
    print("PERCOBAAN 10: NEURAL RENDERING (KONSEP NERF)")
    print("=" * 60)
    print("\n--- 1. Pipeline NeRF ---")
    visualisasi_nerf_pipeline()
    print("\n--- 2. Volume Rendering ---")
    volume_rendering_demo()
    print("\nNeRF menghasilkan novel views dari scene 3D menggunakan MLP.")


if __name__ == "__main__":
    main()
