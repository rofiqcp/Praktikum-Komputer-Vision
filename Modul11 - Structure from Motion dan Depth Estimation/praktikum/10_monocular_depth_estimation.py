"""
==========================================================================
PERCOBAAN 10: MONOCULAR DEPTH ESTIMATION
==========================================================================
Estimasi kedalaman dari satu gambar menggunakan pendekatan klasik/MiDaS.

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


def estimasi_depth_gradient(gray):
    """
    Estimasi depth sederhana dari gradien dan tekstur.
    Asumsi: area dengan gradien tinggi lebih dekat (objek tajam).
    """
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    depth_est = 255 - cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_est = cv2.GaussianBlur(depth_est, (21,21), 0)
    return depth_est


def estimasi_depth_vertical(gray):
    """
    Heuristik: objek di bawah gambar biasanya lebih dekat.
    Membuat depth gradient vertikal.
    """
    h, w = gray.shape
    depth = np.zeros_like(gray, dtype=np.float32)
    for y in range(h):
        depth[y, :] = (y / h) * 255
    return depth.astype(np.uint8)


def main():
    """Fungsi utama: monocular depth estimation."""
    print("=" * 60)
    print("PERCOBAAN 10: MONOCULAR DEPTH ESTIMATION")
    print("=" * 60)
    
    img = load_gambar("indoor_scene.jpg")
    if img is None: img = load_gambar("gambar_depth.png")
    if img is None:
        img = np.random.randint(80, 200, (400, 600, 3), dtype=np.uint8)
        cv2.rectangle(img, (50,200), (200,350), (80,60,40), -1)
        cv2.rectangle(img, (300,100), (500,350), (120,100,80), -1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    d_grad = estimasi_depth_gradient(gray)
    d_vert = estimasi_depth_vertical(gray)
    d_combined = cv2.addWeighted(d_grad, 0.5, d_vert, 0.5, 0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0,0].set_title("Input")
    axes[0,1].imshow(d_grad, cmap='plasma'); axes[0,1].set_title("Depth (Gradient)")
    axes[1,0].imshow(d_vert, cmap='plasma'); axes[1,0].set_title("Depth (Vertical)")
    axes[1,1].imshow(d_combined, cmap='plasma'); axes[1,1].set_title("Depth (Combined)")
    for ax in axes.flat: ax.axis('off')
    plt.suptitle("Monocular Depth Estimation (Heuristic)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_monocular_depth.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/10_monocular_depth.png")
    
    cv2.imshow("Depth", cv2.applyColorMap(d_combined, cv2.COLORMAP_PLASMA))
    cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
