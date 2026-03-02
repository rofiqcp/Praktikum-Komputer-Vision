"""
==========================================================================
PERCOBAAN 8: IMAGE WARPING DENGAN DEPTH
==========================================================================
Forward dan inverse warping menggunakan depth map.

Referensi: Szeliski Ch.14
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


def forward_warping(img, depth, K, R, t):
    """
    Forward warping: untuk setiap piksel di source, hitung posisi di target.
    Masalah: holes dan conflicts.
    """
    h, w = img.shape[:2]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    result = np.zeros_like(img)
    zbuf = np.full((h, w), np.inf)
    
    for v in range(h):
        for u in range(w):
            Z = depth[v, u]
            if Z <= 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            pt3d = R @ np.array([X, Y, Z]) + t
            if pt3d[2] <= 0: continue
            u2 = int(fx * pt3d[0] / pt3d[2] + cx)
            v2 = int(fy * pt3d[1] / pt3d[2] + cy)
            if 0 <= u2 < w and 0 <= v2 < h:
                if pt3d[2] < zbuf[v2, u2]:
                    zbuf[v2, u2] = pt3d[2]
                    result[v2, u2] = img[v, u]
    return result


def inverse_warping(img, depth_target, K, R, t):
    """
    Inverse warping: untuk setiap piksel di target, cari di source.
    Tidak ada holes, tapi perlu depth di target.
    """
    h, w = img.shape[:2]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    result = np.zeros_like(img)
    R_inv = R.T; t_inv = -R.T @ t
    
    for v in range(h):
        for u in range(w):
            Z = depth_target[v, u]
            if Z <= 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            pt = R_inv @ np.array([X, Y, Z]) + t_inv
            if pt[2] <= 0: continue
            us = int(fx * pt[0] / pt[2] + cx)
            vs = int(fy * pt[1] / pt[2] + cy)
            if 0 <= us < w and 0 <= vs < h:
                result[v, u] = img[vs, us]
    return result


def main():
    """Fungsi utama: image warping dengan depth."""
    print("=" * 60)
    print("PERCOBAAN 8: IMAGE WARPING DENGAN DEPTH")
    print("=" * 60)
    
    img = load_gambar("indoor_scene.jpg")
    if img is None: img = load_gambar("multiview_00.png")
    if img is None:
        img = np.random.randint(50, 200, (200, 250, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
        cv2.circle(img, (200, 100), 30, (0, 0, 255), -1)
    
    img = cv2.resize(img, (200, 150))  # kecilkan agar cepat
    h, w = img.shape[:2]
    depth = np.ones((h, w), dtype=np.float32) * 3.0
    depth[30:120, 40:160] = 1.5
    
    K = np.float64([[200, 0, w/2], [0, 200, h/2], [0, 0, 1]])
    R = np.eye(3)
    t = np.array([0.1, 0, 0])  # translasi kecil ke kanan
    
    print("  Forward warping...")
    fwd = forward_warping(img, depth, K, R, t)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("Source"); axes[0].axis('off')
    axes[1].imshow(depth, cmap='plasma'); axes[1].set_title("Depth Map"); axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(fwd, cv2.COLOR_BGR2RGB)); axes[2].set_title("Forward Warped"); axes[2].axis('off')
    plt.suptitle("Image Warping dengan Depth", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_warping.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/08_warping.png")


if __name__ == "__main__":
    main()
