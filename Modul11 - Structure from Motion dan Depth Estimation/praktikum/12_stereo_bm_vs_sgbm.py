"""
==========================================================================
PERCOBAAN 12: PERBANDINGAN STEREO BM VS SGBM
==========================================================================
Membandingkan kualitas dan kecepatan StereoBM vs StereoSGBM.

Referensi: Learning OpenCV
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


import time

def bandingkan_bm_sgbm(img_l, img_r):
    """Membandingkan StereoBM vs StereoSGBM."""
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if len(img_l.shape)==3 else img_l
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if len(img_r.shape)==3 else img_r
    
    bm = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    t0 = time.time()
    disp_bm = bm.compute(gray_l, gray_r).astype(np.float32) / 16.0
    t_bm = time.time() - t0
    
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5,
        P1=8*3*25, P2=32*3*25, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
    t0 = time.time()
    disp_sgbm = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
    t_sgbm = time.time() - t0
    
    return disp_bm, disp_sgbm, t_bm, t_sgbm


def main():
    """Fungsi utama: perbandingan BM vs SGBM."""
    print("=" * 60)
    print("PERCOBAAN 12: PERBANDINGAN STEREO BM VS SGBM")
    print("=" * 60)
    
    img_l = load_gambar("stereo_left.png")
    img_r = load_gambar("stereo_right.png")
    if img_l is None:
        img_l = np.random.randint(80, 200, (400, 500, 3), dtype=np.uint8)
        for _ in range(20): cv2.circle(img_l, (np.random.randint(50,450), np.random.randint(50,350)), np.random.randint(5,30), (np.random.randint(0,255),)*3, -1)
        img_r = np.roll(img_l, -15, axis=1)
    
    disp_bm, disp_sgbm, t_bm, t_sgbm = bandingkan_bm_sgbm(img_l, img_r)
    
    print(f"  StereoBM:   {t_bm*1000:.1f} ms")
    print(f"  StereoSGBM: {t_sgbm*1000:.1f} ms")
    print(f"  Speedup BM: {t_sgbm/max(t_bm,1e-6):.1f}x lebih cepat")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)); axes[0].set_title("Input Kiri"); axes[0].axis('off')
    axes[1].imshow(disp_bm, cmap='jet'); axes[1].set_title(f"BM ({t_bm*1000:.0f}ms)"); axes[1].axis('off')
    axes[2].imshow(disp_sgbm, cmap='jet'); axes[2].set_title(f"SGBM ({t_sgbm*1000:.0f}ms)"); axes[2].axis('off')
    plt.suptitle("Perbandingan StereoBM vs StereoSGBM", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_bm_vs_sgbm.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/12_bm_vs_sgbm.png")


if __name__ == "__main__":
    main()
