"""
==========================================================================
PERCOBAAN 9: SGBM DISPARITY
==========================================================================
Menghitung disparity menggunakan StereoSGBM (Semi-Global Block Matching).

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


def hitung_disparity_sgbm(img_l, img_r, num_disp=64, block_size=5):
    """
    StereoSGBM: Semi-Global Block Matching.
    Lebih akurat dari BM karena mempertimbangkan konsistensi global.
    """
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if len(img_l.shape)==3 else img_l
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if len(img_r.shape)==3 else img_r
    
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block_size,
        P1=8*3*block_size**2, P2=32*3*block_size**2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
    disparity = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
    return disparity


def main():
    """Fungsi utama: SGBM disparity."""
    print("=" * 60)
    print("PERCOBAAN 9: SGBM DISPARITY")
    print("=" * 60)
    
    img_l = load_gambar("stereo_left.png")
    img_r = load_gambar("stereo_right.png")
    if img_l is None:
        img_l = np.random.randint(80, 200, (400, 500, 3), dtype=np.uint8)
        for i in range(10): cv2.rectangle(img_l, (100+i*30,100), (130+i*30,300), (255-i*25,i*25,100), -1)
        img_r = np.roll(img_l, -20, axis=1)
    
    disp_sgbm = hitung_disparity_sgbm(img_l, img_r)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)); axes[0].set_title("Kiri"); axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)); axes[1].set_title("Kanan"); axes[1].axis('off')
    axes[2].imshow(disp_sgbm, cmap='jet'); axes[2].set_title("SGBM Disparity"); axes[2].axis('off')
    plt.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046)
    plt.suptitle("StereoSGBM Disparity Map", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_sgbm_disparity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/09_sgbm_disparity.png")


if __name__ == "__main__":
    main()
