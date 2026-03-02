"""
==========================================================================
PERCOBAAN 8: BLOCK MATCHING DISPARITY
==========================================================================
Menghitung disparity map menggunakan StereoBM (Block Matching).

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


def hitung_disparity_bm(img_l, img_r, num_disp=64, block_size=15):
    """
    Menghitung disparity menggunakan StereoBM.
    numDisparities: range pencarian (kelipatan 16).
    blockSize: ukuran window matching (ganjil).
    """
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if len(img_l.shape)==3 else img_l
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if len(img_r.shape)==3 else img_r
    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    return disparity


def main():
    """Fungsi utama: Block Matching disparity."""
    print("=" * 60)
    print("PERCOBAAN 8: BLOCK MATCHING DISPARITY")
    print("=" * 60)
    
    img_l = load_gambar("stereo_left.png")
    img_r = load_gambar("stereo_right.png")
    if img_l is None:
        img_l = np.random.randint(80, 200, (400, 500, 3), dtype=np.uint8)
        for i in range(10):
            d = np.random.randint(5, 50)
            cv2.rectangle(img_l, (100+i*30, 100), (130+i*30, 300), (255-d*5, d*5, 100), -1)
        img_r = img_l.copy()
        img_r = np.roll(img_r, -20, axis=1)
    
    params = [(64, 9), (64, 15), (128, 15), (128, 21)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (nd, bs) in zip(axes.flat, params):
        disp = hitung_disparity_bm(img_l, img_r, nd, bs)
        ax.imshow(disp, cmap='jet'); ax.set_title(f"numDisp={nd}, block={bs}"); ax.axis('off')
    plt.suptitle("StereoBM Disparity Map", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_bm_disparity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/08_bm_disparity.png")
    
    disp = hitung_disparity_bm(img_l, img_r)
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("BM Disparity", cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET))
    cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
