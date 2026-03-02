"""
==========================================================================
PERCOBAAN 13: WLS FILTER DISPARITY
==========================================================================
Meningkatkan kualitas disparity map menggunakan WLS (Weighted Least Squares) filter.

Referensi: OpenCV ximgproc
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


def wls_filter_simulasi(disp_l, img_l):
    """
    WLS Filter: menghaluskan disparity dengan mempertahankan edge.
    Jika ximgproc tersedia, gunakan. Jika tidak, gunakan bilateral filter.
    """
    try:
        wls = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        wls.setLambda(8000); wls.setSigmaColor(1.5)
        filtered = wls.filter(disp_l.astype(np.int16), img_l)
        return filtered.astype(np.float32) / 16.0, "WLS Filter"
    except:
        gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if len(img_l.shape)==3 else img_l
        disp_norm = cv2.normalize(disp_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        filtered = cv2.ximgproc.guidedFilter(gray, disp_norm, 9, 75) if hasattr(cv2, 'ximgproc') else cv2.bilateralFilter(disp_norm, 9, 75, 75)
        return filtered.astype(np.float32), "Bilateral Filter (fallback)"


def main():
    """Fungsi utama: WLS filter disparity."""
    print("=" * 60)
    print("PERCOBAAN 13: WLS FILTER DISPARITY")
    print("=" * 60)
    
    img_l = load_gambar("stereo_left.png")
    img_r = load_gambar("stereo_right.png")
    if img_l is None:
        img_l = np.random.randint(80,200,(400,500,3), dtype=np.uint8)
        for _ in range(20): cv2.circle(img_l, (np.random.randint(50,450),np.random.randint(50,350)), np.random.randint(5,30), (np.random.randint(0,255),)*3, -1)
        img_r = np.roll(img_l, -15, axis=1)
    
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    bm = cv2.StereoBM_create(64, 15)
    disp_raw = bm.compute(gray_l, gray_r).astype(np.float32) / 16.0
    
    filtered, method = wls_filter_simulasi(disp_raw, img_l)
    print(f"  Metode: {method}")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)); axes[0].set_title("Input"); axes[0].axis('off')
    axes[1].imshow(disp_raw, cmap='jet'); axes[1].set_title("Disparity Raw"); axes[1].axis('off')
    axes[2].imshow(filtered, cmap='jet'); axes[2].set_title(f"Disparity Filtered ({method})"); axes[2].axis('off')
    plt.suptitle("WLS Filter Disparity", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_wls_filter.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/13_wls_filter.png")


if __name__ == "__main__":
    main()
