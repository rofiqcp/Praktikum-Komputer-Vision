"""
==========================================================================
PERCOBAAN 9: VIEW INTERPOLATION
==========================================================================
Interpolasi view antara dua gambar untuk novel view synthesis.

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


def interpolasi_view(img1, img2, alpha=0.5):
    """
    View interpolation sederhana: blending dua gambar.
    alpha=0: gambar 1, alpha=1: gambar 2.
    """
    return cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)


def interpolasi_optical_flow(img1, img2, alpha=0.5):
    """
    View interpolation berbasis optical flow.
    Hitung flow img1→img2, warp kedua gambar ke posisi intermediate.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    h, w = img1.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    map1 = np.stack([x + flow[:,:,0]*alpha, y + flow[:,:,1]*alpha], axis=-1)
    map2 = np.stack([x - flow[:,:,0]*(1-alpha), y - flow[:,:,1]*(1-alpha)], axis=-1)
    
    warped1 = cv2.remap(img1, map1[:,:,0], map1[:,:,1], cv2.INTER_LINEAR)
    warped2 = cv2.remap(img2, map2[:,:,0], map2[:,:,1], cv2.INTER_LINEAR)
    
    result = cv2.addWeighted(warped1, 1-alpha, warped2, alpha, 0)
    return result


def main():
    """Fungsi utama: view interpolation."""
    print("=" * 60)
    print("PERCOBAAN 9: VIEW INTERPOLATION")
    print("=" * 60)
    
    img1 = load_gambar("multiview_00.png")
    img2 = load_gambar("multiview_01.png")
    if img1 is None or img2 is None:
        img1 = np.random.randint(80, 200, (300, 400, 3), dtype=np.uint8)
        cv2.circle(img1, (100, 150), 50, (0, 0, 255), -1)
        cv2.rectangle(img1, (250, 100), (350, 200), (255, 0, 0), -1)
        img2 = img1.copy()
        M = np.float32([[1, 0, 30], [0, 1, -10]])
        img2 = cv2.warpAffine(img2, M, (400, 300))
    else:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(2, len(alphas), figsize=(3*len(alphas), 6))
    for i, a in enumerate(alphas):
        blend = interpolasi_view(img1, img2, a)
        axes[0, i].imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Blend α={a}"); axes[0, i].axis('off')
        
        flow_interp = interpolasi_optical_flow(img1, img2, a)
        axes[1, i].imshow(cv2.cvtColor(flow_interp, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Flow α={a}"); axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel("Linear Blend", fontsize=10)
    axes[1, 0].set_ylabel("Flow-based", fontsize=10)
    plt.suptitle("View Interpolation: Linear vs Optical Flow", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_view_interp.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/09_view_interp.png")


if __name__ == "__main__":
    main()
