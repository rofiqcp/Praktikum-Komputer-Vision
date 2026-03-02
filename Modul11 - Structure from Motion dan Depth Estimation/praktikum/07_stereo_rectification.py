"""
==========================================================================
PERCOBAAN 7: STEREO RECTIFICATION
==========================================================================
Rektifikasi stereo agar baris gambar sejajar (epipolar lines horizontal).

Referensi: Szeliski Ch.11
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


def rektifikasi_sederhana(img1, img2, pts1, pts2):
    """
    Rektifikasi stereo: transformasi agar epipolar lines horizontal.
    Menggunakan cv2.stereoRectifyUncalibrated.
    """
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    if F is None: return img1, img2
    h, w = img1.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w, h))
    rect1 = cv2.warpPerspective(img1, H1, (w, h))
    rect2 = cv2.warpPerspective(img2, H2, (w, h))
    return rect1, rect2


def main():
    """Fungsi utama: stereo rectification."""
    print("=" * 60)
    print("PERCOBAAN 7: STEREO RECTIFICATION")
    print("=" * 60)
    
    img1 = load_gambar("stereo_left.png")
    img2 = load_gambar("stereo_right.png")
    if img1 is None:
        img1 = np.random.randint(80, 200, (400, 500, 3), dtype=np.uint8)
        for _ in range(30): cv2.circle(img1, (np.random.randint(50,450), np.random.randint(50,350)), 8, (np.random.randint(0,255),)*3, -1)
        H = np.float32([[1, 0.02, -30], [-0.01, 1, 5], [0.0001, 0, 1]])
        img2 = cv2.warpPerspective(img1, H, (500, 400))
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(300)
    kp1, d1 = orb.detectAndCompute(gray1, None)
    kp2, d2 = orb.detectAndCompute(gray2, None)
    if d1 is not None and d2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:30]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        rect1, rect2 = rektifikasi_sederhana(img1, img2, pts1, pts2)
        
        # Gambar garis horizontal untuk verifikasi
        for y in range(0, rect1.shape[0], 30):
            cv2.line(rect1, (0, y), (rect1.shape[1], y), (0, 255, 0), 1)
            cv2.line(rect2, (0, y), (rect2.shape[1], y), (0, 255, 0), 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0,0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)); axes[0,0].set_title("Asli Kiri")
        axes[0,1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); axes[0,1].set_title("Asli Kanan")
        axes[1,0].imshow(cv2.cvtColor(rect1, cv2.COLOR_BGR2RGB)); axes[1,0].set_title("Rektifikasi Kiri")
        axes[1,1].imshow(cv2.cvtColor(rect2, cv2.COLOR_BGR2RGB)); axes[1,1].set_title("Rektifikasi Kanan")
        for ax in axes.flat: ax.axis('off')
        plt.suptitle("Stereo Rectification", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "07_rectification.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("  [SAVED] output/07_rectification.png")
        cv2.imshow("Rectified", np.hstack([rect1, rect2]))
        cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
