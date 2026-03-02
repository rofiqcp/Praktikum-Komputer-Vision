"""
==========================================================================
PERCOBAAN 4: EPIPOLAR LINES
==========================================================================
Menggambar dan menganalisis epipolar lines pada pasangan stereo.

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


def deteksi_dan_match(img1, img2):
    """Deteksi fitur ORB dan match dengan BFMatcher."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape)==3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape)==3 else img2
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None: return np.array([]), np.array([])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:30]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def gambar_epipolar(img1, img2, pts1, pts2):
    """Menggambar epipolar lines menggunakan fundamental matrix."""
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    if F is None: return img1, img2
    pts1 = pts1[mask.ravel()==1]; pts2 = pts2[mask.ravel()==1]
    
    lines = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F).reshape(-1,3)
    h, w = img1.shape[:2]
    res1, res2 = img1.copy(), img2.copy()
    for line, p1, p2 in zip(lines, pts1.astype(int), pts2.astype(int)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        a, b, c = line
        x0, x1_val = 0, w
        y0 = int(-c/b) if abs(b) > 1e-6 else 0
        y1 = int(-(c+a*w)/b) if abs(b) > 1e-6 else h
        cv2.line(res1, (x0,y0), (x1_val,y1), color, 1)
        cv2.circle(res1, tuple(p1), 4, color, -1)
        cv2.circle(res2, tuple(p2), 4, color, -1)
    return res1, res2


def main():
    """Fungsi utama: epipolar lines."""
    print("=" * 60)
    print("PERCOBAAN 4: EPIPOLAR LINES")
    print("=" * 60)
    
    img1 = load_gambar("stereo_left.png")
    img2 = load_gambar("stereo_right.png")
    if img1 is None:
        img1 = np.random.randint(100, 200, (400, 500, 3), dtype=np.uint8)
        for _ in range(20): cv2.circle(img1, (np.random.randint(50,450), np.random.randint(50,350)), 10, (np.random.randint(0,255),)*3, -1)
        H = np.float32([[1, 0.02, -25], [-0.01, 1, 5], [0.00005, 0, 1]])
        img2 = cv2.warpPerspective(img1, H, (500, 400))
    
    pts1, pts2 = deteksi_dan_match(img1, img2)
    print(f"  Matches: {len(pts1)}")
    
    if len(pts1) >= 8:
        res1, res2 = gambar_epipolar(img1, img2, pts1, pts2)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)); axes[0].set_title("Kiri + Epipolar Lines"); axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)); axes[1].set_title("Kanan + Titik"); axes[1].axis('off')
        plt.suptitle("Epipolar Lines dari Feature Matching", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "04_epipolar_lines.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("  [SAVED] output/04_epipolar_lines.png")
        cv2.imshow("Epipolar Lines", np.hstack([res1, res2]))
        cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
