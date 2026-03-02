"""
==========================================================================
PERCOBAAN 2: FUNDAMENTAL MATRIX
==========================================================================
Menghitung Fundamental Matrix dari korespondensi titik menggunakan 8-point algorithm.

Referensi: Szeliski Ch.11, Hartley & Zisserman
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


def buat_pasangan_titik():
    """Membuat pasangan titik koresponden sintetis."""
    np.random.seed(42)
    pts1 = np.float32([[100,50],[200,80],[150,200],[300,150],[250,300],
                        [50,250],[350,50],[180,350],[400,200],[120,120]])
    # Simulasi transformasi (translasi + sedikit perspektif)
    H = np.float32([[1.0, 0.05, -30], [-0.02, 1.0, 10], [0.0001, 0, 1]])
    pts2 = []
    for p in pts1:
        pp = H @ np.array([p[0], p[1], 1.0])
        pts2.append([pp[0]/pp[2] + np.random.normal(0,1), pp[1]/pp[2] + np.random.normal(0,1)])
    return pts1, np.float32(pts2)


def hitung_fundamental_8point(pts1, pts2):
    """
    Menghitung Fundamental Matrix menggunakan 8-point algorithm.
    F memenuhi: x'^T F x = 0 untuk semua korespondensi.
    """
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    return F, mask


def gambar_epipolar_lines(img1, img2, pts1, pts2, F):
    """Menggambar epipolar lines pada kedua gambar."""
    h, w = img1.shape[:2]
    # Epipolar lines pada gambar 2 dari titik di gambar 1
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F).reshape(-1,3)
    result1, result2 = img1.copy(), img2.copy()
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
              (0,255,255),(128,0,0),(0,128,0),(0,0,128),(128,128,0)]
    
    for i, (line, p1, p2) in enumerate(zip(lines2, pts1.astype(int), pts2.astype(int))):
        a, b, c = line
        x0, x1_val = 0, w
        y0 = int(-c/b) if abs(b) > 1e-6 else 0
        y1 = int(-(c+a*w)/b) if abs(b) > 1e-6 else h
        color = colors[i % len(colors)]
        cv2.line(result2, (x0,y0), (x1_val,y1), color, 1)
        cv2.circle(result1, tuple(p1), 5, color, -1)
        cv2.circle(result2, tuple(p2), 5, color, -1)
    return result1, result2


def main():
    """Fungsi utama: fundamental matrix."""
    print("=" * 60)
    print("PERCOBAAN 2: FUNDAMENTAL MATRIX")
    print("=" * 60)
    
    img_l = load_gambar("stereo_left.png")
    img_r = load_gambar("stereo_right.png")
    if img_l is None: img_l = np.ones((400, 500, 3), dtype=np.uint8) * 200
    if img_r is None: img_r = np.ones((400, 500, 3), dtype=np.uint8) * 180
    
    print("\n--- 1. Pasangan Titik ---")
    pts1, pts2 = buat_pasangan_titik()
    print(f"  Jumlah korespondensi: {len(pts1)}")
    
    print("\n--- 2. Hitung Fundamental Matrix ---")
    F, mask = hitung_fundamental_8point(pts1, pts2)
    if F is not None:
        print(f"  F =\n{F}")
        print(f"  Rank F: {np.linalg.matrix_rank(F)}")
        print(f"  Det F: {np.linalg.det(F):.6f}")
    
    print("\n--- 3. Epipolar Lines ---")
    res1, res2 = gambar_epipolar_lines(img_l, img_r, pts1, pts2, F)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)); axes[0].set_title("Gambar Kiri + Titik"); axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)); axes[1].set_title("Gambar Kanan + Epipolar Lines"); axes[1].axis('off')
    plt.suptitle("Fundamental Matrix & Epipolar Lines", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_fundamental.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/02_fundamental.png")
    
    cv2.imshow("Epipolar Lines", np.hstack([res1, res2]))
    cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
