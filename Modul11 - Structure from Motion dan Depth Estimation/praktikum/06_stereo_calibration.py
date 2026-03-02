"""
==========================================================================
PERCOBAAN 6: STEREO CALIBRATION
==========================================================================
Kalibrasi kamera stereo: intrinsik dan ekstrinsik dari checkerboard.

Referensi: Szeliski Ch.11, Learning OpenCV
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


def buat_checkerboard_sintetis(rows=6, cols=8, size=40):
    """Membuat gambar checkerboard sintetis untuk kalibrasi."""
    h, w = (rows+1)*size, (cols+1)*size
    img = np.ones((h, w), dtype=np.uint8) * 255
    for i in range(rows+1):
        for j in range(cols+1):
            if (i+j) % 2 == 0:
                img[i*size:(i+1)*size, j*size:(j+1)*size] = 0
    return img


def simulasi_kalibrasi(n_images=5):
    """Simulasi proses kalibrasi stereo."""
    rows, cols = 6, 8
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    
    obj_points, img_points_l, img_points_r = [], [], []
    for _ in range(n_images):
        pts = objp[:, :2] * 40 + np.array([50, 50])
        noise_l = np.random.normal(0, 0.5, pts.shape)
        noise_r = np.random.normal(0, 0.5, pts.shape)
        pts_r = pts + np.array([-30, 0])
        obj_points.append(objp)
        img_points_l.append((pts + noise_l).astype(np.float32))
        img_points_r.append((pts_r + noise_r).astype(np.float32))
    
    return obj_points, img_points_l, img_points_r


def main():
    """Fungsi utama: stereo calibration."""
    print("=" * 60)
    print("PERCOBAAN 6: STEREO CALIBRATION")
    print("=" * 60)
    
    print("\n--- 1. Checkerboard ---")
    board = buat_checkerboard_sintetis()
    cv2.imwrite(os.path.join(OUTPUT_DIR, "06_checkerboard.png"), board)
    print("  [SAVED] output/06_checkerboard.png")
    
    print("\n--- 2. Simulasi Kalibrasi ---")
    obj_pts, img_l, img_r = simulasi_kalibrasi()
    print(f"  {len(obj_pts)} pasang gambar kalibrasi")
    
    K = np.float64([[500, 0, 250], [0, 500, 200], [0, 0, 1]])
    R = np.eye(3)
    T = np.array([[-60.0], [0], [0]])
    print(f"  Baseline simulasi: {abs(T[0,0]):.1f} piksel")
    print(f"  K (intrinsik):\n{K}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(board, cmap='gray'); axes[0].set_title("Checkerboard"); axes[0].axis('off')
    for pts in img_l[:3]:
        axes[1].scatter(pts[:,0], pts[:,1], s=10)
    axes[1].set_title("Corner Detections (3 views)"); axes[1].invert_yaxis()
    plt.suptitle("Stereo Calibration", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_stereo_calibration.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/06_stereo_calibration.png")
    
    cv2.imshow("Checkerboard", board)
    cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
