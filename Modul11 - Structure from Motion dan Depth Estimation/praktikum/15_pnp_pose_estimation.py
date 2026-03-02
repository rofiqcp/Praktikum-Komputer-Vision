"""
==========================================================================
PERCOBAAN 15: PNP POSE ESTIMATION
==========================================================================
Estimasi pose kamera dari korespondensi 2D-3D menggunakan solvePnP.

Referensi: Szeliski
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


def pnp_estimation():
    """
    Estimasi pose kamera dari korespondensi 2D-3D menggunakan solvePnP.
    """
    K = np.float64([[500, 0, 250], [0, 500, 200], [0, 0, 1]])
    dist = np.zeros(5)
    
    # Titik 3D (objek)
    obj_pts = np.float32([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, -1], [1, 0, -1]])
    
    # Proyeksi ke 2D (dengan pose diketahui)
    rvec_gt = np.float64([[0.1], [0.2], [0.05]])
    tvec_gt = np.float64([[0.5], [0.3], [3.0]])
    img_pts, _ = cv2.projectPoints(obj_pts, rvec_gt, tvec_gt, K, dist)
    img_pts = img_pts.reshape(-1, 2)
    img_pts += np.random.normal(0, 0.5, img_pts.shape)  # noise
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts.astype(np.float64), K, dist)
    return rvec, tvec, rvec_gt, tvec_gt, obj_pts, img_pts, K


def main():
    """Fungsi utama: PnP pose estimation."""
    print("=" * 60)
    print("PERCOBAAN 15: PNP POSE ESTIMATION")
    print("=" * 60)
    
    rvec, tvec, rvec_gt, tvec_gt, obj_pts, img_pts, K = pnp_estimation()
    
    print(f"\n  Ground Truth: rvec={rvec_gt.flatten()}, tvec={tvec_gt.flatten()}")
    print(f"  Estimated:    rvec={rvec.flatten()}, tvec={tvec.flatten()}")
    print(f"  Error rvec: {np.linalg.norm(rvec-rvec_gt):.6f}")
    print(f"  Error tvec: {np.linalg.norm(tvec-tvec_gt):.6f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(img_pts[:,0], img_pts[:,1], c='red', s=50, label='2D Points')
    for i, p in enumerate(img_pts):
        axes[0].annotate(f'P{i}', (p[0]+5, p[1]+5), fontsize=8)
    axes[0].set_title("2D Image Points"); axes[0].invert_yaxis(); axes[0].legend()
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(obj_pts[:,0], obj_pts[:,1], obj_pts[:,2], c='blue', s=50, label='3D Object')
    ax2.quiver(tvec[0], tvec[1], tvec[2], 0, 0, -0.5, color='red', label='Camera (est)')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title("3D Object + Camera Pose"); ax2.legend()
    
    plt.suptitle("PnP Pose Estimation", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_pnp.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/15_pnp.png")


if __name__ == "__main__":
    main()
