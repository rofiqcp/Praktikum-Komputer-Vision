"""
==========================================================================
PERCOBAAN 4: ICP REGISTRATION
==========================================================================
Iterative Closest Point: menyelaraskan dua point cloud.

Referensi: Szeliski, Open3D docs
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


def icp_simple(source, target, max_iter=50, threshold=0.01):
    """
    ICP Point-to-Point sederhana.
    1. Cari pasangan terdekat
    2. Hitung transformasi optimal (SVD)
    3. Terapkan transformasi
    4. Ulangi sampai konvergen
    """
    from scipy.spatial import cKDTree
    src = source.copy()
    R_total = np.eye(3); t_total = np.zeros(3)
    errors = []
    
    for it in range(max_iter):
        tree = cKDTree(target)
        dists, idx = tree.query(src)
        error = np.mean(dists)
        errors.append(error)
        if error < threshold: break
        
        matched = target[idx]
        src_mean = src.mean(axis=0); tgt_mean = matched.mean(axis=0)
        src_c = src - src_mean; tgt_c = matched - tgt_mean
        
        H = src_c.T @ tgt_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = tgt_mean - R @ src_mean
        
        src = (R @ src.T).T + t
        R_total = R @ R_total; t_total = R @ t_total + t
    
    return src, R_total, t_total, errors


def main():
    """Fungsi utama: ICP registration."""
    print("=" * 60)
    print("PERCOBAAN 4: ICP REGISTRATION")
    print("=" * 60)
    
    np.random.seed(42)
    target = np.random.randn(300, 3) * 0.5
    angle = np.radians(25)
    R_gt = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    t_gt = np.array([0.3, -0.2, 0.1])
    source = (R_gt @ target.T).T + t_gt
    source += np.random.normal(0, 0.01, source.shape)
    
    try:
        aligned, R_est, t_est, errors = icp_simple(source, target, max_iter=50)
        print(f"  Iterasi: {len(errors)}")
        print(f"  Error awal: {errors[0]:.4f}, akhir: {errors[-1]:.4f}")
        
        fig = plt.figure(figsize=(16, 5))
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(target[:,0], target[:,1], target[:,2], c='blue', s=3, label='Target')
        ax1.scatter(source[:,0], source[:,1], source[:,2], c='red', s=3, label='Source')
        ax1.set_title("Sebelum ICP"); ax1.legend()
        
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(target[:,0], target[:,1], target[:,2], c='blue', s=3, label='Target')
        ax2.scatter(aligned[:,0], aligned[:,1], aligned[:,2], c='green', s=3, label='Aligned')
        ax2.set_title("Setelah ICP"); ax2.legend()
        
        ax3 = fig.add_subplot(133)
        ax3.plot(errors, 'b-o', ms=3); ax3.set_xlabel("Iterasi"); ax3.set_ylabel("Mean Error")
        ax3.set_title("Konvergensi ICP"); ax3.grid(True, alpha=0.3)
        
        plt.suptitle("ICP Registration (Point-to-Point)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "04_icp.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("  [SAVED] output/04_icp.png")
    except ImportError:
        print("  [WARN] scipy diperlukan untuk ICP")


if __name__ == "__main__":
    main()
