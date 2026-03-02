"""
==========================================================================
PERCOBAAN 5: TRIANGULASI TITIK 3D
==========================================================================
Triangulasi: menghitung posisi 3D dari korespondensi 2D pada dua kamera.

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


def setup_stereo_cameras():
    """Setup dua kamera stereo sintetis."""
    K = np.float64([[500, 0, 250], [0, 500, 200], [0, 0, 1]])
    R1 = np.eye(3); t1 = np.zeros((3, 1))
    R2 = np.eye(3); t2 = np.array([[-0.5], [0], [0]])  # baseline 0.5
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])
    return K, P1, P2


def triangulasi(P1, P2, pts1, pts2):
    """
    Triangulasi menggunakan cv2.triangulatePoints.
    Input: projection matrices P1, P2 dan korespondensi 2D.
    Output: titik 3D (homogeneous → euclidean).
    """
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = pts4d[:3] / pts4d[3]
    return pts3d.T


def main():
    """Fungsi utama: triangulasi titik 3D."""
    print("=" * 60)
    print("PERCOBAAN 5: TRIANGULASI TITIK 3D")
    print("=" * 60)
    
    K, P1, P2 = setup_stereo_cameras()
    
    # Buat titik 3D ground truth
    np.random.seed(42)
    pts_3d_gt = np.random.uniform(-2, 2, (15, 3))
    pts_3d_gt[:, 2] = np.random.uniform(3, 8, 15)  # z positif (depan kamera)
    
    # Proyeksi ke 2D
    pts1, pts2 = [], []
    for p in pts_3d_gt:
        ph = np.append(p, 1)
        p1 = P1 @ ph; p1 = p1[:2]/p1[2]
        p2 = P2 @ ph; p2 = p2[:2]/p2[2]
        pts1.append(p1); pts2.append(p2)
    pts1 = np.float64(pts1); pts2 = np.float64(pts2)
    
    print(f"  Titik 3D GT: {len(pts_3d_gt)}")
    
    pts3d = triangulasi(P1, P2, pts1, pts2)
    error = np.sqrt(np.sum((pts3d - pts_3d_gt)**2, axis=1))
    print(f"  Error rata-rata: {error.mean():.4f}")
    print(f"  Error maks: {error.max():.4f}")
    
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pts_3d_gt[:,0], pts_3d_gt[:,1], pts_3d_gt[:,2], c='blue', s=40, label='Ground Truth')
    ax1.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2], c='red', s=40, marker='x', label='Triangulasi')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title("3D Points: GT vs Triangulasi"); ax1.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.bar(range(len(error)), error, color='steelblue')
    ax2.set_xlabel("Titik"); ax2.set_ylabel("Error"); ax2.set_title("Reprojection Error per Titik")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Triangulasi Titik 3D", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_triangulasi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/05_triangulasi.png")


if __name__ == "__main__":
    main()
