"""
==========================================================================
PERCOBAAN 3: ESSENTIAL MATRIX DAN POSE ESTIMATION
==========================================================================
Menghitung Essential Matrix dan dekomposisi menjadi rotasi dan translasi.

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


def buat_kamera_intrinsik():
    """Membuat matriks kamera intrinsik sintetis."""
    fx, fy = 500, 500; cx, cy = 250, 200
    K = np.float64([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def hitung_essential(pts1, pts2, K):
    """
    Menghitung Essential Matrix.
    E = K'^T F K, atau langsung via cv2.findEssentialMat.
    """
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask


def dekomposisi_pose(E, pts1, pts2, K):
    """Dekomposisi E menjadi R dan t menggunakan cv2.recoverPose."""
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t


def main():
    """Fungsi utama: essential matrix dan pose."""
    print("=" * 60)
    print("PERCOBAAN 3: ESSENTIAL MATRIX DAN POSE ESTIMATION")
    print("=" * 60)
    
    K = buat_kamera_intrinsik()
    print(f"\n  K (intrinsik):\n{K}")
    
    np.random.seed(42)
    pts1 = np.float32(np.random.randint(50, 400, (20, 2)))
    H = np.float32([[0.98, 0.05, -20], [-0.03, 0.99, 10], [0.0001, 0, 1]])
    pts2 = cv2.perspectiveTransform(pts1.reshape(-1,1,2), H).reshape(-1,2)
    pts2 += np.random.normal(0, 0.5, pts2.shape).astype(np.float32)
    
    print("\n--- 1. Essential Matrix ---")
    E, mask = hitung_essential(pts1, pts2, K)
    if E is not None:
        print(f"  E =\n{E}")
    
    print("\n--- 2. Dekomposisi Pose ---")
    R, t = dekomposisi_pose(E, pts1, pts2, K)
    print(f"  Rotasi R:\n{R}")
    print(f"  Translasi t: {t.flatten()}")
    
    # Diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(pts1[:,0], pts1[:,1], c='blue', label='Gambar 1', s=30)
    axes[0].scatter(pts2[:,0], pts2[:,1], c='red', label='Gambar 2', s=30, marker='x')
    for p1, p2 in zip(pts1, pts2):
        axes[0].plot([p1[0],p2[0]], [p1[1],p2[1]], 'g-', alpha=0.3)
    axes[0].set_title("Korespondensi Titik"); axes[0].legend(); axes[0].invert_yaxis()
    
    # Visualisasi pose
    axes[1].quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='blue', label='Kamera 1')
    axes[1].quiver(t[0], t[1], R[0,2], R[1,2], angles='xy', scale_units='xy', scale=1, color='red', label='Kamera 2')
    axes[1].set_xlim(-2, 2); axes[1].set_ylim(-2, 2)
    axes[1].set_title("Relative Pose (Top View)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.suptitle("Essential Matrix & Camera Pose", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_essential_pose.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/03_essential_pose.png")


if __name__ == "__main__":
    main()
