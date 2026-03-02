"""
==========================================================================
PERCOBAAN 20: MULTIVIEW RECONSTRUCTION PIPELINE
==========================================================================
Pipeline SfM sederhana: feature matching → F matrix → pose → triangulasi.

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


def sfm_pipeline_sederhana(img1, img2):
    """
    Pipeline SfM sederhana:
    1. Feature detection & matching
    2. Fundamental matrix
    3. Essential matrix → Pose
    4. Triangulasi → 3D points
    """
    K = np.float64([[500, 0, img1.shape[1]//2], [0, 500, img1.shape[0]//2], [0, 0, 1]])
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape)==3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape)==3 else img2
    
    orb = cv2.ORB_create(500)
    kp1, d1 = orb.detectAndCompute(gray1, None)
    kp2, d2 = orb.detectAndCompute(gray2, None)
    if d1 is None or d2 is None: return None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:50]
    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])
    
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999)
    if E is None: return None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K @ np.hstack([R, t])
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    
    return {"pts3d": pts3d, "R": R, "t": t, "n_matches": len(matches), "n_inliers": int(mask_pose.sum())}


def main():
    """Fungsi utama: multiview reconstruction pipeline."""
    print("=" * 60)
    print("PERCOBAAN 20: MULTIVIEW RECONSTRUCTION PIPELINE")
    print("=" * 60)
    
    img1 = load_gambar("stereo_left.png")
    img2 = load_gambar("stereo_right.png")
    if img1 is None:
        img1 = np.random.randint(80,200,(400,500,3), dtype=np.uint8)
        for _ in range(30): cv2.circle(img1, (np.random.randint(50,450),np.random.randint(50,350)),np.random.randint(5,25),(np.random.randint(0,255),)*3,-1)
        H = np.float32([[0.99, 0.02, -25], [-0.01, 0.99, 5], [0.0001, 0, 1]])
        img2 = cv2.warpPerspective(img1, H, (500, 400))
    
    result = sfm_pipeline_sederhana(img1, img2)
    
    if result:
        pts3d = result["pts3d"]
        print(f"  Matches: {result['n_matches']}, Inliers: {result['n_inliers']}")
        print(f"  3D Points: {len(pts3d)}")
        print(f"  Translation: {result['t'].flatten()}")
        
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(131)
        ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)); ax1.set_title("View 1"); ax1.axis('off')
        ax2 = fig.add_subplot(132)
        ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); ax2.set_title("View 2"); ax2.axis('off')
        ax3 = fig.add_subplot(133, projection='3d')
        valid = np.abs(pts3d).max(axis=1) < 50
        p = pts3d[valid]
        if len(p) > 0:
            ax3.scatter(p[:,0], p[:,1], p[:,2], c=p[:,2], cmap='viridis', s=5)
        ax3.set_title(f"3D Reconstruction ({len(p)} pts)")
        plt.suptitle("Structure from Motion Pipeline", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "20_sfm_pipeline.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("  [SAVED] output/20_sfm_pipeline.png")
    
    print("\nPIPELINE: Feature Matching → F Matrix → E Matrix → Pose → Triangulasi → 3D")


if __name__ == "__main__":
    main()
