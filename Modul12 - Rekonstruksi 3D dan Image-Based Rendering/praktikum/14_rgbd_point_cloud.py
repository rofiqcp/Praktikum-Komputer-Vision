"""
==========================================================================
PERCOBAAN 14: RGBD POINT CLOUD
==========================================================================
Membuat colored point cloud dari RGB + Depth (RGBD).

Referensi: Open3D docs, Kinect
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


def rgbd_to_colored_cloud(color_img, depth_img, K):
    """
    Konversi RGBD ke colored point cloud.
    Setiap piksel dengan depth valid → titik 3D berwarna.
    """
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    h, w = depth_img.shape
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = depth_img > 0
    Z = depth_img[valid].astype(np.float64)
    X = (u[valid] - cx) * Z / fx
    Y = (v[valid] - cy) * Z / fy
    
    points = np.column_stack([X, Y, Z])
    if len(color_img.shape) == 3:
        colors = color_img[valid][:, ::-1] / 255.0  # BGR → RGB
    else:
        c = color_img[valid] / 255.0
        colors = np.column_stack([c, c, c])
    return points, colors


def main():
    """Fungsi utama: RGBD point cloud."""
    print("=" * 60)
    print("PERCOBAAN 14: RGBD POINT CLOUD")
    print("=" * 60)
    
    color = load_gambar("rgbd_color_00.png")
    depth = cv2.imread(os.path.join(IMAGE_DIR, "rgbd_depth_00.png"), cv2.IMREAD_GRAYSCALE)
    
    if color is None or depth is None:
        color = np.random.randint(50, 200, (200, 250, 3), dtype=np.uint8)
        cv2.rectangle(color, (50, 50), (120, 150), (0, 255, 0), -1)
        cv2.rectangle(color, (150, 30), (220, 170), (0, 0, 255), -1)
        depth = np.zeros((200, 250), dtype=np.uint8)
        depth[50:150, 50:120] = 80    # dekat
        depth[30:170, 150:220] = 150  # jauh
        depth = cv2.GaussianBlur(depth, (5, 5), 0)
    
    K = np.float64([[300, 0, color.shape[1]/2], [0, 300, color.shape[0]/2], [0, 0, 1]])
    depth_float = depth.astype(np.float32) / 25.0  # skala ke meter
    
    points, colors = rgbd_to_colored_cloud(color, depth_float, K)
    print(f"  Points: {len(points)}")
    
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(131)
    ax1.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)); ax1.set_title("Color"); ax1.axis('off')
    ax2 = fig.add_subplot(132)
    ax2.imshow(depth, cmap='plasma'); ax2.set_title("Depth"); ax2.axis('off')
    ax3 = fig.add_subplot(133, projection='3d')
    idx = np.random.choice(len(points), min(3000, len(points)), replace=False) if len(points) > 3000 else np.arange(len(points))
    ax3.scatter(points[idx,0], points[idx,1], points[idx,2], c=colors[idx], s=2)
    ax3.set_title("Colored Point Cloud"); ax3.set_xlabel('X'); ax3.set_ylabel('Y')
    
    plt.suptitle("RGBD → Colored Point Cloud", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "14_rgbd_cloud.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/14_rgbd_cloud.png")


if __name__ == "__main__":
    main()
