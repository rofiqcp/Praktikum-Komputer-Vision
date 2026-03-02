"""
==========================================================================
PERCOBAAN 13: MESH TEXTURING
==========================================================================
Proyeksi tekstur dari gambar ke mesh 3D.

Referensi: Szeliski, OpenCV docs
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


def tekstur_dari_proyeksi(img, vertices, K, rvec, tvec):
    """
    Memetakan warna dari gambar ke vertices 3D via proyeksi.
    Setiap vertex 3D diproyeksikan ke 2D → ambil warna piksel tersebut.
    """
    dist = np.zeros(5)
    img_pts, _ = cv2.projectPoints(vertices, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1, 2).astype(int)
    h, w = img.shape[:2]
    colors = np.zeros((len(vertices), 3), dtype=np.uint8)
    for i, (u, v) in enumerate(img_pts):
        if 0 <= u < w and 0 <= v < h:
            colors[i] = img[v, u]
    return colors


def main():
    """Fungsi utama: mesh texturing."""
    print("=" * 60)
    print("PERCOBAAN 13: MESH TEXTURING")
    print("=" * 60)
    
    img = load_gambar("multiview_00.png")
    if img is None:
        img = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), -1)
        cv2.circle(img, (300, 150), 50, (0, 0, 255), -1)
    
    # Vertices 3D (grid planar)
    x = np.linspace(-1, 1, 20); y = np.linspace(-1, 1, 15)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) + 3.0
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
    
    K = np.float64([[300, 0, img.shape[1]/2], [0, 300, img.shape[0]/2], [0, 0, 1]])
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.array([0, 0, 0], dtype=np.float64)
    
    colors = tekstur_dari_proyeksi(img, vertices, K, rvec, tvec)
    
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(131)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax1.set_title("Tekstur Source"); ax1.axis('off')
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c='gray', s=5)
    ax2.set_title("Mesh (untextured)"); ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    
    ax3 = fig.add_subplot(133, projection='3d')
    c_rgb = colors[:, ::-1] / 255.0  # BGR → RGB
    ax3.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c=c_rgb, s=20)
    ax3.set_title("Mesh (textured)"); ax3.set_xlabel('X'); ax3.set_ylabel('Y')
    
    plt.suptitle("Mesh Texturing via Projection", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_texturing.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/13_texturing.png")


if __name__ == "__main__":
    main()
