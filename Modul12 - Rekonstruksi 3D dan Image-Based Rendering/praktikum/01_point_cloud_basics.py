"""
==========================================================================
PERCOBAAN 1: POINT CLOUD BASICS
==========================================================================
Dasar point cloud: membuat, memuat, dan memvisualisasikan data 3D.

Referensi: Szeliski Ch.13, Open3D docs
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


def buat_point_cloud_sintetis(n=500):
    """Membuat point cloud sintetis berbagai bentuk."""
    # Bola
    theta = np.random.uniform(0, 2*np.pi, n)
    phi = np.random.uniform(0, np.pi, n)
    r = 1.0
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    sphere = np.column_stack([x, y, z])
    
    # Kubus
    cube = np.random.uniform(-1, 1, (n, 3))
    
    return sphere, cube


def simpan_ply(points, filename, colors=None):
    """Menyimpan point cloud ke format PLY (ASCII)."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(points):
            line = f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
            if colors is not None:
                c = colors[i]
                line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
            f.write(line + "\n")
    print(f"  [SAVED] output/{filename}")


def baca_ply(filename):
    """Membaca point cloud dari file PLY sederhana."""
    path = os.path.join(IMAGE_DIR, filename) if os.path.exists(os.path.join(IMAGE_DIR, filename)) else os.path.join(OUTPUT_DIR, filename)
    points = []
    header_end = False
    with open(path, 'r') as f:
        for line in f:
            if header_end:
                vals = line.strip().split()
                if len(vals) >= 3:
                    points.append([float(vals[0]), float(vals[1]), float(vals[2])])
            elif "end_header" in line:
                header_end = True
    return np.array(points) if points else np.zeros((0, 3))


def main():
    """Fungsi utama: point cloud basics."""
    print("=" * 60)
    print("PERCOBAAN 1: POINT CLOUD BASICS")
    print("=" * 60)
    
    sphere, cube = buat_point_cloud_sintetis(500)
    print(f"  Sphere: {len(sphere)} titik")
    print(f"  Cube: {len(cube)} titik")
    
    colors_s = np.column_stack([np.linspace(0,255,500), np.zeros(500), np.linspace(255,0,500)])
    simpan_ply(sphere, "01_sphere.ply", colors_s.astype(int))
    simpan_ply(cube, "01_cube.ply")
    
    # Coba baca PLY dari image/
    ply_file = os.path.join(IMAGE_DIR, "bunny_point_cloud.ply")
    if os.path.exists(ply_file):
        bunny = baca_ply("bunny_point_cloud.ply")
        if len(bunny) > 0: print(f"  Bunny: {len(bunny)} titik")
    
    fig = plt.figure(figsize=(16, 5))
    for i, (pts, title) in enumerate([(sphere, "Sphere"), (cube, "Cube")]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=pts[:,2], cmap='viridis', s=3)
        ax.set_title(title); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.suptitle("Point Cloud Basics", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_point_cloud.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/01_point_cloud.png")


if __name__ == "__main__":
    main()
