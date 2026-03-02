"""
==========================================================================
PERCOBAAN 18: LIGHT FIELD BASICS
==========================================================================
Dasar Light Field: parameterisasi 4D dari sinar cahaya.

Referensi: Szeliski Ch.14, Levoy & Hanrahan
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


def buat_light_field_sintetis(n_views=5, img_size=100):
    """
    Buat light field sintetis: grid kamera yang mengambil gambar dari posisi berbeda.
    """
    views = []
    for i in range(n_views):
        for j in range(n_views):
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            # Objek bergeser berdasarkan posisi kamera
            cx = 50 + (i - n_views//2) * 5
            cy = 50 + (j - n_views//2) * 5
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), -1)
            cv2.rectangle(img, (cx+20, cy-10), (cx+40, cy+10), (0, 255, 0), -1)
            img[:, :, 0] = np.clip(img[:, :, 0] + 80 + i*10, 0, 255)  # background bervariasi
            views.append(img)
    return views, n_views


def epipolar_plane_image(views, n_views, row=50):
    """
    Buat Epipolar Plane Image (EPI): ambil satu baris dari setiap view horizontal.
    """
    epi = np.zeros((n_views, views[0].shape[1], 3), dtype=np.uint8)
    for i in range(n_views):
        view_idx = i * n_views + n_views // 2  # baris tengah
        if view_idx < len(views):
            epi[i] = views[view_idx][row]
    return epi


def main():
    """Fungsi utama: light field basics."""
    print("=" * 60)
    print("PERCOBAAN 18: LIGHT FIELD BASICS")
    print("=" * 60)
    
    views, n = buat_light_field_sintetis(5, 100)
    print(f"  Light field: {n}×{n} = {len(views)} views")
    
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    for i in range(n):
        for j in range(n):
            axes[i, j].imshow(cv2.cvtColor(views[i*n+j], cv2.COLOR_BGR2RGB))
            axes[i, j].axis('off')
    plt.suptitle(f"Light Field Grid ({n}×{n} views)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_light_field.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/18_light_field.png")
    
    epi = epipolar_plane_image(views, n, 50)
    fig2, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(cv2.cvtColor(cv2.resize(epi, (400, 100)), cv2.COLOR_BGR2RGB))
    ax.set_title("Epipolar Plane Image (EPI) - row 50"); ax.set_xlabel("x"); ax.set_ylabel("view")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_epi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/18_epi.png")


if __name__ == "__main__":
    main()
