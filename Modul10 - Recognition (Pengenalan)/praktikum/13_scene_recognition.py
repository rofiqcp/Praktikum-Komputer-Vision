"""
==========================================================================
PERCOBAAN 13: SCENE RECOGNITION
==========================================================================
Pengenalan jenis scene/pemandangan menggunakan histogram warna dan tekstur.

Referensi: Szeliski Ch.6
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
        print(f"  [WARN] Gambar {nama_file} tidak ditemukan.")
    return img


def ekstrak_fitur_warna(img, bins=32):
    """
    Mengekstrak histogram warna RGB sebagai fitur scene.
    Scene berbeda memiliki distribusi warna yang berbeda.
    """
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
        hist = hist.flatten() / (img.shape[0] * img.shape[1])
        hist_features.extend(hist)
    return np.array(hist_features)


def ekstrak_fitur_tekstur(gray, bins=16):
    """
    Mengekstrak fitur tekstur menggunakan LBP sederhana.
    LBP menangkap pola tekstur lokal yang khas per jenis scene.
    """
    h, w = gray.shape
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for y in range(1, h-1):
        for x in range(1, w-1):
            center = gray[y, x]
            val = 0
            for i, (dy, dx) in enumerate(offsets):
                if gray[y+dy, x+dx] >= center:
                    val |= (1 << i)
            lbp[y-1, x-1] = val
    hist = cv2.calcHist([lbp], [0], None, [bins], [0, 256]).flatten()
    return hist / max(hist.sum(), 1)


def klasifikasi_scene(images, labels, names, test_img):
    """Klasifikasi scene berdasarkan jarak histogram."""
    features = [ekstrak_fitur_warna(img) for img in images]
    test_feat = ekstrak_fitur_warna(test_img)
    dists = [np.sum(np.abs(f - test_feat)) for f in features]
    best_idx = np.argmin(dists)
    return names[labels[best_idx]], dists[best_idx]


def main():
    """Fungsi utama: scene recognition."""
    print("=" * 60)
    print("PERCOBAAN 13: SCENE RECOGNITION")
    print("=" * 60)
    
    scene_files = {"Pantai": "scene_pantai.jpg", "Kota": "scene_kota.jpg", "Hutan": "scene_hutan.jpg"}
    images, labels, names = [], [], list(scene_files.keys())
    
    for i, (name, fname) in enumerate(scene_files.items()):
        img = load_gambar(fname)
        if img is None:
            img = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)
            if i == 0: img[:,:] = [200, 180, 100]  # biru/pantai
            elif i == 1: img[:,:] = [120, 120, 140]  # abu/kota
            else: img[:,:] = [50, 130, 50]  # hijau/hutan
            noise = np.random.normal(0, 20, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(i)
    
    print(f"\n  Scene database: {names}")
    
    print("\n--- 1. Histogram Warna per Scene ---")
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 4*len(names)))
    for i, (name, img) in enumerate(zip(names, images)):
        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(name); axes[i, 0].axis('off')
        for c, color in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([img], [c], None, [64], [0, 256])
            axes[i, 1].plot(hist, color=color, alpha=0.7)
        axes[i, 1].set_title(f"Histogram Warna: {name}")
    plt.suptitle("Scene Recognition - Color Histogram", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_scene_histogram.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/13_scene_histogram.png")
    
    print("\n--- 2. Klasifikasi Scene ---")
    for img, name in zip(images, names):
        pred, dist = klasifikasi_scene(images, labels, names, img)
        print(f"  {name} → Prediksi: {pred} (dist={dist:.3f})")
    
    print("\nRINGKASAN: Scene recognition menggunakan histogram warna")
    print("dan tekstur untuk membedakan jenis pemandangan.")


if __name__ == "__main__":
    main()
