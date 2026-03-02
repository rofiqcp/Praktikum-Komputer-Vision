"""
==========================================================================
PERCOBAAN 12: OBJECT CLASSIFICATION DENGAN BAG OF VISUAL WORDS
==========================================================================
Klasifikasi objek menggunakan pendekatan Bag of Visual Words (BoVW) dengan SIFT/ORB.

Referensi: Learning OpenCV, Szeliski
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


def ekstrak_fitur_orb(images):
    """
    Mengekstrak fitur ORB dari kumpulan gambar.
    ORB = Oriented FAST + Rotated BRIEF (alternatif gratis untuk SIFT).
    """
    orb = cv2.ORB_create(nfeatures=200)
    all_descriptors = []
    per_image_desc = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        kp, desc = orb.detectAndCompute(gray, None)
        if desc is not None:
            all_descriptors.append(desc)
            per_image_desc.append(desc)
        else:
            per_image_desc.append(np.zeros((1, 32), dtype=np.uint8))
    return all_descriptors, per_image_desc


def buat_kamus_visual(all_descriptors, k=20):
    """
    Membuat visual dictionary menggunakan K-Means clustering.
    Setiap cluster center = 1 "visual word".
    """
    descs = np.vstack(all_descriptors).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(descs, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centers


def hitung_histogram_bovw(descriptors, centers):
    """
    Menghitung histogram BoVW: assign setiap descriptor ke cluster terdekat,
    lalu hitung frekuensi per cluster → histogram.
    """
    k = len(centers)
    hist = np.zeros(k, dtype=np.float32)
    for desc in descriptors:
        dists = np.sqrt(np.sum((centers - desc.astype(np.float32))**2, axis=1))
        idx = np.argmin(dists)
        hist[idx] += 1
    # Normalisasi
    if np.sum(hist) > 0:
        hist /= np.sum(hist)
    return hist


def main():
    """Fungsi utama: BoVW classification."""
    print("=" * 60)
    print("PERCOBAAN 12: OBJECT CLASSIFICATION DENGAN BoVW")
    print("=" * 60)
    
    # Buat dataset sintetis (3 kelas)
    np.random.seed(42)
    classes = ["Lingkaran", "Kotak", "Segitiga"]
    images, labels = [], []
    for cls_id in range(3):
        for _ in range(10):
            img = np.ones((100, 100, 3), dtype=np.uint8) * np.random.randint(180, 240)
            if cls_id == 0: cv2.circle(img, (50,50), 30, (np.random.randint(0,100),)*3, -1)
            elif cls_id == 1: cv2.rectangle(img, (20,20), (80,80), (np.random.randint(0,100),)*3, -1)
            else:
                pts = np.array([[50,15],[15,85],[85,85]])
                cv2.fillPoly(img, [pts], (np.random.randint(0,100),)*3)
            images.append(img)
            labels.append(cls_id)
    
    print(f"\n  Dataset: {len(images)} gambar, {len(classes)} kelas")
    
    print("\n--- 1. Ekstrak Fitur ORB ---")
    all_desc, per_desc = ekstrak_fitur_orb(images)
    print(f"  Total descriptors: {sum(len(d) for d in all_desc)}")
    
    print("\n--- 2. Buat Visual Dictionary ---")
    k = 15
    if all_desc:
        centers = buat_kamus_visual(all_desc, k=k)
        print(f"  Visual words: {k}")
        
        print("\n--- 3. BoVW Histograms ---")
        histograms = [hitung_histogram_bovw(d, centers) for d in per_desc]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for cls_id, cls_name in enumerate(classes):
            idx = [i for i, l in enumerate(labels) if l == cls_id][:3]
            for i in idx:
                axes[cls_id].bar(range(k), histograms[i], alpha=0.5)
            axes[cls_id].set_title(f"BoVW: {cls_name}")
            axes[cls_id].set_xlabel("Visual Word")
        plt.suptitle("Bag of Visual Words Histograms", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "12_bovw_histogram.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("  [SAVED] output/12_bovw_histogram.png")
    
    # Tampilkan sampel dataset
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for cls_id in range(3):
        idx = [i for i,l in enumerate(labels) if l==cls_id][:4]
        for j, i in enumerate(idx):
            axes[cls_id, j].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            axes[cls_id, j].set_title(classes[cls_id], fontsize=8)
            axes[cls_id, j].axis('off')
    plt.suptitle("Dataset Sintetis", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_bovw_dataset.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/12_bovw_dataset.png")
    print("\nRINGKASAN: BoVW mengubah gambar menjadi histogram visual words")
    print("untuk klasifikasi berbasis fitur lokal.")


if __name__ == "__main__":
    main()
