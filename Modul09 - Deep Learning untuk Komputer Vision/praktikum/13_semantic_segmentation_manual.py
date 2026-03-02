"""
==========================================================================
PERCOBAAN 13: SEMANTIC SEGMENTATION (MANUAL)
==========================================================================
Program ini mendemonstrasikan konsep semantic segmentation secara manual.
Semantic segmentation memberikan label kelas pada SETIAP piksel gambar.
Berbeda dengan deteksi objek yang hanya memberi bounding box.

Konsep yang dipelajari:
- Segmentasi berbasis warna (thresholding HSV)
- Segmentasi berbasis clustering (K-Means)
- Mask overlay untuk visualisasi segmentasi
- Perbandingan: Detection vs Semantic Segmentation

Referensi: Szeliski Ch.5, Deep Learning for CV (Rosebrock)
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
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
        print(f"  [ERROR] {nama_file} tidak ditemukan!")
    return img


def segmentasi_warna_hsv(img, lower_hsv, upper_hsv):
    """
    Segmentasi berbasis warna di ruang HSV.
    Menghasilkan mask biner: 255 untuk piksel yang masuk range, 0 lainnya.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    # Bersihkan noise dengan morfologi
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def segmentasi_kmeans(img, k=5):
    """
    Segmentasi berbasis K-Means clustering.
    Mengelompokkan piksel berdasarkan warna menjadi k cluster.
    Setiap cluster mewakili satu 'kelas' segmentasi.
    """
    pixel_data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixel_data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)
    label_map = labels.reshape(img.shape[:2])
    return segmented, label_map, centers


def buat_mask_overlay(img, mask, color=(0, 255, 0), alpha=0.4):
    """
    Membuat overlay mask berwarna di atas gambar asli.
    alpha mengontrol transparansi overlay.
    """
    overlay = img.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return blended


def visualisasi_segmentasi_warna(img):
    """
    Melakukan segmentasi warna untuk beberapa range HSV.
    Mensegmentasi: langit (biru), tanaman (hijau), objek terang.
    """
    segmen = {
        "Biru/Langit": ([90, 50, 50], [130, 255, 255], (255, 0, 0)),
        "Hijau/Tanaman": ([35, 50, 50], [85, 255, 255], (0, 255, 0)),
        "Merah/Objek": ([0, 50, 50], [10, 255, 255], (0, 0, 255)),
        "Kuning": ([20, 50, 50], [35, 255, 255], (0, 255, 255)),
    }

    n = len(segmen) + 1
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Gambar Asli")
    axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Gambar Asli")

    for i, (nama, (lo, hi, color)) in enumerate(segmen.items()):
        mask = segmentasi_warna_hsv(img, lo, hi)
        overlay = buat_mask_overlay(img, mask, color)

        axes[0, i + 1].imshow(mask, cmap='gray')
        axes[0, i + 1].set_title(f"Mask: {nama}")
        axes[1, i + 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, i + 1].set_title(f"Overlay: {nama}")

    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle("Semantic Segmentation berbasis Warna (HSV)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_segmentasi_warna.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/13_segmentasi_warna.png")


def visualisasi_segmentasi_kmeans(img, k=5):
    """Menampilkan hasil segmentasi K-Means dengan k cluster."""
    segmented, label_map, centers = segmentasi_kmeans(img, k)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Gambar Asli")
    axes[1].imshow(label_map, cmap='tab10')
    axes[1].set_title(f"Label Map (K={k})")
    axes[2].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Hasil Segmentasi K-Means")

    for ax in axes:
        ax.axis('off')
    plt.suptitle("Semantic Segmentation berbasis K-Means Clustering", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_segmentasi_kmeans.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/13_segmentasi_kmeans.png")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "13_segmented_kmeans.jpg"), segmented)
    print("  [SAVED] output/13_segmented_kmeans.jpg")
    return segmented


def main():
    """Fungsi utama: semantic segmentation manual."""
    print("=" * 60)
    print("PERCOBAAN 13: SEMANTIC SEGMENTATION (MANUAL)")
    print("=" * 60)

    img = load_gambar("scene_outdoor.jpg")
    if img is None:
        img = load_gambar("kucing.jpg")
    if img is None:
        return
    img = cv2.resize(img, (400, 300))

    print("\n--- 1. Segmentasi Warna (HSV) ---")
    visualisasi_segmentasi_warna(img)

    print("\n--- 2. Segmentasi K-Means ---")
    segmented = visualisasi_segmentasi_kmeans(img, k=5)

    cv2.imshow("Segmentasi - Percobaan 13", segmented)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 13")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. Semantic segmentation melabeli setiap piksel dalam gambar
2. Segmentasi HSV efektif untuk objek dengan warna tertentu
3. K-Means clustering mengelompokkan piksel berdasarkan kesamaan warna
4. Mask overlay membantu visualisasi hasil segmentasi
5. Metode modern (FCN, UNet, DeepLab) lebih akurat dan umum

Output: output/13_segmentasi_warna.png, output/13_segmentasi_kmeans.png
""")


if __name__ == "__main__":
    main()
