"""
==========================================================================
PERCOBAAN 11: DETEKSI OBJEK DENGAN HOG (HISTOGRAM OF ORIENTED GRADIENTS)
==========================================================================
Program ini mendemonstrasikan deskriptor HOG untuk deteksi objek.
HOG menghitung distribusi arah gradien dalam sel-sel gambar, menghasilkan
fitur yang kuat untuk mendeteksi bentuk objek (terutama pejalan kaki).

Konsep yang dipelajari:
- Perhitungan gradien gambar (magnitude dan orientasi)
- Histogram orientasi per sel (cell)
- Normalisasi blok untuk invariansi pencahayaan
- HOG + SVM untuk deteksi pejalan kaki

Referensi: Machine Learning for OpenCV (Beyeler), Szeliski Ch.5
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


def hitung_gradien(img_gray):
    """
    Menghitung gradien gambar (magnitude dan orientasi).
    Gradien menunjukkan arah perubahan intensitas terkuat.
    """
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi % 180
    return magnitude, orientation, gx, gy


def visualisasi_gradien(img_gray, magnitude, orientation, gx, gy):
    """
    Menampilkan komponen gradien: Gx, Gy, magnitude, orientasi.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title("Gambar Asli")
    axes[0, 1].imshow(gx, cmap='RdBu_r')
    axes[0, 1].set_title("Gradien X (Horizontal)")
    axes[0, 2].imshow(gy, cmap='RdBu_r')
    axes[0, 2].set_title("Gradien Y (Vertikal)")
    axes[1, 0].imshow(magnitude, cmap='hot')
    axes[1, 0].set_title("Magnitude Gradien")
    axes[1, 1].imshow(orientation, cmap='hsv')
    axes[1, 1].set_title("Orientasi Gradien")

    # Histogram orientasi keseluruhan
    axes[1, 2].hist(orientation.ravel(), bins=18, range=(0, 180),
                     color='steelblue', edgecolor='black')
    axes[1, 2].set_title("Histogram Orientasi")
    axes[1, 2].set_xlabel("Sudut (derajat)")
    axes[1, 2].set_ylabel("Frekuensi")

    for ax in axes.flat[:5]:
        ax.axis('off')
    plt.suptitle("Percobaan 11: Komponen HOG (Gradien)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_hog_gradien.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/11_hog_gradien.png")


def hitung_dan_visualisasi_hog(img_gray):
    """
    Menghitung HOG descriptor menggunakan OpenCV dan memvisualisasikannya.
    HOG descriptor digunakan sebagai input untuk classifier (SVM).
    """
    img_resized = cv2.resize(img_gray, (128, 256))

    # Parameter HOG standar untuk pedestrian detection
    win_size = (128, 256)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    n_bins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
    descriptor = hog.compute(img_resized)
    print(f"  Ukuran HOG descriptor: {descriptor.shape}")

    # Visualisasi manual: hitung orientasi per sel
    h, w = img_resized.shape
    cell_h, cell_w = cell_size
    n_cells_y, n_cells_x = h // cell_h, w // cell_w

    gx = cv2.Sobel(img_resized, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_resized, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi % 180

    # Buat visualisasi HOG
    hog_image = np.zeros_like(img_resized, dtype=np.float32)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y1, y2 = cy * cell_h, (cy + 1) * cell_h
            x1, x2 = cx * cell_w, (cx + 1) * cell_w
            cell_mag = magnitude[y1:y2, x1:x2]
            cell_ori = orientation[y1:y2, x1:x2]
            # Orientasi dominan
            dom_ori = np.average(cell_ori, weights=cell_mag + 1e-5)
            avg_mag = np.mean(cell_mag)
            # Gambar garis representasi
            cx_center = (x1 + x2) // 2
            cy_center = (y1 + y2) // 2
            dx = int(avg_mag * 0.1 * np.cos(np.radians(dom_ori)))
            dy = int(avg_mag * 0.1 * np.sin(np.radians(dom_ori)))
            cv2.line(hog_image, (cx_center - dx, cy_center - dy),
                     (cx_center + dx, cy_center + dy), 255, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    ax1.imshow(img_resized, cmap='gray')
    ax1.set_title("Gambar Input")
    ax1.axis('off')
    ax2.imshow(hog_image, cmap='hot')
    ax2.set_title(f"HOG Visualization\nDescriptor: {descriptor.shape[0]} dim")
    ax2.axis('off')

    plt.suptitle("HOG Descriptor", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_hog_descriptor.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/11_hog_descriptor.png")
    return descriptor


def deteksi_pedestrian(img):
    """
    Deteksi pejalan kaki menggunakan HOG + SVM bawaan OpenCV.
    cv2.HOGDescriptor_getDefaultPeopleDetector() berisi SVM pre-trained.
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    img_resized = cv2.resize(img, (640, 480))
    boxes, weights = hog.detectMultiScale(img_resized, winStride=(8, 8),
                                           padding=(4, 4), scale=1.05)

    img_det = img_resized.copy()
    for (x, y, w, h), weight in zip(boxes, weights):
        cv2.rectangle(img_det, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_det, f"{weight[0]:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print(f"  Jumlah deteksi pedestrian: {len(boxes)}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "11_pedestrian_detection.jpg"), img_det)
    print("  [SAVED] output/11_pedestrian_detection.jpg")
    return img_det


def main():
    """Fungsi utama: deteksi objek HOG."""
    print("=" * 60)
    print("PERCOBAAN 11: DETEKSI OBJEK DENGAN HOG")
    print("=" * 60)

    img = load_gambar("pedestrian.jpg")
    if img is None:
        img = load_gambar("kucing.jpg")
    if img is None:
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("\n--- 1. Komponen Gradien ---")
    mag, ori, gx, gy = hitung_gradien(cv2.resize(img_gray, (224, 224)))
    visualisasi_gradien(cv2.resize(img_gray, (224, 224)), mag, ori, gx, gy)

    print("\n--- 2. HOG Descriptor ---")
    hitung_dan_visualisasi_hog(img_gray)

    print("\n--- 3. Deteksi Pedestrian (HOG+SVM) ---")
    img_det = deteksi_pedestrian(img)

    cv2.imshow("HOG Pedestrian Detection", cv2.resize(img_det, (640, 480)))
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 11")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. HOG menghitung distribusi orientasi gradien per sel
2. Normalisasi blok memberikan invariansi terhadap pencahayaan
3. HOG descriptor berukuran ribuan dimensi
4. cv2.HOGDescriptor menyediakan pedestrian detector pre-trained
5. HOG+SVM adalah baseline kuat sebelum era deep learning

Output: output/11_hog_gradien.png, output/11_hog_descriptor.png,
        output/11_pedestrian_detection.jpg
""")


if __name__ == "__main__":
    main()
