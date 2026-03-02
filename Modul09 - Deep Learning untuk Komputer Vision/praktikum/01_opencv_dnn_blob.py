"""
==========================================================================
PERCOBAAN 1: KLASIFIKASI GAMBAR DENGAN OPENCV DNN (BLOB)
==========================================================================
Program ini mempelajari cara menggunakan modul cv2.dnn untuk membuat
blob (input tensor) dari gambar. Blob adalah format input standar yang
dibutuhkan oleh neural network: tensor 4D (batch, channels, height, width).

Fungsi utama yang dipelajari:
- cv2.dnn.blobFromImage()  : Membuat blob dari satu gambar
- cv2.dnn.blobFromImages() : Membuat blob dari beberapa gambar sekaligus
- Preprocessing gambar (resize, normalisasi, mean subtraction)

Referensi: Mastering OpenCV 4, Ch.13 (DNN Module)
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import time

# Konfigurasi direktori
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_gambar(nama_file):
    """Memuat gambar dari folder image/. Mengembalikan None jika gagal."""
    path = os.path.join(IMAGE_DIR, nama_file)
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] {nama_file} tidak ditemukan! Jalankan download_image.py")
    return img


def buat_blob_standar(img):
    """
    Membuat blob standar dari gambar untuk model ImageNet.
    Blob adalah tensor 4D: (batchSize, channels, height, width).
    scalefactor=1/255 -> normalisasi piksel dari [0,255] ke [0,1]
    size=(224,224) -> ukuran input model ImageNet standar
    swapRB=True -> konversi BGR ke RGB
    """
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0 / 255.0,
        size=(224, 224),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )
    print(f"  Shape gambar asli : {img.shape}")
    print(f"  Shape blob        : {blob.shape}")
    print(f"  Tipe data blob    : {blob.dtype}")
    print(f"  Range nilai       : [{blob.min():.4f}, {blob.max():.4f}]")
    return blob


def variasi_parameter_blob(img):
    """
    Mendemonstrasikan berbagai parameter blobFromImage:
    - Tanpa normalisasi (range 0-255)
    - Dengan mean subtraction ImageNet
    - Dengan standardisasi penuh
    """
    # Blob tanpa normalisasi
    blob_raw = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (0, 0, 0), False, False)
    print(f"  Tanpa normalisasi  : [{blob_raw.min():.1f}, {blob_raw.max():.1f}]")

    # Blob dengan mean subtraction ImageNet (BGR: 103.94, 116.78, 123.68)
    blob_mean = cv2.dnn.blobFromImage(
        img, 1.0, (224, 224), (103.94, 116.78, 123.68), False, False
    )
    print(f"  Dengan mean sub    : [{blob_mean.min():.1f}, {blob_mean.max():.1f}]")

    # Blob dengan normalisasi + mean subtraction
    blob_norm = cv2.dnn.blobFromImage(
        img, 1.0 / 255.0, (224, 224), (0.485, 0.456, 0.406), True, False
    )
    print(f"  Norm + mean sub    : [{blob_norm.min():.4f}, {blob_norm.max():.4f}]")
    return blob_raw, blob_mean, blob_norm


def visualisasi_blob_channels(img, blob):
    """
    Menampilkan channel R, G, B dari blob sebagai gambar terpisah.
    Berguna untuk memahami apa yang 'dilihat' oleh neural network.
    """
    channels = blob[0]  # Ambil batch pertama -> (3, 224, 224)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Gambar Asli")
    axes[0].axis('off')

    nama_ch = [("Channel R", 'Reds'), ("Channel G", 'Greens'), ("Channel B", 'Blues')]
    for i, (nama, cmap) in enumerate(nama_ch):
        axes[i + 1].imshow(channels[i], cmap=cmap)
        axes[i + 1].set_title(nama)
        axes[i + 1].axis('off')

    plt.suptitle("Percobaan 1: Visualisasi Channel Blob", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_blob_channels.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] output/01_blob_channels.png")


def benchmark_blob(img):
    """
    Mengukur waktu pembuatan blob untuk berbagai ukuran input.
    Ukuran lebih besar = waktu lebih lama.
    """
    ukuran = [(224, 224), (299, 299), (416, 416), (640, 640)]
    for size in ukuran:
        start = time.time()
        for _ in range(100):
            cv2.dnn.blobFromImage(img, 1.0 / 255.0, size, swapRB=True)
        elapsed = (time.time() - start) / 100 * 1000
        print(f"  Size {size[0]}x{size[1]} : {elapsed:.2f} ms")


def visualisasi_preprocessing(img):
    """
    Membandingkan berbagai teknik preprocessing DNN secara visual:
    resize, center crop, normalisasi, mean subtraction, histogram equalization.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Gambar asli
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Asli")

    # Resize 224x224
    img_224 = cv2.resize(img, (224, 224))
    axes[0, 1].imshow(cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Resize 224x224")

    # Center crop
    h, w = img.shape[:2]
    s = min(h, w)
    sx, sy = (w - s) // 2, (h - s) // 2
    img_crop = cv2.resize(img[sy:sy + s, sx:sx + s], (224, 224))
    axes[0, 2].imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Center Crop")

    # Normalisasi [0,1]
    img_norm = img_224.astype(np.float32) / 255.0
    axes[1, 0].imshow(cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Normalisasi [0,1]")

    # Mean subtraction
    img_mean = img_224.astype(np.float32) - [103.94, 116.78, 123.68]
    vis = np.clip((img_mean - img_mean.min()) / (img_mean.max() - img_mean.min()), 0, 1)
    axes[1, 1].imshow(vis)
    axes[1, 1].set_title("Mean Subtracted")

    # Histogram equalized
    ycrcb = cv2.cvtColor(img_224, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    axes[1, 2].imshow(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Hist Equalized")

    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle("Perbandingan Teknik Preprocessing DNN", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_preprocessing_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/01_preprocessing_comparison.png")


def main():
    """Fungsi utama: menjalankan semua tahap percobaan blob DNN."""
    print("=" * 60)
    print("PERCOBAAN 1: KLASIFIKASI GAMBAR DENGAN OPENCV DNN (BLOB)")
    print("=" * 60)

    # 1. Memuat gambar
    img = load_gambar("kucing.jpg")
    if img is None:
        return

    # 2. Membuat blob standar
    print("\n--- 1. Membuat Blob Standar ---")
    blob = buat_blob_standar(img)

    # 3. Variasi parameter blob
    print("\n--- 2. Variasi Parameter Blob ---")
    variasi_parameter_blob(img)

    # 4. Visualisasi channel blob
    print("\n--- 3. Visualisasi Channel Blob ---")
    visualisasi_blob_channels(img, blob)

    # 5. Benchmark waktu pembuatan blob
    print("\n--- 4. Benchmark Waktu Blob ---")
    benchmark_blob(img)

    # 6. Visualisasi perbandingan preprocessing
    print("\n--- 5. Visualisasi Preprocessing ---")
    visualisasi_preprocessing(img)

    # 7. Menampilkan gambar asli
    cv2.imshow("Gambar Asli - Percobaan 1", cv2.resize(img, (400, 400)))
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 1")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. cv2.dnn.blobFromImage() membuat tensor 4D (N,C,H,W)
2. Parameter: scalefactor, size, mean, swapRB, crop
3. Blob adalah format input standar neural network di OpenCV
4. Preprocessing yang tepat mempengaruhi hasil klasifikasi
5. Ukuran input mempengaruhi waktu komputasi

Output: output/01_blob_channels.png, output/01_preprocessing_comparison.png
""")


if __name__ == "__main__":
    main()
