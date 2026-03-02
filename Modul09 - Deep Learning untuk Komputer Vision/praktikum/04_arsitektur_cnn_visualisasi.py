"""
==========================================================================
PERCOBAAN 4: VISUALISASI ARSITEKTUR CNN
==========================================================================
Program ini memvisualisasikan komponen arsitektur CNN (Convolutional
Neural Network) secara layer-by-layer. Membantu memahami bagaimana
data mengalir dari input hingga output dalam sebuah CNN.

Konsep yang dipelajari:
- Struktur layer CNN: Conv -> ReLU -> Pool -> FC
- Perubahan dimensi tensor di setiap layer
- Arsitektur terkenal: LeNet-5, VGG, ResNet (diagram)
- Receptive field dan feature map

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


def simulasi_conv_layer(img_gray, kernel_size=3, num_filters=4):
    """
    Mensimulasikan operasi convolutional layer pada gambar grayscale.
    Menggunakan kernel acak untuk menghasilkan feature map.
    Setiap filter mengekstrak fitur berbeda (edge, texture, dll).
    """
    h, w = img_gray.shape
    feature_maps = []
    kernels = []
    for i in range(num_filters):
        # Buat kernel acak yang bermakna (Gabor-like)
        np.random.seed(i * 42)
        kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32)
        kernel /= np.abs(kernel).sum() + 1e-7
        kernels.append(kernel)
        # Terapkan konvolusi
        fmap = cv2.filter2D(img_gray, cv2.CV_32F, kernel)
        feature_maps.append(fmap)
    return feature_maps, kernels


def simulasi_pooling(feature_map, pool_size=2):
    """
    Mensimulasikan max pooling pada feature map.
    Max pooling mengambil nilai maksimum dalam window pool_size x pool_size.
    Efek: mengurangi dimensi spasial dan menambah invariansi.
    """
    h, w = feature_map.shape
    h_new, w_new = h // pool_size, w // pool_size
    pooled = np.zeros((h_new, w_new), dtype=np.float32)
    for i in range(h_new):
        for j in range(w_new):
            region = feature_map[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            pooled[i, j] = np.max(region)
    return pooled


def visualisasi_arsitektur_cnn(img_gray, feature_maps, pooled_maps, kernels):
    """
    Menampilkan alur data melalui CNN: Input -> Conv -> ReLU -> Pool.
    Setiap tahap divisualisasikan untuk memahami transformasi data.
    """
    n = len(feature_maps)
    fig, axes = plt.subplots(4, n + 1, figsize=(4 * (n + 1), 14))

    # Baris 1: Input
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title("Input", fontsize=10)
    for j in range(1, n + 1):
        axes[0, j].imshow(kernels[j - 1], cmap='RdBu_r')
        axes[0, j].set_title(f"Kernel {j}", fontsize=9)

    # Baris 2: Setelah konvolusi
    axes[1, 0].set_title("Conv Output", fontsize=10)
    axes[1, 0].axis('off')
    for j in range(n):
        axes[1, j + 1].imshow(feature_maps[j], cmap='viridis')
        axes[1, j + 1].set_title(f"FMap {j+1}\n{feature_maps[j].shape}", fontsize=8)

    # Baris 3: Setelah ReLU
    axes[2, 0].set_title("ReLU Output", fontsize=10)
    axes[2, 0].axis('off')
    for j in range(n):
        relu = np.maximum(feature_maps[j], 0)
        axes[2, j + 1].imshow(relu, cmap='viridis')
        axes[2, j + 1].set_title(f"ReLU {j+1}", fontsize=8)

    # Baris 4: Setelah Max Pool
    axes[3, 0].set_title("MaxPool Output", fontsize=10)
    axes[3, 0].axis('off')
    for j in range(n):
        axes[3, j + 1].imshow(pooled_maps[j], cmap='viridis')
        axes[3, j + 1].set_title(f"Pool {j+1}\n{pooled_maps[j].shape}", fontsize=8)

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle("Percobaan 4: Alur Data dalam CNN (Conv → ReLU → Pool)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_arsitektur_cnn.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/04_arsitektur_cnn.png")


def diagram_arsitektur_terkenal():
    """
    Membuat diagram perbandingan arsitektur CNN terkenal.
    Menampilkan jumlah layer, parameter, dan keunggulan masing-masing.
    """
    arsitektur = {
        'LeNet-5\n(1998)': {'layers': 7, 'params': 0.06, 'top1': 99.0},
        'AlexNet\n(2012)': {'layers': 8, 'params': 61, 'top1': 63.3},
        'VGG-16\n(2014)': {'layers': 16, 'params': 138, 'top1': 74.4},
        'GoogLeNet\n(2014)': {'layers': 22, 'params': 6.8, 'top1': 74.8},
        'ResNet-50\n(2015)': {'layers': 50, 'params': 25.6, 'top1': 76.2},
        'MobileNetV2\n(2018)': {'layers': 53, 'params': 3.4, 'top1': 72.0},
        'EfficientNet\n(2019)': {'layers': 82, 'params': 5.3, 'top1': 77.1},
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    nama = list(arsitektur.keys())
    layers = [v['layers'] for v in arsitektur.values()]
    params = [v['params'] for v in arsitektur.values()]
    acc = [v['top1'] for v in arsitektur.values()]

    colors = plt.cm.tab10(np.linspace(0, 0.7, len(nama)))
    ax1.bar(nama, layers, color=colors, edgecolor='black')
    ax1.set_ylabel("Jumlah Layer")
    ax1.set_title("Jumlah Layer per Arsitektur")

    ax2.bar(nama, params, color=colors, edgecolor='black')
    ax2.set_ylabel("Parameter (Juta)")
    ax2.set_title("Jumlah Parameter per Arsitektur")
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_arsitektur_terkenal.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/04_arsitektur_terkenal.png")


def main():
    """Fungsi utama: visualisasi arsitektur CNN."""
    print("=" * 60)
    print("PERCOBAAN 4: VISUALISASI ARSITEKTUR CNN")
    print("=" * 60)

    img = load_gambar("kucing.jpg")
    if img is None:
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (224, 224))

    # 1. Simulasi Conv Layer
    print("\n--- 1. Simulasi Convolutional Layer ---")
    feature_maps, kernels = simulasi_conv_layer(img_gray, kernel_size=3, num_filters=4)
    for i, fm in enumerate(feature_maps):
        print(f"  Feature Map {i+1}: shape={fm.shape}, range=[{fm.min():.1f}, {fm.max():.1f}]")

    # 2. Simulasi Pooling
    print("\n--- 2. Simulasi Max Pooling ---")
    pooled_maps = [simulasi_pooling(fm) for fm in feature_maps]
    for i, pm in enumerate(pooled_maps):
        print(f"  Pooled Map {i+1}: shape={pm.shape}")

    # 3. Visualisasi alur CNN
    print("\n--- 3. Visualisasi Alur CNN ---")
    visualisasi_arsitektur_cnn(img_gray, feature_maps, pooled_maps, kernels)

    # 4. Diagram arsitektur terkenal
    print("\n--- 4. Diagram Arsitektur Terkenal ---")
    diagram_arsitektur_terkenal()

    # Tampilkan gambar
    cv2.imshow("Input CNN - Percobaan 4", img_gray)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 4")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. CNN terdiri dari: Conv → ReLU → Pool → FC
2. Convolutional layer mengekstrak fitur lokal (edge, texture)
3. ReLU menghilangkan nilai negatif (non-linearitas)
4. Max Pooling mengurangi dimensi dan menambah invariansi
5. Arsitektur modern (ResNet, EfficientNet) lebih dalam dan efisien

Output: output/04_arsitektur_cnn.png, output/04_arsitektur_terkenal.png
""")


if __name__ == "__main__":
    main()
