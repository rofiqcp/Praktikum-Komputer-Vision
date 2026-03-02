"""
==========================================================================
PERCOBAAN 6: KONVOLUSI DAN POOLING
==========================================================================
Program ini mendemonstrasikan operasi konvolusi dan pooling secara visual.
Konvolusi adalah operasi inti CNN yang mengekstrak fitur lokal dari gambar.
Pooling mengurangi dimensi spasial feature map.

Konsep yang dipelajari:
- Konvolusi 2D dengan berbagai kernel (edge, blur, sharpen)
- Stride dan padding pada konvolusi
- Max Pooling vs Average Pooling
- Feature map dan receptive field

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


def demo_berbagai_kernel(img_gray):
    """
    Menerapkan berbagai kernel konvolusi pada gambar:
    - Identity: tidak mengubah gambar
    - Edge detection: mendeteksi tepi horizontal/vertikal
    - Blur: menghaluskan gambar
    - Sharpen: mempertajam gambar
    """
    kernels = {
        "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
        "Edge Horiz": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
        "Edge Vert": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
        "Blur 3x3": np.ones((3, 3), dtype=np.float32) / 9,
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
    }

    hasil = {}
    for nama, kernel in kernels.items():
        conv = cv2.filter2D(img_gray, cv2.CV_32F, kernel)
        hasil[nama] = conv
        print(f"  {nama:12s} : range [{conv.min():.1f}, {conv.max():.1f}]")
    return hasil, kernels


def visualisasi_konvolusi(img_gray, hasil, kernels):
    """
    Menampilkan hasil konvolusi dengan berbagai kernel dalam grid.
    Termasuk visualisasi kernel itu sendiri.
    """
    n = len(hasil)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))

    for i, (nama, conv) in enumerate(hasil.items()):
        # Baris atas: kernel
        k = kernels[nama]
        axes[0, i].imshow(k, cmap='RdBu_r', interpolation='nearest')
        axes[0, i].set_title(f"Kernel: {nama}", fontsize=8)
        axes[0, i].axis('off')

        # Baris bawah: hasil konvolusi
        axes[1, i].imshow(conv, cmap='gray')
        axes[1, i].set_title(f"Hasil: {nama}", fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle("Percobaan 6: Konvolusi dengan Berbagai Kernel", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_konvolusi_kernel.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/06_konvolusi_kernel.png")


def demo_stride_padding(img_gray):
    """
    Mendemonstrasikan efek stride dan padding pada konvolusi.
    - Stride > 1: output lebih kecil (downsampling)
    - Padding: menjaga ukuran output sama dengan input
    """
    img_small = cv2.resize(img_gray, (64, 64))
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)

    # Stride 1 (standar)
    conv_s1 = cv2.filter2D(img_small, cv2.CV_32F, kernel)

    # Simulasi stride 2 (ambil setiap piksel kedua)
    conv_s2 = conv_s1[::2, ::2]

    # Dilated convolution simulasi
    kernel_dilated = np.zeros((5, 5), dtype=np.float32)
    kernel_dilated[0::2, 0::2] = kernel
    conv_dilated = cv2.filter2D(img_small, cv2.CV_32F, kernel_dilated)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_small, cmap='gray')
    axes[0].set_title(f"Input {img_small.shape}")

    axes[1].imshow(conv_s1, cmap='gray')
    axes[1].set_title(f"Stride=1 {conv_s1.shape}")

    axes[2].imshow(conv_s2, cmap='gray')
    axes[2].set_title(f"Stride=2 {conv_s2.shape}")

    axes[3].imshow(conv_dilated, cmap='gray')
    axes[3].set_title(f"Dilated Conv {conv_dilated.shape}")

    for ax in axes:
        ax.axis('off')
    plt.suptitle("Efek Stride dan Dilated Convolution", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_stride_padding.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/06_stride_padding.png")


def demo_pooling(img_gray):
    """
    Membandingkan Max Pooling vs Average Pooling.
    Max Pooling: mengambil nilai terbesar (mempertahankan fitur kuat)
    Average Pooling: mengambil rata-rata (memperhalus fitur)
    """
    img_small = cv2.resize(img_gray, (128, 128)).astype(np.float32)

    # Max Pooling dengan berbagai ukuran
    pool_sizes = [2, 4, 8]
    fig, axes = plt.subplots(2, len(pool_sizes) + 1, figsize=(16, 8))

    axes[0, 0].imshow(img_small, cmap='gray')
    axes[0, 0].set_title(f"Asli {img_small.shape}")
    axes[1, 0].imshow(img_small, cmap='gray')
    axes[1, 0].set_title(f"Asli {img_small.shape}")

    for i, ps in enumerate(pool_sizes):
        h, w = img_small.shape
        h_new, w_new = h // ps, w // ps
        # Max pooling manual
        max_pool = np.zeros((h_new, w_new))
        avg_pool = np.zeros((h_new, w_new))
        for y in range(h_new):
            for x in range(w_new):
                region = img_small[y * ps:(y + 1) * ps, x * ps:(x + 1) * ps]
                max_pool[y, x] = np.max(region)
                avg_pool[y, x] = np.mean(region)

        axes[0, i + 1].imshow(max_pool, cmap='gray')
        axes[0, i + 1].set_title(f"MaxPool {ps}x{ps}\n{max_pool.shape}")

        axes[1, i + 1].imshow(avg_pool, cmap='gray')
        axes[1, i + 1].set_title(f"AvgPool {ps}x{ps}\n{avg_pool.shape}")

    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle("Perbandingan Max Pooling vs Average Pooling", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_pooling_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/06_pooling_comparison.png")


def main():
    """Fungsi utama: demonstrasi konvolusi dan pooling."""
    print("=" * 60)
    print("PERCOBAAN 6: KONVOLUSI DAN POOLING")
    print("=" * 60)

    img = load_gambar("kucing.jpg")
    if img is None:
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (224, 224))

    print("\n--- 1. Berbagai Kernel Konvolusi ---")
    hasil, kernels = demo_berbagai_kernel(img_gray)

    print("\n--- 2. Visualisasi Konvolusi ---")
    visualisasi_konvolusi(img_gray, hasil, kernels)

    print("\n--- 3. Stride dan Padding ---")
    demo_stride_padding(img_gray)

    print("\n--- 4. Max Pooling vs Average Pooling ---")
    demo_pooling(img_gray)

    cv2.imshow("Input - Percobaan 6", img_gray)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 6")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. Konvolusi mengekstrak fitur lokal (edge, texture, dll)
2. Kernel berbeda menghasilkan fitur berbeda
3. Stride > 1 mengurangi dimensi output
4. Max Pooling mempertahankan fitur terkuat
5. Average Pooling memberikan representasi lebih halus

Output: output/06_konvolusi_kernel.png, output/06_stride_padding.png,
        output/06_pooling_comparison.png
""")


if __name__ == "__main__":
    main()
