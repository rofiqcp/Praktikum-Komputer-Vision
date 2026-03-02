"""
==========================================================================
PERCOBAAN 8: TRANSFER LEARNING (KONSEP DAN VISUALISASI)
==========================================================================
Program ini menjelaskan konsep transfer learning secara visual.
Transfer learning menggunakan model yang sudah dilatih pada dataset
besar (ImageNet) sebagai titik awal untuk tugas klasifikasi baru.

Konsep yang dipelajari:
- Feature Extraction: freeze semua layer, ganti classifier akhir
- Fine-tuning: unfreeze beberapa layer terakhir, train dengan LR kecil
- Perbandingan strategi transfer learning
- Kapan menggunakan masing-masing strategi

Referensi: Deep Learning for CV (Rosebrock), Géron Ch.10
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


def diagram_transfer_learning():
    """
    Membuat diagram yang menjelaskan 3 strategi transfer learning:
    1. Training from scratch (semua layer dilatih)
    2. Feature extraction (hanya classifier dilatih)
    3. Fine-tuning (beberapa layer terakhir + classifier dilatih)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    strategi = [
        ("From Scratch", ["Conv1", "Conv2", "Conv3", "Conv4", "FC1", "FC2"],
         ["train"] * 6, "Semua layer dilatih\nMemerlukan banyak data"),
        ("Feature Extraction", ["Conv1", "Conv2", "Conv3", "Conv4", "FC1*", "FC2*"],
         ["freeze", "freeze", "freeze", "freeze", "train", "train"],
         "Hanya classifier dilatih\nCepat, data sedikit cukup"),
        ("Fine-tuning", ["Conv1", "Conv2", "Conv3*", "Conv4*", "FC1*", "FC2*"],
         ["freeze", "freeze", "train", "train", "train", "train"],
         "Beberapa layer terakhir dilatih\nKeseimbangan terbaik"),
    ]

    for idx, (nama, layers, status, desc) in enumerate(strategi):
        ax = axes[idx]
        colors = {'train': '#4CAF50', 'freeze': '#9E9E9E'}
        for i, (layer, st) in enumerate(zip(layers, status)):
            y = len(layers) - i - 1
            rect = plt.Rectangle((0.1, y * 0.15 + 0.05), 0.8, 0.12,
                                  facecolor=colors[st], edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(0.5, y * 0.15 + 0.11, layer, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(nama, fontsize=13, fontweight='bold')
        ax.text(0.5, 0.01, desc, ha='center', va='bottom', fontsize=8,
                style='italic', color='gray')
        ax.axis('off')

    plt.suptitle("Percobaan 8: Strategi Transfer Learning", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_transfer_learning_strategi.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/08_transfer_learning_strategi.png")


def simulasi_training_curve():
    """
    Mensimulasikan kurva training untuk 3 strategi transfer learning.
    Menunjukkan bahwa transfer learning konvergen lebih cepat dan
    mencapai akurasi lebih tinggi dengan data terbatas.
    """
    epochs = np.arange(1, 31)
    np.random.seed(42)

    # From scratch: lambat konvergen
    scratch_acc = 1 - np.exp(-epochs / 15) * 0.65 + np.random.normal(0, 0.02, 30)
    scratch_acc = np.clip(scratch_acc, 0.3, 0.88)

    # Feature extraction: cepat konvergen, plateu lebih cepat
    feat_acc = 1 - np.exp(-epochs / 4) * 0.25 + np.random.normal(0, 0.01, 30)
    feat_acc = np.clip(feat_acc, 0.7, 0.92)

    # Fine-tuning: cepat dan akurasi tinggi
    fine_acc = 1 - np.exp(-epochs / 5) * 0.2 + np.random.normal(0, 0.01, 30)
    fine_acc = np.clip(fine_acc, 0.75, 0.96)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, scratch_acc * 100, 'r-', linewidth=2, label='From Scratch')
    ax1.plot(epochs, feat_acc * 100, 'b-', linewidth=2, label='Feature Extraction')
    ax1.plot(epochs, fine_acc * 100, 'g-', linewidth=2, label='Fine-tuning')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Akurasi Validasi (%)")
    ax1.set_title("Kurva Training: Akurasi")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss curves
    scratch_loss = np.exp(-epochs / 15) * 2.0 + np.random.normal(0, 0.05, 30)
    feat_loss = np.exp(-epochs / 4) * 0.5 + np.random.normal(0, 0.02, 30)
    fine_loss = np.exp(-epochs / 5) * 0.4 + np.random.normal(0, 0.02, 30)

    ax2.plot(epochs, np.clip(scratch_loss, 0.1, 2.5), 'r-', linewidth=2, label='From Scratch')
    ax2.plot(epochs, np.clip(feat_loss, 0.05, 1), 'b-', linewidth=2, label='Feature Extraction')
    ax2.plot(epochs, np.clip(fine_loss, 0.03, 0.8), 'g-', linewidth=2, label='Fine-tuning')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Kurva Training: Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Simulasi Kurva Training Transfer Learning", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/08_training_curves.png")


def visualisasi_fitur_layer(img):
    """
    Mensimulasikan bagaimana layer berbeda mengekstrak fitur berbeda:
    - Layer awal: edge, corner, texture dasar
    - Layer tengah: pola, bentuk
    - Layer akhir: objek, bagian objek
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (224, 224)).astype(np.float32)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Layer awal: edge detection
    edges_h = cv2.Sobel(img_resized, cv2.CV_32F, 1, 0, ksize=3)
    edges_v = cv2.Sobel(img_resized, cv2.CV_32F, 0, 1, ksize=3)

    # Layer tengah: gabungan fitur
    gabor1 = cv2.getGaborKernel((11, 11), 4, 0, 8, 0.5)
    gabor2 = cv2.getGaborKernel((11, 11), 4, np.pi/4, 8, 0.5)
    feat_mid1 = cv2.filter2D(img_resized, cv2.CV_32F, gabor1)
    feat_mid2 = cv2.filter2D(img_resized, cv2.CV_32F, gabor2)

    # Layer akhir: blob detection
    blob1 = cv2.GaussianBlur(img_resized, (15, 15), 3)
    blob2 = img_resized - blob1  # Difference of Gaussian

    titles = [
        ["Input", "Edge Horiz (L1)", "Edge Vert (L1)", "Edge Mag (L1)"],
        ["Gabor 0° (L3)", "Gabor 45° (L3)", "Blob (L5)", "DoG (L5)"]
    ]
    images = [
        [img_resized, edges_h, edges_v, np.sqrt(edges_h**2 + edges_v**2)],
        [feat_mid1, feat_mid2, blob1, blob2]
    ]

    for i in range(2):
        for j in range(4):
            axes[i, j].imshow(images[i][j], cmap='viridis')
            axes[i, j].set_title(titles[i][j], fontsize=9)
            axes[i, j].axis('off')

    plt.suptitle("Simulasi Fitur di Berbagai Layer CNN", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_fitur_layer.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/08_fitur_layer.png")


def main():
    """Fungsi utama: transfer learning konsep dan visualisasi."""
    print("=" * 60)
    print("PERCOBAAN 8: TRANSFER LEARNING (KONSEP)")
    print("=" * 60)

    print("\n--- 1. Diagram Strategi Transfer Learning ---")
    diagram_transfer_learning()

    print("\n--- 2. Simulasi Kurva Training ---")
    simulasi_training_curve()

    print("\n--- 3. Visualisasi Fitur Layer ---")
    img = load_gambar("kucing.jpg")
    if img is not None:
        visualisasi_fitur_layer(img)
        cv2.imshow("Transfer Learning - Percobaan 8", cv2.resize(img, (400, 400)))
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 8")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. Transfer learning menggunakan model pre-trained sebagai titik awal
2. Feature extraction: cepat, cocok untuk data sedikit
3. Fine-tuning: akurasi lebih tinggi, perlu lebih banyak data
4. Layer awal CNN menangkap fitur umum (edge, texture)
5. Layer akhir menangkap fitur spesifik domain

Output: output/08_transfer_learning_strategi.png, output/08_training_curves.png,
        output/08_fitur_layer.png
""")


if __name__ == "__main__":
    main()
