"""
==========================================================================
PERCOBAAN 5: FUNGSI AKTIVASI
==========================================================================
Program ini memvisualisasikan dan membandingkan berbagai fungsi aktivasi
yang digunakan dalam neural network. Fungsi aktivasi memberikan
non-linearitas sehingga network bisa belajar pola kompleks.

Fungsi aktivasi yang dipelajari:
- ReLU (Rectified Linear Unit) - paling populer saat ini
- Sigmoid - output range (0,1), untuk probabilitas
- Tanh - output range (-1,1)
- Leaky ReLU - mengatasi masalah dying neurons
- Softmax - untuk output klasifikasi multi-kelas

Referensi: Szeliski Ch.5, Géron Ch.10, Deep Learning for CV
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


def relu(x):
    """ReLU: f(x) = max(0, x). Menghilangkan nilai negatif."""
    return np.maximum(0, x)


def sigmoid(x):
    """Sigmoid: f(x) = 1/(1+e^-x). Output antara 0 dan 1."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh_act(x):
    """Tanh: f(x) = tanh(x). Output antara -1 dan 1."""
    return np.tanh(x)


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: f(x) = x jika x>0, alpha*x jika x<=0."""
    return np.where(x > 0, x, alpha * x)


def softmax(x):
    """Softmax: mengubah vektor menjadi distribusi probabilitas."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def turunan_relu(x):
    """Turunan ReLU: 1 jika x>0, 0 jika x<=0."""
    return np.where(x > 0, 1.0, 0.0)


def turunan_sigmoid(x):
    """Turunan Sigmoid: sigmoid(x) * (1 - sigmoid(x))."""
    s = sigmoid(x)
    return s * (1 - s)


def visualisasi_fungsi_aktivasi():
    """
    Membuat grafik perbandingan semua fungsi aktivasi dan turunannya.
    Menampilkan bentuk kurva, range output, dan karakteristik masing-masing.
    """
    x = np.linspace(-6, 6, 300)

    fungsi = [
        ("ReLU", relu(x), "tab:blue"),
        ("Sigmoid", sigmoid(x), "tab:orange"),
        ("Tanh", tanh_act(x), "tab:green"),
        ("Leaky ReLU (α=0.01)", leaky_relu(x), "tab:red"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, (nama, y, warna) in enumerate(fungsi):
        ax = axes[i // 2, i % 2]
        ax.plot(x, y, color=warna, linewidth=2, label=nama)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_title(nama, fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Percobaan 5: Fungsi Aktivasi Neural Network", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_fungsi_aktivasi.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/05_fungsi_aktivasi.png")


def visualisasi_turunan():
    """
    Menampilkan turunan (derivatif) dari fungsi aktivasi.
    Turunan penting untuk backpropagation - menentukan seberapa
    besar gradien yang mengalir balik saat training.
    """
    x = np.linspace(-6, 6, 300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Turunan ReLU
    ax1.plot(x, turunan_relu(x), 'b-', linewidth=2, label="ReLU'")
    ax1.plot(x, np.where(x > 0, 1.0, 0.01), 'r--', linewidth=2, label="Leaky ReLU'")
    ax1.set_title("Turunan ReLU vs Leaky ReLU")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Turunan Sigmoid
    ax2.plot(x, turunan_sigmoid(x), 'g-', linewidth=2, label="Sigmoid'")
    ax2.plot(x, 1 - np.tanh(x) ** 2, 'm--', linewidth=2, label="Tanh'")
    ax2.set_title("Turunan Sigmoid vs Tanh")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Turunan Fungsi Aktivasi (untuk Backpropagation)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_turunan_aktivasi.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/05_turunan_aktivasi.png")


def demo_softmax():
    """
    Mendemonstrasikan fungsi softmax pada vektor logit.
    Softmax mengubah skor mentah menjadi probabilitas yang jumlahnya = 1.
    """
    logits = np.array([2.0, 1.0, 0.5, -1.0, 3.0])
    probs = softmax(logits)
    kelas = ["Kucing", "Anjing", "Burung", "Ikan", "Kelinci"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(kelas, logits, color='steelblue', edgecolor='black')
    ax1.set_title("Logits (skor mentah)")
    ax1.set_ylabel("Nilai")

    ax2.bar(kelas, probs, color='coral', edgecolor='black')
    ax2.set_title(f"Softmax (probabilitas, sum={probs.sum():.2f})")
    ax2.set_ylabel("Probabilitas")

    plt.suptitle("Demo Softmax: Logits → Probabilitas", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_softmax_demo.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/05_softmax_demo.png")

    for k, l, p in zip(kelas, logits, probs):
        print(f"  {k}: logit={l:.1f} → prob={p:.4f}")


def efek_aktivasi_pada_gambar():
    """
    Menerapkan fungsi aktivasi pada gambar grayscale.
    Menunjukkan efek visual dari setiap fungsi aktivasi.
    """
    path = os.path.join(IMAGE_DIR, "kucing.jpg")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("  [SKIP] Gambar tidak tersedia")
        return
    img = cv2.resize(img, (224, 224))
    # Normalisasi ke range [-3, 3] untuk demonstrasi
    img_norm = (img.astype(np.float32) / 255.0 - 0.5) * 6

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Asli")

    aktivasi = [("ReLU", relu), ("Sigmoid", sigmoid), ("Tanh", tanh_act), ("LeakyReLU", leaky_relu)]
    for i, (nama, fn) in enumerate(aktivasi):
        result = fn(img_norm)
        axes[i + 1].imshow(result, cmap='viridis')
        axes[i + 1].set_title(nama)

    for ax in axes:
        ax.axis('off')

    plt.suptitle("Efek Fungsi Aktivasi pada Gambar", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_aktivasi_gambar.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/05_aktivasi_gambar.png")


def main():
    """Fungsi utama: visualisasi fungsi aktivasi."""
    print("=" * 60)
    print("PERCOBAAN 5: FUNGSI AKTIVASI")
    print("=" * 60)

    print("\n--- 1. Grafik Fungsi Aktivasi ---")
    visualisasi_fungsi_aktivasi()

    print("\n--- 2. Turunan Fungsi Aktivasi ---")
    visualisasi_turunan()

    print("\n--- 3. Demo Softmax ---")
    demo_softmax()

    print("\n--- 4. Efek Aktivasi pada Gambar ---")
    efek_aktivasi_pada_gambar()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 5")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. ReLU paling populer karena sederhana dan efektif
2. Sigmoid cocok untuk output probabilitas biner (0-1)
3. Softmax mengubah logits menjadi distribusi probabilitas
4. Turunan aktivasi penting untuk backpropagation
5. Leaky ReLU mengatasi masalah 'dying neurons' pada ReLU

Output: output/05_fungsi_aktivasi.png, output/05_turunan_aktivasi.png,
        output/05_softmax_demo.png, output/05_aktivasi_gambar.png
""")


if __name__ == "__main__":
    main()
