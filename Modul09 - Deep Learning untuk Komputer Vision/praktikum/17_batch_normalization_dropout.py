"""
==========================================================================
PERCOBAAN 17: BATCH NORMALIZATION DAN DROPOUT
==========================================================================
Program ini mendemonstrasikan dua teknik regularisasi penting dalam
deep learning: Batch Normalization dan Dropout.

Konsep yang dipelajari:
- Batch Normalization: menormalisasi output setiap layer
- Dropout: mematikan neuron secara random saat training
- Efek BN terhadap distribusi aktivasi
- Efek Dropout terhadap overfitting
- Kapan menggunakan BN vs Dropout

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


def batch_normalization(x, gamma=1.0, beta=0.0, eps=1e-5):
    """
    Batch Normalization: normalisasi output layer ke mean=0, std=1.
    Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
    gamma dan beta adalah parameter yang dipelajari.
    """
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def dropout(x, rate=0.5, training=True):
    """
    Dropout: mematikan neuron secara random dengan probabilitas 'rate'.
    Saat inference (training=False), semua neuron aktif tetapi
    output diskala dengan (1-rate) untuk kompensasi.
    """
    if not training:
        return x
    mask = (np.random.random(x.shape) > rate).astype(np.float32)
    return x * mask / (1 - rate)  # Inverted dropout


def visualisasi_batch_norm():
    """
    Menampilkan efek Batch Normalization pada distribusi aktivasi.
    Tanpa BN: distribusi bisa sangat bervariasi antar layer.
    Dengan BN: distribusi selalu terpusat di 0 dengan std 1.
    """
    np.random.seed(42)

    # Simulasi 4 layer tanpa BN
    n_samples = 1000
    x = np.random.randn(n_samples, 10)
    layers_no_bn = [x]
    for _ in range(3):
        w = np.random.randn(10, 10) * 2
        x = np.maximum(0, x.dot(w))  # ReLU
        layers_no_bn.append(x)

    # Simulasi 4 layer dengan BN
    x = np.random.randn(n_samples, 10)
    layers_bn = [x]
    for _ in range(3):
        w = np.random.randn(10, 10) * 2
        x = np.maximum(0, x.dot(w))
        x = batch_normalization(x)
        layers_bn.append(x)

    # Visualisasi
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        # Tanpa BN
        data = layers_no_bn[i][:, 0]
        data = data[np.isfinite(data)]
        if len(data) > 0:
            axes[0, i].hist(data, bins=50, color='coral', alpha=0.7)
        axes[0, i].set_title(f"Layer {i} (Tanpa BN)\nmean={np.mean(data):.1f}")

        # Dengan BN
        data = layers_bn[i][:, 0]
        data = data[np.isfinite(data)]
        if len(data) > 0:
            axes[1, i].hist(data, bins=50, color='steelblue', alpha=0.7)
        axes[1, i].set_title(f"Layer {i} (Dengan BN)\nmean={np.mean(data):.2f}")

    plt.suptitle("Efek Batch Normalization pada Distribusi Aktivasi", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_batch_norm_effect.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/17_batch_norm_effect.png")


def visualisasi_dropout():
    """
    Menampilkan efek dropout pada neuron network.
    Random neuron di-'matikan' (set ke 0) saat training.
    """
    np.random.seed(42)
    x = np.random.randn(8, 8)

    rates = [0.0, 0.2, 0.5, 0.8]
    fig, axes = plt.subplots(2, len(rates), figsize=(16, 7))

    for i, rate in enumerate(rates):
        # Aktivasi sebelum dropout
        axes[0, i].imshow(np.abs(x), cmap='Blues', vmin=0, vmax=3)
        axes[0, i].set_title(f"Sebelum\n(rate={rate})")

        # Setelah dropout
        x_drop = dropout(x.copy(), rate=rate, training=True)
        axes[1, i].imshow(np.abs(x_drop), cmap='Blues', vmin=0, vmax=3)
        n_dead = np.sum(x_drop == 0)
        axes[1, i].set_title(f"Setelah Dropout\n({n_dead}/64 mati)")

    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle("Efek Dropout pada Neuron (warna = aktivasi)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_dropout_effect.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/17_dropout_effect.png")


def simulasi_overfitting():
    """
    Mensimulasikan efek dropout terhadap overfitting.
    Tanpa dropout: model overfitting (train bagus, val buruk).
    Dengan dropout: generalisasi lebih baik.
    """
    np.random.seed(42)
    epochs = np.arange(1, 51)

    # Tanpa dropout
    train_no_drop = 1 - np.exp(-epochs / 5) * 0.5 + np.random.normal(0, 0.01, 50)
    val_no_drop = 1 - np.exp(-epochs / 8) * 0.4 + np.random.normal(0, 0.02, 50)
    val_no_drop[20:] -= np.linspace(0, 0.15, 30)  # Overfitting

    # Dengan dropout
    train_drop = 1 - np.exp(-epochs / 6) * 0.5 + np.random.normal(0, 0.01, 50)
    val_drop = 1 - np.exp(-epochs / 7) * 0.5 + np.random.normal(0, 0.015, 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_no_drop * 100, 'b-', linewidth=2, label='Train')
    ax1.plot(epochs, val_no_drop * 100, 'r--', linewidth=2, label='Validation')
    ax1.set_title("Tanpa Dropout (Overfitting)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Akurasi (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.annotate('Gap = Overfitting', xy=(40, 70), fontsize=10, color='red')

    ax2.plot(epochs, train_drop * 100, 'b-', linewidth=2, label='Train')
    ax2.plot(epochs, val_drop * 100, 'r--', linewidth=2, label='Validation')
    ax2.set_title("Dengan Dropout (Lebih Baik)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Akurasi (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Efek Dropout terhadap Overfitting", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_overfitting_dropout.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/17_overfitting_dropout.png")


def main():
    """Fungsi utama: batch normalization dan dropout."""
    print("=" * 60)
    print("PERCOBAAN 17: BATCH NORMALIZATION DAN DROPOUT")
    print("=" * 60)

    print("\n--- 1. Efek Batch Normalization ---")
    visualisasi_batch_norm()

    print("\n--- 2. Efek Dropout ---")
    visualisasi_dropout()

    print("\n--- 3. Dropout vs Overfitting ---")
    simulasi_overfitting()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 17")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. Batch Normalization menstabilkan distribusi aktivasi antar layer
2. BN mempercepat training dan memungkinkan learning rate lebih besar
3. Dropout mencegah overfitting dengan mematikan neuron secara random
4. Dropout rate 0.2-0.5 umumnya optimal
5. BN dan Dropout sering digunakan bersamaan

Output: output/17_batch_norm_effect.png, output/17_dropout_effect.png,
        output/17_overfitting_dropout.png
""")


if __name__ == "__main__":
    main()
