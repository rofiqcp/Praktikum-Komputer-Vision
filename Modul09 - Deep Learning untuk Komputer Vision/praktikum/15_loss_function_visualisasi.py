"""
==========================================================================
PERCOBAAN 15: LOSS FUNCTION VISUALISASI
==========================================================================
Program ini memvisualisasikan berbagai loss function yang digunakan
dalam deep learning. Loss function mengukur seberapa jauh prediksi
model dari nilai yang benar (ground truth).

Loss yang dipelajari:
- MSE (Mean Squared Error) - untuk regresi
- Cross-Entropy - untuk klasifikasi
- Binary Cross-Entropy - untuk klasifikasi biner
- Dice Loss - untuk segmentasi
- Focal Loss - untuk data imbalanced

Referensi: Szeliski Ch.5, Deep Learning for CV (Rosebrock)
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def mse_loss(y_pred, y_true):
    """MSE Loss = mean((y_pred - y_true)^2). Untuk regresi."""
    return np.mean((y_pred - y_true) ** 2)


def binary_cross_entropy(y_pred, y_true):
    """BCE = -[y*log(p) + (1-y)*log(1-p)]. Untuk klasifikasi biner."""
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def categorical_cross_entropy(y_pred, y_true):
    """CCE = -sum(y_true * log(y_pred)). Untuk klasifikasi multi-kelas."""
    y_pred = np.clip(y_pred, 1e-7, 1)
    return -np.sum(y_true * np.log(y_pred))


def dice_loss(y_pred, y_true):
    """Dice Loss = 1 - 2*|A∩B|/(|A|+|B|). Untuk segmentasi."""
    intersection = np.sum(y_pred * y_true)
    return 1 - (2 * intersection + 1e-7) / (np.sum(y_pred) + np.sum(y_true) + 1e-7)


def focal_loss(y_pred, y_true, gamma=2.0, alpha=0.25):
    """Focal Loss = -alpha*(1-p)^gamma*log(p). Untuk data imbalanced."""
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    return -np.mean(alpha_t * (1 - p_t) ** gamma * np.log(p_t))


def visualisasi_loss_vs_prediksi():
    """
    Menampilkan bagaimana setiap loss function berubah terhadap prediksi.
    x-axis: probabilitas prediksi, y-axis: nilai loss.
    """
    p = np.linspace(0.01, 0.99, 200)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # MSE (target = 1)
    mse = (1 - p) ** 2
    axes[0, 0].plot(p, mse, 'b-', linewidth=2)
    axes[0, 0].set_title("MSE Loss (target=1)")
    axes[0, 0].set_xlabel("Prediksi")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # BCE (target = 1)
    bce = -np.log(np.clip(p, 1e-7, 1))
    axes[0, 1].plot(p, bce, 'r-', linewidth=2)
    axes[0, 1].set_title("Binary Cross-Entropy (target=1)")
    axes[0, 1].set_xlabel("Prediksi")
    axes[0, 1].grid(True, alpha=0.3)

    # BCE (target = 0)
    bce0 = -np.log(np.clip(1 - p, 1e-7, 1))
    axes[0, 2].plot(p, bce0, 'g-', linewidth=2)
    axes[0, 2].set_title("Binary Cross-Entropy (target=0)")
    axes[0, 2].set_xlabel("Prediksi")
    axes[0, 2].grid(True, alpha=0.3)

    # Focal Loss (berbagai gamma)
    for gamma in [0, 1, 2, 5]:
        fl = -(1 - p) ** gamma * np.log(np.clip(p, 1e-7, 1))
        axes[1, 0].plot(p, fl, linewidth=2, label=f'γ={gamma}')
    axes[1, 0].set_title("Focal Loss (target=1)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # MSE vs BCE
    axes[1, 1].plot(p, mse, 'b-', linewidth=2, label='MSE')
    axes[1, 1].plot(p, np.clip(bce, 0, 5), 'r-', linewidth=2, label='BCE')
    axes[1, 1].set_title("MSE vs BCE (target=1)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Dice Loss
    dl = 1 - 2 * p / (p + 1 + 1e-7)
    axes[1, 2].plot(p, dl, 'm-', linewidth=2)
    axes[1, 2].set_title("Dice Loss (target=1)")
    axes[1, 2].set_xlabel("Prediksi")
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("Percobaan 15: Visualisasi Loss Functions", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_loss_functions.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/15_loss_functions.png")


def demo_loss_calculation():
    """
    Mendemonstrasikan perhitungan loss untuk kasus nyata.
    Menghitung loss dari prediksi vs ground truth.
    """
    # Kasus klasifikasi: 5 sampel
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.4, 0.3])

    print("  Data: y_true =", y_true)
    print("  Data: y_pred =", [f"{v:.1f}" for v in y_pred])
    print(f"  MSE Loss      : {mse_loss(y_pred, y_true):.4f}")
    print(f"  BCE Loss      : {binary_cross_entropy(y_pred, y_true):.4f}")
    print(f"  Focal Loss    : {focal_loss(y_pred, y_true):.4f}")

    # Kasus segmentasi: mask biner
    mask_true = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
    mask_pred = np.array([[0.9, 0.8, 0.1], [0.7, 0.9, 0.2], [0.1, 0.1, 0.05]], dtype=np.float32)
    print(f"\n  Dice Loss (seg): {dice_loss(mask_pred, mask_true):.4f}")


def visualisasi_landscape_loss():
    """
    Menampilkan loss landscape 3D (surface plot).
    Menunjukkan bagaimana loss berubah terhadap 2 parameter.
    """
    w1 = np.linspace(-3, 3, 100)
    w2 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)
    # Loss = simple quadratic (convex)
    L = W1 ** 2 + W2 ** 2 + 0.5 * W1 * W2

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(W1, W2, L, cmap='viridis', alpha=0.8)
    ax1.set_xlabel("w1")
    ax1.set_ylabel("w2")
    ax1.set_zlabel("Loss")
    ax1.set_title("Loss Landscape 3D (Convex)")

    ax2 = fig.add_subplot(122)
    ax2.contour(W1, W2, L, levels=30, cmap='viridis')
    ax2.plot(0, 0, 'r*', markersize=15, label='Minimum')
    ax2.set_xlabel("w1")
    ax2.set_ylabel("w2")
    ax2.set_title("Contour Loss Landscape")
    ax2.legend()

    plt.suptitle("Loss Landscape (Gradient Descent akan menuju minimum)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_loss_landscape.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/15_loss_landscape.png")


def main():
    """Fungsi utama: visualisasi loss function."""
    print("=" * 60)
    print("PERCOBAAN 15: LOSS FUNCTION VISUALISASI")
    print("=" * 60)

    print("\n--- 1. Grafik Loss Functions ---")
    visualisasi_loss_vs_prediksi()

    print("\n--- 2. Demo Perhitungan Loss ---")
    demo_loss_calculation()

    print("\n--- 3. Loss Landscape ---")
    visualisasi_landscape_loss()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 15")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. MSE cocok untuk regresi (prediksi nilai kontinu)
2. Cross-Entropy cocok untuk klasifikasi (prediksi probabilitas)
3. Dice Loss cocok untuk segmentasi (menghitung overlap)
4. Focal Loss mengatasi masalah class imbalance
5. Loss landscape menentukan sulitnya optimisasi

Output: output/15_loss_functions.png, output/15_loss_landscape.png
""")


if __name__ == "__main__":
    main()
