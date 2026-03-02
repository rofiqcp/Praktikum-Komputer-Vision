"""
==========================================================================
PERCOBAAN 16: OPTIMIZER VISUALISASI
==========================================================================
Program ini memvisualisasikan perilaku berbagai optimizer yang digunakan
untuk melatih neural network. Optimizer menentukan bagaimana weight
diupdate berdasarkan gradien dari loss function.

Optimizer yang dipelajari:
- SGD (Stochastic Gradient Descent) - paling dasar
- SGD + Momentum - mempercepat konvergensi
- Adam (Adaptive Moment Estimation) - paling populer
- RMSProp - adaptive learning rate per parameter

Referensi: Szeliski Ch.5, Géron Ch.10, Deep Learning for CV
==========================================================================
"""

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def loss_function(w1, w2):
    """Fungsi loss non-convex: f(w1,w2) = w1^2 + 10*w2^2 (elongated valley)."""
    return w1 ** 2 + 10 * w2 ** 2


def gradient(w1, w2):
    """Gradien dari loss function."""
    return np.array([2 * w1, 20 * w2])


def sgd_optimizer(w, lr=0.01, steps=100):
    """SGD murni: w = w - lr * gradient."""
    path = [w.copy()]
    for _ in range(steps):
        grad = gradient(w[0], w[1])
        w = w - lr * grad
        path.append(w.copy())
    return np.array(path)


def sgd_momentum_optimizer(w, lr=0.01, momentum=0.9, steps=100):
    """SGD + Momentum: menggunakan moving average dari gradien."""
    path = [w.copy()]
    velocity = np.zeros_like(w)
    for _ in range(steps):
        grad = gradient(w[0], w[1])
        velocity = momentum * velocity - lr * grad
        w = w + velocity
        path.append(w.copy())
    return np.array(path)


def adam_optimizer(w, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, steps=100):
    """Adam: adaptive learning rate + momentum."""
    path = [w.copy()]
    m = np.zeros_like(w)  # First moment (mean)
    v = np.zeros_like(w)  # Second moment (variance)
    for t in range(1, steps + 1):
        grad = gradient(w[0], w[1])
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(w.copy())
    return np.array(path)


def rmsprop_optimizer(w, lr=0.01, decay=0.9, eps=1e-8, steps=100):
    """RMSProp: adaptive learning rate berdasarkan moving avg gradien^2."""
    path = [w.copy()]
    cache = np.zeros_like(w)
    for _ in range(steps):
        grad = gradient(w[0], w[1])
        cache = decay * cache + (1 - decay) * grad ** 2
        w = w - lr * grad / (np.sqrt(cache) + eps)
        path.append(w.copy())
    return np.array(path)


def visualisasi_optimizer_trajectory():
    """
    Menampilkan trajectory setiap optimizer pada loss landscape.
    Optimizer yang baik menuju minimum dengan cepat dan stabil.
    """
    w0 = np.array([4.0, 3.0])

    paths = {
        'SGD (lr=0.02)': sgd_optimizer(w0.copy(), lr=0.02),
        'SGD+Mom (lr=0.02)': sgd_momentum_optimizer(w0.copy(), lr=0.02),
        'Adam (lr=0.2)': adam_optimizer(w0.copy(), lr=0.2),
        'RMSProp (lr=0.1)': rmsprop_optimizer(w0.copy(), lr=0.1),
    }

    # Buat contour plot
    w1 = np.linspace(-5, 5, 200)
    w2 = np.linspace(-4, 4, 200)
    W1, W2 = np.meshgrid(w1, w2)
    L = loss_function(W1, W2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = ['red', 'blue', 'green', 'orange']

    for i, (nama, path) in enumerate(paths.items()):
        ax = axes[i // 2, i % 2]
        ax.contour(W1, W2, L, levels=30, cmap='gray', alpha=0.5)
        ax.plot(path[:, 0], path[:, 1], f'{colors[i][0]}o-', markersize=3, linewidth=1.5,
                label=nama)
        ax.plot(path[0, 0], path[0, 1], 'ko', markersize=10, label='Start')
        ax.plot(0, 0, 'r*', markersize=15, label='Minimum')
        ax.set_title(f"{nama} ({len(path)} steps)")
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")
        ax.legend(fontsize=8)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4, 4)

    plt.suptitle("Percobaan 16: Trajectory Optimizer pada Loss Landscape", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_optimizer_trajectory.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/16_optimizer_trajectory.png")
    return paths


def visualisasi_konvergensi(paths):
    """
    Membandingkan kecepatan konvergensi setiap optimizer.
    Plot loss vs iterasi untuk melihat mana yang tercepat.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'orange']

    for i, (nama, path) in enumerate(paths.items()):
        losses = [loss_function(p[0], p[1]) for p in path]
        ax.plot(losses, color=colors[i], linewidth=2, label=nama)

    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Loss")
    ax.set_title("Kecepatan Konvergensi Optimizer")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_konvergensi_optimizer.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/16_konvergensi_optimizer.png")


def visualisasi_learning_rate_effect():
    """
    Menunjukkan efek learning rate: terlalu kecil (lambat), optimal, terlalu besar (diverge).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lrs = [0.001, 0.02, 0.1]
    titles = ["LR=0.001 (Terlalu Kecil)", "LR=0.02 (Optimal)", "LR=0.1 (Terlalu Besar)"]

    w1 = np.linspace(-5, 5, 200)
    w2 = np.linspace(-4, 4, 200)
    W1, W2 = np.meshgrid(w1, w2)
    L = loss_function(W1, W2)

    for i, (lr, title) in enumerate(zip(lrs, titles)):
        path = sgd_optimizer(np.array([4.0, 3.0]), lr=lr, steps=50)
        axes[i].contour(W1, W2, L, levels=30, cmap='gray', alpha=0.5)
        axes[i].plot(path[:, 0], path[:, 1], 'ro-', markersize=3, linewidth=1.5)
        axes[i].set_title(title)
        axes[i].set_xlim(-5, 5)
        axes[i].set_ylim(-4, 4)

    plt.suptitle("Efek Learning Rate pada SGD", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_learning_rate_effect.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/16_learning_rate_effect.png")


def main():
    """Fungsi utama: visualisasi optimizer."""
    print("=" * 60)
    print("PERCOBAAN 16: OPTIMIZER VISUALISASI")
    print("=" * 60)

    print("\n--- 1. Trajectory Optimizer ---")
    paths = visualisasi_optimizer_trajectory()

    print("\n--- 2. Konvergensi Optimizer ---")
    visualisasi_konvergensi(paths)

    print("\n--- 3. Efek Learning Rate ---")
    visualisasi_learning_rate_effect()

    for nama, path in paths.items():
        final_loss = loss_function(path[-1, 0], path[-1, 1])
        print(f"  {nama:25s} -> Final loss: {final_loss:.6f}")

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 16")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. SGD paling sederhana tapi bisa lambat di elongated valley
2. Momentum mempercepat SGD dengan 'inersia'
3. Adam paling populer karena adaptive dan efektif
4. Learning rate harus dipilih dengan tepat (tidak terlalu besar/kecil)
5. Optimizer yang tepat mempercepat training secara signifikan

Output: output/16_optimizer_trajectory.png, output/16_konvergensi_optimizer.png,
        output/16_learning_rate_effect.png
""")


if __name__ == "__main__":
    main()
