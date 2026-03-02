"""
==========================================================================
PERCOBAAN 9: BACKPROPAGATION VISUALISASI
==========================================================================
Program ini memvisualisasikan proses backpropagation pada neural network
sederhana. Backpropagation menghitung gradien loss terhadap setiap
weight, yang kemudian digunakan untuk update weight saat training.

Konsep yang dipelajari:
- Forward pass: menghitung output dari input
- Loss function: mengukur kesalahan prediksi
- Backward pass: menghitung gradien dengan chain rule
- Weight update: w = w - lr * gradient

Referensi: Szeliski Ch.5, Géron Ch.10
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


def sigmoid(x):
    """Fungsi aktivasi sigmoid: f(x) = 1 / (1 + e^-x)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_deriv(x):
    """Turunan sigmoid: f'(x) = f(x) * (1 - f(x))."""
    s = sigmoid(x)
    return s * (1 - s)


def forward_pass(X, W1, b1, W2, b2):
    """
    Forward pass melalui network 2-layer:
    Input -> Hidden (sigmoid) -> Output (sigmoid)
    Mengembalikan output dan cache untuk backprop.
    """
    z1 = X.dot(W1) + b1           # Linear layer 1
    a1 = sigmoid(z1)               # Aktivasi layer 1
    z2 = a1.dot(W2) + b2          # Linear layer 2
    a2 = sigmoid(z2)               # Aktivasi layer 2 (output)
    cache = (X, z1, a1, z2, a2, W1, W2)
    return a2, cache


def backward_pass(y_true, cache, lr=0.1):
    """
    Backward pass: menghitung gradien dan update weights.
    Menggunakan chain rule untuk propagasi gradien.
    """
    X, z1, a1, z2, a2, W1, W2 = cache
    m = X.shape[0]

    # Gradien output layer
    dz2 = (a2 - y_true) * sigmoid_deriv(z2)
    dW2 = a1.T.dot(dz2) / m
    db2 = np.mean(dz2, axis=0)

    # Gradien hidden layer
    dz1 = dz2.dot(W2.T) * sigmoid_deriv(z1)
    dW1 = X.T.dot(dz1) / m
    db1 = np.mean(dz1, axis=0)

    # Update weights
    W1_new = W1 - lr * dW1
    W2_new = W2 - lr * dW2
    b1_new = b1 - lr * db1
    b2_new = b2 - lr * db2

    gradients = {'dW1': dW1, 'dW2': dW2, 'db1': db1, 'db2': db2}
    return W1_new, b1_new, W2_new, b2_new, gradients


def hitung_loss(y_pred, y_true):
    """Menghitung Mean Squared Error loss."""
    return np.mean((y_pred - y_true) ** 2)


def training_loop(X, y, epochs=500, lr=0.5):
    """
    Training loop: forward -> loss -> backward -> update.
    Melatih network untuk belajar fungsi XOR.
    """
    np.random.seed(42)
    W1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros(4)
    W2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros(1)

    loss_history = []
    grad_history = []

    for epoch in range(epochs):
        # Forward
        y_pred, cache = forward_pass(X, W1, b1, W2, b2)
        loss = hitung_loss(y_pred, y)
        loss_history.append(loss)

        # Backward
        W1, b1, W2, b2, grads = backward_pass(y, cache, lr)
        grad_history.append(np.mean(np.abs(grads['dW1'])))

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss:.6f}")

    return loss_history, grad_history, W1, b1, W2, b2


def visualisasi_loss_gradient(loss_history, grad_history):
    """
    Menampilkan kurva loss dan gradien selama training.
    Loss menurun dan gradien mengecil seiring training berlangsung.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(loss_history, 'b-', linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Loss selama Training")
    ax1.grid(True, alpha=0.3)

    ax2.plot(grad_history, 'r-', linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("|Gradien| rata-rata")
    ax2.set_title("Magnitude Gradien selama Training")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Percobaan 9: Backpropagation Training", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_backprop_training.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/09_backprop_training.png")


def visualisasi_decision_boundary(W1, b1, W2, b2):
    """
    Menampilkan decision boundary yang dipelajari network untuk XOR.
    Menunjukkan bahwa network non-linear dapat memisahkan data XOR.
    """
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred, _ = forward_pass(grid, W1, b1, W2, b2)
    pred = pred.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, pred, levels=50, cmap='RdYlGn', alpha=0.8)
    ax.contour(xx, yy, pred, levels=[0.5], colors='black', linewidths=2)

    # Data points XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', s=200, edgecolors='black', label='Kelas 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='green', s=200, edgecolors='black', label='Kelas 1')
    ax.set_title("Decision Boundary XOR (setelah training)")
    ax.legend()
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_decision_boundary.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/09_decision_boundary.png")


def main():
    """Fungsi utama: visualisasi backpropagation."""
    print("=" * 60)
    print("PERCOBAAN 9: BACKPROPAGATION VISUALISASI")
    print("=" * 60)

    # Dataset XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    print("\n--- 1. Training Network untuk XOR ---")
    loss_history, grad_history, W1, b1, W2, b2 = training_loop(X, y, epochs=500, lr=2.0)

    print("\n--- 2. Hasil Prediksi ---")
    y_pred, _ = forward_pass(X, W1, b1, W2, b2)
    for i in range(len(X)):
        print(f"  Input: {X[i]} -> Prediksi: {y_pred[i][0]:.4f}, Target: {y[i][0]}")

    print("\n--- 3. Visualisasi Loss & Gradien ---")
    visualisasi_loss_gradient(loss_history, grad_history)

    print("\n--- 4. Decision Boundary ---")
    visualisasi_decision_boundary(W1, b1, W2, b2)

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 9")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. Forward pass menghitung output dari input melalui layers
2. Loss function mengukur kesalahan prediksi vs target
3. Backpropagation menghitung gradien menggunakan chain rule
4. Weight update: w = w - lr * gradient
5. Network 2-layer dapat mempelajari fungsi non-linear (XOR)

Output: output/09_backprop_training.png, output/09_decision_boundary.png
""")


if __name__ == "__main__":
    main()
