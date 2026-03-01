"""
==========================================================================
PERCOBAAN 9: BACKPROPAGATION - VISUALISASI
==========================================================================
Program ini mengimplementasikan jaringan saraf tiruan (neural network)
sederhana 2-layer dari nol menggunakan NumPy saja, tanpa framework
deep learning. Jaringan dilatih pada masalah XOR untuk mendemonstrasikan
proses pembelajaran (learning). Visualisasi meliputi evolusi decision
boundary, kurva loss, dan perubahan bobot selama pelatihan.

Fungsi utama yang dipelajari:
- Operasi matriks NumPy    : np.dot(), np.random, np.array
- Fungsi aktivasi sigmoid  : 1 / (1 + exp(-x))
- Turunan sigmoid          : sigmoid(x) * (1 - sigmoid(x))
- Forward pass             : Propagasi input melalui jaringan
- Backward pass            : Menghitung gradien error (backpropagation)
- Gradient descent         : Memperbarui bobot berdasarkan gradien

Konsep yang dipelajari:
- Arsitektur neural network: input layer, hidden layer, output layer
- Proses forward propagation dan backward propagation
- Fungsi loss (Mean Squared Error)
- Gradient descent untuk optimasi bobot
- Masalah XOR sebagai non-linearly separable problem
- Evolusi decision boundary selama training
==========================================================================
"""

# Mengimpor NumPy untuk semua operasi komputasi neural network
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi hasil training
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan header percobaan
print("=" * 60)
print("PERCOBAAN 9: BACKPROPAGATION - VISUALISASI")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep backpropagation
# ============================================================
print("\n--- 1. Konsep Backpropagation ---")

# Menjelaskan konsep backpropagation
print("""
  Backpropagation adalah algoritma untuk melatih neural network:

  1. Forward Pass:
     - Input diteruskan melalui setiap layer
     - Setiap neuron menghitung: output = aktivasi(bobot * input + bias)
     - Hingga menghasilkan output prediksi

  2. Hitung Loss:
     - Membandingkan prediksi dengan target (ground truth)
     - Loss = Mean Squared Error (MSE)

  3. Backward Pass:
     - Menghitung gradien error terhadap setiap bobot
     - Menggunakan chain rule (aturan rantai) dari kalkulus
     - Gradien mengalir mundur dari output ke input

  4. Update Bobot:
     - bobot_baru = bobot_lama - learning_rate * gradien
     - Proses ini disebut Gradient Descent

  Masalah XOR: output = 1 jika input berbeda, 0 jika sama
  [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0
  Ini TIDAK bisa diselesaikan oleh perceptron tunggal (linear)!
""")

# ============================================================
# 2. Mendefinisikan fungsi aktivasi
# ============================================================
print("\n--- 2. Mendefinisikan Fungsi Aktivasi ---")

def sigmoid(x):
    """
    Fungsi aktivasi sigmoid: mengubah nilai ke range (0, 1).
    sigma(x) = 1 / (1 + exp(-x))
    """
    # Menerapkan clipping untuk menghindari overflow
    x = np.clip(x, -500, 500)

    # Menghitung dan mengembalikan nilai sigmoid
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_turunan(x):
    """
    Turunan fungsi sigmoid: digunakan dalam backpropagation.
    sigma'(x) = sigma(x) * (1 - sigma(x))
    """
    # Menghitung sigmoid terlebih dahulu
    s = sigmoid(x)

    # Menghitung dan mengembalikan turunan sigmoid
    return s * (1.0 - s)


# Menampilkan contoh sigmoid dan turunannya
contoh_input = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

# Menghitung output sigmoid untuk contoh input
contoh_sigmoid = sigmoid(contoh_input)

# Menghitung turunan sigmoid untuk contoh input
contoh_turunan = sigmoid_turunan(contoh_input)

# Menampilkan tabel contoh
print("  Contoh fungsi sigmoid dan turunannya:")
print(f"  {'Input':>8s} {'Sigmoid':>10s} {'Turunan':>10s}")
for inp, sig, tur in zip(contoh_input, contoh_sigmoid, contoh_turunan):
    print(f"  {inp:8.1f} {sig:10.4f} {tur:10.4f}")

# ============================================================
# 3. Menyiapkan data XOR
# ============================================================
print("\n--- 3. Menyiapkan Data XOR ---")

# Mendefinisikan input XOR (4 sampel, 2 fitur)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Mendefinisikan target XOR (4 sampel, 1 output)
y = np.array([[0],
              [1],
              [1],
              [0]])

# Menampilkan data XOR
print("  Data XOR:")
print(f"  {'Input 1':>8s} {'Input 2':>8s} {'Target':>8s}")
for i in range(len(X)):
    print(f"  {X[i,0]:8d} {X[i,1]:8d} {y[i,0]:8d}")

# ============================================================
# 4. Inisialisasi bobot neural network
# ============================================================
print("\n--- 4. Inisialisasi Neural Network ---")

# Mendefinisikan arsitektur jaringan
jumlah_input = 2     # 2 neuron input (fitur XOR)
jumlah_hidden = 4    # 4 neuron hidden layer
jumlah_output = 1    # 1 neuron output

# Mengatur seed random untuk reproduksibilitas
np.random.seed(42)

# Menginisialisasi bobot layer 1 (input -> hidden) secara random
# Shape: (2, 4) - setiap input terhubung ke setiap hidden neuron
W1 = np.random.randn(jumlah_input, jumlah_hidden) * 0.5

# Menginisialisasi bias layer 1
# Shape: (1, 4) - satu bias per hidden neuron
b1 = np.zeros((1, jumlah_hidden))

# Menginisialisasi bobot layer 2 (hidden -> output) secara random
# Shape: (4, 1) - setiap hidden neuron terhubung ke output
W2 = np.random.randn(jumlah_hidden, jumlah_output) * 0.5

# Menginisialisasi bias layer 2
# Shape: (1, 1) - satu bias untuk output neuron
b2 = np.zeros((1, jumlah_output))

# Menampilkan informasi arsitektur
print(f"  Arsitektur: {jumlah_input} -> {jumlah_hidden} -> {jumlah_output}")
print(f"  Bobot W1 shape: {W1.shape}")
print(f"  Bias  b1 shape: {b1.shape}")
print(f"  Bobot W2 shape: {W2.shape}")
print(f"  Bias  b2 shape: {b2.shape}")
print(f"  Total parameter: {W1.size + b1.size + W2.size + b2.size}")

# ============================================================
# 5. Implementasi forward pass dan backward pass
# ============================================================
print("\n--- 5. Implementasi Forward & Backward Pass ---")

def forward_pass(X, W1, b1, W2, b2):
    """
    Forward pass: menghitung output jaringan dari input.
    Menyimpan semua nilai antara untuk backward pass.
    """
    # Menghitung input ke hidden layer: z1 = X @ W1 + b1
    z1 = np.dot(X, W1) + b1

    # Menerapkan fungsi aktivasi sigmoid pada hidden layer
    a1 = sigmoid(z1)

    # Menghitung input ke output layer: z2 = a1 @ W2 + b2
    z2 = np.dot(a1, W2) + b2

    # Menerapkan fungsi aktivasi sigmoid pada output layer
    a2 = sigmoid(z2)

    # Menyimpan cache untuk backward pass
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    # Mengembalikan output dan cache
    return a2, cache


def backward_pass(X, y, cache, W1, b1, W2, b2):
    """
    Backward pass: menghitung gradien error terhadap semua bobot.
    Menggunakan chain rule untuk propagasi gradien mundur.
    """
    # Mengambil jumlah sampel
    m = X.shape[0]

    # Mengambil nilai dari cache
    a1 = cache['a1']
    a2 = cache['a2']
    z1 = cache['z1']

    # --- Menghitung gradien di output layer ---
    # Menghitung error output: selisih prediksi dengan target
    dz2 = a2 - y

    # Menghitung gradien bobot W2: dL/dW2 = a1^T @ dz2
    dW2 = np.dot(a1.T, dz2) / m

    # Menghitung gradien bias b2: dL/db2 = mean(dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # --- Menghitung gradien di hidden layer ---
    # Mempropagasi error ke hidden layer: dz1 = dz2 @ W2^T * sigmoid'(z1)
    dz1 = np.dot(dz2, W2.T) * sigmoid_turunan(z1)

    # Menghitung gradien bobot W1: dL/dW1 = X^T @ dz1
    dW1 = np.dot(X.T, dz1) / m

    # Menghitung gradien bias b1: dL/db1 = mean(dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # Menyimpan semua gradien
    gradien = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    # Mengembalikan gradien
    return gradien


def hitung_loss(y_pred, y_true):
    """
    Menghitung Mean Squared Error (MSE) loss.
    Loss = (1/n) * sum((prediksi - target)^2)
    """
    # Menghitung dan mengembalikan MSE
    return np.mean((y_pred - y_true) ** 2)


# Menampilkan penjelasan forward dan backward pass
print("  Forward Pass:")
print("    z1 = X @ W1 + b1 -> a1 = sigmoid(z1)")
print("    z2 = a1 @ W2 + b2 -> a2 = sigmoid(z2)")
print("  Backward Pass:")
print("    dz2 = a2 - y")
print("    dW2 = a1^T @ dz2 / m")
print("    dz1 = (dz2 @ W2^T) * sigmoid'(z1)")
print("    dW1 = X^T @ dz1 / m")

# ============================================================
# 6. Training neural network
# ============================================================
print("\n--- 6. Training Neural Network ---")

# Mendefinisikan hyperparameter training
learning_rate = 2.0    # Kecepatan belajar (agak besar untuk XOR)
jumlah_epoch = 10000   # Jumlah iterasi training

# Menyiapkan list untuk menyimpan riwayat training
riwayat_loss = []
riwayat_W1 = []
riwayat_W2 = []
riwayat_epoch_snapshot = []

# Mendefinisikan epoch-epoch untuk snapshot decision boundary
epoch_snapshot = [0, 100, 500, 1000, 2000, 5000, 10000]

# Menampilkan hyperparameter
print(f"  Learning rate : {learning_rate}")
print(f"  Jumlah epoch  : {jumlah_epoch}")
print(f"  Snapshot epoch: {epoch_snapshot}")

# Memulai proses training
print("\n  Proses Training:")
for epoch in range(1, jumlah_epoch + 1):
    # --- Forward Pass ---
    # Menghitung output prediksi
    y_pred, cache = forward_pass(X, W1, b1, W2, b2)

    # Menghitung loss (MSE)
    loss = hitung_loss(y_pred, y)

    # Menyimpan loss ke riwayat
    riwayat_loss.append(loss)

    # --- Backward Pass ---
    # Menghitung gradien semua bobot
    gradien = backward_pass(X, y, cache, W1, b1, W2, b2)

    # --- Update Bobot (Gradient Descent) ---
    # Memperbarui bobot W1
    W1 = W1 - learning_rate * gradien['dW1']

    # Memperbarui bias b1
    b1 = b1 - learning_rate * gradien['db1']

    # Memperbarui bobot W2
    W2 = W2 - learning_rate * gradien['dW2']

    # Memperbarui bias b2
    b2 = b2 - learning_rate * gradien['db2']

    # Menyimpan snapshot bobot untuk visualisasi
    if epoch in epoch_snapshot or epoch == 1:
        # Menyimpan salinan bobot W1
        riwayat_W1.append(W1.copy())

        # Menyimpan salinan bobot W2
        riwayat_W2.append(W2.copy())

        # Menyimpan epoch snapshot
        riwayat_epoch_snapshot.append(epoch)

    # Menampilkan progress setiap 2000 epoch
    if epoch % 2000 == 0 or epoch == 1:
        print(f"    Epoch {epoch:5d}/{jumlah_epoch}: Loss = {loss:.6f}")

# ============================================================
# 7. Evaluasi hasil training
# ============================================================
print("\n--- 7. Evaluasi Hasil Training ---")

# Menghitung output final setelah training
y_final, _ = forward_pass(X, W1, b1, W2, b2)

# Menampilkan hasil prediksi vs target
print(f"  {'Input':>10s} {'Target':>8s} {'Prediksi':>10s} {'Rounded':>8s}")
for i in range(len(X)):
    # Membulatkan prediksi ke 0 atau 1
    prediksi_bulat = round(y_final[i, 0])
    print(f"  {str(X[i]):>10s} {y[i,0]:8d} {y_final[i,0]:10.4f} {prediksi_bulat:8d}")

# Menghitung akurasi
akurasi = np.mean(np.round(y_final) == y) * 100
print(f"\n  Akurasi: {akurasi:.0f}%")
print(f"  Loss akhir: {riwayat_loss[-1]:.6f}")

# ============================================================
# 8. Visualisasi kurva loss
# ============================================================
print("\n--- 8. Visualisasi Kurva Loss ---")

# Membuat figure untuk kurva loss
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Subplot 1: Kurva loss skala normal ---
# Memplot kurva loss selama training
axes[0].plot(range(1, len(riwayat_loss) + 1), riwayat_loss,
             color='blue', linewidth=1.5, alpha=0.8)

# Mengatur judul subplot
axes[0].set_title("Kurva Loss (Skala Normal)", fontsize=12, fontweight='bold')

# Mengatur label sumbu X
axes[0].set_xlabel("Epoch", fontsize=10)

# Mengatur label sumbu Y
axes[0].set_ylabel("Loss (MSE)", fontsize=10)

# Menambahkan grid
axes[0].grid(True, alpha=0.3)

# --- Subplot 2: Kurva loss skala logaritmik ---
# Memplot kurva loss dalam skala log
axes[1].semilogy(range(1, len(riwayat_loss) + 1), riwayat_loss,
                 color='red', linewidth=1.5, alpha=0.8)

# Mengatur judul subplot
axes[1].set_title("Kurva Loss (Skala Log)", fontsize=12, fontweight='bold')

# Mengatur label sumbu X
axes[1].set_xlabel("Epoch", fontsize=10)

# Mengatur label sumbu Y
axes[1].set_ylabel("Loss (MSE) - Log Scale", fontsize=10)

# Menambahkan grid
axes[1].grid(True, alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 9: Kurva Loss Training Neural Network pada XOR",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi kurva loss
plt.savefig(os.path.join(OUTPUT_DIR, "09_loss_curve.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/09_loss_curve.png")

# ============================================================
# 9. Visualisasi evolusi decision boundary
# ============================================================
print("\n--- 9. Visualisasi Decision Boundary ---")

# Mendefinisikan grid untuk decision boundary
# Membuat range sumbu X dan Y dari -0.5 hingga 1.5
x_range = np.linspace(-0.5, 1.5, 200)
y_range = np.linspace(-0.5, 1.5, 200)

# Membuat meshgrid dari kedua range
xx, yy = np.meshgrid(x_range, y_range)

# Membentuk array titik grid (setiap baris = satu titik)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Mendefinisikan epoch yang akan divisualisasikan (6 snapshot)
epoch_tampil = [0, 100, 500, 1000, 5000, jumlah_epoch]

# Membuat figure dengan 2x3 subplot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Meratakan array axes
axes_flat = axes.flatten()

# Membuat salinan bobot awal untuk re-training snapshot
np.random.seed(42)
W1_init = np.random.randn(jumlah_input, jumlah_hidden) * 0.5
b1_init = np.zeros((1, jumlah_hidden))
W2_init = np.random.randn(jumlah_hidden, jumlah_output) * 0.5
b2_init = np.zeros((1, jumlah_output))

# Menyalin bobot awal untuk training ulang
W1_t = W1_init.copy()
b1_t = b1_init.copy()
W2_t = W2_init.copy()
b2_t = b2_init.copy()

# Membuat variabel untuk menyimpan bobot di epoch snapshot
snapshot_bobot = {}

# Menyimpan bobot awal (epoch 0)
snapshot_bobot[0] = (W1_t.copy(), b1_t.copy(), W2_t.copy(), b2_t.copy())

# Melakukan training ulang dan menyimpan snapshot
for epoch in range(1, jumlah_epoch + 1):
    # Forward pass
    y_pred_t, cache_t = forward_pass(X, W1_t, b1_t, W2_t, b2_t)

    # Backward pass
    grad_t = backward_pass(X, y, cache_t, W1_t, b1_t, W2_t, b2_t)

    # Update bobot
    W1_t = W1_t - learning_rate * grad_t['dW1']
    b1_t = b1_t - learning_rate * grad_t['db1']
    W2_t = W2_t - learning_rate * grad_t['dW2']
    b2_t = b2_t - learning_rate * grad_t['db2']

    # Menyimpan snapshot jika epoch sesuai
    if epoch in epoch_tampil:
        snapshot_bobot[epoch] = (W1_t.copy(), b1_t.copy(), W2_t.copy(), b2_t.copy())

# Memvisualisasikan decision boundary untuk setiap snapshot
for idx, epoch_viz in enumerate(epoch_tampil):
    # Mengambil bobot snapshot
    if epoch_viz in snapshot_bobot:
        W1_s, b1_s, W2_s, b2_s = snapshot_bobot[epoch_viz]
    else:
        continue

    # Menghitung prediksi untuk seluruh grid
    z_grid, _ = forward_pass(grid_points, W1_s, b1_s, W2_s, b2_s)

    # Membentuk ulang prediksi ke shape grid
    z_grid = z_grid.reshape(xx.shape)

    # Memplot decision boundary sebagai contour
    axes_flat[idx].contourf(xx, yy, z_grid, levels=50, cmap='RdYlBu', alpha=0.8)

    # Menambahkan garis batas keputusan (0.5 threshold)
    axes_flat[idx].contour(xx, yy, z_grid, levels=[0.5],
                           colors='black', linewidths=2)

    # Memplot titik data XOR
    for i in range(len(X)):
        # Menentukan warna berdasarkan label
        warna = 'red' if y[i, 0] == 0 else 'blue'
        # Menentukan marker berdasarkan label
        marker = 'o' if y[i, 0] == 0 else 's'
        # Memplot titik data
        axes_flat[idx].scatter(X[i, 0], X[i, 1], c=warna, marker=marker,
                               s=200, edgecolors='black', linewidths=2,
                               zorder=5)

    # Mengatur judul subplot
    axes_flat[idx].set_title(f"Epoch {epoch_viz}", fontsize=12,
                             fontweight='bold')

    # Mengatur label sumbu
    axes_flat[idx].set_xlabel("Input 1", fontsize=9)
    axes_flat[idx].set_ylabel("Input 2", fontsize=9)

    # Mengatur batas sumbu
    axes_flat[idx].set_xlim(-0.5, 1.5)
    axes_flat[idx].set_ylim(-0.5, 1.5)

# Menambahkan judul utama
plt.suptitle("Percobaan 9: Evolusi Decision Boundary Neural Network (XOR)\n"
             "Merah=0, Biru=1 | Garis hitam = batas keputusan (0.5)",
             fontsize=13, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi decision boundary
plt.savefig(os.path.join(OUTPUT_DIR, "09_decision_boundary.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/09_decision_boundary.png")

# ============================================================
# 10. Visualisasi evolusi bobot
# ============================================================
print("\n--- 10. Visualisasi Evolusi Bobot ---")

# Melakukan training ulang dan mencatat bobot setiap epoch
np.random.seed(42)
W1_e = np.random.randn(jumlah_input, jumlah_hidden) * 0.5
b1_e = np.zeros((1, jumlah_hidden))
W2_e = np.random.randn(jumlah_hidden, jumlah_output) * 0.5
b2_e = np.zeros((1, jumlah_output))

# Menyiapkan array untuk menyimpan riwayat bobot per epoch
riwayat_W1_all = np.zeros((jumlah_epoch, jumlah_input * jumlah_hidden))
riwayat_W2_all = np.zeros((jumlah_epoch, jumlah_hidden * jumlah_output))
riwayat_b1_all = np.zeros((jumlah_epoch, jumlah_hidden))
riwayat_b2_all = np.zeros((jumlah_epoch, jumlah_output))

# Melakukan training dan menyimpan bobot setiap epoch
for epoch in range(jumlah_epoch):
    # Menyimpan bobot saat ini
    riwayat_W1_all[epoch] = W1_e.flatten()
    riwayat_W2_all[epoch] = W2_e.flatten()
    riwayat_b1_all[epoch] = b1_e.flatten()
    riwayat_b2_all[epoch] = b2_e.flatten()

    # Forward pass
    y_p, cache_e = forward_pass(X, W1_e, b1_e, W2_e, b2_e)

    # Backward pass
    grad_e = backward_pass(X, y, cache_e, W1_e, b1_e, W2_e, b2_e)

    # Update bobot
    W1_e = W1_e - learning_rate * grad_e['dW1']
    b1_e = b1_e - learning_rate * grad_e['db1']
    W2_e = W2_e - learning_rate * grad_e['dW2']
    b2_e = b2_e - learning_rate * grad_e['db2']

# Membuat figure untuk evolusi bobot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Subplot 1: Evolusi bobot W1 ---
# Memplot setiap elemen bobot W1
for i in range(riwayat_W1_all.shape[1]):
    axes[0, 0].plot(range(jumlah_epoch), riwayat_W1_all[:, i],
                    alpha=0.7, linewidth=1.0, label=f"W1[{i}]")

# Mengatur judul
axes[0, 0].set_title("Evolusi Bobot W1 (Input -> Hidden)",
                      fontsize=11, fontweight='bold')

# Mengatur label
axes[0, 0].set_xlabel("Epoch", fontsize=10)
axes[0, 0].set_ylabel("Nilai Bobot", fontsize=10)

# Menambahkan legend
axes[0, 0].legend(fontsize=7, ncol=2)

# Menambahkan grid
axes[0, 0].grid(True, alpha=0.3)

# --- Subplot 2: Evolusi bobot W2 ---
# Memplot setiap elemen bobot W2
for i in range(riwayat_W2_all.shape[1]):
    axes[0, 1].plot(range(jumlah_epoch), riwayat_W2_all[:, i],
                    alpha=0.7, linewidth=1.5, label=f"W2[{i}]")

# Mengatur judul
axes[0, 1].set_title("Evolusi Bobot W2 (Hidden -> Output)",
                      fontsize=11, fontweight='bold')

# Mengatur label
axes[0, 1].set_xlabel("Epoch", fontsize=10)
axes[0, 1].set_ylabel("Nilai Bobot", fontsize=10)

# Menambahkan legend
axes[0, 1].legend(fontsize=8)

# Menambahkan grid
axes[0, 1].grid(True, alpha=0.3)

# --- Subplot 3: Evolusi bias b1 ---
# Memplot setiap elemen bias b1
for i in range(riwayat_b1_all.shape[1]):
    axes[1, 0].plot(range(jumlah_epoch), riwayat_b1_all[:, i],
                    alpha=0.7, linewidth=1.0, label=f"b1[{i}]")

# Mengatur judul
axes[1, 0].set_title("Evolusi Bias b1 (Hidden Layer)",
                      fontsize=11, fontweight='bold')

# Mengatur label
axes[1, 0].set_xlabel("Epoch", fontsize=10)
axes[1, 0].set_ylabel("Nilai Bias", fontsize=10)

# Menambahkan legend
axes[1, 0].legend(fontsize=8)

# Menambahkan grid
axes[1, 0].grid(True, alpha=0.3)

# --- Subplot 4: Evolusi bias b2 ---
# Memplot elemen bias b2
axes[1, 1].plot(range(jumlah_epoch), riwayat_b2_all[:, 0],
                alpha=0.7, linewidth=1.5, color='purple', label="b2[0]")

# Mengatur judul
axes[1, 1].set_title("Evolusi Bias b2 (Output Layer)",
                      fontsize=11, fontweight='bold')

# Mengatur label
axes[1, 1].set_xlabel("Epoch", fontsize=10)
axes[1, 1].set_ylabel("Nilai Bias", fontsize=10)

# Menambahkan legend
axes[1, 1].legend(fontsize=8)

# Menambahkan grid
axes[1, 1].grid(True, alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 9: Evolusi Bobot dan Bias Neural Network Selama Training",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi evolusi bobot
plt.savefig(os.path.join(OUTPUT_DIR, "09_weight_evolution.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/09_weight_evolution.png")

# ============================================================
# 11. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 9")
print("=" * 60)
print(f"""
Konsep yang telah dipelajari:
1. Neural network 2-layer diimplementasikan dari nol dengan NumPy
2. Fungsi aktivasi sigmoid: sigma(x) = 1/(1+exp(-x))
3. Forward pass: menghitung output dari input melalui setiap layer
4. Backward pass: menghitung gradien error (backpropagation)
5. Gradient descent: memperbarui bobot dengan learning rate
6. Masalah XOR membutuhkan hidden layer (non-linear)
7. Decision boundary berevolusi dari linear menjadi non-linear
8. Loss menurun secara monoton selama training
9. Bobot konvergen ke nilai optimal setelah cukup epoch

Hasil Training:
- Akurasi akhir: {akurasi:.0f}%
- Loss akhir   : {riwayat_loss[-1]:.6f}
- Epoch total  : {jumlah_epoch}
- Learning rate : {learning_rate}

Output disimpan di folder: output/
- 09_loss_curve.png       : Kurva loss selama training
- 09_decision_boundary.png: Evolusi decision boundary
- 09_weight_evolution.png : Evolusi bobot dan bias
""")
