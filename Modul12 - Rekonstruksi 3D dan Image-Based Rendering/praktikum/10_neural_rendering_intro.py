"""
==========================================================
PERCOBAAN 10: PENGENALAN NEURAL RENDERING
Mempelajari konsep dasar neural scene representation.
Implementasi sederhana: representasi gambar sebagai neural
network (image regression) menggunakan NumPy.

Catatan: Ini adalah versi simplified tanpa GPU/PyTorch.
Menggunakan pendekatan manual untuk memahami konsep dasar
neural rendering.

Konsep Utama:
- NeRF (Neural Radiance Fields): Merepresentasikan scene
  3D sebagai fungsi kontinu F(x,y,z,theta,phi) -> (RGB, sigma)
  yang dioptimasi oleh neural network. Setiap koordinat 3D +
  arah pandang menghasilkan warna dan densitas.
- 3D Gaussian Splatting (3DGS): Merepresentasikan scene
  sebagai kumpulan 3D Gaussian primitives yang di-render
  melalui differentiable rasterization, lebih cepat dari NeRF.

Fungsi utama:
- numpy neural network operations (forward pass, backprop)
- cv2.resize()
- matplotlib visualization
==========================================================
"""

# Mengimpor library numpy untuk operasi numerik
import numpy as np

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor library os untuk operasi sistem file
import os

# Mengimpor library matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mengimpor library time untuk mengukur waktu training
import time

# ========================================================
# KONFIGURASI DIREKTORI
# ========================================================

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Menentukan direktori untuk gambar/data input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Menentukan direktori untuk menyimpan hasil output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat direktori output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mencetak header utama percobaan
print("=" * 60)
print("PERCOBAAN 10: PENGENALAN NEURAL RENDERING")
print("=" * 60)
print()


# ========================================================
# 1. KONSEP NERF DAN 3DGS
# ========================================================
print("=" * 60)
print("1. KONSEP NERF DAN 3D GAUSSIAN SPLATTING")
print("=" * 60)

# Mencetak penjelasan konsep NeRF
print("  [NeRF - Neural Radiance Fields]")
print("  - Input: koordinat 3D (x,y,z) + arah pandang (theta, phi)")
print("  - Output: warna RGB + densitas (opacity)")
print("  - Network: MLP dengan positional encoding")
print("  - Rendering: volume rendering (ray marching)")
print()

# Mencetak penjelasan konsep 3DGS
print("  [3D Gaussian Splatting]")
print("  - Representasi: kumpulan Gaussian 3D (mean, covariance, color, opacity)")
print("  - Rendering: differentiable rasterization (splat ke layar)")
print("  - Keunggulan: real-time rendering, training lebih cepat")
print()

# Mencetak penjelasan percobaan ini
print("  [Percobaan ini]")
print("  - Kita akan melatih MLP sederhana untuk 'menghafal' satu gambar")
print("  - Input: koordinat piksel (x, y) -> Output: warna RGB")
print("  - Ini mendemonstrasikan ide dasar neural scene representation")
print()


# ========================================================
# 2. IMPLEMENTASI MLP 2-LAYER DARI SCRATCH
# ========================================================
print("=" * 60)
print("2. IMPLEMENTASI MLP 2-LAYER (NUMPY ONLY)")
print("=" * 60)


class SimpleMLP:
    """
    MLP sederhana 2 layer untuk image regression.
    Input: (x, y) terenkode -> Output: (R, G, B)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001):
        """Menginisialisasi bobot MLP dengan Xavier initialization."""
        # Menyimpan learning rate
        self.lr = lr

        # Menginisialisasi bobot layer 1 (input -> hidden) dengan Xavier
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)

        # Menginisialisasi bias layer 1
        self.b1 = np.zeros((1, hidden_dim))

        # Menginisialisasi bobot layer 2 (hidden -> output) dengan Xavier
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)

        # Menginisialisasi bias layer 2
        self.b2 = np.zeros((1, output_dim))

        # Mencetak arsitektur network
        print(f"  Arsitektur MLP:")
        print(f"    Input dim  : {input_dim}")
        print(f"    Hidden dim : {hidden_dim}")
        print(f"    Output dim : {output_dim}")
        print(f"    Learning rate: {lr}")

        # Menghitung total parameter
        total_params = (input_dim * hidden_dim + hidden_dim +
                        hidden_dim * output_dim + output_dim)
        print(f"    Total parameter: {total_params}")

    def relu(self, x):
        """Fungsi aktivasi ReLU."""
        # Mengembalikan max(0, x) element-wise
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Turunan fungsi ReLU."""
        # Mengembalikan 1 jika x > 0, 0 sebaliknya
        return (x > 0).astype(np.float64)

    def sigmoid(self, x):
        """Fungsi aktivasi Sigmoid untuk output [0,1]."""
        # Menghitung sigmoid dengan clipping untuk stabilitas numerik
        x_clip = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clip))

    def forward(self, X):
        """Forward pass melalui network."""
        # Menghitung pre-aktivasi layer 1
        self.z1 = X @ self.W1 + self.b1

        # Menerapkan aktivasi ReLU pada layer 1
        self.a1 = self.relu(self.z1)

        # Menghitung pre-aktivasi layer 2
        self.z2 = self.a1 @ self.W2 + self.b2

        # Menerapkan sigmoid untuk output RGB [0, 1]
        self.output = self.sigmoid(self.z2)

        # Mengembalikan output prediction
        return self.output

    def backward(self, X, y_true):
        """Backward pass (backpropagation) untuk update bobot."""
        # Menghitung jumlah sampel dalam batch
        m = X.shape[0]

        # Menghitung error output (MSE derivative * sigmoid derivative)
        d_output = (self.output - y_true) * self.output * (1 - self.output)

        # Menghitung gradien bobot layer 2
        dW2 = self.a1.T @ d_output / m

        # Menghitung gradien bias layer 2
        db2 = np.mean(d_output, axis=0, keepdims=True)

        # Menghitung error propagasi ke layer 1
        d_hidden = (d_output @ self.W2.T) * self.relu_derivative(self.z1)

        # Menghitung gradien bobot layer 1
        dW1 = X.T @ d_hidden / m

        # Menghitung gradien bias layer 1
        db1 = np.mean(d_hidden, axis=0, keepdims=True)

        # Memperbarui bobot layer 2
        self.W2 -= self.lr * dW2

        # Memperbarui bias layer 2
        self.b2 -= self.lr * db2

        # Memperbarui bobot layer 1
        self.W1 -= self.lr * dW1

        # Memperbarui bias layer 1
        self.b1 -= self.lr * db1

    def compute_loss(self, y_pred, y_true):
        """Menghitung Mean Squared Error loss."""
        # Menghitung MSE antara prediksi dan ground truth
        return np.mean((y_pred - y_true) ** 2)


print()


# ========================================================
# 3. MEMUAT DAN MENYIAPKAN GAMBAR TARGET
# ========================================================
print("=" * 60)
print("3. MEMUAT DAN MENYIAPKAN GAMBAR TARGET")
print("=" * 60)

# Menentukan path gambar target
target_path = os.path.join(IMAGE_DIR, "multiview_00.png")

# Menjalankan download_image.py otomatis jika gambar target tidak tersedia
if not os.path.exists(target_path):
    print("[WARN] Gambar target tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)

# Memuat gambar target
img_full = cv2.imread(target_path)
if img_full is None:
    raise FileNotFoundError(
        f"[ERROR] {target_path} tidak tersedia.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )
print(f"  Gambar target dimuat: {target_path}")

# Meresize gambar ke ukuran kecil untuk training yang feasible
target_size = 32

# Meresize gambar target
img_small = cv2.resize(img_full, (target_size, target_size))

# Mengkonversi BGR ke RGB
img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

# Menormalisasi nilai piksel ke [0, 1]
img_norm = img_rgb.astype(np.float64) / 255.0

# Mencetak informasi gambar target
print(f"  Ukuran asli: {img_full.shape}")
print(f"  Ukuran training: {target_size}x{target_size}")
print(f"  Total piksel: {target_size * target_size}")
print()


# ========================================================
# 4. POSITIONAL ENCODING KOORDINAT PIKSEL
# ========================================================
print("=" * 60)
print("4. POSITIONAL ENCODING KOORDINAT PIKSEL")
print("=" * 60)


def positional_encoding(coords, num_frequencies=4):
    """
    Menerapkan positional encoding pada koordinat.
    Mengubah (x, y) -> (x, y, sin(2^0*pi*x), cos(2^0*pi*x), ..., sin(2^L*pi*x), ...)
    Ini membantu network menangkap detail frekuensi tinggi.
    """
    # Menyiapkan list untuk fitur terenkode
    encoded = [coords]

    # Mengiterasi setiap frekuensi
    for freq in range(num_frequencies):
        # Menghitung faktor skala frekuensi
        scale = 2.0 ** freq * np.pi

        # Menambahkan sin dari koordinat * frekuensi
        encoded.append(np.sin(scale * coords))

        # Menambahkan cos dari koordinat * frekuensi
        encoded.append(np.cos(scale * coords))

    # Menggabungkan semua fitur secara horizontal
    return np.concatenate(encoded, axis=1)


# Membuat koordinat piksel yang dinormalisasi ke [-1, 1]
y_coords = np.linspace(-1, 1, target_size)
x_coords = np.linspace(-1, 1, target_size)

# Membuat meshgrid dari koordinat
xx, yy = np.meshgrid(x_coords, y_coords)

# Menggabungkan koordinat menjadi array (N, 2)
coords_raw = np.stack([xx.flatten(), yy.flatten()], axis=1)

# Menentukan jumlah frekuensi untuk positional encoding
num_freq = 4

# Menerapkan positional encoding pada koordinat
coords_encoded = positional_encoding(coords_raw, num_freq)

# Menghitung dimensi input setelah encoding
input_dim = coords_encoded.shape[1]

# Mencetak informasi encoding
print(f"  Koordinat raw: {coords_raw.shape} -> (N, 2)")
print(f"  Jumlah frekuensi: {num_freq}")
print(f"  Koordinat encoded: {coords_encoded.shape} -> (N, {input_dim})")
print(f"  Formula: (x,y) + sin/cos(2^k * pi * (x,y)) untuk k=0..{num_freq-1}")
print()

# Menyiapkan data target (RGB per piksel)
target_rgb = img_norm.reshape(-1, 3)

# Mencetak ukuran data target
print(f"  Data target: {target_rgb.shape} -> (N, 3)")
print()


# ========================================================
# 5. TRAINING MLP UNTUK MEREPRESENTASIKAN GAMBAR
# ========================================================
print("=" * 60)
print("5. TRAINING MLP UNTUK MEREPRESENTASIKAN GAMBAR")
print("=" * 60)

# Menentukan parameter training
hidden_dim = 64

# Menentukan jumlah epoch
num_epochs = 500

# Menentukan learning rate
learning_rate = 0.05

# Membuat instance MLP
mlp = SimpleMLP(input_dim, hidden_dim, 3, lr=learning_rate)

# Menyiapkan list untuk menyimpan loss history
loss_history = []

# Menyiapkan list untuk menyimpan prediksi pada epoch tertentu
snapshot_epochs = [10, 50, 100, 200, 350, 500]

# Menyiapkan dictionary untuk snapshot gambar
snapshots = {}

# Mencatat waktu mulai training
start_time = time.time()

# Melakukan training loop
print(f"\n  Training dimulai ({num_epochs} epoch)...")
for epoch in range(1, num_epochs + 1):
    # Forward pass: prediksi RGB dari koordinat
    y_pred = mlp.forward(coords_encoded)

    # Menghitung loss
    loss = mlp.compute_loss(y_pred, target_rgb)

    # Backward pass: update bobot
    mlp.backward(coords_encoded, target_rgb)

    # Menyimpan loss
    loss_history.append(loss)

    # Menyimpan snapshot pada epoch tertentu
    if epoch in snapshot_epochs:
        # Mereshape prediksi menjadi gambar
        pred_img = y_pred.reshape(target_size, target_size, 3)
        # Menyimpan snapshot
        snapshots[epoch] = pred_img.copy()

    # Mencetak progress setiap 50 epoch
    if epoch % 50 == 0 or epoch == 1:
        print(f"    Epoch {epoch:4d}/{num_epochs} | Loss: {loss:.6f}")

# Mencatat waktu selesai training
elapsed = time.time() - start_time

# Mencetak ringkasan training
print(f"\n  Training selesai dalam {elapsed:.2f} detik")
print(f"  Loss awal: {loss_history[0]:.6f}")
print(f"  Loss akhir: {loss_history[-1]:.6f}")
print()


# ========================================================
# 6. VISUALISASI LEARNED VS ORIGINAL PADA TIAP EPOCH
# ========================================================
print("=" * 60)
print("6. VISUALISASI LEARNED VS ORIGINAL PADA TIAP EPOCH")
print("=" * 60)

# Menentukan jumlah snapshot yang akan ditampilkan
n_snapshots = len(snapshot_epochs)

# Membuat figure untuk visualisasi progression
fig, axes = plt.subplots(2, n_snapshots + 1, figsize=(3 * (n_snapshots + 1), 6))

# Menampilkan gambar asli pada kolom pertama (baris atas)
axes[0, 0].imshow(img_norm)
axes[0, 0].set_title("Original", fontsize=10)
axes[0, 0].axis("off")

# Menampilkan gambar asli diperbesar pada baris bawah
img_upscaled = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
axes[1, 0].imshow(img_upscaled)
axes[1, 0].set_title("Original (upscaled)", fontsize=10)
axes[1, 0].axis("off")

# Mengiterasi setiap snapshot epoch
for i, ep in enumerate(snapshot_epochs):
    # Mendapatkan prediksi pada epoch ini
    if ep in snapshots:
        pred = np.clip(snapshots[ep], 0, 1)
    else:
        pred = np.zeros((target_size, target_size, 3))

    # Menampilkan prediksi pada baris atas
    axes[0, i + 1].imshow(pred)
    axes[0, i + 1].set_title(f"Epoch {ep}", fontsize=10)
    axes[0, i + 1].axis("off")

    # Menampilkan versi upscaled pada baris bawah
    pred_up = cv2.resize((pred * 255).astype(np.uint8), (256, 256),
                         interpolation=cv2.INTER_NEAREST)
    axes[1, i + 1].imshow(pred_up)
    axes[1, i + 1].set_title(f"Epoch {ep} (upscaled)", fontsize=10)
    axes[1, i + 1].axis("off")

# Mengatur layout
plt.suptitle("Training Progression: Neural Image Representation",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan visualisasi progression
prog_path = os.path.join(OUTPUT_DIR, "10_training_progression.png")
plt.savefig(prog_path, dpi=150, bbox_inches="tight")
plt.close()

# Mencetak konfirmasi
print(f"  Disimpan: {prog_path}")
print()


# ========================================================
# 7. OVERFITTING = MEMORIZATION = REPRESENTATION
# ========================================================
print("=" * 60)
print("7. OVERFITTING = MEMORIZATION = REPRESENTATION")
print("=" * 60)

# Mencetak penjelasan tentang konsep overfitting dalam neural rendering
print("  Dalam machine learning klasik, overfitting = buruk")
print("  Dalam neural rendering, overfitting = TUJUAN!")
print()
print("  Mengapa?")
print("  - Neural network 'menghafal' representasi scene")
print("  - Setiap piksel/koordinat dipelajari secara akurat")
print("  - Semakin rendah loss = semakin baik representasi")
print()

# Menghitung PSNR (Peak Signal-to-Noise Ratio) hasil akhir
final_pred = mlp.forward(coords_encoded)

# Mereshape prediksi menjadi gambar
final_img = np.clip(final_pred.reshape(target_size, target_size, 3), 0, 1)

# Menghitung MSE antara prediksi dan original
mse_final = np.mean((final_img - img_norm) ** 2)

# Menghitung PSNR
if mse_final > 0:
    psnr = 10 * np.log10(1.0 / mse_final)
else:
    psnr = float('inf')

# Mencetak metrik kualitas
print(f"  MSE akhir: {mse_final:.6f}")
print(f"  PSNR: {psnr:.2f} dB")
print()


# ========================================================
# 8. PERBANDINGAN NEURAL VS DIRECT LOOKUP
# ========================================================
print("=" * 60)
print("8. PERBANDINGAN NEURAL VS DIRECT LOOKUP")
print("=" * 60)

# Mencetak penjelasan mengapa neural representation berguna
print("  Direct lookup (table):")
print("    - Menyimpan RGB per piksel langsung di memori")
print("    - Memori: H x W x 3 bytes")
print("    - Tidak bisa melakukan interpolasi novel view")
print()
print("  Neural representation:")
print("    - Menyimpan bobot network (jauh lebih kompak)")
print("    - Bisa query posisi APAPUN (kontinu, bukan diskrit)")
print("    - Basis untuk novel view synthesis (NeRF/3DGS)")
print()

# Menghitung ukuran memori direct lookup
mem_direct = target_size * target_size * 3 * 4  # float32

# Menghitung ukuran memori neural network
mem_w1 = mlp.W1.size * 8  # float64
mem_b1 = mlp.b1.size * 8
mem_w2 = mlp.W2.size * 8
mem_b2 = mlp.b2.size * 8
mem_nn = mem_w1 + mem_b1 + mem_w2 + mem_b2

# Mencetak perbandingan memori
print(f"  Memori direct lookup ({target_size}x{target_size}): {mem_direct} bytes")
print(f"  Memori neural network: {mem_nn} bytes")
print(f"  Rasio kompresi: {mem_direct / mem_nn:.2f}x")
print()

# Membuat figure perbandingan
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Menampilkan gambar asli
axes[0].imshow(img_norm)
axes[0].set_title("Original (Direct Lookup)", fontsize=11)
axes[0].axis("off")

# Menampilkan hasil neural rendering
axes[1].imshow(np.clip(final_img, 0, 1))
axes[1].set_title(f"Neural Rendering (PSNR: {psnr:.1f} dB)", fontsize=11)
axes[1].axis("off")

# Menampilkan perbedaan (error map)
error_map = np.abs(final_img - img_norm)
# Mengalikan error agar terlihat jelas
axes[2].imshow(error_map * 5, cmap="hot")
axes[2].set_title("Error Map (5x amplified)", fontsize=11)
axes[2].axis("off")

# Mengatur layout
plt.suptitle("Neural Representation vs Direct Lookup", fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan perbandingan
comp_path = os.path.join(OUTPUT_DIR, "10_neural_vs_lookup.png")
plt.savefig(comp_path, dpi=150, bbox_inches="tight")
plt.close()

# Mencetak konfirmasi
print(f"  Disimpan: {comp_path}")
print()


# ========================================================
# 9. MENYIMPAN TRAINING PROGRESSION (LOSS CURVE)
# ========================================================
print("=" * 60)
print("9. MENYIMPAN TRAINING PROGRESSION (LOSS CURVE)")
print("=" * 60)

# Membuat figure untuk loss curve
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Memplot loss history
ax.plot(range(1, num_epochs + 1), loss_history, color="blue", linewidth=1.5)

# Mengatur label sumbu X
ax.set_xlabel("Epoch", fontsize=12)

# Mengatur label sumbu Y
ax.set_ylabel("MSE Loss", fontsize=12)

# Mengatur judul
ax.set_title("Training Loss: Neural Image Representation", fontsize=13)

# Mengatur skala Y ke logaritmik
ax.set_yscale("log")

# Menambahkan grid
ax.grid(True, alpha=0.3)

# Menambahkan anotasi snapshot epochs
for ep in snapshot_epochs:
    if ep <= num_epochs:
        # Menandai epoch snapshot pada grafik
        ax.axvline(x=ep, color="red", linestyle="--", alpha=0.4)
        ax.text(ep, ax.get_ylim()[1], f"Ep.{ep}", fontsize=8,
                ha="center", va="bottom", color="red")

# Mengatur layout
plt.tight_layout()

# Menyimpan loss curve
loss_path = os.path.join(OUTPUT_DIR, "10_training_loss.png")
plt.savefig(loss_path, dpi=150, bbox_inches="tight")
plt.close()

# Mencetak konfirmasi
print(f"  Disimpan: {loss_path}")
print()


# ========================================================
# 10. RINGKASAN KONSEP NEURAL RENDERING
# ========================================================
print("=" * 60)
print("10. RINGKASAN KONSEP NEURAL RENDERING")
print("=" * 60)

# Mencetak ringkasan konsep yang telah dipelajari
print()
print("  +-------------------------------------------+")
print("  | RINGKASAN NEURAL RENDERING                |")
print("  +-------------------------------------------+")
print("  | 1. Neural network bisa merepresentasikan  |")
print("  |    scene sebagai fungsi kontinu            |")
print("  | 2. Positional encoding membantu menangkap |")
print("  |    detail frekuensi tinggi                 |")
print("  | 3. Training = optimasi bobot agar network  |")
print("  |    menghasilkan output yang benar          |")
print("  | 4. Overfitting diinginkan: network harus  |")
print("  |    'menghafal' scene dengan presisi tinggi |")
print("  | 5. Keunggulan: representasi kompak dan     |")
print("  |    bisa query koordinat kontinu (novel view)|")
print("  +-------------------------------------------+")
print()

# Mencetak ringkasan akhir
print("=" * 60)
print("PERCOBAAN 10 SELESAI")
print("=" * 60)
print(f"  Output tersimpan di: {OUTPUT_DIR}")
print(f"  File yang dihasilkan:")
print(f"    - 10_training_progression.png")
print(f"    - 10_neural_vs_lookup.png")
print(f"    - 10_training_loss.png")
