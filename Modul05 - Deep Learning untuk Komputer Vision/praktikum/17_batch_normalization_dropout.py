"""
==========================================================================
PERCOBAAN 17: BATCH NORMALIZATION DAN DROPOUT
==========================================================================
Program ini mengimplementasikan dan memvisualisasikan dua teknik
regularisasi penting dalam deep learning: Batch Normalization dan
Dropout. Batch Normalization menormalkan distribusi fitur antar layer
sehingga training lebih stabil. Dropout secara acak menonaktifkan
sebagian neuron saat training untuk mencegah overfitting.

Fungsi utama yang dipelajari:
- Batch Normalization : Normalisasi, skala, dan shift fitur
- Dropout             : Random mask untuk menonaktifkan neuron
- Operasi NumPy: np.mean(), np.var(), np.sqrt(), np.random
- Matplotlib  : plt.hist(), plt.subplot(), plt.imshow()

Konsep yang dipelajari:
- Cara kerja batch normalization (normalize, scale, shift)
- Efek batch norm pada distribusi fitur (sebelum vs sesudah)
- Cara kerja dropout (random masking)
- Efek dropout rate pada output
- Regularisasi untuk mengatasi overfitting
- Training vs inference mode pada batch norm dan dropout
==========================================================================
"""

# Mengimpor NumPy untuk operasi komputasi dan array
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi grafik dan histogram
import matplotlib.pyplot as plt

# Mengimpor OpenCV untuk memuat dan memproses gambar
import cv2

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
print("PERCOBAAN 17: BATCH NORMALIZATION DAN DROPOUT")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep batch normalization
# ============================================================
print("\n--- 1. Konsep Batch Normalization ---")

# Menjelaskan konsep batch normalization secara detail
print("""
  Batch Normalization (BatchNorm) menormalkan output setiap layer
  sebelum diproses oleh layer berikutnya.

  Langkah-langkah BatchNorm:
  1. Hitung mean batch       : mu = (1/m) * sum(x_i)
  2. Hitung variance batch   : sigma^2 = (1/m) * sum((x_i - mu)^2)
  3. Normalisasi             : x_hat = (x - mu) / sqrt(sigma^2 + eps)
  4. Scale dan shift         : y = gamma * x_hat + beta

  Di mana gamma dan beta adalah parameter yang bisa dipelajari.

  Manfaat BatchNorm:
  - Training lebih cepat (bisa pakai learning rate lebih besar)
  - Mengurangi internal covariate shift
  - Efek regularisasi ringan
  - Membuat gradien lebih stabil
""")

# ============================================================
# 2. Implementasi batch normalization dari nol
# ============================================================
print("\n--- 2. Implementasi Batch Normalization ---")


def batch_normalize(x, gamma=1.0, beta=0.0, epsilon=1e-5):
    """
    Implementasi Batch Normalization dari nol.
    Input x: array (batch_size, features) atau 1D array.
    """
    # Menghitung mean dari batch
    mu = np.mean(x, axis=0)

    # Menghitung variance dari batch
    var = np.var(x, axis=0)

    # Menormalisasi: x_hat = (x - mu) / sqrt(var + eps)
    x_hat = (x - mu) / np.sqrt(var + epsilon)

    # Melakukan scale dan shift: y = gamma * x_hat + beta
    y = gamma * x_hat + beta

    # Mengembalikan hasil normalisasi, mean, dan variance
    return y, mu, var, x_hat


# Membuat data sintetis yang menyimulasikan output layer neural network
np.random.seed(42)

# Menyimulasikan 4 "layer" dengan distribusi fitur yang berbeda
# Setiap layer menghasilkan batch 200 sampel, 1 fitur
layer1_out = np.random.normal(loc=5.0, scale=3.0, size=(200, 1))
layer2_out = np.random.normal(loc=-2.0, scale=7.0, size=(200, 1))
layer3_out = np.random.normal(loc=10.0, scale=1.5, size=(200, 1))
layer4_out = np.random.normal(loc=0.5, scale=15.0, size=(200, 1))

# Menggabungkan menjadi satu array (200 sampel, 4 fitur)
features_sebelum = np.hstack([layer1_out, layer2_out, layer3_out, layer4_out])

# Menampilkan statistik sebelum normalisasi
print("  Statistik SEBELUM Batch Normalization:")
for i in range(4):
    # Menghitung mean dan std untuk setiap fitur
    m = np.mean(features_sebelum[:, i])
    s = np.std(features_sebelum[:, i])
    print(f"    Fitur {i+1}: mean = {m:>8.3f}, std = {s:>8.3f}")

# Menerapkan batch normalization pada setiap fitur
features_sesudah, mu_vals, var_vals, x_hat_vals = batch_normalize(features_sebelum)

# Menampilkan statistik sesudah normalisasi
print("\n  Statistik SESUDAH Batch Normalization:")
for i in range(4):
    # Menghitung mean dan std setelah normalisasi
    m = np.mean(features_sesudah[:, i])
    s = np.std(features_sesudah[:, i])
    print(f"    Fitur {i+1}: mean = {m:>8.3f}, std = {s:>8.3f}")

# ============================================================
# 3. Efek parameter gamma dan beta
# ============================================================
print("\n--- 3. Efek Parameter Gamma dan Beta ---")

# Menerapkan batch norm dengan gamma dan beta berbeda
gamma_values = [0.5, 1.0, 2.0]
beta_values = [-1.0, 0.0, 1.0]

# Menggunakan fitur pertama sebagai contoh
x_sample = features_sebelum[:, 0:1]

# Menampilkan efek gamma dan beta
print("  Efek gamma dan beta pada distribusi (Fitur 1):")
for g in gamma_values:
    for b in beta_values:
        # Menerapkan batch norm dengan parameter tertentu
        y_gb, _, _, _ = batch_normalize(x_sample, gamma=g, beta=b)

        # Menampilkan statistik
        print(f"    gamma={g}, beta={b:>5.1f} -> mean={np.mean(y_gb):.3f}, std={np.std(y_gb):.3f}")

# ============================================================
# 4. Visualisasi batch normalization (Gambar 1)
# ============================================================
print("\n--- 4. Visualisasi Batch Normalization ---")

# Membuat figure dengan 3x4 subplot
fig, axes = plt.subplots(3, 4, figsize=(16, 10))

# Memberikan judul utama
fig.suptitle("Efek Batch Normalization pada Distribusi Fitur",
             fontsize=16, fontweight='bold')

# Mendefinisikan warna untuk setiap fitur
warna_fitur = ['#e53935', '#1e88e5', '#43a047', '#fb8c00']

# --- Baris 1: Distribusi sebelum normalisasi ---
for i in range(4):
    ax = axes[0, i]

    # Menggambar histogram distribusi sebelum normalisasi
    ax.hist(features_sebelum[:, i], bins=30, color=warna_fitur[i],
            alpha=0.7, edgecolor='black', linewidth=0.5)

    # Menggambar garis vertikal untuk mean
    mean_val = np.mean(features_sebelum[:, i])
    ax.axvline(mean_val, color='black', linestyle='--', linewidth=2,
               label=f'μ={mean_val:.1f}')

    # Mengatur judul
    ax.set_title(f"Fitur {i+1} (Sebelum)", fontsize=10, fontweight='bold')

    # Mengatur label y hanya untuk kolom pertama
    if i == 0:
        ax.set_ylabel("Frekuensi")

    # Menambahkan legend
    ax.legend(fontsize=8)

# --- Baris 2: Distribusi sesudah normalisasi ---
for i in range(4):
    ax = axes[1, i]

    # Menggambar histogram distribusi sesudah normalisasi
    ax.hist(features_sesudah[:, i], bins=30, color=warna_fitur[i],
            alpha=0.7, edgecolor='black', linewidth=0.5)

    # Menggambar garis vertikal untuk mean
    mean_val = np.mean(features_sesudah[:, i])
    ax.axvline(mean_val, color='black', linestyle='--', linewidth=2,
               label=f'μ={mean_val:.2f}')

    # Mengatur judul
    ax.set_title(f"Fitur {i+1} (Sesudah BN)", fontsize=10, fontweight='bold')

    # Mengatur label y hanya untuk kolom pertama
    if i == 0:
        ax.set_ylabel("Frekuensi")

    # Menambahkan legend
    ax.legend(fontsize=8)

# --- Baris 3: Efek gamma pada distribusi ---
gammas_demo = [0.5, 1.0, 2.0, 3.0]

for idx, g in enumerate(gammas_demo):
    ax = axes[2, idx]

    # Menerapkan batch norm dengan gamma tertentu
    y_g, _, _, _ = batch_normalize(features_sebelum[:, 0:1], gamma=g, beta=0.0)

    # Menggambar histogram
    ax.hist(y_g, bins=30, color='#7e57c2', alpha=0.7,
            edgecolor='black', linewidth=0.5)

    # Menggambar garis vertikal untuk mean
    ax.axvline(np.mean(y_g), color='black', linestyle='--', linewidth=2)

    # Mengatur judul
    ax.set_title(f"γ={g}, β=0 (std≈{np.std(y_g):.2f})", fontsize=10,
                 fontweight='bold')

    # Mengatur label y hanya untuk kolom pertama
    if idx == 0:
        ax.set_ylabel("Frekuensi")

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_1 = os.path.join(OUTPUT_DIR, "17_batch_norm.png")

# Menyimpan figure
plt.savefig(output_path_1, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_1}")

# ============================================================
# 5. Penjelasan konsep dropout
# ============================================================
print("\n--- 5. Konsep Dropout ---")

# Menjelaskan konsep dropout
print("""
  Dropout secara acak menonaktifkan (set ke 0) sebagian neuron
  selama training:

  Langkah-langkah:
  1. Buat mask acak berisi 0 dan 1
  2. Kalikan output layer dengan mask
  3. Saat inference, TIDAK pakai dropout (gunakan semua neuron)
  4. (Opsional) Inverted dropout: bagi dengan (1-rate) saat training

  Dropout berperan sebagai regularisasi:
  - Mencegah neuron terlalu bergantung pada neuron lain
  - Seperti training "ensemble" dari sub-network berbeda
  - Mengurangi overfitting
""")

# ============================================================
# 6. Implementasi dropout dari nol
# ============================================================
print("\n--- 6. Implementasi Dropout ---")


def dropout_forward(x, rate=0.5, training=True, seed=None):
    """
    Implementasi dropout (inverted dropout).
    rate: fraksi neuron yang dinonaktifkan (0-1).
    """
    if not training or rate == 0.0:
        # Mode inference: mengembalikan input tanpa perubahan
        return x.copy(), np.ones_like(x)

    if seed is not None:
        # Mengatur seed untuk reproducibility
        np.random.seed(seed)

    # Membuat mask acak: 1 (aktif) atau 0 (dropout)
    mask = (np.random.rand(*x.shape) > rate).astype(np.float32)

    # Menerapkan inverted dropout: output = x * mask / (1 - rate)
    output = x * mask / (1.0 - rate)

    # Mengembalikan output dan mask
    return output, mask


# Membuat data fitur sintetis (1 batch, 10 fitur)
np.random.seed(123)
fitur_input = np.random.randn(1, 10).astype(np.float32)

# Menampilkan input asli
print(f"  Input fitur (10 neuron):")
print(f"  {fitur_input[0]}")

# Menerapkan dropout dengan rate 50%
fitur_dropout, mask_dropout = dropout_forward(fitur_input, rate=0.5, seed=42)

# Menampilkan hasil dropout
print(f"\n  Mask dropout (rate=0.5):")
print(f"  {mask_dropout[0]}")
print(f"\n  Output setelah dropout:")
print(f"  {fitur_dropout[0]}")

# Menghitung berapa neuron yang aktif
aktif = int(np.sum(mask_dropout))
total = mask_dropout.size
print(f"\n  Neuron aktif: {aktif}/{total} ({aktif/total*100:.0f}%)")

# ============================================================
# 7. Visualisasi efek dropout pada gambar dan fitur (Gambar 2)
# ============================================================
print("\n--- 7. Visualisasi Dropout ---")

# Mencoba memuat gambar dari IMAGE_DIR
img_path = os.path.join(IMAGE_DIR, "kucing.jpg")
img = None

if os.path.exists(img_path):
    # Memuat gambar
    img = cv2.imread(img_path)

    # Mengkonversi BGR ke RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Meresize gambar ke ukuran standar
    img = cv2.resize(img, (200, 200))

if img is None:
    # Membuat gambar sintetis jika gambar tidak ditemukan
    print("  Gambar tidak ditemukan, membuat gambar sintetis...")
    img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)

    # Menambahkan pola agar lebih menarik
    cv2.circle(img, (100, 100), 60, (255, 100, 50), -1)
    cv2.rectangle(img, (30, 30), (80, 80), (50, 200, 50), -1)

# Membuat figure untuk visualisasi dropout
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Memberikan judul utama
fig.suptitle("Visualisasi Efek Dropout", fontsize=16, fontweight='bold')

# Mendefinisikan berbagai dropout rate
dropout_rates = [0.0, 0.2, 0.5, 0.8]

# --- Baris 1: Efek dropout pada gambar ---
for idx, rate in enumerate(dropout_rates):
    ax = axes[0, idx]

    # Menerapkan dropout pada gambar (per piksel)
    img_float = img.astype(np.float32) / 255.0
    img_dropped, mask_img = dropout_forward(img_float, rate=rate, seed=42 + idx)

    # Memastikan nilai dalam range [0, 1]
    img_dropped = np.clip(img_dropped, 0, 1)

    # Menampilkan gambar hasil dropout
    ax.imshow(img_dropped)

    # Menghitung persentase piksel aktif
    pct_aktif = np.mean(mask_img) * 100

    # Mengatur judul
    ax.set_title(f"Rate={rate} ({pct_aktif:.0f}% aktif)", fontsize=10,
                 fontweight='bold')
    ax.axis('off')

# --- Baris 2: Efek dropout pada feature map ---
# Membuat feature map sintetis
np.random.seed(77)
feature_map = np.random.randn(50, 50).astype(np.float32)

# Menambahkan pola pada feature map
feature_map[10:40, 10:40] += 3.0
feature_map[20:30, 20:30] += 2.0

for idx, rate in enumerate(dropout_rates):
    ax = axes[1, idx]

    # Menerapkan dropout pada feature map
    fm_dropped, mask_fm = dropout_forward(feature_map, rate=rate, seed=99 + idx)

    # Menampilkan feature map hasil dropout
    ax.imshow(fm_dropped, cmap='viridis')

    # Menghitung statistik
    n_aktif = np.sum(mask_fm > 0)
    n_total = mask_fm.size

    # Mengatur judul
    ax.set_title(f"Rate={rate} ({n_aktif}/{n_total})", fontsize=10,
                 fontweight='bold')
    ax.axis('off')

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_2 = os.path.join(OUTPUT_DIR, "17_dropout_visualisasi.png")

# Menyimpan figure
plt.savefig(output_path_2, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_2}")

# ============================================================
# 8. Simulasi efek regularisasi: overfitting vs generalization
# ============================================================
print("\n--- 8. Simulasi Regularisasi ---")

# Membuat data sintetis dengan pola sinusoidal + noise
np.random.seed(42)

# Membuat data training
n_train = 20
x_train = np.sort(np.random.uniform(0, 2 * np.pi, n_train))
y_train_true = np.sin(x_train)
y_train = y_train_true + np.random.normal(0, 0.3, n_train)

# Membuat data test (lebih halus)
x_test = np.linspace(0, 2 * np.pi, 200)
y_test_true = np.sin(x_test)


def polynomial_fit(x, y, degree):
    """
    Fitting polinomial dengan derajat tertentu.
    """
    # Menghitung koefisien polinomial
    coeffs = np.polyfit(x, y, degree)

    # Mengembalikan koefisien
    return coeffs


def predict_polynomial(x, coeffs):
    """
    Prediksi menggunakan koefisien polinomial.
    """
    # Mengevaluasi polinomial
    return np.polyval(coeffs, x)


def simulate_dropout_regularization(x, y, degree, dropout_rate=0.5, n_ensemble=20):
    """
    Simulasi efek regularisasi dropout menggunakan pendekatan ensemble:
    - Latih beberapa model dengan subset acak dari fitur/data
    - Rata-ratakan prediksi (mirip efek dropout)
    """
    # Menyiapkan prediksi ensemble
    predictions = []

    for i in range(n_ensemble):
        # Membuat mask dropout untuk data training
        np.random.seed(i * 10)
        mask = np.random.rand(len(x)) > dropout_rate

        # Memastikan minimal ada degree+1 data point
        if np.sum(mask) <= degree:
            mask[:degree + 1] = True

        # Melakukan fitting pada data yang di-mask
        x_masked = x[mask]
        y_masked = y[mask]

        # Fitting polinomial pada subset data
        try:
            coeffs = polynomial_fit(x_masked, y_masked, min(degree, len(x_masked) - 1))
            pred = predict_polynomial(x_test, coeffs)
            predictions.append(pred)
        except Exception:
            pass

    # Merata-ratakan prediksi dari semua sub-model
    if len(predictions) > 0:
        avg_pred = np.mean(predictions, axis=0)
    else:
        avg_pred = np.zeros_like(x_test)

    # Mengembalikan prediksi rata-rata
    return avg_pred


# Menampilkan penjelasan simulasi
print("  Simulasi: fitting polinomial pada data sinusoidal + noise")
print("  - Derajat rendah  (2)  : underfitting")
print("  - Derajat tinggi  (15) : overfitting")
print("  - Dengan dropout ensemble: regularisasi")

# Melakukan fitting dengan berbagai derajat
coeffs_low = polynomial_fit(x_train, y_train, 2)
coeffs_good = polynomial_fit(x_train, y_train, 5)
coeffs_high = polynomial_fit(x_train, y_train, 15)

# Menghitung prediksi
pred_low = predict_polynomial(x_test, coeffs_low)
pred_good = predict_polynomial(x_test, coeffs_good)
pred_high = predict_polynomial(x_test, coeffs_high)

# Menghitung prediksi dengan regularisasi dropout
pred_dropout = simulate_dropout_regularization(x_train, y_train, 15,
                                                dropout_rate=0.3, n_ensemble=30)

# Menghitung error untuk setiap model
mse_low = np.mean((pred_low - y_test_true) ** 2)
mse_good = np.mean((pred_good - y_test_true) ** 2)
mse_high = np.mean((np.clip(pred_high, -5, 5) - y_test_true) ** 2)
mse_dropout = np.mean((pred_dropout - y_test_true) ** 2)

# Menampilkan hasil
print(f"\n  MSE pada data test:")
print(f"    Underfitting (deg=2)          : {mse_low:.4f}")
print(f"    Good fit (deg=5)              : {mse_good:.4f}")
print(f"    Overfitting (deg=15)          : {mse_high:.4f}")
print(f"    Dengan dropout ensemble       : {mse_dropout:.4f}")

# ============================================================
# 9. Visualisasi regularisasi (Gambar 3)
# ============================================================
print("\n--- 9. Visualisasi Regularisasi ---")

# Membuat figure untuk visualisasi regularisasi
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Memberikan judul utama
fig.suptitle("Regularisasi: Batch Norm dan Dropout vs Overfitting",
             fontsize=16, fontweight='bold')

# --- Subplot 1: Underfitting ---
ax1 = axes[0, 0]

# Menggambar data training
ax1.scatter(x_train, y_train, c='blue', s=40, zorder=5, label='Data train')

# Menggambar fungsi asli
ax1.plot(x_test, y_test_true, 'g--', linewidth=2, label='Fungsi asli')

# Menggambar prediksi underfitting
ax1.plot(x_test, pred_low, 'r-', linewidth=2, label=f'Deg=2 (MSE={mse_low:.3f})')

# Mengatur judul dan label
ax1.set_title("Underfitting (derajat 2)", fontsize=11, fontweight='bold')
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend(fontsize=8)
ax1.set_ylim(-2, 2)
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Good fit ---
ax2 = axes[0, 1]

# Menggambar data training
ax2.scatter(x_train, y_train, c='blue', s=40, zorder=5, label='Data train')

# Menggambar fungsi asli
ax2.plot(x_test, y_test_true, 'g--', linewidth=2, label='Fungsi asli')

# Menggambar prediksi good fit
ax2.plot(x_test, pred_good, 'r-', linewidth=2, label=f'Deg=5 (MSE={mse_good:.3f})')

# Mengatur judul dan label
ax2.set_title("Good Fit (derajat 5)", fontsize=11, fontweight='bold')
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend(fontsize=8)
ax2.set_ylim(-2, 2)
ax2.grid(True, alpha=0.3)

# --- Subplot 3: Overfitting ---
ax3 = axes[0, 2]

# Menggambar data training
ax3.scatter(x_train, y_train, c='blue', s=40, zorder=5, label='Data train')

# Menggambar fungsi asli
ax3.plot(x_test, y_test_true, 'g--', linewidth=2, label='Fungsi asli')

# Menggambar prediksi overfitting (di-clip agar terlihat)
pred_high_clipped = np.clip(pred_high, -3, 3)
ax3.plot(x_test, pred_high_clipped, 'r-', linewidth=2,
         label=f'Deg=15 (MSE={mse_high:.3f})')

# Mengatur judul dan label
ax3.set_title("Overfitting (derajat 15)", fontsize=11, fontweight='bold')
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.legend(fontsize=8)
ax3.set_ylim(-3, 3)
ax3.grid(True, alpha=0.3)

# --- Subplot 4: Dengan dropout regularisasi ---
ax4 = axes[1, 0]

# Menggambar data training
ax4.scatter(x_train, y_train, c='blue', s=40, zorder=5, label='Data train')

# Menggambar fungsi asli
ax4.plot(x_test, y_test_true, 'g--', linewidth=2, label='Fungsi asli')

# Menggambar prediksi dengan dropout
ax4.plot(x_test, pred_dropout, 'r-', linewidth=2,
         label=f'Dropout (MSE={mse_dropout:.3f})')

# Mengatur judul dan label
ax4.set_title("Dengan Dropout Regularisasi", fontsize=11, fontweight='bold')
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.legend(fontsize=8)
ax4.set_ylim(-2, 2)
ax4.grid(True, alpha=0.3)

# --- Subplot 5: Perbandingan MSE ---
ax5 = axes[1, 1]

# Mendefinisikan nama model dan MSE
model_names = ['Underfitting\n(deg=2)', 'Good Fit\n(deg=5)',
               'Overfitting\n(deg=15)', 'Dropout\nEnsemble']
mse_vals = [mse_low, mse_good, mse_high, mse_dropout]

# Mendefinisikan warna bar
bar_colors = ['#e53935', '#43a047', '#e53935', '#1e88e5']

# Menggambar bar chart MSE
bars = ax5.bar(model_names, mse_vals, color=bar_colors)

# Menambahkan label nilai di atas bar
for bar, val in zip(bars, mse_vals):
    ax5.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# Mengatur judul dan label
ax5.set_title("Perbandingan MSE pada Test Set", fontsize=11, fontweight='bold')
ax5.set_ylabel("MSE")
ax5.grid(True, alpha=0.3, axis='y')

# --- Subplot 6: Kurva training loss vs epoch (simulasi) ---
ax6 = axes[1, 2]

# Menyimulasikan kurva training: train loss dan val loss
epochs_sim = np.arange(1, 101)

# Kurva tanpa regularisasi: train turun terus, val naik
train_loss_no_reg = 1.0 * np.exp(-0.05 * epochs_sim) + 0.02
val_loss_no_reg = 0.5 * np.exp(-0.03 * epochs_sim) + 0.1 + 0.004 * epochs_sim

# Kurva dengan regularisasi: keduanya turun dan stabil
train_loss_reg = 1.0 * np.exp(-0.04 * epochs_sim) + 0.08
val_loss_reg = 0.5 * np.exp(-0.03 * epochs_sim) + 0.12

# Menggambar kurva tanpa regularisasi
ax6.plot(epochs_sim, train_loss_no_reg, 'b-', linewidth=2, label='Train (tanpa reg)')
ax6.plot(epochs_sim, val_loss_no_reg, 'b--', linewidth=2, label='Val (tanpa reg)')

# Menggambar kurva dengan regularisasi
ax6.plot(epochs_sim, train_loss_reg, 'r-', linewidth=2, label='Train (dengan reg)')
ax6.plot(epochs_sim, val_loss_reg, 'r--', linewidth=2, label='Val (dengan reg)')

# Menandai area overfitting
ax6.axvspan(60, 100, alpha=0.1, color='red', label='Overfitting zone')

# Mengatur judul dan label
ax6.set_title("Train vs Val Loss (Simulasi)", fontsize=11, fontweight='bold')
ax6.set_xlabel("Epoch")
ax6.set_ylabel("Loss")
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.3)

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_3 = os.path.join(OUTPUT_DIR, "17_regularisasi.png")

# Menyimpan figure
plt.savefig(output_path_3, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_3}")

# ============================================================
# 10. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 17")
print("=" * 60)
print(f"""
Konsep yang telah dipelajari:
1. Batch Normalization menormalkan distribusi fitur antar layer
2. BatchNorm: mean -> 0, std -> 1 (sebelum scale/shift)
3. Parameter gamma (scale) dan beta (shift) bisa dipelajari
4. Dropout menonaktifkan sebagian neuron secara acak
5. Inverted dropout: bagi output dengan (1-rate) saat training
6. Dropout rate tinggi = lebih banyak neuron dinonaktifkan
7. Regularisasi mencegah overfitting pada data training
8. Dropout berfungsi seperti ensemble dari banyak sub-network

Hasil Simulasi Regularisasi:
- Underfitting (deg=2)     : MSE = {mse_low:.4f}
- Good fit (deg=5)         : MSE = {mse_good:.4f}
- Overfitting (deg=15)     : MSE = {mse_high:.4f}
- Dropout ensemble         : MSE = {mse_dropout:.4f}

Output disimpan di folder: output/
- 17_batch_norm.png          : Efek BatchNorm pada distribusi fitur
- 17_dropout_visualisasi.png : Visualisasi efek dropout pada gambar/fitur
- 17_regularisasi.png        : Perbandingan regularisasi vs overfitting
""")
