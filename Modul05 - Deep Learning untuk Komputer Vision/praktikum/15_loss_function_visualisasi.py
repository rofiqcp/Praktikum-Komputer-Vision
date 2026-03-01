"""
==========================================================================
PERCOBAAN 15: LOSS FUNCTION - VISUALISASI
==========================================================================
Program ini mengimplementasikan dan memvisualisasikan berbagai fungsi
loss (loss function) yang umum digunakan dalam deep learning. Fungsi
loss mengukur seberapa jauh prediksi model dari ground truth (target).
Pemahaman loss function sangat penting karena menentukan bagaimana
model belajar dan dioptimasi selama proses training.

Fungsi utama yang dipelajari:
- MSE Loss (Mean Squared Error)         : Rata-rata kuadrat selisih
- Cross-Entropy Loss                     : Loss untuk klasifikasi multi-kelas
- Binary Cross-Entropy Loss              : Loss untuk klasifikasi biner
- Dice Loss                              : Loss untuk segmentasi
- Operasi NumPy: np.mean(), np.log(), np.sum(), np.clip()
- Matplotlib: plt.plot(), plt.contourf(), plt.subplot()

Konsep yang dipelajari:
- Cara kerja dan formula setiap loss function
- Bagaimana loss berubah saat prediksi mendekati ground truth
- Loss landscape (kontur 2D) untuk 2 parameter
- Perbandingan loss function secara visual
- Pengaruh pilihan loss function terhadap gradien
- Dice Loss untuk tugas segmentasi
==========================================================================
"""

# Mengimpor NumPy untuk operasi komputasi numerik dan array
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi grafik dan plot
import matplotlib.pyplot as plt

# Mengimpor matplotlib colormap untuk pewarnaan plot
from matplotlib import cm

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
print("PERCOBAAN 15: LOSS FUNCTION - VISUALISASI")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep loss function
# ============================================================
print("\n--- 1. Konsep Loss Function ---")

# Menjelaskan konsep loss function secara umum
print("""
  Loss Function (Fungsi Kerugian) mengukur seberapa buruk
  prediksi model dibandingkan dengan nilai target (ground truth).

  Tujuan training: MEMINIMALKAN nilai loss

  Jenis-jenis loss function utama:
  1. MSE (Mean Squared Error)    - Regresi
  2. Cross-Entropy               - Klasifikasi multi-kelas
  3. Binary Cross-Entropy (BCE)  - Klasifikasi biner
  4. Dice Loss                   - Segmentasi

  | Loss Function     | Formula                              | Tugas          |
  |-------------------|--------------------------------------|----------------|
  | MSE               | mean((y - y_hat)^2)                  | Regresi        |
  | Cross-Entropy     | -sum(y * log(y_hat))                 | Multi-kelas    |
  | Binary CE         | -[y*log(p) + (1-y)*log(1-p)]         | Biner          |
  | Dice Loss         | 1 - 2*|A∩B| / (|A|+|B|)             | Segmentasi     |
""")

# ============================================================
# 2. Implementasi MSE Loss
# ============================================================
print("\n--- 2. Implementasi MSE Loss ---")


def mse_loss(y_true, y_pred):
    """
    Mean Squared Error Loss.
    Formula: MSE = (1/n) * sum((y_true - y_pred)^2)
    """
    # Menghitung selisih antara target dan prediksi
    selisih = y_true - y_pred

    # Menghitung kuadrat dari selisih
    kuadrat_selisih = selisih ** 2

    # Menghitung rata-rata kuadrat selisih
    rata_rata = np.mean(kuadrat_selisih)

    # Mengembalikan nilai MSE
    return rata_rata


def mse_gradient(y_true, y_pred):
    """
    Gradien MSE terhadap y_pred.
    Formula: d(MSE)/d(y_pred) = -2/n * (y_true - y_pred)
    """
    # Menghitung jumlah elemen
    n = len(y_true)

    # Menghitung gradien MSE
    grad = -2.0 / n * (y_true - y_pred)

    # Mengembalikan rata-rata absolut gradien
    return np.mean(np.abs(grad))


# Mendefinisikan target (ground truth) untuk demonstrasi
y_target = np.array([1.0, 0.0, 1.0, 0.0, 1.0])

# Membuat range prediksi dari 0 sampai 1
prediksi_range = np.linspace(0.0, 2.0, 200)

# Menghitung MSE untuk setiap nilai prediksi konstan
mse_values = []

# Melakukan iterasi untuk setiap nilai prediksi
for p in prediksi_range:
    # Membuat array prediksi dengan nilai konstan
    y_pred = np.full_like(y_target, p)

    # Menghitung MSE loss
    loss = mse_loss(y_target, y_pred)

    # Menyimpan nilai loss
    mse_values.append(loss)

# Mengkonversi list ke array numpy
mse_values = np.array(mse_values)

# Menampilkan contoh perhitungan MSE
print(f"  Target          : {y_target}")
print(f"  Prediksi = 0.5  : MSE = {mse_loss(y_target, np.full_like(y_target, 0.5)):.4f}")
print(f"  Prediksi = 0.0  : MSE = {mse_loss(y_target, np.full_like(y_target, 0.0)):.4f}")
print(f"  Prediksi = 1.0  : MSE = {mse_loss(y_target, np.full_like(y_target, 1.0)):.4f}")

# ============================================================
# 3. Implementasi Cross-Entropy Loss
# ============================================================
print("\n--- 3. Implementasi Cross-Entropy Loss ---")


def cross_entropy_loss(y_true_onehot, y_pred_probs):
    """
    Cross-Entropy Loss untuk klasifikasi multi-kelas.
    Formula: CE = -sum(y_true * log(y_pred))
    """
    # Melakukan clipping prediksi untuk menghindari log(0)
    y_pred_clipped = np.clip(y_pred_probs, 1e-7, 1.0 - 1e-7)

    # Menghitung logaritma dari prediksi
    log_pred = np.log(y_pred_clipped)

    # Menghitung cross-entropy: -sum(y * log(y_hat))
    ce = -np.sum(y_true_onehot * log_pred)

    # Mengembalikan nilai cross-entropy
    return ce


# Mendefinisikan target one-hot untuk 3 kelas (kelas ke-1 benar)
y_true_onehot = np.array([0, 1, 0])

# Membuat range probabilitas untuk kelas benar
prob_range = np.linspace(0.01, 0.99, 200)

# Menghitung cross-entropy untuk setiap probabilitas
ce_values = []

# Melakukan iterasi untuk setiap probabilitas kelas benar
for p in prob_range:
    # Membuat distribusi probabilitas (sisanya merata)
    sisa = (1.0 - p) / 2.0
    y_pred = np.array([sisa, p, sisa])

    # Menghitung cross-entropy loss
    loss = cross_entropy_loss(y_true_onehot, y_pred)

    # Menyimpan nilai loss
    ce_values.append(loss)

# Mengkonversi list ke array numpy
ce_values = np.array(ce_values)

# Menampilkan contoh perhitungan cross-entropy
print(f"  Target (one-hot)   : {y_true_onehot}")
print(f"  Prediksi [0.1, 0.8, 0.1] : CE = {cross_entropy_loss(y_true_onehot, [0.1, 0.8, 0.1]):.4f}")
print(f"  Prediksi [0.33,0.34,0.33]: CE = {cross_entropy_loss(y_true_onehot, [0.33, 0.34, 0.33]):.4f}")
print(f"  Prediksi [0.45,0.1, 0.45]: CE = {cross_entropy_loss(y_true_onehot, [0.45, 0.1, 0.45]):.4f}")

# ============================================================
# 4. Implementasi Binary Cross-Entropy Loss
# ============================================================
print("\n--- 4. Implementasi Binary Cross-Entropy Loss ---")


def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy Loss untuk klasifikasi biner.
    Formula: BCE = -[y*log(p) + (1-y)*log(1-p)]
    """
    # Melakukan clipping prediksi untuk menghindari log(0)
    p = np.clip(y_pred, 1e-7, 1.0 - 1e-7)

    # Menghitung komponen positif: y * log(p)
    term_positif = y_true * np.log(p)

    # Menghitung komponen negatif: (1-y) * log(1-p)
    term_negatif = (1.0 - y_true) * np.log(1.0 - p)

    # Menghitung rata-rata BCE
    bce = -np.mean(term_positif + term_negatif)

    # Mengembalikan nilai BCE
    return bce


def bce_gradient(y_true, y_pred):
    """
    Gradien Binary Cross-Entropy terhadap y_pred.
    Formula: d(BCE)/d(p) = -(y/p) + (1-y)/(1-p)
    """
    # Melakukan clipping prediksi
    p = np.clip(y_pred, 1e-7, 1.0 - 1e-7)

    # Menghitung gradien BCE
    grad = -(y_true / p) + (1.0 - y_true) / (1.0 - p)

    # Mengembalikan rata-rata absolut gradien
    return np.mean(np.abs(grad))


# Mendefinisikan target biner
y_true_biner = 1.0

# Membuat range prediksi probabilitas
prob_biner = np.linspace(0.01, 0.99, 200)

# Menghitung BCE untuk target = 1
bce_target1 = []
for p in prob_biner:
    # Menghitung BCE dengan target = 1
    loss = binary_cross_entropy(np.array([1.0]), np.array([p]))
    bce_target1.append(loss)

# Menghitung BCE untuk target = 0
bce_target0 = []
for p in prob_biner:
    # Menghitung BCE dengan target = 0
    loss = binary_cross_entropy(np.array([0.0]), np.array([p]))
    bce_target0.append(loss)

# Mengkonversi list ke array
bce_target1 = np.array(bce_target1)
bce_target0 = np.array(bce_target0)

# Menampilkan contoh perhitungan BCE
print(f"  BCE(y=1, p=0.9) = {binary_cross_entropy(np.array([1.0]), np.array([0.9])):.4f}")
print(f"  BCE(y=1, p=0.5) = {binary_cross_entropy(np.array([1.0]), np.array([0.5])):.4f}")
print(f"  BCE(y=1, p=0.1) = {binary_cross_entropy(np.array([1.0]), np.array([0.1])):.4f}")
print(f"  BCE(y=0, p=0.1) = {binary_cross_entropy(np.array([0.0]), np.array([0.1])):.4f}")
print(f"  BCE(y=0, p=0.9) = {binary_cross_entropy(np.array([0.0]), np.array([0.9])):.4f}")

# ============================================================
# 5. Implementasi Dice Loss (untuk segmentasi)
# ============================================================
print("\n--- 5. Implementasi Dice Loss ---")


def dice_coefficient(y_true, y_pred):
    """
    Dice Coefficient: mengukur overlap antara prediksi dan target.
    Formula: Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    # Menghitung intersection (irisan)
    intersection = np.sum(y_true * y_pred)

    # Menghitung jumlah elemen di masing-masing set
    sum_a = np.sum(y_true)
    sum_b = np.sum(y_pred)

    # Menghitung Dice coefficient dengan smoothing
    dice = (2.0 * intersection + 1e-7) / (sum_a + sum_b + 1e-7)

    # Mengembalikan nilai Dice coefficient
    return dice


def dice_loss(y_true, y_pred):
    """
    Dice Loss = 1 - Dice Coefficient.
    Semakin kecil loss, semakin baik overlap.
    """
    # Menghitung dice coefficient
    dc = dice_coefficient(y_true, y_pred)

    # Mengembalikan 1 - dice sebagai loss
    return 1.0 - dc


# Membuat mask segmentasi sederhana (5x5) sebagai ground truth
mask_true = np.zeros((5, 5), dtype=np.float32)

# Mengisi area target dengan nilai 1 (objek)
mask_true[1:4, 1:4] = 1.0

# Membuat beberapa prediksi mask dengan tingkat overlap berbeda
mask_pred_sempurna = mask_true.copy()

# Membuat prediksi dengan overlap sebagian
mask_pred_sebagian = np.zeros((5, 5), dtype=np.float32)
mask_pred_sebagian[2:5, 2:5] = 1.0

# Membuat prediksi yang salah total
mask_pred_salah = np.zeros((5, 5), dtype=np.float32)
mask_pred_salah[0, 0] = 1.0

# Menghitung Dice Loss untuk setiap kasus
dice_sempurna = dice_loss(mask_true, mask_pred_sempurna)
dice_sebagian = dice_loss(mask_true, mask_pred_sebagian)
dice_salah = dice_loss(mask_true, mask_pred_salah)

# Menampilkan hasil Dice Loss
print(f"  Dice Loss (overlap sempurna)   : {dice_sempurna:.4f}")
print(f"  Dice Loss (overlap sebagian)   : {dice_sebagian:.4f}")
print(f"  Dice Loss (prediksi salah)     : {dice_salah:.4f}")

# ============================================================
# 6. Visualisasi MSE dan Cross-Entropy (Gambar 1)
# ============================================================
print("\n--- 6. Visualisasi MSE dan Cross-Entropy ---")

# Membuat figure dengan 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Memberikan judul utama pada figure
fig.suptitle("Loss Functions untuk Deep Learning", fontsize=16, fontweight='bold')

# --- Subplot 1: MSE Loss ---
ax1 = axes[0, 0]

# Menggambar kurva MSE loss
ax1.plot(prediksi_range, mse_values, 'b-', linewidth=2, label='MSE Loss')

# Menandai nilai minimum MSE
idx_min = np.argmin(mse_values)
ax1.plot(prediksi_range[idx_min], mse_values[idx_min], 'ro', markersize=10,
         label=f'Min di p={prediksi_range[idx_min]:.2f}')

# Mengatur judul dan label subplot
ax1.set_title("MSE Loss vs Prediksi", fontsize=12, fontweight='bold')
ax1.set_xlabel("Nilai Prediksi (konstan)")
ax1.set_ylabel("MSE Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Cross-Entropy Loss ---
ax2 = axes[0, 1]

# Menggambar kurva cross-entropy loss
ax2.plot(prob_range, ce_values, 'r-', linewidth=2, label='CE Loss')

# Mengatur judul dan label subplot
ax2.set_title("Cross-Entropy Loss vs P(kelas benar)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Probabilitas Kelas Benar")
ax2.set_ylabel("Cross-Entropy Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Subplot 3: Binary Cross-Entropy ---
ax3 = axes[1, 0]

# Menggambar kurva BCE untuk target = 1
ax3.plot(prob_biner, bce_target1, 'g-', linewidth=2, label='BCE (y=1)')

# Menggambar kurva BCE untuk target = 0
ax3.plot(prob_biner, bce_target0, 'm-', linewidth=2, label='BCE (y=0)')

# Mengatur judul dan label subplot
ax3.set_title("Binary Cross-Entropy Loss", fontsize=12, fontweight='bold')
ax3.set_xlabel("Prediksi p")
ax3.set_ylabel("BCE Loss")
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Subplot 4: Dice Loss ---
ax4 = axes[1, 1]

# Membuat range overlap untuk demonstrasi Dice Loss
overlap_range = np.linspace(0.0, 1.0, 200)

# Menghitung Dice Loss untuk setiap tingkat overlap
dice_values = []
for ov in overlap_range:
    # Membuat prediksi dengan overlap tertentu
    pred = np.zeros_like(mask_true)
    # Menghitung jumlah piksel overlap
    n_overlap = int(ov * np.sum(mask_true))
    # Mengisi piksel overlap
    flat_true = mask_true.flatten()
    flat_pred = np.zeros_like(flat_true)
    true_indices = np.where(flat_true == 1.0)[0]
    if n_overlap > 0 and len(true_indices) > 0:
        flat_pred[true_indices[:min(n_overlap, len(true_indices))]] = 1.0
    pred = flat_pred.reshape(mask_true.shape)
    # Menghitung Dice Loss
    dl = dice_loss(mask_true, pred)
    dice_values.append(dl)

# Mengkonversi ke array numpy
dice_values = np.array(dice_values)

# Menggambar kurva Dice Loss
ax4.plot(overlap_range, dice_values, 'c-', linewidth=2, label='Dice Loss')

# Mengatur judul dan label subplot
ax4.set_title("Dice Loss vs Overlap", fontsize=12, fontweight='bold')
ax4.set_xlabel("Tingkat Overlap")
ax4.set_ylabel("Dice Loss")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Mendefinisikan path output file gambar
output_path_1 = os.path.join(OUTPUT_DIR, "15_loss_mse_ce.png")

# Menyimpan figure ke file
plt.savefig(output_path_1, dpi=150, bbox_inches='tight')

# Menutup figure untuk membersihkan memori
plt.close()

# Menampilkan pesan konfirmasi penyimpanan
print(f"  [SAVED] {output_path_1}")

# ============================================================
# 7. Visualisasi Loss Landscape (Gambar 2)
# ============================================================
print("\n--- 7. Visualisasi Loss Landscape ---")

# Mendefinisikan fungsi loss untuk 2 parameter (w1, w2)
# Loss(w1, w2) = (w1-1)^2 + (w2-2)^2 + 0.5*sin(3*w1)*sin(3*w2)


def loss_landscape_func(w1, w2):
    """
    Fungsi loss 2 parameter untuk visualisasi landscape.
    Minimum global di sekitar (1, 2).
    """
    # Menghitung komponen kuadratik
    quadratic = (w1 - 1.0) ** 2 + (w2 - 2.0) ** 2

    # Menambahkan komponen sinusoidal untuk membuat landscape menarik
    sinusoidal = 0.5 * np.sin(3.0 * w1) * np.sin(3.0 * w2)

    # Mengembalikan total loss
    return quadratic + sinusoidal


# Membuat grid 2D untuk parameter w1 dan w2
w1_range = np.linspace(-2, 4, 300)
w2_range = np.linspace(-1, 5, 300)

# Membuat meshgrid dari range parameter
W1, W2 = np.meshgrid(w1_range, w2_range)

# Menghitung loss untuk setiap titik di grid
Z = loss_landscape_func(W1, W2)

# Membuat figure untuk loss landscape
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Memberikan judul utama
fig.suptitle("Loss Landscape (2 Parameter)", fontsize=16, fontweight='bold')

# --- Subplot 1: Contour Plot ---
ax1 = axes[0]

# Menggambar filled contour plot
contour = ax1.contourf(W1, W2, Z, levels=30, cmap='viridis')

# Menambahkan colorbar
plt.colorbar(contour, ax=ax1, label='Loss')

# Menandai titik minimum
ax1.plot(1.0, 2.0, 'r*', markersize=15, label='Minimum (1,2)')

# Mengatur judul dan label
ax1.set_title("Contour Plot Loss", fontsize=12, fontweight='bold')
ax1.set_xlabel("Parameter w1")
ax1.set_ylabel("Parameter w2")
ax1.legend()

# --- Subplot 2: Surface plot (top-down view dengan level lines) ---
ax2 = axes[1]

# Menggambar contour lines
contour_lines = ax2.contour(W1, W2, Z, levels=20, cmap='RdYlBu_r')

# Menambahkan label pada level lines
ax2.clabel(contour_lines, inline=True, fontsize=7)

# Menandai titik minimum
ax2.plot(1.0, 2.0, 'r*', markersize=15, label='Minimum')

# Mensimulasikan jalur gradient descent
gd_path_w1 = [3.5]
gd_path_w2 = [4.0]
lr_gd = 0.05

# Melakukan beberapa langkah gradient descent
for step in range(50):
    # Mengambil posisi saat ini
    cw1 = gd_path_w1[-1]
    cw2 = gd_path_w2[-1]

    # Menghitung gradien secara analitik
    grad_w1 = 2.0 * (cw1 - 1.0) + 1.5 * np.cos(3.0 * cw1) * np.sin(3.0 * cw2)
    grad_w2 = 2.0 * (cw2 - 2.0) + 1.5 * np.sin(3.0 * cw1) * np.cos(3.0 * cw2)

    # Memperbarui parameter
    new_w1 = cw1 - lr_gd * grad_w1
    new_w2 = cw2 - lr_gd * grad_w2

    # Menyimpan posisi baru
    gd_path_w1.append(new_w1)
    gd_path_w2.append(new_w2)

# Menggambar jalur gradient descent
ax2.plot(gd_path_w1, gd_path_w2, 'r.-', markersize=3, linewidth=1.5,
         label='Gradient Descent Path')

# Menandai titik awal
ax2.plot(gd_path_w1[0], gd_path_w2[0], 'gs', markersize=10, label='Start')

# Mengatur judul dan label
ax2.set_title("Gradient Descent Path", fontsize=12, fontweight='bold')
ax2.set_xlabel("Parameter w1")
ax2.set_ylabel("Parameter w2")
ax2.legend(fontsize=8)

# --- Subplot 3: Loss sepanjang jalur GD ---
ax3 = axes[2]

# Menghitung loss di setiap titik jalur GD
loss_path = []
for i in range(len(gd_path_w1)):
    # Menghitung loss di posisi saat ini
    l = loss_landscape_func(gd_path_w1[i], gd_path_w2[i])
    loss_path.append(l)

# Menggambar loss vs iterasi
ax3.plot(range(len(loss_path)), loss_path, 'b-', linewidth=2)

# Mengatur judul dan label
ax3.set_title("Loss vs Iterasi (GD)", fontsize=12, fontweight='bold')
ax3.set_xlabel("Iterasi")
ax3.set_ylabel("Loss")
ax3.grid(True, alpha=0.3)

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_2 = os.path.join(OUTPUT_DIR, "15_loss_landscape.png")

# Menyimpan figure
plt.savefig(output_path_2, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_2}")

# ============================================================
# 8. Perbandingan Loss Function dan Gradien (Gambar 3)
# ============================================================
print("\n--- 8. Perbandingan Loss Function dan Efek Gradien ---")

# Membuat figure untuk perbandingan
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Memberikan judul utama
fig.suptitle("Perbandingan Loss Function dan Gradien", fontsize=16, fontweight='bold')

# --- Baris 1: Perbandingan loss function ---

# Membuat range prediksi untuk perbandingan
p_range = np.linspace(0.01, 0.99, 200)

# Menghitung MSE loss untuk kasus biner (target=1)
mse_biner = [(p - 1.0) ** 2 for p in p_range]

# Menghitung BCE loss untuk target=1
bce_biner = [-np.log(p) for p in p_range]

# Menghitung Hinge loss untuk target=1 (konsep SVM)
hinge_biner = [max(0, 1.0 - p) for p in p_range]

# --- Subplot 1: Semua loss dalam satu plot ---
ax1 = axes[0, 0]

# Menggambar MSE
ax1.plot(p_range, mse_biner, 'b-', linewidth=2, label='MSE')

# Menggambar BCE
ax1.plot(p_range, bce_biner, 'r-', linewidth=2, label='BCE')

# Menggambar Hinge
ax1.plot(p_range, hinge_biner, 'g-', linewidth=2, label='Hinge')

# Mengatur judul dan label
ax1.set_title("Perbandingan Loss (target=1)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Prediksi p")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 5)

# --- Subplot 2: Gradien loss ---
ax2 = axes[0, 1]

# Menghitung gradien MSE: d/dp[(p-1)^2] = 2(p-1)
grad_mse = [abs(2.0 * (p - 1.0)) for p in p_range]

# Menghitung gradien BCE: d/dp[-log(p)] = -1/p
grad_bce = [1.0 / p for p in p_range]

# Menghitung gradien Hinge: -1 jika p < 1, 0 jika p >= 1
grad_hinge = [1.0 if p < 1.0 else 0.0 for p in p_range]

# Menggambar gradien MSE
ax2.plot(p_range, grad_mse, 'b-', linewidth=2, label='|grad MSE|')

# Menggambar gradien BCE
ax2.plot(p_range, grad_bce, 'r-', linewidth=2, label='|grad BCE|')

# Menggambar gradien Hinge
ax2.plot(p_range, grad_hinge, 'g-', linewidth=2, label='|grad Hinge|')

# Mengatur judul dan label
ax2.set_title("Magnitude Gradien (target=1)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Prediksi p")
ax2.set_ylabel("|Gradien|")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 10)

# --- Subplot 3: Perbandingan saat prediksi mendekati target ---
ax3 = axes[0, 2]

# Membuat skenario: prediksi bergerak dari 0.1 ke 0.9
iterasi = np.arange(1, 11)
prediksi_iterasi = np.linspace(0.1, 0.95, 10)

# Menghitung loss di setiap iterasi
mse_iter = [(p - 1.0) ** 2 for p in prediksi_iterasi]
bce_iter = [-np.log(p) for p in prediksi_iterasi]

# Menggambar konvergensi MSE
ax3.plot(iterasi, mse_iter, 'bo-', linewidth=2, label='MSE')

# Menggambar konvergensi BCE
ax3.plot(iterasi, bce_iter, 'rs-', linewidth=2, label='BCE')

# Mengatur judul dan label
ax3.set_title("Konvergensi Loss (p -> 1.0)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Iterasi (p meningkat)")
ax3.set_ylabel("Loss")
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Baris 2: Visualisasi efek pada data berbeda ---

# --- Subplot 4: Loss pada beberapa sampel ---
ax4 = axes[1, 0]

# Mendefinisikan beberapa skenario klasifikasi
skenario_labels = ['Baik\n(p=0.9)', 'Sedang\n(p=0.6)', 'Buruk\n(p=0.3)', 'S.Buruk\n(p=0.1)']
skenario_probs = [0.9, 0.6, 0.3, 0.1]

# Menghitung loss untuk setiap skenario (target=1)
mse_skenario = [(p - 1.0) ** 2 for p in skenario_probs]
bce_skenario = [-np.log(p) for p in skenario_probs]

# Membuat posisi bar
x_pos = np.arange(len(skenario_labels))
bar_width = 0.35

# Menggambar bar chart MSE
ax4.bar(x_pos - bar_width / 2, mse_skenario, bar_width, label='MSE', color='steelblue')

# Menggambar bar chart BCE
ax4.bar(x_pos + bar_width / 2, bce_skenario, bar_width, label='BCE', color='indianred')

# Mengatur label x
ax4.set_xticks(x_pos)
ax4.set_xticklabels(skenario_labels, fontsize=9)

# Mengatur judul dan label
ax4.set_title("Loss per Skenario Prediksi", fontsize=11, fontweight='bold')
ax4.set_ylabel("Loss")
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# --- Subplot 5: Dice Loss visualisasi ---
ax5 = axes[1, 1]

# Membuat beberapa mask prediksi untuk Dice Loss
overlap_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
dice_losses_bar = []

for ov in overlap_levels:
    # Membuat mask prediksi dengan overlap tertentu
    pred_mask = np.zeros_like(mask_true)
    n_ov = int(ov * np.sum(mask_true))
    flat_t = mask_true.flatten()
    flat_p = np.zeros_like(flat_t)
    idx_true = np.where(flat_t == 1.0)[0]
    if n_ov > 0 and len(idx_true) > 0:
        flat_p[idx_true[:min(n_ov, len(idx_true))]] = 1.0
    pred_mask = flat_p.reshape(mask_true.shape)
    # Menghitung Dice Loss
    dl = dice_loss(mask_true, pred_mask)
    dice_losses_bar.append(dl)

# Menggambar bar chart Dice Loss
colors_bar = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c']
ax5.bar(range(len(overlap_levels)), dice_losses_bar, color=colors_bar)

# Mengatur label x
ax5.set_xticks(range(len(overlap_levels)))
ax5.set_xticklabels([f'{int(o*100)}%' for o in overlap_levels])

# Mengatur judul dan label
ax5.set_title("Dice Loss vs Overlap %", fontsize=11, fontweight='bold')
ax5.set_xlabel("Overlap dengan Ground Truth")
ax5.set_ylabel("Dice Loss")
ax5.grid(True, alpha=0.3, axis='y')

# --- Subplot 6: Tabel ringkasan ---
ax6 = axes[1, 2]

# Menyembunyikan axes
ax6.axis('off')

# Membuat tabel ringkasan
tabel_data = [
    ['MSE', 'Regresi', '(y-ŷ)²', 'Smooth'],
    ['CE', 'Multi-kelas', '-Σy·log(ŷ)', 'Tajam'],
    ['BCE', 'Biner', '-[y·log(p)+(1-y)·log(1-p)]', 'Tajam'],
    ['Dice', 'Segmentasi', '1 - 2|A∩B|/(|A|+|B|)', 'Overlap'],
    ['Hinge', 'SVM', 'max(0, 1-y·ŷ)', 'Margin'],
]

# Membuat header tabel
kolom_header = ['Loss', 'Tugas', 'Formula', 'Sifat']

# Menggambar tabel
tabel = ax6.table(cellText=tabel_data, colLabels=kolom_header,
                  loc='center', cellLoc='center')

# Mengatur ukuran font tabel
tabel.auto_set_font_size(False)
tabel.set_fontsize(9)

# Mengatur skala tabel
tabel.scale(1.0, 1.6)

# Mewarnai header tabel
for j in range(len(kolom_header)):
    tabel[0, j].set_facecolor('#4CAF50')
    tabel[0, j].set_text_props(color='white', fontweight='bold')

# Mengatur judul
ax6.set_title("Ringkasan Loss Functions", fontsize=11, fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_3 = os.path.join(OUTPUT_DIR, "15_loss_comparison.png")

# Menyimpan figure
plt.savefig(output_path_3, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_3}")

# ============================================================
# 9. Analisis numerik tambahan
# ============================================================
print("\n--- 9. Analisis Numerik Tambahan ---")

# Menghitung gradien di berbagai titik untuk setiap loss
print("\n  Tabel gradien loss di berbagai prediksi (target=1):")
print(f"  {'Prediksi':<12} {'|grad MSE|':<14} {'|grad BCE|':<14}")
print(f"  {'-'*40}")

# Menghitung dan menampilkan gradien di beberapa titik
for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    # Menghitung gradien MSE
    g_mse = abs(2.0 * (p - 1.0))

    # Menghitung gradien BCE
    g_bce = 1.0 / p

    # Menampilkan hasil
    print(f"  p = {p:<8.1f} {g_mse:<14.4f} {g_bce:<14.4f}")

# Menjelaskan insight dari gradien
print("""
  Insight:
  - BCE memiliki gradien sangat besar saat prediksi jauh dari target
    -> Pembelajaran lebih agresif saat model sangat salah
  - MSE memiliki gradien proporsional terhadap error
    -> Pembelajaran lebih halus dan stabil
  - Untuk klasifikasi, BCE biasanya lebih baik karena
    memberikan sinyal kuat saat model "sangat salah"
""")

# ============================================================
# 10. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 15")
print("=" * 60)
print(f"""
Konsep yang telah dipelajari:
1. MSE Loss: rata-rata kuadrat selisih, cocok untuk regresi
2. Cross-Entropy Loss: mengukur perbedaan distribusi probabilitas
3. Binary Cross-Entropy: CE untuk kasus biner (2 kelas)
4. Dice Loss: mengukur overlap, cocok untuk segmentasi
5. Loss landscape menunjukkan "permukaan" kerugian di ruang parameter
6. Gradient descent mengikuti arah gradien negatif menuju minimum
7. BCE memiliki gradien lebih besar saat prediksi sangat salah
8. Pilihan loss function mempengaruhi kecepatan dan kualitas training

Output disimpan di folder: output/
- 15_loss_mse_ce.png      : Kurva MSE, CE, BCE, dan Dice Loss
- 15_loss_landscape.png   : Loss landscape 2D dan jalur gradient descent
- 15_loss_comparison.png  : Perbandingan loss function dan gradien
""")
