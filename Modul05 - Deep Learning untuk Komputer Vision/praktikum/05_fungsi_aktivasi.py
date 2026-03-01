"""
==========================================================================
PERCOBAAN 5: VISUALISASI FUNGSI AKTIVASI
==========================================================================
Program ini mempelajari berbagai fungsi aktivasi yang digunakan dalam
deep learning. Setiap fungsi aktivasi diimplementasikan secara manual,
divisualisasikan kurvanya, dan diterapkan pada gambar untuk melihat efeknya.

Fungsi utama yang dipelajari:
- ReLU (Rectified Linear Unit)    : max(0, x)
- Sigmoid                         : 1 / (1 + exp(-x))
- Tanh (Hyperbolic Tangent)       : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
- Leaky ReLU                      : max(alpha*x, x)
- Softmax                         : exp(x_i) / sum(exp(x_j))

Konsep yang dipelajari:
- Peran fungsi aktivasi dalam neural network
- Non-linearitas dan mengapa diperlukan
- Pengaruh fungsi aktivasi pada nilai piksel gambar
- Softmax untuk output klasifikasi (konversi skor ke probabilitas)
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi matematika dan array
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi kurva dan gambar
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
print("PERCOBAAN 5: VISUALISASI FUNGSI AKTIVASI")
print("=" * 60)

# ============================================================
# 1. Implementasi fungsi aktivasi
# ============================================================
print("\n--- 1. Implementasi Fungsi Aktivasi ---")

def relu(x):
    """
    ReLU (Rectified Linear Unit): f(x) = max(0, x)
    Fungsi aktivasi paling populer untuk hidden layers CNN.
    Kelebihan: cepat, menghindari vanishing gradient.
    """
    # Mengembalikan maksimum antara 0 dan x
    return np.maximum(0, x)

def sigmoid(x):
    """
    Sigmoid: f(x) = 1 / (1 + exp(-x))
    Output range [0, 1]. Cocok untuk binary classification.
    """
    # Menghitung fungsi sigmoid dengan clipping untuk stabilitas numerik
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def tanh_func(x):
    """
    Tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Output range [-1, 1]. Zero-centered, lebih baik dari sigmoid.
    """
    # Menghitung hyperbolic tangent
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU: f(x) = max(alpha*x, x)
    Mengatasi masalah 'dying ReLU' dengan gradien kecil untuk x < 0.
    """
    # Mengembalikan x jika positif, alpha*x jika negatif
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    """
    Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))
    Mengkonversi vektor skor menjadi distribusi probabilitas.
    Output sum = 1.0, semua nilai positif.
    """
    # Mengurangi nilai maksimum untuk stabilitas numerik
    x_shifted = x - np.max(x)
    # Menghitung eksponen
    exp_x = np.exp(x_shifted)
    # Membagi dengan total sum untuk mendapatkan probabilitas
    return exp_x / exp_x.sum()

def elu(x, alpha=1.0):
    """
    ELU (Exponential Linear Unit): f(x) = x if x > 0, alpha*(exp(x)-1) if x <= 0
    Memiliki output negatif yang smooth.
    """
    # Mengembalikan x jika positif, alpha*(exp(x)-1) jika negatif
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))

# Menampilkan informasi setiap fungsi aktivasi
print("  Fungsi aktivasi yang diimplementasikan:")
print("  1. ReLU       : f(x) = max(0, x)")
print("  2. Sigmoid    : f(x) = 1 / (1 + exp(-x))")
print("  3. Tanh       : f(x) = tanh(x)")
print("  4. Leaky ReLU : f(x) = max(0.01*x, x)")
print("  5. Softmax    : f(x_i) = exp(x_i) / sum(exp(x_j))")
print("  6. ELU        : f(x) = x if x>0, alpha*(exp(x)-1) if x<=0")

# ============================================================
# 2. Visualisasi kurva fungsi aktivasi
# ============================================================
print("\n--- 2. Visualisasi Kurva Fungsi Aktivasi ---")

# Membuat range nilai x untuk plotting kurva
x = np.linspace(-6, 6, 500)

# Menghitung output untuk setiap fungsi aktivasi
y_relu = relu(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh_func(x)
y_leaky = leaky_relu(x, alpha=0.1)
y_elu = elu(x)

# Menghitung turunan (derivative) untuk setiap fungsi
# Turunan ReLU: 0 jika x < 0, 1 jika x > 0
dy_relu = np.where(x > 0, 1.0, 0.0)

# Turunan Sigmoid: sigmoid(x) * (1 - sigmoid(x))
s = sigmoid(x)
dy_sigmoid = s * (1 - s)

# Turunan Tanh: 1 - tanh(x)^2
dy_tanh = 1 - np.tanh(x) ** 2

# Turunan Leaky ReLU: 1 jika x > 0, alpha jika x < 0
dy_leaky = np.where(x > 0, 1.0, 0.1)

# Turunan ELU: 1 jika x > 0, alpha*exp(x) jika x <= 0
dy_elu = np.where(x > 0, 1.0, np.exp(np.clip(x, -500, 500)))

# Membuat figure dengan 2 baris: fungsi dan turunannya
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- Subplot 1: ReLU ---
axes[0, 0].plot(x, y_relu, 'b-', linewidth=2, label='ReLU')
axes[0, 0].plot(x, dy_relu, 'r--', linewidth=1.5, label="ReLU'")
axes[0, 0].axhline(y=0, color='gray', linewidth=0.5)
axes[0, 0].axvline(x=0, color='gray', linewidth=0.5)
axes[0, 0].set_title("ReLU: max(0, x)", fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].set_ylim(-2, 6)
axes[0, 0].grid(alpha=0.3)

# --- Subplot 2: Sigmoid ---
axes[0, 1].plot(x, y_sigmoid, 'b-', linewidth=2, label='Sigmoid')
axes[0, 1].plot(x, dy_sigmoid, 'r--', linewidth=1.5, label="Sigmoid'")
axes[0, 1].axhline(y=0, color='gray', linewidth=0.5)
axes[0, 1].axvline(x=0, color='gray', linewidth=0.5)
axes[0, 1].set_title("Sigmoid: 1/(1+exp(-x))", fontsize=11, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

# --- Subplot 3: Tanh ---
axes[0, 2].plot(x, y_tanh, 'b-', linewidth=2, label='Tanh')
axes[0, 2].plot(x, dy_tanh, 'r--', linewidth=1.5, label="Tanh'")
axes[0, 2].axhline(y=0, color='gray', linewidth=0.5)
axes[0, 2].axvline(x=0, color='gray', linewidth=0.5)
axes[0, 2].set_title("Tanh: (e^x - e^-x)/(e^x + e^-x)", fontsize=11, fontweight='bold')
axes[0, 2].legend(fontsize=9)
axes[0, 2].grid(alpha=0.3)

# --- Subplot 4: Leaky ReLU ---
axes[1, 0].plot(x, y_leaky, 'b-', linewidth=2, label='Leaky ReLU')
axes[1, 0].plot(x, dy_leaky, 'r--', linewidth=1.5, label="Leaky ReLU'")
axes[1, 0].axhline(y=0, color='gray', linewidth=0.5)
axes[1, 0].axvline(x=0, color='gray', linewidth=0.5)
axes[1, 0].set_title("Leaky ReLU: max(0.1x, x)", fontsize=11, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].set_ylim(-2, 6)
axes[1, 0].grid(alpha=0.3)

# --- Subplot 5: ELU ---
axes[1, 1].plot(x, y_elu, 'b-', linewidth=2, label='ELU')
axes[1, 1].plot(x, dy_elu, 'r--', linewidth=1.5, label="ELU'")
axes[1, 1].axhline(y=0, color='gray', linewidth=0.5)
axes[1, 1].axvline(x=0, color='gray', linewidth=0.5)
axes[1, 1].set_title("ELU: x if x>0, exp(x)-1 if x<=0", fontsize=11, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].set_ylim(-2, 6)
axes[1, 1].grid(alpha=0.3)

# --- Subplot 6: Semua fungsi dalam satu plot ---
axes[1, 2].plot(x, y_relu, linewidth=2, label='ReLU')
axes[1, 2].plot(x, y_sigmoid, linewidth=2, label='Sigmoid')
axes[1, 2].plot(x, y_tanh, linewidth=2, label='Tanh')
axes[1, 2].plot(x, y_leaky, linewidth=2, label='Leaky ReLU')
axes[1, 2].plot(x, y_elu, linewidth=2, label='ELU')
axes[1, 2].axhline(y=0, color='gray', linewidth=0.5)
axes[1, 2].axvline(x=0, color='gray', linewidth=0.5)
axes[1, 2].set_title("Semua Fungsi Aktivasi", fontsize=11, fontweight='bold')
axes[1, 2].legend(fontsize=8)
axes[1, 2].set_ylim(-2, 6)
axes[1, 2].grid(alpha=0.3)

# Menambahkan label sumbu pada semua subplot
for ax in axes.flat:
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

# Menambahkan judul utama
plt.suptitle("Percobaan 5: Kurva Fungsi Aktivasi dan Turunannya",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi kurva ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "05_fungsi_aktivasi_kurva.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/05_fungsi_aktivasi_kurva.png")

# ============================================================
# 3. Menerapkan fungsi aktivasi pada gambar
# ============================================================
print("\n--- 3. Menerapkan Fungsi Aktivasi pada Gambar ---")

# Memuat gambar grayscale
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"), cv2.IMREAD_GRAYSCALE)

# Memeriksa apakah gambar berhasil dimuat
if img is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Meresize gambar ke ukuran yang konsisten
img = cv2.resize(img, (256, 256))

# Menampilkan informasi gambar
print(f"  Ukuran gambar: {img.shape}")
print(f"  Range piksel : [{img.min()}, {img.max()}]")

# Menormalisasi gambar ke range [-1, 1] untuk fungsi aktivasi
img_norm = img.astype(np.float32) / 127.5 - 1.0

# Menampilkan range setelah normalisasi
print(f"  Range setelah normalisasi: [{img_norm.min():.2f}, {img_norm.max():.2f}]")

# Menerapkan setiap fungsi aktivasi pada gambar
img_relu = relu(img_norm)
img_sigmoid = sigmoid(img_norm * 5)  # Scaling untuk visualisasi yang jelas
img_tanh = tanh_func(img_norm * 3)   # Scaling untuk visualisasi yang jelas
img_leaky = leaky_relu(img_norm, alpha=0.1)
img_elu = elu(img_norm)

# Menampilkan statistik setiap hasil aktivasi
print(f"\n  Statistik setelah aktivasi:")
print(f"  ReLU       : range=[{img_relu.min():.3f}, {img_relu.max():.3f}], mean={img_relu.mean():.3f}")
print(f"  Sigmoid    : range=[{img_sigmoid.min():.3f}, {img_sigmoid.max():.3f}], mean={img_sigmoid.mean():.3f}")
print(f"  Tanh       : range=[{img_tanh.min():.3f}, {img_tanh.max():.3f}], mean={img_tanh.mean():.3f}")
print(f"  Leaky ReLU : range=[{img_leaky.min():.3f}, {img_leaky.max():.3f}], mean={img_leaky.mean():.3f}")
print(f"  ELU        : range=[{img_elu.min():.3f}, {img_elu.max():.3f}], mean={img_elu.mean():.3f}")

# ============================================================
# 4. Visualisasi efek aktivasi pada gambar
# ============================================================
print("\n--- 4. Visualisasi Efek Aktivasi pada Gambar ---")

# Membuat figure dengan subplot untuk setiap aktivasi
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Menampilkan gambar asli (normalisasi)
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Menampilkan hasil ReLU
axes[0, 1].imshow(img_relu, cmap='gray')
axes[0, 1].set_title("ReLU\n(Nilai negatif = 0)", fontsize=10)
axes[0, 1].axis('off')

# Menampilkan hasil Sigmoid
axes[0, 2].imshow(img_sigmoid, cmap='gray')
axes[0, 2].set_title("Sigmoid\n(Range [0, 1])", fontsize=10)
axes[0, 2].axis('off')

# Menampilkan hasil Tanh
axes[1, 0].imshow(img_tanh, cmap='gray')
axes[1, 0].set_title("Tanh\n(Range [-1, 1])", fontsize=10)
axes[1, 0].axis('off')

# Menampilkan hasil Leaky ReLU
axes[1, 1].imshow(img_leaky, cmap='gray')
axes[1, 1].set_title("Leaky ReLU\n(Gradien kecil untuk negatif)", fontsize=10)
axes[1, 1].axis('off')

# Menampilkan hasil ELU
axes[1, 2].imshow(img_elu, cmap='gray')
axes[1, 2].set_title("ELU\n(Smooth negatif)", fontsize=10)
axes[1, 2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 5: Efek Fungsi Aktivasi pada Gambar Grayscale",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "05_aktivasi_pada_gambar.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/05_aktivasi_pada_gambar.png")

# ============================================================
# 5. Demonstrasi Softmax untuk klasifikasi
# ============================================================
print("\n--- 5. Demonstrasi Softmax untuk Klasifikasi ---")

# Mendefinisikan skor mentah (logits) dari neural network
logits_sampel = np.array([2.5, 1.2, 0.8, 3.1, -0.5, 0.3, -1.0, 1.8, 0.1, -0.3])

# Mendefinisikan nama kelas
nama_kelas = ["Kucing", "Anjing", "Mobil", "Bunga", "Gedung",
              "Burung", "Ikan", "Sepeda", "Pohon", "Rumah"]

# Menerapkan softmax pada logits
probabilitas = softmax(logits_sampel)

# Menampilkan hasil softmax
print(f"  Logits (skor mentah dari network):")
for i, (kelas, logit, prob) in enumerate(zip(nama_kelas, logits_sampel, probabilitas)):
    # Menampilkan kelas, logit, dan probabilitas
    bar = "█" * int(prob * 50)
    print(f"    {kelas:8s}: logit={logit:6.2f} -> prob={prob:.4f} ({prob*100:.1f}%) {bar}")

# Memverifikasi bahwa total probabilitas = 1
print(f"\n  Total probabilitas: {probabilitas.sum():.6f} (harus = 1.0)")

# Menampilkan prediksi teratas
idx_top = np.argmax(probabilitas)
print(f"  Prediksi: {nama_kelas[idx_top]} (confidence: {probabilitas[idx_top]*100:.1f}%)")

# Mendemonstrasikan efek temperature pada softmax
print("\n  Efek Temperature pada Softmax:")
for temp in [0.5, 1.0, 2.0, 5.0]:
    # Membagi logits dengan temperature sebelum softmax
    prob_temp = softmax(logits_sampel / temp)
    # Menampilkan entropy (ukuran ketidakpastian distribusi)
    entropy = -np.sum(prob_temp * np.log(prob_temp + 1e-10))
    print(f"    Temperature={temp:.1f}: max_prob={prob_temp.max():.4f}, entropy={entropy:.4f}")

# ============================================================
# 6. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 5")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. ReLU adalah fungsi aktivasi paling populer: cepat dan efektif
2. Sigmoid menghasilkan output [0,1], cocok untuk binary classification
3. Tanh menghasilkan output [-1,1], zero-centered
4. Leaky ReLU mengatasi 'dying ReLU' dengan gradien kecil untuk negatif
5. ELU memiliki output negatif yang smooth
6. Softmax mengkonversi logits menjadi distribusi probabilitas (sum=1)
7. Temperature scaling mengontrol 'ketajaman' distribusi softmax
8. Fungsi aktivasi memberikan non-linearitas yang diperlukan neural network

Output disimpan di folder: output/
- 05_fungsi_aktivasi_kurva.png  : Kurva fungsi aktivasi dan turunannya
- 05_aktivasi_pada_gambar.png   : Efek aktivasi pada gambar grayscale
""")
