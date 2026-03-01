"""
==========================================================================
PERCOBAAN 19: ONNX DAN DEPLOYMENT - KONSEP
==========================================================================
Program ini mempelajari konsep ONNX (Open Neural Network Exchange) dan
deployment model deep learning. ONNX adalah format standar untuk
pertukaran model antar framework. Program mendemonstrasikan serialisasi
model, kuantisasi, perbandingan tipe data, dan konsep deployment
menggunakan implementasi sederhana berbasis NumPy.

Fungsi utama yang dipelajari:
- np.save() / np.load()     : Menyimpan dan memuat array (bobot model)
- np.float32 / np.float16   : Tipe data presisi berbeda
- np.clip() / np.round()    : Operasi kuantisasi
- Serialisasi model         : Menyimpan arsitektur + bobot
- Inferensi model           : Melakukan prediksi dari bobot tersimpan

Konsep yang dipelajari:
- Format ONNX dan perannya dalam ekosistem deep learning
- Serialisasi dan deserialisasi model (save/load)
- Kuantisasi: mengurangi presisi bilangan untuk efisiensi
- Trade-off model size vs accuracy
- Benchmarking inferensi dengan berbagai tipe data
- Model compression dan dampaknya pada kualitas output
==========================================================================
"""

# Mengimpor NumPy untuk operasi komputasi dan simulasi model
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor time untuk mengukur waktu inferensi
import time

# Mengimpor matplotlib untuk visualisasi grafik
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
print("PERCOBAAN 19: ONNX DAN DEPLOYMENT - KONSEP")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep ONNX dan deployment
# ============================================================
print("\n--- 1. Konsep ONNX dan Deployment ---")

# Menjelaskan konsep ONNX
print("""
  ONNX (Open Neural Network Exchange):
  - Format standar untuk merepresentasikan model deep learning
  - Memungkinkan transfer model antar framework
    (PyTorch -> ONNX -> TensorFlow, dll.)
  - Didukung oleh Microsoft, Facebook, Amazon, dll.

  Alur Deployment Model:
  1. Training   : Latih model di framework (PyTorch/TF)
  2. Export      : Konversi model ke format ONNX
  3. Optimize    : Kuantisasi, pruning, knowledge distillation
  4. Deploy      : Jalankan di target (server, mobile, edge)

  | Aspek            | Training        | Deployment          |
  |------------------|-----------------|---------------------|
  | Prioritas        | Akurasi         | Kecepatan & ukuran  |
  | Presisi          | float32/float64 | float16/int8        |
  | Hardware         | GPU             | CPU/Mobile/Edge     |
  | Framework        | PyTorch/TF      | ONNX Runtime/TFLite |
""")

# ============================================================
# 2. Membangun model sederhana (linear classifier)
# ============================================================
print("\n--- 2. Membangun Model Sederhana ---")

# Mengatur random seed untuk reproducibility
np.random.seed(42)

# Mendefinisikan arsitektur model sederhana (2 layer)
# Layer 1: input(10) -> hidden(32) dengan bias
# Layer 2: hidden(32) -> output(5) dengan bias

# Menginisialisasi bobot layer 1 (Xavier initialization)
w1 = np.random.randn(10, 32).astype(np.float32) * np.sqrt(2.0 / 10)

# Menginisialisasi bias layer 1
b1 = np.zeros(32, dtype=np.float32)

# Menginisialisasi bobot layer 2
w2 = np.random.randn(32, 5).astype(np.float32) * np.sqrt(2.0 / 32)

# Menginisialisasi bias layer 2
b2 = np.zeros(5, dtype=np.float32)


def relu(x):
    """Fungsi aktivasi ReLU."""
    # Mengembalikan max(0, x)
    return np.maximum(0, x)


def softmax(x):
    """Fungsi aktivasi Softmax untuk output probabilitas."""
    # Mengurangi max untuk stabilitas numerik
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

    # Menormalisasi agar jumlah = 1
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def forward_pass(x, w1, b1, w2, b2):
    """
    Forward pass melalui neural network 2-layer.
    """
    # Layer 1: linear + ReLU
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)

    # Layer 2: linear + softmax
    z2 = np.dot(a1, w2) + b2
    output = softmax(z2)

    # Mengembalikan output probabilitas
    return output


# Membuat data input contoh
x_contoh = np.random.randn(5, 10).astype(np.float32)

# Melakukan forward pass
output_contoh = forward_pass(x_contoh, w1, b1, w2, b2)

# Menampilkan informasi model
print(f"  Arsitektur model: Input(10) -> Hidden(32) -> Output(5)")
print(f"  Total parameter : {w1.size + b1.size + w2.size + b2.size}")
print(f"  W1 shape: {w1.shape}, B1 shape: {b1.shape}")
print(f"  W2 shape: {w2.shape}, B2 shape: {b2.shape}")
print(f"  Contoh output (5 sampel):")

# Menampilkan prediksi untuk setiap sampel
for i in range(5):
    # Mendapatkan kelas prediksi
    pred_kelas = np.argmax(output_contoh[i])

    # Mendapatkan probabilitas
    prob = output_contoh[i, pred_kelas]

    # Menampilkan hasil
    print(f"    Sampel {i+1}: kelas={pred_kelas}, prob={prob:.4f}")

# ============================================================
# 3. Serialisasi model (save/load)
# ============================================================
print("\n--- 3. Serialisasi Model ---")

# Mendefinisikan path untuk menyimpan model
model_dir = os.path.join(OUTPUT_DIR, "model_weights")

# Membuat folder model jika belum ada
os.makedirs(model_dir, exist_ok=True)

# Menyimpan bobot model menggunakan np.save
np.save(os.path.join(model_dir, "w1.npy"), w1)
np.save(os.path.join(model_dir, "b1.npy"), b1)
np.save(os.path.join(model_dir, "w2.npy"), w2)
np.save(os.path.join(model_dir, "b2.npy"), b2)

# Menyimpan semua bobot dalam satu file .npz
np.savez(os.path.join(model_dir, "model_lengkap.npz"),
         w1=w1, b1=b1, w2=w2, b2=b2)

# Menghitung ukuran file
ukuran_npy = sum(os.path.getsize(os.path.join(model_dir, f))
                  for f in ["w1.npy", "b1.npy", "w2.npy", "b2.npy"])
ukuran_npz = os.path.getsize(os.path.join(model_dir, "model_lengkap.npz"))

# Menampilkan informasi serialisasi
print(f"  Model disimpan di: {model_dir}")
print(f"  Ukuran .npy terpisah: {ukuran_npy:,} bytes ({ukuran_npy/1024:.1f} KB)")
print(f"  Ukuran .npz gabungan: {ukuran_npz:,} bytes ({ukuran_npz/1024:.1f} KB)")

# Memuat kembali model dari file
print("\n  Memuat model dari file...")

# Memuat bobot dari file .npz
loaded = np.load(os.path.join(model_dir, "model_lengkap.npz"))

# Mengekstrak bobot
w1_loaded = loaded['w1']
b1_loaded = loaded['b1']
w2_loaded = loaded['w2']
b2_loaded = loaded['b2']

# Melakukan forward pass dengan bobot yang dimuat
output_loaded = forward_pass(x_contoh, w1_loaded, b1_loaded, w2_loaded, b2_loaded)

# Memverifikasi bahwa output identik
selisih_max = np.max(np.abs(output_contoh - output_loaded))

# Menampilkan hasil verifikasi
print(f"  Selisih max (asli vs loaded): {selisih_max:.2e}")
print(f"  Verifikasi save/load: {'BERHASIL' if selisih_max < 1e-6 else 'GAGAL'}")

# ============================================================
# 4. Kuantisasi: mengurangi presisi
# ============================================================
print("\n--- 4. Kuantisasi Model ---")


def quantize_to_float16(weights):
    """
    Kuantisasi float32 -> float16.
    """
    # Mengkonversi ke float16
    return weights.astype(np.float16)


def quantize_to_int8(weights, scale=None):
    """
    Kuantisasi float32 -> int8 (linear quantization).
    Formula: q = round(w / scale), dequantize: w_approx = q * scale
    """
    if scale is None:
        # Menghitung scale berdasarkan range bobot
        abs_max = np.max(np.abs(weights))
        scale = abs_max / 127.0

    # Melakukan kuantisasi
    q = np.round(weights / scale).astype(np.int8)

    # Mengembalikan quantized weights dan scale
    return q, scale


def dequantize_int8(q_weights, scale):
    """
    Dequantize int8 kembali ke float32.
    """
    # Mengembalikan ke float32
    return q_weights.astype(np.float32) * scale


# Melakukan kuantisasi float16
w1_fp16 = quantize_to_float16(w1)
b1_fp16 = quantize_to_float16(b1)
w2_fp16 = quantize_to_float16(w2)
b2_fp16 = quantize_to_float16(b2)

# Melakukan kuantisasi int8
w1_int8, s_w1 = quantize_to_int8(w1)
b1_int8, s_b1 = quantize_to_int8(b1)
w2_int8, s_w2 = quantize_to_int8(w2)
b2_int8, s_b2 = quantize_to_int8(b2)

# Menampilkan perbandingan ukuran memori
ukuran_fp32 = w1.nbytes + b1.nbytes + w2.nbytes + b2.nbytes
ukuran_fp16 = w1_fp16.nbytes + b1_fp16.nbytes + w2_fp16.nbytes + b2_fp16.nbytes
ukuran_int8_total = w1_int8.nbytes + b1_int8.nbytes + w2_int8.nbytes + b2_int8.nbytes

# Menampilkan hasil kuantisasi
print(f"  Ukuran memori model:")
print(f"    Float32 (asli)  : {ukuran_fp32:>6,} bytes")
print(f"    Float16         : {ukuran_fp16:>6,} bytes ({ukuran_fp16/ukuran_fp32*100:.0f}%)")
print(f"    Int8            : {ukuran_int8_total:>6,} bytes ({ukuran_int8_total/ukuran_fp32*100:.0f}%)")

# Menghitung error kuantisasi
# Float16
w1_fp16_back = w1_fp16.astype(np.float32)
error_fp16 = np.mean(np.abs(w1 - w1_fp16_back))

# Int8
w1_int8_back = dequantize_int8(w1_int8, s_w1)
error_int8 = np.mean(np.abs(w1 - w1_int8_back))

# Menampilkan error kuantisasi
print(f"\n  Error kuantisasi (rata-rata absolut pada W1):")
print(f"    Float16 -> Float32 : {error_fp16:.6f}")
print(f"    Int8 -> Float32    : {error_int8:.6f}")

# ============================================================
# 5. Perbandingan inferensi dengan berbagai presisi
# ============================================================
print("\n--- 5. Perbandingan Inferensi ---")

# Membuat data input batch besar untuk benchmark
n_benchmark = 1000
x_bench = np.random.randn(n_benchmark, 10).astype(np.float32)

# --- Inferensi float32 ---
t_start = time.time()
for _ in range(100):
    # Melakukan forward pass float32
    out_fp32 = forward_pass(x_bench, w1, b1, w2, b2)
t_fp32 = (time.time() - t_start) / 100

# --- Inferensi float16 ---
x_bench_fp16 = x_bench.astype(np.float16)
t_start = time.time()
for _ in range(100):
    # Melakukan forward pass float16
    z1_16 = np.dot(x_bench_fp16, w1_fp16) + b1_fp16
    a1_16 = np.maximum(0, z1_16)
    z2_16 = np.dot(a1_16, w2_fp16) + b2_fp16
    # Softmax
    exp_z2_16 = np.exp(z2_16.astype(np.float32) - np.max(z2_16.astype(np.float32),
                       axis=-1, keepdims=True))
    out_fp16 = exp_z2_16 / np.sum(exp_z2_16, axis=-1, keepdims=True)
t_fp16 = (time.time() - t_start) / 100

# --- Inferensi int8 (dequantize lalu compute) ---
w1_deq = dequantize_int8(w1_int8, s_w1)
b1_deq = dequantize_int8(b1_int8, s_b1)
w2_deq = dequantize_int8(w2_int8, s_w2)
b2_deq = dequantize_int8(b2_int8, s_b2)

t_start = time.time()
for _ in range(100):
    # Melakukan forward pass dengan bobot dequantized
    out_int8 = forward_pass(x_bench, w1_deq, b1_deq, w2_deq, b2_deq)
t_int8 = (time.time() - t_start) / 100

# Menghitung perbedaan output
diff_fp16 = np.mean(np.abs(out_fp32 - out_fp16))
diff_int8 = np.mean(np.abs(out_fp32 - out_int8))

# Menghitung akurasi relatif (prediksi kelas yang sama)
pred_fp32 = np.argmax(out_fp32, axis=1)
pred_fp16_cls = np.argmax(out_fp16, axis=1)
pred_int8_cls = np.argmax(out_int8, axis=1)

acc_fp16 = np.mean(pred_fp32 == pred_fp16_cls) * 100
acc_int8 = np.mean(pred_fp32 == pred_int8_cls) * 100

# Menampilkan hasil benchmark
print(f"  Benchmark inferensi ({n_benchmark} sampel, rata-rata 100 kali):")
print(f"  {'Presisi':<12} {'Waktu (ms)':<14} {'Perbedaan':<14} {'Kesepakatan':<12}")
print(f"  {'-'*52}")
print(f"  {'Float32':<12} {t_fp32*1000:<14.3f} {'baseline':<14} {'100.0%':<12}")
print(f"  {'Float16':<12} {t_fp16*1000:<14.3f} {diff_fp16:<14.6f} {acc_fp16:<10.1f}%")
print(f"  {'Int8':<12} {t_int8*1000:<14.3f} {diff_int8:<14.6f} {acc_int8:<10.1f}%")

# ============================================================
# 6. Visualisasi serialisasi model (Gambar 1)
# ============================================================
print("\n--- 6. Visualisasi Serialisasi Model ---")

# Membuat figure untuk visualisasi serialisasi
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Memberikan judul utama
fig.suptitle("Model Serialisasi dan Struktur Bobot",
             fontsize=16, fontweight='bold')

# --- Subplot 1: Distribusi bobot W1 ---
ax1 = axes[0, 0]

# Menggambar histogram distribusi bobot W1
ax1.hist(w1.flatten(), bins=50, color='steelblue', alpha=0.7,
         edgecolor='black', linewidth=0.5, label='Float32')

# Menggambar distribusi W1 setelah kuantisasi
w1_deq_flat = dequantize_int8(w1_int8, s_w1).flatten()
ax1.hist(w1_deq_flat, bins=50, color='indianred', alpha=0.5,
         edgecolor='black', linewidth=0.5, label='Int8 (dequant)')

# Mengatur judul dan label
ax1.set_title("Distribusi Bobot W1", fontsize=11, fontweight='bold')
ax1.set_xlabel("Nilai Bobot")
ax1.set_ylabel("Frekuensi")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Heatmap bobot W1 ---
ax2 = axes[0, 1]

# Menampilkan heatmap bobot W1 (10x32)
im = ax2.imshow(w1, aspect='auto', cmap='RdBu_r', interpolation='nearest')

# Menambahkan colorbar
plt.colorbar(im, ax=ax2, label='Nilai Bobot')

# Mengatur judul dan label
ax2.set_title("Heatmap Bobot W1 (Input->Hidden)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Hidden Neuron (32)")
ax2.set_ylabel("Input Feature (10)")

# --- Subplot 3: Perbandingan ukuran file ---
ax3 = axes[1, 0]

# Mendefinisikan data untuk bar chart
format_names = ['.npy\nterpisah', '.npz\ngabungan', 'Float16\n(est.)', 'Int8\n(est.)']
sizes_kb = [ukuran_npy / 1024, ukuran_npz / 1024,
            ukuran_fp16 / 1024, ukuran_int8_total / 1024]
bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

# Menggambar bar chart
bars = ax3.bar(format_names, sizes_kb, color=bar_colors)

# Menambahkan label nilai
for bar, sz in zip(bars, sizes_kb):
    ax3.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.05,
             f'{sz:.1f} KB', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Mengatur judul dan label
ax3.set_title("Perbandingan Ukuran Model", fontsize=11, fontweight='bold')
ax3.set_ylabel("Ukuran (KB)")
ax3.grid(True, alpha=0.3, axis='y')

# --- Subplot 4: Alur deployment ---
ax4 = axes[1, 1]

# Menyembunyikan axes
ax4.axis('off')

# Membuat diagram alur deployment sebagai teks
alur_text = """
Alur Deployment Model Deep Learning:

1. [TRAINING]
   - Latih model di GPU (float32)
   - Evaluasi: akurasi, loss

2. [EXPORT]
   - Simpan bobot (np.save / torch.save)
   - Konversi ke ONNX (opsional)

3. [OPTIMASI]
   - Kuantisasi: float32 → float16 → int8
   - Pruning: hapus bobot kecil
   - Knowledge distillation

4. [DEPLOYMENT]
   - Load model di target platform
   - Inferensi: input → output
   - Monitor: latency, throughput
"""

# Menambahkan teks
ax4.text(0.1, 0.95, alur_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Mengatur judul
ax4.set_title("Alur Deployment", fontsize=11, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_1 = os.path.join(OUTPUT_DIR, "19_model_serialisasi.png")

# Menyimpan figure
plt.savefig(output_path_1, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_1}")

# ============================================================
# 7. Visualisasi efek kuantisasi (Gambar 2)
# ============================================================
print("\n--- 7. Visualisasi Efek Kuantisasi ---")

# Mencoba memuat gambar untuk demonstrasi kuantisasi visual
img_path = os.path.join(IMAGE_DIR, "kucing.jpg")
img = None

if os.path.exists(img_path):
    # Memuat gambar
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))

if img is None:
    print("[ERROR] kucing.jpg tidak ditemukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()

# Membuat figure untuk efek kuantisasi
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Memberikan judul utama
fig.suptitle("Efek Kuantisasi pada Kualitas Output",
             fontsize=16, fontweight='bold')

# --- Baris 1: Kuantisasi gambar ---
# Mendefinisikan jumlah level kuantisasi
quant_levels = [256, 64, 16, 4]

for idx, n_levels in enumerate(quant_levels):
    ax = axes[0, idx]

    # Menghitung step kuantisasi
    step = 256 / n_levels

    # Melakukan kuantisasi gambar
    img_quant = (np.floor(img.astype(np.float32) / step) * step).astype(np.uint8)

    # Menampilkan gambar terkuantisasi
    ax.imshow(img_quant)

    # Menghitung PSNR (Peak Signal-to-Noise Ratio)
    mse_val = np.mean((img.astype(np.float32) - img_quant.astype(np.float32)) ** 2)
    if mse_val > 0:
        psnr = 10 * np.log10(255.0 ** 2 / mse_val)
    else:
        psnr = float('inf')

    # Mengatur judul
    title_str = f"{n_levels} levels" if n_levels < 256 else "Original (256)"
    ax.set_title(f"{title_str}\nPSNR={psnr:.1f}dB", fontsize=10, fontweight='bold')
    ax.axis('off')

# --- Baris 2: Efek kuantisasi pada distribusi bobot ---
quant_bits = [32, 16, 8, 4]
quant_labels = ['Float32\n(asli)', 'Float16', 'Int8\n(dequant)', 'Int4\n(simulasi)']

for idx, (bits, label) in enumerate(zip(quant_bits, quant_labels)):
    ax = axes[1, idx]

    if bits == 32:
        # Float32 asli
        data = w1.flatten()
    elif bits == 16:
        # Float16
        data = w1.astype(np.float16).astype(np.float32).flatten()
    elif bits == 8:
        # Int8 dequantized
        data = dequantize_int8(w1_int8, s_w1).flatten()
    else:
        # Int4 simulasi (hanya 16 level)
        abs_max = np.max(np.abs(w1))
        scale_4 = abs_max / 7.0
        q4 = np.round(w1 / scale_4).astype(np.int8)
        q4 = np.clip(q4, -8, 7)
        data = (q4.astype(np.float32) * scale_4).flatten()

    # Menggambar histogram
    ax.hist(data, bins=50, color=['steelblue', '#FF9800', '#F44336', '#9C27B0'][idx],
            alpha=0.7, edgecolor='black', linewidth=0.5)

    # Menghitung error relatif
    if bits < 32:
        err = np.mean(np.abs(w1.flatten() - data))
        ax.set_title(f"{label}\nerr={err:.4f}", fontsize=10, fontweight='bold')
    else:
        ax.set_title(f"{label}", fontsize=10, fontweight='bold')

    # Mengatur label
    ax.set_xlabel("Nilai Bobot")
    if idx == 0:
        ax.set_ylabel("Frekuensi")

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_2 = os.path.join(OUTPUT_DIR, "19_quantisasi_efek.png")

# Menyimpan figure
plt.savefig(output_path_2, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_2}")

# ============================================================
# 8. Benchmark deployment (Gambar 3)
# ============================================================
print("\n--- 8. Benchmark Deployment ---")

# Melakukan benchmark lebih detail
batch_sizes = [1, 10, 100, 500, 1000]
waktu_fp32_list = []
waktu_fp16_list = []
waktu_int8_list = []

for bs in batch_sizes:
    # Membuat data input dengan batch size tertentu
    x_bs = np.random.randn(bs, 10).astype(np.float32)
    x_bs_fp16 = x_bs.astype(np.float16)

    # Benchmark float32
    t0 = time.time()
    for _ in range(50):
        _ = forward_pass(x_bs, w1, b1, w2, b2)
    waktu_fp32_list.append((time.time() - t0) / 50 * 1000)

    # Benchmark float16
    t0 = time.time()
    for _ in range(50):
        z = np.dot(x_bs_fp16, w1_fp16) + b1_fp16
        a = np.maximum(0, z)
        z2 = np.dot(a, w2_fp16) + b2_fp16
        _ = softmax(z2.astype(np.float32))
    waktu_fp16_list.append((time.time() - t0) / 50 * 1000)

    # Benchmark int8 (dequantize)
    t0 = time.time()
    for _ in range(50):
        _ = forward_pass(x_bs, w1_deq, b1_deq, w2_deq, b2_deq)
    waktu_int8_list.append((time.time() - t0) / 50 * 1000)

# Membuat figure untuk benchmark
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Memberikan judul utama
fig.suptitle("Benchmark Deployment: Presisi vs Performa",
             fontsize=16, fontweight='bold')

# --- Subplot 1: Waktu inferensi vs batch size ---
ax1 = axes[0, 0]

# Menggambar kurva float32
ax1.plot(batch_sizes, waktu_fp32_list, 'bo-', linewidth=2, label='Float32')

# Menggambar kurva float16
ax1.plot(batch_sizes, waktu_fp16_list, 'gs-', linewidth=2, label='Float16')

# Menggambar kurva int8
ax1.plot(batch_sizes, waktu_int8_list, 'r^-', linewidth=2, label='Int8')

# Mengatur judul dan label
ax1.set_title("Waktu Inferensi vs Batch Size", fontsize=11, fontweight='bold')
ax1.set_xlabel("Batch Size")
ax1.set_ylabel("Waktu (ms)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Throughput (samples/sec) ---
ax2 = axes[0, 1]

# Menghitung throughput
throughput_fp32 = [bs / (t / 1000) for bs, t in zip(batch_sizes, waktu_fp32_list)]
throughput_fp16 = [bs / (t / 1000) for bs, t in zip(batch_sizes, waktu_fp16_list)]
throughput_int8 = [bs / (t / 1000) for bs, t in zip(batch_sizes, waktu_int8_list)]

# Menggambar throughput
ax2.plot(batch_sizes, throughput_fp32, 'bo-', linewidth=2, label='Float32')
ax2.plot(batch_sizes, throughput_fp16, 'gs-', linewidth=2, label='Float16')
ax2.plot(batch_sizes, throughput_int8, 'r^-', linewidth=2, label='Int8')

# Mengatur judul dan label
ax2.set_title("Throughput vs Batch Size", fontsize=11, fontweight='bold')
ax2.set_xlabel("Batch Size")
ax2.set_ylabel("Samples/detik")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Subplot 3: Model size vs accuracy trade-off ---
ax3 = axes[1, 0]

# Mendefinisikan data trade-off
size_labels = ['Float32', 'Float16', 'Int8', 'Int4 (est.)']
sizes = [ukuran_fp32, ukuran_fp16, ukuran_int8_total, ukuran_int8_total // 2]
accuracies = [100.0, acc_fp16, acc_int8, max(acc_int8 - 5, 80)]

# Menggambar scatter plot
colors_sc = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
for i in range(len(size_labels)):
    ax3.scatter(sizes[i] / 1024, accuracies[i], s=200, c=colors_sc[i],
                zorder=5, edgecolors='black', linewidths=1.5)
    ax3.annotate(size_labels[i], (sizes[i] / 1024, accuracies[i]),
                 textcoords="offset points", xytext=(10, 5), fontsize=10)

# Mengatur judul dan label
ax3.set_title("Ukuran Model vs Akurasi", fontsize=11, fontweight='bold')
ax3.set_xlabel("Ukuran Model (KB)")
ax3.set_ylabel("Agreement dengan FP32 (%)")
ax3.grid(True, alpha=0.3)
ax3.set_ylim(70, 105)

# --- Subplot 4: Tabel ringkasan ---
ax4 = axes[1, 1]

# Menyembunyikan axes
ax4.axis('off')

# Membuat tabel data
tabel_data = [
    ['Float32', f'{ukuran_fp32/1024:.1f}', '100.0%', f'{t_fp32*1000:.3f}'],
    ['Float16', f'{ukuran_fp16/1024:.1f}', f'{acc_fp16:.1f}%', f'{t_fp16*1000:.3f}'],
    ['Int8', f'{ukuran_int8_total/1024:.1f}', f'{acc_int8:.1f}%', f'{t_int8*1000:.3f}'],
]

# Mendefinisikan header
kolom_header = ['Presisi', 'Ukuran (KB)', 'Agreement', 'Latency (ms)']

# Menggambar tabel
tabel = ax4.table(cellText=tabel_data, colLabels=kolom_header,
                  loc='center', cellLoc='center')

# Mengatur font
tabel.auto_set_font_size(False)
tabel.set_fontsize(11)

# Mengatur skala tabel
tabel.scale(1.0, 2.0)

# Mewarnai header
for j in range(len(kolom_header)):
    tabel[0, j].set_facecolor('#FF5722')
    tabel[0, j].set_text_props(color='white', fontweight='bold')

# Mengatur judul
ax4.set_title("Ringkasan Benchmark", fontsize=11, fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_3 = os.path.join(OUTPUT_DIR, "19_deployment_benchmark.png")

# Menyimpan figure
plt.savefig(output_path_3, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_3}")

# ============================================================
# 9. Membersihkan file temporary
# ============================================================
print("\n--- 9. Membersihkan File Temporary ---")

# Membersihkan file model yang disimpan (opsional)
# File tetap disimpan untuk referensi
print(f"  File model disimpan di: {model_dir}")
print(f"  - w1.npy, b1.npy, w2.npy, b2.npy")
print(f"  - model_lengkap.npz")

# ============================================================
# 10. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 19")
print("=" * 60)
print(f"""
Konsep yang telah dipelajari:
1. ONNX adalah format standar pertukaran model deep learning
2. Model terdiri dari arsitektur + bobot (parameter)
3. np.save/np.load untuk serialisasi array (bobot model)
4. np.savez untuk menyimpan semua bobot dalam satu file
5. Kuantisasi float32->float16 mengurangi ukuran 50%
6. Kuantisasi float32->int8 mengurangi ukuran 75%
7. Trade-off: ukuran lebih kecil vs akurasi sedikit turun
8. Deployment mempertimbangkan latency, throughput, dan ukuran

Hasil Benchmark:
- Float32: ukuran={ukuran_fp32/1024:.1f}KB, agreement=100%
- Float16: ukuran={ukuran_fp16/1024:.1f}KB, agreement={acc_fp16:.1f}%
- Int8   : ukuran={ukuran_int8_total/1024:.1f}KB, agreement={acc_int8:.1f}%

Output disimpan di folder: output/
- 19_model_serialisasi.png   : Visualisasi bobot dan ukuran model
- 19_quantisasi_efek.png     : Efek kuantisasi pada kualitas
- 19_deployment_benchmark.png: Benchmark performa deployment
""")
