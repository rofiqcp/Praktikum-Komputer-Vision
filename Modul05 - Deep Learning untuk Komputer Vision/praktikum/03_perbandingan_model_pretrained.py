"""
==========================================================================
PERCOBAAN 3: PERBANDINGAN MODEL PRE-TRAINED
==========================================================================
Program ini mempelajari perbandingan berbagai arsitektur model deep learning
pre-trained yang umum digunakan untuk klasifikasi gambar. Program juga
mendemonstrasikan perbedaan preprocessing untuk masing-masing arsitektur.

Fungsi utama yang dipelajari:
- cv2.dnn.blobFromImage()  : Membuat blob dengan berbagai parameter
- cv2.resize()             : Mengubah ukuran gambar sesuai kebutuhan model
- time.time()              : Mengukur waktu eksekusi untuk benchmark

Konsep yang dipelajari:
- Arsitektur model: MobileNet, ResNet, GoogLeNet, VGG, EfficientNet
- Perbedaan preprocessing untuk setiap arsitektur model
- Perbandingan ukuran model, kecepatan, dan akurasi
- Pengaruh ukuran input (blob size) terhadap kecepatan dan memori
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan DNN
import cv2

# Mengimpor NumPy untuk operasi array dan komputasi numerik
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi chart perbandingan
import matplotlib.pyplot as plt

# Mengimpor time untuk benchmark kecepatan
import time

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
print("PERCOBAAN 3: PERBANDINGAN MODEL PRE-TRAINED")
print("=" * 60)

# ============================================================
# 1. Informasi arsitektur model populer
# ============================================================
print("\n--- 1. Arsitektur Model Deep Learning Populer ---")

# Mendefinisikan informasi setiap arsitektur model
model_info = {
    "MobileNet V2": {
        "input_size": (224, 224),
        "params_juta": 3.4,
        "top1_accuracy": 71.8,
        "top5_accuracy": 91.0,
        "mean": (127.5, 127.5, 127.5),
        "scale": 1.0 / 127.5,
        "swap_rb": True,
        "deskripsi": "Ringan, cocok untuk mobile/embedded device",
        "tahun": 2018
    },
    "ResNet-50": {
        "input_size": (224, 224),
        "params_juta": 25.6,
        "top1_accuracy": 76.1,
        "top5_accuracy": 92.9,
        "mean": (103.94, 116.78, 123.68),
        "scale": 1.0,
        "swap_rb": False,
        "deskripsi": "Residual connections, model standar industri",
        "tahun": 2015
    },
    "GoogLeNet": {
        "input_size": (224, 224),
        "params_juta": 6.8,
        "top1_accuracy": 69.8,
        "top5_accuracy": 89.9,
        "mean": (104.0, 117.0, 123.0),
        "scale": 1.0,
        "swap_rb": False,
        "deskripsi": "Inception modules, efisien dalam parameter",
        "tahun": 2014
    },
    "VGG-16": {
        "input_size": (224, 224),
        "params_juta": 138.4,
        "top1_accuracy": 71.6,
        "top5_accuracy": 90.4,
        "mean": (103.94, 116.78, 123.68),
        "scale": 1.0,
        "swap_rb": False,
        "deskripsi": "Arsitektur sederhana, sangat besar",
        "tahun": 2014
    },
    "EfficientNet-B0": {
        "input_size": (224, 224),
        "params_juta": 5.3,
        "top1_accuracy": 77.1,
        "top5_accuracy": 93.3,
        "mean": (123.675, 116.28, 103.53),
        "scale": 1.0 / 58.395,
        "swap_rb": True,
        "deskripsi": "Compound scaling, efisien dan akurat",
        "tahun": 2019
    }
}

# Menampilkan tabel informasi model
print(f"\n  {'Model':<18} {'Params(M)':<12} {'Top-1(%)':<10} {'Top-5(%)':<10} {'Tahun':<8}")
print(f"  {'-'*58}")
for nama, info in model_info.items():
    # Menampilkan informasi setiap model dalam format tabel
    print(f"  {nama:<18} {info['params_juta']:<12.1f} {info['top1_accuracy']:<10.1f} "
          f"{info['top5_accuracy']:<10.1f} {info['tahun']:<8}")

# ============================================================
# 2. Perbedaan preprocessing untuk setiap model
# ============================================================
print("\n--- 2. Perbedaan Preprocessing per Arsitektur ---")

# Memuat gambar sample untuk demonstrasi preprocessing
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi gambar asli
print(f"  Gambar input: {img.shape}")

# Mendemonstrasikan preprocessing untuk setiap model
for nama, info in model_info.items():
    # Membuat blob sesuai parameter preprocessing model
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=info['scale'],
        size=info['input_size'],
        mean=info['mean'],
        swapRB=info['swap_rb'],
        crop=False
    )

    # Menampilkan parameter dan hasil preprocessing
    print(f"\n  {nama}:")
    print(f"    Input size  : {info['input_size']}")
    print(f"    Scale       : {info['scale']:.6f}")
    print(f"    Mean        : {info['mean']}")
    print(f"    Swap RB     : {info['swap_rb']}")
    print(f"    Blob range  : [{blob.min():.4f}, {blob.max():.4f}]")
    print(f"    Blob shape  : {blob.shape}")

# ============================================================
# 3. Perbandingan blob creation dengan berbagai ukuran
# ============================================================
print("\n--- 3. Perbandingan Blob Creation untuk Berbagai Ukuran ---")

# Mendefinisikan ukuran input yang berbeda untuk setiap arsitektur
ukuran_blob = [
    (224, 224, "MobileNet/ResNet (224x224)"),
    (299, 299, "Inception V3 (299x299)"),
    (416, 416, "YOLOv3 (416x416)"),
    (512, 512, "Custom Large (512x512)"),
    (640, 640, "YOLOv5 (640x640)")
]

# Memuat semua gambar sample untuk benchmark
gambar_files = ["kucing.jpg", "anjing.jpg", "mobil.jpg", "bunga.jpg", "gedung.jpg"]
gambar_benchmark = []
for fname in gambar_files:
    # Membaca gambar dari folder
    g = cv2.imread(os.path.join(IMAGE_DIR, fname))
    if g is not None:
        # Menambahkan gambar ke list benchmark
        gambar_benchmark.append(g)

# Menyimpan hasil benchmark untuk visualisasi
hasil_benchmark = []

# Menjalankan benchmark untuk setiap ukuran blob
for width, height, nama_model in ukuran_blob:
    # Menghitung rata-rata waktu untuk 50 iterasi
    waktu_total = 0
    memori_blob = 0
    iterasi = 50

    for _ in range(iterasi):
        for g in gambar_benchmark:
            # Mengukur waktu pembuatan blob
            start = time.time()
            blob = cv2.dnn.blobFromImage(g, 1.0/255.0, (width, height), swapRB=True)
            elapsed = time.time() - start
            waktu_total += elapsed
            memori_blob = blob.nbytes

    # Menghitung rata-rata waktu per gambar
    waktu_rata = (waktu_total / (iterasi * len(gambar_benchmark))) * 1000

    # Menyimpan hasil benchmark
    hasil_benchmark.append({
        'nama': nama_model,
        'ukuran': f"{width}x{height}",
        'waktu_ms': waktu_rata,
        'memori_kb': memori_blob / 1024
    })

    # Menampilkan hasil benchmark
    print(f"  {nama_model:35s}: {waktu_rata:.3f} ms, Memori: {memori_blob/1024:.1f} KB")

# ============================================================
# 4. Simulasi perbandingan model pada gambar yang sama
# ============================================================
print("\n--- 4. Simulasi Perbandingan Model ---")

# Mendefinisikan simulasi hasil untuk setiap model
# (Menggunakan variasi preprocessing untuk menunjukkan perbedaan output)
hasil_simulasi = {}

for nama, info in model_info.items():
    # Menyimpan hasil per model
    hasil_per_gambar = []

    for g in gambar_benchmark:
        # Membuat blob sesuai preprocessing model ini
        blob = cv2.dnn.blobFromImage(
            g,
            scalefactor=info['scale'],
            size=info['input_size'],
            mean=info['mean'],
            swapRB=info['swap_rb'],
            crop=False
        )

        # Menghitung statistik blob sebagai 'output simulasi'
        mean_val = np.mean(blob)
        std_val = np.std(blob)
        max_val = np.max(blob)

        # Menyimpan statistik
        hasil_per_gambar.append({
            'mean': mean_val,
            'std': std_val,
            'max': max_val
        })

    # Menyimpan hasil simulasi untuk model ini
    hasil_simulasi[nama] = hasil_per_gambar

    # Menghitung rata-rata statistik untuk model ini
    avg_mean = np.mean([h['mean'] for h in hasil_per_gambar])
    avg_std = np.mean([h['std'] for h in hasil_per_gambar])

    # Menampilkan ringkasan hasil simulasi
    print(f"  {nama:18s}: avg_mean={avg_mean:8.4f}, avg_std={avg_std:8.4f}")

# ============================================================
# 5. Visualisasi perbandingan model
# ============================================================
print("\n--- 5. Visualisasi Perbandingan Model ---")

# Membuat figure dengan 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Subplot 1: Jumlah Parameter ---
nama_model_list = list(model_info.keys())
params_list = [model_info[n]['params_juta'] for n in nama_model_list]

# Mendefinisikan warna untuk setiap model
warna_model = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Membuat bar chart jumlah parameter
bars1 = axes[0, 0].bar(range(len(nama_model_list)), params_list,
                        color=warna_model, edgecolor='gray')

# Menambahkan label nilai di atas setiap bar
for bar, val in zip(bars1, params_list):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}M', ha='center', fontsize=9, fontweight='bold')

# Mengatur label sumbu X
axes[0, 0].set_xticks(range(len(nama_model_list)))
axes[0, 0].set_xticklabels(nama_model_list, rotation=30, ha='right', fontsize=8)

# Menambahkan label dan judul
axes[0, 0].set_ylabel("Jumlah Parameter (Juta)")
axes[0, 0].set_title("Jumlah Parameter Model", fontweight='bold')

# Menambahkan grid untuk keterbacaan
axes[0, 0].grid(axis='y', alpha=0.3)

# --- Subplot 2: Top-1 Accuracy ---
acc_top1 = [model_info[n]['top1_accuracy'] for n in nama_model_list]
acc_top5 = [model_info[n]['top5_accuracy'] for n in nama_model_list]

# Mendefinisikan posisi bar untuk grouped bar chart
x = np.arange(len(nama_model_list))
lebar_bar = 0.35

# Membuat grouped bar chart untuk Top-1 dan Top-5
bars_top1 = axes[0, 1].bar(x - lebar_bar/2, acc_top1, lebar_bar,
                            label='Top-1', color='#3498DB', edgecolor='gray')
bars_top5 = axes[0, 1].bar(x + lebar_bar/2, acc_top5, lebar_bar,
                            label='Top-5', color='#2ECC71', edgecolor='gray')

# Mengatur label sumbu X
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(nama_model_list, rotation=30, ha='right', fontsize=8)

# Menambahkan label dan judul
axes[0, 1].set_ylabel("Akurasi (%)")
axes[0, 1].set_title("Akurasi ImageNet (Top-1 vs Top-5)", fontweight='bold')

# Menambahkan legend
axes[0, 1].legend(fontsize=9)

# Mengatur range sumbu Y untuk memperjelas perbedaan
axes[0, 1].set_ylim(60, 100)

# Menambahkan grid
axes[0, 1].grid(axis='y', alpha=0.3)

# --- Subplot 3: Efisiensi (Accuracy / Parameter) ---
efisiensi = [acc / param for acc, param in zip(acc_top1, params_list)]

# Membuat bar chart efisiensi
bars3 = axes[1, 0].bar(range(len(nama_model_list)), efisiensi,
                        color=warna_model, edgecolor='gray')

# Menambahkan label nilai di atas setiap bar
for bar, val in zip(bars3, efisiensi):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')

# Mengatur label sumbu X
axes[1, 0].set_xticks(range(len(nama_model_list)))
axes[1, 0].set_xticklabels(nama_model_list, rotation=30, ha='right', fontsize=8)

# Menambahkan label dan judul
axes[1, 0].set_ylabel("Accuracy / Parameter (Top-1% / Juta)")
axes[1, 0].set_title("Efisiensi Model (Accuracy per Parameter)", fontweight='bold')

# Menambahkan grid
axes[1, 0].grid(axis='y', alpha=0.3)

# --- Subplot 4: Scatter plot (Params vs Accuracy) ---
# Membuat scatter plot untuk melihat hubungan params vs accuracy
for i, nama in enumerate(nama_model_list):
    # Menggambar titik scatter untuk setiap model
    axes[1, 1].scatter(params_list[i], acc_top1[i], s=200, c=warna_model[i],
                       edgecolors='black', zorder=5, label=nama)
    # Menambahkan label nama model di samping titik
    axes[1, 1].annotate(nama, (params_list[i], acc_top1[i]),
                        textcoords="offset points", xytext=(10, 5), fontsize=8)

# Menambahkan label dan judul
axes[1, 1].set_xlabel("Jumlah Parameter (Juta)")
axes[1, 1].set_ylabel("Top-1 Accuracy (%)")
axes[1, 1].set_title("Trade-off: Parameter vs Akurasi", fontweight='bold')

# Menambahkan grid
axes[1, 1].grid(alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 3: Perbandingan Arsitektur Model Pre-trained",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "03_perbandingan_model.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/03_perbandingan_model.png")

# ============================================================
# 6. Visualisasi benchmark kecepatan blob creation
# ============================================================
print("\n--- 6. Visualisasi Benchmark Kecepatan ---")

# Membuat figure untuk benchmark
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Subplot 1: Waktu blob creation per ukuran ---
nama_ukuran = [h['ukuran'] for h in hasil_benchmark]
waktu_ms = [h['waktu_ms'] for h in hasil_benchmark]

# Membuat bar chart waktu
warna_speed = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(nama_ukuran)))
bars = axes[0].bar(range(len(nama_ukuran)), waktu_ms, color=warna_speed, edgecolor='gray')

# Menambahkan label nilai di atas setiap bar
for bar, val in zip(bars, waktu_ms):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f} ms', ha='center', fontsize=9, fontweight='bold')

# Mengatur label sumbu X
axes[0].set_xticks(range(len(nama_ukuran)))
axes[0].set_xticklabels(nama_ukuran, rotation=30, ha='right', fontsize=9)

# Menambahkan label dan judul
axes[0].set_ylabel("Waktu (ms)")
axes[0].set_title("Waktu Blob Creation per Ukuran Input", fontweight='bold')

# Menambahkan grid
axes[0].grid(axis='y', alpha=0.3)

# --- Subplot 2: Memori blob per ukuran ---
memori_kb = [h['memori_kb'] for h in hasil_benchmark]

# Membuat bar chart memori
warna_mem = plt.cm.Blues(np.linspace(0.3, 0.9, len(nama_ukuran)))
bars2 = axes[1].bar(range(len(nama_ukuran)), memori_kb, color=warna_mem, edgecolor='gray')

# Menambahkan label nilai di atas setiap bar
for bar, val in zip(bars2, memori_kb):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f} KB', ha='center', fontsize=9, fontweight='bold')

# Mengatur label sumbu X
axes[1].set_xticks(range(len(nama_ukuran)))
axes[1].set_xticklabels(nama_ukuran, rotation=30, ha='right', fontsize=9)

# Menambahkan label dan judul
axes[1].set_ylabel("Memori (KB)")
axes[1].set_title("Memori Blob per Ukuran Input", fontweight='bold')

# Menambahkan grid
axes[1].grid(axis='y', alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 3: Benchmark Kecepatan dan Memori Blob Creation",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "03_benchmark_speed.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/03_benchmark_speed.png")

# ============================================================
# 7. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 3")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. Berbagai arsitektur model: MobileNet, ResNet, GoogLeNet, VGG, EfficientNet
2. Setiap model memiliki preprocessing yang berbeda (mean, scale, ukuran)
3. Trade-off antara jumlah parameter, akurasi, dan kecepatan
4. MobileNet paling ringan (3.4M params) cocok untuk perangkat mobile
5. EfficientNet memiliki efisiensi terbaik (akurasi tinggi, params sedikit)
6. Ukuran input blob mempengaruhi waktu komputasi dan penggunaan memori
7. cv2.dnn.blobFromImage() adalah fungsi kunci untuk preprocessing DNN

Output disimpan di folder: output/
- 03_perbandingan_model.png : Chart perbandingan arsitektur model
- 03_benchmark_speed.png    : Benchmark kecepatan dan memori blob
""")
