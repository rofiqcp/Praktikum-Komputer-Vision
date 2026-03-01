"""
==========================================================================
PERCOBAAN 5: DENOISING DENGAN GAUSSIAN BLUR
==========================================================================
Program ini mempelajari teknik denoising (pengurangan noise) menggunakan
Gaussian Blur. Gaussian Blur menggunakan kernel Gaussian untuk merata-
ratakan piksel, sehingga noise berkurang tetapi gambar menjadi lebih blur.

Konsep penting:
- Kernel size: Semakin besar = semakin halus, tapi semakin blur
- Sigma (standar deviasi): Mengontrol lebar distribusi Gaussian
- PSNR (Peak Signal-to-Noise Ratio): Metrik kualitas denoising

Fungsi utama yang dipelajari:
- cv2.GaussianBlur(src, ksize, sigmaX)
- cv2.PSNR(src1, src2) — menghitung rasio sinyal terhadap noise
- Perbandingan kernel size dan sigma terhadap kualitas

Hasil: Perbandingan visual dan PSNR berbagai parameter Gaussian blur
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk manajemen path file
import os

# Mengimpor matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 5: DENOISING DENGAN GAUSSIAN BLUR")
print("=" * 60)

# ============================================================
# 1. Membaca gambar noisy dan ground truth
# ============================================================

# Membaca gambar noisy (mengandung noise Gaussian)
noisy_path = os.path.join(IMAGE_DIR, "noisy_gaussian.png")
noisy_img = cv2.imread(noisy_path)

# Validasi pembacaan gambar noisy
if noisy_img is None:
    print(f"[ERROR] Gagal membaca: {noisy_path}")
    print("[INFO] Jalankan download_image.py terlebih dahulu!")
    exit()

# Membaca gambar asli (ground truth) untuk perhitungan PSNR
gt_path = os.path.join(IMAGE_DIR, "scene_pemandangan.png")
gt_img = cv2.imread(gt_path)

# Validasi pembacaan ground truth
if gt_img is None:
    print(f"[ERROR] Gagal membaca: {gt_path}")
    exit()

# Menyamakan ukuran jika berbeda
if noisy_img.shape != gt_img.shape:
    # Resize ground truth agar sesuai dengan noisy image
    gt_img = cv2.resize(gt_img, (noisy_img.shape[1], noisy_img.shape[0]))

# Menampilkan info gambar
print(f"[INFO] Gambar noisy  : {noisy_img.shape[1]}x{noisy_img.shape[0]}")
print(f"[INFO] Ground truth  : {gt_img.shape[1]}x{gt_img.shape[0]}")

# Menghitung PSNR gambar noisy terhadap ground truth (sebelum denoising)
psnr_noisy = cv2.PSNR(gt_img, noisy_img)
print(f"[INFO] PSNR gambar noisy (sebelum denoising): {psnr_noisy:.2f} dB")

# ============================================================
# 2. Gaussian Blur dengan variasi kernel size
# cv2.GaussianBlur(src, (ksize, ksize), sigmaX)
# - ksize harus ganjil (3, 5, 7, 9, 11, ...)
# - sigmaX = 0 berarti dihitung otomatis dari ksize
# ============================================================

# Daftar kernel size yang akan diuji
kernel_sizes = [3, 5, 7, 9, 11, 15, 21, 31]

# Menyimpan hasil denoising dan PSNR untuk setiap kernel
kernel_results = []
kernel_psnr = []

print("\n--- Variasi Kernel Size (sigma=0, otomatis) ---")
for ksize in kernel_sizes:
    # Menerapkan Gaussian blur dengan kernel tertentu
    # sigmaX=0 membuat OpenCV menghitung sigma dari ksize
    denoised = cv2.GaussianBlur(noisy_img, (ksize, ksize), sigmaX=0)

    # Menghitung PSNR hasil denoising terhadap ground truth
    psnr_val = cv2.PSNR(gt_img, denoised)

    # Menyimpan hasil
    kernel_results.append(denoised)
    kernel_psnr.append(psnr_val)

    # Menampilkan info
    print(f"  Kernel {ksize:2d}x{ksize:2d} — PSNR: {psnr_val:.2f} dB")

# Menemukan kernel size terbaik (PSNR tertinggi)
best_kernel_idx = np.argmax(kernel_psnr)
best_ksize = kernel_sizes[best_kernel_idx]
best_psnr_kernel = kernel_psnr[best_kernel_idx]
print(f"\n[INFO] Kernel terbaik: {best_ksize}x{best_ksize} "
      f"(PSNR: {best_psnr_kernel:.2f} dB)")

# ============================================================
# 3. Gaussian Blur dengan variasi sigma (kernel tetap)
# ============================================================

# Menggunakan kernel size tetap (yang terbaik dari percobaan di atas)
fixed_ksize = best_ksize

# Daftar sigma yang akan diuji
sigma_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

# Menyimpan hasil untuk variasi sigma
sigma_results = []
sigma_psnr = []

print(f"\n--- Variasi Sigma (kernel tetap={fixed_ksize}x{fixed_ksize}) ---")
for sigma in sigma_values:
    # Menerapkan Gaussian blur dengan sigma tertentu
    denoised = cv2.GaussianBlur(noisy_img, (fixed_ksize, fixed_ksize), sigmaX=sigma)

    # Menghitung PSNR
    psnr_val = cv2.PSNR(gt_img, denoised)

    # Menyimpan hasil
    sigma_results.append(denoised)
    sigma_psnr.append(psnr_val)

    # Menampilkan info
    print(f"  Sigma {sigma:5.1f} — PSNR: {psnr_val:.2f} dB")

# Menemukan sigma terbaik
best_sigma_idx = np.argmax(sigma_psnr)
best_sigma = sigma_values[best_sigma_idx]
best_psnr_sigma = sigma_psnr[best_sigma_idx]
print(f"\n[INFO] Sigma terbaik: {best_sigma} "
      f"(PSNR: {best_psnr_sigma:.2f} dB)")

# ============================================================
# 4. Visualisasi perbandingan
# ============================================================

# Membuat figure 3 baris x 4 kolom
fig, axes = plt.subplots(3, 4, figsize=(22, 15))

# --- Baris 1: Variasi kernel size (4 contoh terpilih) ---
# Memilih 4 kernel dari daftar untuk ditampilkan
display_indices = [0, 2, 4, 7]  # kernel 3, 7, 11, 31

for i, idx in enumerate(display_indices):
    # Konversi BGR ke RGB
    rgb = cv2.cvtColor(kernel_results[idx], cv2.COLOR_BGR2RGB)
    axes[0, i].imshow(rgb)
    axes[0, i].set_title(f"Kernel {kernel_sizes[idx]}x{kernel_sizes[idx]}\n"
                         f"PSNR: {kernel_psnr[idx]:.2f} dB", fontsize=9)
    axes[0, i].axis("off")

# --- Baris 2: Variasi sigma (4 contoh terpilih) ---
display_sigma_indices = [0, 2, 4, 7]  # sigma 0.5, 1.5, 3.0, 10.0

for i, idx in enumerate(display_sigma_indices):
    rgb = cv2.cvtColor(sigma_results[idx], cv2.COLOR_BGR2RGB)
    axes[1, i].imshow(rgb)
    axes[1, i].set_title(f"Sigma {sigma_values[idx]}\n"
                         f"PSNR: {sigma_psnr[idx]:.2f} dB", fontsize=9)
    axes[1, i].axis("off")

# --- Baris 3: Perbandingan dan grafik PSNR ---
# Original (ground truth)
gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
axes[2, 0].imshow(gt_rgb)
axes[2, 0].set_title("Ground Truth\n(Gambar Asli)", fontsize=9)
axes[2, 0].axis("off")

# Noisy
noisy_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
axes[2, 1].imshow(noisy_rgb)
axes[2, 1].set_title(f"Noisy (Gaussian)\nPSNR: {psnr_noisy:.2f} dB", fontsize=9)
axes[2, 1].axis("off")

# Grafik PSNR vs Kernel Size
axes[2, 2].plot(kernel_sizes, kernel_psnr, 'bo-', linewidth=2, markersize=6)
axes[2, 2].axhline(y=psnr_noisy, color='r', linestyle='--',
                    label=f'Noisy: {psnr_noisy:.1f} dB')
axes[2, 2].set_title("PSNR vs Kernel Size", fontsize=9)
axes[2, 2].set_xlabel("Kernel Size")
axes[2, 2].set_ylabel("PSNR (dB)")
axes[2, 2].legend(fontsize=8)
axes[2, 2].grid(True, alpha=0.3)

# Grafik PSNR vs Sigma
axes[2, 3].plot(sigma_values, sigma_psnr, 'go-', linewidth=2, markersize=6)
axes[2, 3].axhline(y=psnr_noisy, color='r', linestyle='--',
                    label=f'Noisy: {psnr_noisy:.1f} dB')
axes[2, 3].set_title(f"PSNR vs Sigma (k={fixed_ksize})", fontsize=9)
axes[2, 3].set_xlabel("Sigma")
axes[2, 3].set_ylabel("PSNR (dB)")
axes[2, 3].legend(fontsize=8)
axes[2, 3].grid(True, alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 5: Denoising dengan Gaussian Blur\n"
             "Pengaruh Kernel Size dan Sigma terhadap Kualitas",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi
output_path = os.path.join(OUTPUT_DIR, "05_denoising_gaussian_blur.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 5")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.GaussianBlur(src, (ksize, ksize), sigmaX)")
print("     - ksize: ukuran kernel (harus ganjil)")
print("     - sigmaX: standar deviasi Gaussian")
print("     - sigmaX=0 → sigma dihitung otomatis dari ksize")
print("  2. cv2.PSNR(src1, src2) — Peak Signal-to-Noise Ratio")
print("     - Semakin tinggi PSNR = kualitas semakin baik")
print("     - Umumnya > 30 dB dianggap bagus")
print(f"  3. PSNR noisy (sebelum denoising): {psnr_noisy:.2f} dB")
print(f"  4. PSNR terbaik (kernel): {best_psnr_kernel:.2f} dB "
      f"(kernel {best_ksize}x{best_ksize})")
print(f"  5. PSNR terbaik (sigma): {best_psnr_sigma:.2f} dB "
      f"(sigma={best_sigma})")
print("  6. Trade-off: Kernel besar → noise berkurang TAPI detail hilang")
print("=" * 60)
