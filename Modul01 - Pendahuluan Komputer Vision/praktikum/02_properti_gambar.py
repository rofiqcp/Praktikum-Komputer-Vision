"""
==========================================================================
PERCOBAAN 2: PROPERTI GAMBAR
==========================================================================
Program ini mempelajari cara mengakses dan memahami properti-properti
dasar sebuah gambar digital: dimensi, tipe data, jumlah piksel, memory,
jumlah channel, dan format data.

Fungsi utama yang dipelajari:
- img.shape      : Mendapatkan dimensi gambar (height, width, channels)
- img.dtype      : Tipe data piksel (uint8, float32, dll)
- img.size       : Total jumlah elemen (piksel × channel)
- img.nbytes     : Ukuran memori yang digunakan
- img.ndim       : Jumlah dimensi array
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor library NumPy untuk operasi array
import numpy as np

# Mengimpor library os untuk operasi path
import os

# Mengimpor matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 2: PROPERTI GAMBAR")
print("=" * 60)

# ============================================================
# 1. Membaca gambar berwarna dan grayscale
# ============================================================

# Membaca gambar kucing dalam mode warna (3 channel BGR)
img_color = cv2.imread(os.path.join(IMAGE_DIR, "foto_kucing.jpg"), cv2.IMREAD_COLOR)

# Membaca gambar yang sama dalam mode grayscale (1 channel)
img_gray = cv2.imread(os.path.join(IMAGE_DIR, "foto_kucing.jpg"), cv2.IMREAD_GRAYSCALE)

# Memeriksa apakah gambar berhasil dimuat
if img_color is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

# ============================================================
# 2. Mengakses properti gambar berwarna
# ============================================================
print("\n--- Properti Gambar Berwarna ---")

# img.shape mengembalikan tuple (height, width, channels)
# height = jumlah baris piksel (tinggi)
# width = jumlah kolom piksel (lebar)
# channels = jumlah channel warna (3 untuk BGR)
tinggi, lebar, channels = img_color.shape
print(f"  Dimensi (shape)      : {img_color.shape}")
print(f"  Tinggi (height)      : {tinggi} piksel")
print(f"  Lebar (width)        : {lebar} piksel")
print(f"  Jumlah channel       : {channels} (B, G, R)")

# img.dtype menunjukkan tipe data setiap piksel
# uint8 = unsigned integer 8-bit, range 0-255
print(f"  Tipe data (dtype)    : {img_color.dtype}")

# img.size mengembalikan total elemen = height × width × channels
print(f"  Total elemen (size)  : {img_color.size}")

# img.nbytes mengembalikan ukuran memori dalam bytes
print(f"  Ukuran memori        : {img_color.nbytes} bytes ({img_color.nbytes/1024:.1f} KB)")

# img.ndim mengembalikan jumlah dimensi array
# Gambar berwarna = 3 dimensi (height, width, channels)
print(f"  Jumlah dimensi       : {img_color.ndim}")

# ============================================================
# 3. Mengakses properti gambar grayscale
# ============================================================
print("\n--- Properti Gambar Grayscale ---")

# Gambar grayscale hanya memiliki 2 dimensi (height, width)
print(f"  Dimensi (shape)      : {img_gray.shape}")
print(f"  Tinggi (height)      : {img_gray.shape[0]} piksel")
print(f"  Lebar (width)        : {img_gray.shape[1]} piksel")
print(f"  Tipe data (dtype)    : {img_gray.dtype}")
print(f"  Total elemen (size)  : {img_gray.size}")
print(f"  Ukuran memori        : {img_gray.nbytes} bytes ({img_gray.nbytes/1024:.1f} KB)")
print(f"  Jumlah dimensi       : {img_gray.ndim}")

# ============================================================
# 4. Statistik nilai piksel
# ============================================================
print("\n--- Statistik Nilai Piksel (Gambar Berwarna) ---")

# np.min() dan np.max() mencari nilai minimum dan maksimum
print(f"  Nilai minimum        : {np.min(img_color)}")
print(f"  Nilai maksimum       : {np.max(img_color)}")

# np.mean() menghitung rata-rata nilai piksel
print(f"  Rata-rata            : {np.mean(img_color):.2f}")

# np.std() menghitung standar deviasi nilai piksel
print(f"  Standar deviasi      : {np.std(img_color):.2f}")

# Statistik per channel (B, G, R)
print("\n--- Statistik Per Channel ---")
for i, nama_ch in enumerate(["Blue", "Green", "Red"]):
    # Mengakses channel ke-i menggunakan img[:,:,i]
    ch = img_color[:, :, i]
    print(f"  {nama_ch:6s}: min={np.min(ch):3d}, max={np.max(ch):3d}, "
          f"mean={np.mean(ch):.1f}, std={np.std(ch):.1f}")

# ============================================================
# 5. Perbandingan ukuran memori berbagai tipe data
# ============================================================
print("\n--- Perbandingan Tipe Data ---")

# Mengkonversi gambar ke float32 (range 0.0 - 255.0)
img_float32 = img_color.astype(np.float32)
print(f"  uint8  : {img_color.nbytes/1024:.1f} KB  (range 0-255)")
print(f"  float32: {img_float32.nbytes/1024:.1f} KB (range 0.0-255.0)")

# Mengkonversi gambar ke float64
img_float64 = img_color.astype(np.float64)
print(f"  float64: {img_float64.nbytes/1024:.1f} KB (range 0.0-255.0)")

# Normalisasi ke range 0.0 - 1.0 (umum untuk deep learning)
img_normalized = img_color.astype(np.float32) / 255.0
print(f"  float32 normalized: min={img_normalized.min():.2f}, max={img_normalized.max():.2f}")

# ============================================================
# 6. Resolusi dan Aspect Ratio
# ============================================================
print("\n--- Resolusi dan Aspect Ratio ---")

# Menghitung total piksel (resolusi)
total_piksel = tinggi * lebar
print(f"  Total piksel         : {total_piksel:,} ({total_piksel/1e6:.2f} Megapiksel)")

# Menghitung aspect ratio
from math import gcd
# gcd = greatest common divisor (faktor persekutuan terbesar)
pembagi = gcd(lebar, tinggi)
print(f"  Aspect ratio         : {lebar//pembagi}:{tinggi//pembagi}")

# ============================================================
# 7. Visualisasi ringkasan properti
# ============================================================

# Membuat figure untuk menampilkan gambar beserta propertinya
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Gambar berwarna dengan info properti
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
axes[0].imshow(img_rgb)
axes[0].set_title(f"Berwarna: {lebar}×{tinggi}×{channels}\n"
                  f"dtype={img_color.dtype}, {img_color.nbytes/1024:.0f}KB")
axes[0].axis("off")

# Subplot 2: Gambar grayscale dengan info properti
axes[1].imshow(img_gray, cmap="gray")
axes[1].set_title(f"Grayscale: {img_gray.shape[1]}×{img_gray.shape[0]}\n"
                  f"dtype={img_gray.dtype}, {img_gray.nbytes/1024:.0f}KB")
axes[1].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 2: Properti Gambar Digital", fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path = os.path.join(OUTPUT_DIR, "02_properti_gambar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 2")
print("=" * 60)
print("Properti penting gambar digital:")
print("  1. shape    → Dimensi (h, w, ch) atau (h, w) untuk grayscale")
print("  2. dtype    → Tipe data piksel (uint8, float32, float64)")
print("  3. size     → Total elemen (h × w × ch)")
print("  4. nbytes   → Ukuran memori dalam bytes")
print("  5. ndim     → Jumlah dimensi (2=grayscale, 3=berwarna)")
print("  6. min/max  → Range nilai piksel")
print("  7. mean/std → Statistik distribusi piksel")
print("=" * 60)
