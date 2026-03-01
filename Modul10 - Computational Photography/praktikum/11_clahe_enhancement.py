"""
==========================================================================
PERCOBAAN 11: CLAHE (CONTRAST LIMITED ADAPTIVE HISTOGRAM EQUALIZATION)
==========================================================================
Program ini mempelajari teknik peningkatan kontras menggunakan CLAHE
(Contrast Limited Adaptive Histogram Equalization). Berbeda dengan
histogram equalization biasa yang bekerja secara global, CLAHE membagi
gambar menjadi tile-tile kecil dan menerapkan equalization secara lokal
dengan pembatasan kontras (clip limit) untuk menghindari amplifikasi noise.

Perbandingan metode:
1. Histogram Equalization biasa (cv2.equalizeHist) - global
2. CLAHE dengan variasi clipLimit
3. CLAHE dengan variasi tileGridSize

Fungsi utama yang dipelajari:
- cv2.createCLAHE(clipLimit, tileGridSize) : Membuat objek CLAHE
- clahe.apply(img)                         : Menerapkan CLAHE ke gambar
- cv2.equalizeHist(img)                    : Histogram equalization global
- cv2.cvtColor()                           : Konversi ruang warna (BGR↔LAB)
- cv2.calcHist()                           : Menghitung histogram

Hasil: Perbandingan visual antara HE biasa dan CLAHE dengan berbagai parameter
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi dan penyimpanan grafik
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output hasil percobaan
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 11: CLAHE ENHANCEMENT")
print("=" * 60)

# ============================================================
# 1. Membaca gambar gelap dan gambar kontras rendah
# ============================================================

# Mendefinisikan path gambar gelap
path_gelap = os.path.join(IMAGE_DIR, "gambar_gelap.png")

# Mendefinisikan path gambar kontras rendah
path_low_contrast = os.path.join(IMAGE_DIR, "low_contrast.png")

# Membaca gambar gelap dalam format BGR
img_gelap = cv2.imread(path_gelap)

# Membaca gambar kontras rendah dalam format BGR
img_low = cv2.imread(path_low_contrast)

# Memeriksa apakah gambar berhasil dimuat; jika tidak, download otomatis
if img_gelap is None or img_low is None:
    print("[WARN] Gambar tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    img_gelap = cv2.imread(path_gelap)
    img_low   = cv2.imread(path_low_contrast)
if img_gelap is None:
    raise FileNotFoundError(
        "[ERROR] gambar_gelap.png tidak tersedia setelah download.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )
if img_low is None:
    raise FileNotFoundError(
        "[ERROR] low_contrast.png tidak tersedia setelah download.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )

# Menampilkan informasi dimensi gambar
print(f"[INFO] Ukuran gambar gelap: {img_gelap.shape}")
print(f"[INFO] Ukuran gambar low contrast: {img_low.shape}")

# ============================================================
# 2. Fungsi helper untuk menerapkan CLAHE pada gambar berwarna
# ============================================================

def apply_clahe_color(image, clip_limit=2.0, tile_grid=(8, 8)):
    """Menerapkan CLAHE pada channel L dari ruang warna LAB."""
    # Mengonversi gambar dari BGR ke ruang warna LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Memisahkan channel L, A, B
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Membuat objek CLAHE dengan parameter yang diberikan
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    # Menerapkan CLAHE pada channel L (lightness)
    l_clahe = clahe.apply(l_channel)

    # Menggabungkan kembali channel L yang sudah di-CLAHE dengan A dan B
    lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])

    # Mengonversi kembali dari LAB ke BGR
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Mengembalikan hasil gambar yang sudah ditingkatkan kontrasnya
    return result

def apply_he_color(image):
    """Menerapkan Histogram Equalization biasa pada channel L dari LAB."""
    # Mengonversi gambar dari BGR ke ruang warna LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Memisahkan channel L, A, B
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Menerapkan histogram equalization global pada channel L
    l_eq = cv2.equalizeHist(l_channel)

    # Menggabungkan kembali channel yang sudah di-equalize
    lab_eq = cv2.merge([l_eq, a_channel, b_channel])

    # Mengonversi kembali dari LAB ke BGR
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Mengembalikan hasil
    return result

# ============================================================
# 3. Perbandingan HE biasa vs CLAHE pada gambar gelap
# ============================================================
print("\n[LANGKAH 1] Membandingkan HE vs CLAHE pada gambar gelap...")

# Menerapkan histogram equalization biasa pada gambar gelap
he_gelap = apply_he_color(img_gelap)

# Menerapkan CLAHE dengan clipLimit=2.0 pada gambar gelap
clahe_gelap = apply_clahe_color(img_gelap, clip_limit=2.0, tile_grid=(8, 8))

# Membuat figure untuk perbandingan HE vs CLAHE
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

# Menampilkan gambar asli (gelap)
axes1[0].imshow(cv2.cvtColor(img_gelap, cv2.COLOR_BGR2RGB))
axes1[0].set_title("Original (Gelap)", fontsize=12)
axes1[0].axis("off")

# Menampilkan hasil histogram equalization biasa
axes1[1].imshow(cv2.cvtColor(he_gelap, cv2.COLOR_BGR2RGB))
axes1[1].set_title("Histogram Equalization", fontsize=12)
axes1[1].axis("off")

# Menampilkan hasil CLAHE
axes1[2].imshow(cv2.cvtColor(clahe_gelap, cv2.COLOR_BGR2RGB))
axes1[2].set_title("CLAHE (clip=2.0, tile=8x8)", fontsize=12)
axes1[2].axis("off")

# Menambahkan judul utama
fig1.suptitle("Perbandingan HE vs CLAHE pada Gambar Gelap", fontsize=14, fontweight='bold')

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path1 = os.path.join(OUTPUT_DIR, "11_he_vs_clahe_gelap.png")
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path1}")

# Menutup figure untuk membebaskan memori
plt.close(fig1)

# ============================================================
# 4. Variasi clipLimit pada CLAHE
# ============================================================
print("\n[LANGKAH 2] Variasi clipLimit pada CLAHE...")

# Mendefinisikan daftar nilai clipLimit yang akan diuji
clip_limits = [1.0, 2.0, 4.0, 8.0, 16.0, 40.0]

# Membuat figure untuk variasi clipLimit
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

# Melakukan iterasi untuk setiap nilai clipLimit
for idx, cl in enumerate(clip_limits):
    # Menghitung posisi baris dan kolom pada grid subplot
    row = idx // 3
    col = idx % 3

    # Menerapkan CLAHE dengan clipLimit tertentu
    result = apply_clahe_color(img_gelap, clip_limit=cl, tile_grid=(8, 8))

    # Menampilkan hasil pada subplot yang sesuai
    axes2[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes2[row, col].set_title(f"clipLimit = {cl}", fontsize=11)
    axes2[row, col].axis("off")

# Menambahkan judul utama
fig2.suptitle("Variasi clipLimit pada CLAHE (tileGrid=8x8)", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path2 = os.path.join(OUTPUT_DIR, "11_variasi_cliplimit.png")
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path2}")

# Menutup figure
plt.close(fig2)

# ============================================================
# 5. Variasi tileGridSize pada CLAHE
# ============================================================
print("\n[LANGKAH 3] Variasi tileGridSize pada CLAHE...")

# Mendefinisikan daftar ukuran tile yang akan diuji
tile_sizes = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]

# Membuat figure untuk variasi tileGridSize
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))

# Melakukan iterasi untuk setiap ukuran tile
for idx, ts in enumerate(tile_sizes):
    # Menghitung posisi baris dan kolom pada grid subplot
    row = idx // 3
    col = idx % 3

    # Menerapkan CLAHE dengan ukuran tile tertentu
    result = apply_clahe_color(img_gelap, clip_limit=2.0, tile_grid=ts)

    # Menampilkan hasil pada subplot yang sesuai
    axes3[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes3[row, col].set_title(f"tileGrid = {ts[0]}x{ts[1]}", fontsize=11)
    axes3[row, col].axis("off")

# Menambahkan judul utama
fig3.suptitle("Variasi tileGridSize pada CLAHE (clipLimit=2.0)", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path3 = os.path.join(OUTPUT_DIR, "11_variasi_tilegrid.png")
fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path3}")

# Menutup figure
plt.close(fig3)

# ============================================================
# 6. CLAHE pada gambar low_contrast
# ============================================================
print("\n[LANGKAH 4] CLAHE pada gambar low contrast...")

# Menerapkan HE biasa pada gambar kontras rendah
he_low = apply_he_color(img_low)

# Menerapkan CLAHE pada gambar kontras rendah
clahe_low = apply_clahe_color(img_low, clip_limit=3.0, tile_grid=(8, 8))

# Membuat figure untuk perbandingan pada gambar kontras rendah
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

# Menampilkan gambar asli (kontras rendah)
axes4[0].imshow(cv2.cvtColor(img_low, cv2.COLOR_BGR2RGB))
axes4[0].set_title("Original (Low Contrast)", fontsize=12)
axes4[0].axis("off")

# Menampilkan hasil HE biasa
axes4[1].imshow(cv2.cvtColor(he_low, cv2.COLOR_BGR2RGB))
axes4[1].set_title("Histogram Equalization", fontsize=12)
axes4[1].axis("off")

# Menampilkan hasil CLAHE
axes4[2].imshow(cv2.cvtColor(clahe_low, cv2.COLOR_BGR2RGB))
axes4[2].set_title("CLAHE (clip=3.0, tile=8x8)", fontsize=12)
axes4[2].axis("off")

# Menambahkan judul utama
fig4.suptitle("Perbandingan HE vs CLAHE pada Gambar Low Contrast", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path4 = os.path.join(OUTPUT_DIR, "11_clahe_low_contrast.png")
fig4.savefig(output_path4, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path4}")

# Menutup figure
plt.close(fig4)

# ============================================================
# 7. Perbandingan histogram sebelum dan sesudah CLAHE
# ============================================================
print("\n[LANGKAH 5] Membandingkan histogram sebelum dan sesudah...")

# Mengonversi gambar gelap asli ke grayscale untuk histogram
gray_orig = cv2.cvtColor(img_gelap, cv2.COLOR_BGR2GRAY)

# Menerapkan HE biasa pada grayscale
gray_he = cv2.equalizeHist(gray_orig)

# Membuat objek CLAHE untuk grayscale
clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Menerapkan CLAHE pada grayscale
gray_clahe = clahe_obj.apply(gray_orig)

# Membuat figure untuk histogram
fig5, axes5 = plt.subplots(2, 3, figsize=(15, 8))

# Menampilkan gambar grayscale original
axes5[0, 0].imshow(gray_orig, cmap='gray')
axes5[0, 0].set_title("Original (Grayscale)", fontsize=11)
axes5[0, 0].axis("off")

# Menampilkan gambar setelah HE
axes5[0, 1].imshow(gray_he, cmap='gray')
axes5[0, 1].set_title("Histogram Equalization", fontsize=11)
axes5[0, 1].axis("off")

# Menampilkan gambar setelah CLAHE
axes5[0, 2].imshow(gray_clahe, cmap='gray')
axes5[0, 2].set_title("CLAHE", fontsize=11)
axes5[0, 2].axis("off")

# Menghitung histogram untuk gambar original
hist_orig = cv2.calcHist([gray_orig], [0], None, [256], [0, 256])

# Menghitung histogram untuk gambar HE
hist_he = cv2.calcHist([gray_he], [0], None, [256], [0, 256])

# Menghitung histogram untuk gambar CLAHE
hist_clahe = cv2.calcHist([gray_clahe], [0], None, [256], [0, 256])

# Menampilkan histogram original
axes5[1, 0].plot(hist_orig, color='black')
axes5[1, 0].set_title("Histogram Original", fontsize=11)
axes5[1, 0].set_xlim([0, 256])

# Menampilkan histogram HE
axes5[1, 1].plot(hist_he, color='blue')
axes5[1, 1].set_title("Histogram HE", fontsize=11)
axes5[1, 1].set_xlim([0, 256])

# Menampilkan histogram CLAHE
axes5[1, 2].plot(hist_clahe, color='green')
axes5[1, 2].set_title("Histogram CLAHE", fontsize=11)
axes5[1, 2].set_xlim([0, 256])

# Menambahkan judul utama
fig5.suptitle("Perbandingan Histogram: Original vs HE vs CLAHE", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path5 = os.path.join(OUTPUT_DIR, "11_histogram_comparison.png")
fig5.savefig(output_path5, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path5}")

# Menutup figure
plt.close(fig5)

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 11: CLAHE ENHANCEMENT")
print("=" * 60)
print("Fungsi-fungsi OpenCV yang dipelajari:")
print("1. cv2.createCLAHE(clipLimit, tileGridSize)")
print("   → Membuat objek CLAHE untuk adaptive histogram equalization")
print("2. clahe.apply(img)")
print("   → Menerapkan CLAHE pada gambar grayscale atau single channel")
print("3. cv2.equalizeHist(img)")
print("   → Histogram equalization global (pembanding)")
print("4. cv2.cvtColor(img, cv2.COLOR_BGR2LAB)")
print("   → Konversi ke ruang warna LAB untuk CLAHE pada channel L")
print("5. cv2.calcHist([img], [0], None, [256], [0, 256])")
print("   → Menghitung histogram untuk analisis distribusi intensitas")
print("6. cv2.split() dan cv2.merge()")
print("   → Memisahkan dan menggabungkan channel warna")
print()
print("Kesimpulan:")
print("- CLAHE lebih baik dari HE biasa karena bekerja secara lokal")
print("- clipLimit mengontrol batas amplifikasi kontras")
print("- tileGridSize mengontrol ukuran region lokal")
print("- Nilai clipLimit terlalu tinggi → noise teramplifikasi")
print("- tileGridSize terlalu kecil → artefak blok terlihat")
print("=" * 60)
