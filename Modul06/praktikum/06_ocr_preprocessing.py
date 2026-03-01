"""
==========================================================================
PERCOBAAN 6: OCR PREPROCESSING TECHNIQUES
==========================================================================
Program ini mempelajari teknik-teknik preprocessing gambar teks untuk
meningkatkan kualitas input OCR (Optical Character Recognition). Tahap
preprocessing sangat penting karena kualitas input menentukan akurasi
pengenalan karakter.

Konsep yang dipelajari:
- Pipeline OCR: preprocessing → segmentasi → recognition
- Binarisasi: Otsu thresholding, adaptive thresholding
- Denoising: Gaussian blur, median blur, operasi morfologi
- Deskewing: mendeteksi dan memperbaiki kemiringan teks
- Pengaruh preprocessing terhadap kualitas input OCR

Fungsi utama yang dipelajari:
- cv2.threshold() dengan THRESH_OTSU     : Binarisasi Otsu otomatis
- cv2.adaptiveThreshold()               : Binarisasi adaptif
- cv2.GaussianBlur() / cv2.medianBlur() : Denoising
- cv2.morphologyEx()                    : Operasi morfologi (open/close)
- cv2.HoughLinesP()                     : Deteksi garis untuk deskewing
- cv2.getRotationMatrix2D()             : Rotasi untuk memperbaiki skew

Hasil: Visualisasi perbandingan binarisasi, denoising, dan deskewing
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor math untuk fungsi matematika
import math

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 6: OCR PREPROCESSING TECHNIQUES")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Teks untuk OCR
# ============================================================

print("\n[INFO] Memuat gambar teks...")
print("-" * 50)

# Membaca gambar teks cetak (printed text)
img_printed = cv2.imread(os.path.join(IMAGE_DIR, "teks_printed.jpg"))

# Membaca gambar teks pada scene (sign, plat nomor, dll.)
img_scene = cv2.imread(os.path.join(IMAGE_DIR, "teks_scene.jpg"))

# Membaca gambar teks dengan noise
img_noisy = cv2.imread(os.path.join(IMAGE_DIR, "teks_noisy.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_printed is None:
    print("[ERROR] Gambar teks tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi gambar
print(f"  teks_printed: {img_printed.shape}")
print(f"  teks_scene  : {img_scene.shape}")
print(f"  teks_noisy  : {img_noisy.shape}")

# ============================================================
# 2. Perbandingan Teknik Binarisasi
# ============================================================

print("\n[INFO] Membandingkan teknik binarisasi...")
print("-" * 50)

# Mendefinisikan daftar gambar yang akan diproses
gambar_teks = [
    ("Teks Cetak", img_printed),
    ("Teks Scene", img_scene),
    ("Teks Noisy", img_noisy),
]

# Membuat figure untuk perbandingan binarisasi
fig, axes = plt.subplots(3, 5, figsize=(22, 12))

# Memproses setiap gambar teks
for row, (nama, img) in enumerate(gambar_teks):
    # Mengkonversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Kolom 1: Gambar asli ---
    axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[row, 0].set_title(f"{nama}\nAsli", fontsize=10)
    axes[row, 0].axis("off")

    # --- Kolom 2: Grayscale ---
    axes[row, 1].imshow(gray, cmap='gray')
    axes[row, 1].set_title("Grayscale", fontsize=10)
    axes[row, 1].axis("off")

    # --- Kolom 3: Binary Threshold biasa (threshold=127) ---
    # Menerapkan threshold biasa: piksel > 127 → 255, selain itu → 0
    _, binary_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Menampilkan hasil threshold biasa
    axes[row, 2].imshow(binary_simple, cmap='gray')
    axes[row, 2].set_title("Binary (T=127)", fontsize=10)
    axes[row, 2].axis("off")

    # --- Kolom 4: Otsu Thresholding ---
    # Menerapkan Otsu thresholding (menentukan threshold optimal otomatis)
    # Otsu mencari threshold yang memaksimalkan varians antar kelas
    otsu_val, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Menampilkan hasil Otsu thresholding
    axes[row, 3].imshow(binary_otsu, cmap='gray')
    axes[row, 3].set_title(f"Otsu (T={otsu_val:.0f})", fontsize=10)
    axes[row, 3].axis("off")

    # Menampilkan nilai threshold Otsu
    print(f"  {nama}: Otsu threshold = {otsu_val:.0f}")

    # --- Kolom 5: Adaptive Thresholding ---
    # Menerapkan adaptive thresholding
    # Threshold dihitung per area kecil (local), bukan global
    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,    # Metode: rata-rata tertimbang Gaussian
        cv2.THRESH_BINARY,                  # Tipe: binary
        blockSize=11,                        # Ukuran area lokal (harus ganjil)
        C=2                                  # Konstanta yang dikurangi dari mean
    )

    # Menampilkan hasil adaptive thresholding
    axes[row, 4].imshow(binary_adaptive, cmap='gray')
    axes[row, 4].set_title("Adaptive (b=11, C=2)", fontsize=10)
    axes[row, 4].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 6: Perbandingan Teknik Binarisasi untuk OCR\n"
             "Binary | Otsu (otomatis) | Adaptive (lokal)",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil perbandingan binarisasi
output_path_1 = os.path.join(OUTPUT_DIR, "06_binarisasi_perbandingan.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 3. Teknik Denoising untuk OCR
# ============================================================

print("\n[INFO] Menerapkan teknik denoising...")
print("-" * 50)

# Menggunakan gambar noisy sebagai contoh utama
gray_noisy = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)

# Membuat figure untuk perbandingan denoising
fig, axes = plt.subplots(3, 4, figsize=(18, 12))

# --- Baris 1: Berbagai teknik denoising ---

# Kolom 1: Gambar noisy asli (grayscale)
axes[0, 0].imshow(gray_noisy, cmap='gray')
axes[0, 0].set_title("Noisy (Asli)", fontsize=10)
axes[0, 0].axis("off")

# Kolom 2: Gaussian Blur
# Menerapkan Gaussian blur untuk mengurangi noise
# kernel_size=(5,5): ukuran filter, sigmaX=0: sigma otomatis
gaussian_denoised = cv2.GaussianBlur(gray_noisy, (5, 5), 0)
axes[0, 1].imshow(gaussian_denoised, cmap='gray')
axes[0, 1].set_title("Gaussian Blur (5x5)", fontsize=10)
axes[0, 1].axis("off")

# Kolom 3: Median Blur
# Menerapkan median blur (efektif untuk salt-and-pepper noise)
# ksize=5: ukuran kernel
median_denoised = cv2.medianBlur(gray_noisy, 5)
axes[0, 2].imshow(median_denoised, cmap='gray')
axes[0, 2].set_title("Median Blur (5)", fontsize=10)
axes[0, 2].axis("off")

# Kolom 4: Bilateral Filter
# Menerapkan bilateral filter (mengurangi noise sambil menjaga tepi)
bilateral_denoised = cv2.bilateralFilter(gray_noisy, 9, 75, 75)
axes[0, 3].imshow(bilateral_denoised, cmap='gray')
axes[0, 3].set_title("Bilateral Filter (d=9)", fontsize=10)
axes[0, 3].axis("off")

# --- Baris 2: Denoising lanjut + Binarisasi ---

# Kolom 1: Otsu pada gambar noisy (tanpa denoising)
_, noisy_otsu = cv2.threshold(gray_noisy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
axes[1, 0].imshow(noisy_otsu, cmap='gray')
axes[1, 0].set_title("Noisy → Otsu\n(tanpa denoising)", fontsize=10)
axes[1, 0].axis("off")

# Kolom 2: Gaussian + Otsu
_, gaussian_otsu = cv2.threshold(gaussian_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
axes[1, 1].imshow(gaussian_otsu, cmap='gray')
axes[1, 1].set_title("Gaussian → Otsu", fontsize=10)
axes[1, 1].axis("off")

# Kolom 3: Median + Otsu
_, median_otsu = cv2.threshold(median_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
axes[1, 2].imshow(median_otsu, cmap='gray')
axes[1, 2].set_title("Median → Otsu", fontsize=10)
axes[1, 2].axis("off")

# Kolom 4: Bilateral + Otsu
_, bilateral_otsu = cv2.threshold(bilateral_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
axes[1, 3].imshow(bilateral_otsu, cmap='gray')
axes[1, 3].set_title("Bilateral → Otsu", fontsize=10)
axes[1, 3].axis("off")

# --- Baris 3: Operasi Morfologi untuk pembersihan ---

# Kolom 1: Morphological Opening (menghilangkan noise kecil)
kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph_open = cv2.morphologyEx(median_otsu, cv2.MORPH_OPEN, kernel_morph)
axes[2, 0].imshow(morph_open, cmap='gray')
axes[2, 0].set_title("Morph Opening (3x3)", fontsize=10)
axes[2, 0].axis("off")

# Kolom 2: Morphological Closing (menutup lubang kecil)
morph_close = cv2.morphologyEx(median_otsu, cv2.MORPH_CLOSE, kernel_morph)
axes[2, 1].imshow(morph_close, cmap='gray')
axes[2, 1].set_title("Morph Closing (3x3)", fontsize=10)
axes[2, 1].axis("off")

# Kolom 3: Opening + Closing (kombinasi)
morph_combined = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel_morph)
axes[2, 2].imshow(morph_combined, cmap='gray')
axes[2, 2].set_title("Opening + Closing", fontsize=10)
axes[2, 2].axis("off")

# Kolom 4: Dilasi (menebalkan teks)
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morph_dilate = cv2.dilate(morph_combined, kernel_dilate, iterations=1)
axes[2, 3].imshow(morph_dilate, cmap='gray')
axes[2, 3].set_title("Dilasi (teks tebal)", fontsize=10)
axes[2, 3].axis("off")

# Menampilkan informasi denoising
print("  Teknik denoising yang diterapkan:")
print("  1. Gaussian Blur: menghaluskan dengan kernel Gaussian")
print("  2. Median Blur: efektif untuk salt-and-pepper noise")
print("  3. Bilateral Filter: noise reduction + edge preservation")
print("  4. Morphological Opening: menghilangkan objek kecil (noise)")
print("  5. Morphological Closing: menutup lubang pada objek")

# Menambahkan judul utama
plt.suptitle("Percobaan 6: Denoising dan Morfologi untuk OCR\n"
             "Baris 1: Denoising | Baris 2: +Binarisasi | Baris 3: Morfologi",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil denoising
output_path_2 = os.path.join(OUTPUT_DIR, "06_denoising.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 4. Deskewing (Memperbaiki Kemiringan Teks)
# ============================================================

print("\n[INFO] Melakukan deskewing pada gambar teks...")
print("-" * 50)

# Mendefinisikan fungsi untuk membuat gambar teks miring (untuk demonstrasi)
def buat_gambar_miring(img, sudut):
    """Membuat gambar teks yang diputar/miring sebesar sudut tertentu."""
    # Mendapatkan ukuran gambar
    h, w = img.shape[:2]

    # Menghitung titik pusat gambar
    pusat = (w // 2, h // 2)

    # Membuat matriks rotasi
    M = cv2.getRotationMatrix2D(pusat, sudut, 1.0)

    # Menerapkan rotasi
    img_miring = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    # Mengembalikan gambar yang sudah dimiringkan
    return img_miring

# Mendefinisikan fungsi untuk mendeteksi sudut kemiringan
def deteksi_sudut_skew(img_binary):
    """
    Mendeteksi sudut kemiringan teks menggunakan Hough Line Transform.
    Return: sudut kemiringan dalam derajat.
    """
    # Deteksi tepi menggunakan Canny
    edges = cv2.Canny(img_binary, 50, 150, apertureSize=3)

    # Mendeteksi garis menggunakan Probabilistic Hough Transform
    # minLineLength: panjang minimum garis
    # maxLineGap: jarak maksimum antar segmen garis
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=50, maxLineGap=10)

    # Menghitung sudut rata-rata dari garis yang terdeteksi
    if lines is not None:
        angles = []
        for line in lines:
            # Mengekstrak titik awal dan akhir garis
            x1, y1, x2, y2 = line[0]

            # Menghitung sudut garis terhadap horizontal
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            # Memfilter hanya garis yang hampir horizontal (-45° sampai 45°)
            if abs(angle) < 45:
                angles.append(angle)

        # Menghitung sudut median (lebih robust daripada mean)
        if len(angles) > 0:
            sudut_skew = np.median(angles)
            return sudut_skew

    # Mengembalikan 0 jika tidak ada garis terdeteksi
    return 0.0

# Mendefinisikan fungsi untuk melakukan deskewing
def deskew_image(img, sudut_skew):
    """Meluruskan gambar yang miring berdasarkan sudut yang terdeteksi."""
    # Mendapatkan ukuran gambar
    h, w = img.shape[:2]

    # Menghitung titik pusat
    pusat = (w // 2, h // 2)

    # Membuat matriks rotasi (rotasi negatif untuk meluruskan)
    M = cv2.getRotationMatrix2D(pusat, -sudut_skew, 1.0)

    # Menerapkan rotasi untuk meluruskan gambar
    img_deskewed = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    # Mengembalikan gambar yang sudah diluruskan
    return img_deskewed

# Mendefinisikan sudut-sudut kemiringan untuk demonstrasi
sudut_test = [0, 5, 10, -5, -10, 15]

# Membuat figure untuk visualisasi deskewing
fig, axes = plt.subplots(3, len(sudut_test), figsize=(22, 10))

# Memproses setiap sudut kemiringan
for col, sudut in enumerate(sudut_test):
    # Membuat gambar teks miring
    img_miring = buat_gambar_miring(img_printed, sudut)

    # Mengkonversi ke grayscale
    gray_miring = cv2.cvtColor(img_miring, cv2.COLOR_BGR2GRAY)

    # Menerapkan Otsu thresholding
    _, binary_miring = cv2.threshold(gray_miring, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Menginversi binary (teks hitam → putih pada background hitam)
    binary_inv = cv2.bitwise_not(binary_miring)

    # Mendeteksi sudut skew
    sudut_terdeteksi = deteksi_sudut_skew(binary_inv)

    # Melakukan deskewing
    img_deskewed = deskew_image(img_miring, sudut_terdeteksi)

    # Baris 1: Gambar miring
    axes[0, col].imshow(cv2.cvtColor(img_miring, cv2.COLOR_BGR2RGB))
    axes[0, col].set_title(f"Miring: {sudut}°", fontsize=9)
    axes[0, col].axis("off")

    # Baris 2: Hasil deteksi tepi dan garis Hough
    edges = cv2.Canny(binary_inv, 50, 150, apertureSize=3)
    # Membuat gambar berwarna dari edges untuk menggambar garis
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines[:20]:
            x1, y1, x2, y2 = line[0]
            cv2.line(edges_color, (x1, y1), (x2, y2), (0, 255, 0), 1)

    axes[1, col].imshow(cv2.cvtColor(edges_color, cv2.COLOR_BGR2RGB))
    axes[1, col].set_title(f"Deteksi: {sudut_terdeteksi:.1f}°", fontsize=9)
    axes[1, col].axis("off")

    # Baris 3: Gambar setelah deskewing
    axes[2, col].imshow(cv2.cvtColor(img_deskewed, cv2.COLOR_BGR2RGB))
    axes[2, col].set_title(f"Deskewed", fontsize=9)
    axes[2, col].axis("off")

    # Menampilkan informasi deskewing
    print(f"  Sudut asli: {sudut:+3d}° → Terdeteksi: {sudut_terdeteksi:+6.1f}° → "
          f"Error: {abs(sudut - sudut_terdeteksi):.1f}°")

# Menambahkan judul utama
plt.suptitle("Percobaan 6: Deskewing Gambar Teks\n"
             "Baris 1: Gambar miring | Baris 2: Hough Lines | Baris 3: Deskewed",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil deskewing
output_path_3 = os.path.join(OUTPUT_DIR, "06_deskewing.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 6")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Binarisasi:")
print("     - cv2.threshold(THRESH_BINARY): threshold global sederhana")
print("     - cv2.threshold(THRESH_OTSU): threshold otomatis Otsu")
print("       → Mencari threshold optimal berdasarkan histogram")
print("     - cv2.adaptiveThreshold(): threshold lokal/adaptif")
print("       → Threshold berbeda untuk setiap area kecil")
print("  2. Denoising:")
print("     - cv2.GaussianBlur(): smoothing dengan kernel Gaussian")
print("     - cv2.medianBlur(): efektif untuk salt-and-pepper")
print("     - cv2.bilateralFilter(): noise reduction + edge preservation")
print("  3. Operasi Morfologi:")
print("     - cv2.morphologyEx(MORPH_OPEN): opening (erosi + dilasi)")
print("     - cv2.morphologyEx(MORPH_CLOSE): closing (dilasi + erosi)")
print("     - cv2.dilate(): menebalkan objek/teks")
print("  4. Deskewing:")
print("     - cv2.Canny(): deteksi tepi")
print("     - cv2.HoughLinesP(): deteksi garis (sudut kemiringan)")
print("     - cv2.getRotationMatrix2D() + cv2.warpAffine(): rotasi")
print("  5. Pipeline preprocessing OCR yang baik:")
print("     Grayscale → Denoising → Binarisasi → Morfologi → Deskewing")
print("=" * 60)
