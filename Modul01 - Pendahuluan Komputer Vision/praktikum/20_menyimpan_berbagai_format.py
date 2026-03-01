"""
==========================================================================
PERCOBAAN 20: MENYIMPAN GAMBAR DALAM BERBAGAI FORMAT
==========================================================================
Program ini mempelajari cara menyimpan gambar ke file dengan berbagai
format dan parameter kualitas.

Fungsi utama:
- cv2.imwrite(filename, img, params) → Simpan gambar ke file
  Format ditentukan oleh ekstensi file (.jpg, .png, .bmp, .tiff)
  params → parameter kualitas spesifik per format:
  - JPEG: cv2.IMWRITE_JPEG_QUALITY (0-100)
  - PNG:  cv2.IMWRITE_PNG_COMPRESSION (0-9)
  - WEBP: cv2.IMWRITE_WEBP_QUALITY (1-100)
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 20: MENYIMPAN GAMBAR BERBAGAI FORMAT")
print("=" * 60)

# Membaca gambar
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

img = cv2.resize(img, (400, 400))
print(f"[INFO] Gambar: {img.shape}")

# ============================================================
# 1. Simpan sebagai JPEG dengan kualitas berbeda
# ============================================================
print("\n--- 1. Format JPEG ---")

kualitas_jpeg = [10, 30, 50, 70, 90, 100]

for q in kualitas_jpeg:
    path = os.path.join(OUTPUT_DIR, f"20_jpeg_q{q}.jpg")
    # cv2.IMWRITE_JPEG_QUALITY menentukan kualitas kompresi JPEG
    # Nilai 0 = kompresi maksimum (kualitas rendah)
    # Nilai 100 = kompresi minimum (kualitas tertinggi)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    # Cek ukuran file menggunakan os.path.getsize
    ukuran = os.path.getsize(path) / 1024  # Konversi ke KB
    print(f"  JPEG Q={q:3d} → {ukuran:7.1f} KB")

# ============================================================
# 2. Simpan sebagai PNG dengan kompresi berbeda
# ============================================================
print("\n--- 2. Format PNG ---")

kompresi_png = [0, 3, 6, 9]

for c in kompresi_png:
    path = os.path.join(OUTPUT_DIR, f"20_png_c{c}.png")
    # cv2.IMWRITE_PNG_COMPRESSION menentukan level kompresi PNG
    # Nilai 0 = tanpa kompresi (file besar, simpan cepat)
    # Nilai 9 = kompresi maksimum (file kecil, simpan lambat)
    # PNG menggunakan kompresi lossless (tidak mengurangi kualitas)
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, c])
    ukuran = os.path.getsize(path) / 1024
    print(f"  PNG Comp={c} → {ukuran:7.1f} KB")

# ============================================================
# 3. Simpan sebagai BMP (tanpa kompresi)
# ============================================================
print("\n--- 3. Format BMP ---")

path_bmp = os.path.join(OUTPUT_DIR, "20_format_bmp.bmp")
# BMP tidak menggunakan kompresi, ukuran = lebar × tinggi × channel
cv2.imwrite(path_bmp, img)
ukuran_bmp = os.path.getsize(path_bmp) / 1024
# Ukuran teoritis: 400 × 400 × 3 bytes = 480,000 bytes + header
ukuran_teori = (400 * 400 * 3) / 1024
print(f"  BMP → {ukuran_bmp:.1f} KB (teori: ~{ukuran_teori:.1f} KB)")

# ============================================================
# 4. Simpan sebagai TIFF
# ============================================================
print("\n--- 4. Format TIFF ---")

path_tiff = os.path.join(OUTPUT_DIR, "20_format_tiff.tiff")
# TIFF mendukung berbagai kompresi dan tipe data
cv2.imwrite(path_tiff, img)
ukuran_tiff = os.path.getsize(path_tiff) / 1024
print(f"  TIFF → {ukuran_tiff:.1f} KB")

# ============================================================
# 5. Simpan gambar grayscale
# ============================================================
print("\n--- 5. Simpan Grayscale ---")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"  Grayscale shape: {gray.shape}")

path_gray_jpg = os.path.join(OUTPUT_DIR, "20_grayscale.jpg")
cv2.imwrite(path_gray_jpg, gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
ukuran_gray = os.path.getsize(path_gray_jpg) / 1024

path_color_jpg = os.path.join(OUTPUT_DIR, "20_color.jpg")
cv2.imwrite(path_color_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
ukuran_color = os.path.getsize(path_color_jpg) / 1024

print(f"  Grayscale JPEG: {ukuran_gray:.1f} KB")
print(f"  Color JPEG:     {ukuran_color:.1f} KB")
print(f"  Rasio ukuran:   {ukuran_color / ukuran_gray:.1f}×")

# ============================================================
# 6. Simpan gambar dengan transparansi (RGBA PNG)
# ============================================================
print("\n--- 6. Gambar dengan Alpha Channel (RGBA) ---")

# Membuat channel alpha (transparansi)
# Membuat lingkaran transparan
alpha = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
cv2.circle(alpha, (200, 200), 150, 255, -1)

# cv2.merge menggabungkan 4 channel: B, G, R, Alpha
b, g, r = cv2.split(img)
img_rgba = cv2.merge([b, g, r, alpha])
print(f"  RGBA shape: {img_rgba.shape}")

path_rgba = os.path.join(OUTPUT_DIR, "20_transparansi.png")
# PNG mendukung 4 channel (RGBA) untuk transparansi
cv2.imwrite(path_rgba, img_rgba)
ukuran_rgba = os.path.getsize(path_rgba) / 1024
print(f"  RGBA PNG: {ukuran_rgba:.1f} KB")

# ============================================================
# 7. Simpan gambar float (16-bit dan 32-bit)
# ============================================================
print("\n--- 7. Format 16-bit dan 32-bit ---")

# Konversi ke 16-bit (untuk data dengan presisi lebih tinggi)
gray_16 = gray.astype(np.uint16) * 256
path_16 = os.path.join(OUTPUT_DIR, "20_gray_16bit.png")
cv2.imwrite(path_16, gray_16)
ukuran_16 = os.path.getsize(path_16) / 1024
print(f"  16-bit PNG: {ukuran_16:.1f} KB")

# Konversi ke float32 (OpenCV menyimpan float dalam format EXR/TIFF)
gray_float = gray.astype(np.float32) / 255.0
path_float = os.path.join(OUTPUT_DIR, "20_gray_float32.tiff")
cv2.imwrite(path_float, gray_float)
ukuran_float = os.path.getsize(path_float) / 1024
print(f"  Float32 TIFF: {ukuran_float:.1f} KB")

# ============================================================
# 8. Perbandingan ukuran file
# ============================================================
print("\n--- 8. Ringkasan Ukuran File ---")

# Membaca ukuran semua file yang disimpan
print(f"\n  {'Format':<20} {'Ukuran (KB)':>12} {'Kualitas':<15}")
print(f"  {'-'*47}")

format_info = [
    ("BMP (raw)", ukuran_bmp, "Lossless"),
    ("TIFF", ukuran_tiff, "Lossless"),
    ("PNG C=0", os.path.getsize(os.path.join(OUTPUT_DIR, "20_png_c0.png")) / 1024, "Lossless"),
    ("PNG C=9", os.path.getsize(os.path.join(OUTPUT_DIR, "20_png_c9.png")) / 1024, "Lossless"),
    ("JPEG Q=100", os.path.getsize(os.path.join(OUTPUT_DIR, "20_jpeg_q100.jpg")) / 1024, "Lossy-high"),
    ("JPEG Q=50", os.path.getsize(os.path.join(OUTPUT_DIR, "20_jpeg_q50.jpg")) / 1024, "Lossy-medium"),
    ("JPEG Q=10", os.path.getsize(os.path.join(OUTPUT_DIR, "20_jpeg_q10.jpg")) / 1024, "Lossy-low"),
]

for nama, ukuran, kualitas in format_info:
    print(f"  {nama:<20} {ukuran:>10.1f}  {kualitas:<15}")

# ============================================================
# 9. Visualisasi perbandingan kualitas JPEG
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, q in enumerate([10, 30, 50, 70, 90, 100]):
    baris = idx // 3
    kolom = idx % 3
    # Baca kembali file JPEG yang sudah disimpan
    path = os.path.join(OUTPUT_DIR, f"20_jpeg_q{q}.jpg")
    img_baca = cv2.imread(path)
    ukuran = os.path.getsize(path) / 1024
    axes[baris, kolom].imshow(cv2.cvtColor(img_baca, cv2.COLOR_BGR2RGB))
    axes[baris, kolom].set_title(f"JPEG Q={q} ({ukuran:.1f} KB)")
    axes[baris, kolom].axis("off")

plt.suptitle("Percobaan 20: Perbandingan Kualitas JPEG", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "20_perbandingan_format_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 20")
print("=" * 60)
print("  cv2.imwrite() → Fungsi utama simpan gambar")
print("  JPEG: lossy, IMWRITE_JPEG_QUALITY (0-100)")
print("  PNG:  lossless, IMWRITE_PNG_COMPRESSION (0-9)")
print("  BMP:  raw, tanpa kompresi (file terbesar)")
print("  TIFF: mendukung float32, 16-bit, multi-page")
print("  RGBA PNG: mendukung transparansi (4 channel)")
print("=" * 60)
