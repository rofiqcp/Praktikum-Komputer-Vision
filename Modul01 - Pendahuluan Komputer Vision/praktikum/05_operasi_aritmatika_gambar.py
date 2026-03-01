"""
==========================================================================
PERCOBAAN 5: OPERASI ARITMATIKA GAMBAR
==========================================================================
Program ini mempelajari operasi matematika dasar pada gambar:
penjumlahan, pengurangan, perkalian, pembagian, dan penanganan overflow.

Konsep penting:
- Gambar = array angka, bisa dilakukan operasi matematika
- OpenCV: SATURASI (clip ke 0-255, tidak overflow)
- NumPy : MODULO (wrap-around, 256→0, -1→255)

Fungsi utama:
- cv2.add(img1, img2)       : Penjumlahan dengan saturasi
- cv2.subtract(img1, img2)  : Pengurangan dengan saturasi
- cv2.multiply(img1, img2)  : Perkalian elemen-wise
- cv2.divide(img1, img2)    : Pembagian elemen-wise
- cv2.addWeighted()          : Blending berbobot (alpha blending)
- cv2.absdiff()              : Selisih absolut (|img1 - img2|)
==========================================================================
"""

# Mengimpor library yang dibutuhkan
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 5: OPERASI ARITMATIKA GAMBAR")
print("=" * 60)

# Membaca dua gambar untuk operasi aritmatika
img1 = cv2.imread(os.path.join(IMAGE_DIR, "foto_bunga.jpg"))
img2 = cv2.imread(os.path.join(IMAGE_DIR, "foto_hewan.jpg"))

if img1 is None or img2 is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

# Menyamakan ukuran kedua gambar (harus sama untuk operasi aritmetika)
# cv2.resize() mengubah ukuran gambar
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
print(f"[INFO] Ukuran img1: {img1.shape}, img2: {img2.shape}")

# ============================================================
# 1. Perbedaan cv2.add() vs operator + (NumPy)
# ============================================================
print("\n--- 1. cv2.add() vs NumPy + ---")

# cv2.add() menggunakan SATURASI: 200 + 100 = 255 (clip di 255)
hasil_cv_add = cv2.add(img1, img2)

# Operator + menggunakan MODULO: 200 + 100 = 300 % 256 = 44
# Ini menyebabkan warna yang aneh (wrap-around)
hasil_np_add = img1 + img2

# Demonstrasi perbedaan saturasi vs modulo
print(f"  Contoh piksel img1[100,200]: {img1[100, 200]}")
print(f"  Contoh piksel img2[100,200]: {img2[100, 200]}")
print(f"  cv2.add (saturasi):  {hasil_cv_add[100, 200]}")
print(f"  NumPy + (modulo):    {hasil_np_add[100, 200]}")

# ============================================================
# 2. Pengurangan gambar
# ============================================================
print("\n--- 2. Pengurangan Gambar ---")

# cv2.subtract() dengan saturasi: 50 - 100 = 0 (clip di 0)
hasil_subtract = cv2.subtract(img1, img2)

# Selisih absolut: |img1 - img2| (berguna untuk deteksi perubahan)
# cv2.absdiff() selalu menghasilkan nilai positif
hasil_absdiff = cv2.absdiff(img1, img2)

print(f"  cv2.subtract: min={hasil_subtract.min()}, max={hasil_subtract.max()}")
print(f"  cv2.absdiff:  min={hasil_absdiff.min()}, max={hasil_absdiff.max()}")

# ============================================================
# 3. Menambahkan konstanta (membuat lebih terang/gelap)
# ============================================================
print("\n--- 3. Tambah/Kurang Konstanta ---")

# Menambahkan 80 ke semua piksel → gambar lebih TERANG
# np.ones_like() membuat array berisi 1 dengan shape & dtype sama
img_terang = cv2.add(img1, np.ones_like(img1) * 80)

# Mengurangi 80 dari semua piksel → gambar lebih GELAP
img_gelap = cv2.subtract(img1, np.ones_like(img1) * 80)

print("  Gambar diperbanyak kecerahan +80 (terang)")
print("  Gambar dikurangi kecerahan -80 (gelap)")

# ============================================================
# 4. Perkalian (multiply) - mengubah kontras
# ============================================================
print("\n--- 4. Perkalian Gambar ---")

# cv2.multiply() mengalikan piksel element-wise
# Mengalikan dengan faktor > 1 → kontras naik
# Mengalikan dengan faktor < 1 → kontras turun
# scale=1/255 karena multiply dua uint8 bisa overflow
hasil_multiply = cv2.multiply(img1, np.ones_like(img1) * 2, scale=1.0/255)

# Cara lain: convertScaleAbs untuk mengubah kontras dan brightness
# dst = saturate(|src * alpha + beta|)
# alpha = kontras (1.0 = normal), beta = brightness (0 = normal)
img_kontras_tinggi = cv2.convertScaleAbs(img1, alpha=1.5, beta=0)
img_kontras_rendah = cv2.convertScaleAbs(img1, alpha=0.5, beta=0)

print(f"  Kontras tinggi (alpha=1.5): range {img_kontras_tinggi.min()}-{img_kontras_tinggi.max()}")
print(f"  Kontras rendah (alpha=0.5): range {img_kontras_rendah.min()}-{img_kontras_rendah.max()}")

# ============================================================
# 5. Alpha Blending (addWeighted)
# ============================================================
print("\n--- 5. Alpha Blending ---")

# cv2.addWeighted(src1, alpha, src2, beta, gamma)
# Rumus: dst = src1 * alpha + src2 * beta + gamma
# alpha + beta biasanya = 1.0 agar tidak terlalu terang

# Blending 70% img1 + 30% img2
blend_70_30 = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

# Blending 50% img1 + 50% img2
blend_50_50 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

# Blending 30% img1 + 70% img2
blend_30_70 = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)

print("  Blend 70:30 selesai")
print("  Blend 50:50 selesai")
print("  Blend 30:70 selesai")

# ============================================================
# 6. Invert (negatif) gambar
# ============================================================
print("\n--- 6. Invert/Negatif Gambar ---")

# Invert = 255 - piksel (komplementer)
# cv2.bitwise_not() juga bisa digunakan untuk invert
img_negatif = cv2.bitwise_not(img1)
print(f"  Gambar negatif: piksel asli {img1[100,200]} → {img_negatif[100,200]}")

# ============================================================
# 7. Visualisasi semua hasil
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Operasi dasar
axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar 1 (Asli)")

axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Gambar 2")

axes[0, 2].imshow(cv2.cvtColor(hasil_cv_add, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("cv2.add (Saturasi)")

axes[0, 3].imshow(cv2.cvtColor(hasil_np_add, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("NumPy + (Modulo)\n⚠ Warna aneh!")

# Baris 2: Pengurangan dan brightness
axes[1, 0].imshow(cv2.cvtColor(hasil_subtract, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("cv2.subtract")

axes[1, 1].imshow(cv2.cvtColor(hasil_absdiff, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("cv2.absdiff\n|img1 - img2|")

axes[1, 2].imshow(cv2.cvtColor(img_terang, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Lebih Terang (+80)")

axes[1, 3].imshow(cv2.cvtColor(img_gelap, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Lebih Gelap (-80)")

# Baris 3: Kontras dan blending
axes[2, 0].imshow(cv2.cvtColor(img_kontras_tinggi, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("Kontras Tinggi (×1.5)")

axes[2, 1].imshow(cv2.cvtColor(blend_50_50, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("Blend 50:50")

axes[2, 2].imshow(cv2.cvtColor(blend_70_30, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title("Blend 70:30")

axes[2, 3].imshow(cv2.cvtColor(img_negatif, cv2.COLOR_BGR2RGB))
axes[2, 3].set_title("Negatif (invert)")

for ax in axes.flat:
    ax.axis("off")

plt.suptitle("Percobaan 5: Operasi Aritmatika Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "05_aritmatika_gambar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 5")
print("=" * 60)
print("Operasi aritmatika gambar:")
print("  1. cv2.add()         → Penjumlahan (SATURASI: clip 0-255)")
print("  2. cv2.subtract()    → Pengurangan (SATURASI)")
print("  3. NumPy +/-         → MODULO (wrap-around, HINDARI!)")
print("  4. cv2.absdiff()     → Selisih absolut |a-b|")
print("  5. cv2.addWeighted() → Alpha blending (a*α + b*β + γ)")
print("  6. cv2.convertScaleAbs() → Ubah kontras & brightness")
print("  7. cv2.bitwise_not() → Invert/negatif gambar")
print("=" * 60)
