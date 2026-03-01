"""
==========================================================================
PERCOBAAN 4: AKSES DAN MANIPULASI PIKSEL
==========================================================================
Program ini mempelajari cara mengakses nilai piksel individual maupun
kelompok piksel, serta memodifikasinya secara langsung.

Konsep penting:
- Gambar = array NumPy 2D (grayscale) atau 3D (berwarna)
- Koordinat: img[y, x] (baris, kolom) → BUKAN (x, y)!
- Piksel BGR: img[y, x] = [blue, green, red]
- Piksel Grayscale: img[y, x] = intensitas (0-255)

Fungsi utama:
- img[y, x]         : Akses piksel di posisi (y, x)
- img[y, x] = val   : Mengubah nilai piksel
- img.item(y, x, c) : Akses cepat satu elemen
- img.itemset()      : Ubah cepat satu elemen
- Slicing array      : Akses area/region piksel
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
print("PERCOBAAN 4: AKSES DAN MANIPULASI PIKSEL")
print("=" * 60)

# Membaca gambar kucing dalam mode warna
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

# Membuat salinan gambar untuk manipulasi (agar asli tidak berubah)
# PENTING: Gunakan .copy() bukan assignment langsung!
# img2 = img → hanya membuat referensi (alias), bukan salinan
# img2 = img.copy() → membuat salinan independen
img_modif = img.copy()

# ============================================================
# 1. Mengakses nilai piksel individual
# ============================================================
print("\n--- 1. Akses Piksel Individual ---")

# Mengakses piksel di posisi (y=100, x=200)
# PERHATIAN: urutan adalah [baris/y, kolom/x], BUKAN [x, y]!
piksel = img[100, 200]
print(f"  Piksel di (y=100, x=200) : {piksel}")
print(f"  Blue={piksel[0]}, Green={piksel[1]}, Red={piksel[2]}")

# Mengakses satu channel saja menggunakan indeks ke-3
# Channel 0=Blue, 1=Green, 2=Red
blue_value = img[100, 200, 0]
green_value = img[100, 200, 1]
red_value = img[100, 200, 2]
print(f"  Blue channel  : {blue_value}")
print(f"  Green channel : {green_value}")
print(f"  Red channel   : {red_value}")

# ============================================================
# 2. Cara cepat akses piksel: item() dan itemset()
# ============================================================
print("\n--- 2. Akses Cepat dengan item() dan itemset() ---")

# img.item(y, x, channel) lebih cepat dari img[y, x, c] untuk akses tunggal
blue_fast = img.item(100, 200, 0)
print(f"  Blue (item method): {blue_fast}")

# img.itemset((y, x, channel), value) untuk mengubah satu piksel
img_modif.itemset((100, 200, 0), 255)  # Set blue channel ke 255
img_modif.itemset((100, 200, 1), 0)    # Set green channel ke 0
img_modif.itemset((100, 200, 2), 0)    # Set red channel ke 0
print(f"  Piksel diubah ke biru murni: {img_modif[100, 200]}")

# ============================================================
# 3. Mengubah area piksel (menggunakan slicing NumPy)
# ============================================================
print("\n--- 3. Manipulasi Area Piksel ---")

# Membuat kotak merah di area y:50-150, x:50-150
# Slicing: img[y_start:y_end, x_start:x_end] = [B, G, R]
img_modif[50:150, 50:150] = [0, 0, 255]  # BGR: merah
print("  Kotak merah (100x100 piksel) dibuat di posisi (50,50)")

# Membuat kotak hijau di area lain
img_modif[50:150, 160:260] = [0, 255, 0]  # BGR: hijau
print("  Kotak hijau (100x100 piksel) dibuat di posisi (50,160)")

# Membuat kotak biru
img_modif[50:150, 270:370] = [255, 0, 0]  # BGR: biru
print("  Kotak biru (100x100 piksel) dibuat di posisi (50,270)")

# ============================================================
# 4. Menyalin area piksel (copy region)
# ============================================================
print("\n--- 4. Menyalin Region ---")

# Menyalin bagian gambar dari satu posisi ke posisi lain
# Mengambil region wajah kucing (contoh area)
region = img[150:300, 200:400].copy()  # Salin region
print(f"  Region disalin: ukuran {region.shape}")

# Menempelkan region di posisi baru
img_modif[300:300+region.shape[0], 400:400+region.shape[1]] = region
print("  Region ditempelkan di posisi (300, 400)")

# ============================================================
# 5. Manipulasi piksel berdasarkan kondisi
# ============================================================
print("\n--- 5. Manipulasi Kondisional ---")

# Membuat salinan baru untuk manipulasi kondisional
img_cond = img.copy()

# Mengubah piksel yang grayscale-nya di bawah 100 menjadi hitam
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Membuat mask: True jika piksel gelap (< 100)
mask_gelap = gray < 100

# Menerapkan mask: ubah piksel gelap menjadi merah
# Expand mask ke 3 channel untuk diterapkan ke gambar BGR
img_cond[mask_gelap] = [0, 0, 200]
print("  Piksel gelap (< 100) diubah menjadi merah")

# ============================================================
# 6. Iterasi piksel (lambat, hanya untuk edukasi)
# ============================================================
print("\n--- 6. Iterasi Piksel (Demonstrasi) ---")

# Membuat gambar kecil untuk demonstrasi iterasi
img_kecil = np.zeros((100, 100, 3), dtype=np.uint8)

# Mengisi gambar piksel per piksel dengan gradient
for y in range(100):
    for x in range(100):
        # Menghitung nilai warna berdasarkan posisi
        img_kecil[y, x, 0] = x * 255 // 100      # Blue: gradient horizontal
        img_kecil[y, x, 1] = y * 255 // 100      # Green: gradient vertikal
        img_kecil[y, x, 2] = (x + y) * 255 // 200  # Red: gradient diagonal

print("  Gambar gradient 100x100 dibuat dengan loop piksel")
print("  CATATAN: Iterasi piksel SANGAT LAMBAT! Gunakan operasi NumPy!")

# ============================================================
# 7. Cara cepat (vektorisasi) vs cara lambat (loop)
# ============================================================
print("\n--- 7. Perbandingan Kecepatan ---")

import time

# Membuat gambar tes berukuran 500x500
img_test = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)

# Cara LAMBAT: invert gambar menggunakan loop
start = time.time()
img_slow = img_test.copy()
for y in range(500):
    for x in range(500):
        for c in range(3):
            img_slow[y, x, c] = 255 - img_slow[y, x, c]
waktu_loop = time.time() - start

# Cara CEPAT: invert gambar menggunakan operasi NumPy
start = time.time()
img_fast = 255 - img_test
waktu_numpy = time.time() - start

print(f"  Loop piksel  : {waktu_loop:.4f} detik")
print(f"  Operasi NumPy: {waktu_numpy:.6f} detik")
print(f"  NumPy {waktu_loop/max(waktu_numpy, 1e-6):.0f}x lebih cepat!")

# ============================================================
# 8. Visualisasi hasil
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Gambar asli
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli")
axes[0, 0].axis("off")

# Gambar setelah manipulasi kotak warna + copy region
axes[0, 1].imshow(cv2.cvtColor(img_modif, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Manipulasi: Kotak Warna + Copy Region")
axes[0, 1].axis("off")

# Gambar kondisional (piksel gelap → merah)
axes[0, 2].imshow(cv2.cvtColor(img_cond, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Piksel Gelap → Merah")
axes[0, 2].axis("off")

# Gambar gradient hasil iterasi piksel
axes[1, 0].imshow(cv2.cvtColor(img_kecil, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Gradient (Loop Piksel)")
axes[1, 0].axis("off")

# Gambar invert
axes[1, 1].imshow(cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Invert (NumPy)")
axes[1, 1].axis("off")

# Text perbandingan kecepatan
axes[1, 2].text(0.5, 0.5,
               f"Perbandingan Kecepatan\n\n"
               f"Loop: {waktu_loop:.4f}s\n"
               f"NumPy: {waktu_numpy:.6f}s\n\n"
               f"NumPy {waktu_loop/max(waktu_numpy, 1e-6):.0f}x\nlebih cepat!",
               ha="center", va="center", fontsize=16,
               transform=axes[1, 2].transAxes)
axes[1, 2].axis("off")

plt.suptitle("Percobaan 4: Akses dan Manipulasi Piksel", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "04_manipulasi_piksel_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 4")
print("=" * 60)
print("Cara akses dan manipulasi piksel:")
print("  1. img[y, x]         → Akses piksel (PERHATIKAN: y dulu, bukan x!)")
print("  2. img[y, x, c]      → Akses channel tertentu (0=B, 1=G, 2=R)")
print("  3. img[y1:y2, x1:x2] → Akses area/region (slicing)")
print("  4. img.item(y,x,c)   → Akses cepat satu elemen")
print("  5. img.itemset()     → Ubah cepat satu elemen")
print("  6. mask boolean      → Manipulasi piksel berdasarkan kondisi")
print("  7. SELALU gunakan NumPy daripada loop piksel!")
print("=" * 60)
