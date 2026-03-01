"""
==========================================================================
PERCOBAAN 13: WHITE BALANCE CORRECTION
==========================================================================
Program ini mempelajari metode koreksi white balance pada gambar.
White balance penting agar warna putih terlihat benar-benar putih
dan warna lainnya ditampilkan secara akurat, terlepas dari pencahayaan.

Metode yang dipelajari:
1. Gray World Assumption - menyesuaikan channel agar mean sama
2. Max White Method - normalisasi berdasarkan nilai maksimum channel
3. Color Temperature Adjustment - mengubah suhu warna (hangat/dingin)

Fungsi utama yang dipelajari:
- cv2.cvtColor()          : Konversi ruang warna
- cv2.split() / merge()   : Memisahkan dan menggabungkan channel
- np.mean()               : Menghitung rata-rata channel
- np.clip()               : Membatasi rentang nilai piksel
- cv2.addWeighted()       : Blending gambar

Hasil: Perbandingan visual berbagai metode koreksi white balance
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
print("PERCOBAAN 13: WHITE BALANCE CORRECTION")
print("=" * 60)

# ============================================================
# 1. Membaca gambar input
# ============================================================

# Mendefinisikan path gambar pemandangan
path_img = os.path.join(IMAGE_DIR, "scene_pemandangan.png")

# Membaca gambar dalam format BGR
img = cv2.imread(path_img)

# Memeriksa apakah gambar berhasil dimuat
if img is None:
    # Jika gambar tidak ditemukan, buat gambar sintetis dengan warna bias
    print("[INFO] scene_pemandangan.png tidak ditemukan, membuat gambar sintetis...")
    # Membuat gambar sintetis dengan bias kuning (simulasi cahaya tungsten)
    base = np.random.randint(60, 200, (400, 600, 3), dtype=np.uint8)
    # Menambahkan bias warna hangat (lebih banyak merah dan hijau)
    base[:, :, 2] = np.clip(base[:, :, 2].astype(np.int16) + 40, 0, 255).astype(np.uint8)
    base[:, :, 1] = np.clip(base[:, :, 1].astype(np.int16) + 20, 0, 255).astype(np.uint8)
    img = base

# Menampilkan informasi dimensi dan tipe data gambar
print(f"[INFO] Ukuran gambar: {img.shape}")
print(f"[INFO] Tipe data: {img.dtype}")

# ============================================================
# 2. Metode 1: Gray World Assumption
# ============================================================
print("\n[LANGKAH 1] Menerapkan Gray World White Balance...")

def gray_world(image):
    """Koreksi white balance menggunakan asumsi Gray World."""
    # Mengonversi gambar ke float untuk kalkulasi presisi
    img_float = image.astype(np.float64)

    # Menghitung rata-rata setiap channel (B, G, R)
    mean_b = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_r = np.mean(img_float[:, :, 2])

    # Menghitung rata-rata global dari semua channel
    mean_global = (mean_b + mean_g + mean_r) / 3.0

    # Menghitung faktor koreksi untuk setiap channel
    scale_b = mean_global / (mean_b + 1e-6)
    scale_g = mean_global / (mean_g + 1e-6)
    scale_r = mean_global / (mean_r + 1e-6)

    # Menampilkan faktor koreksi
    print(f"  Mean B={mean_b:.1f}, G={mean_g:.1f}, R={mean_r:.1f}")
    print(f"  Scale B={scale_b:.3f}, G={scale_g:.3f}, R={scale_r:.3f}")

    # Menerapkan faktor koreksi ke setiap channel
    result = img_float.copy()
    result[:, :, 0] = result[:, :, 0] * scale_b
    result[:, :, 1] = result[:, :, 1] * scale_g
    result[:, :, 2] = result[:, :, 2] * scale_r

    # Membatasi nilai ke rentang 0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Mengembalikan hasil koreksi
    return result

# Menerapkan Gray World pada gambar
img_gray_world = gray_world(img)

# ============================================================
# 3. Metode 2: Max White Method
# ============================================================
print("\n[LANGKAH 2] Menerapkan Max White White Balance...")

def max_white(image):
    """Koreksi white balance menggunakan metode Max White."""
    # Mengonversi gambar ke float
    img_float = image.astype(np.float64)

    # Mencari nilai maksimum setiap channel
    max_b = np.max(img_float[:, :, 0])
    max_g = np.max(img_float[:, :, 1])
    max_r = np.max(img_float[:, :, 2])

    # Menampilkan nilai maksimum
    print(f"  Max B={max_b:.1f}, G={max_g:.1f}, R={max_r:.1f}")

    # Menormalisasi setiap channel relatif terhadap maksimumnya
    result = img_float.copy()
    result[:, :, 0] = result[:, :, 0] * (255.0 / (max_b + 1e-6))
    result[:, :, 1] = result[:, :, 1] * (255.0 / (max_g + 1e-6))
    result[:, :, 2] = result[:, :, 2] * (255.0 / (max_r + 1e-6))

    # Membatasi nilai ke rentang 0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Mengembalikan hasil koreksi
    return result

# Menerapkan Max White pada gambar
img_max_white = max_white(img)

# ============================================================
# 4. Metode 3: Color Temperature Adjustment (Warm/Cool)
# ============================================================
print("\n[LANGKAH 3] Menyesuaikan suhu warna...")

def adjust_color_temperature(image, temperature):
    """
    Mengubah suhu warna gambar.
    temperature > 0: lebih hangat (warm/kuning-oranye)
    temperature < 0: lebih dingin (cool/biru)
    """
    # Mengonversi gambar ke float
    img_float = image.astype(np.float64)

    # Membuat salinan hasil
    result = img_float.copy()

    # Jika temperature positif → tambah merah, kurangi biru (hangat)
    if temperature > 0:
        # Menambahkan komponen merah (channel index 2 di BGR)
        result[:, :, 2] = result[:, :, 2] + temperature
        # Mengurangi komponen biru (channel index 0 di BGR)
        result[:, :, 0] = result[:, :, 0] - temperature * 0.5
    else:
        # Jika temperature negatif → tambah biru, kurangi merah (dingin)
        result[:, :, 0] = result[:, :, 0] + abs(temperature)
        # Mengurangi komponen merah
        result[:, :, 2] = result[:, :, 2] - abs(temperature) * 0.5

    # Membatasi nilai ke rentang 0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Mengembalikan hasil
    return result

# Membuat variasi suhu warna: sangat dingin, dingin, hangat, sangat hangat
img_cool_strong = adjust_color_temperature(img, -40)
img_cool = adjust_color_temperature(img, -20)
img_warm = adjust_color_temperature(img, 20)
img_warm_strong = adjust_color_temperature(img, 40)

# ============================================================
# 5. Visualisasi perbandingan metode white balance
# ============================================================
print("\n[LANGKAH 4] Membuat visualisasi perbandingan...")

# Membuat figure untuk perbandingan metode utama
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

# Menampilkan gambar original
axes1[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes1[0].set_title("Original", fontsize=12)
axes1[0].axis("off")

# Menampilkan hasil Gray World
axes1[1].imshow(cv2.cvtColor(img_gray_world, cv2.COLOR_BGR2RGB))
axes1[1].set_title("Gray World", fontsize=12)
axes1[1].axis("off")

# Menampilkan hasil Max White
axes1[2].imshow(cv2.cvtColor(img_max_white, cv2.COLOR_BGR2RGB))
axes1[2].set_title("Max White", fontsize=12)
axes1[2].axis("off")

# Menambahkan judul utama
fig1.suptitle("Perbandingan Metode White Balance", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path1 = os.path.join(OUTPUT_DIR, "13_white_balance_methods.png")
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path1}")

# Menutup figure
plt.close(fig1)

# ============================================================
# 6. Visualisasi variasi suhu warna
# ============================================================
print("\n[LANGKAH 5] Membuat visualisasi variasi suhu warna...")

# Membuat figure untuk variasi suhu warna
fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))

# Daftar gambar dan label suhu warna
temp_images = [img_cool_strong, img_cool, img, img_warm, img_warm_strong]
temp_labels = ["Sangat Dingin (-40)", "Dingin (-20)", "Original (0)", "Hangat (+20)", "Sangat Hangat (+40)"]

# Menampilkan setiap variasi suhu warna
for idx, (t_img, t_label) in enumerate(zip(temp_images, temp_labels)):
    # Menampilkan gambar pada subplot yang sesuai
    axes2[idx].imshow(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
    axes2[idx].set_title(t_label, fontsize=10)
    axes2[idx].axis("off")

# Menambahkan judul utama
fig2.suptitle("Variasi Suhu Warna (Color Temperature)", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path2 = os.path.join(OUTPUT_DIR, "13_color_temperature.png")
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path2}")

# Menutup figure
plt.close(fig2)

# ============================================================
# 7. Analisis distribusi warna sebelum dan sesudah koreksi
# ============================================================
print("\n[LANGKAH 6] Analisis distribusi warna...")

# Membuat figure untuk histogram warna
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

# Daftar gambar untuk analisis histogram
hist_images = [img, img_gray_world, img_max_white]
hist_titles = ["Original", "Gray World", "Max White"]

# Menampilkan histogram setiap metode
for idx, (h_img, h_title) in enumerate(zip(hist_images, hist_titles)):
    # Menghitung histogram untuk channel B
    hist_b = cv2.calcHist([h_img], [0], None, [256], [0, 256])
    # Menghitung histogram untuk channel G
    hist_g = cv2.calcHist([h_img], [1], None, [256], [0, 256])
    # Menghitung histogram untuk channel R
    hist_r = cv2.calcHist([h_img], [2], None, [256], [0, 256])

    # Menampilkan histogram ketiga channel
    axes3[idx].plot(hist_b, color='blue', label='Blue', alpha=0.7)
    axes3[idx].plot(hist_g, color='green', label='Green', alpha=0.7)
    axes3[idx].plot(hist_r, color='red', label='Red', alpha=0.7)
    axes3[idx].set_title(f"Histogram - {h_title}", fontsize=11)
    axes3[idx].set_xlim([0, 256])
    axes3[idx].legend()

# Menambahkan judul utama
fig3.suptitle("Distribusi Warna Sebelum dan Sesudah White Balance", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path3 = os.path.join(OUTPUT_DIR, "13_histogram_warna.png")
fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path3}")

# Menutup figure
plt.close(fig3)

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 13: WHITE BALANCE CORRECTION")
print("=" * 60)
print("Fungsi-fungsi yang dipelajari:")
print("1. Gray World Assumption")
print("   → Menyesuaikan mean setiap channel agar sama")
print("   → Asumsi: rata-rata warna di dunia nyata adalah abu-abu")
print("2. Max White Method")
print("   → Normalisasi berdasarkan nilai maksimum setiap channel")
print("   → Asumsi: piksel paling terang seharusnya putih")
print("3. Color Temperature Adjustment")
print("   → Menggeser keseimbangan merah-biru (hangat/dingin)")
print()
print("Fungsi OpenCV:")
print("- cv2.split() / cv2.merge() → Memisahkan/gabung channel")
print("- cv2.calcHist() → Menghitung histogram distribusi warna")
print("- np.mean(), np.max() → Statistik channel untuk koreksi")
print("- np.clip() → Membatasi nilai agar valid (0-255)")
print()
print("Kesimpulan:")
print("- Gray World baik untuk scene kompleks dengan variasi warna")
print("- Max White baik jika ada area putih di gambar")
print("- Suhu warna penting untuk mood dan kesan visual gambar")
print("=" * 60)
