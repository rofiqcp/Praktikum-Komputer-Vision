"""
==========================================================================
PERCOBAAN 3: KONVERSI RUANG WARNA
==========================================================================
Program ini mempelajari berbagai ruang warna (color space) dalam
pemrosesan gambar dan cara mengkonversi dari satu ruang warna ke lainnya.

Ruang warna yang dipelajari:
- BGR  : Default OpenCV (Blue, Green, Red)
- RGB  : Standard tampilan (Red, Green, Blue)
- GRAY : Grayscale (1 channel, 0-255)
- HSV  : Hue, Saturation, Value (berguna untuk deteksi warna)
- HLS  : Hue, Lightness, Saturation
- LAB  : Lightness, a* (green-red), b* (blue-yellow) - perceptual
- YCrCb: Luminance, Chrominance (digunakan di JPEG)

Fungsi utama:
- cv2.cvtColor(src, code) : Mengkonversi ruang warna
==========================================================================
"""

# Mengimpor library yang dibutuhkan
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Mendapatkan path direktori
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 3: KONVERSI RUANG WARNA")
print("=" * 60)

# ============================================================
# 1. Membaca gambar berwarna
# ============================================================

# Membaca gambar warna-warni yang memiliki banyak warna berbeda
img_bgr = cv2.imread(os.path.join(IMAGE_DIR, "warna_warni.png"))

# Memeriksa apakah gambar berhasil dimuat
if img_bgr is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

print(f"[INFO] Gambar dimuat: {img_bgr.shape}")

# ============================================================
# 2. BGR → RGB (untuk matplotlib / tampilan normal)
# ============================================================

# cv2.cvtColor() mengkonversi gambar dari satu ruang warna ke lainnya
# cv2.COLOR_BGR2RGB : menukar channel Blue dan Red
# OpenCV membaca gambar sebagai BGR (Blue, Green, Red)
# Sedangkan display normal menggunakan RGB (Red, Green, Blue)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print("[INFO] Konversi BGR → RGB selesai")

# ============================================================
# 3. BGR → Grayscale (abu-abu, 1 channel)
# ============================================================

# cv2.COLOR_BGR2GRAY : mengkonversi gambar berwarna ke grayscale
# Rumus: Gray = 0.299*R + 0.587*G + 0.114*B (luminance)
# Mata manusia lebih sensitif terhadap hijau, makanya bobotnya paling besar
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(f"[INFO] Konversi BGR → GRAY selesai: {img_gray.shape}")

# ============================================================
# 4. BGR → HSV (Hue, Saturation, Value)
# ============================================================

# cv2.COLOR_BGR2HSV : mengkonversi ke ruang warna HSV
# HSV sangat berguna untuk deteksi warna karena memisahkan:
# - Hue (H): jenis warna (0-179 di OpenCV, bukan 0-360!)
# - Saturation (S): intensitas/kepekatan warna (0-255)
# - Value (V): kecerahan (0-255)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
print(f"[INFO] Konversi BGR → HSV selesai")

# Menampilkan range HSV untuk masing-masing warna dasar
print("\n--- Range HSV untuk Deteksi Warna ---")
print("  Merah   : H=0-10, 170-179  S=100-255  V=100-255")
print("  Hijau   : H=35-85          S=100-255  V=100-255")
print("  Biru    : H=100-130        S=100-255  V=100-255")
print("  Kuning  : H=20-35          S=100-255  V=100-255")
print("  Cyan    : H=80-100         S=100-255  V=100-255")
print("  Magenta : H=130-170        S=100-255  V=100-255")

# ============================================================
# 5. BGR → HLS (Hue, Lightness, Saturation)
# ============================================================

# cv2.COLOR_BGR2HLS : mirip HSV tapi menggunakan Lightness bukan Value
# Perbedaan: putih memiliki ligthtness tinggi tapi saturation rendah
img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
print(f"[INFO] Konversi BGR → HLS selesai")

# ============================================================
# 6. BGR → LAB (CIE L*a*b*)
# ============================================================

# cv2.COLOR_BGR2LAB : mengkonversi ke ruang warna LAB
# LAB dirancang agar jarak Euclidean sesuai dengan persepsi manusia:
# - L: Lightness (0-255, di OpenCV diskalakan dari 0-100)
# - a: komponen hijau-merah (0-255, 128 = netral)
# - b: komponen biru-kuning (0-255, 128 = netral)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
print(f"[INFO] Konversi BGR → LAB selesai")

# ============================================================
# 7. BGR → YCrCb (Luminance + Chrominance)
# ============================================================

# cv2.COLOR_BGR2YCrCb : digunakan dalam kompresi JPEG dan video
# - Y: Luminance (kecerahan)
# - Cr: Chrominance Red (perbedaan merah dari luminance)
# - Cb: Chrominance Blue (perbedaan biru dari luminance)
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
print(f"[INFO] Konversi BGR → YCrCb selesai")

# ============================================================
# 8. Visualisasi semua ruang warna
# ============================================================

# Membuat figure besar dengan 7 subplot
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Baris 1: BGR, RGB, Grayscale, HSV
axes[0, 0].imshow(img_bgr)  # Sengaja BGR agar terlihat bedanya
axes[0, 0].set_title("BGR (OpenCV default)\n(Warna tertukar di matplotlib)")

axes[0, 1].imshow(img_rgb)
axes[0, 1].set_title("RGB (Warna benar)")

axes[0, 2].imshow(img_gray, cmap="gray")
axes[0, 2].set_title("Grayscale\n(1 channel)")

axes[0, 3].imshow(img_hsv)
axes[0, 3].set_title("HSV\n(H=warna, S=saturasi, V=value)")

# Baris 2: HLS, LAB, YCrCb, Channel HSV terpisah
axes[1, 0].imshow(img_hls)
axes[1, 0].set_title("HLS\n(H=warna, L=lightness, S=saturasi)")

axes[1, 1].imshow(img_lab)
axes[1, 1].set_title("LAB\n(L=light, a=hijau-merah, b=biru-kuning)")

axes[1, 2].imshow(img_ycrcb)
axes[1, 2].set_title("YCrCb\n(Y=luminance, Cr=chrom-red, Cb=chrom-blue)")

# Menampilkan channel Hue dari HSV secara terpisah (berguna untuk deteksi warna)
h_channel = img_hsv[:, :, 0]  # Hue channel saja
axes[1, 3].imshow(h_channel, cmap="hsv")
axes[1, 3].set_title("Hue Channel (HSV)\n(Jenis warna)")

# Menghilangkan axis pada semua subplot
for ax in axes.flat:
    ax.axis("off")

plt.suptitle("Percobaan 3: Konversi Ruang Warna", fontsize=16, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil visualisasi
output_path = os.path.join(OUTPUT_DIR, "03_konversi_ruang_warna_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Visualisasi disimpan di: {output_path}")

# ============================================================
# 9. Visualisasi channel terpisah untuk HSV
# ============================================================

# Memisahkan 3 channel HSV menggunakan cv2.split()
h, s, v = cv2.split(img_hsv)

# Membuat figure untuk channel HSV terpisah
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

# Channel Hue (warna) - range 0-179 di OpenCV
axes2[0].imshow(h, cmap="hsv")
axes2[0].set_title(f"Hue (0-179)\nmin={h.min()}, max={h.max()}")
axes2[0].axis("off")

# Channel Saturation (kepekatan) - range 0-255
axes2[1].imshow(s, cmap="gray")
axes2[1].set_title(f"Saturation (0-255)\nmin={s.min()}, max={s.max()}")
axes2[1].axis("off")

# Channel Value (kecerahan) - range 0-255
axes2[2].imshow(v, cmap="gray")
axes2[2].set_title(f"Value (0-255)\nmin={v.min()}, max={v.max()}")
axes2[2].axis("off")

plt.suptitle("Channel HSV Terpisah", fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan visualisasi channel HSV
output_path2 = os.path.join(OUTPUT_DIR, "03_channel_hsv_terpisah.png")
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Channel HSV disimpan di: {output_path2}")

# ============================================================
# 10. Demonstrasi deteksi warna menggunakan HSV
# ============================================================

# Mendeteksi warna merah pada gambar menggunakan range HSV
# Merah ada di 2 range karena Hue adalah siklus (0 dan 179 berdekatan)
lower_red1 = np.array([0, 100, 100])     # Batas bawah merah range 1
upper_red1 = np.array([10, 255, 255])     # Batas atas merah range 1
lower_red2 = np.array([170, 100, 100])    # Batas bawah merah range 2
upper_red2 = np.array([179, 255, 255])    # Batas atas merah range 2

# cv2.inRange() membuat mask binary: putih jika dalam range, hitam jika tidak
mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)

# Menggabungkan kedua mask menggunakan operasi OR
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Menerapkan mask ke gambar asli
# cv2.bitwise_and() hanya menampilkan piksel yang mask-nya putih
result_red = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_red)

# Mendeteksi warna hijau
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
result_green = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_green)

# Visualisasi hasil deteksi warna
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))

axes3[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axes3[0].set_title("Gambar Asli")
axes3[0].axis("off")

axes3[1].imshow(cv2.cvtColor(result_red, cv2.COLOR_BGR2RGB))
axes3[1].set_title("Deteksi Warna Merah (HSV)")
axes3[1].axis("off")

axes3[2].imshow(cv2.cvtColor(result_green, cv2.COLOR_BGR2RGB))
axes3[2].set_title("Deteksi Warna Hijau (HSV)")
axes3[2].axis("off")

plt.suptitle("Deteksi Warna Menggunakan HSV", fontsize=14, fontweight="bold")
plt.tight_layout()

output_path3 = os.path.join(OUTPUT_DIR, "03_deteksi_warna_hsv.png")
plt.savefig(output_path3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Deteksi warna disimpan di: {output_path3}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 3")
print("=" * 60)
print("Ruang warna yang dipelajari:")
print("  1. BGR/RGB  → Standard warna, 3 channel")
print("  2. Grayscale → 1 channel, intensitas cahaya")
print("  3. HSV      → Hue, Saturation, Value (untuk deteksi warna)")
print("  4. HLS      → Hue, Lightness, Saturation")
print("  5. LAB      → Perceptual uniform color space")
print("  6. YCrCb    → Luminance + Chrominance (JPEG)")
print("\nFungsi utama:")
print("  - cv2.cvtColor(img, code) → Konversi ruang warna")
print("  - cv2.inRange(img, lower, upper) → Mask berdasarkan range")
print("  - cv2.bitwise_and() → Terapkan mask ke gambar")
print("=" * 60)
