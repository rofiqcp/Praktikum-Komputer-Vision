"""
==========================================================================
PERCOBAAN 6: OPERASI BITWISE
==========================================================================
Program ini mempelajari operasi bitwise (operasi bit per bit) pada gambar.
Operasi bitwise sangat berguna untuk membuat mask, overlay logo, dan
menggabungkan gambar secara selektif.

Fungsi utama:
- cv2.bitwise_and(img1, img2)  : AND → Hanya piksel yang sama-sama ON
- cv2.bitwise_or(img1, img2)   : OR  → Piksel yang salah satunya ON
- cv2.bitwise_xor(img1, img2)  : XOR → Piksel yang berbeda saja
- cv2.bitwise_not(img)          : NOT → Invert semua bit

Konsep: Setiap piksel = 8 bit (0000 0000 - 1111 1111)
AND : 1 & 1 = 1, lainnya = 0
OR  : 0 | 0 = 0, lainnya = 1
XOR : sama = 0, beda = 1
NOT : 0 → 1, 1 → 0
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
print("PERCOBAAN 6: OPERASI BITWISE")
print("=" * 60)

# ============================================================
# 1. Membuat dua gambar sederhana untuk demo bitwise
# ============================================================

# Membuat gambar persegi putih di atas latar hitam
img_rect = np.zeros((400, 400), dtype=np.uint8)
# cv2.rectangle(img, titik_kiri_atas, titik_kanan_bawah, warna, ketebalan)
# ketebalan -1 = filled (terisi penuh)
cv2.rectangle(img_rect, (50, 50), (250, 250), 255, -1)

# Membuat gambar lingkaran putih di atas latar hitam
img_circle = np.zeros((400, 400), dtype=np.uint8)
# cv2.circle(img, titik_pusat, radius, warna, ketebalan)
cv2.circle(img_circle, (250, 250), 150, 255, -1)

print("[INFO] Dua gambar mask dibuat: persegi dan lingkaran")

# ============================================================
# 2. Operasi AND - Irisan (intersection)
# ============================================================

# AND: Piksel = putih HANYA jika di kedua gambar pikselnya putih
# Berguna untuk: menampilkan area yang overlap/berpotongan
hasil_and = cv2.bitwise_and(img_rect, img_circle)
print("[INFO] AND: Hanya area yang overlap yang berwarna putih")

# ============================================================
# 3. Operasi OR - Gabungan (union)
# ============================================================

# OR: Piksel = putih jika di salah satu atau kedua gambar pikselnya putih
# Berguna untuk: menggabungkan dua mask
hasil_or = cv2.bitwise_or(img_rect, img_circle)
print("[INFO] OR: Area gabungan kedua bentuk berwarna putih")

# ============================================================
# 4. Operasi XOR - Perbedaan simetris (symmetric difference)
# ============================================================

# XOR: Piksel = putih HANYA jika di salah satu gambar (bukan keduanya)
# Berguna untuk: mendeteksi perbedaan antara dua gambar
hasil_xor = cv2.bitwise_xor(img_rect, img_circle)
print("[INFO] XOR: Hanya area yang berbeda yang berwarna putih")

# ============================================================
# 5. Operasi NOT - Invert (komplemen)
# ============================================================

# NOT: Membalik semua bit → putih jadi hitam, hitam jadi putih
# Berguna untuk: membuat mask invers
hasil_not_rect = cv2.bitwise_not(img_rect)
hasil_not_circle = cv2.bitwise_not(img_circle)
print("[INFO] NOT: Warna dibalik")

# ============================================================
# 6. Visualisasi operasi bitwise dasar
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img_rect, cmap="gray")
axes[0, 0].set_title("Persegi (A)")

axes[0, 1].imshow(img_circle, cmap="gray")
axes[0, 1].set_title("Lingkaran (B)")

axes[0, 2].imshow(hasil_and, cmap="gray")
axes[0, 2].set_title("A AND B\n(Irisan/Intersection)")

axes[1, 0].imshow(hasil_or, cmap="gray")
axes[1, 0].set_title("A OR B\n(Gabungan/Union)")

axes[1, 1].imshow(hasil_xor, cmap="gray")
axes[1, 1].set_title("A XOR B\n(Perbedaan Simetris)")

axes[1, 2].imshow(hasil_not_rect, cmap="gray")
axes[1, 2].set_title("NOT A\n(Invert Persegi)")

for ax in axes.flat:
    ax.axis("off")

plt.suptitle("Percobaan 6: Operasi Bitwise Dasar", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "06_bitwise_dasar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Bitwise dasar: {output_path}")

# ============================================================
# 7. Aplikasi: Overlay Logo menggunakan bitwise
# ============================================================
print("\n--- Aplikasi: Overlay Logo menggunakan Bitwise ---")

# Membaca gambar latar belakang
img_bg = cv2.imread(os.path.join(IMAGE_DIR, "pemandangan.jpg"))
if img_bg is None:
    print("[ERROR] Gambar pemandangan tidak ditemukan!")
    exit()

# Membaca gambar logo (dibuat sebagai lingkaran dengan teks)
img_logo = cv2.imread(os.path.join(IMAGE_DIR, "logo_mask.png"))

# Mengubah ukuran logo agar lebih kecil dari background
# Mengambil 1/4 lebar background sebagai lebar logo
lebar_logo = img_bg.shape[1] // 4
tinggi_logo = int(img_logo.shape[0] * lebar_logo / img_logo.shape[1])
img_logo = cv2.resize(img_logo, (lebar_logo, tinggi_logo))

# Mengkonversi logo ke grayscale untuk membuat mask
logo_gray = cv2.cvtColor(img_logo, cv2.COLOR_BGR2GRAY)

# Membuat mask binary dari logo (putih = ada logo, hitam = tidak)
# cv2.threshold() mengkonversi ke binary: piksel > 10 → putih, sisanya hitam
_, mask_logo = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)

# Membuat mask invers (kebalikan dari mask_logo)
mask_logo_inv = cv2.bitwise_not(mask_logo)

# Menentukan posisi overlay (kanan atas)
y_offset = 10
x_offset = img_bg.shape[1] - lebar_logo - 10

# Mengambil Region of Interest (ROI) dari background di posisi logo
roi = img_bg[y_offset:y_offset + tinggi_logo, x_offset:x_offset + lebar_logo]

# LANGKAH 1: Hapus area logo dari ROI menggunakan AND + mask invers
# Area di mana logo akan ditaruh menjadi hitam
bg_area = cv2.bitwise_and(roi, roi, mask=mask_logo_inv)

# LANGKAH 2: Ambil hanya area logo menggunakan AND + mask
logo_area = cv2.bitwise_and(img_logo, img_logo, mask=mask_logo)

# LANGKAH 3: Gabungkan background (tanpa area logo) + logo
kombinasi = cv2.add(bg_area, logo_area)

# Memasukkan hasil gabungan kembali ke gambar background
img_hasil = img_bg.copy()
img_hasil[y_offset:y_offset + tinggi_logo, x_offset:x_offset + lebar_logo] = kombinasi

print("  Logo berhasil di-overlay ke gambar background!")

# Visualisasi proses overlay
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

axes2[0, 0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title("1. ROI (Area target)")

axes2[0, 1].imshow(mask_logo, cmap="gray")
axes2[0, 1].set_title("2. Mask Logo")

axes2[0, 2].imshow(mask_logo_inv, cmap="gray")
axes2[0, 2].set_title("3. Mask Invers")

axes2[1, 0].imshow(cv2.cvtColor(bg_area, cv2.COLOR_BGR2RGB))
axes2[1, 0].set_title("4. BG tanpa area logo\n(AND + mask invers)")

axes2[1, 1].imshow(cv2.cvtColor(logo_area, cv2.COLOR_BGR2RGB))
axes2[1, 1].set_title("5. Logo saja\n(AND + mask)")

axes2[1, 2].imshow(cv2.cvtColor(img_hasil, cv2.COLOR_BGR2RGB))
axes2[1, 2].set_title("6. Hasil Akhir\n(Background + Logo)")

for ax in axes2.flat:
    ax.axis("off")

plt.suptitle("Proses Overlay Logo dengan Bitwise Operations", fontsize=14, fontweight="bold")
plt.tight_layout()

output_path2 = os.path.join(OUTPUT_DIR, "06_overlay_logo_hasil.png")
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Overlay logo: {output_path2}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 6")
print("=" * 60)
print("Operasi Bitwise:")
print("  1. AND : Mengambil irisan (piksel ON di kedua gambar)")
print("  2. OR  : Mengambil gabungan (piksel ON di salah satu)")
print("  3. XOR : Mengambil perbedaan (piksel ON di satu saja)")
print("  4. NOT : Membalik semua piksel")
print("\nAplikasi utama:")
print("  - Membuat dan menerapkan mask")
print("  - Overlay logo/watermark ke gambar")
print("  - Menggabungkan gambar secara selektif")
print("  - Operasi set (union, intersection, difference)")
print("=" * 60)
