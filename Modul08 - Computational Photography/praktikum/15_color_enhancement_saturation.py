"""
==========================================================================
PERCOBAAN 15: COLOR ENHANCEMENT DAN SATURATION
==========================================================================
Program ini mempelajari teknik peningkatan warna dan saturasi gambar.
Saturasi mengontrol seberapa "kaya" atau "pucat" warna dalam gambar.
Program ini mengeksplorasi manipulasi di ruang warna HSV dan
teknik vibrance yang lebih cerdas.

Teknik yang dipelajari:
1. Saturasi global - mengalikan channel S di HSV
2. Vibrance - boost warna yang kurang saturated lebih banyak
3. Color temperature shifting - mengubah kesan hangat/dingin

Fungsi utama yang dipelajari:
- cv2.cvtColor(src, cv2.COLOR_BGR2HSV) : Konversi BGR ke HSV
- cv2.split() / cv2.merge()            : Memisahkan/gabung channel
- np.clip()                            : Membatasi rentang nilai
- cv2.addWeighted()                    : Blending gambar

Hasil: Perbandingan visual berbagai tingkat saturasi dan vibrance
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
print("PERCOBAAN 15: COLOR ENHANCEMENT & SATURATION")
print("=" * 60)

# ============================================================
# 1. Membaca gambar input
# ============================================================

# Mendefinisikan path gambar pemandangan
path_img = os.path.join(IMAGE_DIR, "scene_pemandangan.png")

# Membaca gambar dalam format BGR
img = cv2.imread(path_img)

# Memeriksa apakah gambar berhasil dimuat; jika tidak, download otomatis
if img is None:
    print("[WARN] scene_pemandangan.png tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    img = cv2.imread(path_img)
if img is None:
    raise FileNotFoundError(
        "[ERROR] scene_pemandangan.png tidak tersedia setelah download.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )

# Menampilkan informasi dimensi gambar
print(f"[INFO] Ukuran gambar: {img.shape}")

# ============================================================
# 2. Saturasi Global via HSV
# ============================================================
print("\n[LANGKAH 1] Variasi saturasi global di ruang warna HSV...")

def adjust_saturation(image, factor):
    """Mengatur saturasi gambar dengan mengalikan channel S di HSV."""
    # Mengonversi gambar dari BGR ke HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)

    # Memisahkan channel H, S, V
    h, s, v = cv2.split(hsv)

    # Mengalikan channel saturation dengan faktor
    s = s * factor

    # Membatasi nilai S ke rentang 0-255
    s = np.clip(s, 0, 255)

    # Menggabungkan kembali channel H, S, V
    hsv_adjusted = cv2.merge([h, s, v]).astype(np.uint8)

    # Mengonversi kembali ke BGR
    result = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    # Mengembalikan hasil
    return result

# Mendefinisikan daftar faktor saturasi yang akan diuji
sat_factors = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]

# Membuat figure untuk variasi saturasi
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))

# Menampilkan setiap variasi saturasi
for idx, sf in enumerate(sat_factors):
    # Menghitung posisi subplot
    row = idx // 3
    col = idx % 3

    # Menerapkan penyesuaian saturasi
    result = adjust_saturation(img, sf)

    # Menampilkan hasil
    axes1[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes1[row, col].set_title(f"Saturasi × {sf}", fontsize=11)
    axes1[row, col].axis("off")

# Menambahkan judul utama
fig1.suptitle("Variasi Saturasi Global (HSV)", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path1 = os.path.join(OUTPUT_DIR, "15_variasi_saturasi.png")
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path1}")

# Menutup figure
plt.show()
plt.show()
plt.show()
plt.close(fig1)

# ============================================================
# 3. Vibrance (Selective Saturation Boost)
# ============================================================
print("\n[LANGKAH 2] Menerapkan Vibrance (boost selektif)...")

def apply_vibrance(image, intensity=1.5):
    """
    Vibrance: boost warna yang kurang saturated lebih banyak,
    sementara warna yang sudah saturated hanya sedikit diubah.
    """
    # Mengonversi gambar dari BGR ke HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)

    # Memisahkan channel H, S, V
    h, s, v = cv2.split(hsv)

    # Menghitung faktor boost berdasarkan saturasi saat ini
    # Semakin rendah saturasi → semakin besar boost
    max_sat = 255.0
    # Menghitung seberapa "kurang saturated" setiap piksel (0-1)
    low_sat_factor = 1.0 - (s / max_sat)

    # Menghitung boost: piksel kurang saturated mendapat boost lebih
    boost = 1.0 + (intensity - 1.0) * low_sat_factor

    # Menerapkan boost pada channel saturation
    s_vibrance = s * boost

    # Membatasi ke rentang 0-255
    s_vibrance = np.clip(s_vibrance, 0, 255)

    # Menggabungkan kembali
    hsv_result = cv2.merge([h, s_vibrance, v]).astype(np.uint8)

    # Mengonversi kembali ke BGR
    result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)

    # Mengembalikan hasil
    return result

# Mendefinisikan daftar intensitas vibrance
vibrance_levels = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

# Membuat figure untuk variasi vibrance
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

# Menampilkan setiap variasi vibrance
for idx, vl in enumerate(vibrance_levels):
    # Menghitung posisi subplot
    row = idx // 3
    col = idx % 3

    # Menerapkan vibrance
    result = apply_vibrance(img, intensity=vl)

    # Menampilkan hasil
    axes2[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes2[row, col].set_title(f"Vibrance = {vl}", fontsize=11)
    axes2[row, col].axis("off")

# Menambahkan judul utama
fig2.suptitle("Variasi Vibrance (Selective Saturation Boost)", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path2 = os.path.join(OUTPUT_DIR, "15_variasi_vibrance.png")
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path2}")

# Menutup figure
plt.show()
plt.show()
plt.show()
plt.close(fig2)

# ============================================================
# 4. Perbandingan Saturasi vs Vibrance
# ============================================================
print("\n[LANGKAH 3] Perbandingan Saturasi vs Vibrance...")

# Menerapkan saturasi 2x
sat_2x = adjust_saturation(img, 2.0)

# Menerapkan vibrance 2x
vib_2x = apply_vibrance(img, 2.0)

# Membuat figure perbandingan
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

# Menampilkan original
axes3[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes3[0].set_title("Original", fontsize=12)
axes3[0].axis("off")

# Menampilkan saturasi 2x
axes3[1].imshow(cv2.cvtColor(sat_2x, cv2.COLOR_BGR2RGB))
axes3[1].set_title("Saturasi × 2.0", fontsize=12)
axes3[1].axis("off")

# Menampilkan vibrance 2x
axes3[2].imshow(cv2.cvtColor(vib_2x, cv2.COLOR_BGR2RGB))
axes3[2].set_title("Vibrance = 2.0", fontsize=12)
axes3[2].axis("off")

# Menambahkan judul utama
fig3.suptitle("Perbandingan: Saturasi Global vs Vibrance", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path3 = os.path.join(OUTPUT_DIR, "15_sat_vs_vibrance.png")
fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path3}")

# Menutup figure
plt.show()
plt.show()
plt.show()
plt.close(fig3)

# ============================================================
# 5. Color Temperature Shift via HSV
# ============================================================
print("\n[LANGKAH 4] Mengubah suhu warna via HSV...")

def shift_hue(image, shift_deg):
    """Menggeser hue gambar (dalam derajat, -180 sampai 180)."""
    # Mengonversi BGR ke HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)

    # Menggeser channel H (dalam OpenCV, H = 0-179)
    # Konversi shift dari derajat ke skala OpenCV (0-179)
    shift_val = int(shift_deg / 2)

    # Menerapkan shift pada channel H
    hsv[:, :, 0] = (hsv[:, :, 0] + shift_val) % 180

    # Mengonversi kembali ke uint8
    hsv = hsv.astype(np.uint8)

    # Mengonversi kembali ke BGR
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Mengembalikan hasil
    return result

# Mendefinisikan variasi pergeseran hue
hue_shifts = [-30, -15, 0, 15, 30, 60]

# Membuat figure untuk variasi hue shift
fig4, axes4 = plt.subplots(2, 3, figsize=(15, 10))

# Menampilkan setiap variasi
for idx, hs in enumerate(hue_shifts):
    # Menghitung posisi subplot
    row = idx // 3
    col = idx % 3

    # Menerapkan hue shift
    result = shift_hue(img, hs)

    # Menampilkan hasil
    axes4[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    label = "Original" if hs == 0 else f"Hue Shift: {hs:+d}°"
    axes4[row, col].set_title(label, fontsize=11)
    axes4[row, col].axis("off")

# Menambahkan judul utama
fig4.suptitle("Variasi Hue Shift (Color Temperature)", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path4 = os.path.join(OUTPUT_DIR, "15_hue_shift.png")
fig4.savefig(output_path4, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path4}")

# Menutup figure
plt.show()
plt.show()
plt.show()
plt.close(fig4)

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 15: COLOR ENHANCEMENT & SATURATION")
print("=" * 60)
print("Fungsi-fungsi yang dipelajari:")
print("1. cv2.cvtColor(src, cv2.COLOR_BGR2HSV)")
print("   → Konversi ke ruang warna HSV untuk manipulasi warna")
print("2. cv2.split() dan cv2.merge()")
print("   → Memisahkan dan menggabungkan channel H, S, V")
print("3. Saturasi global: S_baru = S_lama × faktor")
print("   → Mengubah kekayaan warna secara seragam")
print("4. Vibrance: boost = 1 + (intensity-1) × (1 - S/255)")
print("   → Boost selektif: warna pucat di-boost lebih dari warna kaya")
print("5. Hue shift: H_baru = (H_lama + shift) mod 180")
print("   → Menggeser warna di roda warna")
print()
print("Kesimpulan:")
print("- Saturasi global: mudah tapi bisa oversaturate warna kaya")
print("- Vibrance: lebih natural, menghindari clipping warna")
print("- Hue shift: mengubah keseluruhan palet warna gambar")
print("- Kombinasi ketiganya menghasilkan color grading profesional")
print("=" * 60)
