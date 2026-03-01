"""
==========================================================================
PERCOBAAN 17: PENCIL SKETCH EFFECT
==========================================================================
Program ini mempelajari cara membuat efek sketsa pensil (pencil sketch)
dari foto. Dua pendekatan digunakan:

Metode 1 - Otomatis: cv2.pencilSketch()
    Langsung menghasilkan sketch grayscale dan berwarna

Metode 2 - Manual:
    1. Konversi ke grayscale
    2. Invert grayscale
    3. Gaussian Blur pada inverted
    4. Blend menggunakan Dodge (cv2.divide)
    Formula: sketch = (grayscale * 256) / (256 - blurred_inv)

Fungsi utama yang dipelajari:
- cv2.pencilSketch(src, sigma_s, sigma_r, shade_factor) : Sketch otomatis
- cv2.divide(src1, src2, scale)                         : Operasi divide (dodge blend)
- cv2.bitwise_not(src)                                  : Invert gambar
- cv2.GaussianBlur()                                    : Blur untuk dodge

Hasil: Perbandingan metode otomatis vs manual dan variasi parameter
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
print("PERCOBAAN 17: PENCIL SKETCH EFFECT")
print("=" * 60)

# ============================================================
# 1. Membaca gambar input
# ============================================================

# Mendefinisikan path gambar portrait (atau pemandangan)
path_img = os.path.join(IMAGE_DIR, "portrait.png")

# Membaca gambar dalam format BGR
img = cv2.imread(path_img)

# Memeriksa apakah gambar berhasil dimuat
if img is None:
    # Mencoba gambar alternatif
    path_alt = os.path.join(IMAGE_DIR, "scene_pemandangan.png")
    img = cv2.imread(path_alt)

# Jika masih tidak ditemukan, buat gambar sintetis
if img is None:
    print("[INFO] Gambar tidak ditemukan, membuat gambar sintetis...")
    # Membuat gambar sintetis dengan variasi tonalitas
    img = np.ones((400, 500, 3), dtype=np.uint8) * 200
    # Menambahkan beberapa bentuk
    cv2.circle(img, (250, 180), 80, (160, 140, 130), -1)
    cv2.rectangle(img, (80, 280), (420, 380), (130, 120, 110), -1)
    cv2.ellipse(img, (250, 180), (50, 70), 0, 0, 360, (180, 160, 150), -1)
    # Menambahkan variasi warna
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Menampilkan informasi gambar
print(f"[INFO] Ukuran gambar: {img.shape}")

# ============================================================
# 2. Metode 1: cv2.pencilSketch() - Otomatis
# ============================================================
print("\n[LANGKAH 1] Menerapkan cv2.pencilSketch() otomatis...")

# Menerapkan pencilSketch dengan parameter default
# sigma_s: range spatial (pixel neighborhood), sigma_r: range intensitas
# shade_factor: mengontrol kecerahan output (0-0.1)
sketch_gray, sketch_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)

# Menampilkan informasi hasil
print(f"  Sketch grayscale: {sketch_gray.shape}, dtype={sketch_gray.dtype}")
print(f"  Sketch berwarna: {sketch_color.shape}, dtype={sketch_color.dtype}")

# ============================================================
# 3. Variasi parameter pencilSketch
# ============================================================
print("\n[LANGKAH 2] Variasi parameter pencilSketch...")

# Mendefinisikan kombinasi parameter yang akan diuji
params_list = [
    {"sigma_s": 20, "sigma_r": 0.03, "shade_factor": 0.02},
    {"sigma_s": 60, "sigma_r": 0.07, "shade_factor": 0.05},
    {"sigma_s": 100, "sigma_r": 0.1, "shade_factor": 0.05},
    {"sigma_s": 60, "sigma_r": 0.15, "shade_factor": 0.08},
]

# Membuat figure untuk variasi parameter
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 10))

# Menampilkan variasi sketch grayscale di baris pertama
for idx, params in enumerate(params_list):
    # Menerapkan pencilSketch dengan parameter tertentu
    sk_gray, sk_color = cv2.pencilSketch(img, **params)

    # Menampilkan sketch grayscale di baris pertama
    axes1[0, idx].imshow(sk_gray, cmap='gray')
    axes1[0, idx].set_title(f"Gray: σs={params['sigma_s']}\nσr={params['sigma_r']}, shade={params['shade_factor']}", fontsize=9)
    axes1[0, idx].axis("off")

    # Menampilkan sketch berwarna di baris kedua
    axes1[1, idx].imshow(cv2.cvtColor(sk_color, cv2.COLOR_BGR2RGB))
    axes1[1, idx].set_title(f"Color: σs={params['sigma_s']}\nσr={params['sigma_r']}", fontsize=9)
    axes1[1, idx].axis("off")

# Menambahkan label baris
axes1[0, 0].set_ylabel("Grayscale Sketch", fontsize=12)
axes1[1, 0].set_ylabel("Color Sketch", fontsize=12)

# Menambahkan judul utama
fig1.suptitle("Variasi Parameter cv2.pencilSketch()", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path1 = os.path.join(OUTPUT_DIR, "17_pencilsketch_variasi.png")
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path1}")

# Menutup figure
plt.close(fig1)

# ============================================================
# 4. Metode 2: Sketch Manual (Dodge Blend)
# ============================================================
print("\n[LANGKAH 3] Membuat sketch secara manual (dodge blend)...")

def manual_pencil_sketch(image, blur_sigma=21):
    """Membuat efek pensil sketch secara manual dengan teknik dodge."""
    # Mengonversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Meng-invert gambar grayscale (hitam jadi putih, sebaliknya)
    inverted = cv2.bitwise_not(gray)

    # Menerapkan Gaussian Blur pada gambar inverted
    # Sigma besar → garis sketch lebih halus
    ksize = blur_sigma
    if ksize % 2 == 0:
        ksize += 1
    blurred_inv = cv2.GaussianBlur(inverted, (ksize, ksize), 0)

    # Menerapkan dodge blend: sketch = gray / (255 - blurred_inv) * 256
    # cv2.divide otomatis menangani pembagian dengan nol
    sketch = cv2.divide(gray, 255 - blurred_inv, scale=256.0)

    # Mengembalikan sketch grayscale
    return sketch, gray, inverted, blurred_inv

# Menerapkan sketch manual
sketch_manual, gray_step, inv_step, blur_step = manual_pencil_sketch(img, blur_sigma=21)

# Menampilkan informasi
print(f"  Sketch manual selesai (blur_sigma=21)")

# ============================================================
# 5. Visualisasi langkah-langkah metode manual
# ============================================================
print("\n[LANGKAH 4] Visualisasi langkah-langkah metode manual...")

# Membuat figure untuk langkah manual
fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))

# Menampilkan gambar original
axes2[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes2[0].set_title("1. Original", fontsize=11)
axes2[0].axis("off")

# Menampilkan grayscale
axes2[1].imshow(gray_step, cmap='gray')
axes2[1].set_title("2. Grayscale", fontsize=11)
axes2[1].axis("off")

# Menampilkan inverted
axes2[2].imshow(inv_step, cmap='gray')
axes2[2].set_title("3. Inverted", fontsize=11)
axes2[2].axis("off")

# Menampilkan blurred inverted
axes2[3].imshow(blur_step, cmap='gray')
axes2[3].set_title("4. Blurred Inv", fontsize=11)
axes2[3].axis("off")

# Menampilkan sketch hasil akhir
axes2[4].imshow(sketch_manual, cmap='gray')
axes2[4].set_title("5. Sketch (Dodge)", fontsize=11)
axes2[4].axis("off")

# Menambahkan judul utama
fig2.suptitle("Langkah-langkah Pembuatan Sketch Manual", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path2 = os.path.join(OUTPUT_DIR, "17_manual_sketch_steps.png")
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path2}")

# Menutup figure
plt.close(fig2)

# ============================================================
# 6. Variasi blur sigma pada metode manual
# ============================================================
print("\n[LANGKAH 5] Variasi blur sigma pada sketch manual...")

# Mendefinisikan nilai sigma yang akan diuji
manual_sigmas = [5, 11, 21, 41, 81, 151]

# Membuat figure untuk variasi sigma
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))

# Menampilkan setiap variasi
for idx, ms in enumerate(manual_sigmas):
    # Menghitung posisi subplot
    row = idx // 3
    col = idx % 3

    # Membuat sketch manual dengan sigma tertentu
    sk, _, _, _ = manual_pencil_sketch(img, blur_sigma=ms)

    # Menampilkan hasil
    axes3[row, col].imshow(sk, cmap='gray')
    axes3[row, col].set_title(f"Blur Sigma = {ms}", fontsize=11)
    axes3[row, col].axis("off")

# Menambahkan judul utama
fig3.suptitle("Variasi Blur Sigma pada Sketch Manual", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path3 = os.path.join(OUTPUT_DIR, "17_manual_variasi_sigma.png")
fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path3}")

# Menutup figure
plt.close(fig3)

# ============================================================
# 7. Perbandingan metode otomatis vs manual
# ============================================================
print("\n[LANGKAH 6] Perbandingan otomatis vs manual...")

# Membuat figure perbandingan
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

# Menampilkan original
axes4[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes4[0].set_title("Original", fontsize=12)
axes4[0].axis("off")

# Menampilkan sketch otomatis (pencilSketch)
axes4[1].imshow(sketch_gray, cmap='gray')
axes4[1].set_title("cv2.pencilSketch()\n(σs=60, σr=0.07)", fontsize=11)
axes4[1].axis("off")

# Menampilkan sketch manual (dodge blend)
axes4[2].imshow(sketch_manual, cmap='gray')
axes4[2].set_title("Manual Dodge Blend\n(blur_sigma=21)", fontsize=11)
axes4[2].axis("off")

# Menambahkan judul utama
fig4.suptitle("Perbandingan: cv2.pencilSketch() vs Manual Dodge", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path4 = os.path.join(OUTPUT_DIR, "17_otomatis_vs_manual.png")
fig4.savefig(output_path4, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path4}")

# Menutup figure
plt.close(fig4)

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 17: PENCIL SKETCH EFFECT")
print("=" * 60)
print("Fungsi-fungsi OpenCV yang dipelajari:")
print("1. cv2.pencilSketch(src, sigma_s, sigma_r, shade_factor)")
print("   → Menghasilkan sketch grayscale dan berwarna otomatis")
print("   → sigma_s: spatial neighborhood, sigma_r: intensitas range")
print("   → shade_factor: kecerahan output sketch")
print("2. cv2.bitwise_not(src)")
print("   → Meng-invert gambar (komplemen dari piksel)")
print("3. cv2.divide(src1, src2, scale)")
print("   → Operasi pembagian piksel (dodge blend)")
print("   → sketch = gray * 256 / (256 - blurred_inv)")
print("4. cv2.GaussianBlur(src, ksize, sigma)")
print("   → Blur pada inverted image untuk dodge blend")
print()
print("Kesimpulan:")
print("- cv2.pencilSketch() mudah tapi kurang fleksibel")
print("- Metode manual memberikan kontrol lebih atas parameter")
print("- Blur sigma besar → garis sketch lebih sedikit dan halus")
print("- Blur sigma kecil → lebih banyak detail tertangkap")
print("=" * 60)
