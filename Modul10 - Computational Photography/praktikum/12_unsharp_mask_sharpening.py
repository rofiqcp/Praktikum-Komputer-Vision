"""
==========================================================================
PERCOBAAN 12: IMAGE SHARPENING DENGAN UNSHARP MASK
==========================================================================
Program ini mempelajari teknik penajaman gambar (image sharpening)
menggunakan metode Unsharp Mask. Prinsipnya: detail gambar diperkuat
dengan menambahkan perbedaan antara gambar asli dan versi blur-nya.

Formula Unsharp Mask:
    sharpened = original + amount * (original - blurred)

Metode yang dipelajari:
1. Unsharp Mask manual dengan GaussianBlur
2. cv2.addWeighted() untuk blending
3. cv2.filter2D() dengan custom sharpening kernel

Fungsi utama yang dipelajari:
- cv2.GaussianBlur(src, ksize, sigmaX)  : Membuat versi blur
- cv2.addWeighted(src1, a, src2, b, g)  : Blending dua gambar
- cv2.filter2D(src, ddepth, kernel)      : Konvolusi dengan kernel kustom
- np.clip()                               : Membatasi nilai piksel

Hasil: Perbandingan berbagai parameter sharpening dan metode kernel
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
print("PERCOBAAN 12: UNSHARP MASK SHARPENING")
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
# 2. Fungsi Unsharp Mask
# ============================================================

def unsharp_mask(image, sigma=1.0, amount=1.0):
    """Menerapkan Unsharp Mask: sharpened = orig + amount*(orig - blurred)."""
    # Menentukan ukuran kernel berdasarkan sigma (harus ganjil)
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    # Membuat versi blur dari gambar menggunakan Gaussian Blur
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # Mengonversi ke float untuk menghindari overflow saat kalkulasi
    img_float = image.astype(np.float64)

    # Mengonversi blur ke float juga
    blur_float = blurred.astype(np.float64)

    # Menghitung komponen detail (perbedaan original dan blur)
    detail = img_float - blur_float

    # Menambahkan detail yang diperkuat ke gambar asli
    sharpened = img_float + amount * detail

    # Membatasi nilai piksel ke rentang 0-255
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Mengembalikan gambar yang sudah ditajamkan
    return sharpened

# ============================================================
# 3. Variasi amount (kekuatan sharpening)
# ============================================================
print("\n[LANGKAH 1] Variasi amount (kekuatan sharpening)...")

# Mendefinisikan daftar nilai amount yang akan diuji
amounts = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

# Membuat figure untuk variasi amount
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))

# Melakukan iterasi untuk setiap nilai amount
for idx, amt in enumerate(amounts):
    # Menghitung posisi baris dan kolom
    row = idx // 3
    col = idx % 3

    # Menerapkan unsharp mask dengan amount tertentu dan sigma tetap
    result = unsharp_mask(img, sigma=1.5, amount=amt)

    # Menampilkan hasil pada subplot
    axes1[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes1[row, col].set_title(f"Amount = {amt} (σ=1.5)", fontsize=11)
    axes1[row, col].axis("off")

# Menambahkan judul utama
fig1.suptitle("Variasi Amount pada Unsharp Mask", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path1 = os.path.join(OUTPUT_DIR, "12_variasi_amount.png")
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path1}")

# Menutup figure
plt.close(fig1)

# ============================================================
# 4. Variasi sigma (radius blur)
# ============================================================
print("\n[LANGKAH 2] Variasi sigma (radius blur)...")

# Mendefinisikan daftar nilai sigma yang akan diuji
sigmas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

# Membuat figure untuk variasi sigma
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

# Melakukan iterasi untuk setiap nilai sigma
for idx, sig in enumerate(sigmas):
    # Menghitung posisi baris dan kolom
    row = idx // 3
    col = idx % 3

    # Menerapkan unsharp mask dengan sigma tertentu dan amount tetap
    result = unsharp_mask(img, sigma=sig, amount=1.5)

    # Menampilkan hasil pada subplot
    axes2[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes2[row, col].set_title(f"Sigma = {sig} (amount=1.5)", fontsize=11)
    axes2[row, col].axis("off")

# Menambahkan judul utama
fig2.suptitle("Variasi Sigma pada Unsharp Mask", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path2 = os.path.join(OUTPUT_DIR, "12_variasi_sigma.png")
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path2}")

# Menutup figure
plt.close(fig2)

# ============================================================
# 5. Sharpening dengan cv2.addWeighted()
# ============================================================
print("\n[LANGKAH 3] Sharpening dengan cv2.addWeighted()...")

# Membuat versi blur untuk addWeighted
blurred_for_weighted = cv2.GaussianBlur(img, (0, 0), 3)

# Menerapkan addWeighted: 1.5*original - 0.5*blurred = sharpened
sharp_weighted = cv2.addWeighted(img, 1.5, blurred_for_weighted, -0.5, 0)

# Menerapkan addWeighted dengan faktor lebih kuat
sharp_weighted_strong = cv2.addWeighted(img, 2.0, blurred_for_weighted, -1.0, 0)

# ============================================================
# 6. Sharpening dengan kernel kustom (cv2.filter2D)
# ============================================================
print("\n[LANGKAH 4] Sharpening dengan cv2.filter2D() dan kernel kustom...")

# Mendefinisikan kernel sharpening dasar 3x3
kernel_basic = np.array([
    [0, -1, 0],
    [-1,  5, -1],
    [0, -1, 0]
], dtype=np.float32)

# Mendefinisikan kernel sharpening dengan diagonal (lebih kuat)
kernel_strong = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
], dtype=np.float32)

# Mendefinisikan kernel Laplacian sharpening
kernel_laplacian = np.array([
    [0,  1,  0],
    [1, -4,  1],
    [0,  1,  0]
], dtype=np.float32)

# Menerapkan filter2D dengan kernel dasar
sharp_basic = cv2.filter2D(img, -1, kernel_basic)

# Menerapkan filter2D dengan kernel kuat
sharp_strong = cv2.filter2D(img, -1, kernel_strong)

# Menerapkan Laplacian sharpening: original - laplacian
laplacian_detail = cv2.filter2D(img, cv2.CV_64F, kernel_laplacian)

# Mengurangi detail Laplacian dari gambar asli untuk penajaman
sharp_laplacian = np.clip(img.astype(np.float64) - laplacian_detail, 0, 255).astype(np.uint8)

# ============================================================
# 7. Perbandingan semua metode sharpening
# ============================================================
print("\n[LANGKAH 5] Membuat perbandingan semua metode...")

# Membuat figure untuk perbandingan semua metode
fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))

# Menampilkan gambar original
axes3[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes3[0, 0].set_title("Original", fontsize=11)
axes3[0, 0].axis("off")

# Menampilkan hasil Unsharp Mask
result_um = unsharp_mask(img, sigma=1.5, amount=1.5)
axes3[0, 1].imshow(cv2.cvtColor(result_um, cv2.COLOR_BGR2RGB))
axes3[0, 1].set_title("Unsharp Mask\n(σ=1.5, amt=1.5)", fontsize=10)
axes3[0, 1].axis("off")

# Menampilkan hasil addWeighted
axes3[0, 2].imshow(cv2.cvtColor(sharp_weighted, cv2.COLOR_BGR2RGB))
axes3[0, 2].set_title("addWeighted\n(1.5, -0.5)", fontsize=10)
axes3[0, 2].axis("off")

# Menampilkan hasil addWeighted kuat
axes3[0, 3].imshow(cv2.cvtColor(sharp_weighted_strong, cv2.COLOR_BGR2RGB))
axes3[0, 3].set_title("addWeighted\n(2.0, -1.0)", fontsize=10)
axes3[0, 3].axis("off")

# Menampilkan hasil kernel basic
axes3[1, 0].imshow(cv2.cvtColor(sharp_basic, cv2.COLOR_BGR2RGB))
axes3[1, 0].set_title("Kernel Basic\n[[0,-1,0],[-1,5,-1],[0,-1,0]]", fontsize=9)
axes3[1, 0].axis("off")

# Menampilkan hasil kernel strong
axes3[1, 1].imshow(cv2.cvtColor(sharp_strong, cv2.COLOR_BGR2RGB))
axes3[1, 1].set_title("Kernel Strong\n[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]", fontsize=9)
axes3[1, 1].axis("off")

# Menampilkan hasil Laplacian sharpening
axes3[1, 2].imshow(cv2.cvtColor(sharp_laplacian, cv2.COLOR_BGR2RGB))
axes3[1, 2].set_title("Laplacian Sharpen", fontsize=10)
axes3[1, 2].axis("off")

# Menyembunyikan subplot kosong
axes3[1, 3].axis("off")

# Menambahkan judul utama
fig3.suptitle("Perbandingan Semua Metode Sharpening", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path3 = os.path.join(OUTPUT_DIR, "12_perbandingan_metode.png")
fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path3}")

# Menutup figure
plt.close(fig3)

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 12: UNSHARP MASK SHARPENING")
print("=" * 60)
print("Fungsi-fungsi OpenCV yang dipelajari:")
print("1. cv2.GaussianBlur(src, ksize, sigmaX)")
print("   → Membuat versi blur untuk proses Unsharp Mask")
print("2. cv2.addWeighted(src1, alpha, src2, beta, gamma)")
print("   → Blending dua gambar: dst = alpha*src1 + beta*src2 + gamma")
print("3. cv2.filter2D(src, ddepth, kernel)")
print("   → Konvolusi gambar dengan kernel kustom sharpening")
print("4. np.clip(array, min, max)")
print("   → Membatasi nilai piksel agar tetap dalam rentang valid")
print()
print("Kesimpulan:")
print("- Unsharp Mask: sharpened = orig + amount*(orig - blur)")
print("- Amount mengontrol kekuatan penajaman")
print("- Sigma mengontrol radius detail yang ditajamkan")
print("- Kernel 3x3 memberikan penajaman cepat dan sederhana")
print("- Amount/kekuatan terlalu besar → halo dan artefak muncul")
print("=" * 60)
