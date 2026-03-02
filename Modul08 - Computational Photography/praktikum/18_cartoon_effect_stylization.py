"""
==========================================================================
PERCOBAAN 18: CARTOON EFFECT DAN STYLIZATION
==========================================================================
Program ini mempelajari teknik membuat efek kartun dan stilisasi pada
gambar foto. Beberapa metode digunakan baik otomatis (built-in OpenCV)
maupun manual (kombinasi filter).

Metode yang dipelajari:
1. cv2.stylization() - stilisasi otomatis seperti lukisan
2. Manual cartoon: bilateral filter + edge detection + combine
3. cv2.edgePreservingFilter() - smoothing yang menjaga tepi
4. cv2.detailEnhance() - peningkatan detail gambar

Fungsi utama yang dipelajari:
- cv2.stylization(src, sigma_s, sigma_r)          : Stilisasi otomatis
- cv2.edgePreservingFilter(src, flags, sigma_s, sigma_r) : Filter edge-preserving
- cv2.detailEnhance(src, sigma_s, sigma_r)         : Peningkatan detail
- cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace) : Bilateral filter
- cv2.adaptiveThreshold()                          : Edge detection untuk kartun

Hasil: Galeri efek artistik dari berbagai metode
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
print("PERCOBAAN 18: CARTOON EFFECT & STYLIZATION")
print("=" * 60)

# ============================================================
# 1. Membaca gambar input
# ============================================================

# Mendefinisikan path gambar portrait
path_img = os.path.join(IMAGE_DIR, "portrait.png")

# Membaca gambar dalam format BGR
img = cv2.imread(path_img)

# Memeriksa apakah gambar portrait berhasil dimuat; jika tidak, coba scene_pemandangan
if img is None:
    path_alt = os.path.join(IMAGE_DIR, "scene_pemandangan.png")
    img = cv2.imread(path_alt)

# Jika masih tidak ditemukan, download otomatis
if img is None:
    print("[WARN] Gambar tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    img = cv2.imread(path_img)
    if img is None:
        img = cv2.imread(os.path.join(IMAGE_DIR, "scene_pemandangan.png"))
if img is None:
    raise FileNotFoundError(
        "[ERROR] portrait.png / scene_pemandangan.png tidak tersedia.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )

# Menampilkan informasi gambar
print(f"[INFO] Ukuran gambar: {img.shape}")

# ============================================================
# 2. Metode 1: cv2.stylization()
# ============================================================
print("\n[LANGKAH 1] Menerapkan cv2.stylization()...")

# Menerapkan stilisasi dengan parameter berbeda
style_1 = cv2.stylization(img, sigma_s=60, sigma_r=0.07)

# Menerapkan stilisasi dengan parameter lebih kuat
style_2 = cv2.stylization(img, sigma_s=100, sigma_r=0.15)

# Menerapkan stilisasi dengan sigma_s kecil
style_3 = cv2.stylization(img, sigma_s=30, sigma_r=0.05)

# Menampilkan informasi
print(f"  Stylization selesai dengan 3 variasi parameter")

# ============================================================
# 3. Metode 2: Cartoon Manual (Bilateral + Edge)
# ============================================================
print("\n[LANGKAH 2] Membuat efek kartun secara manual...")

def cartoon_effect(image, num_bilateral=7, block_size=9):
    """
    Membuat efek kartun dari foto.
    Langkah: bilateral filter (smoothing) → edge detection → combine
    """
    # Mengonversi gambar ke grayscale untuk deteksi tepi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Menerapkan median blur untuk mengurangi noise pada edge
    gray_blur = cv2.medianBlur(gray, 7)

    # Mendeteksi tepi menggunakan adaptive threshold
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=2
    )

    # Menerapkan bilateral filter berulang kali untuk smoothing warna
    color = image.copy()
    for i in range(num_bilateral):
        # Bilateral filter mempertahankan tepi sambil menghaluskan area flat
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    # Mengonversi edges ke 3 channel agar bisa di-AND dengan gambar berwarna
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Menggabungkan warna smoothed dengan tepi menggunakan bitwise AND
    cartoon = cv2.bitwise_and(color, edges_3ch)

    # Mengembalikan hasil kartun dan komponen-komponennya
    return cartoon, edges, color

# Menerapkan efek kartun manual
cartoon_result, edges_result, smoothed_result = cartoon_effect(img)

# Menampilkan informasi
print(f"  Cartoon manual selesai")

# ============================================================
# 4. Visualisasi langkah kartun manual
# ============================================================
print("\n[LANGKAH 3] Visualisasi langkah-langkah kartun manual...")

# Membuat figure untuk langkah manual
fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))

# Menampilkan gambar original
axes1[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes1[0].set_title("1. Original", fontsize=11)
axes1[0].axis("off")

# Menampilkan edge map
axes1[1].imshow(edges_result, cmap='gray')
axes1[1].set_title("2. Edge Detection\n(Adaptive Threshold)", fontsize=10)
axes1[1].axis("off")

# Menampilkan smoothed color
axes1[2].imshow(cv2.cvtColor(smoothed_result, cv2.COLOR_BGR2RGB))
axes1[2].set_title("3. Bilateral Smoothed\n(7 iterasi)", fontsize=10)
axes1[2].axis("off")

# Menampilkan hasil kartun
axes1[3].imshow(cv2.cvtColor(cartoon_result, cv2.COLOR_BGR2RGB))
axes1[3].set_title("4. Cartoon Result\n(Smooth + Edge)", fontsize=10)
axes1[3].axis("off")

# Menambahkan judul utama
fig1.suptitle("Langkah Pembuatan Efek Kartun Manual", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path1 = os.path.join(OUTPUT_DIR, "18_cartoon_steps.png")
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path1}")

# Menutup figure
plt.show()
plt.show()
plt.show()
plt.close(fig1)

# ============================================================
# 5. Metode 3: cv2.edgePreservingFilter()
# ============================================================
print("\n[LANGKAH 4] Menerapkan cv2.edgePreservingFilter()...")

# Menerapkan edge-preserving filter dengan flag RECURS_FILTER
ep_recursive = cv2.edgePreservingFilter(img, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)

# Menerapkan edge-preserving filter dengan flag NORMCONV_FILTER
ep_normconv = cv2.edgePreservingFilter(img, flags=cv2.NORMCONV_FILTER, sigma_s=60, sigma_r=0.4)

# Menampilkan informasi
print(f"  Edge-preserving filter selesai (2 mode)")

# ============================================================
# 6. Metode 4: cv2.detailEnhance()
# ============================================================
print("\n[LANGKAH 5] Menerapkan cv2.detailEnhance()...")

# Menerapkan detail enhance dengan parameter berbeda
detail_1 = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

# Menerapkan detail enhance dengan sigma_s lebih besar
detail_2 = cv2.detailEnhance(img, sigma_s=50, sigma_r=0.15)

# Menampilkan informasi
print(f"  Detail enhance selesai (2 variasi)")

# ============================================================
# 7. Galeri semua efek artistik
# ============================================================
print("\n[LANGKAH 6] Membuat galeri semua efek artistik...")

# Membuat figure galeri besar
fig2, axes2 = plt.subplots(3, 3, figsize=(18, 18))

# Menampilkan original
axes2[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title("Original", fontsize=12, fontweight='bold')
axes2[0, 0].axis("off")

# Menampilkan stylization 1
axes2[0, 1].imshow(cv2.cvtColor(style_1, cv2.COLOR_BGR2RGB))
axes2[0, 1].set_title("Stylization\n(σs=60, σr=0.07)", fontsize=10)
axes2[0, 1].axis("off")

# Menampilkan stylization 2
axes2[0, 2].imshow(cv2.cvtColor(style_2, cv2.COLOR_BGR2RGB))
axes2[0, 2].set_title("Stylization\n(σs=100, σr=0.15)", fontsize=10)
axes2[0, 2].axis("off")

# Menampilkan cartoon manual
axes2[1, 0].imshow(cv2.cvtColor(cartoon_result, cv2.COLOR_BGR2RGB))
axes2[1, 0].set_title("Cartoon Manual\n(Bilateral+Edge)", fontsize=10)
axes2[1, 0].axis("off")

# Menampilkan edge-preserving recursive
axes2[1, 1].imshow(cv2.cvtColor(ep_recursive, cv2.COLOR_BGR2RGB))
axes2[1, 1].set_title("Edge Preserving\n(Recursive)", fontsize=10)
axes2[1, 1].axis("off")

# Menampilkan edge-preserving normconv
axes2[1, 2].imshow(cv2.cvtColor(ep_normconv, cv2.COLOR_BGR2RGB))
axes2[1, 2].set_title("Edge Preserving\n(NormConv)", fontsize=10)
axes2[1, 2].axis("off")

# Menampilkan detail enhance 1
axes2[2, 0].imshow(cv2.cvtColor(detail_1, cv2.COLOR_BGR2RGB))
axes2[2, 0].set_title("Detail Enhance\n(σs=10, σr=0.15)", fontsize=10)
axes2[2, 0].axis("off")

# Menampilkan detail enhance 2
axes2[2, 1].imshow(cv2.cvtColor(detail_2, cv2.COLOR_BGR2RGB))
axes2[2, 1].set_title("Detail Enhance\n(σs=50, σr=0.15)", fontsize=10)
axes2[2, 1].axis("off")

# Menampilkan stylization ringan
axes2[2, 2].imshow(cv2.cvtColor(style_3, cv2.COLOR_BGR2RGB))
axes2[2, 2].set_title("Stylization\n(σs=30, σr=0.05)", fontsize=10)
axes2[2, 2].axis("off")

# Menambahkan judul utama
fig2.suptitle("Galeri Efek Artistik - Computational Photography", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path2 = os.path.join(OUTPUT_DIR, "18_galeri_efek_artistik.png")
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path2}")

# Menutup figure
plt.show()
plt.show()
plt.show()
plt.close(fig2)

# ============================================================
# 8. Variasi jumlah iterasi bilateral filter
# ============================================================
print("\n[LANGKAH 7] Variasi iterasi bilateral pada kartun...")

# Mendefinisikan jumlah iterasi yang akan diuji
iterations = [1, 3, 5, 7, 10, 15]

# Membuat figure untuk variasi iterasi
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))

# Menerapkan dan menampilkan setiap variasi
for idx, n_iter in enumerate(iterations):
    # Menghitung posisi subplot
    row = idx // 3
    col = idx % 3

    # Menerapkan efek kartun dengan variasi iterasi
    cart, _, _ = cartoon_effect(img, num_bilateral=n_iter)

    # Menampilkan hasil
    axes3[row, col].imshow(cv2.cvtColor(cart, cv2.COLOR_BGR2RGB))
    axes3[row, col].set_title(f"Bilateral × {n_iter} iterasi", fontsize=11)
    axes3[row, col].axis("off")

# Menambahkan judul utama
fig3.suptitle("Efek Cartoon: Variasi Jumlah Iterasi Bilateral Filter", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path3 = os.path.join(OUTPUT_DIR, "18_variasi_iterasi.png")
fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path3}")

# Menutup figure
plt.show()
plt.show()
plt.show()
plt.close(fig3)

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 18: CARTOON EFFECT & STYLIZATION")
print("=" * 60)
print("Fungsi-fungsi OpenCV yang dipelajari:")
print("1. cv2.stylization(src, sigma_s, sigma_r)")
print("   → Stilisasi otomatis seperti lukisan/cat air")
print("2. cv2.edgePreservingFilter(src, flags, sigma_s, sigma_r)")
print("   → Smoothing yang menjaga struktur tepi")
print("   → RECURS_FILTER: recursive bilateral")
print("   → NORMCONV_FILTER: normalized convolution")
print("3. cv2.detailEnhance(src, sigma_s, sigma_r)")
print("   → Meningkatkan detail tekstur gambar")
print("4. cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)")
print("   → Smoothing warna sambil menjaga tepi untuk kartun")
print("5. cv2.adaptiveThreshold()")
print("   → Deteksi tepi untuk garis kartun")
print("6. cv2.bitwise_and()")
print("   → Menggabungkan warna dan tepi")
print()
print("Kesimpulan:")
print("- cv2.stylization() mudah untuk efek lukisan")
print("- Kartun manual lebih fleksibel dan dapat dikontrol per langkah")
print("- Bilateral filter lebih banyak iterasi → lebih halus")
print("- Kombinasi berbagai filter menghasilkan efek artistik unik")
print("=" * 60)
