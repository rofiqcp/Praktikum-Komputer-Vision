"""
==========================================================================
PERCOBAAN 16: IMAGE ENHANCEMENT PIPELINE
==========================================================================
Program ini menerapkan pipeline peningkatan gambar lengkap yang
menggabungkan beberapa teknik secara berurutan:

Pipeline:
1. White Balance (Gray World) → Koreksi warna
2. Denoising (fastNlMeansDenoisingColored) → Hapus noise
3. CLAHE → Peningkatan kontras lokal
4. Sharpening (Unsharp Mask) → Penajaman detail
5. Color Boost (Vibrance) → Peningkatan warna

Setiap langkah ditampilkan secara berurutan sehingga terlihat
kontribusi masing-masing teknik terhadap hasil akhir.

Fungsi utama yang dipelajari:
- cv2.fastNlMeansDenoisingColored() : Denoising warna
- cv2.createCLAHE()                 : CLAHE untuk kontras
- cv2.GaussianBlur()                : Blur untuk Unsharp Mask
- cv2.cvtColor()                    : Konversi ruang warna

Hasil: Visualisasi setiap tahap pipeline dan perbandingan awal-akhir
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
print("PERCOBAAN 16: IMAGE ENHANCEMENT PIPELINE")
print("=" * 60)

# ============================================================
# 1. Membaca gambar gelap sebagai input pipeline
# ============================================================

# Mendefinisikan path gambar gelap
path_img = os.path.join(IMAGE_DIR, "gambar_gelap.png")

# Membaca gambar dalam format BGR
img = cv2.imread(path_img)

# Memeriksa apakah gambar berhasil dimuat; jika tidak, download otomatis
if img is None:
    print("[WARN] gambar_gelap.png tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    img = cv2.imread(path_img)
if img is None:
    raise FileNotFoundError(
        "[ERROR] gambar_gelap.png tidak tersedia setelah download.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )

# Menampilkan informasi gambar
print(f"[INFO] Ukuran gambar: {img.shape}")
print(f"[INFO] Rata-rata brightness: {np.mean(img):.1f}")

# Menyimpan gambar asli untuk perbandingan akhir
img_original = img.copy()

# Menyimpan daftar gambar setiap tahap pipeline
pipeline_stages = []
stage_names = []

# Menambahkan gambar asli sebagai tahap pertama
pipeline_stages.append(img.copy())
stage_names.append("0. Original")

# ============================================================
# 2. Tahap 1: White Balance (Gray World Assumption)
# ============================================================
print("\n[TAHAP 1] White Balance (Gray World)...")

def gray_world_wb(image):
    """Menerapkan koreksi white balance Gray World."""
    # Mengonversi ke float
    img_float = image.astype(np.float64)

    # Menghitung mean setiap channel
    mean_b = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_r = np.mean(img_float[:, :, 2])

    # Menghitung mean global
    mean_all = (mean_b + mean_g + mean_r) / 3.0

    # Menerapkan koreksi
    result = img_float.copy()
    result[:, :, 0] *= (mean_all / (mean_b + 1e-6))
    result[:, :, 1] *= (mean_all / (mean_g + 1e-6))
    result[:, :, 2] *= (mean_all / (mean_r + 1e-6))

    # Membatasi dan mengembalikan
    return np.clip(result, 0, 255).astype(np.uint8)

# Menerapkan white balance pada gambar
img = gray_world_wb(img)

# Menampilkan informasi setelah white balance
print(f"  Mean setelah WB: B={np.mean(img[:,:,0]):.1f}, G={np.mean(img[:,:,1]):.1f}, R={np.mean(img[:,:,2]):.1f}")

# Menambahkan ke daftar tahap pipeline
pipeline_stages.append(img.copy())
stage_names.append("1. White Balance")

# ============================================================
# 3. Tahap 2: Denoising
# ============================================================
print("\n[TAHAP 2] Denoising (fastNlMeansDenoising)...")

# Menerapkan Non-Local Means Denoising untuk gambar berwarna
# Parameter: h=10 (filter strength), hForColorComponents=10
# templateWindowSize=7, searchWindowSize=21
img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hForColorComponents=10,
                                        templateWindowSize=7, searchWindowSize=21)

# Menampilkan informasi
print(f"  Denoising selesai (h=10, template=7, search=21)")

# Menambahkan ke daftar tahap pipeline
pipeline_stages.append(img.copy())
stage_names.append("2. Denoised")

# ============================================================
# 4. Tahap 3: CLAHE (Contrast Enhancement)
# ============================================================
print("\n[TAHAP 3] CLAHE (Contrast Enhancement)...")

# Mengonversi gambar ke LAB untuk CLAHE pada channel L
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Memisahkan channel L, A, B
l_ch, a_ch, b_ch = cv2.split(lab)

# Membuat objek CLAHE dengan clipLimit=3.0 dan tileGrid=8x8
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Menerapkan CLAHE pada channel L
l_clahe = clahe.apply(l_ch)

# Menggabungkan kembali dan konversi ke BGR
lab_clahe = cv2.merge([l_clahe, a_ch, b_ch])
img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Menampilkan informasi
print(f"  CLAHE diterapkan (clipLimit=3.0, tileGrid=8x8)")
print(f"  Mean brightness setelah CLAHE: {np.mean(img):.1f}")

# Menambahkan ke daftar tahap pipeline
pipeline_stages.append(img.copy())
stage_names.append("3. CLAHE")

# ============================================================
# 5. Tahap 4: Sharpening (Unsharp Mask)
# ============================================================
print("\n[TAHAP 4] Sharpening (Unsharp Mask)...")

# Membuat versi blur dari gambar
sigma = 1.5
ksize = int(6 * sigma + 1)
if ksize % 2 == 0:
    ksize += 1

# Menerapkan Gaussian Blur
blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)

# Menghitung Unsharp Mask: sharpened = original + amount * (original - blurred)
amount = 1.0
img_float = img.astype(np.float64)
blur_float = blurred.astype(np.float64)

# Menghitung detail
detail = img_float - blur_float

# Menambahkan detail yang diperkuat
sharpened = img_float + amount * detail

# Membatasi dan mengonversi ke uint8
img = np.clip(sharpened, 0, 255).astype(np.uint8)

# Menampilkan informasi
print(f"  Unsharp Mask diterapkan (sigma={sigma}, amount={amount})")

# Menambahkan ke daftar tahap pipeline
pipeline_stages.append(img.copy())
stage_names.append("4. Sharpened")

# ============================================================
# 6. Tahap 5: Color Boost (Vibrance)
# ============================================================
print("\n[TAHAP 5] Color Boost (Vibrance)...")

# Mengonversi ke HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)

# Memisahkan channel
h_ch, s_ch, v_ch = cv2.split(hsv)

# Menghitung faktor boost selektif (vibrance)
vibrance_intensity = 1.3
low_sat_factor = 1.0 - (s_ch / 255.0)
boost = 1.0 + (vibrance_intensity - 1.0) * low_sat_factor

# Menerapkan boost pada saturasi
s_boosted = s_ch * boost

# Membatasi ke rentang valid
s_boosted = np.clip(s_boosted, 0, 255)

# Menggabungkan kembali dan konversi ke BGR
hsv_boosted = cv2.merge([h_ch, s_boosted, v_ch]).astype(np.uint8)
img = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

# Menampilkan informasi
print(f"  Vibrance diterapkan (intensity={vibrance_intensity})")

# Menambahkan ke daftar tahap pipeline
pipeline_stages.append(img.copy())
stage_names.append("5. Color Boost")

# ============================================================
# 7. Visualisasi setiap tahap pipeline
# ============================================================
print("\n[LANGKAH 6] Membuat visualisasi pipeline...")

# Membuat figure untuk seluruh tahap pipeline
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))

# Menampilkan setiap tahap pipeline
for idx, (stage_img, stage_name) in enumerate(zip(pipeline_stages, stage_names)):
    # Menghitung posisi subplot
    row = idx // 3
    col = idx % 3

    # Menampilkan gambar pada subplot
    axes1[row, col].imshow(cv2.cvtColor(stage_img, cv2.COLOR_BGR2RGB))
    axes1[row, col].set_title(stage_name, fontsize=12, fontweight='bold')
    axes1[row, col].axis("off")

# Menambahkan judul utama
fig1.suptitle("Image Enhancement Pipeline - Setiap Tahap", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path1 = os.path.join(OUTPUT_DIR, "16_pipeline_stages.png")
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path1}")

# Menutup figure
plt.show()
plt.show()
plt.show()
plt.close(fig1)

# ============================================================
# 8. Perbandingan langsung Original vs Final
# ============================================================
print("\n[LANGKAH 7] Perbandingan Original vs Final...")

# Membuat figure perbandingan
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))

# Menampilkan gambar original
axes2[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
axes2[0].set_title("SEBELUM (Original)", fontsize=14, fontweight='bold')
axes2[0].axis("off")

# Menampilkan hasil akhir pipeline
axes2[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes2[1].set_title("SESUDAH (Enhanced)", fontsize=14, fontweight='bold')
axes2[1].axis("off")

# Menambahkan judul utama
fig2.suptitle("Perbandingan Sebelum dan Sesudah Enhancement Pipeline", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path2 = os.path.join(OUTPUT_DIR, "16_before_after.png")
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path2}")

# Menutup figure
plt.show()
plt.show()
plt.show()
plt.close(fig2)

# ============================================================
# 9. Statistik perubahan setiap tahap
# ============================================================
print("\n[LANGKAH 8] Statistik brightness setiap tahap:")

# Menghitung dan menampilkan statistik setiap tahap
means = []
for stage_img, stage_name in zip(pipeline_stages, stage_names):
    # Menghitung mean brightness
    mean_val = np.mean(stage_img)
    means.append(mean_val)
    # Menampilkan informasi
    print(f"  {stage_name}: mean={mean_val:.1f}")

# Membuat bar chart statistik
fig3, ax3 = plt.subplots(figsize=(10, 5))

# Membuat bar chart
bars = ax3.bar(range(len(means)), means, color=['gray', 'skyblue', 'lightgreen', 'orange', 'salmon', 'violet'])

# Menambahkan label pada setiap bar
for i, (bar, mean) in enumerate(zip(bars, means)):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

# Mengatur label sumbu x
ax3.set_xticks(range(len(stage_names)))
ax3.set_xticklabels(stage_names, rotation=30, ha='right', fontsize=9)

# Mengatur label sumbu y
ax3.set_ylabel("Mean Brightness", fontsize=12)

# Menambahkan judul
ax3.set_title("Statistik Mean Brightness Setiap Tahap Pipeline", fontsize=13, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path3 = os.path.join(OUTPUT_DIR, "16_statistik_pipeline.png")
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
print("RINGKASAN PERCOBAAN 16: IMAGE ENHANCEMENT PIPELINE")
print("=" * 60)
print("Pipeline lengkap yang diterapkan:")
print("1. White Balance (Gray World)")
print("   → Menyeimbangkan mean channel warna")
print("2. Denoising (cv2.fastNlMeansDenoisingColored)")
print("   → Menghapus noise sambil mempertahankan detail")
print("3. CLAHE (cv2.createCLAHE)")
print("   → Meningkatkan kontras lokal secara adaptif")
print("4. Sharpening (Unsharp Mask)")
print("   → Menajamkan detail: sharp = orig + amt*(orig - blur)")
print("5. Color Boost (Vibrance)")
print("   → Meningkatkan saturasi warna secara selektif")
print()
print("Fungsi OpenCV yang digunakan:")
print("- cv2.fastNlMeansDenoisingColored()")
print("- cv2.createCLAHE(clipLimit, tileGridSize)")
print("- cv2.GaussianBlur(src, ksize, sigma)")
print("- cv2.cvtColor() untuk konversi BGR↔LAB↔HSV")
print()
print("Kesimpulan:")
print("- Urutan pipeline mempengaruhi hasil akhir")
print("- Setiap tahap mengatasi masalah spesifik")
print("- Pipeline modular memudahkan tuning per tahap")
print("=" * 60)
