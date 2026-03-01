"""
==========================================================================
PERCOBAAN 14: SYNTHETIC BOKEH (DEPTH OF FIELD) EFFECT
==========================================================================
Program ini mempelajari pembuatan efek bokeh (kedalaman bidang) sintetis
menggunakan depth map. Efek bokeh membuat area di luar fokus menjadi blur,
sementara area yang in-focus tetap tajam, mensimulasikan kamera dengan
aperture lebar.

Prinsip kerja:
1. Memuat portrait.png dan depth_map_portrait.png
2. Menentukan titik fokus (depth level tertentu)
3. Blur variabel berdasarkan selisih kedalaman dari titik fokus
4. Semakin jauh dari fokus → blur semakin kuat

Fungsi utama yang dipelajari:
- cv2.GaussianBlur(src, ksize, sigmaX) : Blur Gaussian dengan kernel bervariasi
- cv2.imread(path, cv2.IMREAD_GRAYSCALE): Membaca depth map sebagai grayscale
- cv2.normalize()                       : Normalisasi nilai depth map
- np.where()                            : Seleksi piksel berdasarkan kondisi

Hasil: Efek bokeh dengan berbagai tingkat kedalaman fokus
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
print("PERCOBAAN 14: SYNTHETIC BOKEH EFFECT")
print("=" * 60)

# ============================================================
# 1. Membaca gambar portrait dan depth map
# ============================================================

# Mendefinisikan path gambar portrait
path_portrait = os.path.join(IMAGE_DIR, "portrait.png")

# Mendefinisikan path depth map portrait
path_depth = os.path.join(IMAGE_DIR, "depth_map_portrait.png")

# Membaca gambar portrait dalam format BGR
img_portrait = cv2.imread(path_portrait)

# Membaca depth map sebagai grayscale (1 channel)
depth_map = cv2.imread(path_depth, cv2.IMREAD_GRAYSCALE)

# Memeriksa apakah gambar portrait berhasil dimuat
if img_portrait is None:
    # Jika portrait tidak ditemukan, buat gambar sintetis
    print("[INFO] portrait.png tidak ditemukan, membuat gambar sintetis...")
    # Membuat gambar sintetis 400x300 dengan latar belakang gradient
    img_portrait = np.zeros((400, 300, 3), dtype=np.uint8)
    # Membuat gradient latar belakang (hijau ke biru)
    for y in range(400):
        img_portrait[y, :, 0] = int(100 + 80 * y / 400)
        img_portrait[y, :, 1] = int(150 - 50 * y / 400)
        img_portrait[y, :, 2] = int(50 + 30 * y / 400)
    # Menambahkan "wajah" sintetis (lingkaran di tengah)
    cv2.circle(img_portrait, (150, 160), 70, (180, 160, 140), -1)
    cv2.circle(img_portrait, (130, 145), 10, (60, 40, 30), -1)
    cv2.circle(img_portrait, (170, 145), 10, (60, 40, 30), -1)
    cv2.ellipse(img_portrait, (150, 180), (25, 10), 0, 0, 180, (100, 60, 60), 2)

# Memeriksa apakah depth map berhasil dimuat
if depth_map is None:
    # Jika depth map tidak ditemukan, buat depth map sintetis
    print("[INFO] depth_map_portrait.png tidak ditemukan, membuat depth map sintetis...")
    h, w = img_portrait.shape[:2]
    # Membuat depth map gradient (atas=jauh, bawah=dekat)
    depth_map = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        # Depth dari 30 (jauh) ke 220 (dekat) dari atas ke bawah
        depth_map[y, :] = int(30 + 190 * y / h)
    # Membuat area subjek (tengah) memiliki depth tertentu (fokus)
    cv2.circle(depth_map, (w // 2, int(h * 0.4)), 70, 180, -1)

# Memastikan depth map memiliki ukuran yang sama dengan portrait
depth_map = cv2.resize(depth_map, (img_portrait.shape[1], img_portrait.shape[0]))

# Menampilkan informasi dimensi gambar
print(f"[INFO] Ukuran portrait: {img_portrait.shape}")
print(f"[INFO] Ukuran depth map: {depth_map.shape}")
print(f"[INFO] Rentang depth: {depth_map.min()} - {depth_map.max()}")

# ============================================================
# 2. Normalisasi depth map ke rentang 0-1
# ============================================================
print("\n[LANGKAH 1] Normalisasi depth map...")

# Mengonversi depth map ke float dan normalisasi ke rentang 0-1
depth_norm = depth_map.astype(np.float64) / 255.0

# Menampilkan statistik depth map
print(f"  Depth min: {depth_norm.min():.3f}, max: {depth_norm.max():.3f}")
print(f"  Depth mean: {depth_norm.mean():.3f}")

# ============================================================
# 3. Fungsi untuk membuat efek bokeh berdasarkan depth
# ============================================================

def apply_bokeh(image, depth, focus_depth, max_blur=31, blur_strength=2.0):
    """
    Menerapkan efek bokeh berdasarkan depth map.
    focus_depth: nilai depth yang menjadi titik fokus (0-1)
    max_blur: ukuran kernel blur maksimum
    blur_strength: pengali kekuatan blur
    """
    # Membuat salinan gambar untuk hasil
    result = image.copy().astype(np.float64)

    # Menghitung jarak setiap piksel dari titik fokus
    distance = np.abs(depth - focus_depth)

    # Mengalikan jarak dengan kekuatan blur
    blur_amount = distance * blur_strength

    # Membatasi blur_amount ke rentang 0-1
    blur_amount = np.clip(blur_amount, 0, 1)

    # Mendefinisikan beberapa level blur (dari sedikit ke banyak)
    blur_levels = 7

    # Membuat daftar gambar blur dengan berbagai tingkat
    blurred_images = []
    for i in range(blur_levels):
        # Menghitung ukuran kernel untuk level ini
        ksize = 1 + 2 * (i * max_blur // blur_levels)
        # Memastikan kernel ganjil dan minimal 1
        ksize = max(1, ksize)
        if ksize % 2 == 0:
            ksize += 1
        # Jika ksize > 1, terapkan blur; jika 1, gunakan gambar asli
        if ksize > 1:
            blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        else:
            blurred = image.copy()
        # Menambahkan ke daftar gambar blur
        blurred_images.append(blurred.astype(np.float64))

    # Menggabungkan gambar berdasarkan blur_amount di setiap piksel
    result = np.zeros_like(image, dtype=np.float64)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Menentukan level blur berdasarkan jarak
            level = int(blur_amount[y, x] * (blur_levels - 1))
            level = min(level, blur_levels - 1)
            # Mengambil piksel dari gambar blur yang sesuai
            result[y, x] = blurred_images[level][y, x]

    # Mengonversi kembali ke uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Mengembalikan hasil
    return result

def apply_bokeh_fast(image, depth, focus_depth, max_blur=31, blur_strength=2.0):
    """
    Versi cepat efek bokeh menggunakan blending antar level blur.
    Menghindari loop per-piksel dengan blending layer.
    """
    # Mendefinisikan jumlah level blur
    num_levels = 5

    # Menghitung jarak setiap piksel dari titik fokus
    distance = np.abs(depth - focus_depth) * blur_strength

    # Membatasi ke rentang 0-1
    distance = np.clip(distance, 0, 1)

    # Membuat gambar hasil dimulai dari gambar asli
    result = image.astype(np.float64)

    # Menerapkan blur bertahap menggunakan level
    for i in range(num_levels):
        # Menghitung ukuran kernel untuk level ini
        ksize = 3 + 2 * i * (max_blur // num_levels)
        if ksize % 2 == 0:
            ksize += 1

        # Menerapkan Gaussian Blur dengan kernel
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0).astype(np.float64)

        # Menghitung batas bawah dan atas threshold untuk level ini
        low = i / num_levels
        high = (i + 1) / num_levels

        # Membuat mask untuk piksel yang masuk level ini
        mask = ((distance >= low) & (distance < high)).astype(np.float64)

        # Memperluas mask ke 3 channel
        mask_3ch = np.stack([mask, mask, mask], axis=2)

        # Menggabungkan: gunakan blurred di area mask, pertahankan result di luar
        result = result * (1 - mask_3ch) + blurred * mask_3ch

    # Mengonversi ke uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Mengembalikan hasil
    return result

# ============================================================
# 4. Membuat efek bokeh dengan berbagai titik fokus
# ============================================================
print("\n[LANGKAH 2] Membuat efek bokeh dengan berbagai titik fokus...")

# Mendefinisikan daftar depth level fokus yang akan diuji
focus_levels = [0.2, 0.4, 0.6, 0.8]

# Membuat figure untuk variasi fokus
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))

# Menampilkan gambar asli
axes1[0, 0].imshow(cv2.cvtColor(img_portrait, cv2.COLOR_BGR2RGB))
axes1[0, 0].set_title("Original", fontsize=11)
axes1[0, 0].axis("off")

# Menampilkan depth map
axes1[0, 1].imshow(depth_map, cmap='jet')
axes1[0, 1].set_title("Depth Map", fontsize=11)
axes1[0, 1].axis("off")

# Menampilkan efek bokeh untuk setiap titik fokus
for idx, fl in enumerate(focus_levels):
    # Menghitung posisi subplot (mulai dari posisi ketiga)
    pos = idx + 2
    row = pos // 3
    col = pos % 3

    # Menerapkan efek bokeh cepat dengan fokus tertentu
    print(f"  Memproses fokus pada depth={fl}...")
    bokeh_result = apply_bokeh_fast(img_portrait, depth_norm, fl, max_blur=25, blur_strength=2.5)

    # Menampilkan hasil pada subplot
    axes1[row, col].imshow(cv2.cvtColor(bokeh_result, cv2.COLOR_BGR2RGB))
    axes1[row, col].set_title(f"Fokus depth={fl}", fontsize=11)
    axes1[row, col].axis("off")

# Menambahkan judul utama
fig1.suptitle("Efek Bokeh Sintetis - Variasi Titik Fokus", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path1 = os.path.join(OUTPUT_DIR, "14_bokeh_variasi_fokus.png")
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path1}")

# Menutup figure
plt.close(fig1)

# ============================================================
# 5. Variasi kekuatan blur (blur strength)
# ============================================================
print("\n[LANGKAH 3] Variasi kekuatan blur...")

# Mendefinisikan daftar kekuatan blur yang akan diuji
strengths = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

# Menentukan depth fokus tetap (misalnya pada subjek utama)
fixed_focus = 0.7

# Membuat figure untuk variasi kekuatan blur
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

# Menerapkan dan menampilkan setiap kekuatan blur
for idx, st in enumerate(strengths):
    # Menghitung posisi subplot
    row = idx // 3
    col = idx % 3

    # Menerapkan bokeh dengan kekuatan blur tertentu
    result = apply_bokeh_fast(img_portrait, depth_norm, fixed_focus, max_blur=31, blur_strength=st)

    # Menampilkan pada subplot
    axes2[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes2[row, col].set_title(f"Strength={st} (focus={fixed_focus})", fontsize=10)
    axes2[row, col].axis("off")

# Menambahkan judul utama
fig2.suptitle("Efek Bokeh - Variasi Kekuatan Blur", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path2 = os.path.join(OUTPUT_DIR, "14_bokeh_variasi_strength.png")
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path2}")

# Menutup figure
plt.close(fig2)

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 14: SYNTHETIC BOKEH EFFECT")
print("=" * 60)
print("Fungsi-fungsi yang dipelajari:")
print("1. cv2.GaussianBlur(src, ksize, sigmaX)")
print("   → Menerapkan blur Gaussian dengan ukuran kernel bervariasi")
print("2. cv2.imread(path, cv2.IMREAD_GRAYSCALE)")
print("   → Membaca depth map sebagai gambar grayscale")
print("3. cv2.normalize(src, dst, alpha, beta, norm_type)")
print("   → Normalisasi nilai depth map")
print("4. cv2.resize(src, dsize)")
print("   → Menyesuaikan ukuran depth map agar cocok dengan gambar")
print()
print("Konsep penting:")
print("- Depth map menyimpan informasi kedalaman setiap piksel")
print("- Focus depth menentukan area yang tetap tajam")
print("- Semakin jauh dari focus → blur semakin kuat")
print("- max_blur mengontrol ukuran kernel maksimum")
print("- blur_strength mengontrol sensitivitas terhadap jarak depth")
print()
print("Kesimpulan:")
print("- Efek bokeh sintetis mensimulasikan depth of field kamera")
print("- Kualitas bergantung pada akurasi depth map")
print("- Metode layer-based lebih cepat dari per-pixel processing")
print("=" * 60)
