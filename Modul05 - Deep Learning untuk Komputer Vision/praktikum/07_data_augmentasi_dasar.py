"""
==========================================================================
PERCOBAAN 7: DATA AUGMENTASI DASAR
==========================================================================
Program ini mempelajari teknik-teknik augmentasi data (data augmentation)
yang umum digunakan dalam pelatihan model deep learning. Augmentasi data
memperbanyak variasi dataset training tanpa mengumpulkan data baru.

Fungsi utama yang dipelajari:
- cv2.flip()                   : Membalik gambar (horizontal/vertikal)
- cv2.getRotationMatrix2D()    : Membuat matriks rotasi
- cv2.warpAffine()             : Menerapkan transformasi affine (rotasi, dll)
- cv2.convertScaleAbs()        : Menyesuaikan brightness/contrast
- cv2.GaussianBlur()           : Menambahkan blur Gaussian
- cv2.cvtColor()               : Konversi ruang warna untuk jittering
- np.random                    : Operasi random untuk augmentasi

Konsep yang dipelajari:
- Augmentasi geometris: flip, rotasi, crop, resize
- Augmentasi warna: brightness, contrast, hue, saturation
- Augmentasi noise: Gaussian noise, random erasing
- Pipeline augmentasi untuk training deep learning
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan random
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi hasil augmentasi
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan header percobaan
print("=" * 60)
print("PERCOBAAN 7: DATA AUGMENTASI DASAR")
print("=" * 60)

# ============================================================
# 1. Memuat gambar untuk augmentasi
# ============================================================
print("\n--- 1. Memuat Gambar Sample ---")

# Memuat gambar augmentasi sample
img = cv2.imread(os.path.join(IMAGE_DIR, "augmentasi_sample.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img is None:
    # Mencoba memuat gambar alternatif jika augmentasi_sample tidak ada
    img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))
    if img is None:
        print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
        exit()

# Meresize gambar ke ukuran standar
img = cv2.resize(img, (300, 300))

# Menampilkan informasi gambar
print(f"  Ukuran gambar: {img.shape}")
print(f"  Tipe data    : {img.dtype}")

# ============================================================
# 2. Augmentasi flip (horizontal dan vertikal)
# ============================================================
print("\n--- 2. Augmentasi Flip ---")

# Melakukan horizontal flip (cermin horizontal)
# flipCode=1 berarti flip terhadap sumbu Y (horizontal)
img_hflip = cv2.flip(img, 1)
print(f"  Horizontal flip (flipCode=1): {img_hflip.shape}")

# Melakukan vertical flip (cermin vertikal)
# flipCode=0 berarti flip terhadap sumbu X (vertikal)
img_vflip = cv2.flip(img, 0)
print(f"  Vertical flip (flipCode=0)  : {img_vflip.shape}")

# Melakukan flip horizontal dan vertikal sekaligus
# flipCode=-1 berarti flip terhadap kedua sumbu
img_hvflip = cv2.flip(img, -1)
print(f"  Both flip (flipCode=-1)     : {img_hvflip.shape}")

# ============================================================
# 3. Augmentasi rotasi
# ============================================================
print("\n--- 3. Augmentasi Rotasi ---")

# Mendapatkan titik pusat gambar untuk rotasi
h, w = img.shape[:2]
center = (w // 2, h // 2)

# Merotasi gambar 15 derajat
# cv2.getRotationMatrix2D(center, angle, scale)
M_15 = cv2.getRotationMatrix2D(center, 15, 1.0)
img_rot15 = cv2.warpAffine(img, M_15, (w, h))
print(f"  Rotasi 15 derajat : {img_rot15.shape}")

# Merotasi gambar 45 derajat
M_45 = cv2.getRotationMatrix2D(center, 45, 1.0)
img_rot45 = cv2.warpAffine(img, M_45, (w, h))
print(f"  Rotasi 45 derajat : {img_rot45.shape}")

# Merotasi gambar 90 derajat
M_90 = cv2.getRotationMatrix2D(center, 90, 1.0)
img_rot90 = cv2.warpAffine(img, M_90, (w, h))
print(f"  Rotasi 90 derajat : {img_rot90.shape}")

# Merotasi gambar -30 derajat (searah jarum jam)
M_n30 = cv2.getRotationMatrix2D(center, -30, 1.0)
img_rotn30 = cv2.warpAffine(img, M_n30, (w, h))
print(f"  Rotasi -30 derajat: {img_rotn30.shape}")

# ============================================================
# 4. Visualisasi flip dan rotasi
# ============================================================
print("\n--- 4. Visualisasi Flip dan Rotasi ---")

# Membuat figure untuk augmentasi flip dan rotasi
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Baris 1: Flip
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Asli", fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img_hflip, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Horizontal Flip")
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(img_vflip, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Vertical Flip")
axes[0, 2].axis('off')

axes[0, 3].imshow(cv2.cvtColor(img_hvflip, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Both Flip")
axes[0, 3].axis('off')

# Baris 2: Rotasi
axes[1, 0].imshow(cv2.cvtColor(img_rot15, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Rotasi 15°")
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(img_rot45, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Rotasi 45°")
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(img_rot90, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Rotasi 90°")
axes[1, 2].axis('off')

axes[1, 3].imshow(cv2.cvtColor(img_rotn30, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Rotasi -30°")
axes[1, 3].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 7: Augmentasi Flip dan Rotasi",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "07_augmentasi_fliprotasi.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/07_augmentasi_fliprotasi.png")

# ============================================================
# 5. Augmentasi brightness dan contrast
# ============================================================
print("\n--- 5. Augmentasi Brightness dan Contrast ---")

# Menyesuaikan brightness (kecerahan) dengan menambah/mengurangi nilai
# cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
# alpha > 1 = kontras lebih tinggi, beta > 0 = lebih terang
img_bright = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
print(f"  Brightness +50: mean={img_bright.mean():.1f}")

# Mengurangi brightness
img_dark = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)
print(f"  Brightness -50: mean={img_dark.mean():.1f}")

# Menambah kontras
img_high_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
print(f"  Contrast x1.5 : mean={img_high_contrast.mean():.1f}")

# Mengurangi kontras
img_low_contrast = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
print(f"  Contrast x0.5 : mean={img_low_contrast.mean():.1f}")

# ============================================================
# 6. Augmentasi warna (hue, saturation)
# ============================================================
print("\n--- 6. Augmentasi Warna (Color Jittering) ---")

# Mengkonversi gambar ke ruang warna HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

# Mengubah Hue (warna) dengan menambahkan offset
hsv_hue = hsv.copy()
hsv_hue[:, :, 0] = (hsv_hue[:, :, 0] + 30) % 180
img_hue_shift = cv2.cvtColor(hsv_hue.astype(np.uint8), cv2.COLOR_HSV2BGR)
print(f"  Hue shift +30  : done")

# Menambah saturasi (kejenuhan warna)
hsv_sat_high = hsv.copy()
hsv_sat_high[:, :, 1] = np.clip(hsv_sat_high[:, :, 1] * 1.5, 0, 255)
img_sat_high = cv2.cvtColor(hsv_sat_high.astype(np.uint8), cv2.COLOR_HSV2BGR)
print(f"  Saturation x1.5: done")

# Mengurangi saturasi
hsv_sat_low = hsv.copy()
hsv_sat_low[:, :, 1] = np.clip(hsv_sat_low[:, :, 1] * 0.5, 0, 255)
img_sat_low = cv2.cvtColor(hsv_sat_low.astype(np.uint8), cv2.COLOR_HSV2BGR)
print(f"  Saturation x0.5: done")

# ============================================================
# 7. Visualisasi augmentasi warna
# ============================================================
print("\n--- 7. Visualisasi Augmentasi Warna ---")

# Membuat figure untuk augmentasi warna
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Baris 1: Brightness dan contrast
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Asli", fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img_bright, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Brightness +50")
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(img_dark, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Brightness -50")
axes[0, 2].axis('off')

axes[0, 3].imshow(cv2.cvtColor(img_high_contrast, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Contrast x1.5")
axes[0, 3].axis('off')

# Baris 2: Warna (HSV)
axes[1, 0].imshow(cv2.cvtColor(img_low_contrast, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Contrast x0.5")
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(img_hue_shift, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Hue Shift +30")
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(img_sat_high, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Saturasi Tinggi")
axes[1, 2].axis('off')

axes[1, 3].imshow(cv2.cvtColor(img_sat_low, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Saturasi Rendah")
axes[1, 3].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 7: Augmentasi Brightness, Contrast, dan Warna",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "07_augmentasi_warna.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/07_augmentasi_warna.png")

# ============================================================
# 8. Augmentasi noise dan random erasing
# ============================================================
print("\n--- 8. Augmentasi Noise dan Random Erasing ---")

def tambah_gaussian_noise(gambar, mean=0, sigma=25):
    """
    Menambahkan Gaussian noise pada gambar.
    Noise Gaussian memiliki distribusi normal dengan mean dan sigma tertentu.
    """
    # Membuat noise random dengan distribusi Gaussian
    noise = np.random.normal(mean, sigma, gambar.shape).astype(np.float32)

    # Menambahkan noise ke gambar
    img_noisy = gambar.astype(np.float32) + noise

    # Memastikan nilai piksel tetap dalam range [0, 255]
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

    # Mengembalikan gambar yang sudah ditambah noise
    return img_noisy

def random_erasing(gambar, prob=0.5, sl=0.02, sh=0.2):
    """
    Menerapkan random erasing (cutout) pada gambar.
    Menghapus area random pada gambar dan mengisinya dengan nilai random.
    """
    # Membuat salinan gambar untuk tidak mengubah aslinya
    img_erased = gambar.copy()

    # Menentukan apakah erasing diterapkan berdasarkan probabilitas
    if np.random.random() > prob:
        return img_erased

    # Mendapatkan dimensi gambar
    h, w = img_erased.shape[:2]

    # Menghitung luas area gambar
    luas_gambar = h * w

    # Menentukan luas area yang akan dihapus secara random
    luas_erase = np.random.uniform(sl, sh) * luas_gambar

    # Menentukan aspect ratio area yang dihapus
    aspect_ratio = np.random.uniform(0.3, 3.3)

    # Menghitung tinggi dan lebar area yang dihapus
    h_erase = int(np.sqrt(luas_erase * aspect_ratio))
    w_erase = int(np.sqrt(luas_erase / aspect_ratio))

    # Memastikan ukuran area tidak melebihi ukuran gambar
    h_erase = min(h_erase, h)
    w_erase = min(w_erase, w)

    # Menentukan posisi random untuk area yang dihapus
    y1 = np.random.randint(0, h - h_erase + 1)
    x1 = np.random.randint(0, w - w_erase + 1)

    # Mengisi area dengan nilai random
    img_erased[y1:y1 + h_erase, x1:x1 + w_erase] = np.random.randint(
        0, 256, (h_erase, w_erase, img_erased.shape[2]), dtype=np.uint8
    )

    # Mengembalikan gambar yang sudah di-erase
    return img_erased

def random_crop_resize(gambar, crop_ratio=0.8):
    """
    Melakukan random crop lalu resize kembali ke ukuran asli.
    """
    # Mendapatkan dimensi gambar
    h, w = gambar.shape[:2]

    # Menghitung ukuran crop
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)

    # Menentukan posisi crop secara random
    y1 = np.random.randint(0, h - crop_h + 1)
    x1 = np.random.randint(0, w - crop_w + 1)

    # Melakukan crop
    cropped = gambar[y1:y1 + crop_h, x1:x1 + crop_w]

    # Meresize kembali ke ukuran asli
    resized = cv2.resize(cropped, (w, h))

    # Mengembalikan gambar yang sudah di-crop dan di-resize
    return resized

# Menerapkan Gaussian noise dengan berbagai intensitas
img_noise_low = tambah_gaussian_noise(img, sigma=15)
print(f"  Gaussian noise (sigma=15) : done")

img_noise_high = tambah_gaussian_noise(img, sigma=50)
print(f"  Gaussian noise (sigma=50) : done")

# Menerapkan random erasing
np.random.seed(42)
img_erased1 = random_erasing(img, prob=1.0, sl=0.05, sh=0.15)
print(f"  Random erasing (5-15%)    : done")

img_erased2 = random_erasing(img, prob=1.0, sl=0.1, sh=0.3)
print(f"  Random erasing (10-30%)   : done")

# Menerapkan random crop dan resize
np.random.seed(42)
img_cropped = random_crop_resize(img, crop_ratio=0.7)
print(f"  Random crop 70%           : done")

# ============================================================
# 9. Visualisasi noise dan erasing
# ============================================================
print("\n--- 9. Visualisasi Noise dan Random Erasing ---")

# Membuat figure untuk augmentasi noise
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Baris 1: Noise
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Asli", fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img_noise_low, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Gaussian Noise (σ=15)")
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(img_noise_high, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Gaussian Noise (σ=50)")
axes[0, 2].axis('off')

# Baris 2: Random erasing dan crop
axes[1, 0].imshow(cv2.cvtColor(img_erased1, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Random Erasing (5-15%)")
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(img_erased2, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Random Erasing (10-30%)")
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Random Crop + Resize")
axes[1, 2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 7: Augmentasi Noise, Random Erasing, dan Crop",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "07_augmentasi_noise.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/07_augmentasi_noise.png")

# ============================================================
# 10. Pipeline augmentasi lengkap
# ============================================================
print("\n--- 10. Pipeline Augmentasi Lengkap ---")

def augmentasi_pipeline(gambar, seed=None):
    """
    Pipeline augmentasi yang menerapkan berbagai teknik secara random.
    Simulasi pipeline training deep learning.
    """
    # Mengatur random seed jika diberikan untuk reproduktibilitas
    if seed is not None:
        np.random.seed(seed)

    # Membuat salinan gambar
    hasil = gambar.copy()

    # Langkah 1: Random horizontal flip (50% kemungkinan)
    if np.random.random() > 0.5:
        hasil = cv2.flip(hasil, 1)

    # Langkah 2: Random rotasi (-15 sampai +15 derajat)
    angle = np.random.uniform(-15, 15)
    h, w = hasil.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    hasil = cv2.warpAffine(hasil, M, (w, h))

    # Langkah 3: Random brightness adjustment
    beta = np.random.randint(-30, 30)
    hasil = cv2.convertScaleAbs(hasil, alpha=1.0, beta=beta)

    # Langkah 4: Random contrast adjustment
    alpha = np.random.uniform(0.8, 1.2)
    hasil = cv2.convertScaleAbs(hasil, alpha=alpha, beta=0)

    # Langkah 5: Random noise (30% kemungkinan)
    if np.random.random() > 0.7:
        noise = np.random.normal(0, 10, hasil.shape).astype(np.float32)
        hasil = np.clip(hasil.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Langkah 6: Random crop dan resize (30% kemungkinan)
    if np.random.random() > 0.7:
        crop_ratio = np.random.uniform(0.8, 0.95)
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        y1 = np.random.randint(0, h - crop_h + 1)
        x1 = np.random.randint(0, w - crop_w + 1)
        hasil = cv2.resize(hasil[y1:y1+crop_h, x1:x1+crop_w], (w, h))

    # Mengembalikan gambar yang sudah diaugmentasi
    return hasil

# Menerapkan pipeline pada gambar 5 kali dengan seed berbeda
print("  Menjalankan pipeline augmentasi 5 kali:")
augmented_list = []
for i in range(5):
    # Menerapkan pipeline dengan seed unik
    img_aug = augmentasi_pipeline(img, seed=i * 10 + 42)
    augmented_list.append(img_aug)
    print(f"    Augmentasi #{i+1}: mean={img_aug.mean():.1f}, std={img_aug.std():.1f}")

# Menerapkan pipeline pada beberapa gambar dari dataset
print("\n  Menjalankan pipeline pada dataset:")
gambar_files = ["kucing.jpg", "anjing.jpg", "mobil.jpg", "bunga.jpg", "gedung.jpg"]
dataset_augmented = []

for fname in gambar_files:
    # Memuat gambar dari dataset
    path = os.path.join(IMAGE_DIR, fname)
    g = cv2.imread(path)
    if g is not None:
        # Meresize ke ukuran standar
        g = cv2.resize(g, (224, 224))
        # Menerapkan augmentasi
        g_aug = augmentasi_pipeline(g, seed=42)
        dataset_augmented.append((fname, g, g_aug))
        print(f"    {fname}: augmentasi berhasil")

# ============================================================
# 11. Visualisasi pipeline augmentasi
# ============================================================
print("\n--- 11. Visualisasi Pipeline Augmentasi ---")

# Membuat figure untuk 5 versi augmentasi dari gambar yang sama
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Meratakan axes
axes_flat = axes.flatten()

# Menampilkan gambar asli
axes_flat[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes_flat[0].set_title("Asli", fontsize=11, fontweight='bold')
axes_flat[0].axis('off')

# Menampilkan 5 versi augmentasi
for i, img_aug in enumerate(augmented_list):
    # Mengkonversi dan menampilkan gambar augmentasi
    axes_flat[i + 1].imshow(cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB))
    axes_flat[i + 1].set_title(f"Augmentasi #{i+1}", fontsize=10)
    axes_flat[i + 1].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 7: Pipeline Augmentasi - 5 Variasi dari Gambar yang Sama",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "07_augmentasi_pipeline.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/07_augmentasi_pipeline.png")

# ============================================================
# 12. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 7")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. Data augmentasi memperbanyak variasi dataset tanpa data baru
2. Augmentasi geometris: cv2.flip(), rotasi dengan warpAffine()
3. Augmentasi warna: brightness (beta), contrast (alpha), hue/saturation
4. Augmentasi noise: Gaussian noise, random erasing (cutout)
5. Random crop + resize mensimulasikan skala objek yang berbeda
6. Pipeline augmentasi menggabungkan beberapa teknik secara random
7. Setiap augmentasi memiliki parameter yang bisa diatur intensitasnya
8. Augmentasi diterapkan secara random saat training untuk generalisasi

Output disimpan di folder: output/
- 07_augmentasi_fliprotasi.png : Augmentasi flip dan rotasi
- 07_augmentasi_warna.png      : Augmentasi brightness, contrast, warna
- 07_augmentasi_noise.png      : Augmentasi noise dan random erasing
- 07_augmentasi_pipeline.png   : Pipeline augmentasi lengkap
""")
