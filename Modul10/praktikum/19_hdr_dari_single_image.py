"""
==========================================================================
PERCOBAAN 19: HDR DARI SINGLE IMAGE (PSEUDO-HDR)
==========================================================================
Program ini mempelajari teknik membuat efek HDR (High Dynamic Range) dari
satu gambar saja (single image), tanpa memerlukan multi-exposure.

Teknik yang dipelajari:
1. Membuat multi-exposure sintetis menggunakan gamma correction
2. Exposure Fusion (Mertens) pada exposure sintetis
3. Local tone mapping menggunakan CLAHE
4. Kombinasi teknik untuk hasil terbaik

Fungsi utama yang dipelajari:
- cv2.createMergeMertens()     : Exposure fusion (tanpa perlu exposure time)
- cv2.createCLAHE()            : CLAHE untuk local tone mapping
- np.power()                   : Gamma correction untuk simulasi exposure
- cv2.normalize()              : Normalisasi gambar

Hasil: Efek HDR sintetis dari single image dengan berbagai teknik
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
print("PERCOBAAN 19: HDR DARI SINGLE IMAGE (PSEUDO-HDR)")
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
    # Mencoba gambar alternatif (gambar gelap)
    path_alt = os.path.join(IMAGE_DIR, "gambar_gelap.png")
    img = cv2.imread(path_alt)

# Jika masih tidak ditemukan, buat gambar sintetis
if img is None:
    print("[INFO] Gambar tidak ditemukan, membuat gambar sintetis...")
    # Membuat gambar sintetis bertema pemandangan
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Membuat langit gradient
    for y in range(200):
        val = int(80 + 120 * (1 - y / 200))
        img[y, :] = [val + 40, val, val - 20]
    # Membuat tanah/rumput
    for y in range(200, 400):
        img[y, :] = [30, 100 + int(40 * (y - 200) / 200), 40]
    # Menambahkan "matahari"
    cv2.circle(img, (450, 80), 40, (50, 200, 255), -1)
    # Noise halus
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Menampilkan informasi gambar
print(f"[INFO] Ukuran gambar: {img.shape}")
print(f"[INFO] Mean brightness: {np.mean(img):.1f}")

# ============================================================
# 2. Membuat multi-exposure sintetis menggunakan gamma correction
# ============================================================
print("\n[LANGKAH 1] Membuat exposure sintetis dengan gamma correction...")

def apply_gamma(image, gamma):
    """Menerapkan gamma correction: output = (input/255)^gamma * 255."""
    # Normalisasi gambar ke rentang 0-1
    normalized = image.astype(np.float64) / 255.0

    # Menerapkan transformasi gamma
    corrected = np.power(normalized, gamma)

    # Mengembalikan ke rentang 0-255
    result = (corrected * 255).astype(np.uint8)

    # Mengembalikan hasil
    return result

# Mendefinisikan nilai gamma untuk simulasi exposure berbeda
# gamma < 1: lebih terang (over-expose)
# gamma > 1: lebih gelap (under-expose)
gammas = [2.5, 1.5, 1.0, 0.7, 0.4]
gamma_labels = ["Under -2", "Under -1", "Normal", "Over +1", "Over +2"]

# Membuat daftar gambar dengan exposure sintetis
synthetic_exposures = []
for g in gammas:
    # Menerapkan gamma correction
    exposed = apply_gamma(img, g)
    # Menambahkan ke daftar
    synthetic_exposures.append(exposed)
    # Menampilkan informasi
    print(f"  Gamma={g:.1f}: mean brightness = {np.mean(exposed):.1f}")

# ============================================================
# 3. Visualisasi exposure sintetis
# ============================================================
print("\n[LANGKAH 2] Visualisasi exposure sintetis...")

# Membuat figure untuk exposure sintetis
fig1, axes1 = plt.subplots(1, 5, figsize=(20, 4))

# Menampilkan setiap exposure
for idx, (exp_img, label) in enumerate(zip(synthetic_exposures, gamma_labels)):
    # Menampilkan pada subplot
    axes1[idx].imshow(cv2.cvtColor(exp_img, cv2.COLOR_BGR2RGB))
    axes1[idx].set_title(f"{label}\n(γ={gammas[idx]})", fontsize=10)
    axes1[idx].axis("off")

# Menambahkan judul utama
fig1.suptitle("Exposure Sintetis via Gamma Correction", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path1 = os.path.join(OUTPUT_DIR, "19_exposure_sintetis.png")
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path1}")

# Menutup figure
plt.close(fig1)

# ============================================================
# 4. Teknik 1: Exposure Fusion (Mertens)
# ============================================================
print("\n[LANGKAH 3] Menerapkan Exposure Fusion (Mertens)...")

# Membuat objek MergeMertens untuk exposure fusion
merge_mertens = cv2.createMergeMertens(
    contrast_weight=1.0,     # Bobot untuk kontras
    saturation_weight=1.0,   # Bobot untuk saturasi
    exposure_weight=1.0      # Bobot untuk kualitas exposure
)

# Menerapkan exposure fusion pada gambar-gambar sintetis
# MergeMertens menerima daftar gambar uint8
fusion_result = merge_mertens.process(synthetic_exposures)

# Hasil fusion dalam rentang 0-1 (float), konversi ke 0-255
fusion_8bit = np.clip(fusion_result * 255, 0, 255).astype(np.uint8)

# Menampilkan informasi
print(f"  Fusion shape: {fusion_result.shape}, range: {fusion_result.min():.3f} - {fusion_result.max():.3f}")
print(f"  Mean brightness setelah fusion: {np.mean(fusion_8bit):.1f}")

# ============================================================
# 5. Teknik 2: Local Tone Mapping dengan CLAHE
# ============================================================
print("\n[LANGKAH 4] Local tone mapping dengan CLAHE...")

def local_tonemapping_clahe(image, clip_limit=3.0, tile_grid=(8, 8)):
    """Menerapkan local tone mapping menggunakan CLAHE pada channel L."""
    # Mengonversi ke LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Memisahkan channel
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Membuat dan menerapkan CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_ch)

    # Menggabungkan kembali
    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Mengembalikan hasil
    return result

# Menerapkan local tone mapping pada gambar original
ltm_result = local_tonemapping_clahe(img, clip_limit=4.0, tile_grid=(8, 8))

# Menampilkan informasi
print(f"  Local tone mapping selesai (CLAHE clip=4.0, tile=8x8)")

# ============================================================
# 6. Teknik 3: Kombinasi Fusion + CLAHE + Color Boost
# ============================================================
print("\n[LANGKAH 5] Kombinasi teknik untuk hasil terbaik...")

# Menerapkan CLAHE pada hasil fusion
combined = local_tonemapping_clahe(fusion_8bit, clip_limit=2.0, tile_grid=(8, 8))

# Menambahkan sedikit peningkatan saturasi (vibrance ringan)
hsv_combined = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV).astype(np.float64)
h_c, s_c, v_c = cv2.split(hsv_combined)

# Menerapkan vibrance ringan (1.2x)
low_sat = 1.0 - (s_c / 255.0)
boost = 1.0 + 0.2 * low_sat
s_c = np.clip(s_c * boost, 0, 255)

# Menggabungkan kembali
hsv_combined = cv2.merge([h_c, s_c, v_c]).astype(np.uint8)
combined = cv2.cvtColor(hsv_combined, cv2.COLOR_HSV2BGR)

# Menampilkan informasi
print(f"  Kombinasi selesai (Fusion + CLAHE + Vibrance)")

# ============================================================
# 7. Perbandingan semua metode
# ============================================================
print("\n[LANGKAH 6] Membuat perbandingan semua metode...")

# Membuat figure perbandingan
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 14))

# Menampilkan original
axes2[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title("Original", fontsize=13, fontweight='bold')
axes2[0, 0].axis("off")

# Menampilkan Exposure Fusion (Mertens)
axes2[0, 1].imshow(cv2.cvtColor(fusion_8bit, cv2.COLOR_BGR2RGB))
axes2[0, 1].set_title("Exposure Fusion (Mertens)", fontsize=13, fontweight='bold')
axes2[0, 1].axis("off")

# Menampilkan Local Tone Mapping CLAHE
axes2[1, 0].imshow(cv2.cvtColor(ltm_result, cv2.COLOR_BGR2RGB))
axes2[1, 0].set_title("Local Tone Mapping (CLAHE)", fontsize=13, fontweight='bold')
axes2[1, 0].axis("off")

# Menampilkan Combined
axes2[1, 1].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
axes2[1, 1].set_title("Combined\n(Fusion+CLAHE+Vibrance)", fontsize=12, fontweight='bold')
axes2[1, 1].axis("off")

# Menambahkan judul utama
fig2.suptitle("Pseudo-HDR dari Single Image: Perbandingan Metode", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path2 = os.path.join(OUTPUT_DIR, "19_perbandingan_pseudo_hdr.png")
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path2}")

# Menutup figure
plt.close(fig2)

# ============================================================
# 8. Variasi parameter Mertens fusion
# ============================================================
print("\n[LANGKAH 7] Variasi parameter Mertens fusion...")

# Mendefinisikan kombinasi parameter
mertens_params = [
    {"contrast_weight": 0.0, "saturation_weight": 1.0, "exposure_weight": 1.0},
    {"contrast_weight": 1.0, "saturation_weight": 0.0, "exposure_weight": 1.0},
    {"contrast_weight": 1.0, "saturation_weight": 1.0, "exposure_weight": 0.0},
    {"contrast_weight": 2.0, "saturation_weight": 1.0, "exposure_weight": 1.0},
    {"contrast_weight": 1.0, "saturation_weight": 2.0, "exposure_weight": 1.0},
    {"contrast_weight": 1.0, "saturation_weight": 1.0, "exposure_weight": 2.0},
]

# Membuat figure untuk variasi parameter Mertens
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))

# Menampilkan setiap variasi
for idx, params in enumerate(mertens_params):
    # Menghitung posisi subplot
    row = idx // 3
    col = idx % 3

    # Membuat objek MergeMertens dengan parameter tertentu
    m = cv2.createMergeMertens(**params)

    # Menerapkan exposure fusion
    res = m.process(synthetic_exposures)
    res_8bit = np.clip(res * 255, 0, 255).astype(np.uint8)

    # Menampilkan hasil
    axes3[row, col].imshow(cv2.cvtColor(res_8bit, cv2.COLOR_BGR2RGB))
    label = f"C={params['contrast_weight']}, S={params['saturation_weight']}, E={params['exposure_weight']}"
    axes3[row, col].set_title(label, fontsize=10)
    axes3[row, col].axis("off")

# Menambahkan judul utama
fig3.suptitle("Variasi Parameter MergeMertens (Exposure Fusion)", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path3 = os.path.join(OUTPUT_DIR, "19_variasi_mertens.png")
fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path3}")

# Menutup figure
plt.close(fig3)

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 19: HDR DARI SINGLE IMAGE")
print("=" * 60)
print("Fungsi-fungsi yang dipelajari:")
print("1. np.power(img/255, gamma) * 255")
print("   → Gamma correction untuk simulasi multi-exposure")
print("   → gamma < 1: terang (over-expose), gamma > 1: gelap")
print("2. cv2.createMergeMertens(contrast_w, saturation_w, exposure_w)")
print("   → Exposure fusion tanpa perlu nilai exposure time")
print("   → Menggabungkan beberapa exposure menjadi satu gambar LDR")
print("3. cv2.createCLAHE(clipLimit, tileGridSize)")
print("   → Local tone mapping untuk meningkatkan detail lokal")
print("4. merge_mertens.process(images)")
print("   → Memproses daftar gambar untuk fusion")
print()
print("Kesimpulan:")
print("- Pseudo-HDR dari single image: buat exposure sintetis → fusion")
print("- Gamma correction efektif untuk simulasi exposure berbeda")
print("- Mertens fusion menghasilkan gambar dengan detail highlight & shadow")
print("- Kombinasi Fusion + CLAHE + Vibrance menghasilkan HDR terbaik")
print("- Parameter Mertens memungkinkan prioritas kontras/saturasi/exposure")
print("=" * 60)
