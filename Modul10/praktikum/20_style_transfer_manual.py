"""
==========================================================================
PERCOBAAN 20: STYLE TRANSFER MANUAL (TANPA DEEP LEARNING)
==========================================================================
Program ini mempelajari teknik transfer gaya (style transfer) secara manual
tanpa menggunakan model deep learning. Pendekatan yang lebih sederhana
namun tetap menghasilkan efek artistik menarik.

Teknik yang dipelajari:
1. Color Transfer - mencocokkan mean dan std channel LAB
   antara gambar konten dan gambar gaya
2. Texture Transfer - menggabungkan edge konten dengan warna gaya
3. Histogram Matching - mencocokkan distribusi warna

Fungsi utama yang dipelajari:
- cv2.cvtColor() (BGR↔LAB)    : Konversi ruang warna untuk color transfer
- cv2.Canny()                  : Deteksi tepi untuk texture transfer
- cv2.calcHist() / equalizeHist() : Histogram matching
- np.mean(), np.std()          : Statistik untuk color transfer

Hasil: Gambar konten dengan gaya visual dari gambar referensi gaya
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
print("PERCOBAAN 20: STYLE TRANSFER MANUAL")
print("=" * 60)

# ============================================================
# 1. Membaca gambar content dan style
# ============================================================

# Mendefinisikan path gambar content (pemandangan)
path_content = os.path.join(IMAGE_DIR, "scene_pemandangan.png")

# Mendefinisikan path gambar style (referensi gaya)
path_style = os.path.join(IMAGE_DIR, "style_reference.png")

# Membaca gambar content
img_content = cv2.imread(path_content)

# Membaca gambar style
img_style = cv2.imread(path_style)

# Memeriksa apakah gambar content berhasil dimuat
if img_content is None:
    print("[INFO] scene_pemandangan.png tidak ditemukan, membuat content sintetis...")
    # Membuat gambar content sintetis (pemandangan)
    img_content = np.zeros((400, 600, 3), dtype=np.uint8)
    # Langit biru
    for y in range(200):
        img_content[y, :] = [180 - y//3, 140 - y//4, 60 + y//4]
    # Tanah hijau
    for y in range(200, 400):
        img_content[y, :] = [40, 120 + (y-200)//4, 50]
    # Menambahkan variasi
    noise = np.random.randint(-10, 10, img_content.shape, dtype=np.int16)
    img_content = np.clip(img_content.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Memeriksa apakah gambar style berhasil dimuat
if img_style is None:
    print("[INFO] style_reference.png tidak ditemukan, membuat style sintetis...")
    # Membuat gambar style sintetis (gaya seni dengan warna cerah)
    img_style = np.zeros((400, 600, 3), dtype=np.uint8)
    # Membuat pola warna-warni (simulasi lukisan)
    for y in range(0, 400, 40):
        for x in range(0, 600, 40):
            # Warna acak terang untuk tiap blok
            color = (
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            )
            cv2.rectangle(img_style, (x, y), (x+40, y+40), color, -1)
    # Menambahkan blur untuk kesan lukisan
    img_style = cv2.GaussianBlur(img_style, (15, 15), 5)

# Menyamakan ukuran gambar style agar sama dengan content
img_style = cv2.resize(img_style, (img_content.shape[1], img_content.shape[0]))

# Menampilkan informasi
print(f"[INFO] Content size: {img_content.shape}")
print(f"[INFO] Style size: {img_style.shape}")

# ============================================================
# 2. Teknik 1: Color Transfer (Reinhard et al.)
# ============================================================
print("\n[LANGKAH 1] Color Transfer menggunakan statistik LAB...")

def color_transfer_lab(content, style):
    """
    Transfer warna dari style ke content menggunakan metode Reinhard.
    Cocokkan mean dan std setiap channel LAB.
    """
    # Mengonversi kedua gambar ke ruang warna LAB (float)
    content_lab = cv2.cvtColor(content, cv2.COLOR_BGR2LAB).astype(np.float64)
    style_lab = cv2.cvtColor(style, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Menghitung mean dan std untuk setiap channel content
    c_mean_l, c_std_l = np.mean(content_lab[:, :, 0]), np.std(content_lab[:, :, 0])
    c_mean_a, c_std_a = np.mean(content_lab[:, :, 1]), np.std(content_lab[:, :, 1])
    c_mean_b, c_std_b = np.mean(content_lab[:, :, 2]), np.std(content_lab[:, :, 2])

    # Menghitung mean dan std untuk setiap channel style
    s_mean_l, s_std_l = np.mean(style_lab[:, :, 0]), np.std(style_lab[:, :, 0])
    s_mean_a, s_std_a = np.mean(style_lab[:, :, 1]), np.std(style_lab[:, :, 1])
    s_mean_b, s_std_b = np.mean(style_lab[:, :, 2]), np.std(style_lab[:, :, 2])

    # Menampilkan statistik
    print(f"  Content LAB mean: L={c_mean_l:.1f}, A={c_mean_a:.1f}, B={c_mean_b:.1f}")
    print(f"  Style LAB mean:   L={s_mean_l:.1f}, A={s_mean_a:.1f}, B={s_mean_b:.1f}")

    # Membuat salinan content LAB untuk modifikasi
    result_lab = content_lab.copy()

    # Menerapkan transfer: result = (content - c_mean) * (s_std / c_std) + s_mean
    # Channel L (Lightness)
    result_lab[:, :, 0] = (result_lab[:, :, 0] - c_mean_l) * (s_std_l / (c_std_l + 1e-6)) + s_mean_l
    # Channel A
    result_lab[:, :, 1] = (result_lab[:, :, 1] - c_mean_a) * (s_std_a / (c_std_a + 1e-6)) + s_mean_a
    # Channel B
    result_lab[:, :, 2] = (result_lab[:, :, 2] - c_mean_b) * (s_std_b / (c_std_b + 1e-6)) + s_mean_b

    # Membatasi ke rentang valid LAB (L: 0-255, A: 0-255, B: 0-255 untuk uint8)
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

    # Mengonversi kembali ke BGR
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    # Mengembalikan hasil
    return result

# Menerapkan color transfer
color_transferred = color_transfer_lab(img_content, img_style)
print(f"  Color transfer selesai")

# ============================================================
# 3. Teknik 2: Texture Transfer (Edge + Color Blend)
# ============================================================
print("\n[LANGKAH 2] Texture Transfer (Edge Content + Color Style)...")

def texture_transfer(content, style, edge_weight=0.4):
    """
    Transfer tekstur: gabungkan struktur tepi content dengan warna style.
    """
    # Mengonversi content ke grayscale
    gray_content = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)

    # Mendeteksi tepi pada content menggunakan Canny
    edges = cv2.Canny(gray_content, 50, 150)

    # Memperlebar tepi agar lebih terlihat
    kernel = np.ones((2, 2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Mengonversi edges ke 3 channel
    edges_3ch = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR).astype(np.float64) / 255.0

    # Membuat versi style yang lebih smooth (seperti lukisan)
    style_smooth = cv2.bilateralFilter(style, 15, 75, 75)

    # Menerapkan color transfer pada style yang sudah smooth
    style_colored = color_transfer_lab(content, style_smooth)

    # Menggabungkan: warna dari style + tepi dari content
    result = style_colored.astype(np.float64)

    # overlay garis tepi (hitam) di atas hasil
    result = result * (1 - edge_weight * edges_3ch)

    # Membatasi ke 0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Mengembalikan hasil beserta komponen
    return result, edges_dilated, style_smooth

# Menerapkan texture transfer
texture_result, edges_vis, style_smooth_vis = texture_transfer(img_content, img_style, edge_weight=0.5)
print(f"  Texture transfer selesai")

# ============================================================
# 4. Teknik 3: Histogram Color Matching
# ============================================================
print("\n[LANGKAH 3] Histogram Color Matching...")

def histogram_matching(content, style):
    """
    Mencocokkan histogram warna content agar mirip dengan style.
    Dilakukan per channel.
    """
    # Membuat salinan content untuk modifikasi
    result = content.copy()

    # Melakukan matching untuk setiap channel (B, G, R)
    for ch in range(3):
        # Menghitung histogram content untuk channel ini
        hist_content = cv2.calcHist([content], [ch], None, [256], [0, 256]).flatten()

        # Menghitung histogram style untuk channel ini
        hist_style = cv2.calcHist([style], [ch], None, [256], [0, 256]).flatten()

        # Menghitung CDF (Cumulative Distribution Function) content
        cdf_content = np.cumsum(hist_content)
        cdf_content = cdf_content / cdf_content[-1]  # Normalisasi ke 0-1

        # Menghitung CDF style
        cdf_style = np.cumsum(hist_style)
        cdf_style = cdf_style / cdf_style[-1]  # Normalisasi ke 0-1

        # Membuat lookup table (LUT) untuk mapping
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Mencari nilai terdekat di CDF style
            diff = np.abs(cdf_style - cdf_content[i])
            lut[i] = np.argmin(diff)

        # Menerapkan LUT pada channel content
        result[:, :, ch] = lut[content[:, :, ch]]

    # Mengembalikan hasil
    return result

# Menerapkan histogram matching
hist_matched = histogram_matching(img_content, img_style)
print(f"  Histogram matching selesai")

# ============================================================
# 5. Visualisasi Input: Content dan Style
# ============================================================
print("\n[LANGKAH 4] Membuat visualisasi input...")

# Membuat figure untuk menampilkan input
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

# Menampilkan content
axes1[0].imshow(cv2.cvtColor(img_content, cv2.COLOR_BGR2RGB))
axes1[0].set_title("Content Image\n(scene_pemandangan.png)", fontsize=12)
axes1[0].axis("off")

# Menampilkan style
axes1[1].imshow(cv2.cvtColor(img_style, cv2.COLOR_BGR2RGB))
axes1[1].set_title("Style Reference\n(style_reference.png)", fontsize=12)
axes1[1].axis("off")

# Menambahkan judul utama
fig1.suptitle("Input: Content dan Style Images", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path1 = os.path.join(OUTPUT_DIR, "20_input_content_style.png")
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path1}")

# Menutup figure
plt.close(fig1)

# ============================================================
# 6. Perbandingan semua metode style transfer
# ============================================================
print("\n[LANGKAH 5] Perbandingan semua metode...")

# Membuat figure perbandingan
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))

# Menampilkan content (original)
axes2[0, 0].imshow(cv2.cvtColor(img_content, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title("Content Original", fontsize=12, fontweight='bold')
axes2[0, 0].axis("off")

# Menampilkan style reference
axes2[0, 1].imshow(cv2.cvtColor(img_style, cv2.COLOR_BGR2RGB))
axes2[0, 1].set_title("Style Reference", fontsize=12, fontweight='bold')
axes2[0, 1].axis("off")

# Menampilkan color transfer (Reinhard)
axes2[0, 2].imshow(cv2.cvtColor(color_transferred, cv2.COLOR_BGR2RGB))
axes2[0, 2].set_title("Color Transfer\n(LAB Reinhard)", fontsize=11)
axes2[0, 2].axis("off")

# Menampilkan texture transfer
axes2[1, 0].imshow(cv2.cvtColor(texture_result, cv2.COLOR_BGR2RGB))
axes2[1, 0].set_title("Texture Transfer\n(Edge + Color)", fontsize=11)
axes2[1, 0].axis("off")

# Menampilkan histogram matching
axes2[1, 1].imshow(cv2.cvtColor(hist_matched, cv2.COLOR_BGR2RGB))
axes2[1, 1].set_title("Histogram Matching\n(Per Channel CDF)", fontsize=11)
axes2[1, 1].axis("off")

# Menampilkan edge map content (komponen texture transfer)
axes2[1, 2].imshow(edges_vis, cmap='gray')
axes2[1, 2].set_title("Edge Map Content\n(Canny)", fontsize=11)
axes2[1, 2].axis("off")

# Menambahkan judul utama
fig2.suptitle("Manual Style Transfer: Perbandingan Semua Metode", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path2 = os.path.join(OUTPUT_DIR, "20_perbandingan_style_transfer.png")
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path2}")

# Menutup figure
plt.close(fig2)

# ============================================================
# 7. Variasi kekuatan transfer (blending dengan original)
# ============================================================
print("\n[LANGKAH 6] Variasi blending strength color transfer...")

# Mendefinisikan daftar alpha blending
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Membuat figure untuk variasi blending
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))

# Menampilkan setiap variasi
for idx, alpha in enumerate(alphas):
    # Menghitung posisi subplot
    row = idx // 3
    col = idx % 3

    # Blending: result = alpha * color_transferred + (1-alpha) * content
    blended = cv2.addWeighted(color_transferred, alpha, img_content, 1 - alpha, 0)

    # Menampilkan hasil
    axes3[row, col].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes3[row, col].set_title(f"Color Transfer α={alpha}", fontsize=11)
    axes3[row, col].axis("off")

# Menambahkan judul utama
fig3.suptitle("Variasi Kekuatan Color Transfer (Blending)", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path3 = os.path.join(OUTPUT_DIR, "20_variasi_blending.png")
fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path3}")

# Menutup figure
plt.close(fig3)

# ============================================================
# 8. Analisis distribusi warna sebelum dan sesudah transfer
# ============================================================
print("\n[LANGKAH 7] Analisis distribusi warna...")

# Membuat figure histogram
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))

# Daftar gambar dan judul untuk histogram
hist_imgs = [img_content, img_style, color_transferred]
hist_titles = ["Content Original", "Style Reference", "Color Transferred"]

# Menampilkan histogram setiap gambar
for idx, (h_img, h_title) in enumerate(zip(hist_imgs, hist_titles)):
    # Menghitung histogram per channel
    for ch, color in enumerate(['blue', 'green', 'red']):
        hist = cv2.calcHist([h_img], [ch], None, [256], [0, 256])
        axes4[idx].plot(hist, color=color, label=color.capitalize(), alpha=0.7)

    # Mengatur label dan judul
    axes4[idx].set_title(f"Histogram - {h_title}", fontsize=11)
    axes4[idx].set_xlim([0, 256])
    axes4[idx].legend()

# Menambahkan judul utama
fig4.suptitle("Distribusi Warna: Content vs Style vs Transferred", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil ke folder output
output_path4 = os.path.join(OUTPUT_DIR, "20_histogram_transfer.png")
fig4.savefig(output_path4, dpi=150, bbox_inches='tight')
print(f"[SAVED] {output_path4}")

# Menutup figure
plt.close(fig4)

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 20: STYLE TRANSFER MANUAL")
print("=" * 60)
print("Teknik-teknik yang dipelajari:")
print("1. Color Transfer (Reinhard et al.)")
print("   → Cocokkan mean & std channel LAB content ke style")
print("   → result = (content - c_mean) * (s_std/c_std) + s_mean")
print("2. Texture Transfer")
print("   → Gabungkan edge content (Canny) + warna style")
print("   → Menghasilkan efek lukisan dengan struktur content")
print("3. Histogram Matching")
print("   → Cocokkan CDF histogram per channel")
print("   → Menghasilkan distribusi warna yang serupa")
print()
print("Fungsi OpenCV yang digunakan:")
print("- cv2.cvtColor(src, cv2.COLOR_BGR2LAB) → konversi LAB")
print("- cv2.Canny(gray, thresh1, thresh2) → deteksi tepi")
print("- cv2.calcHist() → histogram per channel")
print("- cv2.addWeighted() → blending dengan alpha")
print("- cv2.bilateralFilter() → smooth dengan preservasi tepi")
print("- np.mean(), np.std() → statistik untuk color transfer")
print()
print("Kesimpulan:")
print("- Color transfer LAB paling efektif untuk transfer warna global")
print("- Texture transfer mempertahankan struktur konten")
print("- Histogram matching cocokkan distribusi secara presisi")
print("- Blending alpha memungkinkan kontrol kekuatan transfer")
print("- Tanpa deep learning, hasilnya tetap menarik untuk aplikasi sederhana")
print("=" * 60)
