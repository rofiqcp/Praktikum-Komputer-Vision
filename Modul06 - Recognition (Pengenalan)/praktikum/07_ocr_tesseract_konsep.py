"""
==========================================================================
PERCOBAAN 7: OCR DENGAN TESSERACT KONSEP (TEMPLATE MATCHING)
==========================================================================
Program ini mempelajari konsep OCR (Optical Character Recognition) dan
mengimplementasikannya menggunakan pendekatan template matching. Karena
Tesseract mungkin tidak terinstal, program mendemonstrasikan konsep
pengenalan karakter menggunakan teknik template matching dasar.

Konsep yang dipelajari:
- OCR pipeline: segmentasi → recognition → post-processing
- Template matching: mencocokkan gambar karakter dengan template
- Character segmentation: memisahkan karakter individu
- Connected component analysis untuk segmentasi teks
- Evaluasi akurasi pengenalan karakter

Fungsi utama yang dipelajari:
- cv2.putText()                   : Membuat template karakter
- cv2.matchTemplate()             : Template matching per karakter
- cv2.connectedComponentsWithStats() : Connected component analysis
- cv2.boundingRect()              : Mendapatkan bounding box kontur
- cv2.findContours()              : Segmentasi komponen terhubung

Hasil: Visualisasi template, segmentasi, dan hasil OCR
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 7: OCR DENGAN TESSERACT KONSEP (TEMPLATE MATCHING)")
print("=" * 60)

# ============================================================
# 1. Membuat Template Karakter (A-Z, 0-9)
# ============================================================

print("\n[INFO] Membuat template karakter A-Z, 0-9...")
print("-" * 50)

# Mendefinisikan ukuran template karakter
TEMPLATE_SIZE = (30, 40)

# Mendefinisikan daftar karakter yang akan dibuat templatenya
karakter_huruf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
karakter_angka = list("0123456789")
semua_karakter = karakter_huruf + karakter_angka

# Mendefinisikan dictionary untuk menyimpan template
templates = {}

# Membuat template untuk setiap karakter menggunakan cv2.putText
for char in semua_karakter:
    # Membuat canvas putih untuk template karakter
    template = np.ones((TEMPLATE_SIZE[1], TEMPLATE_SIZE[0]), dtype=np.uint8) * 255

    # Mendefinisikan parameter font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # Mendapatkan ukuran teks untuk centering
    (tw, th), baseline = cv2.getTextSize(char, font, font_scale, thickness)

    # Menghitung posisi untuk menempatkan teks di tengah template
    tx = (TEMPLATE_SIZE[0] - tw) // 2
    ty = (TEMPLATE_SIZE[1] + th) // 2

    # Menuliskan karakter pada template (hitam di atas putih)
    cv2.putText(template, char, (tx, ty), font, font_scale, 0, thickness)

    # Menyimpan template ke dictionary
    templates[char] = template

# Menampilkan jumlah template yang dibuat
print(f"  Total template: {len(templates)} karakter")
print(f"  Ukuran template: {TEMPLATE_SIZE[0]}x{TEMPLATE_SIZE[1]} piksel")

# ============================================================
# 2. Visualisasi Template Karakter
# ============================================================

# Membuat figure untuk menampilkan semua template karakter
fig, axes = plt.subplots(4, 9, figsize=(18, 8))
axes_flat = axes.flatten()

# Menampilkan setiap template karakter
for i, char in enumerate(semua_karakter):
    if i < len(axes_flat):
        # Menampilkan template
        axes_flat[i].imshow(templates[char], cmap='gray', vmin=0, vmax=255)
        axes_flat[i].set_title(char, fontsize=10, fontweight='bold')
        axes_flat[i].axis("off")

# Menyembunyikan subplot kosong
for i in range(len(semua_karakter), len(axes_flat)):
    axes_flat[i].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 7: Template Karakter untuk OCR\n"
             "Dibuat menggunakan cv2.putText() - A-Z, 0-9",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi template
output_path_1 = os.path.join(OUTPUT_DIR, "07_template_karakter.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 3. Segmentasi Karakter pada Gambar Teks
# ============================================================

print("\n[INFO] Melakukan segmentasi karakter...")
print("-" * 50)

# Membaca gambar teks cetak
img_teks = cv2.imread(os.path.join(IMAGE_DIR, "teks_printed.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_teks is None:
    print("[ERROR] Gambar teks tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Mengkonversi gambar ke grayscale
gray_teks = cv2.cvtColor(img_teks, cv2.COLOR_BGR2GRAY)

# Menerapkan Otsu thresholding untuk binarisasi
_, binary_teks = cv2.threshold(gray_teks, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Menerapkan operasi morfologi untuk membersihkan hasil binarisasi
kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
binary_clean = cv2.morphologyEx(binary_teks, cv2.MORPH_OPEN, kernel_clean)

# Menerapkan dilasi ringan untuk menghubungkan bagian karakter yang terputus
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
binary_dilated = cv2.dilate(binary_clean, kernel_dilate, iterations=1)

# ============================================================
# 3a. Connected Component Analysis
# ============================================================

# Menerapkan connected component analysis
# Output: jumlah komponen, label map, stats (x,y,w,h,area), centroids
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary_dilated, connectivity=8
)

# Menampilkan jumlah komponen yang ditemukan
print(f"  Jumlah connected components: {num_labels - 1} (tanpa background)")

# Memfilter komponen berdasarkan ukuran (terlalu kecil = noise, terlalu besar = background)
karakter_rects = []

# Memproses setiap komponen (mulai dari 1, karena 0 = background)
for i in range(1, num_labels):
    # Mengekstrak statistik komponen
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    # Memfilter berdasarkan ukuran dan rasio aspek
    # Karakter biasanya memiliki tinggi > lebar dan area tertentu
    if 10 < area < 5000 and 5 < h < 100 and 3 < w < 80:
        # Menghitung rasio aspek
        aspect_ratio = w / float(h)

        # Memfilter rasio aspek yang wajar untuk karakter
        if 0.1 < aspect_ratio < 2.0:
            karakter_rects.append((x, y, w, h, area))

# Mengurutkan karakter dari kiri ke kanan, atas ke bawah
karakter_rects.sort(key=lambda r: (r[1] // 30, r[0]))

# Menampilkan jumlah karakter yang terfilter
print(f"  Karakter tersegmentasi: {len(karakter_rects)}")

# ============================================================
# 3b. Alternatif: Segmentasi dengan findContours
# ============================================================

# Mencari kontur pada gambar biner
contours, hierarchy = cv2.findContours(
    binary_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Memfilter kontur berdasarkan ukuran
kontur_rects = []
for cnt in contours:
    # Mendapatkan bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)

    # Menghitung area kontur
    area = cv2.contourArea(cnt)

    # Memfilter berdasarkan ukuran
    if 10 < area < 5000 and 5 < h < 100 and 3 < w < 80:
        kontur_rects.append((x, y, w, h))

# Mengurutkan kontur
kontur_rects.sort(key=lambda r: (r[1] // 30, r[0]))

# Menampilkan jumlah kontur
print(f"  Kontur karakter (findContours): {len(kontur_rects)}")

# ============================================================
# 4. Visualisasi Segmentasi Karakter
# ============================================================

# Membuat figure untuk visualisasi segmentasi
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Menampilkan gambar teks asli
axes[0, 0].imshow(cv2.cvtColor(img_teks, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Teks Asli", fontsize=11)
axes[0, 0].axis("off")

# Menampilkan hasil binarisasi
axes[0, 1].imshow(binary_teks, cmap='gray')
axes[0, 1].set_title("Binarisasi (Otsu Invers)", fontsize=11)
axes[0, 1].axis("off")

# Menampilkan connected component labels dengan warna
# Membuat gambar label berwarna
label_color = np.zeros((*labels.shape, 3), dtype=np.uint8)
for i in range(1, num_labels):
    # Memberikan warna acak untuk setiap komponen
    warna = np.random.randint(50, 255, 3).tolist()
    label_color[labels == i] = warna

axes[1, 0].imshow(label_color)
axes[1, 0].set_title(f"Connected Components\n({num_labels-1} komponen)", fontsize=11)
axes[1, 0].axis("off")

# Menampilkan hasil segmentasi dengan bounding box
img_segmen = img_teks.copy()

# Menggambar bounding box untuk setiap karakter tersegmentasi
for i, (x, y, w, h, area) in enumerate(karakter_rects[:50]):
    # Mendefinisikan warna berdasarkan indeks
    warna = (0, 255, 0) if i % 2 == 0 else (0, 200, 255)

    # Menggambar rectangle
    cv2.rectangle(img_segmen, (x, y), (x + w, y + h), warna, 1)

axes[1, 1].imshow(cv2.cvtColor(img_segmen, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"Segmentasi Karakter\n({len(karakter_rects)} karakter)", fontsize=11)
axes[1, 1].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 7: Segmentasi Karakter untuk OCR\n"
             "Connected Component Analysis + Contour Detection",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi segmentasi
output_path_2 = os.path.join(OUTPUT_DIR, "07_segmentasi_karakter.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 5. Pengenalan Karakter dengan Template Matching
# ============================================================

print("\n[INFO] Melakukan pengenalan karakter dengan template matching...")
print("-" * 50)

# Mendefinisikan fungsi untuk mengenali satu karakter
def kenali_karakter(roi_char, templates, threshold=0.5):
    """
    Mengenali satu karakter menggunakan template matching.
    Input: roi_char = gambar ROI karakter (grayscale, biner)
    Output: (karakter_prediksi, confidence)
    """
    # Meresize ROI ke ukuran template
    roi_resized = cv2.resize(roi_char, (TEMPLATE_SIZE[0], TEMPLATE_SIZE[1]))

    # Menginversi jika perlu (memastikan teks putih pada background hitam → hitam pada putih)
    if np.mean(roi_resized) < 128:
        roi_resized = cv2.bitwise_not(roi_resized)

    # Menyimpan skor terbaik
    best_char = '?'
    best_score = -1

    # Mencocokkan ROI dengan setiap template
    for char, template in templates.items():
        # Menerapkan template matching menggunakan metode korelasi ternormalisasi
        result = cv2.matchTemplate(roi_resized, template, cv2.TM_CCOEFF_NORMED)

        # Mendapatkan skor matching (nilai tertinggi)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # Memperbarui karakter terbaik jika skor lebih tinggi
        if max_val > best_score:
            best_score = max_val
            best_char = char

    # Mengembalikan karakter prediksi dan confidence
    return best_char, best_score

# Menyiapkan list untuk menyimpan hasil pengenalan
hasil_recognition = []

# Membuat gambar teks tes sederhana untuk demonstrasi
# Membuat gambar dengan teks yang jelas untuk dikenali
img_test_ocr = np.ones((80, 500, 3), dtype=np.uint8) * 255
teks_tes = "ABCD1234"

# Menuliskan teks pada gambar test
cv2.putText(img_test_ocr, teks_tes, (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

# Mengkonversi ke grayscale
gray_test = cv2.cvtColor(img_test_ocr, cv2.COLOR_BGR2GRAY)

# Menerapkan binarisasi invers
_, binary_test = cv2.threshold(gray_test, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Mencari kontur karakter pada gambar test
contours_test, _ = cv2.findContours(binary_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mendapatkan bounding box setiap karakter dan mengurutkan
char_boxes = []
for cnt in contours_test:
    x, y, w, h = cv2.boundingRect(cnt)
    if h > 15 and w > 5:
        char_boxes.append((x, y, w, h))

# Mengurutkan dari kiri ke kanan
char_boxes.sort(key=lambda b: b[0])

# Menampilkan jumlah karakter terdeteksi
print(f"  Teks asli: '{teks_tes}'")
print(f"  Karakter terdeteksi: {len(char_boxes)}")

# Mengenali setiap karakter
teks_hasil = ""
for i, (x, y, w, h) in enumerate(char_boxes):
    # Mengekstrak ROI karakter dari gambar grayscale
    roi = gray_test[y:y+h, x:x+w]

    # Mengenali karakter menggunakan template matching
    char_pred, confidence = kenali_karakter(roi, templates)

    # Menyimpan hasil
    hasil_recognition.append((char_pred, confidence, roi))
    teks_hasil += char_pred

    # Menampilkan hasil per karakter
    if i < len(teks_tes):
        benar = "V" if char_pred == teks_tes[i] else "X"
        print(f"  {benar} Posisi {i}: asli='{teks_tes[i]}' prediksi='{char_pred}' confidence={confidence:.3f}")

# Menampilkan hasil keseluruhan
print(f"\n  Hasil OCR: '{teks_hasil}'")
print(f"  Teks asli: '{teks_tes}'")

# Menghitung akurasi per karakter
benar_count = sum(1 for p, a in zip(teks_hasil, teks_tes) if p == a)
total_count = min(len(teks_hasil), len(teks_tes))
akurasi = benar_count / total_count * 100 if total_count > 0 else 0
print(f"  Akurasi : {benar_count}/{total_count} = {akurasi:.1f}%")

# ============================================================
# 6. Pengenalan pada Gambar Teks Printed
# ============================================================

print("\n[INFO] Menerapkan OCR pada gambar teks printed...")
print("-" * 50)

# Mengambil baris pertama teks dari gambar printed
# Memotong area baris pertama (bagian atas gambar)
baris_pertama = img_teks[20:90, 20:500]

# Mengkonversi ke grayscale
gray_baris = cv2.cvtColor(baris_pertama, cv2.COLOR_BGR2GRAY)

# Menerapkan binarisasi
_, binary_baris = cv2.threshold(gray_baris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Mencari kontur
contours_baris, _ = cv2.findContours(binary_baris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mendapatkan bounding box dan mengurutkan
boxes_baris = []
for cnt in contours_baris:
    x, y, w, h = cv2.boundingRect(cnt)
    if h > 10 and w > 3:
        boxes_baris.append((x, y, w, h))

# Mengurutkan dari kiri ke kanan
boxes_baris.sort(key=lambda b: b[0])

# Mengenali setiap karakter pada baris pertama
teks_baris = ""
skor_baris = []
for (x, y, w, h) in boxes_baris:
    # Mengekstrak ROI
    roi = gray_baris[y:y+h, x:x+w]

    # Mengenali karakter
    char_pred, confidence = kenali_karakter(roi, templates)
    teks_baris += char_pred
    skor_baris.append(confidence)

# Menampilkan hasil OCR baris pertama
print(f"  Hasil OCR baris pertama: '{teks_baris}'")
print(f"  Jumlah karakter: {len(teks_baris)}")
if len(skor_baris) > 0:
    print(f"  Confidence rata-rata: {np.mean(skor_baris):.3f}")

# ============================================================
# 7. Visualisasi Hasil OCR
# ============================================================

# Membuat figure untuk visualisasi hasil OCR
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Baris 1: Gambar test OCR dengan bounding box dan hasil
img_test_vis = img_test_ocr.copy()
for i, (x, y, w, h) in enumerate(char_boxes):
    # Menggambar bounding box
    cv2.rectangle(img_test_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Menambahkan label prediksi
    if i < len(hasil_recognition):
        char_pred, conf, _ = hasil_recognition[i]
        warna_label = (0, 180, 0) if (i < len(teks_tes) and char_pred == teks_tes[i]) else (0, 0, 255)
        cv2.putText(img_test_vis, f"{char_pred}", (x, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna_label, 1)

axes[0].imshow(cv2.cvtColor(img_test_vis, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Gambar Test: '{teks_tes}' → OCR: '{teks_hasil}' (Akurasi: {akurasi:.0f}%)",
                  fontsize=12)
axes[0].axis("off")

# Baris 2: Gambar teks printed dengan segmentasi
img_printed_vis = baris_pertama.copy()
for (x, y, w, h) in boxes_baris:
    cv2.rectangle(img_printed_vis, (x, y), (x + w, y + h), (0, 200, 0), 1)

axes[1].imshow(cv2.cvtColor(img_printed_vis, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Teks Printed → OCR: '{teks_baris}'", fontsize=12)
axes[1].axis("off")

# Baris 3: Confidence score per karakter
if len(hasil_recognition) > 0:
    chars_label = [f"'{r[0]}'" for r in hasil_recognition]
    scores = [r[1] for r in hasil_recognition]

    # Mendefinisikan warna berdasarkan kebenaran
    colors = []
    for i, (char_pred, _, _) in enumerate(hasil_recognition):
        if i < len(teks_tes) and char_pred == teks_tes[i]:
            colors.append('green')
        else:
            colors.append('red')

    # Menampilkan bar chart confidence
    bars = axes[2].bar(range(len(scores)), scores, color=colors, alpha=0.7)
    axes[2].set_xticks(range(len(chars_label)))
    axes[2].set_xticklabels(chars_label, fontsize=10)
    axes[2].set_ylabel("Confidence Score")
    axes[2].set_title("Confidence Score per Karakter (Hijau=Benar, Merah=Salah)", fontsize=12)
    axes[2].set_ylim([0, 1])
    axes[2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Threshold 0.5')
    axes[2].legend()
else:
    axes[2].text(0.5, 0.5, "Tidak ada karakter yang dikenali",
                 ha='center', va='center', transform=axes[2].transAxes, fontsize=14)
    axes[2].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 7: Hasil OCR dengan Template Matching\n"
             "Segmentasi karakter + Pencocokan template",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil OCR
output_path_3 = os.path.join(OUTPUT_DIR, "07_ocr_hasil.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 7")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Template Matching untuk OCR:")
print("     - cv2.putText(): membuat template karakter (A-Z, 0-9)")
print("     - cv2.matchTemplate(TM_CCOEFF_NORMED): mencocokkan template")
print("       → Skor 0-1, semakin tinggi = semakin mirip")
print("  2. Character Segmentation:")
print("     a. cv2.connectedComponentsWithStats():")
print("        - Menghitung komponen terhubung (setiap karakter)")
print("        - Return: labels, stats (x,y,w,h,area), centroids")
print("     b. cv2.findContours() + cv2.boundingRect():")
print("        - Alternatif untuk menemukan bounding box karakter")
print("  3. OCR Pipeline:")
print("     a. Preprocessing: grayscale → binarisasi → cleaning")
print("     b. Segmentasi: isolasi karakter individu")
print("     c. Recognition: template matching per karakter")
print("     d. Post-processing: gabungkan menjadi teks")
print("  4. Keterbatasan Template Matching:")
print("     - Sensitif terhadap ukuran dan font")
print("     - Sulit menangani variasi tulisan tangan")
print("     - Tesseract/DNN lebih akurat untuk kasus nyata")
print(f"\nHasil OCR:")
print(f"  Teks test: '{teks_tes}' → OCR: '{teks_hasil}'")
print(f"  Akurasi: {akurasi:.1f}%")
print("=" * 60)
