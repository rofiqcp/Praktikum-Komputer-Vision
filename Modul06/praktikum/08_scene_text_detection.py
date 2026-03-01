"""
==========================================================================
PERCOBAAN 8: SCENE TEXT DETECTION
==========================================================================
Program ini mempelajari cara mendeteksi region teks pada gambar scene
menggunakan dua pendekatan: morphological operations dan MSER (Maximally
Stable Extremal Regions). Deteksi teks pada scene berbeda dengan dokumen
karena teks muncul di atas latar belakang yang kompleks.

Konsep yang dipelajari:
- Text detection pipeline: grayscale → gradient → threshold → morphology
- Morphological operations untuk menggabungkan region teks
- MSER (Maximally Stable Extremal Regions) untuk mendeteksi blob stabil
- Filtering kandidat teks berdasarkan aspect ratio dan area
- Perbandingan dua metode deteksi teks

Fungsi utama yang dipelajari:
- cv2.morphologyEx()              : Operasi morfologi (closing, dilation)
- cv2.getStructuringElement()     : Membuat kernel morfologi
- cv2.MSER_create()               : Membuat detektor MSER
- cv2.Sobel()                     : Menghitung gradient gambar
- cv2.findContours()              : Menemukan kontur region teks
- cv2.boundingRect()              : Mendapatkan bounding box kontur

Hasil: Visualisasi deteksi teks dengan metode morfologi dan MSER
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan deteksi teks
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu pemrosesan
import time

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 8: SCENE TEXT DETECTION")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Scene dengan Teks
# ============================================================

print("\n[INFO] Memuat gambar scene dengan teks...")
print("-" * 50)

# Membaca gambar scene yang mengandung teks
img_scene = cv2.imread(os.path.join(IMAGE_DIR, "teks_scene.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_scene is None:
    print("[ERROR] Gambar teks_scene.jpg tidak ditemukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi gambar yang dimuat
print(f"  Ukuran gambar: {img_scene.shape[1]}x{img_scene.shape[0]} piksel")
print(f"  Jumlah channel: {img_scene.shape[2]}")

# Mengkonversi gambar dari BGR ke RGB untuk tampilan matplotlib
img_scene_rgb = cv2.cvtColor(img_scene, cv2.COLOR_BGR2RGB)

# Mengkonversi gambar ke grayscale untuk pemrosesan
gray = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Deteksi Teks Menggunakan Operasi Morfologi
# ============================================================

print("\n[INFO] Metode 1: Deteksi teks dengan operasi morfologi...")
print("-" * 50)

# Mencatat waktu mulai untuk metode morfologi
start_morph = time.time()

# --- Langkah 1: Menghitung gradient menggunakan Sobel ---

# Menghitung gradient horizontal (Sobel X) untuk mendeteksi tepi vertikal pada teks
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Menghitung gradient vertikal (Sobel Y) untuk mendeteksi tepi horizontal
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Menghitung magnitude gradient gabungan
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Menormalisasi magnitude gradient ke range 0-255
gradient_norm = np.uint8(255 * gradient_magnitude / (gradient_magnitude.max() + 1e-7))

# --- Langkah 2: Thresholding pada gradient ---

# Menerapkan Otsu thresholding pada magnitude gradient
_, thresh_grad = cv2.threshold(gradient_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- Langkah 3: Operasi morfologi untuk menggabungkan region teks ---

# Membuat kernel horizontal panjang untuk menghubungkan huruf dalam satu kata
kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))

# Menerapkan closing untuk menghubungkan komponen teks yang berdekatan
morph_close = cv2.morphologyEx(thresh_grad, cv2.MORPH_CLOSE, kernel_horizontal)

# Membuat kernel persegi untuk dilation
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Menerapkan dilation untuk memperbesar area teks
morph_dilate = cv2.dilate(morph_close, kernel_dilate, iterations=2)

# --- Langkah 4: Menemukan kontur region kandidat teks ---

# Menemukan kontur pada hasil morfologi
contours_morph, _ = cv2.findContours(morph_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Menampilkan jumlah kontur yang ditemukan
print(f"  Total kontur ditemukan: {len(contours_morph)}")

# --- Langkah 5: Filtering kontur berdasarkan aspect ratio dan area ---

# Mendapatkan tinggi dan lebar gambar untuk filtering
img_h, img_w = gray.shape[:2]

# Menghitung area minimum dan maksimum untuk kandidat teks
min_area = (img_h * img_w) * 0.0005
max_area = (img_h * img_w) * 0.3

# Mendefinisikan range aspect ratio yang valid untuk teks
min_aspect = 0.5
max_aspect = 15.0

# Membuat salinan gambar untuk menggambar bounding box
img_morph_result = img_scene.copy()

# Membuat list untuk menyimpan region teks yang valid
text_regions_morph = []

# Memfilter kontur berdasarkan kriteria teks
for contour in contours_morph:
    # Mendapatkan bounding box dari kontur
    x, y, w, h = cv2.boundingRect(contour)

    # Menghitung area bounding box
    area = w * h

    # Menghitung aspect ratio (lebar/tinggi)
    aspect_ratio = w / float(h) if h > 0 else 0

    # Memeriksa apakah kontur memenuhi kriteria region teks
    if min_area < area < max_area and min_aspect < aspect_ratio < max_aspect:
        # Menambahkan region valid ke list
        text_regions_morph.append((x, y, w, h))

        # Menggambar bounding box hijau pada region teks
        cv2.rectangle(img_morph_result, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Menghitung waktu proses metode morfologi
time_morph = time.time() - start_morph

# Menampilkan hasil filtering
print(f"  Region teks valid: {len(text_regions_morph)}")
print(f"  Waktu proses: {time_morph*1000:.2f} ms")

# Mengkonversi gambar hasil dari BGR ke RGB untuk matplotlib
img_morph_rgb = cv2.cvtColor(img_morph_result, cv2.COLOR_BGR2RGB)

# ============================================================
# 3. Visualisasi Pipeline Deteksi Teks Morfologi
# ============================================================

print("\n[INFO] Menyimpan visualisasi pipeline morfologi...")

# Membuat figure untuk menampilkan pipeline deteksi morfologi
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Menampilkan gambar asli
axes[0, 0].imshow(img_scene_rgb)
axes[0, 0].set_title("1. Gambar Scene Asli", fontsize=11, fontweight="bold")
axes[0, 0].axis("off")

# Menampilkan gambar grayscale
axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("2. Grayscale", fontsize=11, fontweight="bold")
axes[0, 1].axis("off")

# Menampilkan magnitude gradient
axes[0, 2].imshow(gradient_norm, cmap="gray")
axes[0, 2].set_title("3. Gradient Magnitude (Sobel)", fontsize=11, fontweight="bold")
axes[0, 2].axis("off")

# Menampilkan hasil thresholding gradient
axes[1, 0].imshow(thresh_grad, cmap="gray")
axes[1, 0].set_title("4. Otsu Threshold pada Gradient", fontsize=11, fontweight="bold")
axes[1, 0].axis("off")

# Menampilkan hasil operasi morfologi
axes[1, 1].imshow(morph_dilate, cmap="gray")
axes[1, 1].set_title("5. Setelah Closing + Dilation", fontsize=11, fontweight="bold")
axes[1, 1].axis("off")

# Menampilkan hasil akhir dengan bounding box
axes[1, 2].imshow(img_morph_rgb)
axes[1, 2].set_title(f"6. Deteksi Teks ({len(text_regions_morph)} region)", fontsize=11, fontweight="bold")
axes[1, 2].axis("off")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 8: Pipeline Deteksi Teks - Metode Morfologi",
             fontsize=13, fontweight="bold")

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan visualisasi metode morfologi ke file
output_path_1 = os.path.join(OUTPUT_DIR, "08_text_detection_morphology.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure untuk membebaskan memori
plt.close()

# ============================================================
# 4. Deteksi Teks Menggunakan MSER
# ============================================================

print("\n[INFO] Metode 2: Deteksi teks dengan MSER...")
print("-" * 50)

# Mencatat waktu mulai untuk metode MSER
start_mser = time.time()

# Membuat detektor MSER dengan parameter default
mser = cv2.MSER_create()

# Mengatur parameter MSER untuk deteksi teks
mser.setMinArea(60)
mser.setMaxArea(14400)
mser.setDelta(5)

# Mendeteksi region stabil pada gambar grayscale
regions_mser, bboxes_mser = mser.detectRegions(gray)

# Menampilkan jumlah region MSER yang ditemukan
print(f"  Total region MSER: {len(regions_mser)}")

# Membuat salinan gambar untuk menampilkan semua region MSER
img_mser_all = img_scene.copy()

# Membuat mask kosong untuk menggambar semua region MSER
mask_mser = np.zeros_like(gray)

# Menggambar semua region MSER pada mask
for region in regions_mser:
    # Menggambar titik-titik region pada mask
    for point in region:
        # Mengatur piksel pada posisi region menjadi putih
        mask_mser[point[1], point[0]] = 255

# --- Filtering region MSER berdasarkan karakteristik teks ---

# Membuat list untuk menyimpan bounding box teks valid
text_regions_mser = []

# Membuat salinan gambar untuk menggambar bounding box valid
img_mser_result = img_scene.copy()

# Memfilter bounding box MSER berdasarkan aspect ratio dan area
for bbox in bboxes_mser:
    # Mengambil koordinat dan ukuran bounding box
    x, y, w, h = bbox

    # Menghitung area bounding box
    area = w * h

    # Menghitung aspect ratio bounding box
    aspect_ratio = w / float(h) if h > 0 else 0

    # Menghitung extent (rasio area kontur terhadap bounding box)
    min_dim = min(w, h)
    max_dim = max(w, h)

    # Memeriksa apakah dimensi minimum cukup besar (filter noise kecil)
    if min_dim < 8:
        continue

    # Memeriksa apakah aspect ratio sesuai dengan karakter teks (0.1 - 10.0)
    if 0.1 < aspect_ratio < 10.0 and area > 100 and area < max_area:
        # Menambahkan ke list region valid
        text_regions_mser.append((x, y, w, h))

        # Menggambar bounding box merah pada region valid
        cv2.rectangle(img_mser_result, (x, y), (x + w, y + h), (0, 0, 255), 2)

# --- Grouping region MSER yang berdekatan ---

# Mengelompokkan bounding box yang saling overlap menggunakan groupRectangles
if len(text_regions_mser) > 0:
    # Mengkonversi list ke format yang sesuai untuk groupRectangles
    rects_for_group = [(x, y, x + w, y + h) for (x, y, w, h) in text_regions_mser]

    # Membuat array numpy dari bounding box
    rects_array = np.array(rects_for_group)

    # Mengelompokkan box yang berdekatan secara manual
    grouped_boxes = []

    # Mengurutkan bounding box berdasarkan posisi y lalu x
    sorted_indices = np.lexsort((rects_array[:, 0], rects_array[:, 1]))

    # Membuat flag untuk menandai box yang sudah digabungkan
    used = np.zeros(len(rects_array), dtype=bool)

    # Melakukan iterasi untuk mengelompokkan box berdekatan
    for i in sorted_indices:
        # Melewati box yang sudah digabungkan
        if used[i]:
            continue

        # Mendapatkan koordinat box saat ini
        x1, y1, x2, y2 = rects_array[i]

        # Mencari box lain yang berdekatan secara horizontal
        for j in sorted_indices:
            # Melewati box yang sama atau sudah digunakan
            if i == j or used[j]:
                continue

            # Mendapatkan koordinat box pembanding
            bx1, by1, bx2, by2 = rects_array[j]

            # Memeriksa apakah box berada pada baris yang sama (selisih y kecil)
            if abs(y1 - by1) < max(y2 - y1, by2 - by1) * 0.5:
                # Memeriksa apakah box berdekatan secara horizontal
                if abs(x2 - bx1) < (x2 - x1) * 2 or abs(bx2 - x1) < (bx2 - bx1) * 2:
                    # Menggabungkan kedua box
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)

                    # Menandai box sebagai sudah digabungkan
                    used[j] = True

        # Menandai box utama sebagai sudah digunakan
        used[i] = True

        # Menambahkan box gabungan ke list
        grouped_boxes.append((x1, y1, x2 - x1, y2 - y1))

    # Menampilkan jumlah group yang terbentuk
    print(f"  Region terfilter: {len(text_regions_mser)}")
    print(f"  Group teks: {len(grouped_boxes)}")
else:
    # Menginisialisasi list kosong jika tidak ada region valid
    grouped_boxes = []
    print("  Tidak ada region teks yang terdeteksi")

# Menghitung waktu proses metode MSER
time_mser = time.time() - start_mser

# Menampilkan waktu proses MSER
print(f"  Waktu proses: {time_mser*1000:.2f} ms")

# Mengkonversi gambar hasil MSER ke RGB untuk matplotlib
img_mser_rgb = cv2.cvtColor(img_mser_result, cv2.COLOR_BGR2RGB)

# ============================================================
# 5. Visualisasi Hasil Deteksi MSER
# ============================================================

print("\n[INFO] Menyimpan visualisasi deteksi MSER...")

# Membuat figure untuk menampilkan hasil MSER
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Menampilkan gambar asli
axes[0, 0].imshow(img_scene_rgb)
axes[0, 0].set_title("1. Gambar Scene Asli", fontsize=11, fontweight="bold")
axes[0, 0].axis("off")

# Menampilkan mask region MSER
axes[0, 1].imshow(mask_mser, cmap="gray")
axes[0, 1].set_title(f"2. Region MSER ({len(regions_mser)} region)", fontsize=11, fontweight="bold")
axes[0, 1].axis("off")

# Menampilkan bounding box MSER terfilter
axes[1, 0].imshow(img_mser_rgb)
axes[1, 0].set_title(f"3. MSER Terfilter ({len(text_regions_mser)} box)", fontsize=11, fontweight="bold")
axes[1, 0].axis("off")

# Membuat gambar untuk menampilkan grouped boxes
img_grouped = img_scene.copy()

# Menggambar bounding box dari grouped boxes
for (gx, gy, gw, gh) in grouped_boxes:
    # Menggambar kotak biru tebal pada group teks
    cv2.rectangle(img_grouped, (gx, gy), (gx + gw, gy + gh), (255, 0, 0), 3)

# Mengkonversi gambar grouped ke RGB
img_grouped_rgb = cv2.cvtColor(img_grouped, cv2.COLOR_BGR2RGB)

# Menampilkan grouped bounding boxes
axes[1, 1].imshow(img_grouped_rgb)
axes[1, 1].set_title(f"4. Group Teks ({len(grouped_boxes)} group)", fontsize=11, fontweight="bold")
axes[1, 1].axis("off")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 8: Deteksi Teks - Metode MSER",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi MSER ke file
output_path_2 = os.path.join(OUTPUT_DIR, "08_text_detection_mser.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 6. Perbandingan Metode dan Crop Region Teks
# ============================================================

print("\n[INFO] Mengekstrak dan membandingkan region teks...")
print("-" * 50)

# Menggabungkan semua region dari kedua metode untuk cropping
all_regions = text_regions_morph + grouped_boxes

# Mengurutkan region berdasarkan area (terbesar ke terkecil)
all_regions_sorted = sorted(all_regions, key=lambda r: r[2] * r[3], reverse=True)

# Mengambil maksimal 8 region terbesar untuk ditampilkan
top_regions = all_regions_sorted[:min(8, len(all_regions_sorted))]

# Menentukan jumlah region yang akan ditampilkan
n_regions = len(top_regions)

# Memeriksa apakah ada region yang bisa diekstrak
if n_regions > 0:
    # Menghitung jumlah baris dan kolom untuk subplot
    n_cols = min(4, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols

    # Membuat figure untuk menampilkan crop region teks
    fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(4 * n_cols, 4 * (n_rows + 1)))

    # Memastikan axes bisa diakses secara konsisten
    if n_rows + 1 == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Membuat gambar perbandingan kedua metode
    img_compare = img_scene.copy()

    # Menggambar bounding box hijau untuk metode morfologi
    for (x, y, w, h) in text_regions_morph:
        # Menggambar kotak hijau (morfologi)
        cv2.rectangle(img_compare, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Menggambar bounding box merah untuk metode MSER
    for (x, y, w, h) in grouped_boxes:
        # Menggambar kotak merah (MSER)
        cv2.rectangle(img_compare, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Mengkonversi gambar perbandingan ke RGB
    img_compare_rgb = cv2.cvtColor(img_compare, cv2.COLOR_BGR2RGB)

    # Menampilkan perbandingan di baris pertama (gabung semua kolom)
    for col in range(n_cols):
        # Menyembunyikan subplot individual
        axes[0, col].axis("off")

    # Menggunakan subplot pertama untuk menampilkan perbandingan full
    axes[0, 0].imshow(img_compare_rgb)
    axes[0, 0].set_title("Perbandingan: Hijau=Morfologi, Merah=MSER", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    # Menampilkan crop dari setiap region teks
    for i, (x, y, w, h) in enumerate(top_regions):
        # Menghitung indeks baris dan kolom subplot
        row = (i // n_cols) + 1
        col = i % n_cols

        # Memastikan koordinat crop valid
        y1 = max(0, y)
        y2 = min(img_scene.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(img_scene.shape[1], x + w)

        # Melakukan crop pada region teks
        cropped = img_scene[y1:y2, x1:x2]

        # Mengkonversi crop ke RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Menampilkan region teks yang di-crop
        axes[row, col].imshow(cropped_rgb)
        axes[row, col].set_title(f"Region {i+1} ({w}x{h})", fontsize=9)
        axes[row, col].axis("off")

    # Menyembunyikan subplot kosong yang tersisa
    for i in range(n_regions, n_rows * n_cols):
        # Menghitung posisi subplot kosong
        row = (i // n_cols) + 1
        col = i % n_cols
        # Menyembunyikan subplot
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].axis("off")

    # Menambahkan judul utama figure
    plt.suptitle("Percobaan 8: Crop Region Teks yang Terdeteksi",
                 fontsize=13, fontweight="bold")

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan visualisasi region teks ke file
    output_path_3 = os.path.join(OUTPUT_DIR, "08_text_regions.png")
    plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
    print(f"[OUTPUT] Disimpan: {output_path_3}")

    # Menutup figure
    plt.close()
else:
    # Menampilkan pesan jika tidak ada region teks yang ditemukan
    print("  Tidak ada region teks yang bisa diekstrak.")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 8")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Metode Morfologi untuk Deteksi Teks:")
print("     - cv2.Sobel(): menghitung gradient untuk mendeteksi tepi teks")
print("     - cv2.threshold(OTSU): binarisasi adaptif gradient")
print("     - cv2.getStructuringElement(MORPH_RECT): kernel morfologi")
print("     - cv2.morphologyEx(MORPH_CLOSE): menghubungkan huruf")
print("     - cv2.dilate(): memperbesar area teks")
print("     - cv2.findContours(): menemukan region kandidat teks")
print("  2. Metode MSER untuk Deteksi Teks:")
print("     - cv2.MSER_create(): membuat detektor MSER")
print("     - mser.detectRegions(): mendeteksi region stabil")
print("     - MSER mendeteksi blob yang stabil terhadap threshold")
print("  3. Filtering Region Teks:")
print("     - Aspect ratio: teks horizontal biasanya lebar > tinggi")
print("     - Area: terlalu kecil = noise, terlalu besar = bukan teks")
print("     - Grouping: menggabungkan karakter menjadi kata/baris")
print(f"\nHasil deteksi:")
print(f"  - Morfologi : {len(text_regions_morph)} region ({time_morph*1000:.2f} ms)")
print(f"  - MSER      : {len(grouped_boxes)} group ({time_mser*1000:.2f} ms)")
print("=" * 60)
