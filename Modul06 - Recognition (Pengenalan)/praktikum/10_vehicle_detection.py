"""
==========================================================================
PERCOBAAN 10: VEHICLE DETECTION (DETEKSI KENDARAAN)
==========================================================================
Program ini mempelajari cara mendeteksi dan mengklasifikasikan kendaraan
pada gambar menggunakan kombinasi teknik: deteksi berbasis warna, analisis
kontur, dan klasifikasi berdasarkan ukuran. Program mendemonstrasikan
pipeline deteksi kendaraan tanpa deep learning.

Konsep yang dipelajari:
- Deteksi berbasis warna: isolasi warna kendaraan di ruang warna HSV
- Analisis kontur: filtering berdasarkan area dan aspect ratio
- Klasifikasi kendaraan berdasarkan ukuran (mobil, bus, truk)
- Background model sederhana: konsep foreground vs background
- Penghitungan otomatis jumlah kendaraan per jenis

Fungsi utama yang dipelajari:
- cv2.cvtColor(COLOR_BGR2HSV)     : Konversi ke ruang warna HSV
- cv2.inRange()                   : Segmentasi berdasarkan range warna
- cv2.findContours()              : Menemukan kontur objek
- cv2.contourArea()               : Menghitung luas kontur
- cv2.boundingRect()              : Bounding box kontur
- cv2.convexHull()                : Menghitung convex hull kontur

Hasil: Visualisasi deteksi kendaraan berbasis warna, kontur, dan klasifikasi
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan hitungan
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
print("PERCOBAAN 10: VEHICLE DETECTION (DETEKSI KENDARAAN)")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Kendaraan
# ============================================================

print("\n[INFO] Memuat gambar kendaraan...")
print("-" * 50)

# Membaca gambar kendaraan dari file
img_vehicle = cv2.imread(os.path.join(IMAGE_DIR, "kendaraan.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_vehicle is None:
    print("[ERROR] Gambar kendaraan.jpg tidak ditemukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi gambar
print(f"  Ukuran gambar: {img_vehicle.shape[1]}x{img_vehicle.shape[0]} piksel")

# Mengkonversi gambar dari BGR ke RGB untuk tampilan matplotlib
img_vehicle_rgb = cv2.cvtColor(img_vehicle, cv2.COLOR_BGR2RGB)

# Mengkonversi gambar ke grayscale
gray = cv2.cvtColor(img_vehicle, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar ke ruang warna HSV untuk segmentasi warna
hsv = cv2.cvtColor(img_vehicle, cv2.COLOR_BGR2HSV)

# Mendapatkan dimensi gambar
img_h, img_w = gray.shape[:2]

# ============================================================
# 2. Deteksi Kendaraan Berbasis Warna
# ============================================================

print("\n[INFO] Metode 1: Deteksi kendaraan berbasis warna...")
print("-" * 50)

# Mendefinisikan range warna untuk kendaraan umum (HSV)
# Kendaraan biasanya berwarna gelap, putih, merah, atau biru

# Mendefinisikan range untuk warna gelap (hitam/abu-abu gelap)
lower_dark = np.array([0, 0, 0])
upper_dark = np.array([180, 255, 80])

# Membuat mask untuk warna gelap
mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

# Mendefinisikan range untuk warna terang (putih/silver)
lower_light = np.array([0, 0, 180])
upper_light = np.array([180, 40, 255])

# Membuat mask untuk warna terang
mask_light = cv2.inRange(hsv, lower_light, upper_light)

# Mendefinisikan range untuk warna merah (hue wrap-around)
lower_red1 = np.array([0, 70, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 70])
upper_red2 = np.array([180, 255, 255])

# Membuat mask untuk warna merah (gabungan dua range)
mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

# Mendefinisikan range untuk warna biru
lower_blue = np.array([100, 70, 70])
upper_blue = np.array([130, 255, 255])

# Membuat mask untuk warna biru
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# Menggabungkan semua mask warna
mask_combined = mask_dark | mask_light | mask_red | mask_blue

# Membuat kernel untuk operasi morfologi
kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Menerapkan opening untuk menghilangkan noise kecil
mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_clean, iterations=2)

# Menerapkan closing untuk mengisi lubang
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_clean, iterations=3)

# Menemukan kontur pada mask warna yang sudah dibersihkan
contours_color, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Menampilkan jumlah kontur warna yang ditemukan
print(f"  Total kontur warna: {len(contours_color)}")

# Mendefinisikan area minimum untuk kandidat kendaraan
min_vehicle_area = (img_h * img_w) * 0.005

# Mendefinisikan area maksimum untuk kandidat kendaraan
max_vehicle_area = (img_h * img_w) * 0.5

# Membuat salinan gambar untuk menandai deteksi warna
img_color_detect = img_vehicle.copy()

# Membuat list untuk menyimpan kendaraan terdeteksi melalui warna
vehicles_color = []

# Memfilter kontur berdasarkan kriteria kendaraan
for contour in contours_color:
    # Menghitung area kontur
    area = cv2.contourArea(contour)

    # Memeriksa apakah area memenuhi syarat minimum kendaraan
    if area < min_vehicle_area:
        continue

    # Memeriksa apakah area tidak melebihi batas maksimum
    if area > max_vehicle_area:
        continue

    # Mendapatkan bounding box kontur
    x, y, w, h = cv2.boundingRect(contour)

    # Menghitung aspect ratio bounding box
    aspect_ratio = w / float(h) if h > 0 else 0

    # Memeriksa apakah aspect ratio sesuai dengan kendaraan (biasanya lebih lebar dari tinggi)
    if 0.5 < aspect_ratio < 5.0:
        # Menambahkan kendaraan ke list
        vehicles_color.append((x, y, w, h, area))

        # Menggambar bounding box kuning pada kendaraan
        cv2.rectangle(img_color_detect, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Menampilkan jumlah kendaraan terdeteksi melalui warna
print(f"  Kendaraan terdeteksi (warna): {len(vehicles_color)}")

# Mengkonversi gambar hasil ke RGB
img_color_rgb = cv2.cvtColor(img_color_detect, cv2.COLOR_BGR2RGB)

# Membuat figure untuk visualisasi deteksi berbasis warna
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Menampilkan gambar asli
axes[0, 0].imshow(img_vehicle_rgb)
axes[0, 0].set_title("1. Gambar Asli", fontsize=11, fontweight="bold")
axes[0, 0].axis("off")

# Menampilkan mask warna gelap
axes[0, 1].imshow(mask_dark, cmap="gray")
axes[0, 1].set_title("2. Mask Warna Gelap", fontsize=11, fontweight="bold")
axes[0, 1].axis("off")

# Menampilkan mask warna terang
axes[0, 2].imshow(mask_light, cmap="gray")
axes[0, 2].set_title("3. Mask Warna Terang", fontsize=11, fontweight="bold")
axes[0, 2].axis("off")

# Menampilkan mask gabungan
axes[1, 0].imshow(mask_combined, cmap="gray")
axes[1, 0].set_title("4. Mask Gabungan", fontsize=11, fontweight="bold")
axes[1, 0].axis("off")

# Menampilkan mask setelah morfologi
axes[1, 1].imshow(mask_cleaned, cmap="gray")
axes[1, 1].set_title("5. Setelah Morfologi", fontsize=11, fontweight="bold")
axes[1, 1].axis("off")

# Menampilkan hasil deteksi dengan bounding box  
axes[1, 2].imshow(img_color_rgb)
axes[1, 2].set_title(f"6. Deteksi Warna ({len(vehicles_color)} kendaraan)", fontsize=11, fontweight="bold")
axes[1, 2].axis("off")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 10: Deteksi Kendaraan - Metode Warna",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi deteksi berbasis warna
output_path_1 = os.path.join(OUTPUT_DIR, "10_vehicle_color.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 3. Deteksi Kendaraan Berbasis Kontur
# ============================================================

print("\n[INFO] Metode 2: Deteksi kendaraan berbasis kontur...")
print("-" * 50)

# Menerapkan Gaussian blur untuk mengurangi noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Mendeteksi tepi menggunakan Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Membuat kernel untuk operasi morfologi pada tepi
kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))

# Menerapkan dilation pada tepi untuk menghubungkan komponen
dilated_edges = cv2.dilate(edges, kernel_edge, iterations=2)

# Menerapkan closing untuk mengisi gap
closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel_edge, iterations=2)

# Menemukan kontur pada hasil edge detection
contours_edge, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Menampilkan jumlah kontur edge yang ditemukan
print(f"  Total kontur edge: {len(contours_edge)}")

# Membuat salinan gambar untuk menandai deteksi kontur
img_contour_detect = img_vehicle.copy()

# Membuat list untuk menyimpan kendaraan terdeteksi melalui kontur
vehicles_contour = []

# Memfilter kontur berdasarkan kriteria kendaraan
for contour in contours_edge:
    # Menghitung area kontur
    area = cv2.contourArea(contour)

    # Memeriksa area minimum
    if area < min_vehicle_area:
        continue

    # Memeriksa area maksimum
    if area > max_vehicle_area:
        continue

    # Mendapatkan bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Menghitung aspect ratio
    aspect_ratio = w / float(h) if h > 0 else 0

    # Menghitung solidity (area kontur / area convex hull)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area) if hull_area > 0 else 0

    # Memeriksa kriteria kendaraan (aspect ratio dan solidity)
    if 0.5 < aspect_ratio < 5.0 and solidity > 0.3:
        # Menambahkan kendaraan ke list
        vehicles_contour.append((x, y, w, h, area, solidity))

        # Menggambar bounding box cyan pada kendaraan
        cv2.rectangle(img_contour_detect, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Menggambar kontur dengan warna hijau
        cv2.drawContours(img_contour_detect, [contour], -1, (0, 255, 0), 1)

# Menampilkan jumlah kendaraan terdeteksi melalui kontur
print(f"  Kendaraan terdeteksi (kontur): {len(vehicles_contour)}")

# Mengkonversi gambar hasil ke RGB
img_contour_rgb = cv2.cvtColor(img_contour_detect, cv2.COLOR_BGR2RGB)

# Membuat figure untuk visualisasi deteksi kontur
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Menampilkan gambar asli
axes[0, 0].imshow(img_vehicle_rgb)
axes[0, 0].set_title("1. Gambar Asli", fontsize=11, fontweight="bold")
axes[0, 0].axis("off")

# Menampilkan tepi Canny
axes[0, 1].imshow(edges, cmap="gray")
axes[0, 1].set_title("2. Canny Edge Detection", fontsize=11, fontweight="bold")
axes[0, 1].axis("off")

# Menampilkan tepi setelah morfologi
axes[1, 0].imshow(closed_edges, cmap="gray")
axes[1, 0].set_title("3. Edge + Dilation + Closing", fontsize=11, fontweight="bold")
axes[1, 0].axis("off")

# Menampilkan hasil deteksi kontur
axes[1, 1].imshow(img_contour_rgb)
axes[1, 1].set_title(f"4. Deteksi Kontur ({len(vehicles_contour)} kendaraan)", fontsize=11, fontweight="bold")
axes[1, 1].axis("off")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 10: Deteksi Kendaraan - Metode Kontur",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi deteksi kontur
output_path_2 = os.path.join(OUTPUT_DIR, "10_vehicle_contour.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 4. Klasifikasi Jenis Kendaraan Berdasarkan Ukuran
# ============================================================

print("\n[INFO] Mengklasifikasikan jenis kendaraan...")
print("-" * 50)

# Menggabungkan deteksi dari kedua metode
all_vehicles = []

# Menambahkan kendaraan dari metode warna
for (x, y, w, h, area) in vehicles_color:
    all_vehicles.append((x, y, w, h, area, "warna"))

# Menambahkan kendaraan dari metode kontur
for (x, y, w, h, area, sol) in vehicles_contour:
    all_vehicles.append((x, y, w, h, area, "kontur"))

# Menghapus duplikat yang saling overlap menggunakan Non-Maximum Suppression sederhana
def simple_nms(detections, overlap_thresh=0.3):
    """Menghapus bounding box yang saling overlap."""
    # Memeriksa apakah ada deteksi
    if len(detections) == 0:
        return []

    # Mengkonversi ke numpy array untuk komputasi
    boxes = np.array([(x, y, x + w, y + h) for (x, y, w, h, _, _) in detections])

    # Menghitung area setiap box
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Mengurutkan berdasarkan area (terbesar dulu)
    order = areas.argsort()[::-1]

    # Membuat list untuk menyimpan indeks yang dipertahankan
    keep = []

    # Melakukan iterasi NMS
    while len(order) > 0:
        # Mengambil box dengan area terbesar
        i = order[0]
        keep.append(i)

        # Menghitung overlap dengan box lainnya
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        # Menghitung area intersection
        w_inter = np.maximum(0, xx2 - xx1)
        h_inter = np.maximum(0, yy2 - yy1)
        intersection = w_inter * h_inter

        # Menghitung IoU (Intersection over Union)
        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-7)

        # Mempertahankan box dengan IoU rendah (tidak overlap)
        remaining = np.where(iou <= overlap_thresh)[0]
        order = order[remaining + 1]

    # Mengembalikan deteksi yang dipertahankan
    return [detections[i] for i in keep]

# Menerapkan NMS untuk menghapus duplikat
vehicles_nms = simple_nms(all_vehicles)

# Menampilkan jumlah kendaraan setelah NMS
print(f"  Total deteksi sebelum NMS: {len(all_vehicles)}")
print(f"  Total deteksi setelah NMS: {len(vehicles_nms)}")

# Menghitung percentile area untuk threshold klasifikasi
if len(vehicles_nms) > 0:
    # Mengumpulkan semua area kendaraan
    all_areas = [area for (_, _, _, _, area, _) in vehicles_nms]

    # Menghitung threshold klasifikasi berdasarkan ukuran
    area_sorted = sorted(all_areas)
    thresh_small = np.percentile(area_sorted, 40)
    thresh_large = np.percentile(area_sorted, 75)
else:
    # Mendefinisikan threshold default
    thresh_small = (img_h * img_w) * 0.02
    thresh_large = (img_h * img_w) * 0.08

# Mendefinisikan fungsi untuk mengklasifikasikan jenis kendaraan
def classify_vehicle(area, aspect_ratio):
    """Mengklasifikasikan kendaraan berdasarkan area dan aspect ratio."""
    # Menentukan jenis berdasarkan ukuran
    if area < thresh_small:
        return "Mobil", (0, 255, 0)       # Hijau untuk mobil
    elif area < thresh_large:
        if aspect_ratio > 2.0:
            return "Bus", (255, 165, 0)    # Orange untuk bus
        else:
            return "Mobil", (0, 255, 0)    # Hijau untuk mobil
    else:
        if aspect_ratio > 1.8:
            return "Bus", (255, 165, 0)    # Orange untuk bus
        else:
            return "Truk", (0, 0, 255)     # Merah untuk truk

# Membuat salinan gambar untuk menampilkan klasifikasi
img_classify = img_vehicle.copy()

# Mendefinisikan counter untuk setiap jenis kendaraan
count_mobil = 0
count_bus = 0
count_truk = 0

# Mengklasifikasikan dan menandai setiap kendaraan
for (x, y, w, h, area, method) in vehicles_nms:
    # Menghitung aspect ratio
    aspect_ratio = w / float(h) if h > 0 else 0

    # Mengklasifikasikan jenis kendaraan
    jenis, color_bgr = classify_vehicle(area, aspect_ratio)

    # Menghitung jumlah per jenis
    if jenis == "Mobil":
        count_mobil += 1
    elif jenis == "Bus":
        count_bus += 1
    else:
        count_truk += 1

    # Menggambar bounding box dengan warna sesuai jenis
    cv2.rectangle(img_classify, (x, y), (x + w, y + h), color_bgr, 2)

    # Menuliskan label jenis kendaraan
    label = f"{jenis}"
    cv2.putText(img_classify, label, (x, y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

# Menampilkan hasil klasifikasi
print(f"\n  Hasil klasifikasi kendaraan:")
print(f"    Mobil : {count_mobil}")
print(f"    Bus   : {count_bus}")
print(f"    Truk  : {count_truk}")
print(f"    Total : {count_mobil + count_bus + count_truk}")

# Mengkonversi gambar hasil klasifikasi ke RGB
img_classify_rgb = cv2.cvtColor(img_classify, cv2.COLOR_BGR2RGB)

# ============================================================
# 5. Visualisasi Klasifikasi Kendaraan
# ============================================================

print("\n[INFO] Menyimpan visualisasi klasifikasi kendaraan...")

# Membuat figure untuk hasil klasifikasi
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Menampilkan gambar asli
axes[0].imshow(img_vehicle_rgb)
axes[0].set_title("Gambar Asli", fontsize=11, fontweight="bold")
axes[0].axis("off")

# Menampilkan hasil klasifikasi
axes[1].imshow(img_classify_rgb)
axes[1].set_title(f"Klasifikasi Kendaraan ({len(vehicles_nms)} kendaraan)", fontsize=11, fontweight="bold")
axes[1].axis("off")

# Membuat bar chart jumlah kendaraan per jenis
jenis_labels = ["Mobil", "Bus", "Truk"]
jenis_counts = [count_mobil, count_bus, count_truk]
jenis_colors = ["#00ff00", "#ffa500", "#ff0000"]

# Menampilkan bar chart
bars = axes[2].bar(jenis_labels, jenis_counts, color=jenis_colors, edgecolor="black")
axes[2].set_title("Jumlah per Jenis Kendaraan", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Jumlah")
axes[2].set_xlabel("Jenis Kendaraan")

# Menambahkan label angka di atas setiap bar
for bar, count in zip(bars, jenis_counts):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)

# Menambahkan judul utama figure
plt.suptitle("Percobaan 10: Klasifikasi Kendaraan Berdasarkan Ukuran",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi klasifikasi
output_path_3 = os.path.join(OUTPUT_DIR, "10_vehicle_classification.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 10")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Deteksi Berbasis Warna:")
print("     - cv2.cvtColor(COLOR_BGR2HSV): konversi ke ruang warna HSV")
print("     - cv2.inRange(): segmentasi berdasarkan range HSV")
print("     - Warna kendaraan: gelap, terang, merah, biru")
print("  2. Deteksi Berbasis Kontur:")
print("     - cv2.Canny(): deteksi tepi")
print("     - cv2.findContours(): menemukan kontur objek")
print("     - cv2.contourArea(): menghitung luas kontur")
print("     - cv2.convexHull(): menghitung convex hull")
print("     - Solidity = area kontur / area hull")
print("  3. Klasifikasi Kendaraan:")
print("     - Berdasarkan area dan aspect ratio bounding box")
print("     - Mobil: area kecil, Bus: area besar + lebar, Truk: area besar")
print("  4. Non-Maximum Suppression (NMS):")
print("     - Menghapus deteksi duplikat yang overlap")
print("     - IoU (Intersection over Union) sebagai metrik overlap")
print(f"\nHasil deteksi:")
print(f"  - Metode warna  : {len(vehicles_color)} kendaraan")
print(f"  - Metode kontur  : {len(vehicles_contour)} kendaraan")
print(f"  - Setelah NMS    : {len(vehicles_nms)} kendaraan")
print(f"  - Mobil: {count_mobil}, Bus: {count_bus}, Truk: {count_truk}")
print("=" * 60)
