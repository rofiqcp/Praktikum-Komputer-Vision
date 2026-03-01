"""
==========================================================================
PERCOBAAN 9: PEDESTRIAN DETECTION DENGAN HOG+SVM
==========================================================================
Program ini mempelajari deteksi pejalan kaki menggunakan deskriptor HOG
(Histogram of Oriented Gradients) yang dikombinasikan dengan SVM (Support
Vector Machine). HOG merupakan fitur berbasis gradient yang sangat efektif
untuk mendeteksi bentuk objek, terutama manusia.

Konsep yang dipelajari:
- Histogram of Oriented Gradients (HOG): fitur berbasis gradient
- Menghitung gradient gambar secara manual (Sobel)
- Magnitude dan arah gradient
- HOG descriptor cells dan blocks
- Default people detector (pre-trained SVM)
- Pengaruh parameter winStride, padding, dan scale

Fungsi utama yang dipelajari:
- cv2.Sobel()                     : Menghitung gradient gambar
- cv2.cartToPolar()               : Mengkonversi gradient ke magnitude+arah
- cv2.HOGDescriptor()             : Membuat deskriptor HOG
- hog.setSVMDetector()            : Mengatur SVM detector (people)
- hog.detectMultiScale()          : Mendeteksi objek multi-skala
- cv2.rectangle()                 : Menggambar bounding box

Hasil: Visualisasi gradient HOG, deteksi pejalan kaki, dan perbandingan parameter
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan deteksi HOG
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu pemrosesan per parameter
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
print("PERCOBAAN 9: PEDESTRIAN DETECTION DENGAN HOG+SVM")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar dan Memahami Konsep Gradient
# ============================================================

print("\n[INFO] Memuat gambar pedestrian...")
print("-" * 50)

# Membaca gambar pejalan kaki dari file
img_ped = cv2.imread(os.path.join(IMAGE_DIR, "pedestrian.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_ped is None:
    print("[ERROR] Gambar pedestrian.jpg tidak ditemukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi gambar
print(f"  Ukuran gambar: {img_ped.shape[1]}x{img_ped.shape[0]} piksel")

# Mengkonversi gambar ke grayscale untuk komputasi gradient
gray = cv2.cvtColor(img_ped, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar dari BGR ke RGB untuk tampilan matplotlib
img_ped_rgb = cv2.cvtColor(img_ped, cv2.COLOR_BGR2RGB)

# ============================================================
# 2. Menghitung Gradient Gambar Secara Manual
# ============================================================

print("\n[INFO] Menghitung gradient gambar secara manual...")
print("-" * 50)

# Menghitung gradient horizontal menggunakan Sobel (turunan terhadap x)
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)

# Menghitung gradient vertikal menggunakan Sobel (turunan terhadap y)
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)

# Menghitung magnitude gradient: sqrt(gx^2 + gy^2)
magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

# Menormalisasi magnitude ke range 0-255 untuk visualisasi
magnitude_norm = np.uint8(255 * magnitude / (magnitude.max() + 1e-7))

# Menormalisasi arah ke range 0-255 untuk visualisasi (0-360 derajat)
angle_norm = np.uint8(255 * angle / 360.0)

# Menampilkan statistik gradient
print(f"  Magnitude - Min: {magnitude.min():.2f}, Max: {magnitude.max():.2f}")
print(f"  Arah      - Min: {angle.min():.2f}, Max: {angle.max():.2f} derajat")

# ============================================================
# 3. Visualisasi Gradient HOG
# ============================================================

print("\n[INFO] Membuat visualisasi gradient HOG...")

# Membuat visualisasi sederhana HOG dengan menggambar arah gradient pada grid

# Mendefinisikan ukuran cell untuk visualisasi HOG
cell_size = 16

# Menghitung jumlah cell per dimensi
n_cells_y = gray.shape[0] // cell_size
n_cells_x = gray.shape[1] // cell_size

# Membuat canvas untuk visualisasi HOG
hog_vis = np.zeros_like(gray, dtype=np.uint8)

# Menghitung histogram gradient untuk setiap cell
for cy in range(n_cells_y):
    for cx in range(n_cells_x):
        # Menghitung koordinat cell
        y1 = cy * cell_size
        y2 = y1 + cell_size
        x1 = cx * cell_size
        x2 = x1 + cell_size

        # Mengambil magnitude dan angle pada cell ini
        cell_mag = magnitude[y1:y2, x1:x2]
        cell_ang = angle[y1:y2, x1:x2]

        # Menghitung rata-rata magnitude pada cell
        avg_mag = np.mean(cell_mag)

        # Menghitung arah dominan (weighted by magnitude)
        if avg_mag > 0:
            # Menghitung arah rata-rata tertimbang magnitude
            avg_angle_rad = np.average(cell_ang, weights=cell_mag + 1e-7)
            avg_angle_rad = np.deg2rad(avg_angle_rad)

            # Menghitung titik tengah cell
            center_x = x1 + cell_size // 2
            center_y = y1 + cell_size // 2

            # Menghitung panjang garis berdasarkan magnitude (normalisasi)
            line_len = int(min(cell_size // 2, avg_mag / (magnitude.max() + 1e-7) * cell_size))

            # Menghitung titik akhir garis berdasarkan arah
            end_x = int(center_x + line_len * np.cos(avg_angle_rad))
            end_y = int(center_y + line_len * np.sin(avg_angle_rad))

            # Menggambar garis gradient pada canvas
            cv2.line(hog_vis, (center_x, center_y), (end_x, end_y), 255, 1)

# Membuat figure untuk visualisasi gradient
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Menampilkan gambar asli
axes[0, 0].imshow(img_ped_rgb)
axes[0, 0].set_title("1. Gambar Asli", fontsize=11, fontweight="bold")
axes[0, 0].axis("off")

# Menampilkan magnitude gradient
axes[0, 1].imshow(magnitude_norm, cmap="hot")
axes[0, 1].set_title("2. Gradient Magnitude", fontsize=11, fontweight="bold")
axes[0, 1].axis("off")

# Menampilkan arah gradient
axes[1, 0].imshow(angle_norm, cmap="hsv")
axes[1, 0].set_title("3. Gradient Direction (0-360°)", fontsize=11, fontweight="bold")
axes[1, 0].axis("off")

# Menampilkan visualisasi HOG sederhana
axes[1, 1].imshow(hog_vis, cmap="gray")
axes[1, 1].set_title(f"4. Visualisasi HOG (cell={cell_size}px)", fontsize=11, fontweight="bold")
axes[1, 1].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 9: Gradient dan Visualisasi HOG",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi gradient HOG
output_path_1 = os.path.join(OUTPUT_DIR, "09_hog_gradien.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 4. Deteksi Pejalan Kaki dengan HOG+SVM
# ============================================================

print("\n[INFO] Mendeteksi pejalan kaki dengan HOG+SVM...")
print("-" * 50)

# Membuat objek HOG descriptor
hog = cv2.HOGDescriptor()

# Mengatur SVM detector dengan default people detector (pre-trained)
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Menampilkan informasi HOG descriptor
print(f"  Window size: {hog.winSize}")
print(f"  Block size: {hog.blockSize}")
print(f"  Block stride: {hog.blockStride}")
print(f"  Cell size: {hog.cellSize}")
print(f"  Nbins: {hog.nbins}")

# Mencatat waktu mulai deteksi default
start_default = time.time()

# Mendeteksi pejalan kaki dengan parameter default
boxes_default, weights_default = hog.detectMultiScale(
    img_ped,
    winStride=(8, 8),
    padding=(8, 8),
    scale=1.05
)

# Menghitung waktu deteksi default
time_default = time.time() - start_default

# Menampilkan hasil deteksi default
print(f"\n  Deteksi default (winStride=8,8):")
print(f"    Jumlah deteksi: {len(boxes_default)}")
print(f"    Waktu proses: {time_default*1000:.2f} ms")

# Membuat salinan gambar untuk menggambar bounding box
img_detect = img_ped.copy()

# Menggambar bounding box untuk setiap deteksi
for i, (x, y, w, h) in enumerate(boxes_default):
    # Mendapatkan confidence weight untuk deteksi ini
    weight = weights_default[i] if i < len(weights_default) else 0

    # Menentukan warna berdasarkan confidence (hijau ke merah)
    color_intensity = min(255, int(weight * 100))
    color = (0, 255, 0)

    # Menggambar bounding box pada gambar
    cv2.rectangle(img_detect, (x, y), (x + w, y + h), color, 2)

    # Menuliskan confidence score di atas bounding box
    label = f"{weight:.2f}"
    cv2.putText(img_detect, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Mengkonversi gambar hasil ke RGB
img_detect_rgb = cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB)

# Membuat figure untuk menampilkan hasil deteksi
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Menampilkan gambar asli
axes[0].imshow(img_ped_rgb)
axes[0].set_title("Gambar Asli", fontsize=11, fontweight="bold")
axes[0].axis("off")

# Menampilkan hasil deteksi
axes[1].imshow(img_detect_rgb)
axes[1].set_title(f"Deteksi HOG+SVM ({len(boxes_default)} pedestrian)", fontsize=11, fontweight="bold")
axes[1].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 9: Deteksi Pejalan Kaki dengan HOG+SVM",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi deteksi
output_path_2 = os.path.join(OUTPUT_DIR, "09_hog_deteksi.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 5. Perbandingan Parameter HOG
# ============================================================

print("\n[INFO] Membandingkan parameter HOG...")
print("-" * 50)

# Mendefinisikan variasi parameter winStride yang akan diuji
win_strides = [(4, 4), (8, 8), (16, 16)]

# Mendefinisikan variasi padding
paddings = [(4, 4), (8, 8), (16, 16)]

# Mendefinisikan variasi scale
scales = [1.02, 1.05, 1.1]

# Membuat dictionary untuk menyimpan hasil tiap konfigurasi
results = {}

# --- Percobaan variasi winStride ---
print("\n  A. Variasi winStride (padding=8,8, scale=1.05):")

# Menguji setiap nilai winStride
for stride in win_strides:
    # Mencatat waktu mulai
    t_start = time.time()

    # Mendeteksi dengan winStride berbeda
    boxes, weights = hog.detectMultiScale(
        img_ped,
        winStride=stride,
        padding=(8, 8),
        scale=1.05
    )

    # Menghitung waktu proses
    t_elapsed = time.time() - t_start

    # Menyimpan hasil ke dictionary
    key = f"stride_{stride[0]}"
    results[key] = {
        "boxes": boxes, "weights": weights,
        "time": t_elapsed, "label": f"winStride={stride}"
    }

    # Menampilkan hasil
    print(f"    winStride={stride}: {len(boxes)} deteksi, {t_elapsed*1000:.2f} ms")

# --- Percobaan variasi padding ---
print("\n  B. Variasi padding (winStride=8,8, scale=1.05):")

# Menguji setiap nilai padding
for pad in paddings:
    # Mencatat waktu mulai
    t_start = time.time()

    # Mendeteksi dengan padding berbeda
    boxes, weights = hog.detectMultiScale(
        img_ped,
        winStride=(8, 8),
        padding=pad,
        scale=1.05
    )

    # Menghitung waktu proses
    t_elapsed = time.time() - t_start

    # Menyimpan hasil ke dictionary
    key = f"pad_{pad[0]}"
    results[key] = {
        "boxes": boxes, "weights": weights,
        "time": t_elapsed, "label": f"padding={pad}"
    }

    # Menampilkan hasil
    print(f"    padding={pad}: {len(boxes)} deteksi, {t_elapsed*1000:.2f} ms")

# --- Percobaan variasi scale ---
print("\n  C. Variasi scale (winStride=8,8, padding=8,8):")

# Menguji setiap nilai scale
for sc in scales:
    # Mencatat waktu mulai
    t_start = time.time()

    # Mendeteksi dengan scale berbeda
    boxes, weights = hog.detectMultiScale(
        img_ped,
        winStride=(8, 8),
        padding=(8, 8),
        scale=sc
    )

    # Menghitung waktu proses
    t_elapsed = time.time() - t_start

    # Menyimpan hasil ke dictionary
    key = f"scale_{sc}"
    results[key] = {
        "boxes": boxes, "weights": weights,
        "time": t_elapsed, "label": f"scale={sc}"
    }

    # Menampilkan hasil
    print(f"    scale={sc}: {len(boxes)} deteksi, {t_elapsed*1000:.2f} ms")

# ============================================================
# 6. Visualisasi Perbandingan Parameter
# ============================================================

print("\n[INFO] Menyimpan visualisasi perbandingan parameter...")

# Membuat figure 3x3 untuk menampilkan semua variasi parameter
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Mendefinisikan key untuk setiap baris subplot
row_keys = [
    [f"stride_{s[0]}" for s in win_strides],
    [f"pad_{p[0]}" for p in paddings],
    [f"scale_{s}" for s in scales]
]

# Mendefinisikan judul baris
row_titles = ["Variasi winStride", "Variasi Padding", "Variasi Scale"]

# Menampilkan hasil untuk setiap konfigurasi
for row, (keys, row_title) in enumerate(zip(row_keys, row_titles)):
    for col, key in enumerate(keys):
        # Mengambil hasil dari dictionary
        res = results[key]

        # Membuat salinan gambar untuk menggambar deteksi
        img_vis = img_ped.copy()

        # Menggambar bounding box untuk setiap deteksi
        for i, (x, y, w, h) in enumerate(res["boxes"]):
            # Mendapatkan weight untuk deteksi ini
            wt = res["weights"][i] if i < len(res["weights"]) else 0

            # Menggambar bounding box hijau
            cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Menuliskan weight
            cv2.putText(img_vis, f"{wt:.1f}", (x, y - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Mengkonversi ke RGB untuk matplotlib
        img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

        # Menampilkan gambar pada subplot
        axes[row, col].imshow(img_vis_rgb)

        # Membuat judul dengan info parameter
        title = f"{res['label']}\n{len(res['boxes'])} deteksi, {res['time']*1000:.0f}ms"
        axes[row, col].set_title(title, fontsize=9, fontweight="bold")
        axes[row, col].axis("off")

    # Menambahkan label baris di sisi kiri
    axes[row, 0].set_ylabel(row_title, fontsize=11, fontweight="bold", rotation=0,
                            labelpad=80, va="center")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 9: Perbandingan Parameter HOG detectMultiScale",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi perbandingan parameter
output_path_3 = os.path.join(OUTPUT_DIR, "09_hog_parameter_compare.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 9")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Konsep HOG (Histogram of Oriented Gradients):")
print("     - cv2.Sobel(): menghitung gradient Gx dan Gy")
print("     - cv2.cartToPolar(): mengkonversi gradient ke magnitude+arah")
print("     - HOG membagi gambar ke cells → blocks → histogram")
print("     - Fitur HOG robust terhadap perubahan iluminasi")
print("  2. cv2.HOGDescriptor():")
print("     - Window size: 64x128 (ukuran sliding window)")
print("     - Block size: 16x16, Block stride: 8x8")
print("     - Cell size: 8x8, Nbins: 9 (histogram 9 bin orientasi)")
print("  3. hog.detectMultiScale() - Parameter:")
print("     - winStride: langkah sliding window")
print("       (4,4)=teliti/lambat, (16,16)=cepat/kurang akurat")
print("     - padding: padding di sekitar window")
print("     - scale: faktor skala piramida gambar")
print("       Kecil(1.02)=banyak skala/lambat, Besar(1.1)=sedikit skala/cepat")
print("  4. Default People Detector: pre-trained linear SVM")
print(f"\nHasil deteksi (default parameter):")
print(f"  - Jumlah pedestrian: {len(boxes_default)}")
print(f"  - Waktu proses: {time_default*1000:.2f} ms")
if len(weights_default) > 0:
    print(f"  - Confidence min: {min(weights_default):.3f}")
    print(f"  - Confidence max: {max(weights_default):.3f}")
print("=" * 60)
