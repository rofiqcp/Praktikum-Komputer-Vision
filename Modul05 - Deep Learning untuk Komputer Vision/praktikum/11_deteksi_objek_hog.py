"""
==========================================================================
PERCOBAAN 11: HOG (HISTOGRAM OF ORIENTED GRADIENTS) UNTUK DETEKSI
==========================================================================
Program ini mempelajari deskriptor HOG (Histogram of Oriented Gradients)
secara mendalam, mulai dari implementasi manual perhitungan gradien dan
histogram orientasi, hingga penggunaan cv2.HOGDescriptor untuk deteksi
pejalan kaki. HOG adalah fitur yang sangat efektif untuk mendeteksi
bentuk dan struktur objek, terutama manusia.

Fungsi utama yang dipelajari:
- cv2.HOGDescriptor()              : Membuat deskriptor HOG
- cv2.Sobel()                      : Menghitung gradien gambar
- np.arctan2()                     : Menghitung orientasi gradien
- hog.detectMultiScale()           : Deteksi multi-skala dengan HOG
- hog.setSVMDetector()             : Mengatur SVM detector (people)
- cv2.HOGDescriptor_getDefaultPeopleDetector() : Detector pejalan kaki

Konsep yang dipelajari:
- Gradien gambar: magnitude dan orientasi
- Histogram orientasi dalam sel (cell)
- Blok normalisasi untuk invariansi terhadap pencahayaan
- HOG descriptor sebagai representasi shape/form
- Deteksi pejalan kaki menggunakan HOG + SVM
- Pengaruh parameter winStride, padding, scale
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan HOG
import cv2

# Mengimpor NumPy untuk komputasi numerik array
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi hasil
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
print("PERCOBAAN 11: HOG - HISTOGRAM OF ORIENTED GRADIENTS")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep HOG
# ============================================================
print("\n--- 1. Konsep HOG (Histogram of Oriented Gradients) ---")

# Menjelaskan konsep HOG
print("""
  HOG (Histogram of Oriented Gradients) adalah deskriptor fitur
  yang menangkap distribusi arah tepi/gradien dalam gambar.

  Langkah-langkah pembuatan HOG:
  1. Hitung gradien gambar (magnitude & orientasi)
  2. Bagi gambar menjadi sel-sel kecil (misal 8x8 piksel)
  3. Untuk setiap sel, buat histogram orientasi (9 bin, 0-180°)
  4. Kelompokkan sel menjadi blok (misal 2x2 sel)
  5. Normalisasi histogram dalam setiap blok
  6. Gabungkan semua histogram menjadi vektor fitur HOG

  Keunggulan HOG:
  - Robust terhadap perubahan pencahayaan (karena normalisasi)
  - Menangkap bentuk dan struktur objek dengan baik
  - Sangat efektif untuk deteksi pejalan kaki (HOG + SVM)
""")

# ============================================================
# 2. Memuat gambar untuk analisis HOG
# ============================================================
print("\n--- 2. Memuat Gambar ---")

# Memuat gambar pedestrian untuk deteksi
img_ped = cv2.imread(os.path.join(IMAGE_DIR, "pedestrian.jpg"))

# Memeriksa dan membuat gambar sintetis jika tidak ada
if img_ped is None:
    # Menampilkan pesan
    print("  [INFO] pedestrian.jpg tidak ditemukan, membuat gambar sintetis...")

    # Membuat gambar sintetis dengan siluet pejalan kaki
    img_ped = np.ones((400, 600, 3), dtype=np.uint8) * 180

    # Menggambar siluet orang 1 (kiri)
    cv2.rectangle(img_ped, (80, 80), (140, 120), (50, 50, 50), -1)   # kepala
    cv2.rectangle(img_ped, (70, 120), (150, 280), (60, 60, 60), -1)  # badan
    cv2.rectangle(img_ped, (80, 280), (110, 380), (55, 55, 55), -1)  # kaki kiri
    cv2.rectangle(img_ped, (120, 280), (150, 380), (55, 55, 55), -1) # kaki kanan

    # Menggambar siluet orang 2 (tengah)
    cv2.rectangle(img_ped, (270, 100), (330, 140), (45, 45, 45), -1)
    cv2.rectangle(img_ped, (260, 140), (340, 300), (55, 55, 55), -1)
    cv2.rectangle(img_ped, (270, 300), (300, 390), (50, 50, 50), -1)
    cv2.rectangle(img_ped, (310, 300), (340, 390), (50, 50, 50), -1)

    # Menggambar siluet orang 3 (kanan)
    cv2.rectangle(img_ped, (460, 90), (520, 130), (40, 40, 40), -1)
    cv2.rectangle(img_ped, (450, 130), (530, 290), (50, 50, 50), -1)
    cv2.rectangle(img_ped, (460, 290), (490, 385), (45, 45, 45), -1)
    cv2.rectangle(img_ped, (500, 290), (530, 385), (45, 45, 45), -1)

    # Menambahkan noise latar belakang
    noise = np.random.randint(0, 20, img_ped.shape, dtype=np.uint8)
    img_ped = cv2.add(img_ped, noise)

# Meresize gambar ke ukuran standar
img_ped = cv2.resize(img_ped, (600, 400))

# Mengkonversi ke grayscale
gray_ped = cv2.cvtColor(img_ped, cv2.COLOR_BGR2GRAY)

# Memuat gambar kedua untuk analisis HOG manual
img_analisis = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))
if img_analisis is None:
    # Menggunakan gambar pedestrian jika kucing tidak ada
    img_analisis = img_ped.copy()

# Meresize gambar analisis
img_analisis = cv2.resize(img_analisis, (256, 256))

# Mengkonversi ke grayscale
gray_analisis = cv2.cvtColor(img_analisis, cv2.COLOR_BGR2GRAY)

# Menampilkan informasi gambar
print(f"  Gambar pedestrian : {img_ped.shape}")
print(f"  Gambar analisis   : {img_analisis.shape}")

# ============================================================
# 3. Menghitung gradien gambar secara manual
# ============================================================
print("\n--- 3. Menghitung Gradien Gambar ---")

# Mengkonversi gambar analisis ke float untuk presisi
gray_float = gray_analisis.astype(np.float64)

# Menghitung gradien arah X menggunakan Sobel
grad_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=1)

# Menghitung gradien arah Y menggunakan Sobel
grad_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=1)

# Menghitung magnitude gradien (kekuatan tepi)
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Menghitung orientasi gradien (arah tepi) dalam derajat
orientasi = np.arctan2(grad_y, grad_x) * 180.0 / np.pi

# Mengkonversi orientasi ke range 0-180 (unsigned)
orientasi = orientasi % 180

# Menampilkan statistik gradien
print(f"  Gradient X   - Min: {grad_x.min():.1f}, Max: {grad_x.max():.1f}")
print(f"  Gradient Y   - Min: {grad_y.min():.1f}, Max: {grad_y.max():.1f}")
print(f"  Magnitude    - Min: {magnitude.min():.1f}, Max: {magnitude.max():.1f}")
print(f"  Orientasi    - Min: {orientasi.min():.1f}, Max: {orientasi.max():.1f}")

# ============================================================
# 4. Visualisasi gradien
# ============================================================
print("\n--- 4. Visualisasi Gradien ---")

# Membuat figure untuk visualisasi gradien
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- Subplot 1: Gambar asli ---
axes[0, 0].imshow(cv2.cvtColor(img_analisis, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# --- Subplot 2: Grayscale ---
axes[0, 1].imshow(gray_analisis, cmap='gray')
axes[0, 1].set_title("Grayscale", fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# --- Subplot 3: Gradien X ---
# Menormalisasi gradien X untuk visualisasi
grad_x_vis = np.abs(grad_x)
grad_x_vis = (grad_x_vis / grad_x_vis.max() * 255).astype(np.uint8)
axes[0, 2].imshow(grad_x_vis, cmap='hot')
axes[0, 2].set_title("Gradien X (Horizontal)", fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# --- Subplot 4: Gradien Y ---
# Menormalisasi gradien Y untuk visualisasi
grad_y_vis = np.abs(grad_y)
grad_y_vis = (grad_y_vis / grad_y_vis.max() * 255).astype(np.uint8)
axes[1, 0].imshow(grad_y_vis, cmap='hot')
axes[1, 0].set_title("Gradien Y (Vertikal)", fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# --- Subplot 5: Magnitude ---
# Menormalisasi magnitude untuk visualisasi
mag_vis = (magnitude / magnitude.max() * 255).astype(np.uint8)
axes[1, 1].imshow(mag_vis, cmap='hot')
axes[1, 1].set_title("Magnitude Gradien", fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# --- Subplot 6: Orientasi ---
# Memvisualisasikan orientasi dengan colormap HSV
axes[1, 2].imshow(orientasi, cmap='hsv')
axes[1, 2].set_title("Orientasi Gradien (0°-180°)", fontsize=11, fontweight='bold')
axes[1, 2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 11: Komponen Gradien untuk HOG",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi gradien
plt.savefig(os.path.join(OUTPUT_DIR, "11_hog_gradien.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/11_hog_gradien.png")

# ============================================================
# 5. Membangun histogram orientasi per sel (cell)
# ============================================================
print("\n--- 5. Histogram Orientasi per Sel ---")

# Mendefinisikan ukuran sel (cell size)
ukuran_sel = 8  # 8x8 piksel per sel

# Mendefinisikan jumlah bin orientasi
jumlah_bin = 9  # 9 bin (setiap bin = 20 derajat, total 180 derajat)

# Menghitung jumlah sel horizontal dan vertikal
h_img, w_img = gray_analisis.shape
jumlah_sel_y = h_img // ukuran_sel
jumlah_sel_x = w_img // ukuran_sel

# Menampilkan informasi sel
print(f"  Ukuran sel     : {ukuran_sel}x{ukuran_sel} piksel")
print(f"  Jumlah bin     : {jumlah_bin}")
print(f"  Grid sel       : {jumlah_sel_x} x {jumlah_sel_y}")
print(f"  Total sel      : {jumlah_sel_x * jumlah_sel_y}")

# Menyiapkan array untuk menyimpan histogram setiap sel
histogram_sel = np.zeros((jumlah_sel_y, jumlah_sel_x, jumlah_bin))

# Menghitung histogram orientasi untuk setiap sel
for sy in range(jumlah_sel_y):
    for sx in range(jumlah_sel_x):
        # Menentukan batas piksel sel ini
        y_start = sy * ukuran_sel
        y_end = y_start + ukuran_sel
        x_start = sx * ukuran_sel
        x_end = x_start + ukuran_sel

        # Mengambil magnitude dan orientasi untuk sel ini
        mag_sel = magnitude[y_start:y_end, x_start:x_end]
        ori_sel = orientasi[y_start:y_end, x_start:x_end]

        # Menghitung histogram orientasi dengan bobot magnitude
        # Setiap bin mencakup 20 derajat (180/9 = 20)
        for j in range(ukuran_sel):
            for i in range(ukuran_sel):
                # Menentukan bin untuk orientasi piksel ini
                bin_idx = int(ori_sel[j, i] / 20.0)

                # Membatasi bin_idx agar tidak keluar range
                bin_idx = min(bin_idx, jumlah_bin - 1)

                # Menambahkan magnitude sebagai bobot ke histogram
                histogram_sel[sy, sx, bin_idx] += mag_sel[j, i]

# Menampilkan contoh histogram sel tengah
sel_tengah_y = jumlah_sel_y // 2
sel_tengah_x = jumlah_sel_x // 2
print(f"\n  Contoh histogram sel ({sel_tengah_x}, {sel_tengah_y}):")
for b in range(jumlah_bin):
    sudut_awal = b * 20
    sudut_akhir = sudut_awal + 20
    nilai = histogram_sel[sel_tengah_y, sel_tengah_x, b]
    print(f"    Bin {b} ({sudut_awal:3d}°-{sudut_akhir:3d}°): {nilai:.1f}")

# ============================================================
# 6. Visualisasi HOG descriptor
# ============================================================
print("\n--- 6. Visualisasi HOG Descriptor ---")

# Membuat gambar visualisasi HOG dari histogram sel
# Setiap sel akan ditampilkan sebagai bintang/garis orientasi
hog_visual = np.zeros((h_img, w_img), dtype=np.float64)

# Menggambar representasi visual HOG per sel
for sy in range(jumlah_sel_y):
    for sx in range(jumlah_sel_x):
        # Menghitung pusat sel
        cx = sx * ukuran_sel + ukuran_sel // 2
        cy = sy * ukuran_sel + ukuran_sel // 2

        # Mendapatkan histogram sel ini
        hist = histogram_sel[sy, sx]

        # Menormalisasi histogram
        hist_norm = hist / (hist.max() + 1e-7)

        # Menggambar garis untuk setiap bin orientasi
        for b in range(jumlah_bin):
            # Menghitung sudut tengah bin dalam radian
            sudut = (b * 20 + 10) * np.pi / 180.0

            # Menentukan panjang garis berdasarkan nilai histogram
            panjang = int(hist_norm[b] * ukuran_sel * 0.45)

            # Menghitung titik akhir garis
            dx = int(panjang * np.cos(sudut))
            dy = int(panjang * np.sin(sudut))

            # Menggambar garis orientasi
            if panjang > 0:
                cv2.line(hog_visual, (cx - dx, cy - dy), (cx + dx, cy + dy),
                         hist_norm[b], 1)

# Membuat figure untuk visualisasi HOG
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# --- Subplot 1: Gambar asli ---
axes[0].imshow(cv2.cvtColor(img_analisis, cv2.COLOR_BGR2RGB))
axes[0].set_title("Gambar Asli", fontsize=12, fontweight='bold')
axes[0].axis('off')

# --- Subplot 2: HOG descriptor visual ---
axes[1].imshow(hog_visual, cmap='hot')
axes[1].set_title("Visualisasi HOG Descriptor", fontsize=12, fontweight='bold')
axes[1].axis('off')

# --- Subplot 3: Overlay HOG pada gambar ---
# Menormalisasi HOG visual ke 0-255
hog_vis_norm = (hog_visual / (hog_visual.max() + 1e-7) * 255).astype(np.uint8)

# Membuat gambar overlay
overlay = gray_analisis.copy()

# Menambahkan HOG visual di atas gambar grayscale
mask_hog = hog_vis_norm > 10
overlay[mask_hog] = np.clip(
    gray_analisis[mask_hog].astype(int) + hog_vis_norm[mask_hog].astype(int),
    0, 255
).astype(np.uint8)

# Menampilkan overlay
axes[2].imshow(overlay, cmap='gray')
axes[2].set_title("Overlay HOG pada Gambar", fontsize=12, fontweight='bold')
axes[2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 11: Visualisasi HOG Descriptor Manual",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi HOG descriptor
plt.savefig(os.path.join(OUTPUT_DIR, "11_hog_deskriptor.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/11_hog_deskriptor.png")

# ============================================================
# 7. Deteksi pejalan kaki menggunakan cv2.HOGDescriptor
# ============================================================
print("\n--- 7. Deteksi Pejalan Kaki dengan HOG + SVM ---")

# Membuat HOG descriptor untuk deteksi pejalan kaki
hog = cv2.HOGDescriptor()

# Mengatur SVM detector dengan default people detector
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Menampilkan informasi HOG descriptor
print(f"  HOG winSize     : {hog.winSize}")
print(f"  HOG blockSize   : {hog.blockSize}")
print(f"  HOG blockStride : {hog.blockStride}")
print(f"  HOG cellSize    : {hog.cellSize}")
print(f"  HOG nbins       : {hog.nbins}")

# ============================================================
# 8. Deteksi dengan berbagai parameter
# ============================================================
print("\n--- 8. Deteksi dengan Berbagai Parameter ---")

# Mendefinisikan variasi parameter untuk perbandingan
parameter_list = [
    {"winStride": (8, 8), "padding": (8, 8), "scale": 1.05, "nama": "Fine"},
    {"winStride": (4, 4), "padding": (16, 16), "scale": 1.02, "nama": "Very Fine"},
    {"winStride": (16, 16), "padding": (4, 4), "scale": 1.10, "nama": "Coarse"},
]

# Menyiapkan dictionary untuk menyimpan hasil deteksi
hasil_deteksi = {}

# Melakukan deteksi untuk setiap set parameter
for param in parameter_list:
    # Menampilkan parameter yang digunakan
    print(f"\n  Parameter '{param['nama']}':")
    print(f"    winStride: {param['winStride']}")
    print(f"    padding  : {param['padding']}")
    print(f"    scale    : {param['scale']}")

    # Melakukan deteksi multi-skala
    boxes, weights = hog.detectMultiScale(
        gray_ped,
        winStride=param['winStride'],
        padding=param['padding'],
        scale=param['scale']
    )

    # Menampilkan jumlah deteksi
    print(f"    Deteksi  : {len(boxes)} objek")

    # Menyimpan hasil
    hasil_deteksi[param['nama']] = (boxes, weights)

# ============================================================
# 9. Visualisasi hasil deteksi pejalan kaki
# ============================================================
print("\n--- 9. Visualisasi Deteksi Pejalan Kaki ---")

# Membuat figure untuk perbandingan deteksi
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Mendefinisikan warna untuk setiap parameter set
warna_param = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

# Memvisualisasikan hasil untuk setiap parameter
for idx, param in enumerate(parameter_list):
    # Mendapatkan hasil deteksi
    boxes, weights = hasil_deteksi[param['nama']]

    # Membuat salinan gambar
    img_det = img_ped.copy()

    # Menggambar setiap bounding box
    for i, (x, y, w, h) in enumerate(boxes):
        # Menggambar rectangle
        cv2.rectangle(img_det, (x, y), (x + w, y + h), warna_param[idx], 2)

        # Menuliskan skor confidence jika tersedia
        if len(weights) > i:
            skor = weights[i]
            cv2.putText(img_det, f"{skor:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna_param[idx], 2)

    # Mengkonversi dan menampilkan
    axes[idx].imshow(cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(f"{param['nama']}\nwinStride={param['winStride']} "
                        f"scale={param['scale']}\n{len(boxes)} deteksi",
                        fontsize=10, fontweight='bold')
    axes[idx].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 11: Deteksi Pejalan Kaki HOG+SVM - Perbandingan Parameter",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi deteksi
plt.savefig(os.path.join(OUTPUT_DIR, "11_hog_deteksi.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/11_hog_deteksi.png")

# ============================================================
# 10. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 11")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. HOG menangkap distribusi orientasi gradien dalam gambar
2. Gradien dihitung dengan cv2.Sobel() (arah X dan Y)
3. Magnitude = sqrt(gx² + gy²), Orientasi = arctan2(gy, gx)
4. Histogram orientasi dibangun per sel (8x8 piksel)
5. 9 bin orientasi mencakup 0°-180° (setiap bin = 20°)
6. Normalisasi blok membuat HOG robust terhadap pencahayaan
7. cv2.HOGDescriptor + SVM default untuk deteksi pejalan kaki
8. Parameter winStride, padding, scale mempengaruhi kecepatan & akurasi

Output disimpan di folder: output/
- 11_hog_gradien.png    : Visualisasi komponen gradien
- 11_hog_deskriptor.png : Visualisasi HOG descriptor
- 11_hog_deteksi.png    : Deteksi pejalan kaki dengan berbagai parameter
""")
