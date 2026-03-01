"""
==========================================================================
PERCOBAAN 10: DETEKSI OBJEK - SLIDING WINDOW
==========================================================================
Program ini mempelajari konsep sliding window untuk deteksi objek,
yaitu teknik menggeser jendela (window) di seluruh gambar untuk
mendeteksi objek pada berbagai posisi. Implementasi mencakup template
matching sebagai "detektor", Non-Maximum Suppression (NMS), dan
perhitungan Intersection over Union (IoU).

Fungsi utama yang dipelajari:
- cv2.matchTemplate()      : Template matching untuk deteksi objek
- cv2.rectangle()          : Menggambar bounding box
- cv2.minMaxLoc()          : Mencari lokasi nilai min/max
- Manual sliding window    : Implementasi window geser dari nol
- Manual NMS               : Non-Maximum Suppression dari nol
- Manual IoU               : Intersection over Union dari nol

Konsep yang dipelajari:
- Sliding window: jendela geser untuk memindai seluruh gambar
- Template matching: mencocokkan template dengan bagian gambar
- Bounding box: kotak pembatas objek terdeteksi
- Non-Maximum Suppression (NMS): menghilangkan deteksi duplikat
- Intersection over Union (IoU): metrik tumpang tindih bounding box
- Multi-scale detection: deteksi pada berbagai ukuran
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan komputasi numerik
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi hasil deteksi
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
print("PERCOBAAN 10: DETEKSI OBJEK - SLIDING WINDOW")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep sliding window
# ============================================================
print("\n--- 1. Konsep Sliding Window ---")

# Menjelaskan konsep sliding window
print("""
  Sliding Window adalah teknik deteksi objek klasik:

  1. Tentukan ukuran window (misal 64x64 piksel)
  2. Geser window dari kiri-atas ke kanan-bawah gambar
  3. Di setiap posisi, ekstrak fitur dari isi window
  4. Klasifikasikan apakah window berisi objek target
  5. Jika ya, simpan posisi sebagai bounding box
  6. Ulangi untuk ukuran window berbeda (multi-scale)
  7. Terapkan NMS untuk menghilangkan deteksi duplikat

  Dalam percobaan ini, kita gunakan template matching
  sebagai "classifier" sederhana di setiap posisi window.
""")

# ============================================================
# 2. Memuat gambar untuk deteksi
# ============================================================
print("\n--- 2. Memuat Gambar ---")

# Memuat gambar scene outdoor sebagai gambar utama
img = cv2.imread(os.path.join(IMAGE_DIR, "scene_outdoor.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img is None:
    # Mencoba memuat gambar alternatif
    img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
    if img is None:
        # Membuat gambar sintetis dengan beberapa objek jika tidak ada gambar
        print("  [INFO] Membuat gambar sintetis dengan objek-objek...")
        img = np.ones((400, 600, 3), dtype=np.uint8) * 200

        # Menggambar beberapa persegi sebagai "objek"
        cv2.rectangle(img, (50, 50), (130, 130), (0, 0, 200), -1)
        cv2.rectangle(img, (200, 150), (280, 230), (0, 0, 180), -1)
        cv2.rectangle(img, (400, 80), (480, 160), (0, 0, 220), -1)
        cv2.rectangle(img, (300, 280), (380, 360), (0, 0, 190), -1)
        cv2.rectangle(img, (100, 300), (180, 380), (0, 0, 210), -1)

        # Menambahkan noise latar belakang untuk realisme
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

# Meresize gambar agar ukurannya konsisten
img = cv2.resize(img, (600, 400))

# Mengkonversi gambar ke grayscale untuk template matching
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Menampilkan informasi gambar
print(f"  Ukuran gambar  : {img.shape}")
print(f"  Ukuran grayscale: {gray.shape}")

# ============================================================
# 3. Implementasi sliding window
# ============================================================
print("\n--- 3. Implementasi Sliding Window ---")

def sliding_window(gambar, ukuran_window, langkah):
    """
    Generator sliding window yang menghasilkan posisi dan isi window.

    Parameter:
    - gambar: gambar input (2D atau 3D)
    - ukuran_window: tuple (lebar, tinggi) window
    - langkah: jumlah piksel pergeseran per langkah
    """
    # Mendapatkan dimensi gambar
    h_gambar, w_gambar = gambar.shape[:2]

    # Mendapatkan ukuran window
    w_win, h_win = ukuran_window

    # Iterasi vertikal (atas ke bawah)
    for y in range(0, h_gambar - h_win + 1, langkah):
        # Iterasi horizontal (kiri ke kanan)
        for x in range(0, w_gambar - w_win + 1, langkah):
            # Mengekstrak isi window dari gambar
            window = gambar[y:y + h_win, x:x + w_win]

            # Menghasilkan posisi (x, y) dan isi window
            yield (x, y, window)


# Mendefinisikan parameter sliding window
ukuran_window = (80, 80)  # Ukuran window 80x80 piksel
langkah = 40              # Langkah pergeseran 40 piksel

# Menghitung jumlah total posisi window
h_img, w_img = gray.shape
jumlah_posisi = ((h_img - ukuran_window[1]) // langkah + 1) * \
                ((w_img - ukuran_window[0]) // langkah + 1)

# Menampilkan informasi sliding window
print(f"  Ukuran window : {ukuran_window}")
print(f"  Langkah       : {langkah} piksel")
print(f"  Total posisi  : {jumlah_posisi}")

# Mengumpulkan beberapa posisi window untuk visualisasi
posisi_window = []
counter = 0
for (x, y, window) in sliding_window(gray, ukuran_window, langkah):
    # Menyimpan setiap posisi ke-5 untuk visualisasi
    if counter % 5 == 0:
        posisi_window.append((x, y))
    counter += 1

# Menampilkan jumlah posisi yang dikumpulkan
print(f"  Posisi untuk visualisasi: {len(posisi_window)}")

# ============================================================
# 4. Visualisasi sliding window
# ============================================================
print("\n--- 4. Visualisasi Sliding Window ---")

# Membuat figure untuk visualisasi sliding window
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Meratakan axes
axes_flat = axes.flatten()

# Memilih 6 posisi window yang merata untuk ditampilkan
if len(posisi_window) >= 6:
    # Mengambil 6 posisi yang tersebar merata
    step_vis = len(posisi_window) // 6
    posisi_tampil = posisi_window[::step_vis][:6]
else:
    posisi_tampil = posisi_window[:6]

# Memvisualisasikan posisi window pada gambar
for idx, (wx, wy) in enumerate(posisi_tampil):
    # Membuat salinan gambar untuk ditandai
    img_vis = img.copy()

    # Menggambar semua posisi window sebelumnya dengan warna tipis
    for px, py in posisi_window:
        cv2.rectangle(img_vis, (px, py),
                      (px + ukuran_window[0], py + ukuran_window[1]),
                      (200, 200, 200), 1)

    # Menggambar window saat ini dengan warna tebal
    cv2.rectangle(img_vis, (wx, wy),
                  (wx + ukuran_window[0], wy + ukuran_window[1]),
                  (0, 255, 0), 3)

    # Mengkonversi BGR ke RGB untuk matplotlib
    img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

    # Menampilkan gambar dengan window
    axes_flat[idx].imshow(img_rgb)

    # Mengatur judul
    axes_flat[idx].set_title(f"Window di ({wx}, {wy})",
                             fontsize=10, fontweight='bold')

    # Mematikan sumbu
    axes_flat[idx].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 10: Proses Sliding Window pada Gambar\n"
             "Hijau = Window saat ini | Abu = Posisi window lain",
             fontsize=13, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi sliding window
plt.savefig(os.path.join(OUTPUT_DIR, "10_sliding_window.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/10_sliding_window.png")

# ============================================================
# 5. Membuat template untuk deteksi
# ============================================================
print("\n--- 5. Membuat Template untuk Deteksi ---")

# Mengambil bagian gambar sebagai template (simulasi objek target)
# Mengambil region dari pojok kiri atas sebagai template
t_x, t_y = 50, 50
t_w, t_h = 80, 80

# Memastikan template tidak keluar batas gambar
t_x = min(t_x, gray.shape[1] - t_w)
t_y = min(t_y, gray.shape[0] - t_h)

# Mengekstrak template dari gambar grayscale
template = gray[t_y:t_y + t_h, t_x:t_x + t_w]

# Menampilkan informasi template
print(f"  Template diambil dari posisi ({t_x}, {t_y})")
print(f"  Ukuran template: {template.shape}")

# Membuat variasi template yang sedikit berbeda untuk multi-scale
skala_template = [0.8, 1.0, 1.2]
templates = []

for skala in skala_template:
    # Menghitung ukuran baru berdasarkan skala
    new_w = int(t_w * skala)
    new_h = int(t_h * skala)

    # Meresize template ke ukuran baru
    tmpl_resize = cv2.resize(template, (new_w, new_h))

    # Menambahkan ke list templates
    templates.append((tmpl_resize, skala))

    # Menampilkan informasi template
    print(f"  Template skala {skala}: {tmpl_resize.shape}")

# ============================================================
# 6. Template matching untuk deteksi
# ============================================================
print("\n--- 6. Template Matching ---")

# Mendefinisikan threshold untuk deteksi
threshold_deteksi = 0.7

# Menyiapkan list untuk menyimpan semua deteksi
semua_deteksi = []

# Melakukan template matching untuk setiap skala
for tmpl, skala in templates:
    # Mendapatkan ukuran template
    h_tmpl, w_tmpl = tmpl.shape[:2]

    # Memeriksa apakah template lebih kecil dari gambar
    if h_tmpl > gray.shape[0] or w_tmpl > gray.shape[1]:
        print(f"  [SKIP] Template skala {skala} terlalu besar")
        continue

    # Melakukan template matching menggunakan normalized cross-correlation
    result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)

    # Mencari semua lokasi di atas threshold
    lokasi = np.where(result >= threshold_deteksi)

    # Menyimpan setiap deteksi sebagai bounding box dengan skor
    for (y_det, x_det) in zip(*lokasi):
        # Menyimpan (x, y, lebar, tinggi, skor)
        skor = result[y_det, x_det]
        semua_deteksi.append([x_det, y_det, w_tmpl, h_tmpl, skor])

    # Menampilkan jumlah deteksi untuk skala ini
    print(f"  Skala {skala}: {len(lokasi[0])} deteksi (threshold={threshold_deteksi})")

# Mengkonversi ke array numpy
if len(semua_deteksi) > 0:
    semua_deteksi = np.array(semua_deteksi)
else:
    # Membuat deteksi sintetis jika tidak ada yang terdeteksi
    print("  [INFO] Tidak ada deteksi, membuat contoh sintetis...")
    semua_deteksi = np.array([
        [50, 50, 80, 80, 0.95],
        [55, 48, 80, 80, 0.90],
        [60, 52, 80, 80, 0.85],
        [200, 150, 80, 80, 0.92],
        [205, 148, 80, 80, 0.88],
        [400, 80, 80, 80, 0.91],
        [395, 85, 80, 80, 0.86],
        [300, 280, 80, 80, 0.93],
        [100, 300, 80, 80, 0.89],
    ])

# Menampilkan total deteksi sebelum NMS
print(f"\n  Total deteksi sebelum NMS: {len(semua_deteksi)}")

# ============================================================
# 7. Implementasi IoU (Intersection over Union)
# ============================================================
print("\n--- 7. Implementasi IoU ---")

def hitung_iou(box1, box2):
    """
    Menghitung Intersection over Union (IoU) antara dua bounding box.
    Box format: [x, y, w, h]
    IoU = Area_Intersection / Area_Union
    """
    # Menghitung koordinat sudut box 1 (x1, y1, x2, y2)
    x1_min = box1[0]
    y1_min = box1[1]
    x1_max = box1[0] + box1[2]
    y1_max = box1[1] + box1[3]

    # Menghitung koordinat sudut box 2
    x2_min = box2[0]
    y2_min = box2[1]
    x2_max = box2[0] + box2[2]
    y2_max = box2[1] + box2[3]

    # Menghitung koordinat intersection (area tumpang tindih)
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    # Menghitung lebar dan tinggi intersection
    inter_w = max(0, xi_max - xi_min)
    inter_h = max(0, yi_max - yi_min)

    # Menghitung area intersection
    area_intersection = inter_w * inter_h

    # Menghitung area masing-masing box
    area_box1 = box1[2] * box1[3]
    area_box2 = box2[2] * box2[3]

    # Menghitung area union
    area_union = area_box1 + area_box2 - area_intersection

    # Menghitung dan mengembalikan IoU
    iou = area_intersection / (area_union + 1e-7)
    return iou


# Mendemonstrasikan perhitungan IoU
box_a = [100, 100, 80, 80]  # Box A: (100,100) dengan w=80, h=80
box_b = [130, 120, 80, 80]  # Box B: (130,120) dengan w=80, h=80
box_c = [300, 300, 80, 80]  # Box C: (300,300) tidak tumpang tindih

# Menghitung IoU antara Box A dan Box B (tumpang tindih)
iou_ab = hitung_iou(box_a, box_b)
print(f"  IoU(A, B) = {iou_ab:.4f} (tumpang tindih)")

# Menghitung IoU antara Box A dan Box C (tidak tumpang tindih)
iou_ac = hitung_iou(box_a, box_c)
print(f"  IoU(A, C) = {iou_ac:.4f} (tidak tumpang tindih)")

# Menghitung IoU Box A dengan dirinya sendiri
iou_aa = hitung_iou(box_a, box_a)
print(f"  IoU(A, A) = {iou_aa:.4f} (box identik)")

# ============================================================
# 8. Implementasi Non-Maximum Suppression (NMS)
# ============================================================
print("\n--- 8. Implementasi Non-Maximum Suppression (NMS) ---")

def non_maximum_suppression(deteksi, iou_threshold=0.3):
    """
    Implementasi NMS dari nol untuk menghilangkan deteksi duplikat.

    Parameter:
    - deteksi: array Nx5 [x, y, w, h, skor]
    - iou_threshold: threshold IoU untuk menganggap box duplikat

    Return:
    - indeks deteksi yang dipertahankan
    """
    # Memeriksa apakah ada deteksi
    if len(deteksi) == 0:
        return []

    # Mengambil skor confidence dari setiap deteksi
    skor = deteksi[:, 4]

    # Mengurutkan deteksi berdasarkan skor dari tertinggi ke terendah
    indeks_urut = np.argsort(skor)[::-1]

    # Menyiapkan list untuk deteksi yang dipertahankan
    indeks_keep = []

    # Menyiapkan set untuk deteksi yang sudah dihapus
    indeks_hapus = set()

    # Melakukan iterasi dari deteksi dengan skor tertinggi
    for i in indeks_urut:
        # Melewati jika deteksi ini sudah dihapus
        if i in indeks_hapus:
            continue

        # Mempertahankan deteksi ini
        indeks_keep.append(i)

        # Membandingkan dengan semua deteksi lain yang belum dihapus
        for j in indeks_urut:
            # Melewati jika deteksi yang sama atau sudah dihapus
            if j == i or j in indeks_hapus:
                continue

            # Menghitung IoU antara deteksi i dan j
            iou = hitung_iou(deteksi[i, :4], deteksi[j, :4])

            # Menghapus deteksi j jika IoU melebihi threshold
            if iou > iou_threshold:
                indeks_hapus.add(j)

    # Mengembalikan indeks deteksi yang dipertahankan
    return indeks_keep


# Menerapkan NMS pada semua deteksi
indeks_nms = non_maximum_suppression(semua_deteksi, iou_threshold=0.3)

# Mengambil deteksi setelah NMS
deteksi_nms = semua_deteksi[indeks_nms]

# Menampilkan hasil NMS
print(f"  Deteksi sebelum NMS: {len(semua_deteksi)}")
print(f"  Deteksi setelah NMS: {len(deteksi_nms)}")
print(f"  Dihapus oleh NMS   : {len(semua_deteksi) - len(deteksi_nms)}")

# ============================================================
# 9. Visualisasi deteksi template matching
# ============================================================
print("\n--- 9. Visualisasi Deteksi Template Matching ---")

# Membuat figure untuk perbandingan sebelum dan sesudah NMS
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# --- Subplot 1: Template ---
# Menampilkan template yang digunakan
axes[0].imshow(template, cmap='gray')
axes[0].set_title("Template yang Digunakan", fontsize=12, fontweight='bold')
axes[0].axis('off')

# --- Subplot 2: Semua deteksi (sebelum NMS) ---
# Membuat salinan gambar
img_sebelum_nms = img.copy()

# Menggambar semua bounding box deteksi
for det in semua_deteksi:
    # Mengambil koordinat bounding box
    x, y, w, h, skor = det.astype(int)[0:4].tolist() + [det[4]]
    x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])

    # Menggambar rectangle dengan warna berdasarkan skor
    warna = (0, int(255 * skor), 0)
    cv2.rectangle(img_sebelum_nms, (x, y), (x + w, y + h), warna, 2)

# Mengkonversi dan menampilkan
axes[1].imshow(cv2.cvtColor(img_sebelum_nms, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Sebelum NMS ({len(semua_deteksi)} deteksi)",
                  fontsize=12, fontweight='bold')
axes[1].axis('off')

# --- Subplot 3: Deteksi setelah NMS ---
# Membuat salinan gambar
img_setelah_nms = img.copy()

# Menggambar bounding box setelah NMS
for det in deteksi_nms:
    # Mengambil koordinat bounding box
    x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
    skor = det[4]

    # Menggambar rectangle hijau tebal
    cv2.rectangle(img_setelah_nms, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Menuliskan skor di atas bounding box
    cv2.putText(img_setelah_nms, f"{skor:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mengkonversi dan menampilkan
axes[2].imshow(cv2.cvtColor(img_setelah_nms, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Setelah NMS ({len(deteksi_nms)} deteksi)",
                  fontsize=12, fontweight='bold')
axes[2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 10: Template Matching - Sebelum vs Setelah NMS",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi template matching
plt.savefig(os.path.join(OUTPUT_DIR, "10_deteksi_template.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/10_deteksi_template.png")

# ============================================================
# 10. Visualisasi NMS step by step
# ============================================================
print("\n--- 10. Visualisasi NMS Step by Step ---")

# Membuat contoh deteksi yang jelas untuk visualisasi NMS
deteksi_contoh = np.array([
    [100, 80, 90, 90, 0.95],    # Deteksi 1 (skor tertinggi)
    [110, 85, 90, 90, 0.88],    # Deteksi 2 (tumpang tindih dengan 1)
    [115, 90, 90, 90, 0.82],    # Deteksi 3 (tumpang tindih dengan 1)
    [350, 100, 90, 90, 0.93],   # Deteksi 4 (lokasi berbeda)
    [355, 105, 90, 90, 0.85],   # Deteksi 5 (tumpang tindih dengan 4)
    [200, 250, 90, 90, 0.91],   # Deteksi 6 (lokasi berbeda)
    [205, 255, 90, 90, 0.80],   # Deteksi 7 (tumpang tindih dengan 6)
])

# Membuat figure 2x2 untuk visualisasi NMS
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# --- Subplot 1: Semua deteksi ---
# Membuat canvas putih
canvas1 = np.ones((400, 500, 3), dtype=np.uint8) * 240

# Menggambar semua deteksi dengan warna intensitas sesuai skor
for det in deteksi_contoh:
    x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
    skor = det[4]
    # Menghitung intensitas warna berdasarkan skor
    intensitas = int(255 * skor)
    cv2.rectangle(canvas1, (x, y), (x + w, y + h), (0, intensitas, 0), 2)
    # Menuliskan skor
    cv2.putText(canvas1, f"{skor:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Menampilkan canvas
axes[0, 0].imshow(cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"Step 1: Semua Deteksi ({len(deteksi_contoh)})",
                      fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# --- Subplot 2: Urutkan berdasarkan skor ---
# Membuat canvas
canvas2 = np.ones((400, 500, 3), dtype=np.uint8) * 240

# Mengurutkan deteksi berdasarkan skor
urut_skor = np.argsort(deteksi_contoh[:, 4])[::-1]

# Menggambar deteksi terurut dengan nomor urut
for rank, idx in enumerate(urut_skor):
    det = deteksi_contoh[idx]
    x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
    skor = det[4]
    cv2.rectangle(canvas2, (x, y), (x + w, y + h), (0, 200, 0), 2)
    cv2.putText(canvas2, f"#{rank+1} ({skor:.2f})", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1)

# Menampilkan canvas
axes[0, 1].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Step 2: Urutkan Berdasarkan Skor",
                      fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# --- Subplot 3: IoU comparison ---
# Membuat canvas
canvas3 = np.ones((400, 500, 3), dtype=np.uint8) * 240

# Menggambar box 1 dan 2 untuk menunjukkan IoU
det1 = deteksi_contoh[0]
det2 = deteksi_contoh[1]

# Menggambar box hijau (dipertahankan)
x1, y1, w1, h1 = int(det1[0]), int(det1[1]), int(det1[2]), int(det1[3])
cv2.rectangle(canvas3, (x1, y1), (x1+w1, y1+h1), (0, 200, 0), 3)
cv2.putText(canvas3, f"KEEP ({det1[4]:.2f})", (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

# Menggambar box merah (dihapus karena IoU tinggi)
x2, y2, w2, h2 = int(det2[0]), int(det2[1]), int(det2[2]), int(det2[3])
cv2.rectangle(canvas3, (x2, y2), (x2+w2, y2+h2), (0, 0, 200), 2)

# Menghitung dan menuliskan IoU
iou_12 = hitung_iou(det1[:4], det2[:4])
cv2.putText(canvas3, f"REMOVE (IoU={iou_12:.2f})", (x2, y2+h2+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

# Menampilkan canvas
axes[1, 0].imshow(cv2.cvtColor(canvas3, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Step 3: Hapus Box dengan IoU Tinggi",
                      fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# --- Subplot 4: Hasil NMS ---
# Menerapkan NMS pada deteksi contoh
indeks_nms_contoh = non_maximum_suppression(deteksi_contoh, iou_threshold=0.3)
det_nms_contoh = deteksi_contoh[indeks_nms_contoh]

# Membuat canvas
canvas4 = np.ones((400, 500, 3), dtype=np.uint8) * 240

# Menggambar deteksi final setelah NMS
for det in det_nms_contoh:
    x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
    skor = det[4]
    cv2.rectangle(canvas4, (x, y), (x + w, y + h), (0, 200, 0), 3)
    cv2.putText(canvas4, f"{skor:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

# Menampilkan canvas
axes[1, 1].imshow(cv2.cvtColor(canvas4, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"Step 4: Hasil NMS ({len(det_nms_contoh)} deteksi final)",
                      fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 10: Non-Maximum Suppression (NMS) Step by Step",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi NMS
plt.savefig(os.path.join(OUTPUT_DIR, "10_nms_result.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/10_nms_result.png")

# ============================================================
# 11. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 10")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. Sliding window memindai gambar dengan jendela geser
2. Template matching (cv2.matchTemplate) sebagai "detektor" sederhana
3. Bounding box merepresentasikan lokasi objek terdeteksi
4. IoU mengukur tumpang tindih dua bounding box (0=tidak, 1=identik)
5. NMS menghilangkan deteksi duplikat berdasarkan IoU
6. Multi-scale detection menggunakan template berbagai ukuran
7. Threshold confidence menentukan batas minimum skor deteksi
8. NMS mempertahankan deteksi dengan skor tertinggi per area

Output disimpan di folder: output/
- 10_sliding_window.png  : Visualisasi proses sliding window
- 10_deteksi_template.png: Deteksi sebelum dan setelah NMS
- 10_nms_result.png      : NMS step by step
""")
