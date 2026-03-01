"""
==========================================================================
PERCOBAAN 13: SEMANTIC SEGMENTATION - MANUAL
==========================================================================
Program ini mempelajari konsep semantic segmentation, yaitu klasifikasi
setiap piksel gambar ke dalam suatu kelas. Implementasi mencakup
segmentasi berbasis warna (thresholding per channel), segmentasi HSV,
pembuatan mask berwarna, overlay semi-transparan, perhitungan area
per kelas, dan metode watershed sebagai pendekatan lanjut.

Fungsi utama yang dipelajari:
- cv2.inRange()           : Thresholding range warna untuk segmentasi
- cv2.cvtColor()          : Konversi ruang warna (BGR ke HSV, dll)
- cv2.addWeighted()       : Overlay semi-transparan mask pada gambar
- cv2.watershed()         : Segmentasi watershed berbasis marker
- cv2.morphologyEx()      : Operasi morfologi untuk membersihkan mask
- cv2.connectedComponents(): Labeling komponen terhubung

Konsep yang dipelajari:
- Semantic segmentation: klasifikasi per piksel
- Color-based segmentation: threshold pada channel warna
- Segmentasi HSV vs BGR: perbandingan pendekatan
- Mask overlay: visualisasi hasil segmentasi pada gambar asli
- Area calculation: persentase area per kelas
- Watershed: segmentasi berbasis topografi intensitas
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan komputasi numerik
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi hasil segmentasi
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
print("PERCOBAAN 13: SEMANTIC SEGMENTATION - MANUAL")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep semantic segmentation
# ============================================================
print("\n--- 1. Konsep Semantic Segmentation ---")

# Menjelaskan konsep semantic segmentation
print("""
  Semantic Segmentation = klasifikasi per piksel gambar.

  Setiap piksel diklasifikasikan ke salah satu kelas:
  - Langit (sky)
  - Vegetasi (tumbuhan/pohon)
  - Jalan (road)
  - Bangunan (building)
  - Lainnya (other)

  Pendekatan yang digunakan:
  1. Color-based (BGR thresholding): langsung pada channel warna
  2. HSV-based: menggunakan ruang warna HSV (lebih robust)
  3. Watershed: segmentasi berbasis topografi intensitas

  Perbedaan dengan deteksi objek:
  - Deteksi: bounding box di sekitar objek
  - Segmentation: mask piksel-tepat untuk setiap kelas
""")

# ============================================================
# 2. Memuat gambar scene untuk segmentasi
# ============================================================
print("\n--- 2. Memuat Gambar Scene ---")

# Memuat gambar scene outdoor
img = cv2.imread(os.path.join(IMAGE_DIR, "scene_outdoor.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img is None:
    # Mencoba alternatif
    img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
    if img is None:
        # Membuat gambar sintetis scene outdoor
        print("  [INFO] Membuat gambar sintetis scene outdoor...")
        img = np.zeros((400, 600, 3), dtype=np.uint8)

        # Menggambar langit (biru)
        img[:150, :] = [230, 180, 130]  # BGR: biru muda

        # Menggambar awan putih
        cv2.ellipse(img, (150, 60), (80, 30), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (400, 80), (100, 35), 0, 0, 360, (250, 250, 250), -1)

        # Menggambar vegetasi (hijau)
        img[150:250, :] = [30, 130, 30]  # BGR: hijau

        # Menambahkan variasi pada vegetasi
        for i in range(20):
            x = np.random.randint(0, 600)
            y = np.random.randint(150, 250)
            radius = np.random.randint(10, 30)
            hijau = np.random.randint(80, 180)
            cv2.circle(img, (x, y), radius, (20, hijau, 20), -1)

        # Menggambar jalan (abu-abu)
        pts_jalan = np.array([[150, 400], [450, 400],
                              [500, 250], [100, 250]], np.int32)
        cv2.fillPoly(img, [pts_jalan], (100, 100, 100))

        # Menggambar bangunan
        cv2.rectangle(img, (50, 180), (120, 310), (140, 140, 160), -1)
        cv2.rectangle(img, (480, 170), (570, 310), (150, 130, 130), -1)

        # Menambahkan noise ringan
        noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

# Meresize gambar
img = cv2.resize(img, (600, 400))

# Menampilkan informasi gambar
print(f"  Ukuran gambar: {img.shape}")
print(f"  Tipe data    : {img.dtype}")

# ============================================================
# 3. Segmentasi berbasis warna (BGR thresholding)
# ============================================================
print("\n--- 3. Segmentasi Berbasis Warna (BGR) ---")

# Mendefinisikan range warna untuk setiap kelas dalam BGR
# Kelas: Langit (biru dominan)
batas_bawah_langit_bgr = np.array([130, 100, 50])
batas_atas_langit_bgr = np.array([255, 220, 180])

# Kelas: Vegetasi (hijau dominan)
batas_bawah_vegetasi_bgr = np.array([0, 60, 0])
batas_atas_vegetasi_bgr = np.array([100, 200, 80])

# Kelas: Jalan (abu-abu, semua channel mirip)
batas_bawah_jalan_bgr = np.array([60, 60, 60])
batas_atas_jalan_bgr = np.array([140, 140, 140])

# Membuat mask untuk langit menggunakan inRange BGR
mask_langit_bgr = cv2.inRange(img, batas_bawah_langit_bgr, batas_atas_langit_bgr)

# Membuat mask untuk vegetasi
mask_vegetasi_bgr = cv2.inRange(img, batas_bawah_vegetasi_bgr, batas_atas_vegetasi_bgr)

# Membuat mask untuk jalan
mask_jalan_bgr = cv2.inRange(img, batas_bawah_jalan_bgr, batas_atas_jalan_bgr)

# Membersihkan mask dengan operasi morfologi
kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Membersihkan mask langit
mask_langit_bgr = cv2.morphologyEx(mask_langit_bgr, cv2.MORPH_CLOSE, kernel_morph)
mask_langit_bgr = cv2.morphologyEx(mask_langit_bgr, cv2.MORPH_OPEN, kernel_morph)

# Membersihkan mask vegetasi
mask_vegetasi_bgr = cv2.morphologyEx(mask_vegetasi_bgr, cv2.MORPH_CLOSE, kernel_morph)
mask_vegetasi_bgr = cv2.morphologyEx(mask_vegetasi_bgr, cv2.MORPH_OPEN, kernel_morph)

# Membersihkan mask jalan
mask_jalan_bgr = cv2.morphologyEx(mask_jalan_bgr, cv2.MORPH_CLOSE, kernel_morph)
mask_jalan_bgr = cv2.morphologyEx(mask_jalan_bgr, cv2.MORPH_OPEN, kernel_morph)

# Menghitung persentase area per kelas
total_piksel = img.shape[0] * img.shape[1]
persen_langit = np.sum(mask_langit_bgr > 0) / total_piksel * 100
persen_vegetasi = np.sum(mask_vegetasi_bgr > 0) / total_piksel * 100
persen_jalan = np.sum(mask_jalan_bgr > 0) / total_piksel * 100
persen_lainnya = 100 - persen_langit - persen_vegetasi - persen_jalan

# Menampilkan persentase area
print(f"  Segmentasi BGR:")
print(f"    Langit   : {persen_langit:.1f}%")
print(f"    Vegetasi : {persen_vegetasi:.1f}%")
print(f"    Jalan    : {persen_jalan:.1f}%")
print(f"    Lainnya  : {persen_lainnya:.1f}%")

# ============================================================
# 4. Segmentasi berbasis HSV
# ============================================================
print("\n--- 4. Segmentasi Berbasis HSV ---")

# Mengkonversi gambar ke ruang warna HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Mendefinisikan range warna dalam HSV
# Langit: Hue biru (90-130), Saturation rendah-sedang
batas_bawah_langit_hsv = np.array([90, 30, 100])
batas_atas_langit_hsv = np.array([130, 255, 255])

# Vegetasi: Hue hijau (30-85), Saturation sedang-tinggi
batas_bawah_vegetasi_hsv = np.array([30, 40, 30])
batas_atas_vegetasi_hsv = np.array([85, 255, 200])

# Jalan: Saturation sangat rendah (abu-abu), Value sedang
batas_bawah_jalan_hsv = np.array([0, 0, 50])
batas_atas_jalan_hsv = np.array([180, 50, 150])

# Membuat mask untuk langit menggunakan inRange HSV
mask_langit_hsv = cv2.inRange(img_hsv, batas_bawah_langit_hsv, batas_atas_langit_hsv)

# Membuat mask untuk vegetasi HSV
mask_vegetasi_hsv = cv2.inRange(img_hsv, batas_bawah_vegetasi_hsv, batas_atas_vegetasi_hsv)

# Membuat mask untuk jalan HSV
mask_jalan_hsv = cv2.inRange(img_hsv, batas_bawah_jalan_hsv, batas_atas_jalan_hsv)

# Membersihkan mask HSV dengan morfologi
mask_langit_hsv = cv2.morphologyEx(mask_langit_hsv, cv2.MORPH_CLOSE, kernel_morph)
mask_langit_hsv = cv2.morphologyEx(mask_langit_hsv, cv2.MORPH_OPEN, kernel_morph)

mask_vegetasi_hsv = cv2.morphologyEx(mask_vegetasi_hsv, cv2.MORPH_CLOSE, kernel_morph)
mask_vegetasi_hsv = cv2.morphologyEx(mask_vegetasi_hsv, cv2.MORPH_OPEN, kernel_morph)

mask_jalan_hsv = cv2.morphologyEx(mask_jalan_hsv, cv2.MORPH_CLOSE, kernel_morph)
mask_jalan_hsv = cv2.morphologyEx(mask_jalan_hsv, cv2.MORPH_OPEN, kernel_morph)

# Menghitung persentase area HSV
persen_langit_hsv = np.sum(mask_langit_hsv > 0) / total_piksel * 100
persen_vegetasi_hsv = np.sum(mask_vegetasi_hsv > 0) / total_piksel * 100
persen_jalan_hsv = np.sum(mask_jalan_hsv > 0) / total_piksel * 100
persen_lainnya_hsv = 100 - persen_langit_hsv - persen_vegetasi_hsv - persen_jalan_hsv

# Menampilkan persentase area HSV
print(f"  Segmentasi HSV:")
print(f"    Langit   : {persen_langit_hsv:.1f}%")
print(f"    Vegetasi : {persen_vegetasi_hsv:.1f}%")
print(f"    Jalan    : {persen_jalan_hsv:.1f}%")
print(f"    Lainnya  : {persen_lainnya_hsv:.1f}%")

# ============================================================
# 5. Membuat colored segmentation mask
# ============================================================
print("\n--- 5. Membuat Colored Segmentation Mask ---")

# Mendefinisikan warna untuk setiap kelas (BGR)
warna_kelas = {
    'langit': (255, 150, 50),     # Biru muda
    'vegetasi': (50, 200, 50),    # Hijau
    'jalan': (100, 100, 100),     # Abu-abu
    'lainnya': (180, 180, 200),   # Putih keabu-abuan
}

def buat_mask_berwarna(mask_langit, mask_vegetasi, mask_jalan, shape):
    """
    Membuat mask segmentasi berwarna dari mask biner per kelas.
    """
    # Membuat gambar kosong untuk mask berwarna
    mask_warna = np.zeros(shape, dtype=np.uint8)

    # Mengisi warna untuk kelas lainnya (default)
    mask_warna[:] = warna_kelas['lainnya']

    # Mengisi warna untuk jalan (ditimpa di atas lainnya)
    mask_warna[mask_jalan > 0] = warna_kelas['jalan']

    # Mengisi warna untuk vegetasi
    mask_warna[mask_vegetasi > 0] = warna_kelas['vegetasi']

    # Mengisi warna untuk langit
    mask_warna[mask_langit > 0] = warna_kelas['langit']

    # Mengembalikan mask berwarna
    return mask_warna


# Membuat mask berwarna dari segmentasi BGR
mask_warna_bgr = buat_mask_berwarna(mask_langit_bgr, mask_vegetasi_bgr,
                                     mask_jalan_bgr, img.shape)

# Membuat mask berwarna dari segmentasi HSV
mask_warna_hsv = buat_mask_berwarna(mask_langit_hsv, mask_vegetasi_hsv,
                                     mask_jalan_hsv, img.shape)

# Menampilkan informasi mask
print(f"  Mask BGR shape: {mask_warna_bgr.shape}")
print(f"  Mask HSV shape: {mask_warna_hsv.shape}")

# ============================================================
# 6. Visualisasi segmentasi warna (BGR vs HSV)
# ============================================================
print("\n--- 6. Visualisasi Segmentasi BGR vs HSV ---")

# Membuat figure untuk perbandingan
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- Baris 1: Segmentasi BGR ---
# Gambar asli
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Mask BGR individual
mask_gabung_bgr = np.zeros(img.shape[:2], dtype=np.uint8)
mask_gabung_bgr[mask_langit_bgr > 0] = 1
mask_gabung_bgr[mask_vegetasi_bgr > 0] = 2
mask_gabung_bgr[mask_jalan_bgr > 0] = 3

# Menampilkan mask berwarna BGR
axes[0, 1].imshow(cv2.cvtColor(mask_warna_bgr, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Segmentasi BGR\n(Biru=Langit, Hijau=Vegetasi, Abu=Jalan)",
                      fontsize=10, fontweight='bold')
axes[0, 1].axis('off')

# Overlay BGR pada gambar
overlay_bgr = cv2.addWeighted(img, 0.5, mask_warna_bgr, 0.5, 0)
axes[0, 2].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Overlay BGR (50% transparan)",
                      fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# --- Baris 2: Segmentasi HSV ---
# Gambar HSV
axes[1, 0].imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
axes[1, 0].set_title("Gambar dalam HSV", fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# Mask HSV berwarna
axes[1, 1].imshow(cv2.cvtColor(mask_warna_hsv, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Segmentasi HSV\n(Biru=Langit, Hijau=Vegetasi, Abu=Jalan)",
                      fontsize=10, fontweight='bold')
axes[1, 1].axis('off')

# Overlay HSV pada gambar
overlay_hsv = cv2.addWeighted(img, 0.5, mask_warna_hsv, 0.5, 0)
axes[1, 2].imshow(cv2.cvtColor(overlay_hsv, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Overlay HSV (50% transparan)",
                      fontsize=11, fontweight='bold')
axes[1, 2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 13: Perbandingan Segmentasi Warna BGR vs HSV",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi
plt.savefig(os.path.join(OUTPUT_DIR, "13_segmentasi_warna.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/13_segmentasi_warna.png")

# ============================================================
# 7. Visualisasi mask individual per kelas
# ============================================================
print("\n--- 7. Visualisasi Mask Individual ---")

# Membuat figure untuk mask individual (menggunakan HSV yang lebih baik)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- Mask langit ---
axes[0, 0].imshow(mask_langit_hsv, cmap='Blues')
axes[0, 0].set_title(f"Mask Langit ({persen_langit_hsv:.1f}%)",
                      fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# --- Mask vegetasi ---
axes[0, 1].imshow(mask_vegetasi_hsv, cmap='Greens')
axes[0, 1].set_title(f"Mask Vegetasi ({persen_vegetasi_hsv:.1f}%)",
                      fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# --- Mask jalan ---
axes[0, 2].imshow(mask_jalan_hsv, cmap='gray')
axes[0, 2].set_title(f"Mask Jalan ({persen_jalan_hsv:.1f}%)",
                      fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# --- Segmentasi gabungan ---
axes[1, 0].imshow(cv2.cvtColor(mask_warna_hsv, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Mask Gabungan (Berwarna)", fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# --- Pie chart area ---
labels_pie = ['Langit', 'Vegetasi', 'Jalan', 'Lainnya']
sizes_pie = [persen_langit_hsv, persen_vegetasi_hsv,
             persen_jalan_hsv, persen_lainnya_hsv]
colors_pie = ['#3399FF', '#33CC33', '#888888', '#CCCCDD']
explode_pie = (0.05, 0.05, 0.05, 0.02)

# Memplot pie chart
axes[1, 1].pie(sizes_pie, explode=explode_pie, labels=labels_pie,
               colors=colors_pie, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 10})
axes[1, 1].set_title("Persentase Area per Kelas", fontsize=11, fontweight='bold')

# --- Overlay final ---
axes[1, 2].imshow(cv2.cvtColor(overlay_hsv, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Overlay pada Gambar Asli", fontsize=11, fontweight='bold')
axes[1, 2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 13: Mask Segmentasi per Kelas dan Area",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi mask
plt.savefig(os.path.join(OUTPUT_DIR, "13_segmentasi_mask.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/13_segmentasi_mask.png")

# ============================================================
# 8. Segmentasi watershed sebagai metode lanjut
# ============================================================
print("\n--- 8. Segmentasi Watershed ---")

# Mengkonversi gambar ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Menerapkan Gaussian blur untuk mengurangi noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Menerapkan threshold Otsu untuk mendapatkan biner
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Menampilkan informasi threshold
print(f"  Threshold Otsu diterapkan")

# Membersihkan mask dengan morphology (opening untuk menghilangkan noise)
kernel_ws = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_ws, iterations=2)

# Menentukan area yang pasti background (dilasi dari opening)
background_pasti = cv2.dilate(opening, kernel_ws, iterations=3)

# Menentukan area yang pasti foreground (distance transform + threshold)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Menormalisasi distance transform
dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)

# Menentukan threshold untuk foreground pasti
_, foreground_pasti = cv2.threshold(dist_transform, 0.5 * dist_transform.max(),
                                     255, 0)
foreground_pasti = np.uint8(foreground_pasti)

# Menentukan area yang tidak diketahui (unknown)
unknown = cv2.subtract(background_pasti, foreground_pasti)

# Membuat marker untuk watershed
_, markers = cv2.connectedComponents(foreground_pasti)

# Menambahkan 1 ke semua marker agar background bukan 0 (0 = unknown di watershed)
markers = markers + 1

# Menandai area unknown sebagai 0
markers[unknown == 255] = 0

# Menampilkan informasi markers
print(f"  Jumlah marker (region): {markers.max()}")

# Menerapkan algoritma watershed
markers_ws = markers.copy()
cv2.watershed(img, markers_ws)

# Membuat gambar hasil watershed
img_watershed = img.copy()

# Menandai batas watershed dengan warna merah
img_watershed[markers_ws == -1] = [0, 0, 255]

# Membuat mask berwarna berdasarkan marker watershed
mask_watershed = np.zeros(img.shape, dtype=np.uint8)

# Menghasilkan warna random untuk setiap region
np.random.seed(42)
warna_region = {}
for marker_id in range(1, markers_ws.max() + 1):
    # Menghasilkan warna random untuk setiap region
    warna_region[marker_id] = (
        np.random.randint(50, 255),
        np.random.randint(50, 255),
        np.random.randint(50, 255)
    )
    # Mewarnai region pada mask
    mask_watershed[markers_ws == marker_id] = warna_region[marker_id]

# Menampilkan jumlah region terdeteksi
print(f"  Region terdeteksi   : {markers_ws.max()}")

# ============================================================
# 9. Visualisasi segmentasi overlay
# ============================================================
print("\n--- 9. Visualisasi Segmentasi Overlay ---")

# Membuat figure untuk visualisasi overlay
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- Subplot 1: Gambar asli ---
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# --- Subplot 2: Distance transform ---
axes[0, 1].imshow(dist_norm, cmap='jet')
axes[0, 1].set_title("Distance Transform", fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# --- Subplot 3: Markers ---
axes[0, 2].imshow(markers, cmap='nipy_spectral')
axes[0, 2].set_title("Markers untuk Watershed", fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# --- Subplot 4: Hasil watershed ---
axes[1, 0].imshow(cv2.cvtColor(img_watershed, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Watershed (batas merah)", fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# --- Subplot 5: Mask berwarna watershed ---
axes[1, 1].imshow(cv2.cvtColor(mask_watershed, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"Mask Watershed ({markers_ws.max()} region)",
                      fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# --- Subplot 6: Overlay watershed ---
# Membuat overlay watershed semi-transparan
overlay_ws = cv2.addWeighted(img, 0.6, mask_watershed, 0.4, 0)

# Menambahkan garis batas watershed
overlay_ws[markers_ws == -1] = [0, 0, 255]

# Menampilkan overlay
axes[1, 2].imshow(cv2.cvtColor(overlay_ws, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Overlay Watershed", fontsize=11, fontweight='bold')
axes[1, 2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 13: Metode Segmentasi Lanjut - Watershed",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi overlay
plt.savefig(os.path.join(OUTPUT_DIR, "13_segmentasi_overlay.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/13_segmentasi_overlay.png")

# ============================================================
# 10. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 13")
print("=" * 60)
print(f"""
Konsep yang telah dipelajari:
1. Semantic segmentation mengklasifikasikan setiap piksel ke suatu kelas
2. cv2.inRange() membuat mask biner berdasarkan range warna
3. Segmentasi BGR: langsung threshold pada channel Blue, Green, Red
4. Segmentasi HSV: lebih robust karena memisahkan warna (H) dari intensitas (V)
5. Operasi morfologi (open/close) membersihkan mask dari noise
6. cv2.addWeighted() membuat overlay semi-transparan mask pada gambar
7. Persentase area per kelas menunjukkan komposisi scene
8. cv2.watershed() melakukan segmentasi berbasis topografi intensitas

Hasil Segmentasi HSV:
  Langit   : {persen_langit_hsv:.1f}%
  Vegetasi : {persen_vegetasi_hsv:.1f}%
  Jalan    : {persen_jalan_hsv:.1f}%
  Lainnya  : {persen_lainnya_hsv:.1f}%

Output disimpan di folder: output/
- 13_segmentasi_warna.png  : Perbandingan segmentasi BGR vs HSV
- 13_segmentasi_mask.png   : Mask individual per kelas
- 13_segmentasi_overlay.png: Segmentasi watershed overlay
""")
