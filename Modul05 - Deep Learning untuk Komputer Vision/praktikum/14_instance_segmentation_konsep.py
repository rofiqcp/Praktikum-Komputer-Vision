"""
==========================================================================
PERCOBAAN 14: INSTANCE SEGMENTATION - KONSEP
==========================================================================
Program ini mempelajari konsep instance segmentation, yaitu segmentasi
yang tidak hanya mengklasifikasikan setiap piksel ke suatu kelas
(semantic), tetapi juga membedakan setiap instansi (objek individual)
dalam kelas yang sama. Implementasi menggunakan contour detection dan
connected components untuk label dan segmentasi setiap objek.

Fungsi utama yang dipelajari:
- cv2.findContours()               : Menemukan kontur objek
- cv2.connectedComponentsWithStats(): Labeling komponen terhubung + statistik
- cv2.drawContours()               : Menggambar kontur pada gambar
- cv2.boundingRect()               : Bounding box dari kontur
- cv2.contourArea()                : Menghitung area kontur
- cv2.moments()                    : Menghitung momen (centroid) kontur

Konsep yang dipelajari:
- Perbedaan semantic vs instance segmentation
- Contour-based instance segmentation
- Connected components labeling
- Properti per instansi: area, centroid, bounding box
- Pewarnaan unik per instansi
- Perbandingan visual semantic vs instance segmentation
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan komputasi numerik
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor glob untuk pencarian file berdasarkan pola
import glob

# Mengimpor matplotlib untuk visualisasi hasil
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Mendefinisikan path folder dataset
DATASET_DIR = os.path.join(IMAGE_DIR, "dataset")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan header percobaan
print("=" * 60)
print("PERCOBAAN 14: INSTANCE SEGMENTATION - KONSEP")
print("=" * 60)

# ============================================================
# 1. Penjelasan perbedaan semantic vs instance segmentation
# ============================================================
print("\n--- 1. Semantic vs Instance Segmentation ---")

# Menjelaskan perbedaan kedua pendekatan
print("""
  Semantic Segmentation:
  - Mengklasifikasikan setiap piksel ke suatu KELAS
  - Semua objek kelas yang sama memiliki label/warna yang SAMA
  - Contoh: semua "kucing" berwarna hijau, semua "anjing" biru

  Instance Segmentation:
  - Mengklasifikasikan setiap piksel ke INSTANSI INDIVIDUAL
  - Setiap objek memiliki label/warna yang UNIK
  - Contoh: kucing#1 hijau, kucing#2 merah, kucing#3 biru

  Perbedaan kunci:
  | Aspek           | Semantic      | Instance          |
  |-----------------|---------------|-------------------|
  | Output          | N kelas       | M instansi        |
  | Warna sama      | Per kelas     | Per objek         |
  | Jumlah objek    | Tidak tahu    | Tahu pasti        |
  | Contoh metode   | Threshold     | Contour, Conn.Comp|
""")

# ============================================================
# 2. Memuat gambar asli untuk instance segmentation
# ============================================================
print("\n--- 2. Memuat Gambar Asli ---")

# Memuat gambar asli dari dataset (prioritas: lingkaran, lalu gambar lain)
img_sintetis = None
dataset_path = os.path.join(DATASET_DIR, "lingkaran")
if os.path.exists(dataset_path):
    file_list = glob.glob(os.path.join(dataset_path, "*.*"))
    file_list = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(file_list) > 0:
        img_sintetis = cv2.imread(file_list[0])
        if img_sintetis is not None:
            img_sintetis = cv2.resize(img_sintetis, (700, 500))
            print(f"  [OK] Gambar dimuat dari dataset: {os.path.basename(file_list[0])}")

# Fallback: gunakan gambar kucing atau bunga dari IMAGE_DIR
if img_sintetis is None:
    for fn in ["kucing.jpg", "bunga.jpg", "scene_indoor.jpg", "augmentasi_sample.jpg"]:
        img_sintetis = cv2.imread(os.path.join(IMAGE_DIR, fn))
        if img_sintetis is not None:
            img_sintetis = cv2.resize(img_sintetis, (700, 500))
            print(f"  [OK] Gambar dimuat: {fn}")
            break

if img_sintetis is None:
    print("[ERROR] Tidak ada gambar asli ditemukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()

# Mendefinisikan objek_list sebagai placeholder
objek_list = []

# Menambahkan noise ringan sebagai pre-processing deterministik
noise = np.random.randint(0, 5, img_sintetis.shape, dtype=np.uint8)
img_sintetis = cv2.add(img_sintetis, noise)

# Mencoba memuat gambar dari dataset juga
img_dataset = None
if os.path.exists(dataset_path):
    # Mencari gambar pertama di folder lingkaran
    file_list2 = glob.glob(os.path.join(dataset_path, "*.*"))
    file_list2 = [f for f in file_list2 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(file_list2) > 1:
        img_dataset = cv2.imread(file_list2[1])
        if img_dataset is not None:
            img_dataset = cv2.resize(img_dataset, (200, 200))

# Menampilkan informasi gambar
print(f"  Ukuran gambar   : {img_sintetis.shape}")
print("  (Gambar asli digunakan untuk demonstrasi instance segmentation)")

# ============================================================
# 3. Instance segmentation berbasis kontur
# ============================================================
print("\n--- 3. Instance Segmentation Berbasis Kontur ---")

# Mengkonversi gambar sintetis ke grayscale
gray = cv2.cvtColor(img_sintetis, cv2.COLOR_BGR2GRAY)

# Menerapkan Gaussian blur untuk mengurangi noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Menerapkan threshold untuk mendapatkan gambar biner
# Objek gelap pada latar belakang terang
_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

# Membersihkan threshold dengan operasi morfologi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Menemukan kontur pada gambar biner
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

# Menampilkan jumlah kontur ditemukan
print(f"  Kontur ditemukan: {len(contours)}")

# Menyaring kontur berdasarkan area minimum
area_min = 500  # Minimum 500 piksel
contours_valid = []

for contour in contours:
    # Menghitung area kontur
    area = cv2.contourArea(contour)

    # Menyimpan kontur yang cukup besar
    if area >= area_min:
        contours_valid.append(contour)

# Menampilkan jumlah kontur valid
print(f"  Kontur valid (area >= {area_min}): {len(contours_valid)}")

# ============================================================
# 4. Mengekstrak properti setiap instansi
# ============================================================
print("\n--- 4. Properti Setiap Instansi ---")

# Menyiapkan list untuk menyimpan properti instansi
properti_instansi = []

# Menghitung properti untuk setiap kontur valid
for idx, contour in enumerate(contours_valid):
    # Menghitung area kontur
    area = cv2.contourArea(contour)

    # Menghitung keliling kontur
    keliling = cv2.arcLength(contour, True)

    # Menghitung bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Menghitung momen untuk centroid
    M = cv2.moments(contour)

    # Menghitung centroid (pusat massa)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = x + w // 2
        cy = y + h // 2

    # Menghitung circularity (bulatan = 4*pi*area/keliling^2)
    if keliling > 0:
        circularity = 4 * np.pi * area / (keliling ** 2)
    else:
        circularity = 0

    # Menghitung aspect ratio (rasio lebar/tinggi)
    aspect_ratio = float(w) / h if h > 0 else 0

    # Menyimpan properti instansi
    props = {
        'id': idx + 1,
        'area': area,
        'keliling': keliling,
        'bbox': (x, y, w, h),
        'centroid': (cx, cy),
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'contour': contour
    }
    properti_instansi.append(props)

    # Menampilkan properti
    print(f"\n  Instansi #{idx + 1}:")
    print(f"    Area         : {area:.0f} piksel")
    print(f"    Keliling     : {keliling:.1f} piksel")
    print(f"    Bounding Box : ({x}, {y}, {w}, {h})")
    print(f"    Centroid     : ({cx}, {cy})")
    print(f"    Circularity  : {circularity:.3f}")
    print(f"    Aspect Ratio : {aspect_ratio:.2f}")

# ============================================================
# 5. Mewarnai setiap instansi dengan warna unik
# ============================================================
print("\n--- 5. Pewarnaan Instansi Unik ---")

# Menghasilkan warna unik untuk setiap instansi
np.random.seed(42)
warna_instansi = []
for i in range(len(contours_valid)):
    # Menghasilkan warna random yang cukup terang
    warna = (
        np.random.randint(50, 255),
        np.random.randint(50, 255),
        np.random.randint(50, 255)
    )
    warna_instansi.append(warna)

# Membuat mask instance segmentation (setiap objek warna berbeda)
mask_instance = np.zeros(img_sintetis.shape, dtype=np.uint8)

# Mewarnai setiap instansi
for idx, contour in enumerate(contours_valid):
    # Menggambar kontur terisi dengan warna unik
    cv2.drawContours(mask_instance, [contour], -1, warna_instansi[idx], -1)

# Menampilkan informasi
print(f"  Jumlah instansi diberi warna: {len(contours_valid)}")

# ============================================================
# 6. Visualisasi kontur dan properti instansi
# ============================================================
print("\n--- 6. Visualisasi Kontur dan Properti ---")

# Membuat figure untuk visualisasi kontur
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- Subplot 1: Gambar asli ---
axes[0, 0].imshow(cv2.cvtColor(img_sintetis, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli (8 objek)", fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# --- Subplot 2: Threshold biner ---
axes[0, 1].imshow(thresh, cmap='gray')
axes[0, 1].set_title("Threshold Biner", fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# --- Subplot 3: Kontur terdeteksi ---
img_kontur = img_sintetis.copy()

# Menggambar semua kontur dengan warna dan properties
for idx, props in enumerate(properti_instansi):
    # Menggambar kontur
    cv2.drawContours(img_kontur, [props['contour']], -1,
                     warna_instansi[idx], 3)

    # Menggambar bounding box
    x, y, w, h = props['bbox']
    cv2.rectangle(img_kontur, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Menandai centroid
    cx, cy = props['centroid']
    cv2.circle(img_kontur, (cx, cy), 5, (0, 255, 255), -1)

    # Menuliskan ID instansi
    cv2.putText(img_kontur, f"#{props['id']}", (cx - 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Menampilkan kontur
axes[0, 2].imshow(cv2.cvtColor(img_kontur, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"Kontur + BBox + Centroid ({len(contours_valid)} instansi)",
                      fontsize=10, fontweight='bold')
axes[0, 2].axis('off')

# --- Subplot 4: Instance mask (warna unik) ---
axes[1, 0].imshow(cv2.cvtColor(mask_instance, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Instance Segmentation (warna unik)",
                      fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# --- Subplot 5: Overlay instance ---
overlay_inst = cv2.addWeighted(img_sintetis, 0.5, mask_instance, 0.5, 0)
axes[1, 1].imshow(cv2.cvtColor(overlay_inst, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Overlay Instance pada Gambar",
                      fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# --- Subplot 6: Tabel properti ---
axes[1, 2].axis('off')
tabel_data = []
for props in properti_instansi:
    tabel_data.append([
        f"#{props['id']}",
        f"{props['area']:.0f}",
        f"{props['circularity']:.2f}",
        f"{props['aspect_ratio']:.2f}",
        f"({props['centroid'][0]},{props['centroid'][1]})"
    ])

# Membuat tabel properti
if len(tabel_data) > 0:
    tabel = axes[1, 2].table(
        cellText=tabel_data,
        colLabels=['ID', 'Area', 'Circular.', 'Aspect R.', 'Centroid'],
        loc='center',
        cellLoc='center'
    )
    tabel.auto_set_font_size(False)
    tabel.set_fontsize(9)
    tabel.scale(1, 1.5)
    axes[1, 2].set_title("Properti per Instansi", fontsize=11, fontweight='bold')

# Menambahkan judul utama
plt.suptitle("Percobaan 14: Instance Segmentation - Kontur dan Properti",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi kontur
plt.savefig(os.path.join(OUTPUT_DIR, "14_instance_kontour.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/14_instance_kontour.png")

# ============================================================
# 7. Connected Components dengan statistik
# ============================================================
print("\n--- 7. Connected Components with Stats ---")

# Menerapkan connectedComponentsWithStats pada threshold
jumlah_label, labels, stats, centroids = cv2.connectedComponentsWithStats(
    thresh, connectivity=8
)

# Menampilkan informasi connected components
print(f"  Jumlah komponen (termasuk background): {jumlah_label}")
print(f"  Labels shape   : {labels.shape}")
print(f"  Stats shape    : {stats.shape}")
print(f"  Centroids shape: {centroids.shape}")

# Menampilkan properti setiap komponen (skip background = label 0)
print("\n  Properti Connected Components:")
for i in range(1, jumlah_label):
    # Mengambil statistik komponen
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    cx, cy = centroids[i]

    # Menampilkan properti
    print(f"    Komponen {i}: area={area}, bbox=({x},{y},{w},{h}), "
          f"centroid=({cx:.0f},{cy:.0f})")

# Membuat mask berwarna berdasarkan connected components
mask_cc = np.zeros(img_sintetis.shape, dtype=np.uint8)

# Menghasilkan warna untuk setiap komponen
np.random.seed(123)
for i in range(1, jumlah_label):
    # Menghasilkan warna random
    warna = (np.random.randint(50, 255),
             np.random.randint(50, 255),
             np.random.randint(50, 255))

    # Mewarnai piksel komponen i
    mask_cc[labels == i] = warna

# ============================================================
# 8. Visualisasi connected components
# ============================================================
print("\n--- 8. Visualisasi Connected Components ---")

# Membuat figure untuk connected components
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- Subplot 1: Gambar asli ---
axes[0, 0].imshow(cv2.cvtColor(img_sintetis, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# --- Subplot 2: Threshold ---
axes[0, 1].imshow(thresh, cmap='gray')
axes[0, 1].set_title("Threshold Biner", fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# --- Subplot 3: Label map ---
axes[0, 2].imshow(labels, cmap='nipy_spectral')
axes[0, 2].set_title(f"Label Map ({jumlah_label - 1} komponen)",
                      fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# --- Subplot 4: Mask connected components berwarna ---
axes[1, 0].imshow(cv2.cvtColor(mask_cc, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Connected Components (warna unik)",
                      fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# --- Subplot 5: CC dengan bounding box dan centroid ---
img_cc_props = img_sintetis.copy()

# Menggambar properti setiap komponen
for i in range(1, jumlah_label):
    # Mengambil properti
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    cx, cy = int(centroids[i][0]), int(centroids[i][1])

    # Menggambar bounding box
    cv2.rectangle(img_cc_props, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Menandai centroid
    cv2.circle(img_cc_props, (cx, cy), 5, (0, 0, 255), -1)

    # Menuliskan label
    cv2.putText(img_cc_props, f"CC{i}", (cx - 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Menampilkan CC dengan properti
axes[1, 1].imshow(cv2.cvtColor(img_cc_props, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("CC + Bounding Box + Centroid",
                      fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# --- Subplot 6: Overlay CC ---
overlay_cc = cv2.addWeighted(img_sintetis, 0.5, mask_cc, 0.5, 0)
axes[1, 2].imshow(cv2.cvtColor(overlay_cc, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Overlay Connected Components",
                      fontsize=11, fontweight='bold')
axes[1, 2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 14: Connected Components with Stats",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi connected components
plt.savefig(os.path.join(OUTPUT_DIR, "14_instance_komponen.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/14_instance_komponen.png")

# ============================================================
# 9. Perbandingan semantic vs instance segmentation
# ============================================================
print("\n--- 9. Perbandingan Semantic vs Instance ---")

# --- Semantic Segmentation ---
# Membuat mask semantic (semua objek kelas sama = warna sama)
mask_semantic = np.ones(img_sintetis.shape, dtype=np.uint8) * 200  # background

# Mendefinisikan warna per kelas untuk semantic
warna_sem_lingkaran = (0, 0, 255)   # Merah untuk semua lingkaran
warna_sem_persegi = (0, 255, 0)     # Hijau untuk semua persegi
warna_sem_segitiga = (255, 0, 0)    # Biru untuk semua segitiga

# Mewarnai objek berdasarkan kelas (menggunakan warna dari gambar asli)
# Deteksi lingkaran (objek merah di gambar asli)
mask_merah = cv2.inRange(img_sintetis, np.array([0, 0, 150]),
                          np.array([100, 100, 255]))
mask_merah = cv2.morphologyEx(mask_merah, cv2.MORPH_CLOSE, kernel, iterations=2)

# Deteksi persegi (objek hijau di gambar asli)
mask_hijau = cv2.inRange(img_sintetis, np.array([0, 130, 0]),
                          np.array([80, 255, 80]))
mask_hijau = cv2.morphologyEx(mask_hijau, cv2.MORPH_CLOSE, kernel, iterations=2)

# Deteksi segitiga (objek biru di gambar asli)
mask_biru = cv2.inRange(img_sintetis, np.array([150, 0, 0]),
                         np.array([255, 80, 80]))
mask_biru = cv2.morphologyEx(mask_biru, cv2.MORPH_CLOSE, kernel, iterations=2)

# Menerapkan warna semantic
mask_semantic[mask_merah > 0] = warna_sem_lingkaran
mask_semantic[mask_hijau > 0] = warna_sem_persegi
mask_semantic[mask_biru > 0] = warna_sem_segitiga

# --- Membuat figure perbandingan ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- Baris 1: Semantic Segmentation ---
axes[0, 0].imshow(cv2.cvtColor(img_sintetis, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli (8 objek, 3 kelas)",
                      fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Semantic mask
axes[0, 1].imshow(cv2.cvtColor(mask_semantic, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("SEMANTIC Segmentation\n(Kelas sama = warna sama)",
                      fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# Semantic overlay
overlay_sem = cv2.addWeighted(img_sintetis, 0.5, mask_semantic, 0.5, 0)
axes[0, 2].imshow(cv2.cvtColor(overlay_sem, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Overlay Semantic\n3 kelas: merah, hijau, biru",
                      fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# --- Baris 2: Instance Segmentation ---
axes[1, 0].imshow(cv2.cvtColor(img_sintetis, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Gambar Asli (sama)", fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# Instance mask
axes[1, 1].imshow(cv2.cvtColor(mask_instance, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"INSTANCE Segmentation\n({len(contours_valid)} instansi = "
                      f"{len(contours_valid)} warna unik)",
                      fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# Instance overlay
overlay_inst2 = cv2.addWeighted(img_sintetis, 0.5, mask_instance, 0.5, 0)

# Menambahkan label ID pada overlay
for idx, props in enumerate(properti_instansi):
    cx, cy = props['centroid']
    cv2.putText(overlay_inst2, f"#{props['id']}", (cx - 10, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

axes[1, 2].imshow(cv2.cvtColor(overlay_inst2, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Overlay Instance\nSetiap objek ID unik",
                      fontsize=11, fontweight='bold')
axes[1, 2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 14: Perbandingan Semantic vs Instance Segmentation",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi perbandingan
plt.savefig(os.path.join(OUTPUT_DIR, "14_semantic_vs_instance.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/14_semantic_vs_instance.png")

# ============================================================
# 10. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 14")
print("=" * 60)
print(f"""
Konsep yang telah dipelajari:
1. Instance segmentation membedakan setiap objek individual
2. Semantic segmentation mengelompokkan piksel per kelas saja
3. cv2.findContours() menemukan batas luar setiap objek
4. cv2.connectedComponentsWithStats() memberikan label + statistik
5. Properti per instansi: area, keliling, centroid, bounding box
6. Circularity (= 4*pi*area/keliling^2) mengukur kebulatan bentuk
7. Setiap instansi diberi warna unik untuk visualisasi
8. Connected components labeling adalah metode instance segmentation sederhana

Hasil Deteksi:
  Contour-based  : {len(contours_valid)} instansi terdeteksi
  Connected Comp. : {jumlah_label - 1} komponen terdeteksi

Output disimpan di folder: output/
- 14_instance_kontour.png      : Kontur dan properti instansi
- 14_instance_komponen.png     : Connected components
- 14_semantic_vs_instance.png  : Perbandingan semantic vs instance
""")
