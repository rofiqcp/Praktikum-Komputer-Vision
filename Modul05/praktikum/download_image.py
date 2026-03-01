"""
==========================================================================
SCRIPT DOWNLOAD DAN GENERATE GAMBAR SAMPLE
Modul 05 - Deep Learning untuk Komputer Vision
==========================================================================
Script ini menyiapkan semua gambar dan model yang dibutuhkan untuk
20 percobaan Deep Learning.
- Membuat folder 'image/' dan 'output/'
- Men-generate gambar sintetis untuk latihan klasifikasi/deteksi
- Mendownload model pre-trained (opsional)

Jalankan script ini PERTAMA KALI sebelum menjalankan percobaan lainnya.
==========================================================================
"""

# Mengimpor library os untuk operasi file dan folder
import os

# Mengimpor library numpy untuk operasi array/matriks
import numpy as np

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor urllib untuk mendownload file dari internet
import urllib.request

# ============================================================
# LANGKAH 1: Membuat struktur folder yang dibutuhkan
# ============================================================

# Mendapatkan path direktori tempat script ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder image dan output
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Membuat folder 'image' jika belum ada
os.makedirs(IMAGE_DIR, exist_ok=True)

# Membuat folder 'output' jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan pesan bahwa folder berhasil dibuat
print("[INFO] Folder 'image/' dan 'output/' siap.")

# ============================================================
# LANGKAH 2: Generate gambar sintetis untuk klasifikasi
# ============================================================

def buat_gambar_objek(nama, bentuk="lingkaran", warna=(0, 128, 255), ukuran=300):
    """Membuat gambar sintetis dengan bentuk geometris sebagai objek klasifikasi."""
    # Membuat canvas putih
    img = np.ones((ukuran, ukuran, 3), dtype=np.uint8) * 240
    center = ukuran // 2

    if bentuk == "lingkaran":
        # Menggambar lingkaran di tengah canvas
        cv2.circle(img, (center, center), ukuran // 3, warna, -1)
    elif bentuk == "persegi":
        # Menggambar persegi di tengah canvas
        offset = ukuran // 3
        cv2.rectangle(img, (center - offset, center - offset),
                      (center + offset, center + offset), warna, -1)
    elif bentuk == "segitiga":
        # Menggambar segitiga di tengah canvas
        offset = ukuran // 3
        pts = np.array([[center, center - offset],
                        [center - offset, center + offset],
                        [center + offset, center + offset]], np.int32)
        cv2.fillPoly(img, [pts], warna)
    elif bentuk == "bintang":
        # Menggambar bentuk bintang sederhana
        pts_outer = []
        pts_inner = []
        for i in range(5):
            angle_outer = np.radians(i * 72 - 90)
            angle_inner = np.radians(i * 72 - 90 + 36)
            pts_outer.append([int(center + ukuran // 3 * np.cos(angle_outer)),
                              int(center + ukuran // 3 * np.sin(angle_outer))])
            pts_inner.append([int(center + ukuran // 6 * np.cos(angle_inner)),
                              int(center + ukuran // 6 * np.sin(angle_inner))])
        pts = []
        for o, i in zip(pts_outer, pts_inner):
            pts.append(o)
            pts.append(i)
        cv2.fillPoly(img, [np.array(pts, np.int32)], warna)
    elif bentuk == "elips":
        # Menggambar elips
        cv2.ellipse(img, (center, center), (ukuran // 3, ukuran // 5), 0, 0, 360, warna, -1)

    return img


def buat_gambar_scene(nama_scene, ukuran=640):
    """Membuat gambar scene sintetis untuk deteksi objek."""
    # Membuat canvas dengan warna langit
    img = np.ones((480, ukuran, 3), dtype=np.uint8) * 200

    if nama_scene == "outdoor":
        # Langit biru di atas
        img[:240, :] = [230, 180, 130]
        # Rumput hijau di bawah
        img[240:, :] = [60, 160, 60]
        # Matahari
        cv2.circle(img, (500, 80), 50, (0, 200, 255), -1)
        # Pohon
        cv2.rectangle(img, (100, 180), (130, 300), (30, 80, 30), -1)
        cv2.circle(img, (115, 160), 60, (20, 130, 20), -1)
        # Rumah sederhana
        cv2.rectangle(img, (300, 200), (450, 340), (100, 120, 200), -1)
        cv2.rectangle(img, (350, 260), (400, 340), (80, 60, 40), -1)
        # Orang sederhana (stick figure)
        cv2.circle(img, (550, 230), 15, (100, 100, 200), -1)
        cv2.line(img, (550, 245), (550, 300), (100, 100, 200), 3)
        cv2.line(img, (550, 260), (530, 285), (100, 100, 200), 2)
        cv2.line(img, (550, 260), (570, 285), (100, 100, 200), 2)
        cv2.line(img, (550, 300), (535, 340), (100, 100, 200), 2)
        cv2.line(img, (550, 300), (565, 340), (100, 100, 200), 2)

    elif nama_scene == "indoor":
        # Dinding dan lantai
        img[:, :] = [220, 215, 200]
        img[300:, :] = [180, 170, 150]
        # Meja
        cv2.rectangle(img, (100, 250), (500, 280), (60, 100, 140), -1)
        cv2.rectangle(img, (120, 280), (140, 400), (60, 100, 140), -1)
        cv2.rectangle(img, (460, 280), (480, 400), (60, 100, 140), -1)
        # Buku di meja
        cv2.rectangle(img, (200, 230), (280, 250), (200, 50, 50), -1)
        # Laptop di meja
        cv2.rectangle(img, (320, 210), (420, 250), (80, 80, 80), -1)
        cv2.rectangle(img, (325, 215), (415, 245), (200, 200, 200), -1)

    elif nama_scene == "traffic":
        # Jalan
        img[:, :] = [200, 200, 200]
        img[200:400, :] = [80, 80, 80]
        # Garis jalan
        for x in range(0, ukuran, 80):
            cv2.rectangle(img, (x, 295), (x + 40, 305), (255, 255, 255), -1)
        # Mobil 1
        cv2.rectangle(img, (100, 220), (200, 270), (200, 50, 50), -1)
        cv2.rectangle(img, (120, 200), (180, 225), (200, 50, 50), -1)
        cv2.circle(img, (120, 275), 12, (40, 40, 40), -1)
        cv2.circle(img, (180, 275), 12, (40, 40, 40), -1)
        # Mobil 2
        cv2.rectangle(img, (350, 320), (470, 380), (50, 50, 200), -1)
        cv2.rectangle(img, (370, 295), (450, 325), (50, 50, 200), -1)
        cv2.circle(img, (375, 385), 12, (40, 40, 40), -1)
        cv2.circle(img, (445, 385), 12, (40, 40, 40), -1)

    return img


def buat_gambar_wajah_sintetis(ekspresi="netral"):
    """Membuat gambar wajah sintetis sederhana."""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 230

    # Kepala (lingkaran)
    cv2.circle(img, (150, 150), 100, (180, 200, 220), -1)

    # Mata
    cv2.circle(img, (115, 120), 15, (255, 255, 255), -1)
    cv2.circle(img, (185, 120), 15, (255, 255, 255), -1)
    cv2.circle(img, (115, 120), 7, (50, 50, 50), -1)
    cv2.circle(img, (185, 120), 7, (50, 50, 50), -1)

    # Hidung
    pts = np.array([[150, 140], [142, 165], [158, 165]], np.int32)
    cv2.polylines(img, [pts], True, (150, 130, 120), 2)

    if ekspresi == "senang":
        # Mulut tersenyum
        cv2.ellipse(img, (150, 190), (35, 20), 0, 0, 180, (50, 50, 50), 2)
    elif ekspresi == "sedih":
        # Mulut sedih
        cv2.ellipse(img, (150, 210), (35, 20), 0, 180, 360, (50, 50, 50), 2)
    else:
        # Mulut netral
        cv2.line(img, (120, 195), (180, 195), (50, 50, 50), 2)

    return img


def buat_gambar_teks():
    """Membuat gambar berisi teks untuk OCR testing."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Menulis beberapa baris teks
    cv2.putText(img, "Hello World!", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(img, "Computer Vision 2024", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Deep Learning OpenCV", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Python Programming", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "1234567890", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "ABCDEFGHIJ", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    return img


def buat_gambar_pedestrian():
    """Membuat gambar sintetis berisi orang berdiri."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # Latar belakang: jalan
    img[350:, :] = [120, 120, 120]

    # Orang 1
    # Kepala
    cv2.circle(img, (150, 200), 20, (180, 160, 140), -1)
    # Badan
    cv2.rectangle(img, (130, 220), (170, 310), (200, 50, 50), -1)
    # Kaki
    cv2.rectangle(img, (130, 310), (145, 370), (50, 50, 150), -1)
    cv2.rectangle(img, (155, 310), (170, 370), (50, 50, 150), -1)

    # Orang 2
    cv2.circle(img, (350, 210), 18, (180, 160, 140), -1)
    cv2.rectangle(img, (332, 228), (368, 310), (50, 150, 50), -1)
    cv2.rectangle(img, (332, 310), (347, 365), (50, 50, 120), -1)
    cv2.rectangle(img, (353, 310), (368, 365), (50, 50, 120), -1)

    # Orang 3 (lebih jauh)
    cv2.circle(img, (500, 240), 12, (180, 160, 140), -1)
    cv2.rectangle(img, (490, 252), (510, 310), (100, 100, 200), -1)
    cv2.rectangle(img, (490, 310), (498, 345), (50, 50, 80), -1)
    cv2.rectangle(img, (502, 310), (510, 345), (50, 50, 80), -1)

    return img


# ============================================================
# LANGKAH 3: Membuat semua gambar yang diperlukan
# ============================================================

print("\n[INFO] Membuat gambar sintetis untuk percobaan deep learning...")

# --- Gambar untuk klasifikasi ---
# Membuat 5 kategori dengan masing-masing beberapa variasi
kategori = {
    "lingkaran": ("lingkaran", [(0, 128, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (128, 0, 255)]),
    "persegi": ("persegi", [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50), (200, 50, 200)]),
    "segitiga": ("segitiga", [(100, 200, 100), (200, 100, 100), (100, 100, 200), (200, 200, 100), (100, 200, 200)]),
    "bintang": ("bintang", [(0, 200, 200), (200, 0, 200), (200, 200, 0), (100, 200, 200), (200, 100, 200)]),
    "elips": ("elips", [(150, 100, 50), (50, 150, 100), (100, 50, 150), (150, 150, 50), (50, 100, 150)])
}

# Membuat folder dataset per kategori
dataset_dir = os.path.join(IMAGE_DIR, "dataset")
for nama_kat in kategori:
    kat_dir = os.path.join(dataset_dir, nama_kat)
    os.makedirs(kat_dir, exist_ok=True)

    bentuk, warna_list = kategori[nama_kat]
    for i, warna in enumerate(warna_list):
        # Membuat gambar dengan sedikit variasi ukuran
        for j, size in enumerate([200, 250, 300]):
            img = buat_gambar_objek(nama_kat, bentuk, warna, size)
            # Resize ke ukuran standar
            img = cv2.resize(img, (224, 224))
            path = os.path.join(kat_dir, f"{nama_kat}_{i * 3 + j + 1:03d}.jpg")
            cv2.imwrite(path, img)

print(f"  [OK] Dataset klasifikasi: 5 kategori × 15 gambar = 75 gambar")

# --- Gambar scene untuk deteksi objek ---
scenes = ["outdoor", "indoor", "traffic"]
for scene in scenes:
    img = buat_gambar_scene(scene)
    cv2.imwrite(os.path.join(IMAGE_DIR, f"scene_{scene}.jpg"), img)
print(f"  [OK] Gambar scene: {len(scenes)} gambar")

# --- Gambar wajah sintetis ---
for ekspresi in ["netral", "senang", "sedih"]:
    img = buat_gambar_wajah_sintetis(ekspresi)
    cv2.imwrite(os.path.join(IMAGE_DIR, f"wajah_{ekspresi}.jpg"), img)
print(f"  [OK] Gambar wajah: 3 gambar")

# --- Gambar teks untuk OCR ---
img_teks = buat_gambar_teks()
cv2.imwrite(os.path.join(IMAGE_DIR, "teks_sample.jpg"), img_teks)
print(f"  [OK] Gambar teks: 1 gambar")

# --- Gambar pedestrian ---
img_ped = buat_gambar_pedestrian()
cv2.imwrite(os.path.join(IMAGE_DIR, "pedestrian.jpg"), img_ped)
print(f"  [OK] Gambar pedestrian: 1 gambar")

# --- Gambar umum untuk klasifikasi DNN ---
# Gambar kucing sintetis
img_kucing = np.ones((480, 640, 3), dtype=np.uint8) * 240
cv2.circle(img_kucing, (320, 240), 150, (180, 180, 180), -1)
cv2.circle(img_kucing, (270, 200), 25, (50, 50, 50), -1)
cv2.circle(img_kucing, (370, 200), 25, (50, 50, 50), -1)
pts_hidung = np.array([[320, 250], [310, 270], [330, 270]], np.int32)
cv2.fillPoly(img_kucing, [pts_hidung], (180, 130, 200))
cv2.imwrite(os.path.join(IMAGE_DIR, "kucing.jpg"), img_kucing)

# Gambar anjing sintetis
img_anjing = np.ones((480, 640, 3), dtype=np.uint8) * 220
cv2.ellipse(img_anjing, (320, 260), (130, 100), 0, 0, 360, (140, 120, 100), -1)
cv2.circle(img_anjing, (280, 220), 20, (50, 50, 50), -1)
cv2.circle(img_anjing, (360, 220), 20, (50, 50, 50), -1)
cv2.ellipse(img_anjing, (320, 280), (30, 15), 0, 0, 360, (30, 30, 30), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "anjing.jpg"), img_anjing)

# Gambar mobil sintetis
img_mobil = np.ones((480, 640, 3), dtype=np.uint8) * 200
cv2.rectangle(img_mobil, (150, 250), (490, 350), (180, 50, 50), -1)
cv2.rectangle(img_mobil, (220, 180), (420, 260), (180, 50, 50), -1)
cv2.circle(img_mobil, (220, 360), 30, (40, 40, 40), -1)
cv2.circle(img_mobil, (420, 360), 30, (40, 40, 40), -1)
cv2.rectangle(img_mobil, (240, 200), (310, 250), (200, 200, 220), -1)
cv2.rectangle(img_mobil, (330, 200), (400, 250), (200, 200, 220), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "mobil.jpg"), img_mobil)

# Gambar bunga sintetis
img_bunga = np.ones((480, 640, 3), dtype=np.uint8) * 200
for angle in range(0, 360, 45):
    x = int(320 + 80 * np.cos(np.radians(angle)))
    y = int(240 + 80 * np.sin(np.radians(angle)))
    cv2.circle(img_bunga, (x, y), 40, (200, 100, 255), -1)
cv2.circle(img_bunga, (320, 240), 35, (0, 200, 255), -1)
cv2.rectangle(img_bunga, (315, 320), (325, 480), (0, 150, 0), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "bunga.jpg"), img_bunga)

# Gambar gedung sintetis
img_gedung = np.ones((480, 640, 3), dtype=np.uint8)
img_gedung[:240, :] = [230, 180, 130]  # Langit
img_gedung[240:, :] = [120, 120, 120]  # Jalan
cv2.rectangle(img_gedung, (200, 50), (440, 300), (160, 150, 140), -1)
for y in range(80, 280, 40):
    for x in range(220, 420, 50):
        cv2.rectangle(img_gedung, (x, y), (x + 30, y + 25), (200, 200, 220), -1)
cv2.rectangle(img_gedung, (290, 230), (350, 300), (60, 60, 80), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "gedung.jpg"), img_gedung)

print(f"  [OK] Gambar umum: 5 gambar (kucing, anjing, mobil, bunga, gedung)")

# --- Gambar untuk augmentasi ---
# Gambar dengan detail yang terlihat jelas saat di-augmentasi
img_aug = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.rectangle(img_aug, (50, 50), (250, 250), (0, 255, 0), 3)
cv2.circle(img_aug, (150, 150), 60, (255, 0, 0), -1)
cv2.putText(img_aug, "AUG", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.imwrite(os.path.join(IMAGE_DIR, "augmentasi_sample.jpg"), img_aug)
print(f"  [OK] Gambar augmentasi: 1 gambar")

# --- Gambar segmentasi ---
# Gambar dengan area warna berbeda jelas untuk segmentasi
img_seg = np.zeros((480, 640, 3), dtype=np.uint8)
img_seg[:160, :] = [230, 180, 130]  # Biru langit
img_seg[160:320, :] = [60, 160, 60]  # Hijau taman
img_seg[320:, :] = [80, 80, 80]  # Abu-abu jalan
cv2.rectangle(img_seg, (200, 80), (440, 280), (140, 130, 200), -1)  # Bangunan
cv2.circle(img_seg, (100, 120), 40, (0, 200, 255), -1)  # Matahari
cv2.imwrite(os.path.join(IMAGE_DIR, "segmentasi_sample.jpg"), img_seg)
print(f"  [OK] Gambar segmentasi: 1 gambar")

# ============================================================
# LANGKAH 4: Download model pre-trained (opsional)
# ============================================================

def download_file(url, filepath):
    """Mendownload file dari URL ke path tertentu."""
    if os.path.exists(filepath):
        print(f"  [SKIP] {os.path.basename(filepath)} sudah ada.")
        return True
    try:
        print(f"  [DOWNLOAD] {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  [OK] {os.path.basename(filepath)} berhasil didownload.")
        return True
    except Exception as e:
        print(f"  [WARNING] Gagal download {os.path.basename(filepath)}: {e}")
        print(f"  [INFO] Anda bisa download manual dari: {url}")
        return False

# Folder untuk model
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n[INFO] Mencoba download model pre-trained (opsional)...")
print("[INFO] Jika gagal, program percobaan tetap bisa jalan dengan alternatif.\n")

# ImageNet class labels
download_file(
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/classification_classes_ILSVRC2012.txt",
    os.path.join(MODEL_DIR, "classification_classes_ILSVRC2012.txt")
)

print("\n" + "=" * 60)
print("SEMUA PERSIAPAN SELESAI!")
print("=" * 60)
print(f"\nStruktur folder:")
print(f"  {IMAGE_DIR}/")
print(f"    ├── dataset/ (5 kategori × 15 gambar)")
print(f"    ├── scene_*.jpg (3 gambar)")
print(f"    ├── wajah_*.jpg (3 gambar)")
print(f"    ├── teks_sample.jpg")
print(f"    ├── pedestrian.jpg")
print(f"    ├── kucing.jpg, anjing.jpg, mobil.jpg, bunga.jpg, gedung.jpg")
print(f"    ├── augmentasi_sample.jpg")
print(f"    └── segmentasi_sample.jpg")
print(f"  {OUTPUT_DIR}/")
print(f"  {MODEL_DIR}/")
print(f"\nSilakan lanjutkan ke percobaan 01-20!")
