"""
==========================================================================
SCRIPT DOWNLOAD DAN GENERATE GAMBAR SAMPLE
Modul 01 - Pendahuluan Komputer Vision
==========================================================================
Script ini menyiapkan semua gambar yang dibutuhkan untuk 20 percobaan.
- Membuat folder 'image/' dan 'output/'
- Men-generate gambar sintetis untuk latihan
- Mendownload gambar sample dari internet (opsional)

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
# LANGKAH 2: Generate gambar sintetis
# ============================================================

def buat_gambar_kucing_sintetis():
    """Membuat gambar sintetis sederhana sebagai pengganti foto asli."""
    # Membuat canvas putih ukuran 640x480 dengan 3 channel warna (BGR)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240

    # Menggambar lingkaran besar sebagai kepala (warna abu-abu)
    cv2.circle(img, (320, 240), 150, (180, 180, 180), -1)

    # Menggambar dua lingkaran kecil sebagai mata (warna hitam)
    cv2.circle(img, (270, 200), 25, (50, 50, 50), -1)
    cv2.circle(img, (370, 200), 25, (50, 50, 50), -1)

    # Menggambar pupil mata (warna putih)
    cv2.circle(img, (275, 195), 8, (255, 255, 255), -1)
    cv2.circle(img, (375, 195), 8, (255, 255, 255), -1)

    # Menggambar segitiga sebagai hidung (warna pink)
    pts_hidung = np.array([[320, 250], [310, 270], [330, 270]], np.int32)
    cv2.fillPoly(img, [pts_hidung], (180, 130, 200))

    # Menggambar garis sebagai mulut
    cv2.line(img, (320, 270), (320, 290), (100, 100, 100), 2)
    cv2.line(img, (320, 290), (300, 300), (100, 100, 100), 2)
    cv2.line(img, (320, 290), (340, 300), (100, 100, 100), 2)

    # Menggambar kumis (whiskers)
    cv2.line(img, (230, 260), (290, 270), (100, 100, 100), 1)
    cv2.line(img, (230, 280), (290, 280), (100, 100, 100), 1)
    cv2.line(img, (350, 270), (410, 260), (100, 100, 100), 1)
    cv2.line(img, (350, 280), (410, 280), (100, 100, 100), 1)

    # Menggambar telinga (segitiga)
    pts_telinga_kiri = np.array([[200, 130], [250, 50], [300, 130]], np.int32)
    pts_telinga_kanan = np.array([[340, 130], [390, 50], [440, 130]], np.int32)
    cv2.fillPoly(img, [pts_telinga_kiri], (180, 180, 180))
    cv2.fillPoly(img, [pts_telinga_kanan], (180, 180, 180))

    # Menambahkan teks label
    cv2.putText(img, "Sample Image - Kucing", (180, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    return img


def buat_gambar_gradient():
    """Membuat gambar gradient horizontal dari hitam ke putih."""
    # Membuat array 1D dari 0 sampai 255 (gradient horizontal)
    gradient = np.linspace(0, 255, 640, dtype=np.uint8)

    # Mengulang baris gradient ke 480 baris untuk membuat gambar 2D
    img_gray = np.tile(gradient, (480, 1))

    # Mengkonversi grayscale ke BGR agar bisa disimpan sebagai gambar berwarna
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    return img_bgr


def buat_gambar_warna_warni():
    """Membuat gambar dengan blok-blok warna untuk latihan konversi warna."""
    # Membuat canvas hitam ukuran 480x640
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mengisi blok warna: Merah (BGR: 0,0,255)
    img[0:160, 0:213] = (0, 0, 255)

    # Mengisi blok warna: Hijau (BGR: 0,255,0)
    img[0:160, 213:426] = (0, 255, 0)

    # Mengisi blok warna: Biru (BGR: 255,0,0)
    img[0:160, 426:640] = (255, 0, 0)

    # Mengisi blok warna: Kuning (BGR: 0,255,255)
    img[160:320, 0:213] = (0, 255, 255)

    # Mengisi blok warna: Cyan (BGR: 255,255,0)
    img[160:320, 213:426] = (255, 255, 0)

    # Mengisi blok warna: Magenta (BGR: 255,0,255)
    img[160:320, 426:640] = (255, 0, 255)

    # Mengisi blok warna: Putih
    img[320:480, 0:213] = (255, 255, 255)

    # Mengisi blok warna: Abu-abu
    img[320:480, 213:426] = (128, 128, 128)

    # Mengisi blok warna: Orange (BGR: 0,165,255)
    img[320:480, 426:640] = (0, 165, 255)

    return img


def buat_gambar_pemandangan():
    """Membuat gambar pemandangan sintetis sederhana."""
    # Membuat canvas langit biru
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Langit: gradient dari biru tua ke biru muda
    for y in range(300):
        # Menghitung intensitas biru berdasarkan posisi y
        ratio = y / 300.0
        b = int(255 - ratio * 50)
        g = int(180 - ratio * 80)
        r = int(100 - ratio * 50)
        img[y, :] = (max(b, 0), max(g, 0), max(r, 0))

    # Tanah: hijau
    img[300:480, :] = (50, 150, 50)

    # Matahari: lingkaran kuning
    cv2.circle(img, (500, 80), 50, (0, 255, 255), -1)

    # Gunung: segitiga abu-abu
    pts_gunung = np.array([[100, 300], [250, 100], [400, 300]], np.int32)
    cv2.fillPoly(img, [pts_gunung], (150, 150, 150))

    # Puncak gunung bersalju
    pts_salju = np.array([[220, 140], [250, 100], [280, 140]], np.int32)
    cv2.fillPoly(img, [pts_salju], (255, 255, 255))

    # Pohon
    cv2.rectangle(img, (500, 280), (520, 350), (30, 80, 30), -1)
    pts_pohon = np.array([[460, 280], [510, 180], [560, 280]], np.int32)
    cv2.fillPoly(img, [pts_pohon], (30, 120, 30))

    return img


def buat_gambar_teks():
    """Membuat gambar dengan berbagai ukuran dan gaya teks."""
    # Membuat canvas putih
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Menuliskan contoh berbagai font OpenCV
    cv2.putText(img, "HERSHEY_SIMPLEX", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "HERSHEY_PLAIN", (20, 100),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 200), 2)
    cv2.putText(img, "HERSHEY_DUPLEX", (20, 150),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
    cv2.putText(img, "HERSHEY_COMPLEX", (20, 200),
                cv2.FONT_HERSHEY_COMPLEX, 1, (200, 0, 0), 2)
    cv2.putText(img, "HERSHEY_TRIPLEX", (20, 250),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 0, 128), 2)
    cv2.putText(img, "HERSHEY_SCRIPT_SIMPLEX", (20, 300),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 128, 128), 2)
    cv2.putText(img, "HERSHEY_SCRIPT_COMPLEX", (20, 350),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (128, 128, 0), 2)
    cv2.putText(img, "Italic Mode", (20, 420),
                cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC, 1.2, (0, 0, 0), 2)

    return img


def buat_checkerboard():
    """Membuat pola checkerboard untuk kalibrasi dan testing."""
    # Ukuran setiap kotak dalam piksel
    ukuran_kotak = 60

    # Jumlah kotak: 8x8
    baris, kolom = 8, 10

    # Membuat gambar checkerboard
    img = np.zeros((baris * ukuran_kotak, kolom * ukuran_kotak), dtype=np.uint8)

    # Mengisi kotak putih pada posisi genap
    for i in range(baris):
        for j in range(kolom):
            # Jika jumlah i+j genap, isi putih
            if (i + j) % 2 == 0:
                y1 = i * ukuran_kotak
                y2 = (i + 1) * ukuran_kotak
                x1 = j * ukuran_kotak
                x2 = (j + 1) * ukuran_kotak
                img[y1:y2, x1:x2] = 255

    # Mengkonversi ke BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img_bgr


def buat_gambar_bentuk():
    """Membuat gambar berisi berbagai bentuk geometris."""
    # Membuat canvas hitam
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Menggambar persegi panjang merah
    cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 255), 3)

    # Menggambar lingkaran hijau
    cv2.circle(img, (350, 100), 70, (0, 255, 0), -1)

    # Menggambar elips biru
    cv2.ellipse(img, (550, 100), (70, 40), 30, 0, 360, (255, 0, 0), 2)

    # Menggambar garis kuning
    cv2.line(img, (50, 250), (600, 250), (0, 255, 255), 3)

    # Menggambar segitiga magenta
    pts = np.array([[150, 300], [50, 450], [250, 450]], np.int32)
    cv2.polylines(img, [pts], True, (255, 0, 255), 3)

    # Menggambar bintang cyan
    pts_star = np.array([
        [450, 300], [470, 370], [540, 370], [480, 410],
        [500, 470], [450, 430], [400, 470], [420, 410],
        [360, 370], [430, 370]
    ], np.int32)
    cv2.polylines(img, [pts_star], True, (255, 255, 0), 2)

    return img


# ============================================================
# LANGKAH 3: Menyimpan semua gambar yang di-generate
# ============================================================

# Menampilkan pesan proses pembuatan gambar
print("[INFO] Membuat gambar sintetis...")

# Menyimpan gambar kucing sintetis
path_kucing = os.path.join(IMAGE_DIR, "kucing.jpg")
cv2.imwrite(path_kucing, buat_gambar_kucing_sintetis())
print(f"  ✓ {path_kucing}")

# Menyimpan gambar gradient
path_gradient = os.path.join(IMAGE_DIR, "gradient.jpg")
cv2.imwrite(path_gradient, buat_gambar_gradient())
print(f"  ✓ {path_gradient}")

# Menyimpan gambar warna-warni
path_warna = os.path.join(IMAGE_DIR, "warna_warni.png")
cv2.imwrite(path_warna, buat_gambar_warna_warni())
print(f"  ✓ {path_warna}")

# Menyimpan gambar pemandangan
path_pemandangan = os.path.join(IMAGE_DIR, "pemandangan.jpg")
cv2.imwrite(path_pemandangan, buat_gambar_pemandangan())
print(f"  ✓ {path_pemandangan}")

# Menyimpan gambar teks
path_teks = os.path.join(IMAGE_DIR, "contoh_teks.png")
cv2.imwrite(path_teks, buat_gambar_teks())
print(f"  ✓ {path_teks}")

# Menyimpan checkerboard
path_checker = os.path.join(IMAGE_DIR, "checkerboard.png")
cv2.imwrite(path_checker, buat_checkerboard())
print(f"  ✓ {path_checker}")

# Menyimpan gambar bentuk geometris
path_bentuk = os.path.join(IMAGE_DIR, "bentuk_geometris.png")
cv2.imwrite(path_bentuk, buat_gambar_bentuk())
print(f"  ✓ {path_bentuk}")

# Membuat gambar kedua untuk blending (pemandangan kedua)
img_blend2 = np.zeros((480, 640, 3), dtype=np.uint8)
# Latar belakang ungu gelap
img_blend2[:] = (80, 30, 60)
# Lingkaran-lingkaran acak
np.random.seed(42)
for _ in range(30):
    x = np.random.randint(0, 640)
    y = np.random.randint(0, 480)
    r = np.random.randint(10, 60)
    color = tuple(int(c) for c in np.random.randint(100, 255, 3))
    cv2.circle(img_blend2, (x, y), r, color, -1)

path_abstrak = os.path.join(IMAGE_DIR, "abstrak.jpg")
cv2.imwrite(path_abstrak, img_blend2)
print(f"  ✓ {path_abstrak}")

# Membuat gambar logo/mask sederhana
mask_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(mask_img, (320, 240), 150, (255, 255, 255), -1)
cv2.putText(mask_img, "CV", (250, 270), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
path_logo = os.path.join(IMAGE_DIR, "logo_mask.png")
cv2.imwrite(path_logo, mask_img)
print(f"  ✓ {path_logo}")

# Membuat gambar noise untuk testing
img_noise = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
path_noise = os.path.join(IMAGE_DIR, "noise.png")
cv2.imwrite(path_noise, img_noise)
print(f"  ✓ {path_noise}")

# ============================================================
# LANGKAH 4: Download gambar dari internet (opsional)
# ============================================================

# Daftar URL gambar gratis dari Wikimedia Commons
URLS = {
    "lena_color.png": "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
    "cameraman.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Washington_Monument_Dusk_Jan_2006.jpg/320px-Washington_Monument_Dusk_Jan_2006.jpg",
}

print("\n[INFO] Mencoba download gambar dari internet (opsional)...")

# Mengiterasi setiap URL untuk didownload
for nama_file, url in URLS.items():
    path_file = os.path.join(IMAGE_DIR, nama_file)
    # Cek apakah file sudah ada, skip jika sudah
    if os.path.exists(path_file):
        print(f"  ⏭ {nama_file} sudah ada, skip.")
        continue
    try:
        # Mendownload file dari URL menggunakan urllib
        urllib.request.urlretrieve(url, path_file)
        print(f"  ✓ Downloaded: {nama_file}")
    except Exception as e:
        # Menampilkan pesan error jika gagal download
        print(f"  ✗ Gagal download {nama_file}: {e}")
        print(f"    → Tidak masalah, gunakan gambar sintetis yang sudah dibuat.")

# ============================================================
# LANGKAH 5: Verifikasi
# ============================================================

print("\n[INFO] Daftar gambar yang tersedia di folder 'image/':")
# Mendapatkan daftar semua file di folder image
daftar_file = sorted(os.listdir(IMAGE_DIR))

# Menampilkan setiap file beserta ukurannya
for f in daftar_file:
    path = os.path.join(IMAGE_DIR, f)
    # Mendapatkan ukuran file dalam KB
    ukuran_kb = os.path.getsize(path) / 1024
    print(f"  📁 {f} ({ukuran_kb:.1f} KB)")

# Menampilkan total jumlah file
print(f"\n[INFO] Total: {len(daftar_file)} file gambar.")
print("[INFO] Folder 'output/' siap untuk menyimpan hasil percobaan.")
print("\n✅ Setup selesai! Silakan jalankan percobaan 01-20.")
