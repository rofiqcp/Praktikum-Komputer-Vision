"""
==========================================================================
SCRIPT DOWNLOAD DAN GENERATE GAMBAR SAMPLE
Modul 06 - Recognition (Pengenalan)
==========================================================================
Script ini menyiapkan semua gambar yang dibutuhkan untuk 20 percobaan
Recognition/Pengenalan.
- Membuat folder 'image/' dan 'output/'
- Men-generate gambar wajah sintetis untuk face detection/recognition
- Men-generate gambar teks untuk OCR
- Men-generate gambar pedestrian, kendaraan, dan gesture

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

# Membuat folder 'image' dan 'output' jika belum ada
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Folder 'image/' dan 'output/' siap.")

# ============================================================
# LANGKAH 2: Generate gambar wajah sintetis
# ============================================================

def buat_wajah(nama, variasi=0, ukuran=300, warna_kulit=(180, 200, 220),
               ekspresi="netral", kacamata=False, topi=False):
    """Membuat gambar wajah sintetis dengan variasi."""
    # Membuat canvas
    img = np.ones((ukuran, ukuran, 3), dtype=np.uint8) * 230
    cx, cy = ukuran // 2, ukuran // 2

    # Offset acak berdasarkan variasi
    np.random.seed(variasi)
    ox = np.random.randint(-10, 10)
    oy = np.random.randint(-5, 5)

    # Kepala
    radius = ukuran // 3
    cv2.circle(img, (cx + ox, cy + oy), radius, warna_kulit, -1)

    # Rambut
    if topi:
        cv2.rectangle(img, (cx - radius - 10, cy - radius + oy - 20),
                       (cx + radius + 10, cy - radius + oy + 20), (50, 50, 150), -1)
        cv2.rectangle(img, (cx - radius + 20, cy - radius + oy + 10),
                       (cx + radius - 20, cy - radius + oy + 40), (50, 50, 150), -1)
    else:
        # Rambut sederhana
        cv2.ellipse(img, (cx + ox, cy + oy - 20), (radius + 5, radius - 10),
                    0, 180, 360, (40, 30, 20), -1)

    # Mata
    eye_y = cy - radius // 3 + oy
    eye_dist = radius // 3
    # Mata kiri
    cv2.circle(img, (cx - eye_dist + ox, eye_y), 12, (255, 255, 255), -1)
    cv2.circle(img, (cx - eye_dist + ox, eye_y), 6, (50, 50, 50), -1)
    # Mata kanan
    cv2.circle(img, (cx + eye_dist + ox, eye_y), 12, (255, 255, 255), -1)
    cv2.circle(img, (cx + eye_dist + ox, eye_y), 6, (50, 50, 50), -1)

    # Kacamata
    if kacamata:
        cv2.circle(img, (cx - eye_dist + ox, eye_y), 18, (30, 30, 30), 2)
        cv2.circle(img, (cx + eye_dist + ox, eye_y), 18, (30, 30, 30), 2)
        cv2.line(img, (cx - eye_dist + ox + 18, eye_y),
                 (cx + eye_dist + ox - 18, eye_y), (30, 30, 30), 2)

    # Hidung
    nose_y = cy + oy
    pts = np.array([[cx + ox, nose_y - 10], [cx - 8 + ox, nose_y + 10],
                    [cx + 8 + ox, nose_y + 10]], np.int32)
    cv2.polylines(img, [pts], True, (150, 130, 120), 2)

    # Mulut berdasarkan ekspresi
    mouth_y = cy + radius // 3 + oy
    if ekspresi == "senang":
        cv2.ellipse(img, (cx + ox, mouth_y), (25, 12), 0, 0, 180, (50, 50, 200), 2)
    elif ekspresi == "sedih":
        cv2.ellipse(img, (cx + ox, mouth_y + 15), (25, 12), 0, 180, 360, (50, 50, 200), 2)
    elif ekspresi == "terkejut":
        cv2.circle(img, (cx + ox, mouth_y + 5), 12, (50, 50, 200), 2)
    else:
        cv2.line(img, (cx - 20 + ox, mouth_y), (cx + 20 + ox, mouth_y), (50, 50, 200), 2)

    return img


def buat_gambar_grup():
    """Membuat gambar grup berisi beberapa wajah."""
    img = np.ones((400, 700, 3), dtype=np.uint8) * 220

    # Gambar background
    img[300:, :] = [150, 160, 140]

    posisi = [(120, 180), (280, 170), (420, 190), (560, 175)]
    warna_kulit = [(180, 200, 220), (160, 180, 200), (190, 210, 230), (170, 190, 210)]

    for i, (x, y) in enumerate(posisi):
        # Kepala
        cv2.circle(img, (x, y), 45, warna_kulit[i], -1)
        # Mata
        cv2.circle(img, (x - 15, y - 10), 5, (50, 50, 50), -1)
        cv2.circle(img, (x + 15, y - 10), 5, (50, 50, 50), -1)
        # Mulut
        cv2.ellipse(img, (x, y + 15), (12, 6), 0, 0, 180, (50, 50, 50), 2)
        # Badan
        cv2.rectangle(img, (x - 30, y + 45), (x + 30, y + 130),
                       [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50)][i], -1)

    return img


def buat_gambar_teks(tipe="printed"):
    """Membuat gambar teks untuk OCR testing."""
    if tipe == "printed":
        img = np.ones((500, 700, 3), dtype=np.uint8) * 255
        teks = [
            ("Universitas Indonesia", 50, 60, 1.2),
            ("Praktikum Komputer Vision", 50, 120, 1.0),
            ("Modul 06: Recognition", 50, 180, 1.0),
            ("Teknik Informatika 2024", 50, 240, 0.9),
            ("NIM: 12345678", 50, 300, 0.8),
            ("Nama: John Doe", 50, 350, 0.8),
            ("Python OpenCV Deep Learning", 50, 420, 0.8),
            ("1234567890 ABCDEFGHIJ", 50, 470, 0.8),
        ]
        for text, x, y, scale in teks:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)

    elif tipe == "scene":
        img = np.ones((400, 600, 3), dtype=np.uint8) * 180
        # Papan nama
        cv2.rectangle(img, (50, 50), (350, 130), (255, 255, 255), -1)
        cv2.rectangle(img, (50, 50), (350, 130), (0, 0, 0), 3)
        cv2.putText(img, "TOKO BUKU", (80, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 3)
        # Rambu
        cv2.rectangle(img, (380, 150), (550, 220), (50, 150, 50), -1)
        cv2.putText(img, "Jl. Merdeka", (390, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Plat nomor
        cv2.rectangle(img, (200, 280), (400, 340), (20, 20, 20), -1)
        cv2.putText(img, "B 1234 CD", (215, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    elif tipe == "noisy":
        img = np.ones((400, 600, 3), dtype=np.uint8) * 240
        cv2.putText(img, "Teks dengan Noise", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(img, "Komputer Vision", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(img, "Deep Learning 2024", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        # Tambah noise
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

    return img


def buat_gambar_tangan(gesture="open"):
    """Membuat gambar tangan sintetis."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200

    cx, cy = 200, 250

    if gesture == "open":
        # Telapak tangan
        cv2.ellipse(img, (cx, cy), (60, 70), 0, 0, 360, (180, 160, 140), -1)
        # 5 jari
        jari = [(cx - 50, cy - 80, 15), (cx - 25, cy - 110, 13),
                (cx, cy - 120, 14), (cx + 25, cy - 110, 13), (cx + 50, cy - 80, 12)]
        for x, y, r in jari:
            cv2.line(img, (x, cy - 40), (x, y), (180, 160, 140), r * 2)
            cv2.circle(img, (x, y), r, (180, 160, 140), -1)

    elif gesture == "fist":
        # Kepalan
        cv2.ellipse(img, (cx, cy - 20), (55, 65), 0, 0, 360, (180, 160, 140), -1)
        # Ibu jari di samping
        cv2.ellipse(img, (cx - 55, cy - 30), (15, 35), 20, 0, 360, (180, 160, 140), -1)

    elif gesture == "peace":
        # Telapak
        cv2.ellipse(img, (cx, cy), (50, 60), 0, 0, 360, (180, 160, 140), -1)
        # 2 jari (telunjuk dan tengah)
        cv2.line(img, (cx - 15, cy - 40), (cx - 25, cy - 130), (180, 160, 140), 22)
        cv2.circle(img, (cx - 25, cy - 130), 11, (180, 160, 140), -1)
        cv2.line(img, (cx + 15, cy - 40), (cx + 25, cy - 130), (180, 160, 140), 22)
        cv2.circle(img, (cx + 25, cy - 130), 11, (180, 160, 140), -1)

    elif gesture == "thumbsup":
        # Kepalan
        cv2.ellipse(img, (cx, cy), (50, 55), 0, 0, 360, (180, 160, 140), -1)
        # Ibu jari ke atas
        cv2.line(img, (cx - 40, cy - 20), (cx - 50, cy - 100), (180, 160, 140), 26)
        cv2.circle(img, (cx - 50, cy - 100), 14, (180, 160, 140), -1)

    elif gesture == "pointing":
        # Telapak
        cv2.ellipse(img, (cx, cy), (50, 55), 0, 0, 360, (180, 160, 140), -1)
        # Telunjuk ke atas
        cv2.line(img, (cx, cy - 40), (cx, cy - 140), (180, 160, 140), 22)
        cv2.circle(img, (cx, cy - 140), 12, (180, 160, 140), -1)

    return img


def buat_gambar_kendaraan():
    """Membuat gambar berisi kendaraan sintetis."""
    img = np.ones((480, 800, 3), dtype=np.uint8) * 200

    # Jalan
    img[300:, :] = [80, 80, 80]
    # Garis jalan
    for x in range(0, 800, 80):
        cv2.rectangle(img, (x, 385), (x + 40, 395), (255, 255, 255), -1)

    # Mobil 1 (merah)
    cv2.rectangle(img, (80, 310), (200, 370), (60, 60, 220), -1)
    cv2.rectangle(img, (100, 280), (180, 315), (60, 60, 220), -1)
    cv2.rectangle(img, (105, 285), (140, 310), (200, 200, 220), -1)
    cv2.rectangle(img, (145, 285), (175, 310), (200, 200, 220), -1)
    cv2.circle(img, (110, 375), 15, (40, 40, 40), -1)
    cv2.circle(img, (175, 375), 15, (40, 40, 40), -1)

    # Bis (biru)
    cv2.rectangle(img, (280, 260), (480, 370), (200, 100, 50), -1)
    for x in range(295, 460, 40):
        cv2.rectangle(img, (x, 275), (x + 25, 320), (220, 220, 240), -1)
    cv2.circle(img, (310, 375), 18, (40, 40, 40), -1)
    cv2.circle(img, (450, 375), 18, (40, 40, 40), -1)

    # Truk (hijau)
    cv2.rectangle(img, (560, 270), (750, 370), (50, 150, 50), -1)
    cv2.rectangle(img, (700, 290), (760, 370), (100, 100, 100), -1)
    cv2.rectangle(img, (710, 300), (755, 340), (200, 200, 220), -1)
    cv2.circle(img, (590, 375), 18, (40, 40, 40), -1)
    cv2.circle(img, (720, 375), 18, (40, 40, 40), -1)

    return img


# ============================================================
# LANGKAH 3: Membuat semua gambar yang diperlukan
# ============================================================

print("\n[INFO] Membuat gambar sintetis untuk percobaan recognition...")

# --- Gambar wajah individu (untuk face detection/recognition) ---
# Orang 1: Andi (3 ekspresi × 3 variasi = 9 foto)
face_dir = os.path.join(IMAGE_DIR, "faces")
for nama, warna, offset in [("andi", (180, 200, 220), 0),
                              ("budi", (160, 180, 200), 100),
                              ("citra", (190, 210, 230), 200)]:
    person_dir = os.path.join(face_dir, nama)
    os.makedirs(person_dir, exist_ok=True)
    idx = 1
    for ekspresi in ["netral", "senang", "sedih"]:
        for var in range(3):
            img = buat_wajah(nama, variasi=offset + var * 10 + idx,
                            warna_kulit=warna, ekspresi=ekspresi)
            cv2.imwrite(os.path.join(person_dir, f"{nama}_{idx:02d}.jpg"), img)
            idx += 1

print(f"  [OK] Dataset wajah: 3 orang × 9 foto = 27 foto")

# --- Gambar wajah tunggal dan grup ---
img_wajah_single = buat_wajah("test", ekspresi="senang")
cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_single.jpg"), img_wajah_single)

img_wajah_kacamata = buat_wajah("test", ekspresi="netral", kacamata=True)
cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_kacamata.jpg"), img_wajah_kacamata)

img_wajah_topi = buat_wajah("test", ekspresi="senang", topi=True)
cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_topi.jpg"), img_wajah_topi)

img_grup = buat_gambar_grup()
cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_grup.jpg"), img_grup)

print(f"  [OK] Gambar wajah tambahan: 4 gambar")

# --- Gambar teks untuk OCR ---
for tipe in ["printed", "scene", "noisy"]:
    img = buat_gambar_teks(tipe)
    cv2.imwrite(os.path.join(IMAGE_DIR, f"teks_{tipe}.jpg"), img)
print(f"  [OK] Gambar teks: 3 gambar")

# --- Gambar tangan untuk gesture recognition ---
for gesture in ["open", "fist", "peace", "thumbsup", "pointing"]:
    img = buat_gambar_tangan(gesture)
    cv2.imwrite(os.path.join(IMAGE_DIR, f"tangan_{gesture}.jpg"), img)
print(f"  [OK] Gambar tangan: 5 gambar")

# --- Gambar kendaraan ---
img_kendaraan = buat_gambar_kendaraan()
cv2.imwrite(os.path.join(IMAGE_DIR, "kendaraan.jpg"), img_kendaraan)
print(f"  [OK] Gambar kendaraan: 1 gambar")

# --- Gambar pedestrian ---
img_ped = np.ones((480, 640, 3), dtype=np.uint8) * 200
img_ped[350:, :] = [120, 120, 120]
# 4 orang dengan variasi
posisi_ped = [(100, 3), (250, 2), (400, 4), (530, 1)]
for px, skala in posisi_ped:
    h_ped = 60 * skala
    w_ped = 20 * skala
    y_base = 350
    # Kepala
    cv2.circle(img_ped, (px, y_base - int(h_ped) - 10), int(10 * skala / 2), (180, 160, 140), -1)
    # Badan
    cv2.rectangle(img_ped, (px - int(w_ped // 2), y_base - int(h_ped)),
                  (px + int(w_ped // 2), y_base - int(h_ped // 3)),
                  (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200)), -1)
    # Kaki
    cv2.rectangle(img_ped, (px - int(w_ped // 2), y_base - int(h_ped // 3)),
                  (px - 2, y_base), (50, 50, 100), -1)
    cv2.rectangle(img_ped, (px + 2, y_base - int(h_ped // 3)),
                  (px + int(w_ped // 2), y_base), (50, 50, 100), -1)

cv2.imwrite(os.path.join(IMAGE_DIR, "pedestrian.jpg"), img_ped)
print(f"  [OK] Gambar pedestrian: 1 gambar")

# --- Gambar untuk scene recognition ---
scenes = {
    "pantai": ([230, 180, 130], [180, 160, 100], [200, 180, 130]),
    "kota": ([230, 180, 130], [120, 120, 120], [160, 150, 140]),
    "hutan": ([100, 150, 80], [20, 100, 20], [60, 120, 40]),
}
for scene_name, (sky, ground, mid) in scenes.items():
    img = np.ones((480, 640, 3), dtype=np.uint8)
    img[:160, :] = sky
    img[160:320, :] = mid
    img[320:, :] = ground
    # Tambah detail
    if scene_name == "pantai":
        cv2.circle(img, (500, 80), 40, (0, 200, 255), -1)
    elif scene_name == "kota":
        for bx in range(100, 600, 120):
            h = np.random.randint(100, 250)
            cv2.rectangle(img, (bx, 320 - h), (bx + 80, 320), (140 + bx % 40, 130, 140), -1)
    elif scene_name == "hutan":
        for tx in range(50, 600, 100):
            cv2.rectangle(img, (tx, 200), (tx + 15, 400), (30, 80, 30), -1)
            cv2.circle(img, (tx + 7, 180), 40, (20, 120 + tx % 30, 20), -1)
    cv2.imwrite(os.path.join(IMAGE_DIR, f"scene_{scene_name}.jpg"), img)
print(f"  [OK] Gambar scene: 3 gambar")

# --- Gambar objek untuk klasifikasi ---
objek_list = {
    "kucing": (180, 180, 180),
    "anjing": (140, 120, 100),
    "mobil": (180, 50, 50),
    "bunga": (200, 100, 255),
    "gedung": (160, 150, 140),
}
for obj_name, color in objek_list.items():
    img = np.ones((300, 300, 3), dtype=np.uint8) * 230
    cv2.circle(img, (150, 150), 80, color, -1)
    cv2.putText(img, obj_name.upper(), (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(IMAGE_DIR, f"{obj_name}.jpg"), img)
print(f"  [OK] Gambar objek: 5 gambar")

# ============================================================
# LANGKAH 4: Download Haar Cascade dan model (opsional)
# ============================================================

print("\n[INFO] Haar cascade tersedia di OpenCV secara bawaan (cv2.data.haarcascades)")
print("[INFO] Tidak perlu download terpisah untuk Haar cascade.")

# ============================================================
# Selesai
# ============================================================

print("\n" + "=" * 60)
print("SEMUA PERSIAPAN SELESAI!")
print("=" * 60)
print(f"\nStruktur folder:")
print(f"  {IMAGE_DIR}/")
print(f"    ├── faces/ (3 orang × 9 foto)")
print(f"    ├── wajah_*.jpg (4 gambar)")
print(f"    ├── teks_*.jpg (3 gambar)")
print(f"    ├── tangan_*.jpg (5 gambar)")
print(f"    ├── kendaraan.jpg, pedestrian.jpg")
print(f"    ├── scene_*.jpg (3 gambar)")
print(f"    └── kucing.jpg, anjing.jpg, mobil.jpg, bunga.jpg, gedung.jpg")
print(f"  {OUTPUT_DIR}/")
print(f"\nSilakan lanjutkan ke percobaan 01-20!")
