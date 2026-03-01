"""
==========================================================================
DOWNLOAD IMAGE - MODUL 03: PEMROSESAN CITRA
==========================================================================
Script untuk generate gambar sintetis yang digunakan
pada 20 percobaan Modul 03 (Image Processing).

Gambar yang dihasilkan:
  1. kota.jpg        - Pemandangan kota sintetis (gedung + langit)
  2. dokumen.jpg     - Teks dan garis (untuk thresholding)
  3. wajah.jpg       - Wajah sintetis sederhana (lingkaran + fitur)
  4. noise_img.jpg   - Gambar dengan noise untuk denoising
  5. nature.jpg      - Pemandangan alam sintetis
  6. buah.jpg        - Objek berwarna (untuk segmentasi)
  7. teks_buram.jpg  - Teks buram (untuk sharpening)
  8. garis_tepi.jpg  - Pola geometris (untuk edge detection)
  9. biner_noise.png - Gambar biner dengan noise (untuk morfologi)
  10. spektrum.png   - Gambar dengan pola periodik (untuk Fourier)
==========================================================================
"""

import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MODUL 03 - DOWNLOAD / GENERATE IMAGE")
print("=" * 60)

# ============================================================
# 1. kota.jpg - Pemandangan kota sintetis
# ============================================================
img = np.zeros((512, 512, 3), dtype=np.uint8)
# Langit gradient
for y in range(256):
    b = int(180 + 75 * y / 256)
    g = int(100 + 60 * y / 256)
    img[y, :] = [b, g, 40]
# Tanah
img[256:, :] = [60, 80, 70]
# Gedung-gedung
buildings = [(50, 150, 120), (130, 100, 80), (200, 180, 100),
             (310, 130, 70), (400, 160, 90)]
for x, h, w in buildings:
    color = [np.random.randint(80, 180)] * 3
    cv2.rectangle(img, (x, 256 - h), (x + w, 256), tuple(int(c) for c in color), -1)
    # Jendela
    for wy in range(256 - h + 10, 250, 20):
        for wx in range(x + 8, x + w - 5, 15):
            cv2.rectangle(img, (wx, wy), (wx + 8, wy + 12), (0, 200, 255), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "kota.jpg"), img)
print("[OK] kota.jpg")

# ============================================================
# 2. dokumen.jpg - Teks dan garis
# ============================================================
doc = np.full((512, 512, 3), 240, dtype=np.uint8)
# Judul
cv2.putText(doc, "DOKUMEN CONTOH", (80, 60), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (30, 30, 30), 2)
cv2.line(doc, (40, 75), (470, 75), (100, 100, 100), 2)
# Paragraf
lines_text = [
    "Lorem ipsum dolor sit amet,",
    "consectetur adipiscing elit.",
    "Sed do eiusmod tempor incididunt",
    "ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis",
    "nostrud exercitation ullamco.",
]
for i, line in enumerate(lines_text):
    y_pos = 120 + i * 35
    cv2.putText(doc, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (50, 50, 50), 1)
# Tanda tangan
cv2.putText(doc, "Tanda tangan:", (50, 400), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (80, 80, 80), 1)
pts = np.array([[120, 430], [150, 410], [180, 440], [210, 420],
                [240, 435], [270, 415]], np.int32)
cv2.polylines(doc, [pts], False, (0, 0, 150), 2)
# Tambahkan noise ringan agar realistis
noise = np.random.normal(0, 5, doc.shape).astype(np.int16)
doc = np.clip(doc.astype(np.int16) + noise, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(IMAGE_DIR, "dokumen.jpg"), doc)
print("[OK] dokumen.jpg")

# ============================================================
# 3. wajah.jpg - Wajah sintetis
# ============================================================
face = np.full((512, 512, 3), 200, dtype=np.uint8)
# Kepala (oval)
cv2.ellipse(face, (256, 250), (120, 160), 0, 0, 360, (140, 180, 220), -1)
# Mata
cv2.ellipse(face, (210, 220), (25, 15), 0, 0, 360, (255, 255, 255), -1)
cv2.ellipse(face, (300, 220), (25, 15), 0, 0, 360, (255, 255, 255), -1)
cv2.circle(face, (210, 220), 10, (50, 40, 30), -1)
cv2.circle(face, (300, 220), 10, (50, 40, 30), -1)
# Hidung
cv2.line(face, (256, 240), (250, 280), (120, 150, 190), 2)
cv2.line(face, (250, 280), (265, 285), (120, 150, 190), 2)
# Mulut
cv2.ellipse(face, (256, 320), (40, 20), 0, 10, 170, (80, 80, 180), 2)
# Alis
cv2.line(face, (185, 195), (235, 190), (80, 60, 40), 3)
cv2.line(face, (275, 190), (325, 195), (80, 60, 40), 3)
# Rambut
cv2.ellipse(face, (256, 160), (130, 80), 0, 180, 360, (40, 30, 20), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "wajah.jpg"), face)
print("[OK] wajah.jpg")

# ============================================================
# 4. noise_img.jpg - Gambar dengan noise kuat
# ============================================================
clean = np.zeros((512, 512, 3), dtype=np.uint8)
# Gradasi warna
for y in range(512):
    for x in range(512):
        clean[y, x] = [int(x / 2), int(y / 2), int((x + y) / 4)]
# Tambah Gaussian noise kuat
noise_strong = np.random.normal(0, 40, clean.shape).astype(np.int16)
noisy = np.clip(clean.astype(np.int16) + noise_strong, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(IMAGE_DIR, "noise_img.jpg"), noisy)
print("[OK] noise_img.jpg")

# ============================================================
# 5. nature.jpg - Pemandangan alam
# ============================================================
nature = np.zeros((512, 512, 3), dtype=np.uint8)
# Langit
for y in range(300):
    r = int(135 + 120 * (1 - y / 300))
    g = int(180 + 75 * (1 - y / 300))
    b = int(255)
    nature[y, :] = [b, g, r]
# Gunung
pts_mount = np.array([[0, 300], [100, 180], [200, 250], [300, 160],
                       [400, 220], [512, 190], [512, 300]], np.int32)
cv2.fillPoly(nature, [pts_mount], (100, 120, 80))
# Rumput
nature[300:, :] = [40, 140, 60]
# Matahari
cv2.circle(nature, (400, 80), 40, (0, 200, 255), -1)
# Pohon
for tx in [80, 250, 420]:
    cv2.rectangle(nature, (tx - 5, 260), (tx + 5, 310), (30, 60, 80), -1)
    cv2.circle(nature, (tx, 240), 30, (20, 100, 30), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "nature.jpg"), nature)
print("[OK] nature.jpg")

# ============================================================
# 6. buah.jpg - Objek berwarna
# ============================================================
buah = np.full((512, 512, 3), 230, dtype=np.uint8)
# Apel merah
cv2.circle(buah, (130, 300), 70, (30, 30, 200), -1)
cv2.circle(buah, (130, 300), 70, (20, 20, 180), 3)
# Jeruk
cv2.circle(buah, (300, 280), 65, (0, 140, 255), -1)
# Pisang (elips kuning)
cv2.ellipse(buah, (430, 320), (50, 25), -30, 0, 360, (0, 220, 250), -1)
# Anggur hijau
for gx, gy in [(160, 150), (180, 140), (200, 150), (170, 165), (190, 160)]:
    cv2.circle(buah, (gx, gy), 15, (50, 180, 80), -1)
# Bayangan
cv2.ellipse(buah, (256, 430), (200, 30), 0, 0, 360, (180, 180, 180), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "buah.jpg"), buah)
print("[OK] buah.jpg")

# ============================================================
# 7. teks_buram.jpg - Teks buram
# ============================================================
teks = np.full((512, 512, 3), 245, dtype=np.uint8)
texts = ["Python", "OpenCV", "Computer", "Vision", "Image", "Processing"]
for i, t in enumerate(texts):
    y_t = 60 + i * 75
    cv2.putText(teks, t, (60, y_t), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, (30, 30, 30), 3)
# Blur untuk membuatnya buram
teks = cv2.GaussianBlur(teks, (9, 9), 3)
cv2.imwrite(os.path.join(IMAGE_DIR, "teks_buram.jpg"), teks)
print("[OK] teks_buram.jpg")

# ============================================================
# 8. garis_tepi.jpg - Pola geometris untuk edge detection
# ============================================================
edge_img = np.full((512, 512, 3), 240, dtype=np.uint8)
# Kotak bertingkat
for i in range(6):
    s = 40 + i * 35
    color = (30 + i * 30, 30 + i * 20, 200 - i * 25)
    cv2.rectangle(edge_img, (256 - s, 256 - s), (256 + s, 256 + s), color, 3)
# Garis diagonal
cv2.line(edge_img, (0, 0), (512, 512), (50, 50, 50), 2)
cv2.line(edge_img, (512, 0), (0, 512), (50, 50, 50), 2)
# Lingkaran
cv2.circle(edge_img, (256, 256), 200, (80, 80, 80), 2)
cv2.imwrite(os.path.join(IMAGE_DIR, "garis_tepi.jpg"), edge_img)
print("[OK] garis_tepi.jpg")

# ============================================================
# 9. biner_noise.png - Gambar biner dengan noise
# ============================================================
biner = np.zeros((512, 512), dtype=np.uint8)
# Objek-objek biner
cv2.rectangle(biner, (50, 50), (200, 200), 255, -1)
cv2.circle(biner, (350, 150), 80, 255, -1)
cv2.ellipse(biner, (150, 380), (100, 60), 0, 0, 360, 255, -1)
cv2.rectangle(biner, (320, 300), (480, 450), 255, -1)
# Tambah noise salt & pepper
sp_noise = np.random.random(biner.shape)
biner[sp_noise < 0.02] = 255  # salt
biner[sp_noise > 0.98] = 0    # pepper
# Tambah lubang kecil di objek
for _ in range(30):
    cx = np.random.randint(50, 480)
    cy = np.random.randint(50, 480)
    r = np.random.randint(2, 6)
    cv2.circle(biner, (cx, cy), r, 0 if biner[cy, cx] == 255 else 255, -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "biner_noise.png"), biner)
print("[OK] biner_noise.png")

# ============================================================
# 10. spektrum.png - Pola periodik untuk Fourier
# ============================================================
x = np.arange(512)
y = np.arange(512)
xx, yy = np.meshgrid(x, y)
# Gabungan beberapa frekuensi
f1 = np.sin(2 * np.pi * xx / 20)          # frekuensi rendah horizontal
f2 = np.sin(2 * np.pi * yy / 40)          # frekuensi rendah vertikal
f3 = np.sin(2 * np.pi * (xx + yy) / 15)   # diagonal
f4 = np.sin(2 * np.pi * xx / 8)           # frekuensi tinggi
combined = (f1 + f2 + f3 + 0.5 * f4) / 3.5
spektrum = ((combined + 1) / 2 * 255).astype(np.uint8)
cv2.imwrite(os.path.join(IMAGE_DIR, "spektrum.png"), spektrum)
print("[OK] spektrum.png")

print("\n" + "=" * 60)
print("Semua gambar berhasil di-generate!")
print(f"Lokasi: {IMAGE_DIR}")
print("=" * 60)
