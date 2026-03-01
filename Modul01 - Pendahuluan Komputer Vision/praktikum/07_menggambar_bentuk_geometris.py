"""
==========================================================================
PERCOBAAN 7: MENGGAMBAR BENTUK GEOMETRIS
==========================================================================
Program ini mempelajari cara menggambar berbagai bentuk geometris
pada gambar menggunakan fungsi drawing OpenCV.

Fungsi utama:
- cv2.line()       : Menggambar garis lurus
- cv2.rectangle()  : Menggambar persegi panjang
- cv2.circle()     : Menggambar lingkaran
- cv2.ellipse()    : Menggambar elips
- cv2.polylines()  : Menggambar polygon/garis bersambung
- cv2.fillPoly()   : Menggambar polygon terisi
- cv2.arrowedLine(): Menggambar garis berujung panah

Catatan: Semua fungsi drawing MEMODIFIKASI gambar langsung (in-place).
Gunakan .copy() jika ingin menyimpan gambar asli.
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 7: MENGGAMBAR BENTUK GEOMETRIS")
print("=" * 60)

# ============================================================
# 1. Membuat canvas kosong
# ============================================================

# Membuat gambar hitam (canvas) ukuran 600x800 piksel, 3 channel (BGR)
canvas = np.zeros((600, 800, 3), dtype=np.uint8)

# ============================================================
# 2. Menggambar garis (line)
# ============================================================

# cv2.line(img, titik_awal, titik_akhir, warna_BGR, ketebalan)
# titik = tuple (x, y) → PERHATIAN: di sini x dulu, bukan y!
cv2.line(canvas, (50, 50), (200, 50), (0, 255, 0), 2)
print("[INFO] Garis hijau horizontal digambar")

# cv2.LINE_AA = anti-aliased line (lebih halus, tidak bergerigi)
cv2.line(canvas, (50, 80), (200, 150), (0, 255, 255), 2, cv2.LINE_AA)
print("[INFO] Garis kuning diagonal dengan anti-aliasing")

# ============================================================
# 3. Menggambar garis panah (arrowed line)
# ============================================================

# cv2.arrowedLine(img, start, end, color, thickness, tipLength)
# tipLength = panjang ujung panah (fraksi dari panjang garis)
cv2.arrowedLine(canvas, (250, 50), (400, 50), (255, 255, 255), 2, tipLength=0.05)
cv2.arrowedLine(canvas, (250, 80), (400, 130), (0, 165, 255), 2, tipLength=0.1)
print("[INFO] Garis panah digambar")

# ============================================================
# 4. Menggambar persegi panjang (rectangle)
# ============================================================

# cv2.rectangle(img, titik_kiri_atas, titik_kanan_bawah, warna, ketebalan)
# ketebalan > 0 = outline saja
# ketebalan -1 = filled (terisi penuh)
cv2.rectangle(canvas, (50, 180), (200, 280), (0, 0, 255), 3)
print("[INFO] Persegi panjang merah (outline) digambar")

# Persegi terisi (filled)
cv2.rectangle(canvas, (220, 180), (370, 280), (255, 0, 128), -1)
print("[INFO] Persegi panjang ungu (filled) digambar")

# ============================================================
# 5. Menggambar lingkaran (circle)
# ============================================================

# cv2.circle(img, titik_pusat, radius, warna, ketebalan)
cv2.circle(canvas, (500, 100), 50, (255, 0, 0), 2)
print("[INFO] Lingkaran biru (outline) digambar")

# Lingkaran terisi
cv2.circle(canvas, (650, 100), 50, (0, 255, 128), -1)
print("[INFO] Lingkaran hijau (filled) digambar")

# Lingkaran konsentris (beberapa lingkaran dengan pusat sama)
for r in range(10, 60, 10):
    # Menghitung warna berdasarkan radius (gradient warna)
    warna = (r * 4, 255 - r * 4, 128)
    cv2.circle(canvas, (500, 230), r, warna, 2)
print("[INFO] Lingkaran konsentris digambar")

# ============================================================
# 6. Menggambar elips (ellipse)
# ============================================================

# cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
# center = pusat elips
# axes = (setengah_sumbu_horizontal, setengah_sumbu_vertikal)
# angle = sudut rotasi elips (derajat)
# startAngle, endAngle = sudut awal dan akhir (0-360 = penuh)
cv2.ellipse(canvas, (650, 230), (70, 40), 0, 0, 360, (255, 255, 0), 2)
print("[INFO] Elips kuning (penuh) digambar")

# Elips dengan rotasi 45 derajat
cv2.ellipse(canvas, (650, 230), (70, 40), 45, 0, 360, (0, 255, 255), 2)
print("[INFO] Elips kuning muda (rotasi 45°) digambar")

# Setengah elips (arc)
cv2.ellipse(canvas, (500, 350), (60, 30), 0, 0, 180, (255, 128, 0), 3)
print("[INFO] Setengah elips (arc 0-180°) digambar")

# ============================================================
# 7. Menggambar polygon (polylines)
# ============================================================

# Mendefinisikan titik-titik polygon sebagai array NumPy
# Segitiga
pts_segitiga = np.array([[100, 350], [50, 450], [150, 450]], np.int32)
# Reshape ke format yang dibutuhkan: (N, 1, 2)
pts_segitiga = pts_segitiga.reshape((-1, 1, 2))

# cv2.polylines(img, [pts], isClosed, color, thickness)
# isClosed = True → garis terakhir terhubung ke garis pertama
cv2.polylines(canvas, [pts_segitiga], True, (0, 255, 255), 2)
print("[INFO] Segitiga kuning (outline) digambar")

# Pentagon (segi-5) terisi
pts_pentagon = np.array([
    [280, 320], [320, 350], [310, 400], [250, 400], [240, 350]
], np.int32)

# cv2.fillPoly(img, [pts], color) → polygon terisi
cv2.fillPoly(canvas, [pts_pentagon], (128, 0, 255))
print("[INFO] Pentagon pink (filled) digambar")

# ============================================================
# 8. Menggambar bintang (polygon kompleks)
# ============================================================

# Menghitung titik-titik bintang 5 menggunakan trigonometri
pusat_x, pusat_y = 650, 400
radius_luar = 60
radius_dalam = 25
pts_bintang = []

for i in range(10):
    # Sudut untuk setiap titik (bergantian luar-dalam)
    sudut = np.radians(i * 36 - 90)  # Mulai dari atas
    if i % 2 == 0:
        # Titik luar
        x = int(pusat_x + radius_luar * np.cos(sudut))
        y = int(pusat_y + radius_luar * np.sin(sudut))
    else:
        # Titik dalam
        x = int(pusat_x + radius_dalam * np.cos(sudut))
        y = int(pusat_y + radius_dalam * np.sin(sudut))
    pts_bintang.append([x, y])

pts_bintang = np.array(pts_bintang, np.int32)
cv2.fillPoly(canvas, [pts_bintang], (0, 200, 255))
print("[INFO] Bintang orange (filled) digambar")

# ============================================================
# 9. Menggambar marker (titik penanda)
# ============================================================

# cv2.drawMarker(img, position, color, markerType, markerSize, thickness)
# Berbagai tipe marker yang tersedia
marker_types = [
    (cv2.MARKER_CROSS, "CROSS"),
    (cv2.MARKER_TILTED_CROSS, "TILTED"),
    (cv2.MARKER_STAR, "STAR"),
    (cv2.MARKER_DIAMOND, "DIAMOND"),
    (cv2.MARKER_SQUARE, "SQUARE"),
    (cv2.MARKER_TRIANGLE_UP, "TRI_UP"),
]

for i, (marker, nama) in enumerate(marker_types):
    x = 50 + i * 120
    y = 550
    cv2.drawMarker(canvas, (x, y), (255, 255, 255), marker, 30, 2)
    cv2.putText(canvas, nama, (x - 25, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

print("[INFO] 6 tipe marker digambar")

# ============================================================
# 10. Menyimpan dan menampilkan hasil
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 9))
ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
ax.set_title("Percobaan 7: Semua Bentuk Geometris OpenCV", fontsize=14, fontweight="bold")
ax.axis("off")

# Menambahkan label pada gambar
labels = [
    (125, 35, "line / arrowedLine"),
    (125, 165, "rectangle"),
    (570, 15, "circle"),
    (570, 165, "ellipse / arc"),
    (100, 305, "polygon / fillPoly"),
    (650, 305, "bintang"),
    (350, 510, "markers"),
]

for x, y, text in labels:
    ax.annotate(text, (x, y), fontsize=8, color="cyan",
                ha="center", fontweight="bold")

plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "07_bentuk_geometris_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 7")
print("=" * 60)
print("Fungsi drawing OpenCV:")
print("  1. cv2.line()        → Garis lurus")
print("  2. cv2.arrowedLine() → Garis panah")
print("  3. cv2.rectangle()   → Persegi panjang")
print("  4. cv2.circle()      → Lingkaran")
print("  5. cv2.ellipse()     → Elips / arc")
print("  6. cv2.polylines()   → Polygon outline")
print("  7. cv2.fillPoly()    → Polygon terisi")
print("  8. cv2.drawMarker()  → Titik penanda")
print("\nParameter penting:")
print("  - thickness=-1 → bentuk terisi (filled)")
print("  - cv2.LINE_AA  → anti-aliasing (lebih halus)")
print("  - Koordinat: (x, y) untuk fungsi drawing!")
print("=" * 60)
