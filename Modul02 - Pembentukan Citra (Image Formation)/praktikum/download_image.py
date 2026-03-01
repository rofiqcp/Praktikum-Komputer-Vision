"""
==========================================================================
DOWNLOAD IMAGE - MODUL 02: PEMBENTUKAN CITRA
==========================================================================
Script ini membuat gambar sintetis dan menyiapkan folder untuk
praktikum Modul 02 (Pembentukan Citra / Image Formation).

Gambar yang dihasilkan:
  1. gedung.jpg       - Garis-garis perspektif (untuk transformasi)
  2. checkerboard.png - Papan catur (untuk kalibrasi kamera)
  3. kotak_warna.png  - Kotak warna (untuk transformasi warna)
  4. dokumen.jpg      - Simulasi dokumen miring (perspektif)
  5. gradient_radial.png - Gradien radial (untuk distorsi)
  6. grid.png         - Grid garis (untuk transformasi geometris)
  7. objek_3d.png     - Kubus 3D sederhana (untuk proyeksi)
  8. bintang.png      - Pola bintang (untuk rotasi)
  9. lingkaran.png    - Lingkaran konsentris (untuk polar)
  10. teks_miring.jpg - Teks miring (untuk koreksi perspektif)
==========================================================================
"""

import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder yang diperlukan
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DOWNLOAD IMAGE - MODUL 02: PEMBENTUKAN CITRA")
print("=" * 60)

# ----------------------------------------------------------
# 1. Gedung (garis perspektif)
# ----------------------------------------------------------
img = np.ones((400, 400, 3), dtype=np.uint8) * 200
# Langit gradien biru
for y in range(150):
    ratio = y / 150
    img[y, :] = [230 - int(80 * ratio), 200 - int(60 * ratio), 150 - int(50 * ratio)]
# Gedung-gedung
cv2.rectangle(img, (50, 100), (120, 300), (100, 100, 120), -1)
cv2.rectangle(img, (50, 100), (120, 300), (60, 60, 80), 2)
cv2.rectangle(img, (140, 150), (200, 300), (110, 110, 130), -1)
cv2.rectangle(img, (140, 150), (200, 300), (60, 60, 80), 2)
cv2.rectangle(img, (220, 80), (300, 300), (90, 90, 110), -1)
cv2.rectangle(img, (220, 80), (300, 300), (60, 60, 80), 2)
cv2.rectangle(img, (310, 130), (370, 300), (105, 105, 125), -1)
# Jendela
for bx, by1, by2, bw in [(50, 110, 290, 120), (220, 90, 290, 300)]:
    for wy in range(by1, by2, 25):
        for wx in range(bx + 8, bw - 5, 18):
            cv2.rectangle(img, (wx, wy), (wx + 10, wy + 15), (180, 200, 220), -1)
# Jalan
cv2.rectangle(img, (0, 300), (400, 400), (80, 80, 80), -1)
cv2.line(img, (0, 350), (400, 350), (0, 200, 200), 2)
cv2.imwrite(os.path.join(IMAGE_DIR, "gedung.jpg"), img)
print("[OK] gedung.jpg")

# ----------------------------------------------------------
# 2. Checkerboard (papan catur 9×6 untuk kalibrasi)
# ----------------------------------------------------------
cell = 40
rows, cols = 8, 11
cb = np.zeros((rows * cell, cols * cell), dtype=np.uint8)
for r in range(rows):
    for c in range(cols):
        if (r + c) % 2 == 0:
            cb[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = 255
cb_bgr = cv2.cvtColor(cb, cv2.COLOR_GRAY2BGR)
# Tambahkan border putih
cb_bgr = cv2.copyMakeBorder(cb_bgr, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
cv2.imwrite(os.path.join(IMAGE_DIR, "checkerboard.png"), cb_bgr)
print("[OK] checkerboard.png")

# ----------------------------------------------------------
# 3. Kotak warna
# ----------------------------------------------------------
img = np.zeros((300, 300, 3), dtype=np.uint8)
warna = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255),(255,255,0),(128,0,128),(0,128,128),(128,128,0)]
for i, w in enumerate(warna):
    r, c = divmod(i, 3)
    cv2.rectangle(img, (c*100, r*100), ((c+1)*100, (r+1)*100), w, -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "kotak_warna.png"), img)
print("[OK] kotak_warna.png")

# ----------------------------------------------------------
# 4. Dokumen miring (simulasi foto dokumen dari sudut)
# ----------------------------------------------------------
doc = np.ones((300, 250, 3), dtype=np.uint8) * 245
cv2.putText(doc, "DOKUMEN", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
cv2.putText(doc, "Praktikum CV", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1)
for y in range(120, 280, 20):
    cv2.line(doc, (20, y), (230, y), (180, 180, 180), 1)
cv2.rectangle(doc, (5, 5), (245, 295), (0, 0, 0), 2)
# Transformasi perspektif untuk membuat miring
src_pts = np.float32([[0, 0], [250, 0], [250, 300], [0, 300]])
dst_pts = np.float32([[30, 20], [220, 0], [250, 300], [0, 280]])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
doc_miring = cv2.warpPerspective(doc, M, (300, 300), borderValue=(200, 200, 200))
cv2.imwrite(os.path.join(IMAGE_DIR, "dokumen.jpg"), doc_miring)
print("[OK] dokumen.jpg")

# ----------------------------------------------------------
# 5. Gradient radial
# ----------------------------------------------------------
img = np.zeros((400, 400), dtype=np.uint8)
cx, cy = 200, 200
for y in range(400):
    for x in range(400):
        d = np.sqrt((x - cx)**2 + (y - cy)**2)
        img[y, x] = min(255, int(d * 255 / 280))
cv2.imwrite(os.path.join(IMAGE_DIR, "gradient_radial.png"), img)
print("[OK] gradient_radial.png")

# ----------------------------------------------------------
# 6. Grid (garis-garis)
# ----------------------------------------------------------
img = np.ones((400, 400, 3), dtype=np.uint8) * 255
for i in range(0, 401, 25):
    cv2.line(img, (i, 0), (i, 400), (200, 200, 200), 1)
    cv2.line(img, (0, i), (400, i), (200, 200, 200), 1)
for i in range(0, 401, 100):
    cv2.line(img, (i, 0), (i, 400), (100, 100, 100), 2)
    cv2.line(img, (0, i), (400, i), (100, 100, 100), 2)
# Titik di setiap persimpangan grid utama
for y in range(0, 401, 100):
    for x in range(0, 401, 100):
        cv2.circle(img, (x, y), 4, (0, 0, 200), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "grid.png"), img)
print("[OK] grid.png")

# ----------------------------------------------------------
# 7. Objek 3D (kubus wireframe)
# ----------------------------------------------------------
img = np.ones((400, 400, 3), dtype=np.uint8) * 240
# Kubus depan
pts_front = [(120,120),(280,120),(280,280),(120,280)]
pts_back = [(160,80),(320,80),(320,240),(160,240)]
for i in range(4):
    cv2.line(img, pts_front[i], pts_front[(i+1)%4], (200,0,0), 2)
    cv2.line(img, pts_back[i], pts_back[(i+1)%4], (0,0,200), 2)
    cv2.line(img, pts_front[i], pts_back[i], (0,150,0), 2)
cv2.putText(img, "3D Cube", (130, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
cv2.imwrite(os.path.join(IMAGE_DIR, "objek_3d.png"), img)
print("[OK] objek_3d.png")

# ----------------------------------------------------------
# 8. Bintang (pola simetris untuk rotasi)
# ----------------------------------------------------------
img = np.ones((400, 400, 3), dtype=np.uint8) * 240
cx, cy = 200, 200
for i in range(12):
    angle = np.deg2rad(i * 30)
    x2 = int(cx + 180 * np.cos(angle))
    y2 = int(cy + 180 * np.sin(angle))
    warna_line = (int(i * 20), int(255 - i * 20), 128)
    cv2.line(img, (cx, cy), (x2, y2), warna_line, 3)
    cv2.circle(img, (x2, y2), 8, warna_line, -1)
cv2.circle(img, (cx, cy), 15, (0, 0, 200), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "bintang.png"), img)
print("[OK] bintang.png")

# ----------------------------------------------------------
# 9. Lingkaran konsentris (untuk koordinat polar)
# ----------------------------------------------------------
img = np.zeros((400, 400, 3), dtype=np.uint8)
cx, cy = 200, 200
for r in range(20, 200, 20):
    hue = int((r / 200) * 180)
    warna_c = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    cv2.circle(img, (cx, cy), r, warna_c.tolist(), 2)
# Garis radial
for a in range(0, 360, 30):
    rad = np.deg2rad(a)
    x2 = int(cx + 190 * np.cos(rad))
    y2 = int(cy + 190 * np.sin(rad))
    cv2.line(img, (cx, cy), (x2, y2), (100, 100, 100), 1)
cv2.imwrite(os.path.join(IMAGE_DIR, "lingkaran.png"), img)
print("[OK] lingkaran.png")

# ----------------------------------------------------------
# 10. Teks miring (untuk koreksi perspektif)
# ----------------------------------------------------------
img = np.ones((300, 400, 3), dtype=np.uint8) * 250
cv2.putText(img, "KOMPUTER", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
cv2.putText(img, "VISION", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 180), 3)
cv2.putText(img, "Modul 02", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
cv2.rectangle(img, (10, 10), (390, 290), (0, 0, 0), 3)
# Buat miring
M = cv2.getRotationMatrix2D((200, 150), 15, 1.0)
img = cv2.warpAffine(img, M, (400, 300), borderValue=(220, 220, 220))
cv2.imwrite(os.path.join(IMAGE_DIR, "teks_miring.jpg"), img)
print("[OK] teks_miring.jpg")

print(f"\n[SELESAI] 10 gambar disimpan di: {IMAGE_DIR}")
print(f"[INFO] Folder output: {OUTPUT_DIR}")
print("=" * 60)
