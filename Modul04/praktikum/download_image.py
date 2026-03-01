"""
==========================================================================
DOWNLOAD GAMBAR UNTUK MODUL 04: MODEL FITTING DAN OPTIMASI
==========================================================================
Script ini membuat gambar sintetis untuk semua percobaan Modul 04.
Gambar mencakup: pola garis, lingkaran, noise, scene dua view, dll.

Jalankan script ini PERTAMA KALI sebelum menjalankan percobaan.
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
print("DOWNLOAD/GENERATE GAMBAR MODUL 04")
print("=" * 60)

# ------------------------------------------------------------------
# 1. garis_noise.png — titik-titik di sekitar garis lurus + outlier
# ------------------------------------------------------------------
img1 = np.ones((500, 500, 3), dtype=np.uint8) * 255
np.random.seed(42)
# Inlier: titik di sekitar garis y = 0.7x + 50
for _ in range(150):
    x = np.random.randint(50, 450)
    y = int(0.7 * x + 50 + np.random.randn() * 10)
    cv2.circle(img1, (x, y), 3, (0, 0, 200), -1)
# Outlier
for _ in range(40):
    x = np.random.randint(50, 450)
    y = np.random.randint(50, 450)
    cv2.circle(img1, (x, y), 3, (200, 0, 0), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "garis_noise.png"), img1)
print("[OK] garis_noise.png")

# ------------------------------------------------------------------
# 2. lingkaran_noise.png — titik-titik di sekitar lingkaran + outlier
# ------------------------------------------------------------------
img2 = np.ones((500, 500, 3), dtype=np.uint8) * 255
center_x, center_y, radius = 250, 250, 150
for _ in range(200):
    theta = np.random.uniform(0, 2 * np.pi)
    r = radius + np.random.randn() * 8
    x = int(center_x + r * np.cos(theta))
    y = int(center_y + r * np.sin(theta))
    cv2.circle(img2, (x, y), 3, (0, 150, 0), -1)
for _ in range(50):
    x = np.random.randint(30, 470)
    y = np.random.randint(30, 470)
    cv2.circle(img2, (x, y), 3, (0, 0, 200), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "lingkaran_noise.png"), img2)
print("[OK] lingkaran_noise.png")

# ------------------------------------------------------------------
# 3. jalan.png — gambar kota dengan garis-garis jalan
# ------------------------------------------------------------------
img3 = np.zeros((500, 700, 3), dtype=np.uint8)
# Langit
img3[:200] = (180, 140, 100)
# Jalan
cv2.rectangle(img3, (0, 200), (700, 500), (80, 80, 80), -1)
# Garis marka jalan
cv2.line(img3, (150, 500), (300, 250), (0, 255, 255), 3)
cv2.line(img3, (550, 500), (400, 250), (0, 255, 255), 3)
cv2.line(img3, (350, 500), (350, 280), (255, 255, 255), 2, cv2.LINE_AA)
# Gedung-gedung
for bx in range(50, 650, 100):
    h = np.random.randint(50, 180)
    cv2.rectangle(img3, (bx, 200 - h), (bx + 60, 200), (60, 60, 80), -1)
    cv2.rectangle(img3, (bx, 200 - h), (bx + 60, 200), (100, 100, 120), 1)
cv2.imwrite(os.path.join(IMAGE_DIR, "jalan.png"), img3)
print("[OK] jalan.png")

# ------------------------------------------------------------------
# 4. koin.png — lingkaran-lingkaran untuk Hough Circle
# ------------------------------------------------------------------
img4 = np.ones((500, 500, 3), dtype=np.uint8) * 240
coins = [(120, 120, 50), (300, 150, 70), (200, 350, 60),
         (420, 300, 45), (380, 120, 55), (100, 400, 40)]
for cx, cy, r in coins:
    color = tuple(np.random.randint(30, 180, 3).tolist())
    cv2.circle(img4, (cx, cy), r, color, -1)
    cv2.circle(img4, (cx, cy), r, (0, 0, 0), 2)
cv2.imwrite(os.path.join(IMAGE_DIR, "koin.png"), img4)
print("[OK] koin.png")

# ------------------------------------------------------------------
# 5. papan_catur.png — checkerboard untuk homography
# ------------------------------------------------------------------
img5 = np.zeros((480, 640, 3), dtype=np.uint8)
sq = 60
for i in range(8):
    for j in range(8):
        if (i + j) % 2 == 0:
            cv2.rectangle(img5, (j * sq + 80, i * sq),
                          ((j + 1) * sq + 80, (i + 1) * sq), (255, 255, 255), -1)
        else:
            cv2.rectangle(img5, (j * sq + 80, i * sq),
                          ((j + 1) * sq + 80, (i + 1) * sq), (0, 0, 0), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "papan_catur.png"), img5)
print("[OK] papan_catur.png")

# ------------------------------------------------------------------
# 6. scene_a.png & scene_b.png — dua view untuk homography/matching
# ------------------------------------------------------------------
base = np.zeros((400, 600, 3), dtype=np.uint8)
# Background gradient
for y in range(400):
    base[y] = (int(50 + y * 0.3), int(100 + y * 0.2), int(150 - y * 0.2))
# Objek-objek
cv2.rectangle(base, (100, 80), (250, 200), (0, 200, 200), -1)
cv2.circle(base, (400, 150), 60, (200, 100, 50), -1)
cv2.drawContours(base, [np.array([[300, 300], [350, 250], [400, 300]])], 0, (50, 200, 50), -1)
cv2.putText(base, "VIEW", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
scene_a = base.copy()

# Scene B: rotasi + translasi kecil
M = cv2.getRotationMatrix2D((300, 200), 10, 0.95)
M[0, 2] += 30
M[1, 2] += 15
scene_b = cv2.warpAffine(base, M, (600, 400), borderValue=(0, 0, 0))

cv2.imwrite(os.path.join(IMAGE_DIR, "scene_a.png"), scene_a)
cv2.imwrite(os.path.join(IMAGE_DIR, "scene_b.png"), scene_b)
print("[OK] scene_a.png & scene_b.png")

# ------------------------------------------------------------------
# 7. frame1.png & frame2.png — dua frame untuk optical flow
# ------------------------------------------------------------------
frame1 = np.zeros((400, 600, 3), dtype=np.uint8)
frame1[:] = (40, 40, 40)
# Objek bergerak: beberapa lingkaran
objs = [(100, 200, 30), (300, 150, 40), (450, 300, 25), (200, 350, 35)]
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
for (cx, cy, r), color in zip(objs, colors):
    cv2.circle(frame1, (cx, cy), r, color, -1)
# Latar belakang: grid kecil
for x in range(0, 600, 40):
    cv2.line(frame1, (x, 0), (x, 400), (60, 60, 60), 1)
for y in range(0, 400, 40):
    cv2.line(frame1, (0, y), (600, y), (60, 60, 60), 1)

frame2 = np.zeros_like(frame1)
frame2[:] = (40, 40, 40)
for x in range(0, 600, 40):
    cv2.line(frame2, (x, 0), (x, 400), (60, 60, 60), 1)
for y in range(0, 400, 40):
    cv2.line(frame2, (0, y), (600, y), (60, 60, 60), 1)
# Objek bergerak ke kanan dan sedikit turun
shifts = [(20, 5), (15, 10), (-10, 15), (25, -5)]
for (cx, cy, r), color, (dx, dy) in zip(objs, colors, shifts):
    cv2.circle(frame2, (cx + dx, cy + dy), r, color, -1)

cv2.imwrite(os.path.join(IMAGE_DIR, "frame1.png"), frame1)
cv2.imwrite(os.path.join(IMAGE_DIR, "frame2.png"), frame2)
print("[OK] frame1.png & frame2.png")

# ------------------------------------------------------------------
# 8. noisy_img.png — gambar dengan noise untuk denoising MRF
# ------------------------------------------------------------------
clean = np.zeros((300, 300), dtype=np.uint8)
# Buat pola bersih: 4 region
clean[:150, :150] = 50
clean[:150, 150:] = 100
clean[150:, :150] = 150
clean[150:, 150:] = 200
# Tambah noise Gaussian
noise = np.random.randn(300, 300) * 30
noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(IMAGE_DIR, "noisy_img.png"), noisy)
cv2.imwrite(os.path.join(IMAGE_DIR, "clean_img.png"), clean)
print("[OK] noisy_img.png & clean_img.png")

# ------------------------------------------------------------------
# 9. template.png & target.png — untuk template matching
# ------------------------------------------------------------------
target = np.ones((400, 600, 3), dtype=np.uint8) * 200
# Beberapa bentuk
cv2.rectangle(target, (50, 50), (150, 150), (0, 0, 180), -1)
cv2.circle(target, (350, 100), 40, (0, 180, 0), -1)
cv2.rectangle(target, (250, 250), (350, 350), (0, 0, 180), -1)  # sama dgn template
cv2.circle(target, (500, 300), 50, (180, 0, 0), -1)
# Template: crop dari target
template = target[250:350, 250:350].copy()
cv2.imwrite(os.path.join(IMAGE_DIR, "target.png"), target)
cv2.imwrite(os.path.join(IMAGE_DIR, "template.png"), template)
print("[OK] target.png & template.png")

# ------------------------------------------------------------------
# 10. bentuk.png — berbagai bentuk untuk contour fitting
# ------------------------------------------------------------------
img10 = np.zeros((500, 500, 3), dtype=np.uint8)
# Ellips
cv2.ellipse(img10, (150, 150), (80, 50), 30, 0, 360, (255, 255, 255), -1)
# Persegi panjang miring
pts_rect = np.array([[300, 80], [420, 120], [400, 220], [280, 180]])
cv2.fillPoly(img10, [pts_rect], (255, 255, 255))
# Bintang
angles = np.linspace(0, 2 * np.pi, 11)[:-1]
pts_star = []
for i, a in enumerate(angles):
    r = 70 if i % 2 == 0 else 35
    pts_star.append([int(150 + r * np.cos(a)), int(400 + r * np.sin(a))])
cv2.fillPoly(img10, [np.array(pts_star)], (255, 255, 255))
# Lingkaran
cv2.circle(img10, (400, 380), 60, (255, 255, 255), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "bentuk.png"), img10)
print("[OK] bentuk.png")

print("\n" + "=" * 60)
print(f"Semua gambar disimpan di: {IMAGE_DIR}")
print(f"Folder output siap di: {OUTPUT_DIR}")
print("=" * 60)
