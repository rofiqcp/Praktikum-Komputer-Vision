"""
==========================================================================
PERCOBAAN 19: MASKING GAMBAR
==========================================================================
Program ini mempelajari cara membuat dan menerapkan mask pada gambar.
Mask adalah gambar biner (hitam-putih) yang menentukan area mana
yang akan diproses atau ditampilkan.

Fungsi utama:
- cv2.bitwise_and(src1, src2, mask=mask) → Terapkan mask
- cv2.threshold() → Buat mask dari threshold
- cv2.inRange() → Buat mask dari rentang warna
- np.zeros/ones → Buat mask manual
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 19: MASKING GAMBAR")
print("=" * 60)

# Membaca gambar
img = cv2.imread(os.path.join(IMAGE_DIR, "foto_orang2.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

img = cv2.resize(img, (300, 300))
h, w = img.shape[:2]
print(f"[INFO] Gambar: {w}×{h}")

# ============================================================
# 1. Mask persegi panjang
# ============================================================
print("\n--- 1. Mask Persegi Panjang ---")

# Membuat mask hitam (semua piksel 0)
mask_rect = np.zeros((h, w), dtype=np.uint8)
# Menggambar persegi putih di tengah (area yang akan ditampilkan)
# cv2.rectangle dengan ketebalan -1 = diisi penuh
cv2.rectangle(mask_rect, (50, 50), (250, 250), 255, -1)

# cv2.bitwise_and menerapkan mask: piksel putih = tampil, hitam = hilang
img_rect = cv2.bitwise_and(img, img, mask=mask_rect)
print(f"  Mask persegi: piksel putih = {cv2.countNonZero(mask_rect)}")

# ============================================================
# 2. Mask lingkaran
# ============================================================
print("\n--- 2. Mask Lingkaran ---")

# Membuat mask lingkaran
mask_circle = np.zeros((h, w), dtype=np.uint8)
# cv2.circle dengan -1 = lingkaran diisi penuh
cv2.circle(mask_circle, (w // 2, h // 2), 120, 255, -1)

# Terapkan mask lingkaran
img_circle = cv2.bitwise_and(img, img, mask=mask_circle)
print(f"  Mask lingkaran: piksel putih = {cv2.countNonZero(mask_circle)}")

# ============================================================
# 3. Mask elips
# ============================================================
print("\n--- 3. Mask Elips ---")

mask_elips = np.zeros((h, w), dtype=np.uint8)
# cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
cv2.ellipse(mask_elips, (w // 2, h // 2), (130, 90), 0, 0, 360, 255, -1)

img_elips = cv2.bitwise_and(img, img, mask=mask_elips)
print(f"  Mask elips: piksel putih = {cv2.countNonZero(mask_elips)}")

# ============================================================
# 4. Mask dari threshold
# ============================================================
print("\n--- 4. Mask dari Threshold ---")

# Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.threshold membuat mask biner berdasarkan ambang batas
# Piksel > 127 → putih (255), sisanya → hitam (0)
_, mask_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Terapkan mask: hanya area terang yang tampil
img_thresh = cv2.bitwise_and(img, img, mask=mask_thresh)
print(f"  Threshold > 127: piksel putih = {cv2.countNonZero(mask_thresh)}")

# ============================================================
# 5. Mask dari rentang warna (inRange)
# ============================================================
print("\n--- 5. Mask Warna (inRange) ---")

# Konversi ke HSV untuk deteksi warna yang lebih akurat
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Deteksi warna biru (Hue 100-130 dalam OpenCV HSV)
# cv2.inRange membuat mask: piksel dalam rentang → putih, lainnya → hitam
batas_bawah = np.array([100, 50, 50])
batas_atas = np.array([130, 255, 255])
mask_biru = cv2.inRange(img_hsv, batas_bawah, batas_atas)

# Deteksi warna merah (Hue 0-10 dan 170-180)
mask_merah1 = cv2.inRange(img_hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
mask_merah2 = cv2.inRange(img_hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
# Gabungkan dua mask merah menggunakan bitwise_or
mask_merah = cv2.bitwise_or(mask_merah1, mask_merah2)

# Terapkan mask warna
img_biru = cv2.bitwise_and(img, img, mask=mask_biru)
img_merah = cv2.bitwise_and(img, img, mask=mask_merah)

print(f"  Warna biru: {cv2.countNonZero(mask_biru)} piksel")
print(f"  Warna merah: {cv2.countNonZero(mask_merah)} piksel")

# ============================================================
# 6. Kombinasi mask (AND, OR, NOT)
# ============================================================
print("\n--- 6. Operasi Logika pada Mask ---")

# NOT: inversi mask (area yang tadinya putih jadi hitam)
mask_not = cv2.bitwise_not(mask_circle)

# AND: irisan dua mask (area putih yang sama-sama ada di kedua mask)
mask_and = cv2.bitwise_and(mask_rect, mask_circle)

# OR: gabungan dua mask (area putih di salah satu mask)
mask_or = cv2.bitwise_or(mask_rect, mask_circle)

# XOR: perbedaan dua mask (area putih di salah satu tapi bukan keduanya)
mask_xor = cv2.bitwise_xor(mask_rect, mask_circle)

# Terapkan mask kombinasi
img_and = cv2.bitwise_and(img, img, mask=mask_and)
img_or = cv2.bitwise_and(img, img, mask=mask_or)

print(f"  NOT:  {cv2.countNonZero(mask_not)} piksel")
print(f"  AND:  {cv2.countNonZero(mask_and)} piksel")
print(f"  OR:   {cv2.countNonZero(mask_or)} piksel")
print(f"  XOR:  {cv2.countNonZero(mask_xor)} piksel")

# ============================================================
# 7. Mask polygon (bentuk bebas)
# ============================================================
print("\n--- 7. Mask Polygon ---")

mask_poly = np.zeros((h, w), dtype=np.uint8)

# Titik-titik polygon berbentuk bintang
cx, cy = w // 2, h // 2
pts = []
for i in range(5):
    # Titik luar bintang
    angle_luar = np.deg2rad(i * 72 - 90)
    pts.append([int(cx + 120 * np.cos(angle_luar)),
                int(cy + 120 * np.sin(angle_luar))])
    # Titik dalam bintang
    angle_dalam = np.deg2rad(i * 72 - 90 + 36)
    pts.append([int(cx + 50 * np.cos(angle_dalam)),
                int(cy + 50 * np.sin(angle_dalam))])

# cv2.fillPoly mengisi polygon dengan warna putih
pts_array = np.array([pts], dtype=np.int32)
cv2.fillPoly(mask_poly, pts_array, 255)

img_poly = cv2.bitwise_and(img, img, mask=mask_poly)
print(f"  Mask bintang: {cv2.countNonZero(mask_poly)} piksel")

# ============================================================
# 8. Aplikasi: Background replacement
# ============================================================
print("\n--- 8. Background Replacement ---")

# Buat background baru (gradien biru-ungu)
bg = np.zeros_like(img)
for y_pos in range(h):
    ratio = y_pos / h
    bg[y_pos, :, 0] = int(200 * ratio)      # Blue
    bg[y_pos, :, 1] = int(50 * (1 - ratio))  # Green
    bg[y_pos, :, 2] = int(150 * ratio)       # Red

# Gunakan mask lingkaran: foreground dari gambar asli, background dari gradien
# Foreground: gambar asli di area mask putih
fg = cv2.bitwise_and(img, img, mask=mask_circle)
# Background: gradien di area mask hitam (inverse mask)
bg_part = cv2.bitwise_and(bg, bg, mask=mask_not)
# Gabungkan foreground + background menggunakan cv2.add
img_composite = cv2.add(fg, bg_part)

print("  Background replacement selesai")

# ============================================================
# 9. Visualisasi
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 1].imshow(cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Mask Persegi")
axes[0, 2].imshow(cv2.cvtColor(img_circle, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Mask Lingkaran")
axes[0, 3].imshow(cv2.cvtColor(img_elips, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Mask Elips")

axes[1, 0].imshow(cv2.cvtColor(img_thresh, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Mask Threshold")
axes[1, 1].imshow(cv2.cvtColor(img_and, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Rect AND Circle")
axes[1, 2].imshow(cv2.cvtColor(img_or, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Rect OR Circle")
axes[1, 3].imshow(mask_xor, cmap="gray")
axes[1, 3].set_title("Mask XOR")

axes[2, 0].imshow(cv2.cvtColor(img_poly, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("Mask Bintang")
axes[2, 1].imshow(cv2.cvtColor(img_biru, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("Mask Warna Biru")
axes[2, 2].imshow(cv2.cvtColor(img_merah, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title("Mask Warna Merah")
axes[2, 3].imshow(cv2.cvtColor(img_composite, cv2.COLOR_BGR2RGB))
axes[2, 3].set_title("BG Replacement")

for ax in axes.flat:
    ax.axis("off")

plt.suptitle("Percobaan 19: Masking Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "19_masking_gambar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 19")
print("=" * 60)
print("  cv2.bitwise_and(img, img, mask=m) → Terapkan mask")
print("  cv2.threshold()     → Mask dari ambang batas")
print("  cv2.inRange()       → Mask dari rentang warna HSV")
print("  Operasi logika: AND, OR, NOT, XOR pada mask")
print("  cv2.fillPoly()      → Mask bentuk polygon")
print("=" * 60)
