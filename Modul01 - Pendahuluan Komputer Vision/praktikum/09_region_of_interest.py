"""
==========================================================================
PERCOBAAN 9: REGION OF INTEREST (ROI)
==========================================================================
Program ini mempelajari konsep Region of Interest (ROI) yaitu
memilih dan memanipulasi area tertentu dari gambar.

ROI sangat berguna untuk:
- Memproses hanya bagian tertentu dari gambar (efisien)
- Crop objek yang diinginkan
- Copy-paste area gambar
- Deteksi dan tracking objek di area tertentu

Cara membuat ROI:
- Slicing array: img[y1:y2, x1:x2]
- Menggunakan mask
- cv2.selectROI() untuk seleksi interaktif
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
print("PERCOBAAN 9: REGION OF INTEREST (ROI)")
print("=" * 60)

# Membaca gambar
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

tinggi, lebar = img.shape[:2]
print(f"[INFO] Gambar dimuat: {lebar}×{tinggi}")

# ============================================================
# 1. ROI dengan slicing array (cara paling umum)
# ============================================================
print("\n--- 1. ROI dengan Slicing ---")

# Memilih ROI: area tengah gambar
# Format: img[y_start:y_end, x_start:x_end]
# Menghitung koordinat tengah
y_center = tinggi // 2
x_center = lebar // 2
roi_size = 150

# Menentukan batas ROI (area persegi di tengah)
y1 = y_center - roi_size
y2 = y_center + roi_size
x1 = x_center - roi_size
x2 = x_center + roi_size

# Mengambil ROI menggunakan slicing NumPy
roi_tengah = img[y1:y2, x1:x2].copy()
print(f"  ROI Tengah: ({x1},{y1}) sampai ({x2},{y2})")
print(f"  Ukuran ROI: {roi_tengah.shape}")

# ============================================================
# 2. Beberapa ROI dari posisi berbeda
# ============================================================
print("\n--- 2. Beberapa ROI ---")

# ROI kiri atas (sudut kiri atas gambar)
roi_kiri_atas = img[0:150, 0:200].copy()
print(f"  ROI Kiri Atas: ukuran {roi_kiri_atas.shape}")

# ROI kanan bawah
roi_kanan_bawah = img[tinggi-150:tinggi, lebar-200:lebar].copy()
print(f"  ROI Kanan Bawah: ukuran {roi_kanan_bawah.shape}")

# ============================================================
# 3. Memanipulasi ROI (mengubah area tertentu)
# ============================================================
print("\n--- 3. Manipulasi ROI ---")

# Membuat salinan gambar untuk modifikasi
img_modif = img.copy()

# Mengkonversi ROI tengah ke grayscale lalu kembali ke BGR
roi_gray = cv2.cvtColor(roi_tengah, cv2.COLOR_BGR2GRAY)
roi_gray_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

# Memasukkan ROI grayscale kembali ke gambar
img_modif[y1:y2, x1:x2] = roi_gray_bgr
print("  Area tengah diubah ke grayscale")

# Menggambar kotak penanda ROI
cv2.rectangle(img_modif, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(img_modif, "ROI", (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# ============================================================
# 4. Copy-paste ROI (memindahkan area)
# ============================================================
print("\n--- 4. Copy-Paste ROI ---")

img_copaste = img.copy()

# Menyalin area kiri atas ke kanan bawah
src_roi = img[50:200, 50:250].copy()
h_roi, w_roi = src_roi.shape[:2]

# Menempelkan ROI di posisi baru (kanan bawah)
dst_y = tinggi - h_roi - 10
dst_x = lebar - w_roi - 10
img_copaste[dst_y:dst_y + h_roi, dst_x:dst_x + w_roi] = src_roi

# Menandai area sumber dan tujuan
cv2.rectangle(img_copaste, (50, 50), (250, 200), (0, 0, 255), 2)
cv2.putText(img_copaste, "Sumber", (55, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.rectangle(img_copaste, (dst_x, dst_y), (dst_x + w_roi, dst_y + h_roi),
              (255, 0, 0), 2)
cv2.putText(img_copaste, "Tujuan", (dst_x + 5, dst_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

print("  ROI disalin dari kiri-atas ke kanan-bawah")

# ============================================================
# 5. ROI dengan efek blur (privasi/sensor)
# ============================================================
print("\n--- 5. ROI Blur (Efek Sensor) ---")

img_blur = img.copy()

# Menentukan area yang ingin di-blur (simulasi area privasi)
blur_y1, blur_y2 = 100, 300
blur_x1, blur_x2 = 150, 400

# Mengambil ROI dan menerapkan Gaussian Blur
roi_to_blur = img_blur[blur_y1:blur_y2, blur_x1:blur_x2]

# cv2.GaussianBlur(src, ksize, sigmaX)
# ksize = ukuran kernel (harus ganjil), semakin besar semakin blur
roi_blurred = cv2.GaussianBlur(roi_to_blur, (51, 51), 0)

# Memasukkan ROI yang sudah di-blur kembali ke gambar
img_blur[blur_y1:blur_y2, blur_x1:blur_x2] = roi_blurred

# Menandai area blur
cv2.rectangle(img_blur, (blur_x1, blur_y1), (blur_x2, blur_y2), (0, 255, 255), 2)
cv2.putText(img_blur, "BLURRED", (blur_x1 + 5, blur_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

print("  Area tertentu berhasil di-blur")

# ============================================================
# 6. ROI dengan mask lingkaran
# ============================================================
print("\n--- 6. ROI Lingkaran (Mask) ---")

# Membuat mask lingkaran (hitam dengan lingkaran putih)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
center = (lebar // 2, tinggi // 2)
radius = min(lebar, tinggi) // 3

# Membuat lingkaran putih pada mask
cv2.circle(mask, center, radius, 255, -1)

# Menerapkan mask ke gambar menggunakan bitwise_and
img_circle_roi = cv2.bitwise_and(img, img, mask=mask)

print(f"  ROI lingkaran: pusat={center}, radius={radius}")

# ============================================================
# 7. ROI dengan resize (zoom)
# ============================================================
print("\n--- 7. ROI Zoom ---")

# Mengambil ROI kecil dan memperbesarnya
roi_kecil = img[y_center-50:y_center+50, x_center-50:x_center+50].copy()

# Memperbesar ROI 4x menggunakan cv2.resize
roi_zoom = cv2.resize(roi_kecil, (400, 400), interpolation=cv2.INTER_CUBIC)
print(f"  ROI kecil (100×100) diperbesar ke (400×400)")

# ============================================================
# 8. Visualisasi semua hasil
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Baris 1
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli")
axes[0, 0].axis("off")

axes[0, 1].imshow(cv2.cvtColor(roi_tengah, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"ROI Tengah\n{roi_tengah.shape[1]}×{roi_tengah.shape[0]}")
axes[0, 1].axis("off")

axes[0, 2].imshow(cv2.cvtColor(img_modif, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("ROI → Grayscale")
axes[0, 2].axis("off")

axes[0, 3].imshow(cv2.cvtColor(img_copaste, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Copy-Paste ROI")
axes[0, 3].axis("off")

# Baris 2
axes[1, 0].imshow(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("ROI Blur (Sensor)")
axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(img_circle_roi, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("ROI Lingkaran (Mask)")
axes[1, 1].axis("off")

axes[1, 2].imshow(cv2.cvtColor(roi_kecil, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f"ROI Kecil\n{roi_kecil.shape[1]}×{roi_kecil.shape[0]}")
axes[1, 2].axis("off")

axes[1, 3].imshow(cv2.cvtColor(roi_zoom, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title(f"ROI Zoom 4×\n{roi_zoom.shape[1]}×{roi_zoom.shape[0]}")
axes[1, 3].axis("off")

plt.suptitle("Percobaan 9: Region of Interest (ROI)", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "09_roi_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 9")
print("=" * 60)
print("  1. img[y1:y2, x1:x2]    → Seleksi ROI dengan slicing")
print("  2. roi.copy()            → Salin ROI (hindari referensi)")
print("  3. img[y1:y2, x1:x2]=roi → Tempelkan ROI ke gambar")
print("  4. cv2.GaussianBlur()    → Blur pada ROI tertentu")
print("  5. cv2.bitwise_and(mask) → ROI berbentuk non-persegi")
print("  6. cv2.resize(roi, size) → Zoom ROI")
print("=" * 60)
