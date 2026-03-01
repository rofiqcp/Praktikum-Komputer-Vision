"""
==========================================================================
PERCOBAAN 11: CROPPING GAMBAR
==========================================================================
Program ini mempelajari teknik cropping (pemotongan) gambar untuk
mengambil bagian tertentu. Cropping di OpenCV dilakukan dengan
slicing array NumPy: img[y1:y2, x1:x2]

Teknik yang dipelajari:
- Cropping manual (koordinat tetap)
- Cropping proporsional (persentase)
- Cropping tengah (center crop)
- Cropping dengan padding jika area di luar batas
- Crop dan resize (thumbnail generation)
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
print("PERCOBAAN 11: CROPPING GAMBAR")
print("=" * 60)

# Membaca gambar pemandangan
img = cv2.imread(os.path.join(IMAGE_DIR, "pemandangan.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

tinggi, lebar = img.shape[:2]
print(f"[INFO] Gambar asli: {lebar}×{tinggi}")

# ============================================================
# 1. Cropping manual (koordinat tetap)
# ============================================================
print("\n--- 1. Cropping Manual ---")

# Crop area langit (bagian atas gambar)
crop_langit = img[0:150, 0:lebar]
print(f"  Langit: {crop_langit.shape[1]}×{crop_langit.shape[0]}")

# Crop area tanah (bagian bawah gambar)
crop_tanah = img[300:tinggi, 0:lebar]
print(f"  Tanah: {crop_tanah.shape[1]}×{crop_tanah.shape[0]}")

# Crop area gunung (bagian tengah)
crop_gunung = img[80:310, 50:450]
print(f"  Gunung: {crop_gunung.shape[1]}×{crop_gunung.shape[0]}")

# ============================================================
# 2. Center crop (crop simetris dari tengah)
# ============================================================
print("\n--- 2. Center Crop ---")

def center_crop(img, crop_w, crop_h):
    """Memotong gambar dari bagian tengah."""
    h, w = img.shape[:2]
    # Menghitung titik awal (sudut kiri atas) dari area crop
    start_x = max(0, (w - crop_w) // 2)
    start_y = max(0, (h - crop_h) // 2)
    # Memastikan tidak melebihi batas gambar
    end_x = min(w, start_x + crop_w)
    end_y = min(h, start_y + crop_h)
    return img[start_y:end_y, start_x:end_x].copy()

# Center crop 300x300
crop_center = center_crop(img, 300, 300)
print(f"  Center crop 300×300: {crop_center.shape[1]}×{crop_center.shape[0]}")

# Center crop 200x200
crop_center_small = center_crop(img, 200, 200)
print(f"  Center crop 200×200: {crop_center_small.shape[1]}×{crop_center_small.shape[0]}")

# ============================================================
# 3. Cropping proporsional (berdasarkan persentase)
# ============================================================
print("\n--- 3. Cropping Proporsional ---")

def proportional_crop(img, top_pct=0, bottom_pct=0, left_pct=0, right_pct=0):
    """Memotong gambar berdasarkan persentase dari setiap sisi."""
    h, w = img.shape[:2]
    # Menghitung jumlah piksel yang dipotong dari setiap sisi
    top = int(h * top_pct / 100)
    bottom = h - int(h * bottom_pct / 100)
    left = int(w * left_pct / 100)
    right = w - int(w * right_pct / 100)
    return img[top:bottom, left:right].copy()

# Crop 10% dari setiap sisi
crop_10pct = proportional_crop(img, 10, 10, 10, 10)
print(f"  Crop 10% semua sisi: {crop_10pct.shape[1]}×{crop_10pct.shape[0]}")

# Crop 20% dari atas (hilangkan langit)
crop_tanpa_langit = proportional_crop(img, top_pct=30)
print(f"  Crop 30% atas: {crop_tanpa_langit.shape[1]}×{crop_tanpa_langit.shape[0]}")

# ============================================================
# 4. Crop dan resize (pembuatan thumbnail)
# ============================================================
print("\n--- 4. Thumbnail Generation ---")

def buat_thumbnail(img, thumb_size=150):
    """Membuat thumbnail persegi dari gambar apapun."""
    h, w = img.shape[:2]
    # Menentukan dimensi terkecil
    min_dim = min(h, w)
    # Center crop ke bentuk persegi
    crop_sq = center_crop(img, min_dim, min_dim)
    # Resize ke ukuran thumbnail
    return cv2.resize(crop_sq, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)

# Membuat thumbnail 150x150
thumb = buat_thumbnail(img, 150)
print(f"  Thumbnail 150×150: {thumb.shape}")

# Membuat grid thumbnail dari berbagai crop
thumbs = [
    buat_thumbnail(crop_langit, 100),
    buat_thumbnail(crop_gunung, 100),
    buat_thumbnail(crop_tanah, 100),
]

# ============================================================
# 5. Crop dengan aspect ratio tertentu
# ============================================================
print("\n--- 5. Crop dengan Aspect Ratio ---")

def crop_aspect_ratio(img, ratio_w=16, ratio_h=9):
    """Crop gambar agar sesuai dengan aspect ratio tertentu."""
    h, w = img.shape[:2]
    target_ratio = ratio_w / ratio_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Gambar terlalu lebar → crop horizontal
        new_w = int(h * target_ratio)
        start_x = (w - new_w) // 2
        return img[:, start_x:start_x + new_w].copy()
    else:
        # Gambar terlalu tinggi → crop vertikal
        new_h = int(w / target_ratio)
        start_y = (h - new_h) // 2
        return img[start_y:start_y + new_h, :].copy()

# Crop ke aspect ratio 16:9
crop_16_9 = crop_aspect_ratio(img, 16, 9)
print(f"  16:9: {crop_16_9.shape[1]}×{crop_16_9.shape[0]}")

# Crop ke aspect ratio 1:1 (persegi)
crop_1_1 = crop_aspect_ratio(img, 1, 1)
print(f"  1:1:  {crop_1_1.shape[1]}×{crop_1_1.shape[0]}")

# Crop ke aspect ratio 4:3
crop_4_3 = crop_aspect_ratio(img, 4, 3)
print(f"  4:3:  {crop_4_3.shape[1]}×{crop_4_3.shape[0]}")

# ============================================================
# 6. Visualisasi
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"Asli ({lebar}×{tinggi})")
axes[0, 0].axis("off")

axes[0, 1].imshow(cv2.cvtColor(crop_langit, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Crop Langit (atas)")
axes[0, 1].axis("off")

axes[0, 2].imshow(cv2.cvtColor(crop_gunung, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Crop Gunung (manual)")
axes[0, 2].axis("off")

axes[0, 3].imshow(cv2.cvtColor(crop_center, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Center Crop 300×300")
axes[0, 3].axis("off")

axes[1, 0].imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Thumbnail 150×150")
axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(crop_16_9, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"Ratio 16:9\n{crop_16_9.shape[1]}×{crop_16_9.shape[0]}")
axes[1, 1].axis("off")

axes[1, 2].imshow(cv2.cvtColor(crop_1_1, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f"Ratio 1:1\n{crop_1_1.shape[1]}×{crop_1_1.shape[0]}")
axes[1, 2].axis("off")

axes[1, 3].imshow(cv2.cvtColor(crop_4_3, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title(f"Ratio 4:3\n{crop_4_3.shape[1]}×{crop_4_3.shape[0]}")
axes[1, 3].axis("off")

plt.suptitle("Percobaan 11: Cropping Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "11_cropping_gambar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 11")
print("=" * 60)
print("  1. img[y1:y2, x1:x2]    → Crop manual dengan slicing")
print("  2. Center crop           → Potong simetris dari tengah")
print("  3. Proportional crop     → Potong berdasarkan persentase")
print("  4. Aspect ratio crop     → Potong ke rasio tertentu")
print("  5. Thumbnail             → Center crop + resize")
print("=" * 60)
