"""
==========================================================================
PERCOBAAN 13: FLIP GAMBAR (PENCERMINAN)
==========================================================================
Program ini mempelajari cara membalik/mencerminkan gambar secara
horizontal, vertikal, dan kombinasi keduanya.

Fungsi utama:
- cv2.flip(src, flipCode) → Cerminkan gambar
  flipCode = 0  : vertikal (atas-bawah)
  flipCode = 1  : horizontal (kiri-kanan)
  flipCode = -1 : keduanya (vertikal + horizontal)
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
print("PERCOBAAN 13: FLIP GAMBAR (PENCERMINAN)")
print("=" * 60)

# Membaca gambar berwarna
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

# ============================================================
# 1. Flip horizontal (cermin kiri-kanan)
# ============================================================
print("\n--- 1. Flip Horizontal ---")

# cv2.flip(src, 1) → mencerminkan gambar secara horizontal
# flipCode=1 berarti membalik sumbu Y (kiri jadi kanan)
img_flip_h = cv2.flip(img, 1)
print("  flipCode=1 → horizontal (kiri ↔ kanan)")

# ============================================================
# 2. Flip vertikal (cermin atas-bawah)
# ============================================================
print("\n--- 2. Flip Vertikal ---")

# cv2.flip(src, 0) → mencerminkan gambar secara vertikal
# flipCode=0 berarti membalik sumbu X (atas jadi bawah)
img_flip_v = cv2.flip(img, 0)
print("  flipCode=0 → vertikal (atas ↔ bawah)")

# ============================================================
# 3. Flip keduanya (horizontal + vertikal)
# ============================================================
print("\n--- 3. Flip Horizontal + Vertikal ---")

# cv2.flip(src, -1) → cermin horizontal DAN vertikal
# Hasilnya sama dengan rotasi 180°
img_flip_hv = cv2.flip(img, -1)
print("  flipCode=-1 → horizontal + vertikal (setara rotasi 180°)")

# ============================================================
# 4. Verifikasi flip -1 = rotasi 180°
# ============================================================
print("\n--- 4. Verifikasi Flip -1 == Rotasi 180° ---")

# Rotasi 180° menggunakan cv2.rotate
img_rot180 = cv2.rotate(img, cv2.ROTATE_180)

# Menghitung perbedaan pixel antara flip(-1) dan rotasi 180°
# cv2.absdiff menghitung |A - B| per pixel
perbedaan = cv2.absdiff(img_flip_hv, img_rot180)
# np.sum menjumlahkan seluruh nilai piksel perbedaan
total_diff = np.sum(perbedaan)
print(f"  Total perbedaan pixel: {total_diff}")
print(f"  Flip -1 identik dengan rotasi 180°: {total_diff == 0}")

# ============================================================
# 5. Flip menggunakan NumPy (alternatif)
# ============================================================
print("\n--- 5. Flip Menggunakan NumPy ---")

# np.fliplr → flip left-right (horizontal)
img_np_h = np.fliplr(img)
# np.flipud → flip up-down (vertikal)
img_np_v = np.flipud(img)
# np.fliplr + np.flipud → keduanya
img_np_hv = np.flipud(np.fliplr(img))

# Verifikasi hasilnya sama
diff_h = np.sum(cv2.absdiff(img_flip_h, img_np_h))
diff_v = np.sum(cv2.absdiff(img_flip_v, img_np_v))
print(f"  NumPy fliplr == cv2.flip(1): {diff_h == 0}")
print(f"  NumPy flipud == cv2.flip(0): {diff_v == 0}")

# ============================================================
# 6. Flip dengan array slicing (cara paling dasar)
# ============================================================
print("\n--- 6. Flip dengan Slicing ---")

# Horizontal flip: balik urutan kolom [:, ::-1]
img_slice_h = img[:, ::-1, :]
# Vertikal flip: balik urutan baris [::-1]
img_slice_v = img[::-1, :, :]
# Keduanya: balik baris dan kolom
img_slice_hv = img[::-1, ::-1, :]

print("  img[:, ::-1] → horizontal flip")
print("  img[::-1, :] → vertikal flip")
print("  img[::-1, ::-1] → keduanya")

# ============================================================
# 7. Aplikasi: Membuat efek refleksi air
# ============================================================
print("\n--- 7. Efek Refleksi Air ---")

# Ambil setengah bawah gambar
tinggi = img.shape[0]
# Bagian atas gambar tetap
atas = img[:tinggi // 2, :]
# Flip vertikal bagian atas untuk efek refleksi
refleksi = cv2.flip(atas, 0)

# Buat efek transparan pada refleksi
# Buat gradien fade untuk efek air
fade = np.linspace(0.7, 0.1, refleksi.shape[0]).reshape(-1, 1, 1)
# Kalikan refleksi dengan gradien fade
refleksi_fade = (refleksi * fade).astype(np.uint8)

# Tambahkan efek biru pada refleksi (tint air)
refleksi_biru = refleksi_fade.copy()
# Garis biru tipis untuk efek riak air
for y in range(0, refleksi_biru.shape[0], 4):
    # Geser piksel sedikit horizontal untuk efek riak
    shift = int(2 * np.sin(y * 0.1))
    # np.roll menggeser array secara circular
    refleksi_biru[y] = np.roll(refleksi_biru[y], shift, axis=0)

# Gabungkan atas + refleksi
# np.vstack menggabungkan array secara vertikal
img_refleksi = np.vstack([atas, refleksi_biru])
print(f"  Ukuran: {img_refleksi.shape[1]}×{img_refleksi.shape[0]}")

# ============================================================
# 8. Visualisasi
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Baris 1: Flip dasar
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(cv2.cvtColor(img_flip_h, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Flip Horizontal (1)")
axes[0, 1].axis("off")

axes[0, 2].imshow(cv2.cvtColor(img_flip_v, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Flip Vertikal (0)")
axes[0, 2].axis("off")

axes[0, 3].imshow(cv2.cvtColor(img_flip_hv, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Flip Keduanya (-1)")
axes[0, 3].axis("off")

# Baris 2: Alternatif & aplikasi
axes[1, 0].imshow(cv2.cvtColor(img_np_h, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("NumPy fliplr")
axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(img_slice_v, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Slicing [::-1]")
axes[1, 1].axis("off")

axes[1, 2].imshow(cv2.cvtColor(img_rot180, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Rotasi 180°")
axes[1, 2].axis("off")

axes[1, 3].imshow(cv2.cvtColor(img_refleksi, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Efek Refleksi Air")
axes[1, 3].axis("off")

plt.suptitle("Percobaan 13: Flip Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "13_flip_gambar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 13")
print("=" * 60)
print("  cv2.flip(src, 1)   → Horizontal (kiri ↔ kanan)")
print("  cv2.flip(src, 0)   → Vertikal (atas ↔ bawah)")
print("  cv2.flip(src, -1)  → Keduanya (= rotasi 180°)")
print("  np.fliplr/flipud   → Alternatif NumPy")
print("  img[::-1,::-1]     → Alternatif slicing")
print("=" * 60)
