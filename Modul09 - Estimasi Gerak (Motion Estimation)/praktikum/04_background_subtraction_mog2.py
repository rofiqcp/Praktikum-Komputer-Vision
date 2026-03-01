"""
==========================================================================
PERCOBAAN 4: BACKGROUND SUBTRACTION DENGAN MOG2
==========================================================================
Program ini mempelajari cara melakukan background subtraction menggunakan
metode MOG2 (Mixture of Gaussians versi 2). Background subtraction
memisahkan objek bergerak (foreground) dari latar belakang (background)
secara otomatis tanpa perlu background referensi.

MOG2 memodelkan setiap piksel sebagai campuran distribusi Gaussian.
Piksel yang tidak cocok dengan model background dianggap foreground.

Fungsi utama yang dipelajari:
- cv2.createBackgroundSubtractorMOG2()  : Membuat model background MOG2
  - history: jumlah frame untuk membangun model
  - varThreshold: threshold varians untuk klasifikasi foreground
  - detectShadows: aktifkan deteksi bayangan (abu-abu di mask)
- subtractor.apply()                    : Menerapkan model ke frame baru
- cv2.morphologyEx()                    : Membersihkan noise pada mask

Hasil: Visualisasi foreground mask MOG2 untuk beberapa frame video
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan video dan gambar
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file
import os

# Mengimpor matplotlib untuk menyimpan visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input dan output
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 4: BACKGROUND SUBTRACTION DENGAN MOG2")
print("=" * 60)

# ============================================================
# 1. Membuka video dan menyiapkan model MOG2
# ============================================================

# Membuka file video yang berisi objek bergerak
video_path = os.path.join(IMAGE_DIR, "video_orang.avi")
cap = cv2.VideoCapture(video_path)

# Memeriksa apakah video berhasil dibuka
if not cap.isOpened():
    print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Membaca informasi video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"[INFO] Video berhasil dibuka: {frame_width}x{frame_height}")
print(f"[INFO] Total frame: {total_frames}")

# ============================================================
# 2. Membuat Background Subtractor MOG2
# cv2.createBackgroundSubtractorMOG2():
#   - history: jumlah frame terakhir yang digunakan membangun model (default 500)
#   - varThreshold: threshold varians, semakin kecil semakin sensitif (default 16)
#   - detectShadows: jika True, bayangan ditandai abu-abu (nilai 127) di mask
# ============================================================

# Membuat objek background subtractor MOG2 dengan parameter kustom
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,         # Menggunakan 500 frame terakhir untuk model
    varThreshold=16,     # Threshold varians standar
    detectShadows=True   # Mengaktifkan deteksi bayangan
)

print("[INFO] Model MOG2 dibuat dengan history=500, varThreshold=16, detectShadows=True")

# ============================================================
# 3. Mendefinisikan kernel untuk operasi morfologi
# ============================================================

# Membuat kernel elips 5x5 untuk operasi morfologi pembersihan noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ============================================================
# 4. Memproses video frame per frame
# ============================================================

# Menyimpan frame hasil untuk visualisasi
original_frames = []
fg_mask_frames = []
fg_clean_frames = []
frame_indices = []
frame_count = 0

# Menentukan frame mana saja yang akan divisualisasikan
# Frame awal model belum stabil, frame akhir model sudah stabil
target_frames = [10, 30, 60, 100]

print("[INFO] Memproses background subtraction frame per frame...")

while True:
    # Membaca frame berikutnya dari video
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Menerapkan background subtractor MOG2 ke frame saat ini
    # apply() mengembalikan foreground mask:
    #   - 255 (putih) = foreground (objek bergerak)
    #   - 127 (abu-abu) = bayangan (jika detectShadows=True)
    #   - 0 (hitam) = background (latar belakang)
    fg_mask = mog2.apply(frame)

    # Membersihkan noise pada foreground mask menggunakan opening
    # Opening = erosi diikuti dilasi, menghilangkan noise kecil
    fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Menerapkan closing untuk menutup lubang kecil pada objek
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, kernel)

    # Menyimpan frame yang ditargetkan untuk divisualisasikan
    if frame_count in target_frames:
        original_frames.append(frame.copy())
        fg_mask_frames.append(fg_mask.copy())
        fg_clean_frames.append(fg_clean.copy())
        frame_indices.append(frame_count)

    # Berhenti jika sudah mencapai target terakhir + buffer
    if frame_count > max(target_frames) + 10:
        break

# Melepaskan video capture setelah selesai
cap.release()

print(f"[INFO] Selesai memproses {frame_count} frame.")

# ============================================================
# 5. Visualisasi hasil: Original vs FG Mask vs FG Clean
# ============================================================

# Membuat figure dengan grid 4 baris x 3 kolom
fig, axes = plt.subplots(len(frame_indices), 3, figsize=(15, 4 * len(frame_indices)))

# Memastikan axes selalu 2D meskipun hanya 1 baris
if len(frame_indices) == 1:
    axes = axes.reshape(1, -1)

for idx in range(len(frame_indices)):
    # Kolom 1: Frame asli (konversi BGR ke RGB untuk matplotlib)
    rgb_frame = cv2.cvtColor(original_frames[idx], cv2.COLOR_BGR2RGB)
    axes[idx, 0].imshow(rgb_frame)
    axes[idx, 0].set_title(f"Frame {frame_indices[idx]} - Original", fontsize=10)
    axes[idx, 0].axis("off")

    # Kolom 2: Foreground mask mentah (putih=FG, abu=bayangan, hitam=BG)
    axes[idx, 1].imshow(fg_mask_frames[idx], cmap="gray")
    axes[idx, 1].set_title(f"Frame {frame_indices[idx]} - FG Mask (Raw)", fontsize=10)
    axes[idx, 1].axis("off")

    # Kolom 3: Foreground mask setelah dibersihkan morfologi
    axes[idx, 2].imshow(fg_clean_frames[idx], cmap="gray")
    axes[idx, 2].set_title(f"Frame {frame_indices[idx]} - FG Mask (Clean)", fontsize=10)
    axes[idx, 2].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 4: Background Subtraction MOG2\nPutih=Foreground, Abu-abu=Bayangan, Hitam=Background",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil visualisasi ke file
output_path = os.path.join(OUTPUT_DIR, "04_background_subtraction_mog2.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# 6. Visualisasi pengaruh parameter varThreshold
# ============================================================

# Membuka kembali video untuk membandingkan parameter yang berbeda
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Gagal membuka ulang video.")
    exit()

# Mendefinisikan variasi varThreshold yang akan dibandingkan
var_thresholds = [8, 16, 50, 100]

# Membuat subtractor untuk setiap variasi threshold
subtractors = []
for vt in var_thresholds:
    # Membuat model MOG2 dengan threshold berbeda
    sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=vt, detectShadows=False
    )
    subtractors.append(sub)

# Memproses beberapa frame awal agar model stabil
frame_count2 = 0
comparison_frame = None
comparison_masks = [None] * len(var_thresholds)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count2 += 1

    # Menerapkan setiap subtractor ke frame yang sama
    masks = []
    for sub in subtractors:
        m = sub.apply(frame)
        masks.append(m)

    # Menyimpan hasil di frame ke-60 (model sudah cukup stabil)
    if frame_count2 == 60:
        comparison_frame = frame.copy()
        comparison_masks = [m.copy() for m in masks]
        break

# Melepaskan video capture
cap.release()

# Membuat figure untuk perbandingan threshold
fig2, axes2 = plt.subplots(1, len(var_thresholds) + 1, figsize=(18, 4))

# Menampilkan frame asli di kolom pertama
rgb_comp = cv2.cvtColor(comparison_frame, cv2.COLOR_BGR2RGB)
axes2[0].imshow(rgb_comp)
axes2[0].set_title("Frame Asli (ke-60)", fontsize=10)
axes2[0].axis("off")

# Menampilkan mask untuk setiap threshold
for i, (vt, mask) in enumerate(zip(var_thresholds, comparison_masks)):
    axes2[i + 1].imshow(mask, cmap="gray")
    axes2[i + 1].set_title(f"varThreshold={vt}", fontsize=10)
    axes2[i + 1].axis("off")

# Menambahkan judul utama untuk perbandingan
plt.suptitle("Pengaruh varThreshold pada MOG2\nSemakin kecil = semakin sensitif (lebih banyak foreground)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil perbandingan threshold
output_path2 = os.path.join(OUTPUT_DIR, "04_mog2_varthreshold_comparison.png")
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Perbandingan threshold disimpan di: {output_path2}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 4")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.createBackgroundSubtractorMOG2()  → Membuat model BG MOG2")
print("     - history: jumlah frame untuk membangun model (default 500)")
print("     - varThreshold: threshold varians (kecil = sensitif)")
print("     - detectShadows: deteksi bayangan (abu-abu di mask)")
print("  2. subtractor.apply(frame)               → Menghasilkan foreground mask")
print("     - 255 = foreground, 127 = bayangan, 0 = background")
print("  3. cv2.morphologyEx()                    → Membersihkan noise mask")
print("     - MORPH_OPEN: menghilangkan noise kecil")
print("     - MORPH_CLOSE: menutup lubang kecil")
print("  4. MOG2 adaptif: model background terus diperbarui seiring waktu")
print("  5. varThreshold kecil → lebih sensitif (banyak false positive)")
print("     varThreshold besar → kurang sensitif (bisa miss objek kecil)")
print("=" * 60)
