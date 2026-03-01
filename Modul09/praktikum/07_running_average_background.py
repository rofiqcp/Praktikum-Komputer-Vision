"""
==========================================================================
PERCOBAAN 7: RUNNING AVERAGE BACKGROUND MODEL
==========================================================================
Program ini mempelajari cara membangun model background secara adaptif
menggunakan running average (rata-rata tertimbang). Model ini terus
memperbarui estimasi background setiap frame baru datang.

Rumus: background = alpha * frame + (1 - alpha) * background
- alpha kecil (0.01): background sangat lambat berubah → lebih stabil
- alpha besar (0.5): background cepat berubah → adaptif tapi noisy

Objek yang bergerak terdeteksi dari perbedaan antara frame saat ini
dan model background yang sudah dipelajari.

Fungsi utama yang dipelajari:
- cv2.accumulateWeighted()        : Memperbarui running average
  - dst = alpha * src + (1 - alpha) * dst
- cv2.absdiff()                   : Menghitung perbedaan frame vs background
- cv2.convertScaleAbs()           : Konversi ke format uint8 absolut
- cv2.threshold()                 : Mendapatkan mask biner foreground

Hasil: Perbandingan efek parameter alpha pada background model
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
print("PERCOBAAN 7: RUNNING AVERAGE BACKGROUND MODEL")
print("=" * 60)

# ============================================================
# 1. Membuka video dan membaca frame pertama
# ============================================================

# Membuka file video
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

# Membaca frame pertama
ret, first_frame = cap.read()
if not ret:
    print("[ERROR] Gagal membaca frame pertama.")
    exit()

# Mengkonversi ke grayscale sebagai format kerja
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Menginisialisasi model background
# Model diinisialisasi dengan frame pertama (tipe float32)
# cv2.accumulateWeighted() membutuhkan input float32
# ============================================================

# Menginisialisasi background model dari frame pertama (konversi ke float32)
bg_model = first_gray.astype(np.float32)

print("[INFO] Background model diinisialisasi dari frame pertama.")

# ============================================================
# 3. Mendefinisikan parameter
# ============================================================

# Alpha: learning rate untuk running average
# Semakin kecil alpha, semakin lambat background berubah
ALPHA = 0.05

# Threshold untuk klasifikasi foreground/background
THRESH_VALUE = 30

# Membuat kernel untuk operasi morfologi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print(f"[INFO] Alpha (learning rate): {ALPHA}")
print(f"[INFO] Threshold foreground: {THRESH_VALUE}")

# ============================================================
# 4. Memproses video frame per frame
# ============================================================

# Menyimpan frame untuk visualisasi
original_frames = []
bg_frames = []
diff_frames = []
fg_frames = []
frame_indices = []
frame_count = 0

# Frame yang ingin divisualisasikan
target_frames = [5, 20, 50, 90]

print("[INFO] Memproses running average background model...")

while True:
    # Membaca frame berikutnya
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Mengkonversi frame saat ini ke grayscale
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ============================================================
    # Memperbarui background model dengan running average
    # cv2.accumulateWeighted(src, dst, alpha)
    # dst = alpha * src + (1 - alpha) * dst
    # - src: frame saat ini (harus float32 atau sesuai dst)
    # - dst: background model (float32, diupdate in-place)
    # - alpha: bobot frame baru (learning rate)
    # ============================================================
    cv2.accumulateWeighted(curr_gray, bg_model, ALPHA)

    # Mengkonversi background model ke uint8 untuk komputasi
    bg_uint8 = cv2.convertScaleAbs(bg_model)

    # ============================================================
    # Menghitung perbedaan antara frame saat ini dan background model
    # Piksel yang berbeda jauh = foreground (objek bergerak)
    # ============================================================
    diff = cv2.absdiff(curr_gray, bg_uint8)

    # Menerapkan threshold untuk mendapatkan binary foreground mask
    _, fg_mask = cv2.threshold(diff, THRESH_VALUE, 255, cv2.THRESH_BINARY)

    # Membersihkan noise dengan operasi morfologi
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Menyimpan frame target untuk visualisasi
    if frame_count in target_frames:
        original_frames.append(frame.copy())
        bg_frames.append(bg_uint8.copy())
        diff_frames.append(diff.copy())
        fg_frames.append(fg_mask.copy())
        frame_indices.append(frame_count)

    # Berhenti setelah melewati target terakhir
    if frame_count > max(target_frames) + 10:
        break

# Melepaskan video capture
cap.release()

print(f"[INFO] Selesai memproses {frame_count} frame.")

# ============================================================
# 5. Visualisasi hasil running average
# ============================================================

# Membuat figure 4 baris x 4 kolom
fig, axes = plt.subplots(len(frame_indices), 4, figsize=(18, 4 * len(frame_indices)))

if len(frame_indices) == 1:
    axes = axes.reshape(1, -1)

for idx in range(len(frame_indices)):
    # Kolom 1: Frame asli
    rgb_frame = cv2.cvtColor(original_frames[idx], cv2.COLOR_BGR2RGB)
    axes[idx, 0].imshow(rgb_frame)
    axes[idx, 0].set_title(f"Frame {frame_indices[idx]} - Original", fontsize=9)
    axes[idx, 0].axis("off")

    # Kolom 2: Background model saat ini
    axes[idx, 1].imshow(bg_frames[idx], cmap="gray")
    axes[idx, 1].set_title(f"Frame {frame_indices[idx]} - BG Model", fontsize=9)
    axes[idx, 1].axis("off")

    # Kolom 3: Difference (frame vs background)
    axes[idx, 2].imshow(diff_frames[idx], cmap="hot")
    axes[idx, 2].set_title(f"Frame {frame_indices[idx]} - Difference", fontsize=9)
    axes[idx, 2].axis("off")

    # Kolom 4: Foreground mask biner
    axes[idx, 3].imshow(fg_frames[idx], cmap="gray")
    axes[idx, 3].set_title(f"Frame {frame_indices[idx]} - FG Mask", fontsize=9)
    axes[idx, 3].axis("off")

# Menambahkan judul utama
plt.suptitle(f"Percobaan 7: Running Average Background (alpha={ALPHA})\nbg = alpha * frame + (1-alpha) * bg",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil visualisasi
output_path = os.path.join(OUTPUT_DIR, "07_running_average_background.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# 6. Perbandingan efek parameter alpha
# ============================================================

# Mendefinisikan variasi alpha yang akan dibandingkan
alpha_values = [0.01, 0.05, 0.2, 0.5]

# Menyimpan hasil untuk setiap alpha
alpha_bg_results = []
alpha_fg_results = []

for alpha_val in alpha_values:
    # Membuka video dari awal untuk setiap alpha
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        continue

    # Membaca frame pertama sebagai inisialisasi
    ret, f0 = cap.read()
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)

    # Inisialisasi background model
    bg = g0.astype(np.float32)

    fc = 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        fc += 1

        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        # Memperbarui model dengan alpha tertentu
        cv2.accumulateWeighted(g, bg, alpha_val)

        # Menyimpan hasil di frame ke-60
        if fc == 60:
            bg_u8 = cv2.convertScaleAbs(bg)
            df = cv2.absdiff(g, bg_u8)
            _, fg = cv2.threshold(df, THRESH_VALUE, 255, cv2.THRESH_BINARY)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

            alpha_bg_results.append(bg_u8.copy())
            alpha_fg_results.append(fg.copy())
            break

    cap.release()

# Membuat figure untuk perbandingan alpha
fig2, axes2 = plt.subplots(2, len(alpha_values), figsize=(16, 7))

for i, alpha_val in enumerate(alpha_values):
    # Baris 1: Background model
    axes2[0, i].imshow(alpha_bg_results[i], cmap="gray")
    axes2[0, i].set_title(f"BG Model\nalpha={alpha_val}", fontsize=10)
    axes2[0, i].axis("off")

    # Baris 2: Foreground mask
    axes2[1, i].imshow(alpha_fg_results[i], cmap="gray")
    axes2[1, i].set_title(f"FG Mask\nalpha={alpha_val}", fontsize=10)
    axes2[1, i].axis("off")

# Menambahkan label baris
axes2[0, 0].set_ylabel("Background", fontsize=11)
axes2[1, 0].set_ylabel("Foreground", fontsize=11)

# Menambahkan judul utama
plt.suptitle("Pengaruh Alpha pada Running Average Background Model (Frame ke-60)\nAlpha kecil=stabil, Alpha besar=adaptif",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan perbandingan alpha
output_path2 = os.path.join(OUTPUT_DIR, "07_alpha_comparison.png")
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Perbandingan alpha disimpan di: {output_path2}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 7")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.accumulateWeighted(src, dst, alpha)")
print("     → Memperbarui running average: dst = alpha*src + (1-alpha)*dst")
print("     - src: frame saat ini")
print("     - dst: background model (float32)")
print("     - alpha: learning rate (0.0 - 1.0)")
print("  2. cv2.absdiff()              → Perbedaan frame vs background")
print("  3. cv2.convertScaleAbs()      → Konversi float32 → uint8")
print("  4. Efek parameter alpha:")
print("     - alpha kecil (0.01): BG sangat stabil, lambat adaptasi")
print("     - alpha sedang (0.05): keseimbangan stabilitas & adaptasi")
print("     - alpha besar (0.5): BG cepat berubah, noisy")
print("  5. Running average = metode sederhana tapi efektif")
print("     Cocok untuk kamera statis dengan background yang perlahan berubah")
print("=" * 60)
