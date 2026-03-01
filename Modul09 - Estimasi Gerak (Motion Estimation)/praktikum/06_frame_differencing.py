"""
==========================================================================
PERCOBAAN 6: FRAME DIFFERENCING UNTUK DETEKSI GERAKAN
==========================================================================
Program ini mempelajari teknik paling dasar untuk deteksi gerakan, yaitu
frame differencing. Prinsipnya sederhana: menghitung perbedaan absolut
antara frame saat ini dan frame sebelumnya, lalu menerapkan threshold
untuk mendapatkan mask biner area yang bergerak.

|frame_t - frame_{t-1}| > threshold → gerakan terdeteksi

Teknik ini sangat cepat namun sensitif terhadap noise dan perubahan
pencahayaan. Operasi morfologi digunakan untuk membersihkan noise.

Fungsi utama yang dipelajari:
- cv2.absdiff()                   : Menghitung perbedaan absolut dua frame
- cv2.threshold()                 : Mengubah grayscale ke biner dengan threshold
- cv2.morphologyEx()              : Membersihkan noise (opening/closing)
- cv2.dilate()                    : Menebalkan area foreground
- cv2.findContours()              : Menemukan kontur objek bergerak

Hasil: Visualisasi difference image dan binary motion mask
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
print("PERCOBAAN 6: FRAME DIFFERENCING UNTUK DETEKSI GERAKAN")
print("=" * 60)

# ============================================================
# 1. Membuka video dan membaca frame pertama
# ============================================================

# Membuka file video
video_path = os.path.join(IMAGE_DIR, "video_bola.avi")
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
ret, prev_frame = cap.read()
if not ret:
    print("[ERROR] Gagal membaca frame pertama.")
    exit()

# Mengkonversi frame pertama ke grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Membuat kernel untuk operasi morfologi
# ============================================================

# Membuat kernel persegi panjang 5x5 untuk operasi morfologi
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Membuat kernel lebih besar untuk dilasi (menebalkan objek)
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

# ============================================================
# 3. Memproses frame differencing
# ============================================================

# Menyimpan frame untuk visualisasi
original_frames = []
diff_frames = []
binary_frames = []
contour_frames = []
frame_indices = []
frame_count = 0

# Frame target untuk divisualisasikan
target_frames = [10, 30, 50, 80]

# Mendefinisikan threshold untuk klasifikasi gerakan
THRESH_VALUE = 30  # Piksel dengan perbedaan > 30 dianggap bergerak

print(f"[INFO] Threshold gerakan: {THRESH_VALUE}")
print("[INFO] Memproses frame differencing...")

while True:
    # Membaca frame berikutnya
    ret, curr_frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Mengkonversi frame saat ini ke grayscale
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # ============================================================
    # Menghitung perbedaan absolut antara frame saat ini dan sebelumnya
    # cv2.absdiff(src1, src2) = |src1 - src2|
    # Hasilnya: piksel terang = ada perubahan, gelap = tidak berubah
    # ============================================================
    diff = cv2.absdiff(curr_gray, prev_gray)

    # ============================================================
    # Menerapkan threshold ke difference image
    # cv2.threshold(src, thresh, maxval, type)
    # Piksel > THRESH_VALUE → 255 (putih, bergerak)
    # Piksel <= THRESH_VALUE → 0 (hitam, diam)
    # ============================================================
    _, binary_mask = cv2.threshold(diff, THRESH_VALUE, 255, cv2.THRESH_BINARY)

    # ============================================================
    # Membersihkan noise menggunakan operasi morfologi
    # MORPH_OPEN: menghilangkan titik-titik noise kecil (erosi + dilasi)
    # MORPH_CLOSE: menutup lubang kecil di dalam objek (dilasi + erosi)
    # ============================================================
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # Menebalkan area foreground agar objek lebih terlihat jelas
    clean_mask = cv2.dilate(clean_mask, kernel_dilate, iterations=1)

    # ============================================================
    # Menemukan kontur objek bergerak dan menggambar bounding box
    # ============================================================
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Membuat salinan frame untuk digambar kontur
    contour_display = curr_frame.copy()

    for cnt in contours:
        # Menghitung luas kontur, abaikan kontur yang terlalu kecil
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter noise: hanya objek > 500 piksel
            # Mendapatkan bounding box dari kontur
            x, y, w, h = cv2.boundingRect(cnt)

            # Menggambar kotak hijau di sekitar objek bergerak
            cv2.rectangle(contour_display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Menambahkan teks "Gerakan" di atas kotak
            cv2.putText(contour_display, "Gerakan", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Menyimpan frame target untuk visualisasi
    if frame_count in target_frames:
        original_frames.append(curr_frame.copy())
        diff_frames.append(diff.copy())
        binary_frames.append(clean_mask.copy())
        contour_frames.append(contour_display.copy())
        frame_indices.append(frame_count)

    # Memperbarui frame sebelumnya untuk iterasi berikutnya
    prev_gray = curr_gray.copy()

    # Berhenti setelah melewati target terakhir
    if frame_count > max(target_frames) + 10:
        break

# Melepaskan video capture
cap.release()

print(f"[INFO] Selesai memproses {frame_count} frame.")

# ============================================================
# 4. Visualisasi hasil frame differencing
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

    # Kolom 2: Difference image (grayscale)
    axes[idx, 1].imshow(diff_frames[idx], cmap="hot")
    axes[idx, 1].set_title(f"Frame {frame_indices[idx]} - Difference", fontsize=9)
    axes[idx, 1].axis("off")

    # Kolom 3: Binary motion mask
    axes[idx, 2].imshow(binary_frames[idx], cmap="gray")
    axes[idx, 2].set_title(f"Frame {frame_indices[idx]} - Motion Mask", fontsize=9)
    axes[idx, 2].axis("off")

    # Kolom 4: Frame dengan bounding box
    rgb_contour = cv2.cvtColor(contour_frames[idx], cv2.COLOR_BGR2RGB)
    axes[idx, 3].imshow(rgb_contour)
    axes[idx, 3].set_title(f"Frame {frame_indices[idx]} - Deteksi", fontsize=9)
    axes[idx, 3].axis("off")

# Menambahkan judul utama
plt.suptitle(f"Percobaan 6: Frame Differencing (threshold={THRESH_VALUE})\n|frame_t - frame_{{t-1}}| > threshold → gerakan",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil visualisasi
output_path = os.path.join(OUTPUT_DIR, "06_frame_differencing.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# 5. Visualisasi pengaruh threshold
# ============================================================

# Membuka kembali video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Gagal membuka ulang video.")
    exit()

# Membaca dua frame berturutan di area frame ke-50
for i in range(49):
    cap.read()

ret1, f1 = cap.read()
ret2, f2 = cap.read()
cap.release()

if ret1 and ret2:
    # Mengkonversi ke grayscale
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # Menghitung difference
    diff_comp = cv2.absdiff(g1, g2)

    # Membandingkan berbagai threshold
    thresholds = [10, 20, 30, 50]

    fig2, axes2 = plt.subplots(1, len(thresholds) + 1, figsize=(18, 4))

    # Menampilkan difference image
    axes2[0].imshow(diff_comp, cmap="hot")
    axes2[0].set_title("Difference Image", fontsize=10)
    axes2[0].axis("off")

    # Menampilkan mask untuk setiap threshold
    for i, th in enumerate(thresholds):
        _, mask_th = cv2.threshold(diff_comp, th, 255, cv2.THRESH_BINARY)
        axes2[i + 1].imshow(mask_th, cmap="gray")
        axes2[i + 1].set_title(f"Threshold = {th}", fontsize=10)
        axes2[i + 1].axis("off")

    plt.suptitle("Pengaruh Threshold pada Frame Differencing\nThreshold kecil = sensitif, Threshold besar = hanya gerakan besar",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Menyimpan hasil perbandingan threshold
    output_path2 = os.path.join(OUTPUT_DIR, "06_threshold_comparison.png")
    plt.savefig(output_path2, dpi=150, bbox_inches="tight")
    print(f"[OUTPUT] Perbandingan threshold disimpan di: {output_path2}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 6")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.absdiff(frame1, frame2)  → Perbedaan absolut dua frame")
print("     - Hasil: |frame1 - frame2| per piksel")
print("     - Terang = ada perubahan, gelap = tidak berubah")
print("  2. cv2.threshold()              → Konversi ke biner")
print("     - > threshold → 255 (gerakan)")
print("     - <= threshold → 0 (diam)")
print("  3. cv2.morphologyEx()           → Membersihkan noise")
print("     - MORPH_OPEN: hilangkan noise kecil")
print("     - MORPH_CLOSE: tutup lubang dalam objek")
print("  4. cv2.findContours()           → Menemukan objek bergerak")
print("  5. Frame differencing SANGAT sederhana tapi:")
print("     - Sensitif terhadap noise dan perubahan cahaya")
print("     - Hanya mendeteksi TEPI gerakan, bukan seluruh objek")
print("     - Cocok untuk skenario sederhana dengan background statis")
print("=" * 60)
