"""
==========================================================================
PERCOBAAN 11: MOTION HISTORY IMAGE (MHI)
==========================================================================
Program ini mempelajari representasi Motion History Image (MHI) untuk
merekam jejak gerakan dalam video. MHI menyimpan informasi temporal
gerakan di mana piksel yang baru bergerak bernilai terang (bright)
dan piksel yang sudah lama bergerak bernilai gelap (dark).

Konsep MHI:
- Setiap frame, deteksi area yang bergerak (frame differencing)
- Update MHI: area bergerak → nilai timestamp maksimum
- Area tidak bergerak → perlahan memudar (decay)
- Hasil: gambar grayscale yang menunjukkan KAPAN gerakan terjadi

Fungsi utama yang dipelajari:
- cv2.absdiff()                   : Menghitung perbedaan antar frame
- cv2.threshold()                 : Threshold untuk deteksi gerakan
- cv2.motempl.updateMotionHistory(): Update MHI (jika tersedia)
- cv2.applyColorMap()             : Memberikan warna pada MHI
- cv2.GaussianBlur()              : Menghaluskan noise pada mask

Hasil: Visualisasi MHI dalam mode grayscale dan colormap
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

# Mengimpor time untuk pencatatan timestamp
import time

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input dan output
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 11: MOTION HISTORY IMAGE (MHI)")
print("=" * 60)

# ============================================================
# 1. Membuka video dan menyiapkan variabel
# ============================================================

# Membuka file video yang berisi objek bergerak
video_path = os.path.join(IMAGE_DIR, "video_bola.avi")
cap = cv2.VideoCapture(video_path)

# Memeriksa apakah video berhasil dibuka
if not cap.isOpened():
    print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Membaca informasi video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"[INFO] Video berhasil dibuka: {width}x{height}")
print(f"[INFO] FPS: {fps}, Total frame: {total_frames}")

# Membaca frame pertama sebagai referensi awal
ret, prev_frame = cap.read()
if not ret:
    print("[ERROR] Gagal membaca frame pertama.")
    exit()

# Mengkonversi frame pertama ke grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Inisialisasi Motion History Image (MHI)
# ============================================================

# Membuat array MHI dengan ukuran sama dengan frame, tipe float32
# MHI menyimpan nilai timestamp (waktu) terakhir gerakan terdeteksi
mhi = np.zeros((height, width), dtype=np.float32)

# Durasi gerakan tetap terlihat di MHI (dalam satuan timestamp)
# Semakin besar = jejak gerakan bertahan lebih lama
MHI_DURATION = 30.0

# Threshold untuk mendeteksi gerakan dari perbedaan frame
DIFF_THRESHOLD = 25

# Variabel untuk mencatat timestamp (dinaikkan setiap frame)
timestamp = 0

# Memeriksa ketersediaan modul motempl di OpenCV
has_motempl = hasattr(cv2, 'motempl')
if has_motempl:
    print("[INFO] cv2.motempl tersedia, menggunakan updateMotionHistory()")
else:
    print("[INFO] cv2.motempl TIDAK tersedia, menggunakan implementasi manual MHI")

# ============================================================
# 3. Memproses video frame per frame dan mengupdate MHI
# ============================================================

# Menyimpan beberapa frame MHI untuk visualisasi
mhi_snapshots = []
frame_snapshots = []
diff_snapshots = []
snapshot_indices = []

# Menentukan frame mana yang akan disimpan untuk visualisasi
target_frames = [10, 30, 60, 90]

print("[INFO] Memproses video dan menghitung MHI...")

# Counter frame
frame_count = 0

while True:
    # Membaca frame berikutnya dari video
    ret, frame = cap.read()
    if not ret:
        break

    # Menaikkan counter frame
    frame_count += 1

    # Menaikkan timestamp untuk setiap frame
    timestamp += 1

    # Mengkonversi frame saat ini ke grayscale
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ============================================================
    # Langkah 3a: Menghitung perbedaan absolut antar frame
    # cv2.absdiff() menghasilkan nilai |frame_t - frame_{t-1}|
    # Piksel yang berubah (bergerak) akan memiliki nilai besar
    # ============================================================
    frame_diff = cv2.absdiff(curr_gray, prev_gray)

    # Menghaluskan perbedaan dengan Gaussian blur untuk mengurangi noise
    frame_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)

    # ============================================================
    # Langkah 3b: Threshold pada perbedaan untuk membuat binary mask
    # cv2.threshold() mengkonversi ke hitam-putih (0 atau 255)
    # Piksel > DIFF_THRESHOLD dianggap bergerak
    # ============================================================
    _, motion_mask = cv2.threshold(frame_diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # ============================================================
    # Langkah 3c: Update Motion History Image
    # ============================================================
    if has_motempl:
        # Menggunakan fungsi bawaan OpenCV untuk update MHI
        # updateMotionHistory(silhouette, mhi, timestamp, duration)
        # - silhouette: binary mask gerakan
        # - mhi: MHI yang diupdate (in-place)
        # - timestamp: waktu saat ini
        # - duration: berapa lama jejak bertahan
        cv2.motempl.updateMotionHistory(motion_mask, mhi, timestamp, MHI_DURATION)
    else:
        # Implementasi manual MHI
        # Untuk piksel yang bergerak (mask == 255): set nilai MHI ke timestamp saat ini
        mhi[motion_mask == 255] = timestamp

        # Untuk piksel yang TIDAK bergerak: cek apakah sudah melewati durasi
        # Jika (timestamp - mhi_value) > MHI_DURATION → set ke 0 (memudar)
        expired = (timestamp - mhi) > MHI_DURATION
        mhi[expired] = 0

    # ============================================================
    # Langkah 3d: Normalisasi MHI untuk visualisasi (0-255)
    # Piksel dengan timestamp terbaru → terang (255)
    # Piksel dengan timestamp lama → gelap
    # Piksel tanpa gerakan → hitam (0)
    # ============================================================

    # Menghitung MHI yang dinormalisasi
    mhi_vis = np.clip((mhi - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1)
    mhi_vis = (mhi_vis * 255).astype(np.uint8)

    # Menyimpan snapshot pada frame-frame tertentu
    if frame_count in target_frames:
        mhi_snapshots.append(mhi_vis.copy())
        frame_snapshots.append(frame.copy())
        diff_snapshots.append(motion_mask.copy())
        snapshot_indices.append(frame_count)

    # Memperbarui frame sebelumnya untuk iterasi berikutnya
    prev_gray = curr_gray.copy()

# Melepaskan video capture
cap.release()

print(f"[INFO] Selesai memproses {frame_count} frame.")

# ============================================================
# 4. Visualisasi MHI pada beberapa titik waktu
# ============================================================

# Membuat figure utama: MHI Grayscale pada beberapa frame
fig1, axes1 = plt.subplots(2, 4, figsize=(18, 9))

for idx in range(min(4, len(mhi_snapshots))):
    # Baris atas: frame asli
    rgb_frame = cv2.cvtColor(frame_snapshots[idx], cv2.COLOR_BGR2RGB)
    axes1[0, idx].imshow(rgb_frame)
    axes1[0, idx].set_title(f"Frame Asli #{snapshot_indices[idx]}", fontsize=10)
    axes1[0, idx].axis("off")

    # Baris bawah: MHI grayscale
    axes1[1, idx].imshow(mhi_snapshots[idx], cmap='gray', vmin=0, vmax=255)
    axes1[1, idx].set_title(f"MHI Grayscale #{snapshot_indices[idx]}", fontsize=10)
    axes1[1, idx].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 11: Motion History Image (MHI) - Grayscale\n"
             "Terang = gerakan baru, Gelap = gerakan lama", fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan figure pertama
output_path_1 = os.path.join(OUTPUT_DIR, "11_mhi_grayscale.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Hasil MHI grayscale disimpan di: {output_path_1}")

# ============================================================
# 5. Visualisasi MHI dengan colormap
# ============================================================

# Membuat figure kedua: perbandingan MHI dengan berbagai colormap
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))

# Menggunakan snapshot MHI terakhir yang tersedia
last_mhi = mhi_snapshots[-1] if mhi_snapshots else np.zeros((height, width), dtype=np.uint8)

# Daftar colormap yang akan digunakan
colormaps_cv = [
    (cv2.COLORMAP_JET, "JET"),
    (cv2.COLORMAP_HOT, "HOT"),
    (cv2.COLORMAP_BONE, "BONE"),
]

# Daftar colormap matplotlib
colormaps_mpl = ["jet", "hot", "viridis"]

# Baris atas: MHI dengan colormap OpenCV
for idx, (cmap_id, cmap_name) in enumerate(colormaps_cv):
    # Menerapkan colormap pada MHI menggunakan cv2.applyColorMap()
    mhi_colored = cv2.applyColorMap(last_mhi, cmap_id)
    # Mengkonversi BGR ke RGB untuk matplotlib
    mhi_colored_rgb = cv2.cvtColor(mhi_colored, cv2.COLOR_BGR2RGB)
    axes2[0, idx].imshow(mhi_colored_rgb)
    axes2[0, idx].set_title(f"MHI + Colormap {cmap_name} (OpenCV)", fontsize=10)
    axes2[0, idx].axis("off")

# Baris bawah: MHI dengan colormap matplotlib
for idx, cmap_name in enumerate(colormaps_mpl):
    axes2[1, idx].imshow(last_mhi, cmap=cmap_name, vmin=0, vmax=255)
    axes2[1, idx].set_title(f"MHI + Colormap {cmap_name} (Matplotlib)", fontsize=10)
    axes2[1, idx].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 11: MHI dengan Berbagai Colormap\n"
             "Warna menunjukkan waktu terjadinya gerakan", fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan figure kedua
output_path_2 = os.path.join(OUTPUT_DIR, "11_mhi_colormap.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Hasil MHI colormap disimpan di: {output_path_2}")

# ============================================================
# 6. Visualisasi motion mask vs MHI
# ============================================================

# Membuat figure ketiga: perbandingan motion mask dan MHI
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

# Menampilkan motion mask (binary)
if diff_snapshots:
    axes3[0].imshow(diff_snapshots[-1], cmap='gray')
    axes3[0].set_title("Motion Mask (Binary)", fontsize=11)
    axes3[0].axis("off")

# Menampilkan MHI grayscale
axes3[1].imshow(last_mhi, cmap='gray', vmin=0, vmax=255)
axes3[1].set_title("MHI Grayscale", fontsize=11)
axes3[1].axis("off")

# Menampilkan MHI dengan colormap JET
mhi_jet = cv2.applyColorMap(last_mhi, cv2.COLORMAP_JET)
mhi_jet_rgb = cv2.cvtColor(mhi_jet, cv2.COLOR_BGR2RGB)
axes3[2].imshow(mhi_jet_rgb)
axes3[2].set_title("MHI + JET Colormap", fontsize=11)
axes3[2].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 11: Perbandingan Motion Mask vs MHI", fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan figure ketiga
output_path_3 = os.path.join(OUTPUT_DIR, "11_mhi_comparison.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Hasil perbandingan disimpan di: {output_path_3}")

# Menampilkan statistik MHI
print(f"\n[INFO] Statistik MHI terakhir:")
print(f"  - Nilai minimum MHI : {last_mhi.min()}")
print(f"  - Nilai maksimum MHI: {last_mhi.max()}")
print(f"  - Nilai rata-rata   : {last_mhi.mean():.2f}")
print(f"  - Piksel bergerak   : {np.count_nonzero(last_mhi)} dari {width*height}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 11")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.absdiff()                    → Perbedaan absolut antar frame")
print("  2. cv2.threshold()                  → Membuat binary mask gerakan")
print("  3. cv2.motempl.updateMotionHistory() → Update MHI (jika tersedia)")
print("  4. cv2.applyColorMap()              → Pewarnaan MHI dengan colormap")
print("  5. cv2.GaussianBlur()               → Menghaluskan noise")
print("Konsep MHI:")
print("  - MHI = gambar di mana pixel terang = gerakan baru")
print("  - Pixel gelap = gerakan yang sudah lama terjadi")
print("  - MHI menyimpan informasi TEMPORAL (waktu)")
print("  - Berguna untuk mengenali pola gerakan (gesture recognition)")
print("  - MHI_DURATION mengatur berapa lama jejak bertahan")
print("=" * 60)
