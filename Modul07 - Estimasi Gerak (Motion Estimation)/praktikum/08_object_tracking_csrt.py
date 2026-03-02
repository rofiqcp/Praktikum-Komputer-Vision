"""
==========================================================================
PERCOBAAN 8: SINGLE OBJECT TRACKING DENGAN CSRT
==========================================================================
Program ini mempelajari cara melacak (tracking) satu objek dalam video
menggunakan algoritma CSRT (Channel and Spatial Reliability Tracker).
CSRT adalah salah satu tracker paling akurat di OpenCV, cocok untuk
objek yang berubah bentuk atau mengalami oklusi parsial.

Langkah tracking:
1. Memilih ROI (Region of Interest) awal di frame pertama
2. Menginisialisasi tracker dengan ROI tersebut
3. Memperbarui posisi objek di setiap frame berikutnya

Fungsi utama yang dipelajari:
- cv2.TrackerCSRT_create() / cv2.legacy.TrackerCSRT_create()
  : Membuat tracker CSRT
- tracker.init(frame, bbox)       : Inisialisasi tracker dengan ROI awal
- tracker.update(frame)           : Memperbarui posisi objek di frame baru
  → return (success, bbox)        : status sukses dan bounding box baru

Hasil: Visualisasi pelacakan objek dengan bounding box di beberapa frame
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
print("PERCOBAAN 8: SINGLE OBJECT TRACKING DENGAN CSRT")
print("=" * 60)

# ============================================================
# 1. Membuat tracker CSRT
# OpenCV versi berbeda memiliki API yang berbeda:
# - OpenCV 4.5+: cv2.legacy.TrackerCSRT_create()
# - OpenCV 4.x: cv2.TrackerCSRT_create()
# Menggunakan try/except untuk kompatibilitas
# ============================================================

def create_csrt_tracker():
    """Membuat CSRT tracker dengan kompatibilitas versi OpenCV."""
    try:
        # Mencoba API OpenCV baru (4.5+) yang ada di modul legacy
        tracker = cv2.legacy.TrackerCSRT_create()
        print("[INFO] Menggunakan cv2.legacy.TrackerCSRT_create()")
    except AttributeError:
        try:
            # Mencoba API OpenCV lama (4.x)
            tracker = cv2.TrackerCSRT_create()
            print("[INFO] Menggunakan cv2.TrackerCSRT_create()")
        except AttributeError:
            print("[ERROR] CSRT tracker tidak tersedia di versi OpenCV ini.")
            print("[INFO] Install opencv-contrib-python: pip install opencv-contrib-python")
            return None
    return tracker

# ============================================================
# 2. Membuka video dan membaca frame pertama
# ============================================================

# Membuka file video yang berisi objek bergerak (bola)
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
ret, first_frame = cap.read()
if not ret:
    print("[ERROR] Gagal membaca frame pertama.")
    exit()

# ============================================================
# 3. Mendefinisikan bounding box awal secara manual
# bbox = (x, y, width, height) dari objek yang akan dilacak
# Disesuaikan dengan posisi bola di frame pertama video_bola.avi
# ============================================================

# Mendefinisikan bounding box awal untuk bola (x, y, w, h)
# Posisi ini disesuaikan dengan video sintetis dari download_image.py
bbox_initial = (70, 70, 60, 60)

print(f"[INFO] Bounding box awal: {bbox_initial}")

# ============================================================
# 4. Menginisialisasi tracker CSRT
# ============================================================

# Membuat objek tracker CSRT
tracker = create_csrt_tracker()
if tracker is None:
    exit()

# Menginisialisasi tracker dengan frame pertama dan bounding box awal
# tracker.init(frame, bbox) → melatih model appearance objek
tracker.init(first_frame, bbox_initial)

print("[INFO] Tracker CSRT berhasil diinisialisasi.")

# ============================================================
# 5. Melacak objek frame per frame
# ============================================================

# Menyimpan frame hasil untuk visualisasi
tracked_frames = []
frame_indices = []
bbox_history = [bbox_initial]  # Menyimpan riwayat posisi bbox
success_count = 0
fail_count = 0
frame_count = 0

# Frame target untuk divisualisasikan
target_frames = [1, 20, 50, 80, 110, 140]

# Menyimpan frame pertama dengan bbox awal
display_first = first_frame.copy()
x, y, w, h = [int(v) for v in bbox_initial]
cv2.rectangle(display_first, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(display_first, "CSRT: Frame 1 (Init)", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
tracked_frames.append(display_first)
frame_indices.append(1)

print("[INFO] Memulai tracking objek...")

while True:
    # Membaca frame berikutnya
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ============================================================
    # Memperbarui posisi tracker di frame baru
    # tracker.update(frame) mengembalikan:
    #   - success (bool): True jika tracking berhasil
    #   - bbox (tuple): (x, y, w, h) posisi baru objek
    # ============================================================
    success, bbox = tracker.update(frame)

    # Membuat salinan frame untuk digambar
    display = frame.copy()

    if success:
        # Tracking berhasil: menggambar bounding box hijau
        success_count += 1
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, f"CSRT: Frame {frame_count + 1} (OK)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        bbox_history.append(bbox)
    else:
        # Tracking gagal: menampilkan pesan merah
        fail_count += 1
        cv2.putText(display, f"CSRT: Frame {frame_count + 1} (LOST)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Menyimpan frame target untuk visualisasi
    if (frame_count + 1) in target_frames:
        tracked_frames.append(display.copy())
        frame_indices.append(frame_count + 1)

# Melepaskan video capture
cap.release()

print(f"[INFO] Tracking selesai: {frame_count + 1} frame diproses.")
print(f"[INFO] Berhasil: {success_count} frame, Gagal: {fail_count} frame")
print(f"[INFO] Akurasi tracking: {success_count / (success_count + fail_count) * 100:.1f}%")

# ============================================================
# 6. Visualisasi hasil tracking
# ============================================================

# Menentukan jumlah frame hasil yang tersedia
n_results = min(len(tracked_frames), 6)
n_cols = 3
n_rows = (n_results + n_cols - 1) // n_cols

# Membuat figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten() if n_results > 1 else [axes]

for idx in range(n_results):
    # Konversi BGR ke RGB
    rgb = cv2.cvtColor(tracked_frames[idx], cv2.COLOR_BGR2RGB)
    axes[idx].imshow(rgb)
    axes[idx].set_title(f"Frame {frame_indices[idx]}", fontsize=11)
    axes[idx].axis("off")

# Menyembunyikan subplot kosong
for idx in range(n_results, len(axes)):
    axes[idx].axis("off")

# Menambahkan judul utama
plt.suptitle(f"Percobaan 8: Single Object Tracking dengan CSRT\nBerhasil: {success_count}, Gagal: {fail_count}",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil visualisasi tracking
output_path = os.path.join(OUTPUT_DIR, "08_object_tracking_csrt.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[OUTPUT] Hasil tracking disimpan di: {output_path}")

# ============================================================
# 7. Visualisasi trajectory (lintasan) objek
# ============================================================

# Membuat plot trajectory dari riwayat bounding box
if len(bbox_history) > 1:
    # Mengekstrak posisi tengah bbox dari riwayat
    centers_x = [b[0] + b[2] / 2 for b in bbox_history]
    centers_y = [b[1] + b[3] / 2 for b in bbox_history]

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))

    # Menggambar lintasan objek
    ax2.plot(centers_x, centers_y, 'b-', linewidth=1.5, alpha=0.7, label="Lintasan")

    # Menandai posisi awal dan akhir
    ax2.plot(centers_x[0], centers_y[0], 'go', markersize=12, label="Mulai")
    ax2.plot(centers_x[-1], centers_y[-1], 'ro', markersize=12, label="Akhir")

    # Menggambar titik-titik di setiap frame
    scatter = ax2.scatter(centers_x, centers_y, c=range(len(centers_x)),
                          cmap='viridis', s=20, zorder=5)
    plt.colorbar(scatter, ax=ax2, label="Nomor Frame")

    # Membalik sumbu y agar sesuai dengan koordinat gambar (origin di kiri atas)
    ax2.invert_yaxis()
    ax2.set_xlabel("Posisi X (piksel)", fontsize=11)
    ax2.set_ylabel("Posisi Y (piksel)", fontsize=11)
    ax2.set_title("Trajectory Objek yang Dilacak (CSRT Tracker)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Menyimpan trajectory
    output_path2 = os.path.join(OUTPUT_DIR, "08_csrt_trajectory.png")
    plt.savefig(output_path2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[OUTPUT] Trajectory disimpan di: {output_path2}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 8")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.legacy.TrackerCSRT_create() → Membuat CSRT tracker")
print("     atau cv2.TrackerCSRT_create() untuk OpenCV versi lama")
print("  2. tracker.init(frame, bbox)       → Inisialisasi tracker")
print("     - frame: frame pertama (BGR)")
print("     - bbox: (x, y, w, h) bounding box awal")
print("  3. tracker.update(frame)           → Update posisi objek")
print("     - Return: (success, bbox)")
print("     - success: True jika tracking berhasil")
print("     - bbox: posisi baru (x, y, w, h)")
print("  4. CSRT = Channel and Spatial Reliability Tracker")
print("     - Akurasi tinggi, tapi lebih lambat")
print("     - Menggunakan spatial reliability map")
print("     - Tahan terhadap perubahan skala & deformasi")
print("  5. Perlu bounding box awal (manual atau dari detektor)")
print("=" * 60)
