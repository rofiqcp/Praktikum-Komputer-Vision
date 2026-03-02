"""
==========================================================================
PERCOBAAN 10: MULTI-OBJECT TRACKING
==========================================================================
Program ini mempelajari cara melacak beberapa objek secara bersamaan
dalam video. Setiap objek memiliki tracker tersendiri dan ditandai
dengan bounding box berwarna berbeda.

Pendekatan yang digunakan:
- Membuat beberapa tracker individual untuk setiap objek
- Mengelola tracker secara manual dalam loop (kompatibel semua versi)
- Juga mencoba cv2.legacy.MultiTracker_create() jika tersedia

Fungsi utama yang dipelajari:
- cv2.legacy.MultiTracker_create()  : Membuat multi-tracker (jika tersedia)
- multiTracker.add()                : Menambahkan tracker untuk satu objek
- multiTracker.update()             : Update semua tracker sekaligus
- Manual multi-tracking dengan list of trackers

Hasil: Visualisasi pelacakan multi-objek dengan warna berbeda per objek
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

# Mengimpor time untuk mengukur kecepatan
import time

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input dan output
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 10: MULTI-OBJECT TRACKING")
print("=" * 60)

# ============================================================
# 1. Fungsi helper untuk membuat tracker individu
# ============================================================

def create_single_tracker(tracker_type="CSRT"):
    """
    Membuat satu objek tracker.
    Mendukung CSRT dan KCF dengan kompatibilitas versi OpenCV.
    """
    if tracker_type == "CSRT":
        try:
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            try:
                return cv2.TrackerCSRT_create()
            except AttributeError:
                return None
    elif tracker_type == "KCF":
        try:
            return cv2.legacy.TrackerKCF_create()
        except AttributeError:
            try:
                return cv2.TrackerKCF_create()
            except AttributeError:
                return None
    return None

# ============================================================
# 2. Membuka video multi-objek
# ============================================================

# Membuka file video yang berisi beberapa objek bergerak
video_path = os.path.join(IMAGE_DIR, "video_multi_objek.avi")
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
# 3. Mendefinisikan bounding box awal untuk setiap objek
# Setiap objek memiliki (x, y, w, h) dan warna yang berbeda
# Disesuaikan dengan video_multi_objek.avi dari download_image.py
# ============================================================

# Daftar bounding box awal untuk setiap objek yang ingin dilacak
initial_bboxes = [
    (80, 80, 50, 50),     # Objek 1 (bola merah): posisi kiri atas
    (300, 200, 50, 50),   # Objek 2 (bola hijau): posisi tengah
    (500, 100, 50, 50),   # Objek 3 (bola biru): posisi kanan atas
]

# Warna berbeda untuk setiap objek (BGR format)
object_colors = [
    (0, 0, 255),    # Merah untuk objek 1
    (0, 255, 0),    # Hijau untuk objek 2
    (255, 0, 0),    # Biru untuk objek 3
]

# Label nama untuk setiap objek
object_labels = ["Objek-1", "Objek-2", "Objek-3"]

print(f"[INFO] Jumlah objek yang akan dilacak: {len(initial_bboxes)}")

# ============================================================
# 4. Inisialisasi multi-tracker secara manual
# Membuat satu tracker untuk setiap objek
# ============================================================

# Daftar untuk menyimpan semua tracker
trackers = []
tracker_type = "CSRT"

for i, bbox in enumerate(initial_bboxes):
    # Membuat tracker baru untuk objek ke-i
    tracker = create_single_tracker(tracker_type)
    if tracker is None:
        print(f"[ERROR] Gagal membuat tracker untuk {object_labels[i]}")
        continue

    # Menginisialisasi tracker dengan frame pertama dan bbox awal
    tracker.init(first_frame, bbox)
    trackers.append(tracker)
    print(f"[INFO] Tracker {tracker_type} untuk {object_labels[i]} diinisialisasi: bbox={bbox}")

# Memeriksa apakah ada tracker yang berhasil dibuat
if len(trackers) == 0:
    print("[ERROR] Tidak ada tracker yang berhasil dibuat!")
    exit()

# ============================================================
# 5. Melacak semua objek frame per frame
# ============================================================

# Menyimpan frame untuk visualisasi
tracked_frames = []
frame_indices = []
frame_count = 0

# Frame target untuk divisualisasikan
target_frames = [1, 20, 40, 60, 80, 100]

# Menyimpan riwayat posisi setiap objek untuk trajectory
trajectories = [[] for _ in range(len(trackers))]

# Menyimpan statistik per objek
obj_success = [0] * len(trackers)
obj_fail = [0] * len(trackers)

# Menyimpan frame pertama dengan bbox awal
display_first = first_frame.copy()
for i, bbox in enumerate(initial_bboxes):
    x, y, w, h = [int(v) for v in bbox]
    color = object_colors[i] if i < len(object_colors) else (255, 255, 255)
    cv2.rectangle(display_first, (x, y), (x + w, y + h), color, 2)
    cv2.putText(display_first, object_labels[i], (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Menyimpan pusat bbox awal ke trajectory
    cx, cy = x + w // 2, y + h // 2
    trajectories[i].append((cx, cy))
tracked_frames.append(display_first)
frame_indices.append(1)

print("[INFO] Memulai multi-object tracking...")
start_time = time.time()

while True:
    # Membaca frame berikutnya
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Membuat salinan frame untuk digambar
    display = frame.copy()

    # Memperbarui setiap tracker secara individual
    for i, tracker in enumerate(trackers):
        # Memperbarui posisi tracker ke-i
        success, bbox = tracker.update(frame)

        if success:
            obj_success[i] += 1
            x, y, w, h = [int(v) for v in bbox]
            color = object_colors[i] if i < len(object_colors) else (255, 255, 255)

            # Menggambar bounding box dengan warna unik per objek
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            # Menambahkan label objek
            cv2.putText(display, object_labels[i], (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Menyimpan pusat bbox ke trajectory
            cx, cy = x + w // 2, y + h // 2
            trajectories[i].append((cx, cy))
        else:
            obj_fail[i] += 1

    # Menambahkan informasi frame
    cv2.putText(display, f"Frame: {frame_count + 1}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Menyimpan frame target
    if (frame_count + 1) in target_frames:
        tracked_frames.append(display.copy())
        frame_indices.append(frame_count + 1)

# Melepaskan video capture
cap.release()
total_time = time.time() - start_time

print(f"[INFO] Multi-tracking selesai: {frame_count + 1} frame dalam {total_time:.2f} detik")
for i in range(len(trackers)):
    total_obj = obj_success[i] + obj_fail[i]
    acc = obj_success[i] / total_obj * 100 if total_obj > 0 else 0
    print(f"  {object_labels[i]}: Berhasil={obj_success[i]}, Gagal={obj_fail[i]}, Akurasi={acc:.1f}%")

# ============================================================
# 6. Visualisasi hasil multi-object tracking
# ============================================================

# Menentukan jumlah frame hasil
n_results = min(len(tracked_frames), 6)
n_cols = 3
n_rows = (n_results + n_cols - 1) // n_cols

# Membuat figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

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
plt.suptitle(f"Percobaan 10: Multi-Object Tracking ({len(trackers)} Objek, {tracker_type})\nSetiap objek memiliki warna bounding box berbeda",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil visualisasi
output_path = os.path.join(OUTPUT_DIR, "10_multi_object_tracking.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[OUTPUT] Hasil tracking disimpan di: {output_path}")

# ============================================================
# 7. Visualisasi trajectory semua objek
# ============================================================

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))

# Warna RGB untuk matplotlib (konversi dari BGR)
mpl_colors = [(c[2]/255, c[1]/255, c[0]/255) for c in object_colors]

for i in range(len(trackers)):
    if len(trajectories[i]) > 1:
        # Mengekstrak koordinat x dan y dari trajectory
        traj_x = [p[0] for p in trajectories[i]]
        traj_y = [p[1] for p in trajectories[i]]

        # Menggambar lintasan objek
        color = mpl_colors[i] if i < len(mpl_colors) else 'gray'
        ax2.plot(traj_x, traj_y, '-', color=color, linewidth=2,
                 label=object_labels[i], alpha=0.8)

        # Menandai posisi awal (lingkaran) dan akhir (bintang)
        ax2.plot(traj_x[0], traj_y[0], 'o', color=color, markersize=10)
        ax2.plot(traj_x[-1], traj_y[-1], '*', color=color, markersize=14)

# Membalik sumbu y agar sesuai koordinat gambar
ax2.invert_yaxis()
ax2.set_xlabel("Posisi X (piksel)", fontsize=11)
ax2.set_ylabel("Posisi Y (piksel)", fontsize=11)
ax2.set_title("Trajectory Semua Objek yang Dilacak", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10, loc="best")
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Menyimpan trajectory
output_path2 = os.path.join(OUTPUT_DIR, "10_multi_tracking_trajectory.png")
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
plt.show()
print(f"[OUTPUT] Trajectory disimpan di: {output_path2}")

# ============================================================
# 8. Ringkasan statistik per objek
# ============================================================

fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))

# Data akurasi per objek
accuracies = []
for i in range(len(trackers)):
    total_obj = obj_success[i] + obj_fail[i]
    acc = obj_success[i] / total_obj * 100 if total_obj > 0 else 0
    accuracies.append(acc)

# Membuat bar chart akurasi per objek
bars = ax3.bar(object_labels[:len(trackers)], accuracies,
               color=[mpl_colors[i] for i in range(len(trackers))],
               edgecolor="black")

# Menambahkan label nilai
for bar, val in zip(bars, accuracies):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{val:.1f}%", ha="center", fontsize=11)

ax3.set_ylabel("Akurasi Tracking (%)", fontsize=11)
ax3.set_title(f"Akurasi Tracking Per Objek (Tracker: {tracker_type})",
              fontsize=13, fontweight="bold")
ax3.set_ylim(0, 110)
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout()

# Menyimpan statistik
output_path3 = os.path.join(OUTPUT_DIR, "10_multi_tracking_stats.png")
plt.savefig(output_path3, dpi=150, bbox_inches="tight")
plt.show()
print(f"[OUTPUT] Statistik disimpan di: {output_path3}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 10")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Multi-object tracking = melacak beberapa objek sekaligus")
print("  2. Pendekatan manual (kompatibel semua versi):")
print("     - Membuat list of tracker individu")
print("     - Loop: tracker.init() dan tracker.update() per objek")
print("  3. cv2.legacy.MultiTracker_create() (jika tersedia):")
print("     - multiTracker.add(tracker, frame, bbox)")
print("     - multiTracker.update(frame) → update semua sekaligus")
print("  4. Setiap objek memiliki:")
print("     - Tracker tersendiri (CSRT/KCF)")
print("     - Bounding box dengan warna unik")
print("     - Statistik tracking independen")
print("  5. Tantangan multi-tracking:")
print("     - Oklusi antar objek (saling menutupi)")
print("     - ID switching (tracker tertukar objek)")
print("     - Komputasi bertambah seiring jumlah objek")
print("=" * 60)
