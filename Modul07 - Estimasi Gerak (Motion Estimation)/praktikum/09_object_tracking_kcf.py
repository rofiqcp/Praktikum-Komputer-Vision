"""
==========================================================================
PERCOBAAN 9: SINGLE OBJECT TRACKING KCF DAN PERBANDINGAN DENGAN CSRT
==========================================================================
Program ini mempelajari cara melacak objek menggunakan algoritma KCF
(Kernelized Correlation Filter) dan membandingkan performanya dengan
CSRT dari percobaan sebelumnya.

KCF lebih cepat dari CSRT namun kurang akurat untuk objek yang berubah
skala. Perbandingan dilakukan dari sisi kecepatan (FPS) dan akurasi
(jumlah frame berhasil di-track).

Fungsi utama yang dipelajari:
- cv2.TrackerKCF_create() / cv2.legacy.TrackerKCF_create()
  : Membuat tracker KCF
- tracker.init(frame, bbox)       : Inisialisasi tracker
- tracker.update(frame)           : Update posisi objek
- time.time()                     : Mengukur kecepatan tracking

Hasil: Perbandingan visual dan statistik KCF vs CSRT
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

# Mengimpor time untuk mengukur kecepatan tracking
import time

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input dan output
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 9: TRACKING KCF DAN PERBANDINGAN DENGAN CSRT")
print("=" * 60)

# ============================================================
# 1. Fungsi helper untuk membuat tracker
# ============================================================

def create_tracker(tracker_type):
    """
    Membuat objek tracker berdasarkan tipe.
    Mendukung: 'KCF', 'CSRT'
    Menggunakan try/except untuk kompatibilitas versi OpenCV.
    """
    if tracker_type == "KCF":
        try:
            # Mencoba API legacy (OpenCV 4.5+)
            tracker = cv2.legacy.TrackerKCF_create()
        except AttributeError:
            try:
                # Mencoba API lama (OpenCV 4.x)
                tracker = cv2.TrackerKCF_create()
            except AttributeError:
                print(f"[ERROR] Tracker {tracker_type} tidak tersedia.")
                return None
    elif tracker_type == "CSRT":
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            try:
                tracker = cv2.TrackerCSRT_create()
            except AttributeError:
                print(f"[ERROR] Tracker {tracker_type} tidak tersedia.")
                return None
    else:
        print(f"[ERROR] Tipe tracker tidak dikenal: {tracker_type}")
        return None

    return tracker

# ============================================================
# 2. Fungsi untuk menjalankan tracking pada video
# ============================================================

def run_tracking(video_path, tracker_type, bbox_initial, target_frames):
    """
    Menjalankan tracking pada video dan mengumpulkan metrik.
    Return: tracked_frames, frame_indices, stats (dict)
    """
    # Membuka video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Video tidak ditemukan: {video_path}")
        return [], [], {}

    # Membaca frame pertama
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return [], [], {}

    # Membuat dan menginisialisasi tracker
    tracker = create_tracker(tracker_type)
    if tracker is None:
        cap.release()
        return [], [], {}

    # Menginisialisasi tracker dengan bbox awal
    tracker.init(first_frame, bbox_initial)

    # Menyimpan frame pertama
    display_first = first_frame.copy()
    x, y, w, h = [int(v) for v in bbox_initial]
    cv2.rectangle(display_first, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(display_first, f"{tracker_type}: Frame 1 (Init)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    tracked_frames = [display_first]
    frame_indices = [1]
    success_count = 0
    fail_count = 0
    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Mengukur waktu tracking per frame
        start_time = time.time()
        success, bbox = tracker.update(frame)
        elapsed = time.time() - start_time
        total_time += elapsed

        # Membuat display frame
        display = frame.copy()
        if success:
            success_count += 1
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, f"{tracker_type}: Frame {frame_count + 1}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            fail_count += 1
            cv2.putText(display, f"{tracker_type}: LOST", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Menyimpan frame target
        if (frame_count + 1) in target_frames:
            tracked_frames.append(display.copy())
            frame_indices.append(frame_count + 1)

    cap.release()

    # Menghitung statistik
    total_processed = success_count + fail_count
    avg_fps = total_processed / total_time if total_time > 0 else 0
    accuracy = success_count / total_processed * 100 if total_processed > 0 else 0

    stats = {
        "total_frames": total_processed,
        "success": success_count,
        "fail": fail_count,
        "avg_fps": avg_fps,
        "total_time": total_time,
        "accuracy": accuracy
    }

    return tracked_frames, frame_indices, stats

# ============================================================
# 3. Menjalankan tracking KCF dan CSRT
# ============================================================

# Path video
video_path = os.path.join(IMAGE_DIR, "video_bola.avi")

# Bounding box awal (disesuaikan dengan video_bola.avi)
bbox_initial = (70, 70, 60, 60)

# Frame target untuk visualisasi
target_frames = [1, 30, 60, 90, 120]

print(f"[INFO] Video: {video_path}")
print(f"[INFO] Bounding box awal: {bbox_initial}")

# Menjalankan tracking dengan KCF
print("\n[INFO] Menjalankan tracking KCF...")
kcf_frames, kcf_indices, kcf_stats = run_tracking(
    video_path, "KCF", bbox_initial, target_frames
)
print(f"  → FPS: {kcf_stats.get('avg_fps', 0):.1f}, Akurasi: {kcf_stats.get('accuracy', 0):.1f}%")

# Menjalankan tracking dengan CSRT
print("[INFO] Menjalankan tracking CSRT...")
csrt_frames, csrt_indices, csrt_stats = run_tracking(
    video_path, "CSRT", bbox_initial, target_frames
)
print(f"  → FPS: {csrt_stats.get('avg_fps', 0):.1f}, Akurasi: {csrt_stats.get('accuracy', 0):.1f}%")

# ============================================================
# 4. Visualisasi perbandingan KCF vs CSRT
# ============================================================

# Menentukan jumlah frame untuk ditampilkan
n_compare = min(len(kcf_frames), len(csrt_frames), 5)

# Membuat figure 2 baris: baris atas KCF, baris bawah CSRT
fig, axes = plt.subplots(2, n_compare, figsize=(4 * n_compare, 7))

# Memastikan axes 2D
if n_compare == 1:
    axes = axes.reshape(2, 1)

for i in range(n_compare):
    # Baris 1: KCF
    if i < len(kcf_frames):
        rgb_kcf = cv2.cvtColor(kcf_frames[i], cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(rgb_kcf)
        axes[0, i].set_title(f"KCF - Frame {kcf_indices[i]}", fontsize=9)
    axes[0, i].axis("off")

    # Baris 2: CSRT
    if i < len(csrt_frames):
        rgb_csrt = cv2.cvtColor(csrt_frames[i], cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(rgb_csrt)
        axes[1, i].set_title(f"CSRT - Frame {csrt_indices[i]}", fontsize=9)
    axes[1, i].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 9: Perbandingan Tracking KCF vs CSRT",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil perbandingan visual
output_path = os.path.join(OUTPUT_DIR, "09_tracking_kcf_vs_csrt.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[OUTPUT] Perbandingan visual disimpan di: {output_path}")

# ============================================================
# 5. Visualisasi statistik perbandingan
# ============================================================

# Membuat figure untuk bar chart perbandingan
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# Data untuk chart
tracker_names = ["KCF", "CSRT"]
colors_bar = ["#2196F3", "#FF9800"]

# Chart 1: FPS (kecepatan)
fps_vals = [kcf_stats.get("avg_fps", 0), csrt_stats.get("avg_fps", 0)]
bars1 = axes2[0].bar(tracker_names, fps_vals, color=colors_bar, edgecolor="black")
axes2[0].set_title("Kecepatan (FPS)", fontsize=12, fontweight="bold")
axes2[0].set_ylabel("Frame per Second")
# Menambahkan label nilai di atas bar
for bar, val in zip(bars1, fps_vals):
    axes2[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                  f"{val:.1f}", ha="center", fontsize=11)

# Chart 2: Akurasi (%)
acc_vals = [kcf_stats.get("accuracy", 0), csrt_stats.get("accuracy", 0)]
bars2 = axes2[1].bar(tracker_names, acc_vals, color=colors_bar, edgecolor="black")
axes2[1].set_title("Akurasi (%)", fontsize=12, fontweight="bold")
axes2[1].set_ylabel("Persentase (%)")
axes2[1].set_ylim(0, 110)
for bar, val in zip(bars2, acc_vals):
    axes2[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                  f"{val:.1f}%", ha="center", fontsize=11)

# Chart 3: Total waktu (detik)
time_vals = [kcf_stats.get("total_time", 0), csrt_stats.get("total_time", 0)]
bars3 = axes2[2].bar(tracker_names, time_vals, color=colors_bar, edgecolor="black")
axes2[2].set_title("Total Waktu (detik)", fontsize=12, fontweight="bold")
axes2[2].set_ylabel("Detik")
for bar, val in zip(bars3, time_vals):
    axes2[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"{val:.3f}s", ha="center", fontsize=11)

# Menambahkan judul utama
plt.suptitle("Perbandingan Performa: KCF vs CSRT\nKCF lebih cepat, CSRT lebih akurat",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan statistik
output_path2 = os.path.join(OUTPUT_DIR, "09_kcf_vs_csrt_stats.png")
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
plt.show()
print(f"[OUTPUT] Statistik disimpan di: {output_path2}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 9")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.legacy.TrackerKCF_create()  → Membuat KCF tracker")
print("     atau cv2.TrackerKCF_create() untuk OpenCV versi lama")
print("  2. KCF = Kernelized Correlation Filter")
print("     - Sangat cepat (> 100 FPS)")
print("     - Berbasis correlation filter di domain Fourier")
print("     - Kurang baik untuk perubahan skala")
print("  3. Perbandingan KCF vs CSRT:")
print(f"     - KCF  → FPS: {kcf_stats.get('avg_fps', 0):.1f}, Akurasi: {kcf_stats.get('accuracy', 0):.1f}%")
print(f"     - CSRT → FPS: {csrt_stats.get('avg_fps', 0):.1f}, Akurasi: {csrt_stats.get('accuracy', 0):.1f}%")
print("  4. Trade-off kecepatan vs akurasi:")
print("     - Butuh real-time? Gunakan KCF")
print("     - Butuh akurasi tinggi? Gunakan CSRT")
print("=" * 60)
