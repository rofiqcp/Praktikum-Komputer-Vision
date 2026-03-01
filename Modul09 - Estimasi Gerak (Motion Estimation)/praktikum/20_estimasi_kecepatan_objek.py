"""
==========================================================================
PERCOBAAN 20: ESTIMASI KECEPATAN OBJEK DARI OPTICAL FLOW
==========================================================================
Program ini mempelajari cara mengestimasi kecepatan objek bergerak
menggunakan informasi optical flow. Dengan menghitung magnitude flow
pada area objek, kita dapat mengetahui kecepatan gerak objek dalam
satuan piksel/frame dan mengkonversinya ke satuan yang lebih bermakna.

Pipeline Estimasi Kecepatan:
1. Hitung dense optical flow (Farneback)
2. Threshold magnitude → deteksi area bergerak
3. Temukan kontur objek bergerak
4. Hitung rata-rata vektor kecepatan per objek
5. Konversi ke kecepatan perkiraan (dengan asumsi skala)

Fungsi utama yang dipelajari:
- cv2.calcOpticalFlowFarneback()  : Dense optical flow
- cv2.cartToPolar()               : Konversi vektor ke magnitude+arah
- cv2.findContours()              : Deteksi kontur objek
- cv2.arrowedLine()               : Menggambar vektor kecepatan
- cv2.moments()                   : Menghitung centroid objek

Hasil: Visualisasi vektor kecepatan pada setiap objek bergerak
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan video
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file
import os

# Mengimpor matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input dan output
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 20: ESTIMASI KECEPATAN OBJEK DARI OPTICAL FLOW")
print("=" * 60)

# ============================================================
# 1. Membuka video
# ============================================================

# Menggunakan video multi-objek untuk estimasi kecepatan beberapa objek
video_path = os.path.join(IMAGE_DIR, "video_multi_objek.avi")
cap = cv2.VideoCapture(video_path)

# Memeriksa apakah video berhasil dibuka
if not cap.isOpened():
    print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Membaca properti video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"[INFO] Video: {width}x{height}, FPS={fps}, Total={total_frames} frame")

# ============================================================
# 2. Asumsi skala untuk konversi kecepatan
# ============================================================

# Asumsi: lebar video merepresentasikan area tertentu di dunia nyata
# Ini adalah ESTIMASI dan bergantung pada kalibrasi kamera
ASSUMED_REAL_WIDTH_M = 5.0   # Asumsi lebar scene = 5 meter
PIXEL_TO_METER = ASSUMED_REAL_WIDTH_M / width  # Konversi piksel ke meter

print(f"[INFO] Asumsi skala: {width} piksel = {ASSUMED_REAL_WIDTH_M} meter")
print(f"[INFO] 1 piksel = {PIXEL_TO_METER:.4f} meter")
print(f"[INFO] FPS video = {fps}")

# ============================================================
# 3. Parameter deteksi
# ============================================================

# Threshold magnitude optical flow untuk mendeteksi gerakan
MAG_THRESHOLD = 1.5  # piksel/frame

# Area minimum kontur untuk dianggap sebagai objek
MIN_CONTOUR_AREA = 300

# Kernel untuk operasi morfologi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ============================================================
# 4. Memproses video frame per frame
# ============================================================

print("[INFO] Memproses video untuk estimasi kecepatan...")

# Membaca frame pertama
ret, prev_frame = cap.read()
if not ret:
    print("[ERROR] Gagal membaca frame pertama.")
    exit()

# Mengkonversi ke grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Menyimpan data untuk setiap frame
all_speeds = []           # Kecepatan rata-rata semua objek per frame
all_object_count = []     # Jumlah objek per frame

# Menyimpan frame untuk visualisasi
vis_data = []
target_vis = [15, 30, 60, 90]

frame_count = 0

while True:
    # Membaca frame berikutnya
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Mengkonversi ke grayscale
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ============================================================
    # Langkah 4a: Menghitung dense optical flow
    # ============================================================
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # Memisahkan komponen flow
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    # Menghitung magnitude dan arah
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)

    # ============================================================
    # Langkah 4b: Threshold magnitude untuk area bergerak
    # ============================================================

    # Membuat mask area bergerak
    motion_mask = (magnitude > MAG_THRESHOLD).astype(np.uint8) * 255

    # Operasi morfologi untuk membersihkan mask
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    # ============================================================
    # Langkah 4c: Menemukan kontur objek bergerak
    # ============================================================
    contours, _ = cv2.findContours(
        motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Membuat salinan frame untuk menggambar
    vis_frame = frame.copy()

    # Counter objek dan penyimpan kecepatan per frame
    obj_count = 0
    frame_speeds_px = []   # Kecepatan dalam piksel/frame
    frame_speeds_ms = []   # Kecepatan dalam m/s

    for contour in contours:
        # Menghitung area kontur
        area = cv2.contourArea(contour)

        # Memfilter kontur kecil
        if area < MIN_CONTOUR_AREA:
            continue

        obj_count += 1

        # Mendapatkan bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # ============================================================
        # Langkah 4d: Menghitung kecepatan rata-rata objek
        # Mengambil flow hanya di dalam area kontur
        # ============================================================

        # Membuat mask untuk kontur ini saja
        obj_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(obj_mask, [contour], -1, 255, -1)

        # Mengambil flow_x dan flow_y di dalam area objek
        obj_flow_x = flow_x[obj_mask > 0]
        obj_flow_y = flow_y[obj_mask > 0]
        obj_magnitude = magnitude[obj_mask > 0]
        obj_angle = angle[obj_mask > 0]

        # Menghitung rata-rata kecepatan (piksel/frame)
        avg_speed_px = obj_magnitude.mean()
        avg_vx = obj_flow_x.mean()
        avg_vy = obj_flow_y.mean()
        avg_direction = obj_angle.mean()

        # Mengkonversi ke meter/detik
        # speed_m/s = speed_px/frame * PIXEL_TO_METER * FPS
        avg_speed_ms = avg_speed_px * PIXEL_TO_METER * fps

        frame_speeds_px.append(avg_speed_px)
        frame_speeds_ms.append(avg_speed_ms)

        # ============================================================
        # Langkah 4e: Visualisasi — bounding box dan vektor kecepatan
        # ============================================================

        # Menggambar bounding box (hijau)
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Menghitung centroid objek
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        # Menggambar vektor kecepatan menggunakan arrowedLine
        # Panjang panah proporsional dengan kecepatan
        arrow_scale = 5  # Faktor skala untuk visualisasi
        end_x = int(cx + avg_vx * arrow_scale)
        end_y = int(cy + avg_vy * arrow_scale)
        cv2.arrowedLine(vis_frame, (cx, cy), (end_x, end_y),
                        (0, 0, 255), 2, tipLength=0.3)

        # Menambahkan label kecepatan
        speed_label = f"Obj{obj_count}: {avg_speed_px:.1f} px/f"
        cv2.putText(vis_frame, speed_label, (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        speed_label_ms = f"~{avg_speed_ms:.2f} m/s"
        cv2.putText(vis_frame, speed_label_ms, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Menggambar titik centroid
        cv2.circle(vis_frame, (cx, cy), 4, (255, 0, 0), -1)

    # Menambahkan info frame
    cv2.putText(vis_frame, f"Frame {frame_count} | Objek: {obj_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Menyimpan statistik
    all_object_count.append(obj_count)
    if frame_speeds_px:
        all_speeds.append(np.mean(frame_speeds_px))
    else:
        all_speeds.append(0)

    # Menyimpan snapshot
    if frame_count in target_vis:
        vis_data.append((frame_count, frame.copy(), motion_mask.copy(),
                        vis_frame.copy(), magnitude.copy()))

    # Memperbarui frame sebelumnya
    prev_gray = curr_gray.copy()

# Menutup video capture
cap.release()

print(f"[INFO] Selesai memproses {frame_count} frame.")

# ============================================================
# 5. Visualisasi hasil estimasi kecepatan
# ============================================================

# Figure 1: Pipeline deteksi dan estimasi kecepatan
num_vis = min(4, len(vis_data))
fig1, axes1 = plt.subplots(3, num_vis, figsize=(4.5 * num_vis, 12))

if num_vis == 1:
    axes1 = axes1.reshape(3, 1)

for col in range(num_vis):
    fnum, orig, mask, detected, mag = vis_data[col]

    # Baris 1: Frame asli
    axes1[0, col].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes1[0, col].set_title(f"Frame #{fnum}", fontsize=10)
    axes1[0, col].axis("off")

    # Baris 2: Magnitude flow
    im = axes1[1, col].imshow(mag, cmap='hot', vmin=0, vmax=mag.max())
    axes1[1, col].set_title(f"Flow Magnitude #{fnum}", fontsize=10)
    axes1[1, col].axis("off")

    # Baris 3: Deteksi dengan vektor kecepatan
    axes1[2, col].imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
    axes1[2, col].set_title(f"Estimasi Kecepatan #{fnum}", fontsize=10)
    axes1[2, col].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 20: Estimasi Kecepatan Objek dari Optical Flow\n"
             "Panah merah = vektor kecepatan, Label = kecepatan",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan figure pertama
output_path_1 = os.path.join(OUTPUT_DIR, "20_estimasi_kecepatan.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Estimasi kecepatan disimpan di: {output_path_1}")

# ============================================================
# 6. Grafik kecepatan dan jumlah objek
# ============================================================

# Membuat figure: grafik statistik
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Kecepatan rata-rata per frame
axes2[0, 0].plot(all_speeds, color='red', linewidth=1.5, alpha=0.7)
axes2[0, 0].fill_between(range(len(all_speeds)), all_speeds, alpha=0.2, color='red')
axes2[0, 0].set_xlabel("Nomor Frame", fontsize=10)
axes2[0, 0].set_ylabel("Kecepatan (piksel/frame)", fontsize=10)
axes2[0, 0].set_title("Kecepatan Rata-rata Objek per Frame", fontsize=11)
axes2[0, 0].grid(True, alpha=0.3)

# Subplot 2: Jumlah objek per frame
axes2[0, 1].plot(all_object_count, color='steelblue', linewidth=1.5, alpha=0.7)
axes2[0, 1].set_xlabel("Nomor Frame", fontsize=10)
axes2[0, 1].set_ylabel("Jumlah Objek", fontsize=10)
axes2[0, 1].set_title("Jumlah Objek Terdeteksi per Frame", fontsize=11)
axes2[0, 1].grid(True, alpha=0.3)

# Subplot 3: Histogram kecepatan
nonzero_speeds = [s for s in all_speeds if s > 0]
if nonzero_speeds:
    axes2[1, 0].hist(nonzero_speeds, bins=30, color='coral', alpha=0.7, edgecolor='darkred')
axes2[1, 0].set_xlabel("Kecepatan (piksel/frame)", fontsize=10)
axes2[1, 0].set_ylabel("Jumlah Frame", fontsize=10)
axes2[1, 0].set_title("Distribusi Kecepatan Objek", fontsize=11)
axes2[1, 0].grid(True, alpha=0.3)

# Subplot 4: Kecepatan dalam m/s (konversi)
speeds_ms = [s * PIXEL_TO_METER * fps for s in all_speeds]
axes2[1, 1].plot(speeds_ms, color='green', linewidth=1.5, alpha=0.7)
axes2[1, 1].fill_between(range(len(speeds_ms)), speeds_ms, alpha=0.2, color='green')
axes2[1, 1].set_xlabel("Nomor Frame", fontsize=10)
axes2[1, 1].set_ylabel("Kecepatan Estimasi (m/s)", fontsize=10)
axes2[1, 1].set_title(f"Estimasi Kecepatan (asumsi: {ASSUMED_REAL_WIDTH_M}m lebar scene)", fontsize=10)
axes2[1, 1].grid(True, alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 20: Statistik Kecepatan Objek",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan figure kedua
output_path_2 = os.path.join(OUTPUT_DIR, "20_statistik_kecepatan.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Statistik kecepatan disimpan di: {output_path_2}")

# Menampilkan statistik akhir
print(f"\n[INFO] ============ STATISTIK KECEPATAN ============")
print(f"  Kecepatan rata-rata (px/frame) : {np.mean(all_speeds):.2f}")
print(f"  Kecepatan maksimum (px/frame)  : {np.max(all_speeds):.2f}")
print(f"  Kecepatan rata-rata (m/s)      : {np.mean(speeds_ms):.3f}")
print(f"  Kecepatan maksimum (m/s)       : {np.max(speeds_ms):.3f}")
print(f"  Rata-rata objek per frame      : {np.mean(all_object_count):.1f}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 20")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.calcOpticalFlowFarneback()  → Dense optical flow")
print("  2. cv2.cartToPolar()               → Hitung magnitude & arah")
print("  3. cv2.findContours()              → Deteksi kontur objek bergerak")
print("  4. cv2.arrowedLine()               → Gambar vektor kecepatan")
print("  5. cv2.moments()                   → Hitung centroid objek")
print("Pipeline Estimasi Kecepatan:")
print("  1. Hitung dense optical flow antar frame")
print("  2. Threshold magnitude → mask area bergerak")
print("  3. findContours → identifikasi objek")
print("  4. Rata-rata flow di area objek → kecepatan (px/frame)")
print("  5. Konversi: speed_m/s = speed_px * scale * FPS")
print("Catatan:")
print("  - Konversi px → m/s butuh KALIBRASI kamera")
print("  - Asumsi skala memberikan estimasi kasar")
print("  - Untuk akurasi tinggi, gunakan referensi ukuran nyata")
print("=" * 60)
