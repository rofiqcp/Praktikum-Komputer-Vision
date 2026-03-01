"""
==========================================================================
PERCOBAAN 18: MULTI-FACE DETECTION DAN TRACKING CONCEPT
==========================================================================
Program ini mempelajari konsep multi-face detection dan tracking sederhana.
Wajah dideteksi pada serangkaian frame sintetis, lalu dilacak menggunakan
centroid distance untuk mengasosiasikan deteksi antar frame. Program juga
menangani wajah yang muncul dan menghilang.

Konsep yang dipelajari:
- Face detection pada multiple frames (simulasi video)
- Simple tracking: menetapkan ID unik ke setiap wajah
- Centroid-based association: mencocokkan deteksi berdasarkan jarak centroid
- Track management: membuat track baru, memperbarui, dan menghapus
- Handling appearance/disappearance: wajah masuk dan keluar frame
- Statistik deteksi dan tracking sepanjang waktu

Fungsi utama yang dipelajari:
- cv2.CascadeClassifier.detectMultiScale() : Deteksi wajah per frame
- np.linalg.norm()                          : Menghitung jarak centroid
- np.argmin()                               : Mencari jarak minimum
- cv2.rectangle() / cv2.putText()           : Visualisasi tracking
- plt.subplot()                             : Multi-panel visualization

Hasil: Visualisasi tracking sequence dan statistik deteksi
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan deteksi wajah
import cv2

# Mengimpor NumPy untuk operasi array dan perhitungan jarak
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu pemrosesan
import time

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 18: MULTI-FACE DETECTION DAN TRACKING CONCEPT")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Wajah untuk Membuat Frame Sintetis
# ============================================================

print("\n[INFO] Memuat gambar wajah untuk simulasi video...")
print("-" * 50)

# Memuat Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Memeriksa apakah cascade berhasil dimuat
print(f"  Face cascade dimuat: {not face_cascade.empty()}")

# Mendefinisikan ukuran frame sintetis
FRAME_W = 640
FRAME_H = 480

# Menetapkan seed random untuk reprodusibilitas
np.random.seed(42)

# Mendefinisikan jumlah "wajah sintetis" yang bergerak
n_synth_faces = 4

# Mendefinisikan ukuran wajah sintetis
FACE_SIZE = 80

# Mendefinisikan warna untuk setiap wajah sintetis
face_colors_bgr = [
    (200, 170, 150),  # Skin tone 1
    (180, 155, 135),  # Skin tone 2
    (220, 190, 170),  # Skin tone 3
    (190, 165, 145),  # Skin tone 4
]

# Membuat posisi awal acak untuk setiap wajah sintetis
initial_positions = []
for i in range(n_synth_faces):
    # Menghasilkan posisi x dan y acak
    x = np.random.randint(FACE_SIZE, FRAME_W - FACE_SIZE * 2)
    y = np.random.randint(FACE_SIZE, FRAME_H - FACE_SIZE * 2)

    # Menyimpan posisi awal
    initial_positions.append([x, y])

# Mengkonversi ke numpy array
positions = np.array(initial_positions, dtype=float)

# Membuat kecepatan acak untuk pergerakan wajah
velocities = np.random.uniform(-5, 5, size=(n_synth_faces, 2))

# Mendefinisikan jumlah frame yang akan disimulasikan
n_frames = 30

# Mendefinisikan pada frame berapa wajah muncul dan menghilang
# Wajah 0: selalu ada, Wajah 1: frame 5-25, Wajah 2: frame 0-20, Wajah 3: frame 10-30
face_visibility = {
    0: (0, 30),   # Wajah 0: selalu ada (frame 0-30)
    1: (5, 25),   # Wajah 1: muncul frame 5, hilang frame 25
    2: (0, 20),   # Wajah 2: ada sejak awal, hilang frame 20
    3: (10, 30),  # Wajah 3: muncul frame 10, tetap sampai akhir
}

print(f"  Ukuran frame: {FRAME_W}x{FRAME_H}")
print(f"  Jumlah wajah sintetis: {n_synth_faces}")
print(f"  Jumlah frame: {n_frames}")
print(f"  Jadwal visibility:")
for face_id, (start, end) in face_visibility.items():
    print(f"    Wajah {face_id}: frame {start} - {end}")


# ============================================================
# 2. Fungsi Membuat Frame Sintetis dengan Wajah Bergerak
# ============================================================

print("\n[INFO] Membuat frame sintetis dengan wajah bergerak...")
print("-" * 50)


# Mendefinisikan fungsi untuk membuat satu frame sintetis
def buat_frame_sintetis(frame_idx, positions, face_visibility, frame_w, frame_h):
    """
    Membuat frame sintetis dengan wajah-wajah sederhana yang bergerak.
    Wajah digambar sebagai elips dengan fitur wajah sederhana.
    """
    # Membuat frame kosong dengan background abu-abu
    frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 180

    # Menambahkan tekstur noise pada background
    noise = np.random.randint(0, 20, (frame_h, frame_w, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)

    # Membuat list wajah yang visible pada frame ini
    visible_faces = []

    # Menggambar setiap wajah yang visible
    for face_id, (start_frame, end_frame) in face_visibility.items():
        # Memeriksa apakah wajah visible pada frame ini
        if start_frame <= frame_idx < end_frame:
            # Mengambil posisi wajah
            x = int(positions[face_id, 0])
            y = int(positions[face_id, 1])

            # Memastikan posisi dalam batas frame
            x = max(10, min(frame_w - FACE_SIZE - 10, x))
            y = max(10, min(frame_h - FACE_SIZE - 10, y))

            # Mengambil warna kulit untuk wajah ini
            skin_color = face_colors_bgr[face_id]

            # Menghitung pusat dan radius elips wajah
            cx = x + FACE_SIZE // 2
            cy = y + FACE_SIZE // 2
            rx = FACE_SIZE // 2
            ry = int(FACE_SIZE * 0.6)

            # Menggambar elips wajah (kepala)
            cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, skin_color, -1)

            # Menggambar border wajah
            cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360,
                        (100, 80, 60), 2)

            # Menggambar mata kiri
            eye_y = cy - ry // 4
            cv2.circle(frame, (cx - rx // 3, eye_y), 5, (40, 30, 20), -1)

            # Menggambar mata kanan
            cv2.circle(frame, (cx + rx // 3, eye_y), 5, (40, 30, 20), -1)

            # Menggambar mulut (garis melengkung)
            mouth_y = cy + ry // 3
            cv2.ellipse(frame, (cx, mouth_y), (rx // 3, 5), 0, 0, 180,
                        (80, 50, 40), 2)

            # Menyimpan bounding box wajah yang visible
            visible_faces.append({
                "id": face_id,
                "bbox": (x, y, FACE_SIZE, FACE_SIZE),
                "centroid": (cx, cy)
            })

    # Mengembalikan frame dan info wajah visible
    return frame, visible_faces


# ============================================================
# 3. Simple Centroid-Based Tracker
# ============================================================

print("\n[INFO] Mengimplementasikan simple centroid tracker...")
print("-" * 50)


# Mendefinisikan kelas SimpleTracker untuk tracking wajah
class SimpleTracker:
    """
    Tracker sederhana berbasis centroid distance.
    Mencocokkan deteksi baru dengan track yang ada berdasarkan
    jarak centroid terdekat.
    """

    # Menginisialisasi tracker
    def __init__(self, max_distance=100, max_disappeared=5):
        # Menyimpan threshold jarak maksimum untuk asosiasi
        self.max_distance = max_distance

        # Menyimpan jumlah frame maksimum track boleh hilang
        self.max_disappeared = max_disappeared

        # Membuat counter untuk ID track unik
        self.next_id = 0

        # Membuat dictionary untuk menyimpan track aktif
        self.tracks = {}

        # Membuat dictionary untuk menghitung frame tanpa deteksi per track
        self.disappeared = {}

    # Mendefinisikan fungsi untuk memperbarui tracker
    def update(self, detections):
        """
        Memperbarui tracker dengan deteksi baru.
        detections: list of (cx, cy, w, h) centroid dan ukuran.
        """
        # Jika tidak ada deteksi, tambahkan disappeared count
        if len(detections) == 0:
            # Menambah counter disappeared untuk semua track
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1

                # Menghapus track jika sudah terlalu lama hilang
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]

            # Mengembalikan track yang tersisa
            return self.tracks.copy()

        # Mengambil centroid dari deteksi baru
        det_centroids = np.array([(d[0], d[1]) for d in detections])

        # Jika belum ada track, buat track baru untuk setiap deteksi
        if len(self.tracks) == 0:
            for det in detections:
                # Membuat track baru dengan ID unik
                self.tracks[self.next_id] = {
                    "centroid": (det[0], det[1]),
                    "bbox": (det[0] - det[2] // 2, det[1] - det[3] // 2,
                             det[2], det[3])
                }
                self.disappeared[self.next_id] = 0
                self.next_id += 1

            return self.tracks.copy()

        # Mengambil centroid dari track yang ada
        track_ids = list(self.tracks.keys())
        track_centroids = np.array(
            [self.tracks[tid]["centroid"] for tid in track_ids]
        )

        # Menghitung matriks jarak antara track dan deteksi
        n_tracks = len(track_centroids)
        n_dets = len(det_centroids)
        dist_matrix = np.zeros((n_tracks, n_dets))

        # Mengisi matriks jarak
        for i in range(n_tracks):
            for j in range(n_dets):
                # Menghitung jarak Euclidean antara track dan deteksi
                dist_matrix[i, j] = np.linalg.norm(
                    track_centroids[i] - det_centroids[j]
                )

        # Mencocokkan track dengan deteksi (greedy matching)
        matched_tracks = set()
        matched_dets = set()

        # Mendapatkan urutan pasangan berdasarkan jarak terkecil
        flat_indices = np.argsort(dist_matrix.flatten())

        # Melakukan matching greedy
        for flat_idx in flat_indices:
            # Mengkonversi flat index ke (row, col)
            t_idx = flat_idx // n_dets
            d_idx = flat_idx % n_dets

            # Melewatkan jika sudah di-match
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue

            # Melewatkan jika jarak terlalu jauh
            if dist_matrix[t_idx, d_idx] > self.max_distance:
                continue

            # Memperbarui track dengan deteksi yang cocok
            tid = track_ids[t_idx]
            det = detections[d_idx]
            self.tracks[tid]["centroid"] = (det[0], det[1])
            self.tracks[tid]["bbox"] = (det[0] - det[2] // 2,
                                        det[1] - det[3] // 2,
                                        det[2], det[3])
            self.disappeared[tid] = 0

            # Menandai sebagai sudah di-match
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        # Menangani track yang tidak ter-match (menambah disappeared)
        for t_idx in range(n_tracks):
            if t_idx not in matched_tracks:
                tid = track_ids[t_idx]
                self.disappeared[tid] += 1

                # Menghapus track yang terlalu lama hilang
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]

        # Menangani deteksi yang tidak ter-match (membuat track baru)
        for d_idx in range(n_dets):
            if d_idx not in matched_dets:
                # Membuat track baru
                det = detections[d_idx]
                self.tracks[self.next_id] = {
                    "centroid": (det[0], det[1]),
                    "bbox": (det[0] - det[2] // 2, det[1] - det[3] // 2,
                             det[2], det[3])
                }
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        # Mengembalikan track yang diperbarui
        return self.tracks.copy()


# Membuat instance tracker
tracker = SimpleTracker(max_distance=80, max_disappeared=3)

print(f"  Tracker dibuat: max_distance=80, max_disappeared=3")

# ============================================================
# 4. Menjalankan Tracking pada Semua Frame
# ============================================================

print("\n[INFO] Menjalankan tracking pada semua frame...")
print("-" * 50)

# Membuat list untuk menyimpan hasil per frame
frame_results = []
tracking_stats = {
    "frame_idx": [],
    "n_detected": [],
    "n_tracked": [],
    "n_visible": []
}

# Mendefinisikan warna untuk track ID
track_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)
]

# Mencatat waktu mulai
start_time = time.time()

# Memproses setiap frame
for frame_idx in range(n_frames):
    # Memperbarui posisi wajah dengan velocity
    positions += velocities

    # Memantulkan wajah jika keluar batas frame
    for i in range(n_synth_faces):
        # Memantulkan di batas horizontal
        if positions[i, 0] < 10 or positions[i, 0] > FRAME_W - FACE_SIZE - 10:
            velocities[i, 0] *= -1
            positions[i, 0] = np.clip(positions[i, 0], 10, FRAME_W - FACE_SIZE - 10)

        # Memantulkan di batas vertikal
        if positions[i, 1] < 10 or positions[i, 1] > FRAME_H - FACE_SIZE - 10:
            velocities[i, 1] *= -1
            positions[i, 1] = np.clip(positions[i, 1], 10, FRAME_H - FACE_SIZE - 10)

    # Membuat frame sintetis
    frame, visible_faces = buat_frame_sintetis(
        frame_idx, positions, face_visibility, FRAME_W, FRAME_H
    )

    # Mengkonversi ke grayscale untuk deteksi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah menggunakan Haar Cascade
    detections_raw = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )

    # Mengkonversi deteksi ke format centroid
    det_list = []
    if len(detections_raw) > 0:
        for (x, y, w, h) in detections_raw:
            # Menghitung centroid
            cx = x + w // 2
            cy = y + h // 2
            det_list.append((cx, cy, w, h))

    # Jika Haar cascade tidak mendeteksi, gunakan ground truth
    if len(det_list) == 0 and len(visible_faces) > 0:
        for vf in visible_faces:
            # Menggunakan centroid dari wajah sintetis sebagai deteksi
            cx, cy = vf["centroid"]
            det_list.append((cx, cy, FACE_SIZE, FACE_SIZE))

    # Memperbarui tracker dengan deteksi baru
    active_tracks = tracker.update(det_list)

    # Menggambar tracking pada frame
    frame_viz = frame.copy()

    # Menggambar bounding box dan ID untuk setiap track
    for track_id, track_info in active_tracks.items():
        # Mengambil bounding box
        bx, by, bw, bh = track_info["bbox"]

        # Memilih warna berdasarkan track ID
        color = track_colors[track_id % len(track_colors)]

        # Menggambar bounding box
        cv2.rectangle(frame_viz, (int(bx), int(by)),
                      (int(bx + bw), int(by + bh)), color, 2)

        # Menambahkan label ID
        label = f"ID:{track_id}"
        cv2.putText(frame_viz, label, (int(bx), int(by) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Menyimpan hasil frame
    frame_results.append({
        "frame_idx": frame_idx,
        "frame_viz": frame_viz,
        "n_detected": len(det_list),
        "n_tracked": len(active_tracks),
        "n_visible": len(visible_faces),
        "tracks": active_tracks.copy()
    })

    # Menyimpan statistik
    tracking_stats["frame_idx"].append(frame_idx)
    tracking_stats["n_detected"].append(len(det_list))
    tracking_stats["n_tracked"].append(len(active_tracks))
    tracking_stats["n_visible"].append(len(visible_faces))

# Menghitung total waktu pemrosesan
total_time = time.time() - start_time

# Menghitung FPS
fps = n_frames / total_time

# Menampilkan hasil tracking
print(f"  Total frame diproses: {n_frames}")
print(f"  Waktu total: {total_time:.3f} detik")
print(f"  FPS: {fps:.1f}")
print(f"  Total ID unik yang dibuat: {tracker.next_id}")

# ============================================================
# 5. Visualisasi 1: Frame Sequence dengan Tracking
# ============================================================

print("\n[INFO] Membuat visualisasi frame sequence...")
print("-" * 50)

# Memilih 8 frame untuk ditampilkan (sampel merata)
sample_indices = np.linspace(0, n_frames - 1, 8, dtype=int)

# Membuat figure dengan 2x4 subplot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Meratakan array axes
axes_flat = axes.flatten()

# Menampilkan setiap frame sampel
for plot_idx, frame_idx in enumerate(sample_indices):
    # Mengambil data frame
    result = frame_results[frame_idx]

    # Mengkonversi BGR ke RGB untuk matplotlib
    frame_rgb = cv2.cvtColor(result["frame_viz"], cv2.COLOR_BGR2RGB)

    # Menampilkan frame
    axes_flat[plot_idx].imshow(frame_rgb)

    # Mengatur judul dengan informasi tracking
    axes_flat[plot_idx].set_title(
        f"Frame {frame_idx}\n"
        f"Deteksi: {result['n_detected']}, "
        f"Track: {result['n_tracked']}",
        fontsize=9, fontweight='bold'
    )

    # Menyembunyikan ticks
    axes_flat[plot_idx].set_xticks([])
    axes_flat[plot_idx].set_yticks([])

# Menambahkan judul utama
plt.suptitle("Percobaan 18: Multi-Face Tracking - Sequence Frame",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi frame sequence
output_path_1 = os.path.join(OUTPUT_DIR, "18_tracking_frame_sequence.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 6. Visualisasi 2: Statistik Tracking
# ============================================================

print("\n[INFO] Membuat visualisasi statistik tracking...")
print("-" * 50)

# Membuat figure dengan 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Panel kiri atas: Jumlah deteksi dan track per frame ---
# Menggambar garis jumlah wajah visible (ground truth)
axes[0, 0].plot(tracking_stats["frame_idx"], tracking_stats["n_visible"],
                'g-', linewidth=2, label='Visible (GT)', marker='o', markersize=3)

# Menggambar garis jumlah deteksi
axes[0, 0].plot(tracking_stats["frame_idx"], tracking_stats["n_detected"],
                'b--', linewidth=2, label='Detected', marker='s', markersize=3)

# Menggambar garis jumlah track aktif
axes[0, 0].plot(tracking_stats["frame_idx"], tracking_stats["n_tracked"],
                'r:', linewidth=2, label='Tracked', marker='^', markersize=3)

# Mengatur sumbu dan label
axes[0, 0].set_xlabel("Frame", fontsize=10)
axes[0, 0].set_ylabel("Jumlah Wajah", fontsize=10)
axes[0, 0].set_title("Jumlah Wajah per Frame", fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(-0.5, n_frames - 0.5)

# --- Panel kanan atas: Timeline visibility wajah ---
# Menggambar timeline untuk setiap wajah
for face_id, (start, end) in face_visibility.items():
    # Menggambar bar horizontal untuk visibility
    axes[0, 1].barh(face_id, end - start, left=start, height=0.5,
                    color=plt.cm.Set2(face_id), edgecolor='black', linewidth=0.5,
                    label=f'Wajah {face_id}')

    # Menambahkan label range
    axes[0, 1].text((start + end) / 2, face_id,
                    f"Frame {start}-{end}", ha='center', va='center',
                    fontsize=9, fontweight='bold')

# Mengatur sumbu dan label
axes[0, 1].set_xlabel("Frame", fontsize=10)
axes[0, 1].set_ylabel("Face ID (GT)", fontsize=10)
axes[0, 1].set_title("Timeline Visibility Wajah", fontsize=11, fontweight='bold')
axes[0, 1].set_yticks(range(n_synth_faces))
axes[0, 1].set_yticklabels([f'Wajah {i}' for i in range(n_synth_faces)])
axes[0, 1].grid(axis='x', alpha=0.3)
axes[0, 1].set_xlim(-0.5, n_frames + 0.5)

# --- Panel kiri bawah: Kumulatif ID unik ---
# Menghitung kumulatif ID unik yang terbuat per frame
unique_ids_over_time = []
seen_ids = set()

# Menghitung ID unik kumulatif
for result in frame_results:
    for tid in result["tracks"]:
        seen_ids.add(tid)
    unique_ids_over_time.append(len(seen_ids))

# Menggambar grafik kumulatif ID unik
axes[1, 0].plot(range(n_frames), unique_ids_over_time, 'b-o', linewidth=2,
                markersize=4, label='Kumulatif ID Unik')

# Menambahkan garis referensi jumlah wajah sebenarnya
axes[1, 0].axhline(y=n_synth_faces, color='red', linestyle='--', linewidth=1.5,
                    label=f'Total wajah sebenarnya ({n_synth_faces})')

# Mengatur sumbu dan label
axes[1, 0].set_xlabel("Frame", fontsize=10)
axes[1, 0].set_ylabel("Jumlah ID Unik", fontsize=10)
axes[1, 0].set_title("Kumulatif Track ID Unik", fontsize=11, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(-0.5, n_frames - 0.5)

# --- Panel kanan bawah: Tabel ringkasan ---
# Menyembunyikan sumbu
axes[1, 1].axis('off')

# Membuat data tabel ringkasan
summary_data = [
    ["Total Frame", str(n_frames)],
    ["Total Wajah Sintetis", str(n_synth_faces)],
    ["Total Track ID Dibuat", str(tracker.next_id)],
    ["Rata-rata Deteksi/Frame", f"{np.mean(tracking_stats['n_detected']):.1f}"],
    ["Rata-rata Track/Frame", f"{np.mean(tracking_stats['n_tracked']):.1f}"],
    ["Max Deteksi pada 1 Frame", str(max(tracking_stats['n_detected']))],
    ["Max Track pada 1 Frame", str(max(tracking_stats['n_tracked']))],
    ["Waktu Total", f"{total_time:.3f} detik"],
    ["FPS", f"{fps:.1f}"],
]

# Membuat tabel
table = axes[1, 1].table(cellText=summary_data,
                          colLabels=["Parameter", "Nilai"],
                          cellLoc='center', loc='center',
                          colWidths=[0.55, 0.35])

# Mengatur ukuran font dan skala
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.5)

# Mengatur warna header
for j in range(2):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Mengatur warna baris data
for i in range(1, len(summary_data) + 1):
    for j in range(2):
        if i % 2 == 0:
            table[i, j].set_facecolor('#D6E4F0')
        else:
            table[i, j].set_facecolor('#EDF2F9')

# Menambahkan judul tabel
axes[1, 1].set_title("Ringkasan Statistik Tracking", fontsize=11, fontweight='bold')

# Menambahkan judul utama
plt.suptitle("Percobaan 18: Statistik Multi-Face Tracking",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi statistik
output_path_2 = os.path.join(OUTPUT_DIR, "18_tracking_statistics.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 18")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Multi-Face Detection:")
print("     - Haar Cascade pada setiap frame simulasi")
print("     - detectMultiScale() untuk mendeteksi semua wajah")
print("  2. Simple Centroid Tracker:")
print("     - Menghitung centroid setiap deteksi")
print("     - Mencocokkan deteksi baru dengan track lama")
print("     - Menggunakan jarak Euclidean terdekat (greedy)")
print("     - Membuat track baru jika tidak ada yang cocok")
print("     - Menghapus track jika hilang terlalu lama")
print("  3. Track Management:")
print(f"     - Total ID unik dibuat: {tracker.next_id}")
print(f"     - Wajah sintetis sebenarnya: {n_synth_faces}")
print("  4. Handling Appear/Disappear:")
for face_id, (start, end) in face_visibility.items():
    print(f"     - Wajah {face_id}: frame {start}-{end} "
          f"({end - start} frame)")
print(f"\nPerforma: {fps:.1f} FPS ({total_time:.3f} detik untuk {n_frames} frame)")
print("=" * 60)
