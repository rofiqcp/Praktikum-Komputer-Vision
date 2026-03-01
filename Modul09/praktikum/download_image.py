"""
==========================================================================
SCRIPT DOWNLOAD DAN GENERATE GAMBAR/VIDEO SAMPLE
Modul 09 - Estimasi Gerak (Motion Estimation)
==========================================================================
Script ini menyiapkan semua gambar dan video sintetis yang dibutuhkan
untuk 20 percobaan Motion Estimation.
- Membuat folder 'image/' dan 'output/'
- Men-generate gambar sintetis dan video dummy untuk latihan
- Membuat sekuens frame untuk simulasi optical flow & tracking

Jalankan script ini PERTAMA KALI sebelum menjalankan percobaan lainnya.
==========================================================================
"""

# Mengimpor library os untuk operasi file dan folder
import os

# Mengimpor library numpy untuk operasi array/matriks
import numpy as np

# Mengimpor library OpenCV untuk pemrosesan gambar & video
import cv2

# Mengimpor math untuk operasi trigonometri
import math

# ============================================================
# LANGKAH 1: Membuat struktur folder yang dibutuhkan
# ============================================================

# Mendapatkan path direktori tempat script ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder image dan output
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Membuat folder 'image' jika belum ada
os.makedirs(IMAGE_DIR, exist_ok=True)

# Membuat folder 'output' jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Folder 'image/' dan 'output/' siap.")

# ============================================================
# LANGKAH 2: Generate video sintetis dengan objek bergerak
# ============================================================

def buat_video_bola_bergerak(filename="video_bola.avi", width=640, height=480, fps=30, durasi=5):
    """
    Membuat video sintetis dengan bola yang bergerak secara diagonal.
    Video ini digunakan untuk latihan optical flow dan tracking.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    # Codec XVID untuk format AVI
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    total_frames = fps * durasi
    # Posisi awal bola
    x, y = 100, 100
    # Kecepatan bola (piksel per frame)
    vx, vy = 3, 2
    # Radius bola
    radius = 30

    for i in range(total_frames):
        # Membuat frame dengan background biru tua
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (40, 30, 20)

        # Menambahkan grid sebagai tekstur background agar optical flow bekerja
        for gx in range(0, width, 50):
            cv2.line(frame, (gx, 0), (gx, height), (60, 50, 40), 1)
        for gy in range(0, height, 50):
            cv2.line(frame, (0, gy), (width, gy), (60, 50, 40), 1)

        # Mengupdate posisi bola
        x += vx
        y += vy

        # Memantulkan bola saat menyentuh tepi
        if x - radius <= 0 or x + radius >= width:
            vx = -vx
        if y - radius <= 0 or y + radius >= height:
            vy = -vy

        # Menggambar bola berwarna merah
        cv2.circle(frame, (int(x), int(y)), radius, (0, 0, 255), -1)
        # Menambahkan highlight pada bola
        cv2.circle(frame, (int(x) - 8, int(y) - 8), 8, (100, 100, 255), -1)

        # Menulis frame ke video
        out.write(frame)

    out.release()
    print(f"[OK] Video '{filename}' berhasil dibuat ({total_frames} frame, {durasi}s).")
    return filepath


def buat_video_multi_objek(filename="video_multi_objek.avi", width=640, height=480, fps=30, durasi=5):
    """
    Membuat video dengan beberapa objek bergerak secara independen.
    Digunakan untuk multi-object tracking dan background subtraction.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    total_frames = fps * durasi
    # Mendefinisikan beberapa objek: [x, y, vx, vy, radius, warna]
    objek_list = [
        [100, 100, 4, 2, 25, (0, 0, 255)],    # Merah
        [400, 300, -3, 3, 20, (0, 255, 0)],    # Hijau
        [300, 200, 2, -4, 30, (255, 0, 0)],    # Biru
        [500, 100, -2, 2, 22, (0, 255, 255)],  # Kuning
    ]

    for i in range(total_frames):
        # Background statis (pemandangan sederhana)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200

        # Menambahkan lantai dan langit
        frame[0:height//2, :] = (230, 200, 180)  # Langit
        frame[height//2:, :] = (100, 160, 100)    # Rumput

        # Menambahkan pola agar tracking dan optical flow bekerja baik
        for gx in range(0, width, 80):
            cv2.line(frame, (gx, 0), (gx, height), (180, 180, 180), 1)

        # Mengupdate dan menggambar setiap objek
        for obj in objek_list:
            obj[0] += obj[2]
            obj[1] += obj[3]
            # Pantulkan saat menyentuh tepi
            if obj[0] - obj[4] <= 0 or obj[0] + obj[4] >= width:
                obj[2] = -obj[2]
            if obj[1] - obj[4] <= 0 or obj[1] + obj[4] >= height:
                obj[3] = -obj[3]
            # Menggambar objek sebagai lingkaran
            cv2.circle(frame, (int(obj[0]), int(obj[1])), obj[4], obj[5], -1)

        out.write(frame)

    out.release()
    print(f"[OK] Video '{filename}' berhasil dibuat ({total_frames} frame, {durasi}s).")
    return filepath


def buat_video_orang_berjalan(filename="video_orang.avi", width=640, height=480, fps=30, durasi=5):
    """
    Membuat video sintetis dengan objek mirip orang berjalan.
    Untuk background subtraction dan motion detection.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    total_frames = fps * durasi

    for i in range(total_frames):
        # Background statis (ruangan)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 220
        # Lantai
        cv2.rectangle(frame, (0, 350), (width, height), (180, 170, 160), -1)
        # Dinding pattern
        for wx in range(0, width, 100):
            cv2.rectangle(frame, (wx, 0), (wx+2, 350), (200, 200, 200), -1)

        # Orang 1 berjalan dari kiri ke kanan
        px1 = int((i * 3) % (width + 100)) - 50
        # Kepala
        cv2.circle(frame, (px1, 250), 20, (150, 130, 120), -1)
        # Badan
        cv2.rectangle(frame, (px1-15, 270), (px1+15, 340), (80, 80, 180), -1)
        # Kaki (animasi sederhana)
        leg_offset = int(10 * math.sin(i * 0.3))
        cv2.line(frame, (px1-5, 340), (px1-5+leg_offset, 380), (60, 60, 60), 4)
        cv2.line(frame, (px1+5, 340), (px1+5-leg_offset, 380), (60, 60, 60), 4)

        # Orang 2 berjalan dari kanan ke kiri (lebih lambat)
        px2 = width - int((i * 2) % (width + 100)) + 50
        cv2.circle(frame, (px2, 260), 18, (130, 120, 110), -1)
        cv2.rectangle(frame, (px2-12, 278), (px2+12, 340), (180, 80, 80), -1)
        leg_offset2 = int(8 * math.sin(i * 0.25))
        cv2.line(frame, (px2-4, 340), (px2-4+leg_offset2, 375), (60, 60, 60), 3)
        cv2.line(frame, (px2+4, 340), (px2+4-leg_offset2, 375), (60, 60, 60), 3)

        out.write(frame)

    out.release()
    print(f"[OK] Video '{filename}' berhasil dibuat ({total_frames} frame, {durasi}s).")
    return filepath


def buat_video_panning(filename="video_panning.avi", width=640, height=480, fps=30, durasi=4):
    """
    Membuat video simulasi kamera panning (bergerak horizontal).
    Untuk video stabilization dan global motion estimation.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    # Membuat panorama lebar (2x width)
    panorama_w = width * 3
    panorama = np.ones((height, panorama_w, 3), dtype=np.uint8) * 200

    # Menggambar pemandangan di panorama
    # Langit
    panorama[0:height//2, :] = (230, 200, 160)
    # Tanah
    panorama[height//2:, :] = (80, 140, 80)

    # Menambahkan objek-objek statis ke panorama
    for tx in range(0, panorama_w, 200):
        # Pohon
        cv2.rectangle(panorama, (tx+80, 200), (tx+100, 350), (40, 80, 40), -1)
        cv2.circle(panorama, (tx+90, 180), 50, (30, 120, 30), -1)
    for tx in range(0, panorama_w, 300):
        # Rumah
        cv2.rectangle(panorama, (tx+130, 250), (tx+230, 350), (60, 60, 180), -1)
        pts = np.array([[tx+120, 250], [tx+180, 190], [tx+240, 250]], np.int32)
        cv2.fillPoly(panorama, [pts], (50, 50, 150))

    total_frames = fps * durasi
    for i in range(total_frames):
        # Simulasi kamera panning + sedikit goyang (shake)
        offset_x = int(i * (panorama_w - width) / total_frames)
        shake_x = int(3 * math.sin(i * 0.5))
        shake_y = int(2 * math.cos(i * 0.7))

        # Crop dari panorama sesuai posisi kamera
        x_start = max(0, min(offset_x + shake_x, panorama_w - width))
        y_start = max(0, min(shake_y, 0))

        frame = panorama[0:height, x_start:x_start+width].copy()
        out.write(frame)

    out.release()
    print(f"[OK] Video '{filename}' berhasil dibuat ({total_frames} frame, {durasi}s).")
    return filepath


def buat_frame_pair(filename1="frame_t0.png", filename2="frame_t1.png", width=640, height=480):
    """
    Membuat sepasang frame untuk percobaan optical flow statis.
    Frame kedua memiliki objek yang sedikit bergeser.
    """
    # Frame 1: objek di posisi awal
    frame1 = np.ones((height, width, 3), dtype=np.uint8) * 180
    # Menambahkan tekstur (kotak-kotak)
    for i in range(0, height, 40):
        for j in range(0, width, 40):
            if (i // 40 + j // 40) % 2 == 0:
                cv2.rectangle(frame1, (j, i), (j+40, i+40), (160, 160, 160), -1)

    # Menggambar objek
    cv2.rectangle(frame1, (200, 150), (350, 300), (0, 0, 200), -1)
    cv2.circle(frame1, (480, 240), 60, (200, 0, 0), -1)

    # Frame 2: objek bergeser
    frame2 = np.ones((height, width, 3), dtype=np.uint8) * 180
    for i in range(0, height, 40):
        for j in range(0, width, 40):
            if (i // 40 + j // 40) % 2 == 0:
                cv2.rectangle(frame2, (j, i), (j+40, i+40), (160, 160, 160), -1)

    # Objek bergeser 20px ke kanan dan 10px ke bawah
    cv2.rectangle(frame2, (220, 160), (370, 310), (0, 0, 200), -1)
    cv2.circle(frame2, (500, 250), 60, (200, 0, 0), -1)

    path1 = os.path.join(IMAGE_DIR, filename1)
    path2 = os.path.join(IMAGE_DIR, filename2)
    cv2.imwrite(path1, frame1)
    cv2.imwrite(path2, frame2)
    print(f"[OK] Frame pair '{filename1}' dan '{filename2}' berhasil dibuat.")


def buat_gambar_textured(filename="textured_scene.png", width=640, height=480):
    """
    Membuat gambar dengan banyak tekstur untuk optical flow yang baik.
    """
    img = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
    # Menambahkan noise Gaussian untuk tekstur
    noise = np.random.randn(height, width, 3) * 20
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # Blur sedikit
    img = cv2.GaussianBlur(img, (5, 5), 1.0)

    # Menggambar beberapa objek geometris
    cv2.rectangle(img, (100, 80), (250, 200), (0, 0, 180), -1)
    cv2.circle(img, (400, 150), 70, (0, 180, 0), -1)
    cv2.ellipse(img, (300, 350), (80, 50), 30, 0, 360, (180, 0, 0), -1)

    path = os.path.join(IMAGE_DIR, filename)
    cv2.imwrite(path, img)
    print(f"[OK] Gambar '{filename}' berhasil dibuat.")


def buat_video_rotasi(filename="video_rotasi.avi", width=640, height=480, fps=30, durasi=4):
    """
    Membuat video dengan objek yang berotasi.
    Untuk analisis rotational motion.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    total_frames = fps * durasi
    center = (width // 2, height // 2)

    for i in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 220

        # Menambahkan grid background
        for gx in range(0, width, 60):
            cv2.line(frame, (gx, 0), (gx, height), (200, 200, 200), 1)
        for gy in range(0, height, 60):
            cv2.line(frame, (0, gy), (width, gy), (200, 200, 200), 1)

        # Menggambar objek yang berotasi (persegi panjang)
        angle = i * 3  # 3 derajat per frame
        rect_size = (150, 80)
        box = cv2.boxPoints(((center[0], center[1]), rect_size, angle))
        box = np.int32(box)
        cv2.fillPoly(frame, [box], (0, 100, 200))
        cv2.polylines(frame, [box], True, (0, 50, 150), 2)

        # Menambahkan lingkaran kecil yang mengorbit
        orbit_r = 150
        ox = int(center[0] + orbit_r * math.cos(math.radians(angle * 2)))
        oy = int(center[1] + orbit_r * math.sin(math.radians(angle * 2)))
        cv2.circle(frame, (ox, oy), 15, (0, 200, 0), -1)

        out.write(frame)

    out.release()
    print(f"[OK] Video '{filename}' berhasil dibuat ({total_frames} frame, {durasi}s).")


def buat_video_zoom(filename="video_zoom.avi", width=640, height=480, fps=30, durasi=3):
    """
    Membuat video simulasi zoom in/out.
    Untuk analisis scaling motion.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    total_frames = fps * durasi
    # Membuat gambar dasar yang lebih besar
    big_size = max(width, height) * 3
    base = np.ones((big_size, big_size, 3), dtype=np.uint8) * 200

    # Menggambar pattern di gambar dasar
    for i in range(0, big_size, 100):
        for j in range(0, big_size, 100):
            color = ((i * 37 + j * 53) % 150 + 50, (i * 23 + j * 67) % 150 + 50, (i * 47 + j * 31) % 150 + 50)
            cv2.rectangle(base, (j, i), (j+80, i+80), color, -1)

    center_x, center_y = big_size // 2, big_size // 2
    for i in range(total_frames):
        # Zoom in: crop semakin kecil
        t = i / total_frames
        scale = 1.0 + t * 2.0  # zoom dari 1x ke 3x
        crop_w = int(width / scale)
        crop_h = int(height / scale)

        x1 = center_x - crop_w // 2
        y1 = center_y - crop_h // 2
        crop = base[y1:y1+crop_h, x1:x1+crop_w]

        # Resize crop ke ukuran frame
        frame = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        out.write(frame)

    out.release()
    print(f"[OK] Video '{filename}' berhasil dibuat ({total_frames} frame, {durasi}s).")


# ============================================================
# LANGKAH 3: Generate semua video dan gambar
# ============================================================

print("\n" + "=" * 60)
print("GENERATING ASSETS UNTUK MODUL 09")
print("=" * 60)

# Video utama dengan bola bergerak
buat_video_bola_bergerak()

# Video dengan beberapa objek bergerak
buat_video_multi_objek()

# Video simulasi orang berjalan
buat_video_orang_berjalan()

# Video kamera panning/goyang
buat_video_panning()

# Video objek berotasi
buat_video_rotasi()

# Video efek zoom
buat_video_zoom()

# Sepasang frame untuk optical flow statis
buat_frame_pair()

# Gambar bertekstur
buat_gambar_textured()

# ============================================================
# LANGKAH 4: Verifikasi semua file
# ============================================================
print("\n" + "=" * 60)
print("VERIFIKASI FILE")
print("=" * 60)

for f in os.listdir(IMAGE_DIR):
    filepath = os.path.join(IMAGE_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  [✓] {f} ({size_kb:.1f} KB)")

print(f"\n[SELESAI] Semua asset untuk Modul 09 berhasil dibuat!")
print(f"[INFO] Folder image: {IMAGE_DIR}")
print(f"[INFO] Folder output: {OUTPUT_DIR}")
print("[INFO] Silakan jalankan percobaan 01-20.")
