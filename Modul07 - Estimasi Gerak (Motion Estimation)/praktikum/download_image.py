"""
==========================================================================
SCRIPT DOWNLOAD GAMBAR DAN VIDEO ASLI
Modul 09 - Estimasi Gerak (Motion Estimation)
==========================================================================
Script ini men-download video dan gambar ASLI dari internet untuk digunakan
pada 20 percobaan Motion Estimation.

Video yang didownload (footage nyata):
  - video_bola.avi        : Footage olahraga/aksi - objek bergerak cepat
                            (untuk optical flow LK, dense flow, frame diff,
                             tracking CSRT/KCF, MHI, feature trajectory)
  - video_orang.avi       : Footage outdoor dengan figur manusia bergerak
                            (untuk background subtraction MOG2/KNN,
                             running average background)
  - video_multi_objek.avi : Footage jalanan dengan banyak objek bergerak
                            (untuk multi-object tracking, optical flow
                             magnitude, bg subtraction comparison,
                             motion contour detection)
  - video_panning.avi     : Footage outdoor dengan gerakan kamera lateral
                            (untuk video stabilization)

Gambar static yang didownload / diekstrak:
  - frame_t0.png          : Frame real dari video (optical flow statis,
                            frame interpolation linear/flow)
  - frame_t1.png          : Frame berikutnya dari video (frame pair)
  - textured_scene.png    : Foto landscape dengan tekstur kaya (feature tracking)

Sumber: Google Developers Sample Videos (freely available for developers)
        Wikimedia Commons (CC Licensed)
Jalankan script ini PERTAMA KALI sebelum menjalankan percobaan 01-20.
==========================================================================
"""

import os
import sys
import urllib.request
import urllib.error
import cv2
import numpy as np

# ============================================================
# KONFIGURASI DIREKTORI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("[INFO] Folder 'image/' dan 'output/' siap.")

# ============================================================
# FUNGSI UTILITAS
# ============================================================

def download_file(url, dest_path, desc=""):
    """
    Download file dari URL ke dest_path.
    Menggunakan User-Agent agar tidak diblokir server.
    """
    label = desc if desc else os.path.basename(dest_path)
    print(f"\n  [DOWNLOAD] {label}")
    print(f"  URL: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                ),
                'Accept': '*/*',
            }
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            data = response.read()

        with open(dest_path, 'wb') as f:
            f.write(data)

        size_kb = os.path.getsize(dest_path) / 1024
        print(f"  [OK] Berhasil: {size_kb:.1f} KB")
        return True

    except urllib.error.HTTPError as e:
        print(f"  [WARN] HTTP {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"  [WARN] URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"  [WARN] Error: {e}")
        return False


def convert_mp4_to_avi(mp4_path, avi_path, max_frames=180, target_size=(640, 480)):
    """
    Konversi video MP4 (real footage) ke format AVI menggunakan OpenCV.
    Mengambil max_frames pertama dari video sumber.
    Video sumber adalah footage nyata yang baru didownload.
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"  [WARN] Tidak dapat membuka: {mp4_path}")
        return False

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_src if 5 < fps_src <= 60 else 30.0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(avi_path, fourcc, fps, target_size)
    if not out.isOpened():
        # Coba codec alternatif
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(avi_path, fourcc, fps, target_size)

    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, target_size)
        out.write(frame_resized)
        count += 1

    cap.release()
    out.release()

    if count > 0:
        size_kb = os.path.getsize(avi_path) / 1024
        print(f"  [OK] AVI dibuat: {os.path.basename(avi_path)} ({count} frame, {size_kb:.0f} KB)")
        return True
    return False


def extract_frame_pair(video_path, idx_a=30, idx_b=33,
                       name_a="frame_t0.png", name_b="frame_t1.png"):
    """
    Ekstrak dua frame dari video nyata sebagai pasangan frame untuk
    percobaan optical flow statis dan frame interpolation.
    Frame diambil dari footage real yang sudah didownload.
    """
    path_a = os.path.join(IMAGE_DIR, name_a)
    path_b = os.path.join(IMAGE_DIR, name_b)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    saved_a = saved_b = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count == idx_a:
            cv2.imwrite(path_a, cv2.resize(frame, (640, 480)))
            saved_a = True
        if frame_count == idx_b:
            cv2.imwrite(path_b, cv2.resize(frame, (640, 480)))
            saved_b = True
        if saved_a and saved_b:
            break
        frame_count += 1

    cap.release()

    if saved_a and saved_b:
        print(f"  [OK] Frame pair diekstrak dari footage nyata:")
        print(f"       {name_a} (frame #{idx_a})")
        print(f"       {name_b} (frame #{idx_b})")
        return True
    return False


def download_image(url, dest_path, resize=None):
    """
    Download gambar dari URL dan simpan. Opsional resize ke ukuran tertentu.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
                )
            }
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = np.frombuffer(resp.read(), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            return False
        if resize:
            img = cv2.resize(img, resize)
        cv2.imwrite(dest_path, img)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"  [OK] {os.path.basename(dest_path)} ({img.shape[1]}x{img.shape[0]}, {size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  [WARN] Gagal download gambar: {e}")
        return False


# ============================================================
# LANGKAH 1: DOWNLOAD VIDEO ASLI
# ============================================================

print("\n" + "=" * 60)
print("MODUL 09 - DOWNLOAD VIDEO DAN GAMBAR ASLI")
print("=" * 60)

# Sumber: Google Developers Sample Videos
# Video-video ini adalah footage nyata (real footage), bebas digunakan
# untuk keperluan development/pendidikan
GOOGLE_CDN = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample"

# ForBiggerBlazes.mp4  → aksi outdoor / objek bergerak → video_bola.avi
# ForBiggerEscapes.mp4 → figur manusia outdoor bergerak → video_orang.avi
# ForBiggerJoyrides.mp4 → perjalanan outdoor multi-subjek → video_multi_objek.avi
# ForBiggerMeltdowns.mp4 → outdoor dengan pan kamera → video_panning.avi

print("\n--- VIDEO ---")

video_configs = [
    {
        "url":      f"{GOOGLE_CDN}/ForBiggerBlazes.mp4",
        "temp":     "temp_v1.mp4",
        "avi":      "video_bola.avi",
        "desc": (
            "Footage aksi outdoor (ForBiggerBlazes.mp4) → video_bola.avi\n"
            "  Digunakan: optical flow LK/dense, tracking CSRT/KCF, MHI,\n"
            "  frame diff, feature trajectory, realtime optical flow"
        ),
    },
    {
        "url":      f"{GOOGLE_CDN}/ForBiggerEscapes.mp4",
        "temp":     "temp_v2.mp4",
        "avi":      "video_orang.avi",
        "desc": (
            "Footage figur manusia outdoor (ForBiggerEscapes.mp4) → video_orang.avi\n"
            "  Digunakan: background subtraction MOG2/KNN, running average BG"
        ),
    },
    {
        "url":      f"{GOOGLE_CDN}/ForBiggerJoyrides.mp4",
        "temp":     "temp_v3.mp4",
        "avi":      "video_multi_objek.avi",
        "desc": (
            "Footage perjalanan outdoor (ForBiggerJoyrides.mp4) → video_multi_objek.avi\n"
            "  Digunakan: multi-object tracking, optical flow magnitude/direction,\n"
            "  background subtraction comparison, deteksi gerakan contour"
        ),
    },
    {
        "url":      f"{GOOGLE_CDN}/ForBiggerMeltdowns.mp4",
        "temp":     "temp_v4.mp4",
        "avi":      "video_panning.avi",
        "desc": (
            "Footage outdoor dengan gerakan kamera (ForBiggerMeltdowns.mp4) → video_panning.avi\n"
            "  Digunakan: video stabilization"
        ),
    },
]

first_avi_path = None  # Untuk ekstrak frame pair

for cfg in video_configs:
    print(f"\n{'─'*55}")
    print(f"  {cfg['desc']}")

    temp_path = os.path.join(IMAGE_DIR, cfg["temp"])
    avi_path  = os.path.join(IMAGE_DIR, cfg["avi"])

    # Skip jika sudah ada
    if os.path.exists(avi_path) and os.path.getsize(avi_path) > 50_000:
        print(f"  [SKIP] {cfg['avi']} sudah ada.")
        if first_avi_path is None:
            first_avi_path = avi_path
        continue

    # Download MP4 asli
    ok = download_file(cfg["url"], temp_path)
    if ok and os.path.getsize(temp_path) > 50_000:
        # Konversi ke AVI (160 frame ≈ 5-6 detik @ 30fps)
        ok2 = convert_mp4_to_avi(temp_path, avi_path, max_frames=160)
        if ok2 and first_avi_path is None:
            first_avi_path = avi_path
        # Hapus file temp
        try:
            os.remove(temp_path)
        except Exception:
            pass
    else:
        print(f"  [WARN] Download gagal, {cfg['avi']} tidak tersedia.")
        # Hapus file kosong
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

# ============================================================
# LANGKAH 2: EKSTRAK FRAME PAIR DARI VIDEO NYATA
# ============================================================

print(f"\n{'─'*55}")
print("  Ekstrak frame pair dari footage nyata → frame_t0.png + frame_t1.png")
print("  Digunakan: visualisasi optical flow statis, frame interpolation")

frame_t0 = os.path.join(IMAGE_DIR, "frame_t0.png")
frame_t1 = os.path.join(IMAGE_DIR, "frame_t1.png")

if (os.path.exists(frame_t0) and os.path.exists(frame_t1) and
        os.path.getsize(frame_t0) > 1000):
    print("  [SKIP] frame_t0.png + frame_t1.png sudah ada.")
elif first_avi_path and os.path.exists(first_avi_path):
    ok = extract_frame_pair(first_avi_path, 30, 33)
    if not ok:
        # Coba dengan index berbeda
        extract_frame_pair(first_avi_path, 5, 8)
else:
    print("  [WARN] Tidak ada video sumber untuk ekstrak frame pair.")

# ============================================================
# LANGKAH 3: DOWNLOAD GAMBAR BERTEKSTUR ASLI
# ============================================================

print(f"\n{'─'*55}")
print("  Download gambar bertekstur → textured_scene.png")
print("  Digunakan: feature tracking, pengujian optical flow")

textured_path = os.path.join(IMAGE_DIR, "textured_scene.png")

if os.path.exists(textured_path) and os.path.getsize(textured_path) > 10_000:
    print("  [SKIP] textured_scene.png sudah ada.")
else:
    # Foto Sahara Desert dunes - tekstur pasir sangat kaya untuk feature detection
    # Sumber: Wikimedia Commons (Public Domain)
    TEXTURE_URLS = [
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/"
            "Sand_dunes_in_the_Sahara.jpg/640px-Sand_dunes_in_the_Sahara.jpg",
            "Sahara desert dunes (Wikimedia Commons, PD)"
        ),
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/"
            "24701-nature-natural-beauty.jpg/640px-24701-nature-natural-beauty.jpg",
            "Nature forest texture (Wikimedia Commons, CC)"
        ),
    ]
    ok = False
    for url, label in TEXTURE_URLS:
        print(f"\n  Mencoba: {label}")
        ok = download_image(url, textured_path, resize=(640, 480))
        if ok:
            break
    if not ok:
        print("  [WARN] Semua URL tekstur gagal.")

# ============================================================
# VERIFIKASI AKHIR
# ============================================================

print("\n" + "=" * 60)
print("VERIFIKASI FILE - MODUL 09")
print("=" * 60)

required = [
    ("video_bola.avi",        "Optical flow, CSRT/KCF tracking, MHI, frame diff"),
    ("video_orang.avi",       "Background subtraction MOG2/KNN, running avg BG"),
    ("video_multi_objek.avi", "Multi-object tracking, flow magnitude, motion contour"),
    ("video_panning.avi",     "Video stabilization"),
    ("frame_t0.png",          "Optical flow statis, frame interpolation"),
    ("frame_t1.png",          "Optical flow statis, frame interpolation"),
    ("textured_scene.png",    "Feature tracking, optical flow test"),
]

all_ok = True
for fname, usage in required:
    fpath = os.path.join(IMAGE_DIR, fname)
    exists = os.path.exists(fpath)
    size_kb = os.path.getsize(fpath) / 1024 if exists else 0
    status = "✓" if (exists and size_kb > 1) else "✗"
    mark = "" if (exists and size_kb > 1) else "  ← PERLU DOWNLOAD ULANG"
    print(f"  [{status}] {fname:<28} {size_kb:>8.1f} KB  | {usage}{mark}")
    if not (exists and size_kb > 1):
        all_ok = False

print(f"\n{'='*60}")
if all_ok:
    print("[SELESAI] Semua asset Modul 09 berhasil didownload!")
    print("[INFO]    Semua video berasal dari footage nyata (real video).")
else:
    print("[PERHATIAN] Beberapa file belum tersedia.")
    print("            Pastikan koneksi internet aktif lalu jalankan ulang.")
print(f"[INFO] Folder image : {IMAGE_DIR}")
print(f"[INFO] Folder output: {OUTPUT_DIR}")
print("[INFO] Silakan jalankan percobaan 01-20.")
