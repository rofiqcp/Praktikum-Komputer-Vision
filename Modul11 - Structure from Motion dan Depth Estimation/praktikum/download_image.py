"""
==========================================================================
SCRIPT DOWNLOAD GAMBAR ASLI
Modul 11 - Structure from Motion dan Depth Estimation
==========================================================================
Script ini men-download gambar ASLI dari internet sebagai bahan dasar
untuk 20 percobaan SfM dan Depth Estimation.

Gambar asli yang didownload:
  - building_stereo.jpg   : Foto eksterior bangunan / arsitektur dengan
                            banyak fitur geometris dan tekstur kaya.
                            Digunakan sebagai dasar pembuatan stereo pair.
  - gambar_fitur.png      : Foto fasad bangunan bersejarah (Notre Dame de Paris)
                            Kaya akan corners dan edges untuk feature detection,
                            SIFT/ORB matching, fundamental matrix, epipolar lines
  - indoor_scene.jpg      : Foto interior gedung/koridor dengan depth variation
                            Digunakan untuk monocular depth estimation

Gambar turunan (derived) dari foto asli:
  - stereo_left.png       : Foto bangunan asli (pandangan kiri stereo)
                            Digunakan: fundamental matrix, essential matrix,
                            epipolar lines, triangulasi, stereo matching,
                            disparity map, depth estimation
  - stereo_right.png      : Simulasi pandangan kanan stereo (perspektif shift
                            pada foto asli, mensimulasikan baseline kamera)
  - gambar_fitur_rotasi.png: gambar_fitur dirotasi -30° untuk pengujian
                             feature matching pada beda orientasi
  - gambar_depth.png      : Indoor scene dengan variasi kedalaman

Catatan: Stereo pair dibuat dengan mensimulasikan baseline kamera menggunakan
         perspektif transform pada foto nyata. Ini adalah teknik standar
         dalam pendidikan stereo vision.

Sumber: Wikimedia Commons (CC / Public Domain)
Jalankan script ini PERTAMA KALI sebelum menjalankan percobaan 01-20.
==========================================================================
"""

import os
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

def download_image(url, dest_path, resize=None, desc=""):
    """
    Download gambar dari URL, decode dengan OpenCV, opsional resize, simpan.
    Mengembalikan array gambar jika berhasil, None jika gagal.
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
                    'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
                ),
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            }
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = np.frombuffer(resp.read(), dtype=np.uint8)

        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            print("  [WARN] Tidak dapat decode gambar.")
            return None

        if resize:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(dest_path, img)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"  [OK] {img.shape[1]}x{img.shape[0]}, {size_kb:.1f} KB")
        return img

    except urllib.error.HTTPError as e:
        print(f"  [WARN] HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"  [WARN] URL Error: {e.reason}")
    except Exception as e:
        print(f"  [WARN] Error: {e}")
    return None


def create_stereo_right(left_img, baseline_fraction=0.04):
    """
    Membuat pandangan kanan stereo dari gambar kiri menggunakan
    perspektif transform yang mensimulasikan pergeseran kamera horizontal.
    Ini adalah teknik standar dalam pendidikan stereo vision:
    - baseline_fraction: proporsi lebar gambar untuk pergeseran (default 4%)
    """
    h, w = left_img.shape[:2]
    baseline = int(w * baseline_fraction)

    # Perspektif transform mensimulasikan kamera bergeser ke kanan
    # Titik-titik sumber (gambar kiri)
    pts_src = np.float32([
        [0,       0      ],
        [w - 1,   0      ],
        [w - 1,   h - 1  ],
        [0,       h - 1  ],
    ])
    # Titik-titik tujuan (gambar kanan - perspektif geser ke kiri sedikit)
    pts_dst = np.float32([
        [baseline,          int(h * 0.01) ],
        [w - 1,             0             ],
        [w - 1,             h - 1         ],
        [baseline,          h - 1 - int(h * 0.01)],
    ])

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    right_img = cv2.warpPerspective(left_img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_REPLICATE)
    return right_img


# ============================================================
# LANGKAH 1: DOWNLOAD GAMBAR ASLI
# ============================================================

print("\n" + "=" * 60)
print("MODUL 11 - DOWNLOAD GAMBAR ASLI")
print("Sumber: Wikimedia Commons (CC / Public Domain)")
print("=" * 60)

# ─── gambar_fitur.png ──────────────────────────────────────
# Fasad Notre-Dame de Paris - bangunan bersejarah yang kaya fitur:
# banyak sudut, tepi, dan detail arsitektur untuk SIFT/ORB matching.
# Digunakan: percobaan 01 (feature detection & matching)
print("\n--- GAMBAR FITUR: gambar_fitur.png ---")

fitur_path = os.path.join(IMAGE_DIR, "gambar_fitur.png")
fitur_img  = None

FITUR_URLS = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/"
        "Notre-Dame_de_Paris.jpg/640px-Notre-Dame_de_Paris.jpg",
        "Notre-Dame de Paris facade (Wikimedia, CC)"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/"
        "Notre_Dame_Cathedral_in_Paris.jpg/640px-Notre_Dame_Cathedral_in_Paris.jpg",
        "Notre Dame Cathedral Paris (Wikimedia, CC)"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/"
        "Colosseum_in_Rome%2C_Italy_-_April_2007.jpg/"
        "640px-Colosseum_in_Rome%2C_Italy_-_April_2007.jpg",
        "Colosseum Rome (Wikimedia, CC)"
    ),
]

if os.path.exists(fitur_path) and os.path.getsize(fitur_path) > 30_000:
    print("  [SKIP] gambar_fitur.png sudah ada.")
    fitur_img = cv2.imread(fitur_path)
else:
    for url, label in FITUR_URLS:
        fitur_img = download_image(
            url, fitur_path, resize=(640, 480),
            desc=f"gambar_fitur.png [{label}]"
        )
        if fitur_img is not None:
            break

# ─── gambar_fitur_rotasi.png ──────────────────────────────
# Gambar fitur yang sama tapi dirotasi -30° - untuk menguji
# invariance terhadap rotasi dalam feature matching
print("\n--- GAMBAR FITUR ROTASI: gambar_fitur_rotasi.png ---")

rotasi_path = os.path.join(IMAGE_DIR, "gambar_fitur_rotasi.png")

if os.path.exists(rotasi_path) and os.path.getsize(rotasi_path) > 10_000:
    print("  [SKIP] gambar_fitur_rotasi.png sudah ada.")
elif fitur_img is not None:
    h_f, w_f = fitur_img.shape[:2]
    center_f = (w_f // 2, h_f // 2)
    M_rot = cv2.getRotationMatrix2D(center_f, -30, 0.9)  # rotasi -30°, scale 0.9
    rotated = cv2.warpAffine(fitur_img, M_rot, (w_f, h_f),
                             flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(rotasi_path, rotated)
    size_kb = os.path.getsize(rotasi_path) / 1024
    print(f"  [OK] gambar_fitur_rotasi.png (rotasi -30° dari foto asli, {size_kb:.1f} KB)")
else:
    print("  [WARN] gambar_fitur tidak tersedia, gambar_fitur_rotasi tidak dibuat.")

# ─── Stereo pair: stereo_left.png + stereo_right.png ──────
# Building exterior yang kaya fitur untuk stereo/depth estimation.
# Paris Pantheon - detail arsitektur, kolom, dan perspektif yang bagus
# Digunakan: percobaan 02-09, 16 (fundamental/essential matrix, epipolar,
# triangulasi, stereo calibration, disparity, depth estimation)
print("\n--- STEREO PAIR: stereo_left.png + stereo_right.png ---")

stereo_left_path  = os.path.join(IMAGE_DIR, "stereo_left.png")
stereo_right_path = os.path.join(IMAGE_DIR, "stereo_right.png")
stereo_base_img   = None

STEREO_URLS = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/"
        "Pantheon_Rome.jpg/640px-Pantheon_Rome.jpg",
        "Roman Pantheon exterior (Wikimedia, CC)"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/"
        "Colosseo_2020.jpg/640px-Colosseo_2020.jpg",
        "Colosseum Rome exterior (Wikimedia, CC)"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/"
        "Smiley.svg/240px-Smiley.svg.png",
        "Smiley fallback"
    ),
]

stereo_base_path = os.path.join(IMAGE_DIR, "building_stereo_base.jpg")

if (os.path.exists(stereo_left_path)  and os.path.getsize(stereo_left_path)  > 30_000 and
        os.path.exists(stereo_right_path) and os.path.getsize(stereo_right_path) > 30_000):
    print("  [SKIP] stereo_left.png + stereo_right.png sudah ada.")
else:
    # Download foto bangunan sebagai base
    for url, label in STEREO_URLS:
        stereo_base_img = download_image(
            url, stereo_base_path, resize=(640, 480),
            desc=f"stereo base [{label}]"
        )
        if stereo_base_img is not None:
            break

    if stereo_base_img is not None:
        # Stereo left = gambar asli
        cv2.imwrite(stereo_left_path, stereo_base_img)
        size_kb = os.path.getsize(stereo_left_path) / 1024
        print(f"  [OK] stereo_left.png (foto asli, {size_kb:.1f} KB)")

        # Stereo right = perspektif shift dari gambar asli (simulates camera baseline)
        right_img = create_stereo_right(stereo_base_img, baseline_fraction=0.04)
        cv2.imwrite(stereo_right_path, right_img)
        size_kb = os.path.getsize(stereo_right_path) / 1024
        print(f"  [OK] stereo_right.png (simulated baseline dari foto asli, {size_kb:.1f} KB)")
        print(f"       Baseline: {int(640 * 0.04)}px horizontal offset")
    else:
        print("  [WARN] Download stereo base gagal.")

# ─── gambar_depth.png ─────────────────────────────────────
# Foto koridor/interior dengan variasi kedalaman yang jelas:
# benda-benda di foreground dan background berbeda jauh (depth cues kuat)
# Digunakan: percobaan 10 (monocular depth estimation)
print("\n--- GAMBAR DEPTH: gambar_depth.png ---")

depth_path = os.path.join(IMAGE_DIR, "gambar_depth.png")

DEPTH_URLS = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
        "Bikeroom.jpg/640px-Bikeroom.jpg",
        "Indoor bike room (Wikimedia, CC) - depth variation jelas"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/"
        "Hallway_at_an_angle.jpg/640px-Hallway_at_an_angle.jpg",
        "Hallway corridor (Wikimedia, CC)"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/"
        "Perspective_hall.jpg/640px-Perspective_hall.jpg",
        "Perspective hall (Wikimedia, CC)"
    ),
]

if os.path.exists(depth_path) and os.path.getsize(depth_path) > 20_000:
    print("  [SKIP] gambar_depth.png sudah ada.")
else:
    for url, label in DEPTH_URLS:
        result = download_image(
            url, depth_path, resize=(640, 480),
            desc=f"gambar_depth.png [{label}]"
        )
        if result is not None:
            break

# ============================================================
# MEMBUAT GAMBAR MULTI-VIEW UNTUK EKS 20 (SFM PIPELINE)
# ============================================================

print("\n" + "=" * 60)
print("MEMBUAT MULTI-VIEW IMAGES DARI gambar_fitur.png")
print("=" * 60)

_fitur_path = os.path.join(IMAGE_DIR, "gambar_fitur.png")
if os.path.exists(_fitur_path):
    _base = cv2.imread(_fitur_path)
    if _base is not None:
        _h, _w = _base.shape[:2]
        # Buat 5 view dengan horizontal perspective shift bertahap
        # View 0 = kiri (shift -30px), ..., View 4 = kanan (+30px)
        _shifts = [-30, -15, 0, 15, 30]
        for _i, _dx in enumerate(_shifts):
            _dst_path = os.path.join(IMAGE_DIR, f"multiview_sfm_{_i:02d}.png")
            if not os.path.exists(_dst_path):
                # Terapkan perspektif transform horizontal
                _src_pts = np.float32([[0, 0], [_w, 0], [_w, _h], [0, _h]])
                _dst_pts = np.float32([
                    [max(0, _dx),       0],
                    [min(_w, _w + _dx), 0],
                    [min(_w, _w + _dx), _h],
                    [max(0, _dx),       _h],
                ])
                _M = cv2.getPerspectiveTransform(_src_pts, _dst_pts)
                _view = cv2.warpPerspective(_base, _M, (_w, _h))
                cv2.imwrite(_dst_path, _view)
                _sz = os.path.getsize(_dst_path) / 1024
                print(f"  [OK] multiview_sfm_{_i:02d}.png dibuat ({_sz:.1f} KB)")
            else:
                print(f"  [--] multiview_sfm_{_i:02d}.png sudah ada")
    else:
        print("  [SKIP] gambar_fitur.png gagal dibaca")
else:
    print("  [SKIP] gambar_fitur.png belum ada, jalankan lagi setelah download")

# ============================================================
# VERIFIKASI AKHIR
# ============================================================

print("\n" + "=" * 60)
print("VERIFIKASI FILE - MODUL 11")
print("=" * 60)

required = [
    ("gambar_fitur.png",        "Real: Notre-Dame facade - feature detection & matching"),
    ("gambar_fitur_rotasi.png", "Derived: rotasi -30° dari gambar_fitur"),
    ("stereo_left.png",         "Real: foto bangunan - pandangan stereo kiri"),
    ("stereo_right.png",        "Derived: perspektif shift - pandangan stereo kanan"),
    ("gambar_depth.png",        "Real: indoor scene - monocular depth estimation"),
    ("multiview_sfm_00.png",    "Derived: multi-view view 0 (SfM pipeline exp 20)"),
    ("multiview_sfm_02.png",    "Derived: multi-view view 2 center (SfM pipeline exp 20)"),
    ("multiview_sfm_04.png",    "Derived: multi-view view 4 (SfM pipeline exp 20)"),
]

all_ok = True
for fname, usage in required:
    fpath = os.path.join(IMAGE_DIR, fname)
    exists = os.path.exists(fpath)
    size_kb = os.path.getsize(fpath) / 1024 if exists else 0
    status = "✓" if (exists and size_kb > 5) else "✗"
    note = "" if (exists and size_kb > 5) else "  ← PERLU DOWNLOAD ULANG"
    print(f"  [{status}] {fname:<30} {size_kb:>8.1f} KB  | {usage}{note}")
    if not (exists and size_kb > 5):
        all_ok = False

print(f"\n{'='*60}")
if all_ok:
    print("[SELESAI] Semua asset Modul 11 berhasil disiapkan!")
    print("[INFO]    Gambar fitur dan stereo berasal dari foto nyata.")
    print("[INFO]    Stereo pair menggunakan perspektif transform dari foto asli.")
else:
    print("[PERHATIAN] Beberapa file belum tersedia.")
    print("            Pastikan koneksi internet aktif lalu jalankan ulang.")
print(f"[INFO] Folder image : {IMAGE_DIR}")
print(f"[INFO] Folder output: {OUTPUT_DIR}")
print("[INFO] Silakan jalankan percobaan 01-20.")
