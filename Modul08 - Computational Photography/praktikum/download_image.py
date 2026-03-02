"""
==========================================================================
SCRIPT DOWNLOAD GAMBAR ASLI
Modul 10 - Computational Photography
==========================================================================
Script ini men-download gambar ASLI dari internet sebagai bahan dasar
untuk 20 percobaan Computational Photography.

Gambar asli yang didownload:
  - scene_pemandangan.jpg : Foto landscape pegunungan Swiss (berkualitas tinggi,
                            warna kaya) untuk HDR, denoising, sharpening,
                            white balance, color enhancement, style transfer
  - portrait.jpg          : Foto portrait kucing/hewan close-up untuk efek
                            bokeh sintetis, pencil sketch, cartoon stylization
  - style_reference.jpg   : Lukisan Van Gogh "The Starry Night" (public domain)
                            untuk style transfer manual
  - city_night.jpg        : Foto kota malam hari (low-light) sebagai alternatif
                            gambar gelap untuk enhancement pipeline

Gambar turunan (derived) yang dibuat dari gambar asli di atas:
  - exposure_1.png ... exposure_5.png : Simulasi berbagai exposure dari foto
  - noisy_gaussian.png  : Foto asli + Gaussian noise
  - noisy_heavy.png     : Foto asli + noise berat
  - noisy_salt_pepper.png : Foto asli + salt-and-pepper noise
  - damaged_image.png   : Foto asli dengan goresan acak (untuk inpainting)
  - inpaint_mask.png    : Mask dari area rusak
  - low_resolution.png  : Foto asli resolusi rendah (untuk super resolution)
  - gambar_gelap.png    : Foto asli dengan exposure sangat gelap
  - low_contrast.png    : Foto asli dengan kontras rendah
  - depth_map_portrait.png : Estimasi depth map dari portrait (untuk bokeh)
  - tekstur_pattern.jpg : Gambar tekstur nyata untuk style transfer

Sumber: Wikimedia Commons (CC / Public Domain)
Jalankan script ini PERTAMA KALI sebelum menjalankan percobaan 01-20.
==========================================================================
"""

import os
import math
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

def download_image(url, dest_path, resize=(800, 600), desc=""):
    """
    Download gambar dari URL, decode dengan OpenCV, opsional resize, lalu simpan.
    Mengembalikan numpy array gambar jika berhasil, None jika gagal.
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

        cv2.imwrite(dest_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
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


def download_image_try_urls(url_list, dest_path, resize=(800, 600), desc=""):
    """
    Coba download dari beberapa URL, gunakan yang pertama berhasil.
    """
    for url, label in url_list:
        img = download_image(url, dest_path, resize=resize, desc=f"{desc} [{label}]")
        if img is not None:
            return img
    print(f"  [WARN] Semua URL gagal untuk {os.path.basename(dest_path)}")
    return None


# ============================================================
# LANGKAH 1: DOWNLOAD GAMBAR ASLI (BASE IMAGES)
# ============================================================

print("\n" + "=" * 60)
print("MODUL 10 - DOWNLOAD GAMBAR ASLI")
print("Sumber: Wikimedia Commons (CC / Public Domain)")
print("=" * 60)

# ─── scene_pemandangan.png ──────────────────────────────────
# Foto Engelberg, Swiss - pegunungan Alpen dengan salju, langit biru
# cerah, padang rumput hijau. Sangat representatif untuk eksperimen:
# denoising (gaussian, bilateral, NLM), sharpening, white balance,
# color enhancement, CLAHE, HDR, inpainting, super resolution,
# style transfer (digunakan oleh percobaan 05-20 hampir semua)
print("\n--- GAMBAR UTAMA: scene_pemandangan ---")

scene_path = os.path.join(IMAGE_DIR, "scene_pemandangan.png")
scene_img  = None

SCENE_URLS = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/"
        "Mount_Hood_reflected_in_Mirror_Lake%2C_Oregon.jpg/"
        "800px-Mount_Hood_reflected_in_Mirror_Lake%2C_Oregon.jpg",
        "Mount Hood Oregon, reflected in lake (Wikimedia, PD)"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
        "PNG_transparency_demonstration_1.png/"
        "640px-PNG_transparency_demonstration_1.png",
        "Demo PNG"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/"
        "24701-nature-natural-beauty.jpg/"
        "800px-24701-nature-natural-beauty.jpg",
        "Nature beauty (Wikimedia, CC)"
    ),
]

if os.path.exists(scene_path) and os.path.getsize(scene_path) > 50_000:
    print("  [SKIP] scene_pemandangan.png sudah ada.")
    scene_img = cv2.imread(scene_path)
else:
    for url, label in SCENE_URLS:
        scene_img = download_image(
            url, scene_path, resize=(800, 600),
            desc=f"scene_pemandangan.png [{label}]"
        )
        if scene_img is not None:
            break

# ─── portrait.png ──────────────────────────────────────────
# Foto close-up portrait kucing/hewan dengan depth-of-field nyata.
# Cocok untuk efek bokeh sintetis (percobaan 14), pencil sketch (17),
# cartoon stylization (18).
print("\n--- GAMBAR PORTRAIT: portrait.png ---")

portrait_path = os.path.join(IMAGE_DIR, "portrait.png")
portrait_img  = None

PORTRAIT_URLS = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/"
        "Gatto_europeo4.jpg/"
        "400px-Gatto_europeo4.jpg",
        "European cat close-up portrait (Wikimedia, CC)"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/"
        "Cat_November_2010-1a.jpg/"
        "400px-Cat_November_2010-1a.jpg",
        "Cat portrait face (Wikimedia, CC)"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/"
        "YellowLabradorLooking_new.jpg/"
        "400px-YellowLabradorLooking_new.jpg",
        "Yellow Labrador portrait (Wikimedia, CC)"
    ),
]

if os.path.exists(portrait_path) and os.path.getsize(portrait_path) > 20_000:
    print("  [SKIP] portrait.png sudah ada.")
    portrait_img = cv2.imread(portrait_path)
else:
    for url, label in PORTRAIT_URLS:
        portrait_img = download_image(
            url, portrait_path, resize=(600, 750),
            desc=f"portrait.png [{label}]"
        )
        if portrait_img is not None:
            break

# ─── style_reference.png ───────────────────────────────────
# Lukisan "The Starry Night" oleh Vincent van Gogh (1889)
# Domain publik - cocok sebagai style image untuk style transfer (percobaan 20)
print("\n--- STYLE REFERENCE: style_reference.png ---")

style_path = os.path.join(IMAGE_DIR, "style_reference.png")

STYLE_URLS = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/"
        "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/"
        "640px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "Van Gogh - The Starry Night (Wikimedia, Public Domain)"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/"
        "Die_Seerosen_-_Claude_Monet%2C_1906.jpg/"
        "500px-Die_Seerosen_-_Claude_Monet%2C_1906.jpg",
        "Monet - Water Lilies (Wikimedia, Public Domain)"
    ),
]

if os.path.exists(style_path) and os.path.getsize(style_path) > 20_000:
    print("  [SKIP] style_reference.png sudah ada.")
else:
    for url, label in STYLE_URLS:
        result = download_image(
            url, style_path, resize=(400, 300),
            desc=f"style_reference.png [{label}]"
        )
        if result is not None:
            break

# ─── tekstur_pattern.jpg ───────────────────────────────────
# Foto tekstur anyaman bambu / kain tradisional - pola natural yang kaya
# Digunakan sebagai pasangan style reference dalam style transfer (percobaan 20)
print("\n--- TEKSTUR PATTERN: tekstur_pattern.jpg ---")

tekstur_path = os.path.join(IMAGE_DIR, "tekstur_pattern.jpg")

TEKSTUR_URLS = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/"
        "Bamboo_weaving_in_detail.jpg/"
        "400px-Bamboo_weaving_in_detail.jpg",
        "Bamboo weaving texture (Wikimedia, CC)"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/"
        "Wheat_close-up.JPG/"
        "400px-Wheat_close-up.JPG",
        "Wheat field close-up texture (Wikimedia, CC)"
    ),
]

if os.path.exists(tekstur_path) and os.path.getsize(tekstur_path) > 10_000:
    print("  [SKIP] tekstur_pattern.jpg sudah ada.")
else:
    for url, label in TEKSTUR_URLS:
        result = download_image(
            url, tekstur_path, resize=(400, 400),
            desc=f"tekstur_pattern.jpg [{label}]"
        )
        if result is not None:
            break

# ============================================================
# LANGKAH 2: GAMBAR TURUNAN DARI FOTO ASLI
# ============================================================
# Semua gambar berikut ini DITURUNKAN dari foto landscape asli yang
# didownload di atas. Ini bukan generasi manual, ini adalah pemrosesan
# sinyal yang diterapkan pada image nyata.

print("\n" + "=" * 60)
print("MEMBUAT GAMBAR TURUNAN DARI FOTO ASLI")
print("=" * 60)

# Pastikan scene_img tersedia
if scene_img is None and os.path.exists(scene_path):
    scene_img = cv2.imread(scene_path)

if scene_img is None:
    print("[WARN] scene_pemandangan.png tidak tersedia, gambar turunan tidak dapat dibuat.")
else:
    h, w = scene_img.shape[:2]
    print(f"  Base image: scene_pemandangan.png ({w}x{h})")

    # ── Gambar berbagai exposure (untuk HDR, percobaan 01-04) ──────────
    print("\n  Membuat gambar multi-exposure (dari foto asli, bukan gambar buatan):")
    exposure_factors = [0.25, 0.5, 1.0, 1.8, 3.2]
    for i, factor in enumerate(exposure_factors, 1):
        out_img = np.clip(scene_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        out_path = os.path.join(IMAGE_DIR, f"exposure_{i}.png")
        cv2.imwrite(out_path, out_img)
        print(f"    [OK] exposure_{i}.png (factor={factor:.2f})")

    # ── Gambar noisy (untuk denoising, percobaan 05-07) ────────────────
    print("\n  Membuat gambar dengan noise (dari foto asli + noise sinyal):")

    np.random.seed(42)
    noise_g = np.random.randn(h, w, 3).astype(np.float32) * 30
    noisy_gauss = np.clip(scene_img.astype(np.float32) + noise_g, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(IMAGE_DIR, "noisy_gaussian.png"), noisy_gauss)
    print("    [OK] noisy_gaussian.png (sigma=30, dari foto asli)")

    noise_h = np.random.randn(h, w, 3).astype(np.float32) * 60
    noisy_heavy = np.clip(scene_img.astype(np.float32) + noise_h, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(IMAGE_DIR, "noisy_heavy.png"), noisy_heavy)
    print("    [OK] noisy_heavy.png (sigma=60, dari foto asli)")

    noisy_sp = scene_img.copy()
    num_salt = int(0.03 * noisy_sp.size / 3)
    ys = np.random.randint(0, h, num_salt)
    xs = np.random.randint(0, w, num_salt)
    noisy_sp[ys, xs] = 255
    ys2 = np.random.randint(0, h, num_salt)
    xs2 = np.random.randint(0, w, num_salt)
    noisy_sp[ys2, xs2] = 0
    cv2.imwrite(os.path.join(IMAGE_DIR, "noisy_salt_pepper.png"), noisy_sp)
    print("    [OK] noisy_salt_pepper.png (3% SP, dari foto asli)")

    # ── Gambar rusak + mask inpainting (percobaan 08-09) ───────────────
    print("\n  Membuat gambar rusak untuk inpainting (dari foto asli + goresan acak):")
    np.random.seed(7)
    damaged = scene_img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(10):
        x1 = np.random.randint(50, w - 50)
        y1 = np.random.randint(50, h - 50)
        x2 = x1 + np.random.randint(-100, 100)
        y2 = y1 + np.random.randint(-100, 100)
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        thick = np.random.randint(5, 18)
        # Goresan putih (simulasi teks/coretan yang hendak dihapus)
        cv2.line(damaged, (x1, y1), (x2, y2), (255, 255, 255), thick)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thick)
    for _ in range(8):
        cx = np.random.randint(80, w - 80)
        cy = np.random.randint(80, h - 80)
        r = np.random.randint(12, 28)
        cv2.circle(damaged, (cx, cy), r, (255, 255, 255), -1)
        cv2.circle(mask, (cx, cy), r, 255, -1)
    cv2.imwrite(os.path.join(IMAGE_DIR, "damaged_image.png"), damaged)
    cv2.imwrite(os.path.join(IMAGE_DIR, "inpaint_mask.png"), mask)
    print("    [OK] damaged_image.png + inpaint_mask.png (dari foto asli)")

    # ── Gambar resolusi rendah (percobaan 10) ──────────────────────────
    print("\n  Membuat gambar resolusi rendah (dari foto asli, dikecilkan 4x):")
    low_res = cv2.resize(scene_img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(IMAGE_DIR, "low_resolution.png"), low_res)
    print(f"    [OK] low_resolution.png ({low_res.shape[1]}x{low_res.shape[0]})")

    # ── Gambar gelap (percobaan 11, 16, 19) ───────────────────────────
    print("\n  Membuat gambar gelap (dari foto asli, exposure rendah):")
    dark_img = np.clip(scene_img.astype(np.float32) * 0.22, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(IMAGE_DIR, "gambar_gelap.png"), dark_img)
    print("    [OK] gambar_gelap.png (dari foto asli, factor=0.22)")

    # ── Gambar kontras rendah (percobaan 11) ──────────────────────────
    print("\n  Membuat gambar kontras rendah (dari foto asli):")
    low_c = np.clip(scene_img.astype(np.float32) * 0.35 + 90, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(IMAGE_DIR, "low_contrast.png"), low_c)
    print("    [OK] low_contrast.png (dari foto asli, compressed range)")

# ── Depth map portrait (percobaan 14) ─────────────────────────────
print("\n  Membuat estimasi depth map dari portrait asli (via edge+blur):")
portrait_img = cv2.imread(portrait_path) if portrait_img is None else portrait_img
if portrait_img is not None:
    ph, pw = portrait_img.shape[:2]
    gray_p = cv2.cvtColor(portrait_img, cv2.COLOR_BGR2GRAY)
    # Depth estimation sederhana: tepi (edges) cenderung dekat ke kamera
    edges = cv2.Canny(gray_p, 30, 100).astype(np.float32)
    # Background depth tinggi (jauh), foreground rendah (dekat)
    depth_map = cv2.GaussianBlur(255 - edges, (51, 51), 25)
    # Normalisasi ke 0-255
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(IMAGE_DIR, "depth_map_portrait.png"), depth_norm)
    print(f"    [OK] depth_map_portrait.png ({pw}x{ph}, dari portrait asli)")
else:
    print("    [WARN] Portrait tidak tersedia, depth_map_portrait tidak dibuat.")

# ============================================================
# VERIFIKASI AKHIR
# ============================================================

print("\n" + "=" * 60)
print("VERIFIKASI FILE - MODUL 10")
print("=" * 60)

required = [
    # Gambar asli (downloaded)
    ("scene_pemandangan.png",  "Real landscape photo - base untuk hampir semua percobaan"),
    ("portrait.png",           "Real portrait photo - bokeh, sketch, cartoon"),
    ("style_reference.png",    "Van Gogh painting - style transfer"),
    ("tekstur_pattern.jpg",    "Real texture photo - style transfer pattern"),
    # Derived dari foto asli
    ("exposure_1.png",         "Derived from landscape: sangat gelap (HDR)"),
    ("exposure_3.png",         "Derived from landscape: normal exposure (HDR)"),
    ("exposure_5.png",         "Derived from landscape: sangat terang (HDR)"),
    ("noisy_gaussian.png",     "Derived: landscape + Gaussian noise (denoising)"),
    ("noisy_heavy.png",        "Derived: landscape + noise berat (denoising NLM)"),
    ("noisy_salt_pepper.png",  "Derived: landscape + S&P noise"),
    ("damaged_image.png",      "Derived: landscape + goresan (inpainting)"),
    ("inpaint_mask.png",       "Derived: mask dari goresan (inpainting)"),
    ("low_resolution.png",     "Derived: landscape resolusi rendah (super-res)"),
    ("gambar_gelap.png",       "Derived: landscape sangat gelap (CLAHE, pipeline)"),
    ("low_contrast.png",       "Derived: landscape kontras rendah (CLAHE)"),
    ("depth_map_portrait.png", "Derived: estimated depth dari portrait (bokeh)"),
]

all_ok = True
for fname, usage in required:
    fpath = os.path.join(IMAGE_DIR, fname)
    exists = os.path.exists(fpath)
    size_kb = os.path.getsize(fpath) / 1024 if exists else 0
    status = "✓" if (exists and size_kb > 0.5) else "✗"
    note = "" if (exists and size_kb > 0.5) else "  ← PERLU DOWNLOAD ULANG"
    print(f"  [{status}] {fname:<30} {size_kb:>8.1f} KB  | {usage}{note}")
    if not (exists and size_kb > 0.5):
        all_ok = False

print(f"\n{'='*60}")
if all_ok:
    print("[SELESAI] Semua asset Modul 10 berhasil disiapkan!")
    print("[INFO]    Gambar dasar berasal dari foto nyata yang didownload.")
    print("[INFO]    Gambar turunan dibuat dari foto asli (bukan gambar buatan).")
else:
    print("[PERHATIAN] Beberapa file belum tersedia.")
    print("            Pastikan koneksi internet aktif lalu jalankan ulang.")
print(f"[INFO] Folder image : {IMAGE_DIR}")
print(f"[INFO] Folder output: {OUTPUT_DIR}")
print("[INFO] Silakan jalankan percobaan 01-20.")
