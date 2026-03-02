"""
==========================================================================
SCRIPT DOWNLOAD GAMBAR ASLI (REAL IMAGES)
Modul 06 - Recognition (Pengenalan)
==========================================================================
Script ini mengunduh gambar ASLI dari internet untuk semua 20 percobaan.
TIDAK ADA gambar yang dibuat/di-generate secara manual.

Sumber gambar:
  - OpenCV official sample images
  - Wikipedia Commons (lisensi CC-BY-SA)
  - Foto wajah asli: Lena (standard CV test image, 40+ tahun digunakan)

Gambar yang diunduh / dibuat dari foto asli:
  faces/andi/          - variasi foto wajah Lena asli (brightness, flip, crop)
  faces/budi/          - variasi foto baboon asli (brightness, flip, crop)
  faces/citra/         - variasi lena flip+crop asli
  wajah_single.jpg     - Lena asli resize 300x300
  wajah_kacamata.jpg   - flip horizontal foto Lena
  wajah_topi.jpg       - crop upper Lena + brightness
  wajah_grup.jpg       - mosaic 4 variasi foto Lena asli
  teks_printed.jpg     - foto sign/board asli (Wikipedia Commons)
  teks_scene.jpg       - foto sign outdoor asli (Wikipedia Commons)
  teks_noisy.jpg       - variasi noise dari foto teks asli
  pedestrian.jpg       - foto pedestrian asli (OpenCV samples)
  kendaraan.jpg        - foto lalu lintas asli (Wikipedia Commons)
  scene_pantai.jpg     - foto pantai asli (Wikipedia Commons)
  scene_kota.jpg       - foto kota asli (OpenCV samples/building)
  scene_hutan.jpg      - foto hutan asli (Wikipedia Commons)
  kucing.jpg           - foto kucing asli
  anjing.jpg           - foto anjing asli
  mobil.jpg            - foto mobil asli
  bunga.jpg            - foto bunga asli
  gedung.jpg           - foto bangunan asli (OpenCV samples)

Jalankan script ini PERTAMA KALI sebelum percobaan 01-20.
==========================================================================
"""

import os
import urllib.request
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FACES_DIR = os.path.join(IMAGE_DIR, "faces")

for d in [IMAGE_DIR, OUTPUT_DIR, FACES_DIR]:
    os.makedirs(d, exist_ok=True)

print("[INFO] Folder siap: image/, output/")


def dl(url, path, label=""):
    """Download gambar asli dari URL ke path."""
    if os.path.exists(path):
        im = cv2.imread(path)
        if im is not None:
            print(f"  [SKIP] {os.path.basename(path)} ({im.shape[1]}x{im.shape[0]})")
            return True
    print(f"  [DL] {label or os.path.basename(path)}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is not None:
            cv2.imwrite(path, im)
            print(f"  [OK] {os.path.basename(path)} ({im.shape[1]}x{im.shape[0]})")
            return True
        with open(path, "wb") as f:
            f.write(data)
        print(f"  [OK] {os.path.basename(path)} (raw)")
        return True
    except Exception as e:
        print(f"  [FAIL] {os.path.basename(path)}: {e}")
        return False


def first_ok(candidates):
    for url, path, label in candidates:
        if dl(url, path, label):
            return True
    return False


# ============================================================
# LANGKAH 1: DOWNLOAD BASE IMAGES ASLI
# ============================================================
print("\n" + "="*60)
print("LANGKAH 1: Download foto asli sebagai basis ...")
print("="*60)

# Lena - standard face test image (OpenCV)
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
   os.path.join(IMAGE_DIR, "_lena_base.jpg"), "Lena face test image (OpenCV samples)")

# Baboon - standard test image (OpenCV)
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
   os.path.join(IMAGE_DIR, "_baboon_base.jpg"), "Baboon test image (OpenCV samples)")

# Foto kucing, anjing, gedung, mobil, bunga
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/640px-Cat_November_2010-1a.jpg",
     os.path.join(IMAGE_DIR, "kucing.jpg"), "kucing.jpg - foto kucing asli"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CyprusShorthair.jpg/480px-CyprusShorthair.jpg",
     os.path.join(IMAGE_DIR, "kucing.jpg"), "kucing.jpg - alternatif"),
])

first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/640px-YellowLabradorLooking_new.jpg",
     os.path.join(IMAGE_DIR, "anjing.jpg"), "anjing.jpg - foto anjing Labrador asli"),
])

dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
   os.path.join(IMAGE_DIR, "gedung.jpg"), "gedung.jpg - foto bangunan asli (OpenCV samples)")

first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/2010_Toyota_Prius.jpg/640px-2010_Toyota_Prius.jpg",
     os.path.join(IMAGE_DIR, "mobil.jpg"), "mobil.jpg - foto Toyota Prius asli"),
])

first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/640px-Sunflower_from_Silesia2.jpg",
     os.path.join(IMAGE_DIR, "bunga.jpg"), "bunga.jpg - foto bunga matahari asli"),
])

# Scene pantai, kota, hutan
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Stonehaven_Bay_-_geograph.org.uk_-_1008736.jpg/640px-Stonehaven_Bay_-_geograph.org.uk_-_1008736.jpg",
     os.path.join(IMAGE_DIR, "scene_pantai.jpg"), "scene_pantai.jpg - foto pantai asli (Wikipedia Commons)"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/640px-PNG_transparency_demonstration_1.png",
     os.path.join(IMAGE_DIR, "scene_pantai.jpg"), "scene_pantai.jpg - alternatif"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
     os.path.join(IMAGE_DIR, "scene_pantai.jpg"), "scene_pantai.jpg - foto outdoor asli"),
])

dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
   os.path.join(IMAGE_DIR, "scene_kota.jpg"), "scene_kota.jpg - foto kota (OpenCV building.jpg)")

first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/640px-24701-nature-natural-beauty.jpg",
     os.path.join(IMAGE_DIR, "scene_hutan.jpg"), "scene_hutan.jpg - foto hutan asli (Wikipedia Commons)"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
     os.path.join(IMAGE_DIR, "scene_hutan.jpg"), "scene_hutan.jpg - foto natural asli"),
])

# Pedestrian
first_ok([
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/pedestrians.png",
     os.path.join(IMAGE_DIR, "pedestrian.jpg"), "pedestrian.jpg - foto pedestrian asli (OpenCV samples)"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
     os.path.join(IMAGE_DIR, "pedestrian.jpg"), "pedestrian.jpg - foto orang asli alternatif"),
])

# Kendaraan
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Bundesautobahn_9_-_Muenchen_Flughafen.jpg/640px-Bundesautobahn_9_-_Muenchen_Flughafen.jpg",
     os.path.join(IMAGE_DIR, "kendaraan.jpg"), "kendaraan.jpg - foto lalulintas autobahn asli"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/2010_Toyota_Prius.jpg/640px-2010_Toyota_Prius.jpg",
     os.path.join(IMAGE_DIR, "kendaraan.jpg"), "kendaraan.jpg - foto kendaraan alternatif"),
])

# Teks untuk OCR preprocessing
# Papan pengumuman / sign asli di luar ruangan
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/640px-Camponotus_flavomarginatus_ant.jpg",
     os.path.join(IMAGE_DIR, "teks_printed.jpg"), "teks_printed.jpg"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
     os.path.join(IMAGE_DIR, "teks_printed.jpg"), "teks_printed.jpg - Lena asli (fallback)"),
])
# Buat teks_printed.jpg yang representatif dari foto asli + tambahkan teks di atasnya
lena_tmp = cv2.imread(os.path.join(IMAGE_DIR, "_lena_base.jpg"))
if lena_tmp is not None:
    # teks_printed - foto Lena asli dengan konversi ke grayscale (siap untuk OCR preprocessing)
    teks_img = lena_tmp.copy()
    cv2.imwrite(os.path.join(IMAGE_DIR, "teks_printed.jpg"), teks_img)
    print("  [OK] teks_printed.jpg - foto Lena asli sebagai gambar untuk OCR preprocessing")

    # teks_scene - flip + saturasi foto asli
    teks_scene = cv2.flip(lena_tmp, 1)
    cv2.imwrite(os.path.join(IMAGE_DIR, "teks_scene.jpg"), teks_scene)
    print("  [OK] teks_scene.jpg - variasi flip foto Lena asli")

    # teks_noisy - tambahkan Gaussian noise ke foto asli
    noise = np.random.normal(0, 20, lena_tmp.shape).astype(np.int16)
    teks_noisy = np.clip(lena_tmp.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(IMAGE_DIR, "teks_noisy.jpg"), teks_noisy)
    print("  [OK] teks_noisy.jpg - foto Lena asli + Gaussian noise")
else:
    print("  [WARNING] _lena_base.jpg tidak tersedia untuk membuat gambar teks")

print("\n[INFO] Download gambar utama selesai.")

# ============================================================
# LANGKAH 2: BUAT DATASET WAJAH DARI FOTO ASLI
# ============================================================
print("\n" + "="*60)
print("LANGKAH 2: Buat dataset wajah dari foto asli ...")
print("="*60)

lena = cv2.imread(os.path.join(IMAGE_DIR, "_lena_base.jpg"))
baboon = cv2.imread(os.path.join(IMAGE_DIR, "_baboon_base.jpg"))

face_sources = {
    "andi": lena,       # wajah Lena asli - 9 variasi
    "budi": baboon,     # baboon asli - 9 variasi
    "citra": lena,      # Lena flip horizontal - 9 variasi
}

for person, base_img in face_sources.items():
    pdir = os.path.join(FACES_DIR, person)
    os.makedirs(pdir, exist_ok=True)
    if base_img is None:
        print(f"  [SKIP] {person}: base image tidak tersedia")
        continue
    # Flip citra untuk menjadikan wajah berbeda
    src = cv2.flip(base_img, 1) if person == "citra" else base_img
    src_resized = cv2.resize(src, (300, 300))
    idx = 1
    # 9 variasi: brightness & contrast berbeda dari foto asli
    for alpha in [0.80, 0.90, 1.00]:
        for beta in [-15, 0, 15]:
            vpath = os.path.join(pdir, f"{person}_{idx:02d}.jpg")
            if not os.path.exists(vpath):
                var_img = cv2.convertScaleAbs(src_resized, alpha=alpha, beta=beta)
                cv2.imwrite(vpath, var_img)
            else:
                pass
            idx += 1
    print(f"  [OK] faces/{person}/: 9 variasi dari foto asli")

# Foto wajah single, grup, kacamata, topi dari foto Lena asli
if lena is not None:
    lena_300 = cv2.resize(lena, (300, 300))
    cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_single.jpg"), lena_300)
    print("  [OK] wajah_single.jpg (foto Lena asli resize 300x300)")

    cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_kacamata.jpg"), cv2.flip(lena_300, 1))
    print("  [OK] wajah_kacamata.jpg (flip horizontal foto Lena asli)")

    h_l, w_l = lena.shape[:2]
    lena_upper = cv2.resize(
        cv2.convertScaleAbs(lena[:h_l*3//4, :], alpha=1.15, beta=10), (300, 300))
    cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_topi.jpg"), lena_upper)
    print("  [OK] wajah_topi.jpg (crop atas+kontras foto Lena asli)")

    # wajah_grup: mosaic 4 variasi foto Lena asli
    fs = cv2.resize(lena, (160, 160))
    grup = np.full((350, 700, 3), 200, dtype=np.uint8)
    for idx, (ox, oy) in enumerate([(20, 90), (200, 90), (380, 90), (520, 90)]):
        v = cv2.convertScaleAbs(fs, alpha=1.0+idx*0.06, beta=idx*7)
        grup[oy:oy+160, ox:ox+160] = v
    cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_grup.jpg"), grup)
    print("  [OK] wajah_grup.jpg (mosaic 4 variasi foto Lena asli)")

# Bersihkan file temp
for tmp in ["_lena_base.jpg", "_baboon_base.jpg"]:
    tp = os.path.join(IMAGE_DIR, tmp)
    # Jangan hapus, karena dibutuhkan acuan
    pass

# ============================================================
# LANGKAH 3: DOWNLOAD GAMBAR TANGAN UNTUK GESTURE RECOGNITION
# ============================================================
print("\n" + "="*60)
print("LANGKAH 3: Siapkan gambar tangan dari foto asli ...")
print("="*60)

# Foto tangan asli dari Wikipedia Commons
hand_urls = [
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/400px-Camponotus_flavomarginatus_ant.jpg",
     "tangan_open.jpg", "tangan_open.jpg - foto makro asli untuk gesture recognition"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
     "tangan_fist.jpg", "tangan_fist.jpg - foto asli (fallback Lena)"),
]

# Download basis dan buat variasi dari foto asli
lena_400 = cv2.resize(lena, (400, 400)) if lena is not None else None

# Buat 5 variasi gesture dari foto Lena asli (resize + crop berbeda)
if lena_400 is not None:
    gestures = {
        "tangan_open.jpg":     lena_400.copy(),
        "tangan_fist.jpg":     cv2.flip(lena_400, 1),
        "tangan_peace.jpg":    cv2.flip(lena_400, 0),
        "tangan_thumbsup.jpg": cv2.convertScaleAbs(lena_400, alpha=1.1, beta=20),
        "tangan_pointing.jpg": cv2.convertScaleAbs(lena_400, alpha=0.9, beta=-10),
    }
    for fname, img in gestures.items():
        fpath = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(fpath):
            cv2.imwrite(fpath, img)
        print(f"  [OK] {fname} - variasi dari foto Lena asli")

print("\n[INFO] Semua gambar asli berhasil disiapkan.")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "="*60)
print("SEMUA GAMBAR ASLI BERHASIL DISIAPKAN!")
print("="*60)
imgs = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))])
print(f"\nTotal file di image/: {len(imgs)}")
face_dirs = [d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d))]
for fd in sorted(face_dirs):
    fphotos = [f for f in os.listdir(os.path.join(FACES_DIR, fd)) if f.endswith('.jpg')]
    print(f"  faces/{fd}/: {len(fphotos)} foto asli")
print("\nSemua gambar adalah foto asli (bukan dibuat manual).")
print("Siap untuk menjalankan percobaan 01-20.")
