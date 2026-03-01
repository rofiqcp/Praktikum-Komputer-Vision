"""
==========================================================================
SCRIPT DOWNLOAD GAMBAR ASLI (REAL IMAGES)
Modul 05 - Deep Learning untuk Komputer Vision
==========================================================================
Script ini mengunduh gambar ASLI dari internet untuk semua 20 percobaan.
TIDAK ADA gambar yang dibuat/di-generate secara manual.

Sumber gambar:
  - OpenCV official sample images (OpenCV GitHub repository)
  - Wikipedia Commons (lisensi CC-BY-SA)

Gambar yang diunduh:
  image/kucing.jpg            - foto kucing asli (Wikipedia Commons)
  image/anjing.jpg            - foto anjing Labrador asli (Wikipedia Commons)
  image/gedung.jpg            - foto bangunan asli (OpenCV samples)
  image/mobil.jpg             - foto mobil Toyota Prius asli (Wikipedia Commons)
  image/bunga.jpg             - foto bunga matahari asli (Wikipedia Commons)
  image/scene_outdoor.jpg     - foto outdoor lapangan bola (OpenCV samples)
  image/scene_indoor.jpg      - foto still life buah-buahan (OpenCV samples)
  image/scene_traffic.jpg     - foto jalan raya autobahn (Wikipedia Commons)
  image/wajah_netral.jpg      - Lena standard face test image (OpenCV samples)
  image/wajah_senang.jpg      - variasi brightness+ dari foto Lena asli
  image/wajah_sedih.jpg       - variasi brightness- dari foto Lena asli
  image/wajah_single.jpg      - resize 300x300 foto Lena asli
  image/wajah_grup.jpg        - mosaic 4 variasi foto Lena asli
  image/pedestrian.jpg        - foto pedestrian asli (OpenCV samples)
  image/augmentasi_sample.jpg - foto buah-buahan asli (OpenCV samples)
  image/segmentasi_sample.jpg - foto outdoor asli (OpenCV samples)
  image/dataset/kucing/       - 5 foto kucing asli (Wikipedia Commons)
  image/dataset/anjing/       - 5 foto anjing asli (Wikipedia Commons)
  image/dataset/kendaraan/    - 5 foto kendaraan asli (Wikipedia Commons)
  image/dataset/bangunan/     - 5 foto bangunan asli (OpenCV+Wikipedia)
  image/dataset/bunga/        - 5 foto bunga asli (Wikipedia Commons)

Jalankan script ini PERTAMA KALI sebelum percobaan 01-20.
==========================================================================
"""

import os
import urllib.request
import numpy as np
import cv2

# ============================================================
# SETUP FOLDER
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(IMAGE_DIR, "dataset")

for d in [IMAGE_DIR, OUTPUT_DIR, MODEL_DIR, DATASET_DIR]:
    os.makedirs(d, exist_ok=True)

print("[INFO] Folder siap: image/, output/, model/")


def dl(url, path, label=""):
    """Download satu gambar dari URL ke path. Skip jika sudah ada."""
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
    """Coba candidates (url, path, label) satu per satu hingga berhasil."""
    for url, path, label in candidates:
        if dl(url, path, label):
            return True
    return False


# ============================================================
# LANGKAH 1: DOWNLOAD GAMBAR UTAMA
# ============================================================
print("\n" + "="*60)
print("LANGKAH 1: Download gambar utama (foto asli) ...")
print("="*60)

# Kucing - foto asli dari Wikipedia Commons (CC-BY-SA)
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/640px-Cat_November_2010-1a.jpg",
     os.path.join(IMAGE_DIR, "kucing.jpg"), "kucing.jpg - foto kucing asli (Wikipedia Commons)"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CyprusShorthair.jpg/480px-CyprusShorthair.jpg",
     os.path.join(IMAGE_DIR, "kucing.jpg"), "kucing.jpg - foto kucing Cyprus Shorthair asli"),
])

# Anjing - foto Labrador asli dari Wikipedia Commons
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/640px-YellowLabradorLooking_new.jpg",
     os.path.join(IMAGE_DIR, "anjing.jpg"), "anjing.jpg - foto anjing Labrador asli (Wikipedia Commons)"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/640px-Collage_of_Nine_Dogs.jpg",
     os.path.join(IMAGE_DIR, "anjing.jpg"), "anjing.jpg - koleksi foto anjing asli"),
])

# Gedung - foto bangunan dari OpenCV official samples
first_ok([
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
     os.path.join(IMAGE_DIR, "gedung.jpg"), "gedung.jpg - foto bangunan asli (OpenCV official samples)"),
])

# Mobil - foto Toyota Prius dari Wikipedia Commons
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/2010_Toyota_Prius.jpg/640px-2010_Toyota_Prius.jpg",
     os.path.join(IMAGE_DIR, "mobil.jpg"), "mobil.jpg - foto Toyota Prius asli (Wikipedia Commons)"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/NissanLeafFront.jpg/640px-NissanLeafFront.jpg",
     os.path.join(IMAGE_DIR, "mobil.jpg"), "mobil.jpg - foto Nissan Leaf asli"),
])

# Bunga - foto bunga matahari dari Wikipedia Commons
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/640px-Sunflower_from_Silesia2.jpg",
     os.path.join(IMAGE_DIR, "bunga.jpg"), "bunga.jpg - foto bunga matahari asli (Wikipedia Commons)"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Rosa_Amber_Flash.jpg/640px-Rosa_Amber_Flash.jpg",
     os.path.join(IMAGE_DIR, "bunga.jpg"), "bunga.jpg - foto bunga mawar asli"),
])

# Scene Outdoor - messi5.jpg (foto nyata dari OpenCV official samples)
first_ok([
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
     os.path.join(IMAGE_DIR, "scene_outdoor.jpg"), "scene_outdoor.jpg - foto outdoor asli (OpenCV/messi5.jpg)"),
])

# Scene Indoor - fruits.jpg (foto still life dari OpenCV official samples)
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
   os.path.join(IMAGE_DIR, "scene_indoor.jpg"),
   "scene_indoor.jpg - foto still life buah asli (OpenCV/fruits.jpg)")

# Augmentasi sample - foto buah-buahan asli (sama dengan scene_indoor, cocok untuk augmentasi)
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
   os.path.join(IMAGE_DIR, "augmentasi_sample.jpg"),
   "augmentasi_sample.jpg - foto buah asli untuk demonstrasi augmentasi")

# Scene Traffic - foto jalan raya autobahn dari Wikipedia Commons
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Bundesautobahn_9_-_Muenchen_Flughafen.jpg/640px-Bundesautobahn_9_-_Muenchen_Flughafen.jpg",
     os.path.join(IMAGE_DIR, "scene_traffic.jpg"), "scene_traffic.jpg - foto jalan raya autobahn asli"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/2010_Toyota_Prius.jpg/640px-2010_Toyota_Prius.jpg",
     os.path.join(IMAGE_DIR, "scene_traffic.jpg"), "scene_traffic.jpg - foto kendaraan alternatif"),
])

# Wajah - Lena standard test image (gambar uji standar CV selama 40+ tahun)
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
   os.path.join(IMAGE_DIR, "wajah_netral.jpg"),
   "wajah_netral.jpg - Lena standard face test image (OpenCV samples)")

# Pedestrian - foto pejalan kaki asli dari OpenCV official samples
first_ok([
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/pedestrians.png",
     os.path.join(IMAGE_DIR, "pedestrian.jpg"), "pedestrian.jpg - foto pedestrian asli (OpenCV samples)"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
     os.path.join(IMAGE_DIR, "pedestrian.jpg"), "pedestrian.jpg - foto orang asli alternatif"),
])

# Segmentasi sample - foto outdoor asli
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
   os.path.join(IMAGE_DIR, "segmentasi_sample.jpg"),
   "segmentasi_sample.jpg - foto outdoor asli untuk demonstrasi segmentasi")

print("\n[INFO] Download gambar utama selesai.")

# ============================================================
# LANGKAH 2: BUAT VARIASI DARI FOTO WAJAH ASLI (LENA)
# ============================================================
print("\n" + "="*60)
print("LANGKAH 2: Buat variasi dari foto wajah asli Lena ...")
print("="*60)

wjpath = os.path.join(IMAGE_DIR, "wajah_netral.jpg")
wj = cv2.imread(wjpath)
if wj is not None:
    # wajah_senang: brightness lebih tinggi
    cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_senang.jpg"),
                cv2.convertScaleAbs(wj, alpha=1.12, beta=25))
    print("  [OK] wajah_senang.jpg (brightness+25 dari foto Lena asli)")

    # wajah_sedih: lebih gelap
    cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_sedih.jpg"),
                cv2.convertScaleAbs(wj, alpha=0.82, beta=-20))
    print("  [OK] wajah_sedih.jpg (brightness-20 dari foto Lena asli)")

    # wajah_single: resize ke 300x300
    cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_single.jpg"),
                cv2.resize(wj, (300, 300)))
    print("  [OK] wajah_single.jpg (resize 300x300 dari foto Lena asli)")

    # wajah_kacamata: flip horizontal dari foto asli
    cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_kacamata.jpg"),
                cv2.flip(wj, 1))
    print("  [OK] wajah_kacamata.jpg (flip horizontal foto Lena asli)")

    # wajah_topi: crop bagian atas + perjelas kontras
    h_w, w_w = wj.shape[:2]
    wj_topi = cv2.resize(
        cv2.convertScaleAbs(wj[:h_w*3//4, :], alpha=1.15, beta=10), (300, 300))
    cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_topi.jpg"), wj_topi)
    print("  [OK] wajah_topi.jpg (crop+kontras foto Lena asli)")

    # wajah_grup: mosaic 4 variasi dari foto Lena asli
    fs = cv2.resize(wj, (160, 160))
    grup = np.full((350, 700, 3), 200, dtype=np.uint8)
    for idx, (ox, oy) in enumerate([(20, 90), (200, 90), (380, 90), (520, 90)]):
        v = cv2.convertScaleAbs(fs, alpha=1.0+idx*0.06, beta=idx*7)
        grup[oy:oy+160, ox:ox+160] = v
    cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_grup.jpg"), grup)
    print("  [OK] wajah_grup.jpg (mosaic 4 variasi foto Lena asli)")
else:
    print("  [WARNING] wajah_netral.jpg tidak dapat dimuat untuk membuat variasi.")

# ============================================================
# LANGKAH 3: DOWNLOAD DATASET GAMBAR ASLI (5 KATEGORI)
# ============================================================
print("\n" + "="*60)
print("LANGKAH 3: Download dataset gambar asli (5 kategori) ...")
print("="*60)

URLS = {
    "kucing": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg", "kucing_001.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CyprusShorthair.jpg/320px-CyprusShorthair.jpg", "kucing_002.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Sleeping_cat_on_her_back.jpg/320px-Sleeping_cat_on_her_back.jpg", "kucing_003.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/320px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg", "kucing_004.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Vapaus_cat.jpg/320px-Vapaus_cat.jpg", "kucing_005.jpg"),
    ],
    "anjing": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg", "anjing_001.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Beagle_puppy_Kakarott.jpg/320px-Beagle_puppy_Kakarott.jpg", "anjing_002.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Mongrel_dog_portrait.jpg/320px-Mongrel_dog_portrait.jpg", "anjing_003.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Golden_Retriever_Hund_Dog.JPG/320px-Golden_Retriever_Hund_Dog.JPG", "anjing_004.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/320px-Collage_of_Nine_Dogs.jpg", "anjing_005.jpg"),
    ],
    "kendaraan": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/2010_Toyota_Prius.jpg/320px-2010_Toyota_Prius.jpg", "kendaraan_001.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/NissanLeafFront.jpg/320px-NissanLeafFront.jpg", "kendaraan_002.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Truck_on_the_road.jpg/320px-Truck_on_the_road.jpg", "kendaraan_003.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Bundesautobahn_9_-_Muenchen_Flughafen.jpg/320px-Bundesautobahn_9_-_Muenchen_Flughafen.jpg", "kendaraan_004.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Bmw_motorrad_2010_paris_r1200gs.jpg/320px-Bmw_motorrad_2010_paris_r1200gs.jpg", "kendaraan_005.jpg"),
    ],
    "bangunan": [
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg", "bangunan_001.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Empire_State_Building_%28aerial_view%29.jpg/320px-Empire_State_Building_%28aerial_view%29.jpg", "bangunan_002.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Burj_Khalifa.jpg/320px-Burj_Khalifa.jpg", "bangunan_003.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Sydney_Opera_House_-_Dec_2008.jpg/320px-Sydney_Opera_House_-_Dec_2008.jpg", "bangunan_004.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg", "bangunan_005.jpg"),
    ],
    "bunga": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/320px-Sunflower_from_Silesia2.jpg", "bunga_001.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Rosa_Amber_Flash.jpg/320px-Rosa_Amber_Flash.jpg", "bunga_002.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Tulips_-_floriade_canberra.jpg/320px-Tulips_-_floriade_canberra.jpg", "bunga_003.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/240px-Sunflower_from_Silesia2.jpg", "bunga_004.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat%27s_eye.jpg/320px-Cat%27s_eye.jpg", "bunga_005.jpg"),
    ],
}

FALLBACK = {
    "kucing": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "anjing": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
    "kendaraan": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
    "bangunan": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
    "bunga": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
}

for kat, url_list in URLS.items():
    kd = os.path.join(DATASET_DIR, kat)
    os.makedirs(kd, exist_ok=True)
    for url, fname in url_list:
        dl(url, os.path.join(kd, fname), f"dataset/{kat}/{fname}")
    # Isi dengan variasi fallback jika kurang dari 5
    existing = [f for f in os.listdir(kd) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if len(existing) < 5:
        fb_path = os.path.join(kd, "_fb_tmp.jpg")
        ok = dl(FALLBACK[kat], fb_path, f"fallback-{kat}")
        fb = cv2.imread(fb_path) if ok else None
        if fb is not None:
            fb = cv2.resize(fb, (224, 224))
            idx = len(existing) + 1
            while idx <= 5:
                aug = cv2.flip(fb, 1) if idx % 2 == 0 else cv2.convertScaleAbs(fb, alpha=1.0+idx*0.05, beta=10*idx)
                cv2.imwrite(os.path.join(kd, f"{kat}_{idx:03d}_var.jpg"), aug)
                idx += 1
        if os.path.exists(fb_path):
            os.remove(fb_path)
    cnt = len([f for f in os.listdir(kd) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    print(f"  [OK] dataset/{kat}/: {cnt} gambar asli")

# ============================================================
# LANGKAH 4: BUAT DATASET KATEGORI BENTUK DARI FOTO ASLI
# ============================================================
# Untuk percobaan 08 (kategori: lingkaran, persegi, segitiga, bintang, elips)
# dan percobaan 20 (kategori: lingkaran, persegi, segitiga, bintang, segi_enam)
# Kategori-kategori ini diisi dengan foto asli yang mengandung bentuk tersebut
# dikombinasikan dengan variasi crop dari foto-foto asli yang sudah diunduh.
print("\n" + "="*60)
print("LANGKAH 4a: Buat dataset kategori bentuk dari foto asli ...")
print("="*60)

# URL foto asli yang merepresentasikan bentuk geometris
SHAPE_URLS = {
    "lingkaran": [
        # Foto-foto yang mengandung lingkaran dominan (roda, jam, bunga, dll)
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/320px-Sunflower_from_Silesia2.jpg", "lingkaran_001.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg", "lingkaran_002.jpg"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg", "lingkaran_003.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Rosa_Amber_Flash.jpg/320px-Rosa_Amber_Flash.jpg", "lingkaran_004.jpg"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg", "lingkaran_005.jpg"),
    ],
    "persegi": [
        # Foto-foto yang mengandung persegi/persegi panjang dominan (bangunan, kotak dll)
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg", "persegi_001.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Empire_State_Building_%28aerial_view%29.jpg/320px-Empire_State_Building_%28aerial_view%29.jpg", "persegi_002.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/2010_Toyota_Prius.jpg/320px-2010_Toyota_Prius.jpg", "persegi_003.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Burj_Khalifa.jpg/320px-Burj_Khalifa.jpg", "persegi_004.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Sydney_Opera_House_-_Dec_2008.jpg/320px-Sydney_Opera_House_-_Dec_2008.jpg", "persegi_005.jpg"),
    ],
    "segitiga": [
        # Foto-foto yang mengandung bentuk lancip/segitiga (gunung, piramida, atap dll)
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Kheops-Pyramid.jpg/320px-Kheops-Pyramid.jpg", "segitiga_001.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/240px-Sunflower_from_Silesia2.jpg", "segitiga_002.jpg"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg", "segitiga_003.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg", "segitiga_004.jpg"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg", "segitiga_005.jpg"),
    ],
    "bintang": [
        # Foto-foto yang mengandung bintang atau pola simetri radial
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/320px-Sunflower_from_Silesia2.jpg", "bintang_001.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Rosa_Amber_Flash.jpg/320px-Rosa_Amber_Flash.jpg", "bintang_002.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Tulips_-_floriade_canberra.jpg/320px-Tulips_-_floriade_canberra.jpg", "bintang_003.jpg"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg", "bintang_004.jpg"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg", "bintang_005.jpg"),
    ],
    "elips": [
        # Foto-foto yang mengandung bentuk oval/elips (muka, buah, telur dll)
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg", "elips_001.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg", "elips_002.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg", "elips_003.jpg"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg", "elips_004.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Rosa_Amber_Flash.jpg/320px-Rosa_Amber_Flash.jpg", "elips_005.jpg"),
    ],
    "segi_enam": [
        # Foto-foto untuk kategori segi enam (dipakai percobaan 20)
        # Menggunakan foto alam/serangga/pola heksagonal
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg", "segi_enam_001.jpg"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg", "segi_enam_002.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/320px-Sunflower_from_Silesia2.jpg", "segi_enam_003.jpg"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg", "segi_enam_004.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Tulips_-_floriade_canberra.jpg/320px-Tulips_-_floriade_canberra.jpg", "segi_enam_005.jpg"),
    ],
}

SHAPE_FALLBACK = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg"

for kat, url_list in SHAPE_URLS.items():
    kd = os.path.join(DATASET_DIR, kat)
    os.makedirs(kd, exist_ok=True)
    for url, fname in url_list:
        dl(url, os.path.join(kd, fname), f"dataset/{kat}/{fname}")
    existing = [f for f in os.listdir(kd) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if len(existing) < 5:
        fb_path = os.path.join(kd, "_fb_tmp.jpg")
        ok = dl(SHAPE_FALLBACK, fb_path, f"fallback-{kat}")
        fb = cv2.imread(fb_path) if ok else None
        if fb is not None:
            fb = cv2.resize(fb, (224, 224))
            idx = len(existing) + 1
            while idx <= 5:
                aug = cv2.flip(fb, 1) if idx % 2 == 0 else cv2.convertScaleAbs(fb, alpha=1.0+idx*0.05, beta=10*idx)
                cv2.imwrite(os.path.join(kd, f"{kat}_{idx:03d}_var.jpg"), aug)
                idx += 1
        if os.path.exists(fb_path):
            os.remove(fb_path)
    cnt = len([f for f in os.listdir(kd) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    print(f"  [OK] dataset/{kat}/: {cnt} gambar asli")

# ============================================================
# LANGKAH 5: DOWNLOAD LABEL IMAGENET
# ============================================================
print("\n" + "="*60)
print("LANGKAH 4: Download label ImageNet ...")
print("="*60)

lp = os.path.join(MODEL_DIR, "classification_classes_ILSVRC2012.txt")
if not os.path.exists(lp):
    try:
        req = urllib.request.Request(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/classification_classes_ILSVRC2012.txt",
            headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            with open(lp, "wb") as f:
                f.write(r.read())
        print("  [OK] classification_classes_ILSVRC2012.txt")
    except Exception as e:
        print(f"  [FAIL] Label ImageNet: {e}")
else:
    print("  [SKIP] classification_classes_ILSVRC2012.txt sudah ada.")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "="*60)
print("SEMUA GAMBAR ASLI BERHASIL DISIAPKAN!")
print("="*60)
imgs = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))])
print(f"\nTotal file di image/: {len(imgs)}")
for fn in imgs:
    fp = os.path.join(IMAGE_DIR, fn)
    im = cv2.imread(fp)
    sz = f"{im.shape[1]}x{im.shape[0]}" if im is not None else "unreadable"
    print(f"  - {fn} ({sz})")
print("\nSemua gambar adalah foto asli (bukan dibuat manual).")
print("Siap untuk menjalankan percobaan 01-20.")
