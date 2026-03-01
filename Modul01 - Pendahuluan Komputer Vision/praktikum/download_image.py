"""
==========================================================================
DOWNLOAD IMAGE - MODUL 01: PENDAHULUAN KOMPUTER VISION
==========================================================================
Script ini mendownload semua gambar NYATA yang dibutuhkan untuk 20
percobaan Modul 01 dari OpenCV GitHub dan picsum.photos.

Semua gambar adalah foto asli (bukan gambar sintetis/generated).

Pemetaan percobaan -> gambar nyata:
  01 loading & tampilkan    -> foto_kucing.jpg      (portrait kucing asli)
  02 properti gambar        -> foto_kucing.jpg      (same portrait)
  03 konversi ruang warna   -> foto_bunga2.jpg      (bunga warna-warni)
  04 akses & manipulasi     -> foto_kucing.jpg      (detail bulu kucing)
  05 operasi aritmatika     -> foto_bunga.jpg + foto_hewan.jpg
  06 operasi bitwise        -> foto_alam.jpg + foto_orang.jpg
  07 gambar geometris       -> (canvas kosong, tidak perlu input)
  08 menulis teks           -> foto_kucing.jpg      (kanvas foto)
  09 region of interest     -> foto_pasar.jpg       (scene ramai)
  10 resize & scaling       -> foto_tekstur.jpg     (foto tekstur)
  11 cropping               -> foto_alam2.jpg       (lanskap luas)
  12 rotasi                 -> foto_burung.jpg      (burung terbang)
  13 flip                   -> foto_arsitektur.jpg  (bangunan simetris)
  14 padding/border         -> fruits.jpg           (buah berwarna)
  15 splitting channel      -> foto_bunga2.jpg      (bunga warna-warni)
  16 blending               -> foto_malam.jpg + foto_siang.jpg
  17 brightness & contrast  -> foto_gelap.jpg       (foto minim cahaya)
  18 histogram              -> lena.jpg             (potret klasik OpenCV)
  19 masking                -> foto_orang2.jpg      (potret orang)
  20 menyimpan format       -> lena.jpg             (potret klasik OpenCV)

Jalankan script ini PERTAMA KALI sebelum menjalankan percobaan lainnya.
==========================================================================
"""

import os
import ssl
import urllib.request

# ============================================================
# Membuat struktur folder
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(IMAGE_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MODUL 01 - DOWNLOAD GAMBAR NYATA")
print("=" * 60)
print(f"[INFO] Folder image  : {IMAGE_DIR}")
print(f"[INFO] Folder output : {OUTPUT_DIR}")

# ============================================================
# Helper download
# ============================================================
def _download(filename, url, timeout=30):
    """Download satu file dari URL ke IMAGE_DIR. Skip jika sudah ada."""
    dest = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(dest):
        size_kb = os.path.getsize(dest) / 1024
        print(f"  skip  {filename} (sudah ada {size_kb:.0f} KB)")
        return True
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        size_kb = len(data) / 1024
        print(f"  OK   {filename}  ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  FAIL {filename}: {e}")
        return False


# ============================================================
# Daftar gambar nyata & URL
# ============================================================
IMAGES = {
    # OpenCV classic test images
    "lena.jpg":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "baboon.jpg":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
    "fruits.jpg":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
    "building.jpg":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
    # Foto tematik dari picsum.photos
    "foto_kucing.jpg":
        "https://picsum.photos/seed/cat0101/640/480",
    "foto_alam.jpg":
        "https://picsum.photos/seed/nature1101/640/480",
    "foto_alam2.jpg":
        "https://picsum.photos/seed/nature1116/640/480",
    "foto_kota.jpg":
        "https://picsum.photos/seed/city1102/640/480",
    "foto_orang.jpg":
        "https://picsum.photos/seed/people1103/480/640",
    "foto_orang2.jpg":
        "https://picsum.photos/seed/people1104/640/480",
    "foto_bunga.jpg":
        "https://picsum.photos/seed/flower1105/640/480",
    "foto_bunga2.jpg":
        "https://picsum.photos/seed/flower1106/640/480",
    "foto_hewan.jpg":
        "https://picsum.photos/seed/animal1107/640/480",
    "foto_tekstur.jpg":
        "https://picsum.photos/seed/texture1108/640/480",
    "foto_lanskap.jpg":
        "https://picsum.photos/seed/landscape1109/640/480",
    "foto_gelap.jpg":
        "https://picsum.photos/seed/night1110/640/480",
    "foto_malam.jpg":
        "https://picsum.photos/seed/night1111/640/480",
    "foto_siang.jpg":
        "https://picsum.photos/seed/day1112/640/480",
    "foto_arsitektur.jpg":
        "https://picsum.photos/seed/arch1113/640/480",
    "foto_pasar.jpg":
        "https://picsum.photos/seed/market1114/640/480",
    "foto_burung.jpg":
        "https://picsum.photos/seed/bird1115/640/480",
}

# ============================================================
# Eksekusi download
# ============================================================
print(f"\n[INFO] Mendownload {len(IMAGES)} gambar nyata...\n")
ok, fail = 0, 0
for fname, url in IMAGES.items():
    if _download(fname, url):
        ok += 1
    else:
        fail += 1

# ============================================================
# Verifikasi
# ============================================================
print("\n" + "=" * 60)
print("[INFO] Daftar gambar di folder 'image/':")
for f in sorted(os.listdir(IMAGE_DIR)):
    path = os.path.join(IMAGE_DIR, f)
    kb   = os.path.getsize(path) / 1024
    print(f"  {f:<30} {kb:>7.1f} KB")

print(f"\n  Total berhasil : {ok}")
print(f"  Total gagal    : {fail}")
print(f"\nSetup selesai! Silakan jalankan percobaan 01-20.")
print("=" * 60)