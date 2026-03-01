"""
==========================================================================
PERCOBAAN 1: DETEKSI FITUR DAN FEATURE MATCHING
==========================================================================
Program ini mempelajari cara mendeteksi fitur pada gambar menggunakan
detektor SIFT dan ORB, kemudian mencocokkan fitur antara dua gambar
menggunakan BFMatcher dan FLANN-based Matcher.

Konsep utama:
- SIFT (Scale-Invariant Feature Transform) mendeteksi keypoint yang
  invarian terhadap skala dan rotasi, menghasilkan deskriptor 128-dim.
- ORB (Oriented FAST and Rotated BRIEF) adalah alternatif cepat dan
  bebas paten, menghasilkan deskriptor biner 32-byte.
- Lowe's Ratio Test menyaring match yang ambigu berdasarkan rasio
  jarak nearest neighbor pertama terhadap kedua.

Fungsi utama yang dipelajari:
- cv2.SIFT_create()               : Membuat detektor SIFT
- cv2.ORB_create()                : Membuat detektor ORB
- detectAndCompute()              : Mendeteksi keypoint & deskriptor
- cv2.BFMatcher()                 : Brute-Force Matcher
- cv2.FlannBasedMatcher()         : FLANN-based Matcher
- knnMatch()                      : K-Nearest Neighbor matching
- cv2.drawMatches()               : Menggambar garis pencocokan fitur

Hasil: Visualisasi keypoint dan garis pencocokan fitur antara dua gambar
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk membuat dan menyimpan visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan judul percobaan
print("=" * 60)
print("PERCOBAAN 1: DETEKSI FITUR DAN FEATURE MATCHING")
print("=" * 60)

# ============================================================
# 1. Memuat gambar input
# ============================================================

# Mendefinisikan path gambar pertama (gambar fitur asli)
path_gambar1 = os.path.join(IMAGE_DIR, "gambar_fitur.png")

# Mendefinisikan path gambar kedua (gambar fitur yang sudah dirotasi)
path_gambar2 = os.path.join(IMAGE_DIR, "gambar_fitur_rotasi.png")

# Membaca gambar pertama dalam format BGR
gambar1 = cv2.imread(path_gambar1)

# Membaca gambar kedua dalam format BGR
gambar2 = cv2.imread(path_gambar2)

# Memeriksa apakah gambar berhasil dimuat
if gambar1 is None or gambar2 is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi dimensi gambar pertama
print(f"\n[INFO] Ukuran gambar 1: {gambar1.shape}")

# Menampilkan informasi dimensi gambar kedua
print(f"[INFO] Ukuran gambar 2: {gambar2.shape}")

# Mengkonversi gambar pertama ke grayscale untuk deteksi fitur
gray1 = cv2.cvtColor(gambar1, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kedua ke grayscale untuk deteksi fitur
gray2 = cv2.cvtColor(gambar2, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Deteksi fitur menggunakan SIFT
# ============================================================

# Menampilkan informasi tahap deteksi SIFT
print("\n--- Deteksi Fitur SIFT ---")

# Membuat objek detektor SIFT dengan parameter default
sift = cv2.SIFT_create()

# Mendeteksi keypoint dan menghitung deskriptor pada gambar 1
kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)

# Mendeteksi keypoint dan menghitung deskriptor pada gambar 2
kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)

# Menampilkan jumlah keypoint yang terdeteksi SIFT pada gambar 1
print(f"[SIFT] Keypoint gambar 1: {len(kp1_sift)}")

# Menampilkan jumlah keypoint yang terdeteksi SIFT pada gambar 2
print(f"[SIFT] Keypoint gambar 2: {len(kp2_sift)}")

# Menampilkan dimensi deskriptor SIFT (seharusnya 128)
print(f"[SIFT] Dimensi deskriptor: {des1_sift.shape[1]}")

# ============================================================
# 3. Deteksi fitur menggunakan ORB
# ============================================================

# Menampilkan informasi tahap deteksi ORB
print("\n--- Deteksi Fitur ORB ---")

# Membuat objek detektor ORB dengan maksimal 500 keypoint
orb = cv2.ORB_create(nfeatures=500)

# Mendeteksi keypoint dan menghitung deskriptor ORB pada gambar 1
kp1_orb, des1_orb = orb.detectAndCompute(gray1, None)

# Mendeteksi keypoint dan menghitung deskriptor ORB pada gambar 2
kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)

# Menampilkan jumlah keypoint yang terdeteksi ORB pada gambar 1
print(f"[ORB] Keypoint gambar 1: {len(kp1_orb)}")

# Menampilkan jumlah keypoint yang terdeteksi ORB pada gambar 2
print(f"[ORB] Keypoint gambar 2: {len(kp2_orb)}")

# Menampilkan dimensi deskriptor ORB (seharusnya 32 byte)
print(f"[ORB] Dimensi deskriptor: {des1_orb.shape[1]}")

# ============================================================
# 4. Feature matching dengan BFMatcher + Lowe's Ratio Test (SIFT)
# ============================================================

# Menampilkan informasi tahap BFMatcher
print("\n--- BFMatcher (SIFT) + Lowe's Ratio Test ---")

# Membuat BFMatcher dengan norma L2 (cocok untuk deskriptor float SIFT)
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Melakukan knnMatch dengan k=2 untuk mendapatkan 2 match terdekat
matches_bf_sift = bf_sift.knnMatch(des1_sift, des2_sift, k=2)

# Menerapkan Lowe's Ratio Test untuk menyaring match yang ambigu
good_bf_sift = []
for m, n in matches_bf_sift:
    # Jika jarak match terbaik < 0.75 * jarak match kedua, dianggap baik
    if m.distance < 0.75 * n.distance:
        good_bf_sift.append(m)

# Menampilkan jumlah total match dan match yang lolos ratio test
print(f"[BF-SIFT] Total match: {len(matches_bf_sift)}")
print(f"[BF-SIFT] Match setelah ratio test: {len(good_bf_sift)}")

# ============================================================
# 5. Feature matching dengan FLANN-based Matcher (SIFT)
# ============================================================

# Menampilkan informasi tahap FLANN Matcher
print("\n--- FLANN-based Matcher (SIFT) ---")

# Mendefinisikan parameter indeks FLANN untuk algoritma KDTree
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

# Mendefinisikan parameter pencarian FLANN
search_params = dict(checks=50)

# Membuat objek FLANN-based Matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Melakukan knnMatch dengan FLANN
matches_flann = flann.knnMatch(des1_sift, des2_sift, k=2)

# Menerapkan Lowe's Ratio Test pada hasil FLANN
good_flann = []
for m, n in matches_flann:
    # Menyaring match dengan ratio test threshold 0.75
    if m.distance < 0.75 * n.distance:
        good_flann.append(m)

# Menampilkan jumlah match FLANN setelah ratio test
print(f"[FLANN] Total match: {len(matches_flann)}")
print(f"[FLANN] Match setelah ratio test: {len(good_flann)}")

# ============================================================
# 6. Feature matching dengan BFMatcher (ORB - Hamming distance)
# ============================================================

# Menampilkan informasi tahap BFMatcher ORB
print("\n--- BFMatcher (ORB) + Lowe's Ratio Test ---")

# Membuat BFMatcher dengan norma Hamming (cocok untuk deskriptor biner ORB)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Melakukan knnMatch dengan k=2 pada deskriptor ORB
matches_bf_orb = bf_orb.knnMatch(des1_orb, des2_orb, k=2)

# Menerapkan Lowe's Ratio Test pada match ORB
good_bf_orb = []
for m, n in matches_bf_orb:
    # Menggunakan threshold 0.75 untuk filtering
    if m.distance < 0.75 * n.distance:
        good_bf_orb.append(m)

# Menampilkan jumlah match ORB setelah ratio test
print(f"[BF-ORB] Total match: {len(matches_bf_orb)}")
print(f"[BF-ORB] Match setelah ratio test: {len(good_bf_orb)}")

# ============================================================
# 7. Visualisasi hasil
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi ---")

# Mengkonversi gambar dari BGR ke RGB untuk ditampilkan matplotlib
gambar1_rgb = cv2.cvtColor(gambar1, cv2.COLOR_BGR2RGB)
gambar2_rgb = cv2.cvtColor(gambar2, cv2.COLOR_BGR2RGB)

# Menggambar keypoint SIFT pada gambar 1
img_kp1_sift = cv2.drawKeypoints(gambar1_rgb, kp1_sift, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Menggambar keypoint ORB pada gambar 1
img_kp1_orb = cv2.drawKeypoints(gambar1_rgb, kp1_orb, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Menggambar garis pencocokan BFMatcher SIFT (top 30 match)
sorted_bf_sift = sorted(good_bf_sift, key=lambda x: x.distance)[:30]
img_bf_sift = cv2.drawMatches(gambar1_rgb, kp1_sift, gambar2_rgb, kp2_sift,
                               sorted_bf_sift, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Menggambar garis pencocokan FLANN SIFT (top 30 match)
sorted_flann = sorted(good_flann, key=lambda x: x.distance)[:30]
img_flann = cv2.drawMatches(gambar1_rgb, kp1_sift, gambar2_rgb, kp2_sift,
                             sorted_flann, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Menggambar garis pencocokan BFMatcher ORB (top 30 match)
sorted_bf_orb = sorted(good_bf_orb, key=lambda x: x.distance)[:30]
img_bf_orb = cv2.drawMatches(gambar1_rgb, kp1_orb, gambar2_rgb, kp2_orb,
                              sorted_bf_orb, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Membuat figure dengan 5 subplot untuk semua visualisasi
fig, axes = plt.subplots(3, 2, figsize=(18, 20))

# Menampilkan keypoint SIFT pada subplot pertama
axes[0, 0].imshow(img_kp1_sift)
axes[0, 0].set_title(f"Keypoint SIFT ({len(kp1_sift)} titik)", fontsize=12)
axes[0, 0].axis("off")

# Menampilkan keypoint ORB pada subplot kedua
axes[0, 1].imshow(img_kp1_orb)
axes[0, 1].set_title(f"Keypoint ORB ({len(kp1_orb)} titik)", fontsize=12)
axes[0, 1].axis("off")

# Menampilkan hasil BFMatcher SIFT pada subplot ketiga
axes[1, 0].imshow(img_bf_sift)
axes[1, 0].set_title(f"BFMatcher SIFT ({len(good_bf_sift)} match)", fontsize=12)
axes[1, 0].axis("off")

# Menampilkan hasil FLANN SIFT pada subplot keempat
axes[1, 1].imshow(img_flann)
axes[1, 1].set_title(f"FLANN SIFT ({len(good_flann)} match)", fontsize=12)
axes[1, 1].axis("off")

# Menampilkan hasil BFMatcher ORB pada subplot kelima
axes[2, 0].imshow(img_bf_orb)
axes[2, 0].set_title(f"BFMatcher ORB ({len(good_bf_orb)} match)", fontsize=12)
axes[2, 0].axis("off")

# Menampilkan perbandingan statistik pada subplot keenam
axes[2, 1].axis("off")
# Membuat tabel perbandingan sebagai teks
info_text = (
    "PERBANDINGAN METODE\n"
    "=" * 40 + "\n\n"
    f"SIFT Keypoints : {len(kp1_sift)} / {len(kp2_sift)}\n"
    f"ORB Keypoints  : {len(kp1_orb)} / {len(kp2_orb)}\n\n"
    f"BF-SIFT Match  : {len(good_bf_sift)}\n"
    f"FLANN Match    : {len(good_flann)}\n"
    f"BF-ORB Match   : {len(good_bf_orb)}\n\n"
    f"SIFT Deskriptor: 128-dim (float)\n"
    f"ORB Deskriptor : 32-dim (binary)\n"
)
# Menampilkan teks perbandingan pada subplot
axes[2, 1].text(0.1, 0.5, info_text, fontsize=13, fontfamily='monospace',
                verticalalignment='center', transform=axes[2, 1].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[2, 1].set_title("Ringkasan Perbandingan", fontsize=12)

# Mengatur layout agar subplot tidak saling tumpang tindih
plt.suptitle("Percobaan 1: Deteksi Fitur dan Feature Matching", fontsize=16, fontweight='bold')
plt.tight_layout()

# Menyimpan figure ke file output
output_path = os.path.join(OUTPUT_DIR, "01_deteksi_fitur_dan_matching.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[SAVED] Hasil disimpan di: {output_path}")

# Menampilkan figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 1: DETEKSI FITUR DAN FEATURE MATCHING")
print("=" * 60)
print(f"1. SIFT mendeteksi {len(kp1_sift)} keypoint (gambar 1), {len(kp2_sift)} (gambar 2)")
print(f"2. ORB mendeteksi {len(kp1_orb)} keypoint (gambar 1), {len(kp2_orb)} (gambar 2)")
print(f"3. SIFT deskriptor: 128-dim float, ORB: 32-dim biner")
print(f"4. BFMatcher SIFT: {len(good_bf_sift)} match lolos Lowe's ratio test")
print(f"5. FLANN Matcher : {len(good_flann)} match lolos Lowe's ratio test")
print(f"6. BFMatcher ORB : {len(good_bf_orb)} match lolos Lowe's ratio test")
print(f"7. Lowe's Ratio Test (threshold=0.75) menyaring match ambigu")
print(f"8. BFMatcher cocok untuk dataset kecil, FLANN untuk dataset besar")
print("=" * 60)
