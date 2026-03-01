"""
==========================================================================
PERCOBAAN 6: BRUTE-FORCE FEATURE MATCHING
==========================================================================
Program ini mempelajari cara mencocokkan fitur antar dua gambar
menggunakan Brute-Force Matcher. BF Matcher menghitung jarak antara
setiap deskriptor pada gambar pertama dengan semua deskriptor pada
gambar kedua secara exhaustive, kemudian memilih pasangan terdekat.

Konsep yang dipelajari:
- Brute-Force matching: pencocokan exhaustive seluruh deskriptor
- Norma jarak: NORM_L2 untuk deskriptor float (SIFT), NORM_HAMMING
  untuk deskriptor biner (ORB)
- crossCheck: memfilter match yang tidak saling terbaik
- Sortir match berdasarkan jarak (kualitas pencocokan)
- Perbandingan SIFT vs ORB matching

Fungsi utama yang dipelajari:
- cv2.BFMatcher()       : Membuat objek Brute-Force matcher
- bf.match()            : Mencocokkan deskriptor secara one-to-one
- cv2.drawMatches()     : Menggambar garis pencocokan antar dua gambar
- sorted(matches, key)  : Mengurutkan match berdasarkan jarak

Hasil: Visualisasi pencocokan fitur BF dengan SIFT dan ORB
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan pencocokan fitur
import cv2

# Mengimpor NumPy untuk operasi array dan statistik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu pemrosesan
import time

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
print("PERCOBAAN 6: BRUTE-FORCE FEATURE MATCHING")
print("=" * 60)

# ============================================================
# 1. Memuat Pasangan Gambar Overlapping
# ============================================================

# Membaca gambar kiri dari pasangan overlapping
img_left = cv2.imread(os.path.join(IMAGE_DIR, "scene_left.jpg"))

# Membaca gambar kanan dari pasangan overlapping
img_right = cv2.imread(os.path.join(IMAGE_DIR, "scene_right.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_left is None or img_right is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi ukuran gambar
print(f"[INFO] Ukuran scene_left: {img_left.shape}")
print(f"[INFO] Ukuran scene_right: {img_right.shape}")

# Mengkonversi gambar kiri ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kanan ke grayscale
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Deteksi SIFT dan BF Matching
# ============================================================

# Membuat detektor SIFT
sift = cv2.SIFT_create()

# Mendeteksi keypoints dan descriptor pada gambar kiri
kp_left_sift, desc_left_sift = sift.detectAndCompute(gray_left, None)

# Mendeteksi keypoints dan descriptor pada gambar kanan
kp_right_sift, desc_right_sift = sift.detectAndCompute(gray_right, None)

# Menampilkan jumlah keypoint
print(f"\n[SIFT] Keypoints kiri: {len(kp_left_sift)}")
print(f"[SIFT] Keypoints kanan: {len(kp_right_sift)}")

# Membuat BF Matcher dengan norma L2 (cocok untuk SIFT) dan crossCheck=True
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Mencatat waktu mulai matching
t_start = time.time()

# Mencocokkan descriptor
matches_sift = bf_sift.match(desc_left_sift, desc_right_sift)

# Menghitung waktu matching
match_time_sift = time.time() - t_start

# Mengurutkan matches berdasarkan jarak (kualitas terbaik di depan)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)

# Menampilkan informasi matching
print(f"[SIFT] Total matches: {len(matches_sift)}")
print(f"[SIFT] Waktu matching: {match_time_sift*1000:.2f} ms")

# Menghitung statistik jarak match
distances_sift = [m.distance for m in matches_sift]

# Menampilkan statistik jarak
print(f"[SIFT] Jarak - min: {min(distances_sift):.2f}, max: {max(distances_sift):.2f}, "
      f"mean: {np.mean(distances_sift):.2f}")

# ============================================================
# 3. Visualisasi SIFT Matching dengan Berbagai Jumlah Match
# ============================================================

# Mendefinisikan jumlah match yang akan ditampilkan
match_counts = [20, 50, 100]

# Membuat figure untuk berbagai jumlah match
fig, axes = plt.subplots(len(match_counts), 1, figsize=(16, 5*len(match_counts)))

# Melakukan iterasi untuk setiap jumlah match
for i, n_match in enumerate(match_counts):
    # Mengambil top-N matches
    top_matches = matches_sift[:n_match]

    # Menggambar matches
    img_matches = cv2.drawMatches(img_left, kp_left_sift, img_right, kp_right_sift,
                                   top_matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Menampilkan pada subplot
    axes[i].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))

    # Menghitung rata-rata jarak top-N
    avg_dist = np.mean([m.distance for m in top_matches])

    # Memberikan judul
    axes[i].set_title(f"Top {n_match} SIFT Matches (avg distance: {avg_dist:.2f})", fontsize=12)

    # Menonaktifkan sumbu
    axes[i].axis('off')

# Memberikan judul utama
fig.suptitle("BF Matching dengan SIFT (NORM_L2, crossCheck)", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi SIFT matching
plt.savefig(os.path.join(OUTPUT_DIR, "06_bf_match_sift.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 06_bf_match_sift.png")

# Menutup figure
plt.close()

# ============================================================
# 4. Histogram Jarak Match SIFT
# ============================================================

# Membuat figure untuk histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Menggambar histogram jarak match SIFT
axes[0].hist(distances_sift, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

# Memberikan judul dan label
axes[0].set_title(f"Distribusi Jarak Match SIFT\n({len(matches_sift)} matches)", fontsize=12)
axes[0].set_xlabel("Jarak (L2)", fontsize=11)
axes[0].set_ylabel("Frekuensi", fontsize=11)

# Menambahkan garis rata-rata
axes[0].axvline(np.mean(distances_sift), color='red', linestyle='--',
                label=f"Mean: {np.mean(distances_sift):.2f}")

# Menambahkan garis median
axes[0].axvline(np.median(distances_sift), color='green', linestyle='--',
                label=f"Median: {np.median(distances_sift):.2f}")

# Menampilkan legend
axes[0].legend()

# Mengaktifkan grid
axes[0].grid(True, alpha=0.3)

# Menggambar cumulative distribution
sorted_dists = sorted(distances_sift)

# Menghitung CDF
cdf_y = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)

# Menggambar CDF
axes[1].plot(sorted_dists, cdf_y, color='steelblue', linewidth=2)

# Memberikan judul
axes[1].set_title("Cumulative Distribution Function (CDF)", fontsize=12)
axes[1].set_xlabel("Jarak (L2)", fontsize=11)
axes[1].set_ylabel("Proporsi Kumulatif", fontsize=11)
axes[1].grid(True, alpha=0.3)

# Memberikan judul utama
fig.suptitle("Analisis Jarak Match SIFT", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan histogram
plt.savefig(os.path.join(OUTPUT_DIR, "06_match_histogram.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"[SAVED] 06_match_histogram.png")

# Menutup figure
plt.close()

# ============================================================
# 5. Deteksi ORB dan BF Matching
# ============================================================

# Membuat detektor ORB
orb = cv2.ORB_create(nfeatures=1000)

# Mendeteksi keypoints dan descriptor ORB pada gambar kiri
kp_left_orb, desc_left_orb = orb.detectAndCompute(gray_left, None)

# Mendeteksi keypoints dan descriptor ORB pada gambar kanan
kp_right_orb, desc_right_orb = orb.detectAndCompute(gray_right, None)

# Menampilkan jumlah keypoint ORB
print(f"\n[ORB] Keypoints kiri: {len(kp_left_orb)}")
print(f"[ORB] Keypoints kanan: {len(kp_right_orb)}")

# Membuat BF Matcher dengan NORM_HAMMING (cocok untuk deskriptor biner) dan crossCheck
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Mencatat waktu mulai matching ORB
t_start = time.time()

# Mencocokkan descriptor ORB
matches_orb = bf_orb.match(desc_left_orb, desc_right_orb)

# Menghitung waktu matching ORB
match_time_orb = time.time() - t_start

# Mengurutkan matches ORB berdasarkan jarak
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

# Menampilkan informasi matching ORB
print(f"[ORB] Total matches: {len(matches_orb)}")
print(f"[ORB] Waktu matching: {match_time_orb*1000:.2f} ms")

# Menghitung statistik jarak ORB
distances_orb = [m.distance for m in matches_orb]

# Menampilkan statistik
print(f"[ORB] Jarak - min: {min(distances_orb):.2f}, max: {max(distances_orb):.2f}, "
      f"mean: {np.mean(distances_orb):.2f}")

# Membuat figure untuk ORB matching
fig, axes = plt.subplots(len(match_counts), 1, figsize=(16, 5*len(match_counts)))

# Melakukan iterasi untuk setiap jumlah match
for i, n_match in enumerate(match_counts):
    # Mengambil top-N matches ORB
    top_matches_orb = matches_orb[:min(n_match, len(matches_orb))]

    # Menggambar matches ORB
    img_matches_orb = cv2.drawMatches(img_left, kp_left_orb, img_right, kp_right_orb,
                                       top_matches_orb, None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Menampilkan pada subplot
    axes[i].imshow(cv2.cvtColor(img_matches_orb, cv2.COLOR_BGR2RGB))

    # Menghitung rata-rata jarak
    avg_dist_orb = np.mean([m.distance for m in top_matches_orb])

    # Memberikan judul
    axes[i].set_title(f"Top {len(top_matches_orb)} ORB Matches (avg distance: {avg_dist_orb:.2f})", fontsize=12)

    # Menonaktifkan sumbu
    axes[i].axis('off')

# Memberikan judul utama
fig.suptitle("BF Matching dengan ORB (NORM_HAMMING, crossCheck)", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ORB matching
plt.savefig(os.path.join(OUTPUT_DIR, "06_bf_match_orb.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 06_bf_match_orb.png")

# Menutup figure
plt.close()

# ============================================================
# 6. Perbandingan SIFT vs ORB Matching
# ============================================================

# Menampilkan header perbandingan
print(f"\n--- Perbandingan SIFT vs ORB Matching ---")

# Membuat figure untuk perbandingan
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Menampilkan top-50 SIFT matches
top50_sift = matches_sift[:50]
img_sift_50 = cv2.drawMatches(img_left, kp_left_sift, img_right, kp_right_sift,
                               top50_sift, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Menampilkan pada subplot
axes[0, 0].imshow(cv2.cvtColor(img_sift_50, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"SIFT Top 50 Matches", fontsize=12)
axes[0, 0].axis('off')

# Menampilkan top-50 ORB matches
top50_orb = matches_orb[:min(50, len(matches_orb))]
img_orb_50 = cv2.drawMatches(img_left, kp_left_orb, img_right, kp_right_orb,
                              top50_orb, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Menampilkan pada subplot
axes[0, 1].imshow(cv2.cvtColor(img_orb_50, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"ORB Top 50 Matches", fontsize=12)
axes[0, 1].axis('off')

# Menggambar histogram jarak perbandingan
axes[1, 0].hist(distances_sift, bins=25, color='steelblue', alpha=0.7,
                label=f'SIFT (n={len(matches_sift)})', edgecolor='black')
axes[1, 0].set_title("Distribusi Jarak SIFT", fontsize=12)
axes[1, 0].set_xlabel("Jarak (L2)")
axes[1, 0].set_ylabel("Frekuensi")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Menggambar histogram jarak ORB
axes[1, 1].hist(distances_orb, bins=25, color='coral', alpha=0.7,
                label=f'ORB (n={len(matches_orb)})', edgecolor='black')
axes[1, 1].set_title("Distribusi Jarak ORB", fontsize=12)
axes[1, 1].set_xlabel("Jarak (Hamming)")
axes[1, 1].set_ylabel("Frekuensi")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Memberikan judul utama
fig.suptitle("Perbandingan BF Matching: SIFT vs ORB", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan perbandingan
plt.savefig(os.path.join(OUTPUT_DIR, "06_sift_vs_orb.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"[SAVED] 06_sift_vs_orb.png")

# Menutup figure
plt.close()

# ============================================================
# 7. Test pada Pasangan dengan Perubahan Iluminasi
# ============================================================

# Menampilkan header tes iluminasi
print(f"\n--- Tes Pasangan Iluminasi ---")

# Membaca gambar buku dengan brightness normal
img_bright_0 = cv2.imread(os.path.join(IMAGE_DIR, "buku_bright+0.jpg"))

# Membaca gambar buku dengan brightness +80
img_bright_80 = cv2.imread(os.path.join(IMAGE_DIR, "buku_bright+80.jpg"))

# Memeriksa apakah gambar tersedia
if img_bright_0 is not None and img_bright_80 is not None:
    # Mengkonversi ke grayscale
    gray_b0 = cv2.cvtColor(img_bright_0, cv2.COLOR_BGR2GRAY)
    gray_b80 = cv2.cvtColor(img_bright_80, cv2.COLOR_BGR2GRAY)

    # Mendeteksi SIFT pada kedua gambar
    kp_b0, desc_b0 = sift.detectAndCompute(gray_b0, None)
    kp_b80, desc_b80 = sift.detectAndCompute(gray_b80, None)

    # Mencocokkan dengan BF Matcher
    bf_illum = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_illum = bf_illum.match(desc_b0, desc_b80)

    # Mengurutkan berdasarkan jarak
    matches_illum = sorted(matches_illum, key=lambda x: x.distance)

    # Menampilkan info
    print(f"  Keypoints normal: {len(kp_b0)}, bright+80: {len(kp_b80)}")
    print(f"  Matches: {len(matches_illum)}")

    # Menghitung rata-rata jarak
    if len(matches_illum) > 0:
        avg_dist_illum = np.mean([m.distance for m in matches_illum])
        print(f"  Rata-rata jarak: {avg_dist_illum:.2f}")
else:
    print("  [SKIP] Gambar iluminasi tidak tersedia")

# ============================================================
# 8. Test pada Pasangan dengan Viewpoint Berbeda
# ============================================================

# Menampilkan header tes viewpoint
print(f"\n--- Tes Pasangan Viewpoint Berbeda ---")

# Membaca gambar buku rotasi 0 dan rotasi 90
img_rot0 = cv2.imread(os.path.join(IMAGE_DIR, "buku_rot0.jpg"))
img_rot90 = cv2.imread(os.path.join(IMAGE_DIR, "buku_rot90.jpg"))

# Memeriksa apakah gambar tersedia
if img_rot0 is not None and img_rot90 is not None:
    # Mengkonversi ke grayscale
    gray_r0 = cv2.cvtColor(img_rot0, cv2.COLOR_BGR2GRAY)
    gray_r90 = cv2.cvtColor(img_rot90, cv2.COLOR_BGR2GRAY)

    # Mendeteksi SIFT
    kp_r0, desc_r0 = sift.detectAndCompute(gray_r0, None)
    kp_r90, desc_r90 = sift.detectAndCompute(gray_r90, None)

    # Mencocokkan
    bf_view = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_view = bf_view.match(desc_r0, desc_r90)

    # Mengurutkan
    matches_view = sorted(matches_view, key=lambda x: x.distance)

    # Menampilkan info
    print(f"  Keypoints rot0: {len(kp_r0)}, rot90: {len(kp_r90)}")
    print(f"  Matches: {len(matches_view)}")

    # Menghitung rata-rata jarak
    if len(matches_view) > 0:
        avg_dist_view = np.mean([m.distance for m in matches_view])
        print(f"  Rata-rata jarak: {avg_dist_view:.2f}")
else:
    print("  [SKIP] Gambar rotasi tidak tersedia")

# ============================================================
# 9. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 6: BRUTE-FORCE FEATURE MATCHING")

# Menampilkan garis pemisah
print("=" * 60)

# Menjelaskan konsep BF Matcher
print("1. BF Matcher menghitung jarak setiap deskriptor di gambar 1")
print("   dengan semua deskriptor di gambar 2 secara exhaustive")

# Menjelaskan norma jarak
print("2. Norma jarak:")
print("   - NORM_L2: untuk deskriptor float (SIFT, SURF)")
print("   - NORM_HAMMING: untuk deskriptor biner (ORB, BRIEF, AKAZE)")

# Menjelaskan crossCheck
print("3. crossCheck=True memastikan match saling terbaik:")
print("   A cocok B dan B cocok A -> valid match")

# Menjelaskan perbandingan
print("4. Perbandingan SIFT vs ORB matching:")
print(f"   SIFT: {len(matches_sift)} matches, avg dist: {np.mean(distances_sift):.2f}")
print(f"   ORB:  {len(matches_orb)} matches, avg dist: {np.mean(distances_orb):.2f}")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 06_bf_match_sift.png")
print("  - 06_bf_match_orb.png")
print("  - 06_match_histogram.png")
print("  - 06_sift_vs_orb.png")

# Menampilkan garis penutup
print("=" * 60)
