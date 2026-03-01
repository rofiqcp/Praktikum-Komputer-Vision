"""
==========================================================================
PERCOBAAN 13: PERBANDINGAN KOMPREHENSIF DESKRIPTOR FITUR
==========================================================================
Program ini membandingkan secara menyeluruh berbagai detektor dan
deskriptor fitur: SIFT, ORB, AKAZE, FAST+BRIEF (jika tersedia),
serta Harris+SIFT. Perbandingan mencakup waktu deteksi, waktu
matching, jumlah keypoint, jumlah good matches, dimensi deskriptor,
tipe data, dan penggunaan memori.

Konsep yang dipelajari:
- Perbandingan multi-kriteria antar detektor/deskriptor
- Trade-off antara akurasi dan kecepatan
- Dimensi deskriptor: SIFT=128 float, ORB=32 binary, AKAZE=61 binary
- Penggunaan memori per deskriptor
- Ranking detektor berdasarkan berbagai kriteria
- Hybrid detektor: menggunakan detektor A + deskriptor B

Fungsi utama yang dipelajari:
- cv2.SIFT_create()         : Detektor & deskriptor SIFT (128-D float)
- cv2.ORB_create()           : Detektor & deskriptor ORB (32-D binary)
- cv2.AKAZE_create()         : Detektor & deskriptor AKAZE (binary)
- cv2.FastFeatureDetector_create() : Detektor FAST (hanya keypoint)
- cv2.cornerHarris()         : Deteksi corner Harris
- cv2.BFMatcher()            : Brute-Force matcher

Hasil: Tabel perbandingan komprehensif dan chart peringkat detektor
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan fitur
import cv2

# Mengimpor NumPy untuk operasi array dan matriks numerik
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
print("PERCOBAAN 13: PERBANDINGAN KOMPREHENSIF DESKRIPTOR FITUR")
print("=" * 60)

# ============================================================
# 1. Memuat Pasangan Gambar untuk Pengujian
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
# 2. Mendefinisikan Semua Detektor/Deskriptor
# ============================================================

# Menyiapkan dictionary untuk menyimpan hasil setiap metode
hasil_semua = {}

# --- SIFT ---
# Menampilkan proses SIFT
print(f"\n--- Pengujian SIFT ---")

# Membuat detektor SIFT
sift = cv2.SIFT_create()

# Mengukur waktu deteksi SIFT pada gambar kiri
t_start = time.time()
kp_sift_l, desc_sift_l = sift.detectAndCompute(gray_left, None)
t_det_l = time.time() - t_start

# Mengukur waktu deteksi SIFT pada gambar kanan
t_start = time.time()
kp_sift_r, desc_sift_r = sift.detectAndCompute(gray_right, None)
t_det_r = time.time() - t_start

# Menghitung rata-rata waktu deteksi
t_det_sift = (t_det_l + t_det_r) / 2

# Membuat matcher BF L2 untuk SIFT
matcher_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Mengukur waktu matching SIFT
t_start = time.time()
matches_sift = matcher_sift.knnMatch(desc_sift_l, desc_sift_r, k=2)
t_match_sift = time.time() - t_start

# Menerapkan ratio test untuk SIFT
good_sift = []
for pair in matches_sift:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_sift.append(m)

# Menyimpan hasil SIFT
hasil_semua['SIFT'] = {
    'kp_left': len(kp_sift_l), 'kp_right': len(kp_sift_r),
    'good_matches': len(good_sift),
    'waktu_deteksi': t_det_sift, 'waktu_matching': t_match_sift,
    'dimensi': desc_sift_l.shape[1] if desc_sift_l is not None else 0,
    'dtype': str(desc_sift_l.dtype) if desc_sift_l is not None else "N/A",
    'memori_per_desc': desc_sift_l[0].nbytes if desc_sift_l is not None else 0,
    'desc_left': desc_sift_l, 'desc_right': desc_sift_r,
    'kp_obj_left': kp_sift_l, 'kp_obj_right': kp_sift_r,
    'matches': good_sift
}

# Menampilkan hasil SIFT
print(f"  KP: {len(kp_sift_l)}/{len(kp_sift_r)}, Good: {len(good_sift)}, "
      f"Det: {t_det_sift * 1000:.1f}ms, Match: {t_match_sift * 1000:.1f}ms")

# --- ORB ---
# Menampilkan proses ORB
print(f"\n--- Pengujian ORB ---")

# Membuat detektor ORB dengan 1000 fitur
orb = cv2.ORB_create(nfeatures=1000)

# Mengukur waktu deteksi ORB pada gambar kiri
t_start = time.time()
kp_orb_l, desc_orb_l = orb.detectAndCompute(gray_left, None)
t_det_l = time.time() - t_start

# Mengukur waktu deteksi ORB pada gambar kanan
t_start = time.time()
kp_orb_r, desc_orb_r = orb.detectAndCompute(gray_right, None)
t_det_r = time.time() - t_start

# Menghitung rata-rata waktu deteksi
t_det_orb = (t_det_l + t_det_r) / 2

# Membuat matcher Hamming untuk ORB
matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Mengukur waktu matching ORB
t_start = time.time()
matches_orb = matcher_orb.knnMatch(desc_orb_l, desc_orb_r, k=2)
t_match_orb = time.time() - t_start

# Menerapkan ratio test untuk ORB
good_orb = []
for pair in matches_orb:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_orb.append(m)

# Menyimpan hasil ORB
hasil_semua['ORB'] = {
    'kp_left': len(kp_orb_l), 'kp_right': len(kp_orb_r),
    'good_matches': len(good_orb),
    'waktu_deteksi': t_det_orb, 'waktu_matching': t_match_orb,
    'dimensi': desc_orb_l.shape[1] if desc_orb_l is not None else 0,
    'dtype': str(desc_orb_l.dtype) if desc_orb_l is not None else "N/A",
    'memori_per_desc': desc_orb_l[0].nbytes if desc_orb_l is not None else 0,
    'desc_left': desc_orb_l, 'desc_right': desc_orb_r,
    'kp_obj_left': kp_orb_l, 'kp_obj_right': kp_orb_r,
    'matches': good_orb
}

# Menampilkan hasil ORB
print(f"  KP: {len(kp_orb_l)}/{len(kp_orb_r)}, Good: {len(good_orb)}, "
      f"Det: {t_det_orb * 1000:.1f}ms, Match: {t_match_orb * 1000:.1f}ms")

# --- AKAZE ---
# Menampilkan proses AKAZE
print(f"\n--- Pengujian AKAZE ---")

# Membuat detektor AKAZE
akaze = cv2.AKAZE_create()

# Mengukur waktu deteksi AKAZE pada gambar kiri
t_start = time.time()
kp_akaze_l, desc_akaze_l = akaze.detectAndCompute(gray_left, None)
t_det_l = time.time() - t_start

# Mengukur waktu deteksi AKAZE pada gambar kanan
t_start = time.time()
kp_akaze_r, desc_akaze_r = akaze.detectAndCompute(gray_right, None)
t_det_r = time.time() - t_start

# Menghitung rata-rata waktu deteksi
t_det_akaze = (t_det_l + t_det_r) / 2

# Membuat matcher Hamming untuk AKAZE
matcher_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Mengukur waktu matching AKAZE
t_start = time.time()
matches_akaze = matcher_akaze.knnMatch(desc_akaze_l, desc_akaze_r, k=2)
t_match_akaze = time.time() - t_start

# Menerapkan ratio test untuk AKAZE
good_akaze = []
for pair in matches_akaze:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_akaze.append(m)

# Menyimpan hasil AKAZE
hasil_semua['AKAZE'] = {
    'kp_left': len(kp_akaze_l), 'kp_right': len(kp_akaze_r),
    'good_matches': len(good_akaze),
    'waktu_deteksi': t_det_akaze, 'waktu_matching': t_match_akaze,
    'dimensi': desc_akaze_l.shape[1] if desc_akaze_l is not None else 0,
    'dtype': str(desc_akaze_l.dtype) if desc_akaze_l is not None else "N/A",
    'memori_per_desc': desc_akaze_l[0].nbytes if desc_akaze_l is not None else 0,
    'desc_left': desc_akaze_l, 'desc_right': desc_akaze_r,
    'kp_obj_left': kp_akaze_l, 'kp_obj_right': kp_akaze_r,
    'matches': good_akaze
}

# Menampilkan hasil AKAZE
print(f"  KP: {len(kp_akaze_l)}/{len(kp_akaze_r)}, Good: {len(good_akaze)}, "
      f"Det: {t_det_akaze * 1000:.1f}ms, Match: {t_match_akaze * 1000:.1f}ms")

# --- FAST + BRIEF (jika tersedia) ---
# Menampilkan proses FAST+BRIEF
print(f"\n--- Pengujian FAST+BRIEF ---")

# Mencoba membuat FAST + BRIEF (BRIEF mungkin tidak tersedia di semua build)
try:
    # Membuat detektor FAST
    fast = cv2.FastFeatureDetector_create(threshold=25)

    # Membuat deskriptor BRIEF (mungkin memerlukan contrib)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # Mengukur waktu deteksi FAST pada gambar kiri
    t_start = time.time()
    kp_fast_l = fast.detect(gray_left, None)
    kp_fast_l, desc_fast_l = brief.compute(gray_left, kp_fast_l)
    t_det_l = time.time() - t_start

    # Mengukur waktu deteksi FAST pada gambar kanan
    t_start = time.time()
    kp_fast_r = fast.detect(gray_right, None)
    kp_fast_r, desc_fast_r = brief.compute(gray_right, kp_fast_r)
    t_det_r = time.time() - t_start

    # Menghitung rata-rata waktu deteksi
    t_det_fast = (t_det_l + t_det_r) / 2

    # Membuat matcher Hamming untuk BRIEF
    matcher_fast = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Mengukur waktu matching
    t_start = time.time()
    matches_fast = matcher_fast.knnMatch(desc_fast_l, desc_fast_r, k=2)
    t_match_fast = time.time() - t_start

    # Menerapkan ratio test
    good_fast = []
    for pair in matches_fast:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_fast.append(m)

    # Menyimpan hasil FAST+BRIEF
    hasil_semua['FAST+BRIEF'] = {
        'kp_left': len(kp_fast_l), 'kp_right': len(kp_fast_r),
        'good_matches': len(good_fast),
        'waktu_deteksi': t_det_fast, 'waktu_matching': t_match_fast,
        'dimensi': desc_fast_l.shape[1] if desc_fast_l is not None else 0,
        'dtype': str(desc_fast_l.dtype) if desc_fast_l is not None else "N/A",
        'memori_per_desc': desc_fast_l[0].nbytes if desc_fast_l is not None else 0,
        'desc_left': desc_fast_l, 'desc_right': desc_fast_r,
        'kp_obj_left': kp_fast_l, 'kp_obj_right': kp_fast_r,
        'matches': good_fast
    }

    # Menampilkan hasil FAST+BRIEF
    print(f"  KP: {len(kp_fast_l)}/{len(kp_fast_r)}, Good: {len(good_fast)}, "
          f"Det: {t_det_fast * 1000:.1f}ms, Match: {t_match_fast * 1000:.1f}ms")

except Exception as e:
    # Menampilkan pesan jika BRIEF tidak tersedia
    print(f"  [SKIP] FAST+BRIEF tidak tersedia: {e}")
    print(f"  (Memerlukan opencv-contrib-python)")

# --- Harris + SIFT (hybrid) ---
# Menampilkan proses Harris+SIFT
print(f"\n--- Pengujian Harris+SIFT (Hybrid) ---")

# Membuat detektor SIFT untuk komputasi deskriptor saja
sift_desc = cv2.SIFT_create()

# Mengukur waktu deteksi Harris pada gambar kiri
t_start = time.time()

# Menghitung respons Harris pada gambar kiri
harris_l = cv2.cornerHarris(np.float32(gray_left), blockSize=2, ksize=3, k=0.04)

# Mendilasi respons Harris
harris_l_dil = cv2.dilate(harris_l, None)

# Mendapatkan lokasi corner Harris yang melewati threshold
threshold_harris = 0.01 * harris_l_dil.max()
corner_locs_l = np.argwhere(harris_l_dil > threshold_harris)

# Mengkonversi lokasi corner menjadi KeyPoint
kp_harris_l = [cv2.KeyPoint(float(c[1]), float(c[0]), 7) for c in corner_locs_l[:500]]

# Menghitung deskriptor SIFT pada keypoint Harris
kp_harris_l, desc_harris_l = sift_desc.compute(gray_left, kp_harris_l)
t_det_l = time.time() - t_start

# Mengukur waktu deteksi Harris pada gambar kanan
t_start = time.time()

# Menghitung respons Harris pada gambar kanan
harris_r = cv2.cornerHarris(np.float32(gray_right), blockSize=2, ksize=3, k=0.04)

# Mendilasi respons Harris
harris_r_dil = cv2.dilate(harris_r, None)

# Mendapatkan lokasi corner Harris
threshold_harris_r = 0.01 * harris_r_dil.max()
corner_locs_r = np.argwhere(harris_r_dil > threshold_harris_r)

# Mengkonversi lokasi corner menjadi KeyPoint
kp_harris_r = [cv2.KeyPoint(float(c[1]), float(c[0]), 7) for c in corner_locs_r[:500]]

# Menghitung deskriptor SIFT pada keypoint Harris
kp_harris_r, desc_harris_r = sift_desc.compute(gray_right, kp_harris_r)
t_det_r = time.time() - t_start

# Menghitung rata-rata waktu deteksi
t_det_harris = (t_det_l + t_det_r) / 2

# Memeriksa ketersediaan deskriptor Harris+SIFT
if desc_harris_l is not None and desc_harris_r is not None and len(desc_harris_l) > 0:
    # Membuat matcher L2 untuk deskriptor SIFT
    matcher_harris = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Mengukur waktu matching
    t_start = time.time()
    matches_harris = matcher_harris.knnMatch(desc_harris_l, desc_harris_r, k=2)
    t_match_harris = time.time() - t_start

    # Menerapkan ratio test
    good_harris = []
    for pair in matches_harris:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_harris.append(m)

    # Menyimpan hasil Harris+SIFT
    hasil_semua['Harris+SIFT'] = {
        'kp_left': len(kp_harris_l), 'kp_right': len(kp_harris_r),
        'good_matches': len(good_harris),
        'waktu_deteksi': t_det_harris, 'waktu_matching': t_match_harris,
        'dimensi': desc_harris_l.shape[1] if desc_harris_l is not None else 0,
        'dtype': str(desc_harris_l.dtype) if desc_harris_l is not None else "N/A",
        'memori_per_desc': desc_harris_l[0].nbytes if desc_harris_l is not None else 0,
        'desc_left': desc_harris_l, 'desc_right': desc_harris_r,
        'kp_obj_left': kp_harris_l, 'kp_obj_right': kp_harris_r,
        'matches': good_harris
    }

    # Menampilkan hasil Harris+SIFT
    print(f"  KP: {len(kp_harris_l)}/{len(kp_harris_r)}, Good: {len(good_harris)}, "
          f"Det: {t_det_harris * 1000:.1f}ms, Match: {t_match_harris * 1000:.1f}ms")
else:
    # Menampilkan pesan jika Harris+SIFT gagal
    print(f"  [SKIP] Harris+SIFT tidak menghasilkan deskriptor yang cukup")

# ============================================================
# 3. Visualisasi Match untuk Setiap Metode
# ============================================================

# Menghitung jumlah metode yang tersedia
n_metode = len(hasil_semua)

# Menghitung layout subplot
n_rows = (n_metode + 1) // 2
n_cols = 2

# Membuat figure untuk visualisasi
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))

# Meratakan axes menjadi 1D
if n_rows == 1:
    axes_flat = [axes[0], axes[1]] if n_cols == 2 else [axes]
else:
    axes_flat = axes.flatten()

# Melakukan iterasi untuk setiap metode
for idx, (nama, data) in enumerate(hasil_semua.items()):
    # Mengambil keypoint dan matches
    kp_l = data['kp_obj_left']
    kp_r = data['kp_obj_right']
    good = data['matches']

    # Menggambar matches (maksimal 30 untuk keterbacaan)
    img_match = cv2.drawMatches(img_left, kp_l, img_right, kp_r,
                                good[:30], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Menampilkan pada subplot
    axes_flat[idx].imshow(cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB))

    # Memberikan judul
    axes_flat[idx].set_title(f"{nama}: {data['good_matches']} good matches "
                             f"(Det: {data['waktu_deteksi'] * 1000:.1f}ms)", fontsize=11)

    # Menonaktifkan sumbu
    axes_flat[idx].axis('off')

# Menonaktifkan subplot kosong
for idx in range(n_metode, len(axes_flat)):
    axes_flat[idx].axis('off')

# Memberikan judul utama
fig.suptitle("Perbandingan Visual: Matching Berbagai Metode", fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file
plt.savefig(os.path.join(OUTPUT_DIR, "13_perbandingan_visual.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"\n[SAVED] 13_perbandingan_visual.png")

# Menutup figure
plt.close()

# ============================================================
# 4. Chart Perbandingan Multi-Kriteria
# ============================================================

# Mengambil nama-nama metode
metode_names = list(hasil_semua.keys())

# Membuat figure 2x2 untuk chart perbandingan
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mendefinisikan warna untuk setiap metode
colors = plt.cm.Set2(np.linspace(0, 1, len(metode_names)))

# --- Chart 1: Jumlah Good Matches ---
# Mengambil data good matches
vals_match = [hasil_semua[n]['good_matches'] for n in metode_names]

# Menggambar bar chart
bars1 = axes[0, 0].bar(metode_names, vals_match, color=colors, edgecolor='black')

# Menambahkan label angka
for bar, val in zip(bars1, vals_match):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha='center', va='bottom', fontsize=9)

# Memberikan judul
axes[0, 0].set_title("Jumlah Good Matches", fontsize=12)

# Memberikan label sumbu Y
axes[0, 0].set_ylabel("Good Matches")

# Menambahkan grid
axes[0, 0].grid(axis='y', alpha=0.3)

# Merotasi label sumbu X
axes[0, 0].tick_params(axis='x', rotation=15)

# --- Chart 2: Waktu Deteksi ---
# Mengambil data waktu deteksi
vals_det = [hasil_semua[n]['waktu_deteksi'] * 1000 for n in metode_names]

# Menggambar bar chart
bars2 = axes[0, 1].bar(metode_names, vals_det, color=colors, edgecolor='black')

# Menambahkan label angka
for bar, val in zip(bars2, vals_det):
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{val:.1f}", ha='center', va='bottom', fontsize=9)

# Memberikan judul
axes[0, 1].set_title("Waktu Deteksi (ms)", fontsize=12)

# Memberikan label sumbu Y
axes[0, 1].set_ylabel("Waktu (ms)")

# Menambahkan grid
axes[0, 1].grid(axis='y', alpha=0.3)

# Merotasi label
axes[0, 1].tick_params(axis='x', rotation=15)

# --- Chart 3: Jumlah Keypoint (kiri) ---
# Mengambil data keypoint
vals_kp = [hasil_semua[n]['kp_left'] for n in metode_names]

# Menggambar bar chart
bars3 = axes[1, 0].bar(metode_names, vals_kp, color=colors, edgecolor='black')

# Menambahkan label angka
for bar, val in zip(bars3, vals_kp):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha='center', va='bottom', fontsize=9)

# Memberikan judul
axes[1, 0].set_title("Jumlah Keypoint (Gambar Kiri)", fontsize=12)

# Memberikan label sumbu Y
axes[1, 0].set_ylabel("Keypoints")

# Menambahkan grid
axes[1, 0].grid(axis='y', alpha=0.3)

# Merotasi label
axes[1, 0].tick_params(axis='x', rotation=15)

# --- Chart 4: Memori per Deskriptor ---
# Mengambil data memori
vals_mem = [hasil_semua[n]['memori_per_desc'] for n in metode_names]

# Menggambar bar chart
bars4 = axes[1, 1].bar(metode_names, vals_mem, color=colors, edgecolor='black')

# Menambahkan label angka
for bar, val in zip(bars4, vals_mem):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val}", ha='center', va='bottom', fontsize=9)

# Memberikan judul
axes[1, 1].set_title("Memori per Deskriptor (bytes)", fontsize=12)

# Memberikan label sumbu Y
axes[1, 1].set_ylabel("Bytes")

# Menambahkan grid
axes[1, 1].grid(axis='y', alpha=0.3)

# Merotasi label
axes[1, 1].tick_params(axis='x', rotation=15)

# Memberikan judul utama
fig.suptitle("Perbandingan Multi-Kriteria Detektor/Deskriptor", fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan chart ke file
plt.savefig(os.path.join(OUTPUT_DIR, "13_perbandingan_chart.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"[SAVED] 13_perbandingan_chart.png")

# Menutup figure
plt.close()

# ============================================================
# 5. Tabel Perbandingan Komprehensif
# ============================================================

# Membuat figure untuk tabel
fig, ax = plt.subplots(figsize=(16, 4 + len(hasil_semua) * 0.6))

# Menonaktifkan sumbu
ax.axis('off')

# Mendefinisikan header tabel
tabel_header = ["Metode", "KP Kiri", "KP Kanan", "Good Match",
                "Det (ms)", "Match (ms)", "Dimensi", "Tipe", "Memori (B)"]

# Menyusun data tabel
tabel_data = []
for nama in metode_names:
    d = hasil_semua[nama]
    # Menyusun baris data
    tabel_data.append([
        nama,
        str(d['kp_left']),
        str(d['kp_right']),
        str(d['good_matches']),
        f"{d['waktu_deteksi'] * 1000:.1f}",
        f"{d['waktu_matching'] * 1000:.1f}",
        str(d['dimensi']),
        d['dtype'],
        str(d['memori_per_desc'])
    ])

# Membuat tabel pada figure
table = ax.table(cellText=tabel_data, colLabels=tabel_header,
                 cellLoc='center', loc='center')

# Mengatur ukuran font tabel
table.auto_set_font_size(False)
table.set_fontsize(9)

# Mengatur skala tabel
table.scale(1.0, 1.6)

# Memberi warna header
for j in range(len(tabel_header)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Memberi warna baris bergantian
for i in range(len(tabel_data)):
    for j in range(len(tabel_header)):
        if i % 2 == 0:
            table[i + 1, j].set_facecolor('#D6E4F0')

# Memberikan judul tabel
ax.set_title("Tabel Perbandingan Komprehensif Detektor/Deskriptor",
             fontsize=14, fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Menyimpan tabel ke file
plt.savefig(os.path.join(OUTPUT_DIR, "13_perbandingan_tabel.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"[SAVED] 13_perbandingan_tabel.png")

# Menutup figure
plt.close()

# ============================================================
# 6. Ranking Detektor
# ============================================================

# Menampilkan header ranking
print(f"\n--- Ranking Detektor ---")

# Ranking berdasarkan jumlah good matches (terbanyak terbaik)
rank_match = sorted(metode_names, key=lambda n: hasil_semua[n]['good_matches'], reverse=True)
print(f"\nBerdasarkan Good Matches:")
for i, n in enumerate(rank_match):
    print(f"  {i + 1}. {n}: {hasil_semua[n]['good_matches']}")

# Ranking berdasarkan kecepatan deteksi (tercepat terbaik)
rank_speed = sorted(metode_names, key=lambda n: hasil_semua[n]['waktu_deteksi'])
print(f"\nBerdasarkan Kecepatan Deteksi:")
for i, n in enumerate(rank_speed):
    print(f"  {i + 1}. {n}: {hasil_semua[n]['waktu_deteksi'] * 1000:.1f} ms")

# Ranking berdasarkan efisiensi memori (terkecil terbaik)
rank_mem = sorted(metode_names, key=lambda n: hasil_semua[n]['memori_per_desc'])
print(f"\nBerdasarkan Efisiensi Memori:")
for i, n in enumerate(rank_mem):
    print(f"  {i + 1}. {n}: {hasil_semua[n]['memori_per_desc']} bytes/desc")

# ============================================================
# 7. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 13: PERBANDINGAN KOMPREHENSIF DESKRIPTOR FITUR")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan trade-off
print("1. Terdapat trade-off antara akurasi dan kecepatan:")
print("   - SIFT: akurat tapi lambat (128-D float)")
print("   - ORB: cepat tapi kurang akurat (32-D binary)")
print("   - AKAZE: keseimbangan antara keduanya")

# Menampilkan penjelasan dimensi deskriptor
print("2. Dimensi deskriptor mempengaruhi memori dan kecepatan:")
print("   - Float descriptor (SIFT): lebih informatif, lebih boros memori")
print("   - Binary descriptor (ORB, AKAZE): efisien, matching cepat (Hamming)")

# Menampilkan tentang hybrid detector
print("3. Hybrid detector (Harris+SIFT) memungkinkan kombinasi")
print("   kekuatan detektor A dengan deskriptor B.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 13_perbandingan_visual.png")
print("  - 13_perbandingan_chart.png")
print("  - 13_perbandingan_tabel.png")

# Menampilkan garis penutup
print("=" * 60)
