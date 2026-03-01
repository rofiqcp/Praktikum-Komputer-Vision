"""
==========================================================================
PERCOBAAN 8: ESTIMASI HOMOGRAPHY DENGAN RANSAC
==========================================================================
Program ini mempelajari cara menghitung matriks homography (3x3) yang
memetakan titik-titik dari satu bidang planar ke bidang lain menggunakan
algoritma RANSAC (Random Sample Consensus). Matriks homography memiliki
8 derajat kebebasan (DOF) dan membutuhkan minimal 4 pasangan titik.

Konsep yang dipelajari:
- Matriks homography 3x3 dengan 8 DOF (derajat kebebasan)
- RANSAC: memilih sampel acak, fit model, hitung inlier, iterasi
- Pengaruh threshold RANSAC terhadap jumlah inlier
- Perbandingan RANSAC vs LMEDS (Least Median of Squares)
- Warp perspektif menggunakan homography
- Bounding box transformasi perspektif

Fungsi utama yang dipelajari:
- cv2.findHomography()        : Menghitung matriks homography dengan RANSAC/LMEDS
- cv2.warpPerspective()       : Mentransformasi gambar menggunakan homography
- cv2.perspectiveTransform()  : Mentransformasikan titik-titik menggunakan homography
- cv2.drawMatchesKnn()        : Menggambar hasil matching
- cv2.polylines()             : Menggambar poligon bounding box

Hasil: Visualisasi homography, inlier/outlier, warp, dan variasi threshold
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
print("PERCOBAAN 8: ESTIMASI HOMOGRAPHY DENGAN RANSAC")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Template dan Scene
# ============================================================

# Membaca gambar template (objek buku) dari file
img_template = cv2.imread(os.path.join(IMAGE_DIR, "objek_buku.jpg"))

# Membaca gambar scene yang mengandung objek buku dari file
img_scene = cv2.imread(os.path.join(IMAGE_DIR, "scene_buku.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_template is None or img_scene is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi ukuran gambar template
print(f"[INFO] Ukuran template (objek_buku): {img_template.shape}")

# Menampilkan informasi ukuran gambar scene
print(f"[INFO] Ukuran scene (scene_buku): {img_scene.shape}")

# Mengkonversi gambar template ke grayscale
gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar scene ke grayscale
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Deteksi SIFT dan Matching FLANN + Ratio Test
# ============================================================

# Membuat detektor SIFT untuk mendeteksi keypoints dan descriptor
sift = cv2.SIFT_create()

# Mendeteksi keypoints dan descriptor pada gambar template
kp_template, desc_template = sift.detectAndCompute(gray_template, None)

# Mendeteksi keypoints dan descriptor pada gambar scene
kp_scene, desc_scene = sift.detectAndCompute(gray_scene, None)

# Menampilkan jumlah keypoint yang terdeteksi
print(f"\n[SIFT] Keypoints template: {len(kp_template)}")
print(f"[SIFT] Keypoints scene: {len(kp_scene)}")

# Mendefinisikan parameter FLANN untuk deskriptor SIFT (tipe float)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Membuat matcher FLANN dengan parameter yang sudah ditentukan
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Melakukan KNN matching dengan k=2 untuk ratio test
matches_knn = flann.knnMatch(desc_template, desc_scene, k=2)

# Menampilkan jumlah match mentah
print(f"[MATCH] Total KNN matches: {len(matches_knn)}")

# Menerapkan Lowe's Ratio Test untuk memfilter match yang ambigu
ratio_threshold = 0.7
good_matches = []
for m, n in matches_knn:
    # Memeriksa apakah rasio jarak memenuhi threshold
    if m.distance < ratio_threshold * n.distance:
        # Menambahkan match yang lolos ratio test
        good_matches.append(m)

# Menampilkan jumlah match setelah ratio test
print(f"[MATCH] Good matches (ratio < {ratio_threshold}): {len(good_matches)}")

# ============================================================
# 3. Menghitung Homography dengan RANSAC
# ============================================================

# Memeriksa apakah jumlah match cukup (minimal 4 untuk homography)
if len(good_matches) >= 4:
    # Mengekstrak lokasi keypoint dari match template
    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Mengekstrak lokasi keypoint dari match scene
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Menghitung matriks homography menggunakan RANSAC dengan threshold 5.0
    H, mask_ransac = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Mengkonversi mask ke array 1D untuk memisahkan inlier dan outlier
    matches_mask = mask_ransac.ravel().tolist()

    # Menghitung jumlah inlier (mask = 1)
    num_inliers = sum(matches_mask)

    # Menghitung jumlah outlier (mask = 0)
    num_outliers = len(matches_mask) - num_inliers

    # Menampilkan hasil homography
    print(f"\n[HOMOGRAPHY] Matriks H (3x3):")
    print(H)
    print(f"[HOMOGRAPHY] Inliers: {num_inliers}, Outliers: {num_outliers}")
    print(f"[HOMOGRAPHY] Rasio inlier: {num_inliers / len(matches_mask) * 100:.1f}%")
else:
    # Menampilkan pesan error jika match tidak cukup
    print("[ERROR] Tidak cukup good matches untuk menghitung homography!")
    exit()

# ============================================================
# 4. Visualisasi Inlier (Hijau) dan Outlier (Merah)
# ============================================================

# Membuat gambar hasil match dengan warna inlier hijau dan outlier merah
# Menyiapkan parameter gambar: inlier = hijau, outlier = merah
draw_params_inlier = dict(
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matches_mask,
    flags=cv2.DrawMatchesFlags_DEFAULT
)

# Menggambar hanya inlier (hijau) pada gambar match
img_inlier = cv2.drawMatches(img_template, kp_template, img_scene, kp_scene,
                             good_matches, None, **draw_params_inlier)

# Membuat mask outlier (kebalikan dari mask inlier)
outlier_mask = [1 - m for m in matches_mask]

# Menyiapkan parameter gambar untuk outlier (merah)
draw_params_outlier = dict(
    matchColor=(0, 0, 255),
    singlePointColor=None,
    matchesMask=outlier_mask,
    flags=cv2.DrawMatchesFlags_DEFAULT
)

# Menggambar outlier (merah) pada gambar match terpisah
img_outlier = cv2.drawMatches(img_template, kp_template, img_scene, kp_scene,
                              good_matches, None, **draw_params_outlier)

# Membuat figure untuk menampilkan inlier dan outlier
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Menampilkan gambar inlier pada subplot atas
axes[0].imshow(cv2.cvtColor(img_inlier, cv2.COLOR_BGR2RGB))

# Memberikan judul pada subplot inlier
axes[0].set_title(f"Inlier Matches (Hijau) - {num_inliers} titik", fontsize=14)

# Menonaktifkan sumbu pada subplot inlier
axes[0].axis('off')

# Menampilkan gambar outlier pada subplot bawah
axes[1].imshow(cv2.cvtColor(img_outlier, cv2.COLOR_BGR2RGB))

# Memberikan judul pada subplot outlier
axes[1].set_title(f"Outlier Matches (Merah) - {num_outliers} titik", fontsize=14)

# Menonaktifkan sumbu pada subplot outlier
axes[1].axis('off')

# Memberikan judul utama figure
fig.suptitle("Homography RANSAC: Inlier vs Outlier", fontsize=16, fontweight='bold')

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan gambar hasil ke file
plt.savefig(os.path.join(OUTPUT_DIR, "08_homography_matches.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 08_homography_matches.png")

# Menutup figure untuk menghemat memori
plt.close()

# ============================================================
# 5. Warp Template Menggunakan Homography
# ============================================================

# Mendapatkan ukuran gambar scene sebagai target warp
h_scene, w_scene = img_scene.shape[:2]

# Melakukan warp perspektif pada template menggunakan matriks homography
img_warped = cv2.warpPerspective(img_template, H, (w_scene, h_scene))

# Membuat overlay dengan menggabungkan warp dan scene menggunakan blending
alpha = 0.5

# Membuat salinan scene untuk overlay
img_overlay = img_scene.copy()

# Membuat mask dari piksel non-hitam pada gambar warped
mask_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY) > 0

# Melakukan blending alpha pada area warp di scene
img_overlay[mask_warped] = cv2.addWeighted(
    img_scene[mask_warped], 1 - alpha,
    img_warped[mask_warped], alpha, 0
)

# Mendapatkan ukuran template untuk menentukan corner bounding box
h_tmpl, w_tmpl = img_template.shape[:2]

# Mendefinisikan 4 titik corner dari template
corners_template = np.float32([[0, 0], [w_tmpl, 0], [w_tmpl, h_tmpl], [0, h_tmpl]]).reshape(-1, 1, 2)

# Mentransformasikan corner template ke koordinat scene menggunakan homography
corners_scene = cv2.perspectiveTransform(corners_template, H)

# Menggambar bounding box pada scene menggunakan polylines
img_bbox = img_scene.copy()

# Mengkonversi corner ke integer untuk polylines
corners_int = np.int32(corners_scene)

# Menggambar polyline hijau sebagai bounding box
cv2.polylines(img_bbox, [corners_int], isClosed=True, color=(0, 255, 0), thickness=3)

# Membuat figure untuk menampilkan hasil warp
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Menampilkan gambar template asli
axes[0, 0].imshow(cv2.cvtColor(img_template, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Template (objek_buku)", fontsize=12)
axes[0, 0].axis('off')

# Menampilkan gambar scene asli
axes[0, 1].imshow(cv2.cvtColor(img_scene, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Scene (scene_buku)", fontsize=12)
axes[0, 1].axis('off')

# Menampilkan gambar warped template
axes[1, 0].imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Warp Template ke Scene", fontsize=12)
axes[1, 0].axis('off')

# Menampilkan gambar scene dengan bounding box
axes[1, 1].imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Bounding Box pada Scene", fontsize=12)
axes[1, 1].axis('off')

# Memberikan judul utama figure
fig.suptitle("Warp Perspektif dan Bounding Box Homography", fontsize=16, fontweight='bold')

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan gambar warp ke file
plt.savefig(os.path.join(OUTPUT_DIR, "08_homography_warp.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"[SAVED] 08_homography_warp.png")

# Menutup figure
plt.close()

# ============================================================
# 6. Variasi Threshold RANSAC dan Plot Jumlah Inlier
# ============================================================

# Mendefinisikan daftar threshold RANSAC yang akan diuji
thresholds = [1.0, 3.0, 5.0, 10.0, 20.0]

# Menyiapkan list untuk menyimpan jumlah inlier per threshold
inlier_counts = []

# Menyiapkan list untuk menyimpan rasio inlier per threshold
inlier_ratios = []

# Menampilkan header tabel hasil variasi threshold
print(f"\n{'Threshold':>10} {'Inliers':>10} {'Outliers':>10} {'Rasio (%)':>10}")
print("-" * 45)

# Melakukan iterasi untuk setiap threshold RANSAC
for thresh in thresholds:
    # Menghitung homography dengan threshold saat ini
    H_t, mask_t = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, thresh)

    # Menghitung jumlah inlier dari mask
    n_inlier = int(mask_t.sum())

    # Menghitung jumlah outlier
    n_outlier = len(mask_t) - n_inlier

    # Menghitung rasio inlier dalam persen
    rasio = n_inlier / len(mask_t) * 100

    # Menyimpan jumlah inlier ke list
    inlier_counts.append(n_inlier)

    # Menyimpan rasio inlier ke list
    inlier_ratios.append(rasio)

    # Menampilkan hasil untuk threshold ini
    print(f"{thresh:>10.1f} {n_inlier:>10} {n_outlier:>10} {rasio:>10.1f}")

# ============================================================
# 7. Perbandingan RANSAC vs LMEDS
# ============================================================

# Menampilkan header perbandingan metode
print(f"\n--- Perbandingan Metode Estimasi Homography ---")

# Menghitung homography dengan metode RANSAC
t_start = time.time()
H_ransac, mask_ransac_cmp = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
t_ransac = time.time() - t_start

# Menghitung jumlah inlier RANSAC
n_inlier_ransac = int(mask_ransac_cmp.sum())

# Menghitung homography dengan metode LMEDS (Least Median of Squares)
t_start = time.time()
H_lmeds, mask_lmeds = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
t_lmeds = time.time() - t_start

# Menghitung jumlah inlier LMEDS
n_inlier_lmeds = int(mask_lmeds.sum())

# Menampilkan perbandingan RANSAC vs LMEDS
print(f"  RANSAC: Inliers = {n_inlier_ransac}, Waktu = {t_ransac * 1000:.2f} ms")
print(f"  LMEDS:  Inliers = {n_inlier_lmeds}, Waktu = {t_lmeds * 1000:.2f} ms")

# Membuat figure untuk grafik variasi threshold dan perbandingan
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Membuat bar chart jumlah inlier per threshold
bars = axes[0].bar([str(t) for t in thresholds], inlier_counts, color='steelblue', edgecolor='black')

# Menambahkan label angka di atas setiap bar
for bar, count in zip(bars, inlier_counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=10)

# Memberikan label sumbu X
axes[0].set_xlabel("RANSAC Threshold (piksel)", fontsize=12)

# Memberikan label sumbu Y
axes[0].set_ylabel("Jumlah Inlier", fontsize=12)

# Memberikan judul chart
axes[0].set_title("Pengaruh Threshold RANSAC terhadap Inlier", fontsize=13)

# Menambahkan grid pada chart
axes[0].grid(axis='y', alpha=0.3)

# Membuat bar chart perbandingan RANSAC vs LMEDS
metode_names = ['RANSAC', 'LMEDS']
metode_inliers = [n_inlier_ransac, n_inlier_lmeds]
metode_colors = ['steelblue', 'coral']

# Menggambar bar chart perbandingan
bars2 = axes[1].bar(metode_names, metode_inliers, color=metode_colors, edgecolor='black')

# Menambahkan label angka di atas setiap bar
for bar, count in zip(bars2, metode_inliers):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=10)

# Memberikan label sumbu Y
axes[1].set_ylabel("Jumlah Inlier", fontsize=12)

# Memberikan judul chart
axes[1].set_title("Perbandingan RANSAC vs LMEDS", fontsize=13)

# Menambahkan grid pada chart
axes[1].grid(axis='y', alpha=0.3)

# Memberikan judul utama figure
fig.suptitle("Analisis Threshold dan Metode Homography", fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan grafik ke file
plt.savefig(os.path.join(OUTPUT_DIR, "08_ransac_threshold.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 08_ransac_threshold.png")

# Menutup figure
plt.close()

# ============================================================
# 8. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 8: ESTIMASI HOMOGRAPHY DENGAN RANSAC")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan tentang homography
print("1. Matriks homography (3x3) memetakan titik dari bidang planar")
print("   satu ke bidang lain dengan 8 derajat kebebasan (DOF).")

# Menampilkan penjelasan tentang RANSAC
print("2. RANSAC memilih 4 sampel acak, menghitung homography,")
print("   lalu menghitung inlier. Proses diulang hingga model terbaik.")

# Menampilkan pengaruh threshold
print("3. Threshold RANSAC mengontrol toleransi error reprojeksi:")
print("   - Threshold kecil: ketat, sedikit inlier tapi akurat")
print("   - Threshold besar: longgar, banyak inlier tapi kurang presisi")

# Menampilkan perbandingan metode
print("4. LMEDS tidak memerlukan threshold, menggunakan median residual")
print("   sebagai kriteria, cocok saat outlier kurang dari 50%.")

# Menampilkan daftar file yang disimpan
print("\nFile output yang dihasilkan:")
print("  - 08_homography_matches.png")
print("  - 08_homography_warp.png")
print("  - 08_ransac_threshold.png")

# Menampilkan garis penutup
print("=" * 60)
