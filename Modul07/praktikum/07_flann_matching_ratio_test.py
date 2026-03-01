"""
==========================================================================
PERCOBAAN 7: FLANN MATCHING + LOWE'S RATIO TEST
==========================================================================
Program ini mempelajari pencocokan fitur menggunakan FLANN (Fast Library
for Approximate Nearest Neighbors) yang lebih cepat dari Brute-Force
untuk dataset besar, serta Lowe's Ratio Test untuk memfilter match
yang ambigu.

Konsep yang dipelajari:
- FLANN: approximate nearest neighbor search menggunakan KD-Tree
  (lebih cepat dari BF untuk deskriptor berdimensi tinggi)
- KNN Match (k=2): mencari 2 tetangga terdekat untuk setiap deskriptor
- Lowe's Ratio Test: match dianggap baik jika d1/d2 < ratio threshold
  (menolak match ambigu dimana dua tetangga terdekat berjarak mirip)
- Pengaruh ratio threshold terhadap jumlah dan kualitas match
- Perbandingan performa BF vs FLANN

Fungsi utama yang dipelajari:
- cv2.FlannBasedMatcher()   : Membuat matcher berbasis FLANN
- flann.knnMatch()          : KNN matching (k tetangga terdekat)
- Ratio test filtering      : Memfilter match berdasarkan rasio jarak
- cv2.drawMatchesKnn()      : Menggambar hasil KNN matching

Hasil: Visualisasi FLANN matching dengan ratio test dan perbandingan BF
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
print("PERCOBAAN 7: FLANN MATCHING + LOWE'S RATIO TEST")
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
# 2. Deteksi SIFT Keypoints dan Descriptors
# ============================================================

# Membuat detektor SIFT
sift = cv2.SIFT_create()

# Mendeteksi keypoints dan descriptor pada gambar kiri
kp_left, desc_left = sift.detectAndCompute(gray_left, None)

# Mendeteksi keypoints dan descriptor pada gambar kanan
kp_right, desc_right = sift.detectAndCompute(gray_right, None)

# Menampilkan jumlah keypoint
print(f"\n[SIFT] Keypoints kiri: {len(kp_left)}")
print(f"[SIFT] Keypoints kanan: {len(kp_right)}")
print(f"[SIFT] Descriptor shape kiri: {desc_left.shape}")

# ============================================================
# 3. Setup FLANN Matcher
# ============================================================

# Mendefinisikan konstanta FLANN_INDEX_KDTREE
FLANN_INDEX_KDTREE = 1

# Mendefinisikan parameter index untuk FLANN (KD-Tree dengan 5 pohon)
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

# Mendefinisikan parameter pencarian (50 kali pengecekan)
search_params = dict(checks=50)

# Membuat FLANN matcher dengan parameter yang telah didefinisikan
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Menampilkan informasi parameter FLANN
print(f"\n[FLANN] Algorithm: KD-Tree (FLANN_INDEX_KDTREE={FLANN_INDEX_KDTREE})")
print(f"[FLANN] Trees: 5, Checks: 50")

# ============================================================
# 4. KNN Match dengan k=2
# ============================================================

# Mencatat waktu mulai matching FLANN
t_start_flann = time.time()

# Melakukan KNN matching dengan k=2 (2 tetangga terdekat)
matches_knn = flann.knnMatch(desc_left, desc_right, k=2)

# Menghitung waktu matching FLANN
flann_time = time.time() - t_start_flann

# Menampilkan info KNN matching
print(f"\n[KNN] Total pasangan KNN: {len(matches_knn)}")
print(f"[KNN] Waktu FLANN matching: {flann_time*1000:.2f} ms")

# ============================================================
# 5. Menerapkan Lowe's Ratio Test (ratio=0.75)
# ============================================================

# Mendefinisikan threshold rasio Lowe
ratio_threshold = 0.75

# Menyiapkan list untuk menyimpan good matches
good_matches = []

# Menerapkan ratio test: match dianggap baik jika d1/d2 < ratio
for m, n in matches_knn:
    # Memeriksa apakah jarak match pertama < ratio * jarak match kedua
    if m.distance < ratio_threshold * n.distance:
        # Menambahkan ke good matches jika lolos ratio test
        good_matches.append(m)

# Menampilkan hasil ratio test
print(f"\n[RATIO TEST] Threshold: {ratio_threshold}")
print(f"[RATIO TEST] Good matches: {len(good_matches)} dari {len(matches_knn)} "
      f"({len(good_matches)/len(matches_knn)*100:.1f}%)")

# Mengurutkan good matches berdasarkan jarak
good_matches = sorted(good_matches, key=lambda x: x.distance)

# Menghitung statistik jarak good matches
if len(good_matches) > 0:
    good_distances = [m.distance for m in good_matches]
    print(f"[RATIO TEST] Jarak - min: {min(good_distances):.2f}, "
          f"max: {max(good_distances):.2f}, mean: {np.mean(good_distances):.2f}")

# ============================================================
# 6. Visualisasi Good Matches
# ============================================================

# Membuat figure untuk visualisasi good matches
fig, axes = plt.subplots(3, 1, figsize=(16, 15))

# Menampilkan semua KNN matches (tanpa filter) - mengambil match pertama saja
all_first_matches = [m for m, n in matches_knn]

# Mengurutkan semua matches
all_first_matches_sorted = sorted(all_first_matches, key=lambda x: x.distance)

# Menggambar top-50 tanpa ratio test
img_all = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                           all_first_matches_sorted[:50], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Menampilkan hasil tanpa ratio test
axes[0].imshow(cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Tanpa Ratio Test - Top 50 dari {len(all_first_matches)} matches", fontsize=12)
axes[0].axis('off')

# Menggambar good matches setelah ratio test (top-50)
n_show = min(50, len(good_matches))
img_good = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                            good_matches[:n_show], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Menampilkan good matches
axes[1].imshow(cv2.cvtColor(img_good, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Setelah Ratio Test (ratio={ratio_threshold}) - "
                  f"Top {n_show} dari {len(good_matches)} good matches", fontsize=12)
axes[1].axis('off')

# Menampilkan semua good matches
img_all_good = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                                good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Menampilkan semua good matches
axes[2].imshow(cv2.cvtColor(img_all_good, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Semua Good Matches ({len(good_matches)} matches)", fontsize=12)
axes[2].axis('off')

# Memberikan judul utama
fig.suptitle("FLANN Matching dengan Lowe's Ratio Test", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi FLANN matches
plt.savefig(os.path.join(OUTPUT_DIR, "07_flann_matches.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 07_flann_matches.png")

# Menutup figure
plt.close()

# ============================================================
# 7. Variasi Ratio Threshold
# ============================================================

# Mendefinisikan daftar ratio threshold yang akan diuji
ratio_values = [0.5, 0.6, 0.7, 0.8, 0.9]

# Menampilkan header variasi ratio
print(f"\n--- Variasi Ratio Threshold ---")

# Menyimpan jumlah good matches untuk setiap ratio
ratio_counts = []

# Menyimpan jarak rata-rata untuk setiap ratio
ratio_avg_dists = []

# Melakukan iterasi untuk setiap ratio
for ratio in ratio_values:
    # Menyiapkan list good matches untuk ratio ini
    gm = []

    # Menerapkan ratio test
    for m, n in matches_knn:
        if m.distance < ratio * n.distance:
            gm.append(m)

    # Menyimpan jumlah good matches
    ratio_counts.append(len(gm))

    # Menghitung rata-rata jarak
    if len(gm) > 0:
        avg_d = np.mean([m.distance for m in gm])
    else:
        avg_d = 0

    # Menyimpan rata-rata jarak
    ratio_avg_dists.append(avg_d)

    # Menampilkan ke konsol
    print(f"  ratio={ratio}: {len(gm)} good matches ({len(gm)/len(matches_knn)*100:.1f}%), "
          f"avg dist: {avg_d:.2f}")

# Membuat figure untuk variasi ratio
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Menggambar grafik jumlah good matches vs ratio
axes[0].plot(ratio_values, ratio_counts, 'o-', color='steelblue', linewidth=2, markersize=8)

# Memberikan judul dan label
axes[0].set_title("Good Matches vs Ratio Threshold", fontsize=12)
axes[0].set_xlabel("Ratio Threshold", fontsize=11)
axes[0].set_ylabel("Jumlah Good Matches", fontsize=11)
axes[0].grid(True, alpha=0.3)

# Menambahkan anotasi pada setiap titik
for r, c in zip(ratio_values, ratio_counts):
    axes[0].annotate(f'{c}', (r, c), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

# Menggambar grafik jarak rata-rata vs ratio
axes[1].plot(ratio_values, ratio_avg_dists, 's-', color='coral', linewidth=2, markersize=8)

# Memberikan judul dan label
axes[1].set_title("Jarak Rata-rata vs Ratio Threshold", fontsize=12)
axes[1].set_xlabel("Ratio Threshold", fontsize=11)
axes[1].set_ylabel("Jarak Rata-rata", fontsize=11)
axes[1].grid(True, alpha=0.3)

# Menggambar grafik persentase vs ratio
percentages = [c / len(matches_knn) * 100 for c in ratio_counts]

# Menggambar bar chart persentase
bars = axes[2].bar(ratio_values, percentages, width=0.07, color='mediumseagreen',
                    edgecolor='black', alpha=0.8)

# Memberikan judul dan label
axes[2].set_title("Persentase Lolos Ratio Test", fontsize=12)
axes[2].set_xlabel("Ratio Threshold", fontsize=11)
axes[2].set_ylabel("Persentase (%)", fontsize=11)
axes[2].grid(True, alpha=0.3, axis='y')

# Menambahkan label persen pada bar
for bar, pct in zip(bars, percentages):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

# Memberikan judul utama
fig.suptitle("Pengaruh Ratio Threshold pada Lowe's Ratio Test", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi variasi ratio
plt.savefig(os.path.join(OUTPUT_DIR, "07_ratio_test_variasi.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 07_ratio_test_variasi.png")

# Menutup figure
plt.close()

# ============================================================
# 8. Benchmark BF vs FLANN
# ============================================================

# Menentukan jumlah iterasi benchmark
n_runs = 10

# Menampilkan header benchmark
print(f"\n--- Benchmark BF vs FLANN ({n_runs} runs) ---")

# Benchmark BF Matcher (crossCheck)
bf_times = []
bf_match_counts = []
for run in range(n_runs):
    # Membuat BF Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Mencatat waktu mulai
    t_start = time.time()

    # Melakukan matching
    matches_bf = bf.match(desc_left, desc_right)

    # Menghitung waktu
    bf_times.append(time.time() - t_start)

    # Menyimpan jumlah match
    bf_match_counts.append(len(matches_bf))

# Menghitung rata-rata waktu BF
avg_bf_time = np.mean(bf_times)

# Menampilkan hasil BF
print(f"  BF (crossCheck): {avg_bf_time*1000:.2f} ms, "
      f"{int(np.mean(bf_match_counts))} matches")

# Benchmark FLANN + Ratio Test
flann_times = []
flann_good_counts = []
for run in range(n_runs):
    # Membuat FLANN Matcher
    flann_b = cv2.FlannBasedMatcher(index_params, search_params)

    # Mencatat waktu mulai
    t_start = time.time()

    # Melakukan KNN matching
    knn_b = flann_b.knnMatch(desc_left, desc_right, k=2)

    # Menerapkan ratio test
    gm_b = []
    for m, n in knn_b:
        if m.distance < 0.75 * n.distance:
            gm_b.append(m)

    # Menghitung waktu total (matching + ratio test)
    flann_times.append(time.time() - t_start)

    # Menyimpan jumlah good matches
    flann_good_counts.append(len(gm_b))

# Menghitung rata-rata waktu FLANN
avg_flann_time = np.mean(flann_times)

# Menampilkan hasil FLANN
print(f"  FLANN (ratio=0.75): {avg_flann_time*1000:.2f} ms, "
      f"{int(np.mean(flann_good_counts))} good matches")

# Menghitung speedup
speedup = avg_bf_time / avg_flann_time if avg_flann_time > 0 else 0

# Menampilkan speedup
print(f"  FLANN {speedup:.2f}x {'lebih cepat' if speedup > 1 else 'lebih lambat'} dari BF")

# ============================================================
# 9. Perbandingan Kualitas: BF+crossCheck vs FLANN+Ratio Test
# ============================================================

# Menampilkan header perbandingan kualitas
print(f"\n--- Perbandingan Kualitas Match ---")

# Mendapatkan BF matches
bf_final = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_bf_final = bf_final.match(desc_left, desc_right)
matches_bf_final = sorted(matches_bf_final, key=lambda x: x.distance)

# Mendapatkan FLANN good matches
flann_final = cv2.FlannBasedMatcher(index_params, search_params)
knn_final = flann_final.knnMatch(desc_left, desc_right, k=2)
good_final = []
for m, n in knn_final:
    if m.distance < 0.75 * n.distance:
        good_final.append(m)
good_final = sorted(good_final, key=lambda x: x.distance)

# Menghitung statistik untuk BF
bf_distances = [m.distance for m in matches_bf_final]

# Menghitung statistik untuk FLANN
flann_distances = [m.distance for m in good_final]

# Menampilkan perbandingan
print(f"  BF+crossCheck: {len(matches_bf_final)} matches, "
      f"avg dist: {np.mean(bf_distances):.2f}")
print(f"  FLANN+ratio:   {len(good_final)} matches, "
      f"avg dist: {np.mean(flann_distances):.2f}")

# Membuat figure untuk perbandingan BF vs FLANN
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Menampilkan top-50 BF matches
n_vis = 50
img_bf_vis = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                              matches_bf_final[:n_vis], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Menampilkan BF matches
axes[0, 0].imshow(cv2.cvtColor(img_bf_vis, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"BF + crossCheck\nTop {n_vis} dari {len(matches_bf_final)} matches", fontsize=12)
axes[0, 0].axis('off')

# Menampilkan top-50 FLANN good matches
img_flann_vis = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                                 good_final[:n_vis], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Menampilkan FLANN matches
axes[0, 1].imshow(cv2.cvtColor(img_flann_vis, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"FLANN + Ratio Test (0.75)\nTop {n_vis} dari {len(good_final)} matches", fontsize=12)
axes[0, 1].axis('off')

# Menggambar histogram perbandingan jarak
axes[1, 0].hist(bf_distances, bins=25, color='steelblue', alpha=0.6,
                label=f'BF (n={len(matches_bf_final)})', edgecolor='black')
axes[1, 0].hist(flann_distances, bins=25, color='coral', alpha=0.6,
                label=f'FLANN (n={len(good_final)})', edgecolor='black')

# Memberikan judul dan label
axes[1, 0].set_title("Distribusi Jarak Match", fontsize=12)
axes[1, 0].set_xlabel("Jarak (L2)")
axes[1, 0].set_ylabel("Frekuensi")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Membuat bar chart perbandingan waktu dan jumlah match
methods = ['BF+crossCheck', 'FLANN+ratio']
match_counts_cmp = [len(matches_bf_final), len(good_final)]
time_cmp = [avg_bf_time * 1000, avg_flann_time * 1000]

# Membuat twin axis untuk dua metrik pada satu plot
ax_left = axes[1, 1]
ax_right = ax_left.twinx()

# Menggambar bar jumlah match
x_pos = np.arange(len(methods))
bars1 = ax_left.bar(x_pos - 0.2, match_counts_cmp, 0.35, color='steelblue',
                     alpha=0.8, label='Jumlah Match')

# Menggambar bar waktu
bars2 = ax_right.bar(x_pos + 0.2, time_cmp, 0.35, color='coral',
                      alpha=0.8, label='Waktu (ms)')

# Memberikan label sumbu
ax_left.set_ylabel('Jumlah Match', color='steelblue')
ax_right.set_ylabel('Waktu (ms)', color='coral')

# Mengatur label sumbu x
ax_left.set_xticks(x_pos)
ax_left.set_xticklabels(methods)

# Memberikan judul
axes[1, 1].set_title("Perbandingan Metrik", fontsize=12)

# Menggabungkan legend
lines1, labels1 = ax_left.get_legend_handles_labels()
lines2, labels2 = ax_right.get_legend_handles_labels()
ax_left.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

# Mengaktifkan grid
ax_left.grid(True, alpha=0.3, axis='y')

# Memberikan judul utama
fig.suptitle("Perbandingan BF Matcher vs FLANN Matcher", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan perbandingan BF vs FLANN
plt.savefig(os.path.join(OUTPUT_DIR, "07_bf_vs_flann.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 07_bf_vs_flann.png")

# Menutup figure
plt.close()

# ============================================================
# 10. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 7: FLANN MATCHING + RATIO TEST")

# Menampilkan garis pemisah
print("=" * 60)

# Menjelaskan FLANN
print("1. FLANN menggunakan KD-Tree untuk approximate nearest neighbor")
print("   search yang lebih cepat dari BF untuk deskriptor dimensi tinggi")

# Menjelaskan parameter FLANN
print("2. Parameter FLANN:")
print("   - trees: jumlah KD-Tree (lebih banyak = lebih akurat tapi lambat)")
print("   - checks: jumlah pengecekan (lebih banyak = lebih akurat)")

# Menjelaskan ratio test
print("3. Lowe's Ratio Test (d1/d2 < threshold):")
print("   - ratio kecil (0.5): sedikit match tapi sangat akurat")
print("   - ratio besar (0.9): banyak match tapi banyak false positive")
print(f"   - ratio optimal biasanya 0.7-0.8")

# Menampilkan hasil variasi ratio
print("\n   Hasil variasi ratio:")
for r, c, pct in zip(ratio_values, ratio_counts, percentages):
    print(f"   ratio={r}: {c} matches ({pct:.1f}%)")

# Menjelaskan perbandingan BF vs FLANN
print(f"\n4. BF vs FLANN:")
print(f"   BF+crossCheck: {avg_bf_time*1000:.2f} ms, {len(matches_bf_final)} matches")
print(f"   FLANN+ratio:   {avg_flann_time*1000:.2f} ms, {len(good_final)} matches")
print(f"   FLANN {speedup:.2f}x {'lebih cepat' if speedup > 1 else 'lebih lambat'}")

# Menjelaskan rekomendasi
print("\n5. Rekomendasi:")
print("   - BF+crossCheck: dataset kecil, akurasi tinggi dibutuhkan")
print("   - FLANN+ratio test: dataset besar, kecepatan penting")
print("   - Ratio test memberikan kontrol lebih atas kualitas match")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 07_flann_matches.png")
print("  - 07_ratio_test_variasi.png")
print("  - 07_bf_vs_flann.png")

# Menampilkan garis penutup
print("=" * 60)
