"""
==========================================================================
PERCOBAAN 10: INVARIANSI FITUR TERHADAP ROTASI
==========================================================================
Program ini menguji seberapa robust (tahan) berbagai detektor fitur
terhadap perubahan rotasi gambar. Gambar template dicocokkan dengan
versi yang diputar pada berbagai sudut (0, 15, 30, 45, 90 derajat).

Konsep yang dipelajari:
- Rotation invariance: kemampuan detektor mengenali fitur meski diputar
- SIFT: invariant terhadap rotasi karena menggunakan orientasi dominan
- ORB: menggunakan oriented FAST + steered BRIEF untuk invariansi rotasi
- AKAZE: menggunakan nonlinear scale space, mendukung rotasi
- Perbandingan kuantitatif antar detektor pada gambar terputar

Fungsi utama yang dipelajari:
- cv2.SIFT_create()   : Membuat detektor SIFT (rotation-invariant)
- cv2.ORB_create()     : Membuat detektor ORB (oriented FAST + BRIEF)
- cv2.AKAZE_create()   : Membuat detektor AKAZE (nonlinear scale space)
- cv2.BFMatcher()      : Membuat Brute-Force matcher
- cv2.drawMatches()    : Menggambar hasil matching

Hasil: Chart perbandingan jumlah match per detektor di setiap rotasi
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
print("PERCOBAAN 10: INVARIANSI FITUR TERHADAP ROTASI")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Template dan Versi Rotasi
# ============================================================

# Mendefinisikan sudut-sudut rotasi yang tersedia
sudut_rotasi = [0, 15, 30, 45, 90]

# Menyiapkan dictionary untuk menyimpan gambar per sudut
gambar_rotasi = {}

# Memuat setiap gambar rotasi dari file
for sudut in sudut_rotasi:
    # Membaca gambar rotasi dari file
    path_img = os.path.join(IMAGE_DIR, f"buku_rot{sudut}.jpg")
    img = cv2.imread(path_img)

    # Memeriksa apakah gambar berhasil dimuat
    if img is None:
        print(f"[ERROR] File buku_rot{sudut}.jpg tidak ditemukan!")
        exit()

    # Menyimpan gambar ke dictionary
    gambar_rotasi[sudut] = img

    # Menampilkan informasi ukuran gambar
    print(f"[INFO] buku_rot{sudut}.jpg: {img.shape}")

# Mengambil gambar template (rotasi 0 derajat) sebagai referensi
img_template = gambar_rotasi[0]

# ============================================================
# 2. Mendefinisikan Detektor dan Matcher
# ============================================================

# Mendefinisikan daftar nama detektor yang akan diuji
nama_detektor = ["SIFT", "ORB", "AKAZE"]

# Menyiapkan dictionary untuk menyimpan objek detektor
detektors = {}

# Menyiapkan dictionary untuk menyimpan objek matcher
matchers = {}

# Membuat detektor SIFT
detektors["SIFT"] = cv2.SIFT_create()

# Membuat matcher untuk SIFT (menggunakan norm L2 karena float descriptor)
matchers["SIFT"] = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Membuat detektor ORB dengan jumlah fitur 1000
detektors["ORB"] = cv2.ORB_create(nfeatures=1000)

# Membuat matcher untuk ORB (menggunakan norm Hamming karena binary descriptor)
matchers["ORB"] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Membuat detektor AKAZE
detektors["AKAZE"] = cv2.AKAZE_create()

# Membuat matcher untuk AKAZE (menggunakan norm Hamming karena binary descriptor)
matchers["AKAZE"] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ============================================================
# 3. Menjalankan Matching untuk Setiap Detektor dan Sudut
# ============================================================

# Menyiapkan dictionary untuk menyimpan jumlah good matches
hasil_match = {det: [] for det in nama_detektor}

# Menyiapkan dictionary untuk menyimpan jumlah keypoint terdeteksi
hasil_kp = {det: [] for det in nama_detektor}

# Menampilkan header tabel hasil
print(f"\n{'Detektor':>10} {'Sudut':>8} {'KP Tmpl':>10} {'KP Rot':>10} {'Good Match':>12}")
print("-" * 55)

# Melakukan iterasi untuk setiap detektor
for nama_det in nama_detektor:
    # Mengambil objek detektor
    det = detektors[nama_det]

    # Mengambil objek matcher
    matcher = matchers[nama_det]

    # Mengkonversi template ke grayscale
    gray_tmpl = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    # Mendeteksi keypoints dan descriptor pada template
    kp_tmpl, desc_tmpl = det.detectAndCompute(gray_tmpl, None)

    # Melakukan iterasi untuk setiap sudut rotasi
    for sudut in sudut_rotasi:
        # Mengkonversi gambar rotasi ke grayscale
        gray_rot = cv2.cvtColor(gambar_rotasi[sudut], cv2.COLOR_BGR2GRAY)

        # Mendeteksi keypoints dan descriptor pada gambar rotasi
        kp_rot, desc_rot = det.detectAndCompute(gray_rot, None)

        # Menyimpan jumlah keypoint
        hasil_kp[nama_det].append(len(kp_rot))

        # Memeriksa apakah deskriptor tersedia
        if desc_tmpl is None or desc_rot is None or len(kp_tmpl) < 2 or len(kp_rot) < 2:
            # Menambahkan 0 jika tidak ada deskriptor
            hasil_match[nama_det].append(0)
            print(f"{nama_det:>10} {sudut:>8}° {len(kp_tmpl):>10} {len(kp_rot):>10} {0:>12}")
            continue

        # Melakukan KNN matching dengan k=2
        matches_knn = matcher.knnMatch(desc_tmpl, desc_rot, k=2)

        # Menerapkan Lowe's Ratio Test
        good = []
        for pair in matches_knn:
            # Memeriksa apakah ada 2 match untuk ratio test
            if len(pair) == 2:
                m, n = pair
                # Menerapkan threshold ratio 0.75
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        # Menyimpan jumlah good matches
        hasil_match[nama_det].append(len(good))

        # Menampilkan hasil
        print(f"{nama_det:>10} {sudut:>8}° {len(kp_tmpl):>10} {len(kp_rot):>10} {len(good):>12}")

# ============================================================
# 4. Visualisasi Matches SIFT pada Setiap Sudut
# ============================================================

# Membuat figure untuk visualisasi matching SIFT
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Meratakan array axes untuk iterasi mudah
axes_flat = axes.flatten()

# Mendefinisikan sudut yang akan divisualisasikan (selain 0)
sudut_vis = [15, 30, 45, 90]

# Membuat detektor SIFT dan matcher untuk visualisasi
sift_vis = cv2.SIFT_create()
matcher_vis = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Mengkonversi template ke grayscale
gray_tmpl_vis = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

# Mendeteksi fitur pada template
kp_vis, desc_vis = sift_vis.detectAndCompute(gray_tmpl_vis, None)

# Melakukan iterasi untuk setiap sudut visualisasi
for idx, sudut in enumerate(sudut_vis):
    # Mengkonversi gambar rotasi ke grayscale
    gray_rot_vis = cv2.cvtColor(gambar_rotasi[sudut], cv2.COLOR_BGR2GRAY)

    # Mendeteksi fitur pada gambar rotasi
    kp_rot_vis, desc_rot_vis = sift_vis.detectAndCompute(gray_rot_vis, None)

    # Memeriksa ketersediaan deskriptor
    if desc_vis is not None and desc_rot_vis is not None:
        # Melakukan KNN matching
        m_knn = matcher_vis.knnMatch(desc_vis, desc_rot_vis, k=2)

        # Menerapkan ratio test
        good_vis = []
        for pair in m_knn:
            if len(pair) == 2:
                a, b = pair
                if a.distance < 0.75 * b.distance:
                    good_vis.append(a)

        # Menggambar matches pada gambar
        img_matches = cv2.drawMatches(img_template, kp_vis,
                                      gambar_rotasi[sudut], kp_rot_vis,
                                      good_vis[:30], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Menampilkan gambar match pada subplot
        axes_flat[idx].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    else:
        # Menampilkan pesan jika tidak ada match
        axes_flat[idx].text(0.5, 0.5, "Tidak ada match", ha='center', va='center')

    # Memberikan judul
    axes_flat[idx].set_title(f"SIFT: Template vs Rotasi {sudut}° ({len(good_vis)} match)", fontsize=11)

    # Menonaktifkan sumbu
    axes_flat[idx].axis('off')

# Memberikan judul utama
fig.suptitle("Matching SIFT pada Berbagai Sudut Rotasi", fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi SIFT ke file
plt.savefig(os.path.join(OUTPUT_DIR, "10_rotasi_sift.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"\n[SAVED] 10_rotasi_sift.png")

# Menutup figure
plt.close()

# ============================================================
# 5. Visualisasi Matches ORB pada Setiap Sudut
# ============================================================

# Membuat figure untuk visualisasi matching ORB
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Meratakan array axes
axes_flat = axes.flatten()

# Membuat detektor ORB dan matcher untuk visualisasi
orb_vis = cv2.ORB_create(nfeatures=1000)
matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Mengkonversi template ke grayscale
gray_tmpl_orb = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

# Mendeteksi fitur ORB pada template
kp_orb_tmpl, desc_orb_tmpl = orb_vis.detectAndCompute(gray_tmpl_orb, None)

# Melakukan iterasi untuk setiap sudut
for idx, sudut in enumerate(sudut_vis):
    # Mengkonversi gambar rotasi ke grayscale
    gray_rot_orb = cv2.cvtColor(gambar_rotasi[sudut], cv2.COLOR_BGR2GRAY)

    # Mendeteksi fitur ORB pada gambar rotasi
    kp_orb_rot, desc_orb_rot = orb_vis.detectAndCompute(gray_rot_orb, None)

    # Memeriksa ketersediaan deskriptor
    good_orb = []
    if desc_orb_tmpl is not None and desc_orb_rot is not None:
        # Melakukan KNN matching
        m_knn_orb = matcher_orb.knnMatch(desc_orb_tmpl, desc_orb_rot, k=2)

        # Menerapkan ratio test
        for pair in m_knn_orb:
            if len(pair) == 2:
                a, b = pair
                if a.distance < 0.75 * b.distance:
                    good_orb.append(a)

        # Menggambar matches
        img_matches_orb = cv2.drawMatches(img_template, kp_orb_tmpl,
                                          gambar_rotasi[sudut], kp_orb_rot,
                                          good_orb[:30], None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Menampilkan gambar pada subplot
        axes_flat[idx].imshow(cv2.cvtColor(img_matches_orb, cv2.COLOR_BGR2RGB))
    else:
        # Menampilkan pesan jika tidak ada match
        axes_flat[idx].text(0.5, 0.5, "Tidak ada match", ha='center', va='center')

    # Memberikan judul
    axes_flat[idx].set_title(f"ORB: Template vs Rotasi {sudut}° ({len(good_orb)} match)", fontsize=11)

    # Menonaktifkan sumbu
    axes_flat[idx].axis('off')

# Memberikan judul utama
fig.suptitle("Matching ORB pada Berbagai Sudut Rotasi", fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ORB ke file
plt.savefig(os.path.join(OUTPUT_DIR, "10_rotasi_orb.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"[SAVED] 10_rotasi_orb.png")

# Menutup figure
plt.close()

# ============================================================
# 6. Grafik Perbandingan Semua Detektor terhadap Rotasi
# ============================================================

# Membuat figure untuk grafik perbandingan
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mendefinisikan warna untuk setiap detektor
warna_detektor = {'SIFT': 'steelblue', 'ORB': 'coral', 'AKAZE': 'seagreen'}

# Mendefinisikan marker untuk setiap detektor
marker_detektor = {'SIFT': 'o', 'ORB': 's', 'AKAZE': '^'}

# Menggambar line chart jumlah match vs sudut rotasi
for nama_det in nama_detektor:
    # Menggambar garis untuk detektor ini
    axes[0].plot(sudut_rotasi, hasil_match[nama_det],
                 color=warna_detektor[nama_det],
                 marker=marker_detektor[nama_det],
                 linewidth=2, markersize=8, label=nama_det)

# Memberikan label sumbu X
axes[0].set_xlabel("Sudut Rotasi (derajat)", fontsize=12)

# Memberikan label sumbu Y
axes[0].set_ylabel("Jumlah Good Matches", fontsize=12)

# Memberikan judul chart
axes[0].set_title("Good Matches vs Sudut Rotasi", fontsize=13)

# Menambahkan legenda
axes[0].legend(fontsize=11)

# Menambahkan grid
axes[0].grid(True, alpha=0.3)

# Menentukan detektor terbaik berdasarkan rata-rata match
rata_rata = {}
for nama_det in nama_detektor:
    # Menghitung rata-rata match (hanya sudut > 0)
    rata_rata[nama_det] = np.mean(hasil_match[nama_det][1:])

# Membuat bar chart rata-rata match per detektor
bar_names = list(rata_rata.keys())
bar_values = list(rata_rata.values())
bar_colors = [warna_detektor[n] for n in bar_names]

# Menggambar bar chart
bars = axes[1].bar(bar_names, bar_values, color=bar_colors, edgecolor='black')

# Menambahkan label angka di atas bar
for bar, val in zip(bars, bar_values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha='center', va='bottom', fontsize=10)

# Memberikan label sumbu Y
axes[1].set_ylabel("Rata-rata Good Matches", fontsize=12)

# Memberikan judul chart
axes[1].set_title("Rata-rata Match pada Gambar Terputar", fontsize=13)

# Menambahkan grid
axes[1].grid(axis='y', alpha=0.3)

# Menemukan detektor terbaik
detektor_terbaik = max(rata_rata, key=rata_rata.get)

# Menampilkan informasi detektor terbaik
print(f"\n[ANALISIS] Detektor paling robust terhadap rotasi: {detektor_terbaik}")
print(f"  Rata-rata match: {rata_rata[detektor_terbaik]:.1f}")

# Memberikan judul utama
fig.suptitle("Perbandingan Invariansi Rotasi: SIFT vs ORB vs AKAZE",
             fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan grafik perbandingan ke file
plt.savefig(os.path.join(OUTPUT_DIR, "10_rotasi_chart.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"[SAVED] 10_rotasi_chart.png")

# Menutup figure
plt.close()

# ============================================================
# 7. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 10: INVARIANSI FITUR TERHADAP ROTASI")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan invariansi rotasi
print("1. Rotation invariance berarti detektor dapat mengenali fitur")
print("   yang sama meskipun orientasi gambar berubah.")

# Menampilkan penjelasan SIFT
print("2. SIFT menggunakan orientasi dominan gradient sehingga")
print("   deskriptornya rotation-invariant secara desain.")

# Menampilkan penjelasan ORB
print("3. ORB menggunakan oriented FAST + steered BRIEF yang")
print("   mempertimbangkan orientasi keypoint untuk invariansi.")

# Menampilkan penjelasan AKAZE
print("4. AKAZE menggunakan nonlinear diffusion yang menjaga")
print("   struktur gambar dan mendukung invariansi rotasi.")

# Menampilkan hasil
print(f"5. Pada percobaan ini, {detektor_terbaik} menunjukkan performa")
print(f"   terbaik dengan rata-rata {rata_rata[detektor_terbaik]:.1f} good matches.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 10_rotasi_sift.png")
print("  - 10_rotasi_orb.png")
print("  - 10_rotasi_chart.png")

# Menampilkan garis penutup
print("=" * 60)
