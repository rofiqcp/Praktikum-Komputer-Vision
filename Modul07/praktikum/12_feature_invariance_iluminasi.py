"""
==========================================================================
PERCOBAAN 12: INVARIANSI FITUR TERHADAP ILUMINASI
==========================================================================
Program ini menguji seberapa robust (tahan) berbagai detektor fitur
terhadap perubahan pencahayaan (brightness). Gambar template dicocokkan
dengan versi yang memiliki tingkat kecerahan berbeda (-80 hingga +80).

Konsep yang dipelajari:
- Illumination invariance: ketahanan fitur terhadap perubahan cahaya
- SIFT: berbasis gradient sehingga relatif tahan terhadap perubahan global
- ORB: menggunakan FAST (intensity comparison) yang sensitif terhadap cahaya
- AKAZE: berbasis nonlinear diffusion yang cukup robust terhadap pencahayaan
- Pengaruh brightness offset terhadap kualitas matching
- Perbandingan kuantitatif robustness antar detektor

Fungsi utama yang dipelajari:
- cv2.SIFT_create()   : Detektor SIFT (gradient-based, relatif robust)
- cv2.ORB_create()     : Detektor ORB (intensity-based, lebih sensitif)
- cv2.AKAZE_create()   : Detektor AKAZE (nonlinear diffusion)
- cv2.BFMatcher()      : Brute-Force matcher untuk pencocokan deskriptor
- cv2.drawMatches()    : Menggambar hasil matching

Hasil: Visualisasi matching pada berbagai brightness dan chart robustness
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
print("PERCOBAAN 12: INVARIANSI FITUR TERHADAP ILUMINASI")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar pada Berbagai Tingkat Kecerahan
# ============================================================

# Mendefinisikan daftar offset brightness yang tersedia
brightness_list = [-80, -40, 0, 40, 80]

# Menyiapkan dictionary untuk menyimpan gambar per brightness
gambar_bright = {}

# Memuat setiap gambar brightness dari file
for bright in brightness_list:
    # Membaca gambar brightness dari file
    path_img = os.path.join(IMAGE_DIR, f"buku_bright{bright:+d}.jpg")
    img = cv2.imread(path_img)

    # Memeriksa apakah gambar berhasil dimuat
    if img is None:
        print(f"[ERROR] File buku_bright{bright:+d}.jpg tidak ditemukan!")
        exit()

    # Menyimpan gambar ke dictionary
    gambar_bright[bright] = img

    # Menampilkan informasi gambar
    gray_info = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[INFO] buku_bright{bright:+d}.jpg: {img.shape}, "
          f"mean brightness: {gray_info.mean():.1f}")

# Mengambil gambar template (brightness 0) sebagai referensi
img_template = gambar_bright[0]

# ============================================================
# 2. Mendefinisikan Detektor dan Matcher
# ============================================================

# Mendefinisikan daftar nama detektor yang akan diuji
nama_detektor = ["SIFT", "ORB", "AKAZE"]

# Menyiapkan dictionary untuk objek detektor
detektors = {}

# Menyiapkan dictionary untuk objek matcher
matchers = {}

# Membuat detektor SIFT
detektors["SIFT"] = cv2.SIFT_create()

# Membuat matcher SIFT (L2 norm)
matchers["SIFT"] = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Membuat detektor ORB
detektors["ORB"] = cv2.ORB_create(nfeatures=1000)

# Membuat matcher ORB (Hamming norm)
matchers["ORB"] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Membuat detektor AKAZE
detektors["AKAZE"] = cv2.AKAZE_create()

# Membuat matcher AKAZE (Hamming norm)
matchers["AKAZE"] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ============================================================
# 3. Menjalankan Matching untuk Setiap Detektor dan Brightness
# ============================================================

# Menyiapkan dictionary untuk menyimpan jumlah good matches
hasil_match = {det: [] for det in nama_detektor}

# Menyiapkan dictionary untuk menyimpan jumlah keypoint
hasil_kp = {det: [] for det in nama_detektor}

# Menampilkan header tabel hasil
print(f"\n{'Detektor':>10} {'Bright':>8} {'KP Tmpl':>10} {'KP Bright':>10} {'Good Match':>12}")
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

    # Melakukan iterasi untuk setiap brightness
    for bright in brightness_list:
        # Mengkonversi gambar brightness ke grayscale
        gray_bright = cv2.cvtColor(gambar_bright[bright], cv2.COLOR_BGR2GRAY)

        # Mendeteksi keypoints dan descriptor pada gambar brightness
        kp_bright, desc_bright = det.detectAndCompute(gray_bright, None)

        # Menyimpan jumlah keypoint
        hasil_kp[nama_det].append(len(kp_bright))

        # Memeriksa ketersediaan deskriptor
        if desc_tmpl is None or desc_bright is None or len(kp_tmpl) < 2 or len(kp_bright) < 2:
            # Menambahkan 0 jika tidak ada deskriptor
            hasil_match[nama_det].append(0)
            print(f"{nama_det:>10} {bright:>+7d} {len(kp_tmpl):>10} {len(kp_bright):>10} {0:>12}")
            continue

        # Melakukan KNN matching dengan k=2
        matches_knn = matcher.knnMatch(desc_tmpl, desc_bright, k=2)

        # Menerapkan Lowe's Ratio Test
        good = []
        for pair in matches_knn:
            if len(pair) == 2:
                m, n = pair
                # Menerapkan threshold ratio 0.75
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        # Menyimpan jumlah good matches
        hasil_match[nama_det].append(len(good))

        # Menampilkan hasil
        print(f"{nama_det:>10} {bright:>+7d} {len(kp_tmpl):>10} {len(kp_bright):>10} {len(good):>12}")

# ============================================================
# 4. Visualisasi Matches pada Setiap Tingkat Kecerahan
# ============================================================

# Membuat figure grid untuk visualisasi matching pada berbagai brightness
fig, axes = plt.subplots(3, 3, figsize=(18, 16))

# Membuat detektor dan matcher untuk visualisasi
det_vis_list = [cv2.SIFT_create(), cv2.ORB_create(nfeatures=1000), cv2.AKAZE_create()]
matcher_vis_list = [
    cv2.BFMatcher(cv2.NORM_L2, crossCheck=False),
    cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
    cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
]

# Mendefinisikan brightness yang akan divisualisasikan
bright_vis = [-80, 0, 80]

# Melakukan iterasi untuk setiap detektor dan brightness
for row, (nama_det, det_v, mat_v) in enumerate(zip(nama_detektor, det_vis_list, matcher_vis_list)):
    # Mengkonversi template ke grayscale
    g_tmpl = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    # Mendeteksi fitur pada template
    kp_t, desc_t = det_v.detectAndCompute(g_tmpl, None)

    for col, bright in enumerate(bright_vis):
        # Mengkonversi gambar brightness ke grayscale
        g_bright = cv2.cvtColor(gambar_bright[bright], cv2.COLOR_BGR2GRAY)

        # Mendeteksi fitur pada gambar brightness
        kp_b, desc_b = det_v.detectAndCompute(g_bright, None)

        # Menyiapkan jumlah match
        n_good = 0

        # Memeriksa ketersediaan deskriptor
        if desc_t is not None and desc_b is not None and len(kp_t) >= 2 and len(kp_b) >= 2:
            # Melakukan KNN matching
            mknn = mat_v.knnMatch(desc_t, desc_b, k=2)

            # Menerapkan ratio test
            good_v = []
            for pair in mknn:
                if len(pair) == 2:
                    a, b = pair
                    if a.distance < 0.75 * b.distance:
                        good_v.append(a)

            # Menghitung jumlah
            n_good = len(good_v)

            # Menggambar matches
            img_m = cv2.drawMatches(img_template, kp_t,
                                    gambar_bright[bright], kp_b,
                                    good_v[:20], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Menampilkan pada subplot
            axes[row, col].imshow(cv2.cvtColor(img_m, cv2.COLOR_BGR2RGB))
        else:
            # Menampilkan pesan tidak ada match
            axes[row, col].text(0.5, 0.5, "Tidak ada match", ha='center', va='center',
                                transform=axes[row, col].transAxes)

        # Memberikan judul subplot
        axes[row, col].set_title(f"{nama_det} | Bright {bright:+d} ({n_good} match)", fontsize=10)

        # Menonaktifkan sumbu
        axes[row, col].axis('off')

# Memberikan judul utama
fig.suptitle("Matching pada Berbagai Tingkat Kecerahan", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file
plt.savefig(os.path.join(OUTPUT_DIR, "12_iluminasi_matches.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"\n[SAVED] 12_iluminasi_matches.png")

# Menutup figure
plt.close()

# ============================================================
# 5. Grafik Robustness terhadap Iluminasi
# ============================================================

# Membuat figure untuk grafik perbandingan
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mendefinisikan warna untuk setiap detektor
warna_detektor = {'SIFT': 'steelblue', 'ORB': 'coral', 'AKAZE': 'seagreen'}

# Mendefinisikan marker untuk setiap detektor
marker_detektor = {'SIFT': 'o', 'ORB': 's', 'AKAZE': '^'}

# Menggambar line chart jumlah match vs brightness offset
for nama_det in nama_detektor:
    # Menggambar garis untuk detektor ini
    axes[0].plot(brightness_list, hasil_match[nama_det],
                 color=warna_detektor[nama_det],
                 marker=marker_detektor[nama_det],
                 linewidth=2, markersize=8, label=nama_det)

# Memberikan label sumbu X
axes[0].set_xlabel("Brightness Offset", fontsize=12)

# Memberikan label sumbu Y
axes[0].set_ylabel("Jumlah Good Matches", fontsize=12)

# Memberikan judul chart
axes[0].set_title("Good Matches vs Brightness Offset", fontsize=13)

# Menambahkan legenda
axes[0].legend(fontsize=11)

# Menambahkan grid
axes[0].grid(True, alpha=0.3)

# Menghitung rata-rata match untuk setiap detektor (non-zero brightness)
rata_rata = {}
for nama_det in nama_detektor:
    # Mengambil match untuk brightness selain 0
    match_non0 = [hasil_match[nama_det][i] for i, b in enumerate(brightness_list) if b != 0]
    # Menghitung rata-rata
    rata_rata[nama_det] = np.mean(match_non0) if match_non0 else 0

# Membuat bar chart rata-rata robustness
bar_names = list(rata_rata.keys())
bar_values = list(rata_rata.values())
bar_colors = [warna_detektor[n] for n in bar_names]

# Menggambar bar chart
bars = axes[1].bar(bar_names, bar_values, color=bar_colors, edgecolor='black')

# Menambahkan label angka
for bar, val in zip(bars, bar_values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha='center', va='bottom', fontsize=10)

# Memberikan label sumbu Y
axes[1].set_ylabel("Rata-rata Good Matches", fontsize=12)

# Memberikan judul chart
axes[1].set_title("Rata-rata Match pada Brightness Berubah", fontsize=13)

# Menambahkan grid
axes[1].grid(axis='y', alpha=0.3)

# Menemukan detektor terbaik
detektor_terbaik = max(rata_rata, key=rata_rata.get)

# Menampilkan informasi detektor terbaik
print(f"\n[ANALISIS] Detektor paling robust terhadap iluminasi: {detektor_terbaik}")
print(f"  Rata-rata match: {rata_rata[detektor_terbaik]:.1f}")

# Memberikan judul utama
fig.suptitle("Analisis Robustness terhadap Perubahan Iluminasi",
             fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan grafik ke file
plt.savefig(os.path.join(OUTPUT_DIR, "12_iluminasi_chart.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"[SAVED] 12_iluminasi_chart.png")

# Menutup figure
plt.close()

# ============================================================
# 6. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 12: INVARIANSI FITUR TERHADAP ILUMINASI")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan illumination invariance
print("1. Illumination invariance berarti detektor dapat mengenali")
print("   fitur meskipun pencahayaan gambar berubah (terang/gelap).")

# Menampilkan penjelasan SIFT terhadap iluminasi
print("2. SIFT berbasis gradient (arah perubahan intensitas) sehingga")
print("   relatif tahan terhadap perubahan brightness global.")

# Menampilkan penjelasan ORB terhadap iluminasi
print("3. ORB menggunakan perbandingan intensitas piksel (FAST corner)")
print("   sehingga lebih sensitif terhadap perubahan pencahayaan.")

# Menampilkan penjelasan AKAZE terhadap iluminasi
print("4. AKAZE menggunakan nonlinear diffusion yang memfilter noise")
print("   tapi tetap dipengaruhi perubahan brightness yang ekstrem.")

# Menampilkan hasil analisis
print(f"5. Pada percobaan ini, {detektor_terbaik} menunjukkan performa")
print(f"   terbaik dengan rata-rata {rata_rata[detektor_terbaik]:.1f} match.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 12_iluminasi_matches.png")
print("  - 12_iluminasi_chart.png")

# Menampilkan garis penutup
print("=" * 60)
