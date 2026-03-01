"""
==========================================================================
PERCOBAAN 11: INVARIANSI FITUR TERHADAP SKALA
==========================================================================
Program ini menguji seberapa robust (tahan) berbagai detektor fitur
terhadap perubahan skala (ukuran) gambar. Gambar template dicocokkan
dengan versi yang diperbesar dan diperkecil pada berbagai faktor skala.

Konsep yang dipelajari:
- Scale invariance: kemampuan detektor mengenali fitur meski ukuran berubah
- SIFT: membangun piramida Gaussian (scale space) untuk deteksi multi-skala
- ORB: menggunakan piramida gambar tapi deskriptor BRIEF kurang scale-robust
- AKAZE: membangun nonlinear scale space untuk ketahanan terhadap skala
- Distribusi ukuran keypoint pada berbagai skala
- Perbandingan kuantitatif antar detektor pada gambar dengan skala berbeda

Fungsi utama yang dipelajari:
- cv2.SIFT_create()   : Detektor SIFT (multi-scale detection)
- cv2.ORB_create()     : Detektor ORB (image pyramid)
- cv2.AKAZE_create()   : Detektor AKAZE (nonlinear scale space)
- cv2.resize()         : Mengubah ukuran gambar
- cv2.BFMatcher()      : Brute-Force matcher

Hasil: Chart perbandingan match per detektor dan distribusi ukuran keypoint
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
print("PERCOBAAN 11: INVARIANSI FITUR TERHADAP SKALA")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar pada Berbagai Skala
# ============================================================

# Mendefinisikan daftar persentase skala yang tersedia
skala_list = [50, 75, 100, 150, 200]

# Menyiapkan dictionary untuk menyimpan gambar per skala
gambar_skala = {}

# Memuat setiap gambar skala dari file
for skala in skala_list:
    # Membaca gambar skala dari file
    path_img = os.path.join(IMAGE_DIR, f"buku_scale{skala}.jpg")
    img = cv2.imread(path_img)

    # Memeriksa apakah gambar berhasil dimuat
    if img is None:
        print(f"[ERROR] File buku_scale{skala}.jpg tidak ditemukan!")
        exit()

    # Menyimpan gambar ke dictionary
    gambar_skala[skala] = img

    # Menampilkan informasi ukuran gambar
    print(f"[INFO] buku_scale{skala}.jpg: {img.shape}")

# Mengambil gambar template (skala 100%) sebagai referensi
img_template = gambar_skala[100]

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

# Membuat matcher SIFT (L2 norm untuk float descriptor)
matchers["SIFT"] = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Membuat detektor ORB dengan 1000 fitur
detektors["ORB"] = cv2.ORB_create(nfeatures=1000)

# Membuat matcher ORB (Hamming norm untuk binary descriptor)
matchers["ORB"] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Membuat detektor AKAZE
detektors["AKAZE"] = cv2.AKAZE_create()

# Membuat matcher AKAZE (Hamming norm untuk binary descriptor)
matchers["AKAZE"] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ============================================================
# 3. Menjalankan Matching untuk Setiap Detektor dan Skala
# ============================================================

# Menyiapkan dictionary untuk menyimpan jumlah good matches
hasil_match = {det: [] for det in nama_detektor}

# Menyiapkan dictionary untuk menyimpan ukuran keypoint
ukuran_kp = {det: {s: [] for s in skala_list} for det in nama_detektor}

# Menampilkan header tabel hasil
print(f"\n{'Detektor':>10} {'Skala':>8} {'KP Tmpl':>10} {'KP Skala':>10} {'Good Match':>12}")
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

    # Menyimpan ukuran keypoint template
    ukuran_kp[nama_det][100] = [kp.size for kp in kp_tmpl]

    # Melakukan iterasi untuk setiap skala
    for skala in skala_list:
        # Mengkonversi gambar skala ke grayscale
        gray_scaled = cv2.cvtColor(gambar_skala[skala], cv2.COLOR_BGR2GRAY)

        # Mendeteksi keypoints dan descriptor pada gambar skala
        kp_scaled, desc_scaled = det.detectAndCompute(gray_scaled, None)

        # Menyimpan ukuran keypoint pada skala ini
        ukuran_kp[nama_det][skala] = [kp.size for kp in kp_scaled]

        # Memeriksa ketersediaan deskriptor
        if desc_tmpl is None or desc_scaled is None or len(kp_tmpl) < 2 or len(kp_scaled) < 2:
            # Menambahkan 0 jika tidak ada deskriptor
            hasil_match[nama_det].append(0)
            print(f"{nama_det:>10} {skala:>7}% {len(kp_tmpl):>10} {len(kp_scaled):>10} {0:>12}")
            continue

        # Melakukan KNN matching dengan k=2
        matches_knn = matcher.knnMatch(desc_tmpl, desc_scaled, k=2)

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
        print(f"{nama_det:>10} {skala:>7}% {len(kp_tmpl):>10} {len(kp_scaled):>10} {len(good):>12}")

# ============================================================
# 4. Visualisasi Matches SIFT pada Berbagai Skala
# ============================================================

# Membuat figure untuk visualisasi matching SIFT
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Meratakan array axes
axes_flat = axes.flatten()

# Mendefinisikan skala yang akan divisualisasikan (selain 100%)
skala_vis = [50, 75, 150, 200]

# Membuat detektor SIFT dan matcher untuk visualisasi
sift_vis = cv2.SIFT_create()
matcher_vis = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Mengkonversi template ke grayscale
gray_tmpl_vis = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

# Mendeteksi fitur pada template
kp_vis, desc_vis = sift_vis.detectAndCompute(gray_tmpl_vis, None)

# Melakukan iterasi untuk setiap skala visualisasi
for idx, skala in enumerate(skala_vis):
    # Mengkonversi gambar skala ke grayscale
    gray_sc = cv2.cvtColor(gambar_skala[skala], cv2.COLOR_BGR2GRAY)

    # Mendeteksi fitur pada gambar skala
    kp_sc, desc_sc = sift_vis.detectAndCompute(gray_sc, None)

    # Menyiapkan list good matches
    good_vis = []

    # Memeriksa ketersediaan deskriptor
    if desc_vis is not None and desc_sc is not None:
        # Melakukan KNN matching
        m_knn = matcher_vis.knnMatch(desc_vis, desc_sc, k=2)

        # Menerapkan ratio test
        for pair in m_knn:
            if len(pair) == 2:
                a, b = pair
                if a.distance < 0.75 * b.distance:
                    good_vis.append(a)

        # Menggambar matches
        img_matches = cv2.drawMatches(img_template, kp_vis,
                                      gambar_skala[skala], kp_sc,
                                      good_vis[:30], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Menampilkan gambar pada subplot
        axes_flat[idx].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    else:
        # Menampilkan pesan jika tidak ada match
        axes_flat[idx].text(0.5, 0.5, "Tidak ada match", ha='center', va='center')

    # Memberikan judul
    axes_flat[idx].set_title(f"SIFT: 100% vs {skala}% ({len(good_vis)} match)", fontsize=11)

    # Menonaktifkan sumbu
    axes_flat[idx].axis('off')

# Memberikan judul utama
fig.suptitle("Matching SIFT pada Berbagai Skala", fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi SIFT ke file
plt.savefig(os.path.join(OUTPUT_DIR, "11_skala_sift.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"\n[SAVED] 11_skala_sift.png")

# Menutup figure
plt.close()

# ============================================================
# 5. Visualisasi Matches ORB pada Berbagai Skala
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

# Melakukan iterasi untuk setiap skala
for idx, skala in enumerate(skala_vis):
    # Mengkonversi gambar skala ke grayscale
    gray_sc_orb = cv2.cvtColor(gambar_skala[skala], cv2.COLOR_BGR2GRAY)

    # Mendeteksi fitur ORB pada gambar skala
    kp_orb_sc, desc_orb_sc = orb_vis.detectAndCompute(gray_sc_orb, None)

    # Menyiapkan list good matches
    good_orb = []

    # Memeriksa ketersediaan deskriptor
    if desc_orb_tmpl is not None and desc_orb_sc is not None:
        # Melakukan KNN matching
        m_knn_orb = matcher_orb.knnMatch(desc_orb_tmpl, desc_orb_sc, k=2)

        # Menerapkan ratio test
        for pair in m_knn_orb:
            if len(pair) == 2:
                a, b = pair
                if a.distance < 0.75 * b.distance:
                    good_orb.append(a)

        # Menggambar matches
        img_matches_orb = cv2.drawMatches(img_template, kp_orb_tmpl,
                                          gambar_skala[skala], kp_orb_sc,
                                          good_orb[:30], None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Menampilkan gambar pada subplot
        axes_flat[idx].imshow(cv2.cvtColor(img_matches_orb, cv2.COLOR_BGR2RGB))
    else:
        # Menampilkan pesan jika tidak ada match
        axes_flat[idx].text(0.5, 0.5, "Tidak ada match", ha='center', va='center')

    # Memberikan judul
    axes_flat[idx].set_title(f"ORB: 100% vs {skala}% ({len(good_orb)} match)", fontsize=11)

    # Menonaktifkan sumbu
    axes_flat[idx].axis('off')

# Memberikan judul utama
fig.suptitle("Matching ORB pada Berbagai Skala", fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ORB ke file
plt.savefig(os.path.join(OUTPUT_DIR, "11_skala_orb.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"[SAVED] 11_skala_orb.png")

# Menutup figure
plt.close()

# ============================================================
# 6. Grafik Perbandingan Semua Detektor terhadap Skala
# ============================================================

# Membuat figure untuk grafik perbandingan
fig, ax = plt.subplots(figsize=(10, 6))

# Mendefinisikan warna untuk setiap detektor
warna_detektor = {'SIFT': 'steelblue', 'ORB': 'coral', 'AKAZE': 'seagreen'}

# Mendefinisikan marker untuk setiap detektor
marker_detektor = {'SIFT': 'o', 'ORB': 's', 'AKAZE': '^'}

# Menggambar line chart jumlah match vs skala
for nama_det in nama_detektor:
    # Menggambar garis untuk detektor ini
    ax.plot(skala_list, hasil_match[nama_det],
            color=warna_detektor[nama_det],
            marker=marker_detektor[nama_det],
            linewidth=2, markersize=8, label=nama_det)

# Memberikan label sumbu X
ax.set_xlabel("Faktor Skala (%)", fontsize=12)

# Memberikan label sumbu Y
ax.set_ylabel("Jumlah Good Matches", fontsize=12)

# Memberikan judul chart
ax.set_title("Good Matches vs Faktor Skala per Detektor", fontsize=14)

# Menambahkan legenda
ax.legend(fontsize=11)

# Menambahkan grid
ax.grid(True, alpha=0.3)

# Menentukan detektor terbaik berdasarkan rata-rata match tanpa skala 100%
rata_rata = {}
for nama_det in nama_detektor:
    # Mengambil match untuk skala selain 100%
    match_non100 = [hasil_match[nama_det][i] for i, s in enumerate(skala_list) if s != 100]
    # Menghitung rata-rata
    rata_rata[nama_det] = np.mean(match_non100) if match_non100 else 0

# Menemukan detektor terbaik
detektor_terbaik = max(rata_rata, key=rata_rata.get)

# Menampilkan informasi detektor terbaik
print(f"\n[ANALISIS] Detektor paling robust terhadap skala: {detektor_terbaik}")
print(f"  Rata-rata match (non-100%): {rata_rata[detektor_terbaik]:.1f}")

# Mengatur layout
plt.tight_layout()

# Menyimpan grafik ke file
plt.savefig(os.path.join(OUTPUT_DIR, "11_skala_chart.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"[SAVED] 11_skala_chart.png")

# Menutup figure
plt.close()

# ============================================================
# 7. Distribusi Ukuran Keypoint pada Berbagai Skala
# ============================================================

# Membuat figure untuk distribusi ukuran keypoint
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Melakukan iterasi untuk setiap detektor
for idx, nama_det in enumerate(nama_detektor):
    # Menyiapkan data untuk boxplot
    data_box = []
    labels_box = []

    for skala in skala_list:
        # Mengambil ukuran keypoint pada skala ini
        sizes = ukuran_kp[nama_det][skala]

        # Menambahkan data jika tersedia
        if len(sizes) > 0:
            data_box.append(sizes)
            labels_box.append(f"{skala}%")

    # Membuat boxplot distribusi ukuran keypoint
    if data_box:
        bp = axes[idx].boxplot(data_box, labels=labels_box, patch_artist=True)

        # Mewarnai box sesuai detektor
        for patch in bp['boxes']:
            patch.set_facecolor(warna_detektor[nama_det])
            patch.set_alpha(0.6)

    # Memberikan label sumbu X
    axes[idx].set_xlabel("Skala", fontsize=11)

    # Memberikan label sumbu Y
    axes[idx].set_ylabel("Ukuran Keypoint (piksel)", fontsize=11)

    # Memberikan judul
    axes[idx].set_title(f"Distribusi Ukuran KP: {nama_det}", fontsize=12)

    # Menambahkan grid
    axes[idx].grid(axis='y', alpha=0.3)

# Memberikan judul utama
fig.suptitle("Distribusi Ukuran Keypoint pada Berbagai Skala", fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan distribusi ke file
plt.savefig(os.path.join(OUTPUT_DIR, "11_skala_distribusi.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan
print(f"[SAVED] 11_skala_distribusi.png")

# Menutup figure
plt.close()

# ============================================================
# 8. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 11: INVARIANSI FITUR TERHADAP SKALA")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan scale invariance
print("1. Scale invariance berarti detektor mengenali fitur yang sama")
print("   meskipun ukuran gambar berubah (diperbesar/diperkecil).")

# Menampilkan penjelasan SIFT
print("2. SIFT membangun piramida Gaussian (scale space) sehingga")
print("   fitur terdeteksi pada skala optimalnya, sangat scale-robust.")

# Menampilkan penjelasan ORB
print("3. ORB menggunakan piramida gambar sederhana, sehingga")
print("   ketahanan terhadap skala terbatas dibanding SIFT.")

# Menampilkan penjelasan AKAZE
print("4. AKAZE membangun nonlinear scale space yang menjaga")
print("   tepi dan detail, memberikan ketahanan skala yang baik.")

# Menampilkan hasil analisis
print(f"5. Pada percobaan ini, {detektor_terbaik} menunjukkan performa")
print(f"   terbaik terhadap perubahan skala.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 11_skala_sift.png")
print("  - 11_skala_orb.png")
print("  - 11_skala_chart.png")
print("  - 11_skala_distribusi.png")

# Menampilkan garis penutup
print("=" * 60)
