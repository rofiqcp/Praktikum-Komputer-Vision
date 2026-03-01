"""
==========================================================================
PERCOBAAN 4: ORB FEATURE DETECTION AND DESCRIPTION
==========================================================================
Program ini mempelajari deteksi dan deskripsi fitur menggunakan metode
ORB (Oriented FAST and Rotated BRIEF). ORB adalah alternatif cepat
dan efisien untuk SIFT/SURF yang menghasilkan deskriptor biner.

Konsep yang dipelajari:
- FAST keypoint detector: mendeteksi sudut dengan segment test
- Oriented FAST: menambahkan orientasi pada keypoint FAST
- Rotated BRIEF: membuat deskriptor biner yang tahan rotasi
- Perbandingan ORB vs SIFT dalam kecepatan dan akurasi
- Pengaruh parameter nfeatures, scaleFactor, dan nlevels

Fungsi utama yang dipelajari:
- cv2.ORB_create()              : Membuat objek detektor ORB
- orb.detectAndCompute()        : Mendeteksi keypoint dan menghitung descriptor
- cv2.drawKeypoints()           : Menggambar keypoint pada gambar
- time.time()                   : Mengukur waktu eksekusi

Hasil: Visualisasi fitur ORB, perbandingan dengan SIFT, dan benchmark waktu
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan deteksi fitur
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
print("PERCOBAAN 4: ORB FEATURE DETECTION AND DESCRIPTION")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar
# ============================================================

# Membaca gambar bangunan dari file
img_bangunan = cv2.imread(os.path.join(IMAGE_DIR, "bangunan.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_bangunan is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi ukuran gambar
print(f"[INFO] Ukuran bangunan: {img_bangunan.shape}")

# Mengkonversi gambar ke grayscale
gray_bangunan = cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Deteksi ORB Dasar
# ============================================================

# Membuat objek detektor ORB dengan 500 fitur
orb = cv2.ORB_create(nfeatures=500)

# Mendeteksi keypoint dan menghitung descriptor
kp_orb, desc_orb = orb.detectAndCompute(gray_bangunan, None)

# Menampilkan jumlah keypoint ORB
print(f"\n[HASIL] Jumlah keypoint ORB: {len(kp_orb)}")

# Menampilkan informasi descriptor
print(f"[HASIL] Shape descriptor: {desc_orb.shape}")

# Menampilkan tipe data descriptor (uint8 = biner)
print(f"[HASIL] Tipe descriptor: {desc_orb.dtype}")

# Menampilkan 5 nilai pertama dari descriptor pertama
print(f"[HASIL] Descriptor pertama (5 nilai awal): {desc_orb[0][:5]}")

# Menampilkan panjang descriptor dalam bit
print(f"[HASIL] Panjang descriptor: {desc_orb.shape[1]} bytes = {desc_orb.shape[1]*8} bits")

# Menggambar rich keypoints (menampilkan ukuran dan orientasi)
img_orb_rich = cv2.drawKeypoints(img_bangunan, kp_orb, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Menggambar keypoint sederhana
img_orb_simple = cv2.drawKeypoints(img_bangunan, kp_orb, None,
                                    color=(0, 0, 255))

# ============================================================
# 3. Visualisasi Keypoint ORB Dasar
# ============================================================

# Membuat figure dengan 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Menampilkan gambar asli
axes[0].imshow(cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2RGB))
axes[0].set_title("Gambar Asli", fontsize=12)
axes[0].axis('off')

# Menampilkan keypoint sederhana
axes[1].imshow(cv2.cvtColor(img_orb_simple, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"ORB Keypoints ({len(kp_orb)})", fontsize=12)
axes[1].axis('off')

# Menampilkan rich keypoints
axes[2].imshow(cv2.cvtColor(img_orb_rich, cv2.COLOR_BGR2RGB))
axes[2].set_title("Rich Keypoints (size + orientation)", fontsize=12)
axes[2].axis('off')

# Memberikan judul utama
fig.suptitle("Percobaan 4: ORB Feature Detection", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi keypoint ORB
plt.savefig(os.path.join(OUTPUT_DIR, "04_orb_keypoints.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 04_orb_keypoints.png")

# Menutup figure
plt.close()

# ============================================================
# 4. Variasi Parameter nfeatures
# ============================================================

# Mendefinisikan daftar nilai nfeatures
nfeatures_list = [100, 500, 1000, 5000]

# Menampilkan header variasi nfeatures
print(f"\n--- Variasi nfeatures ORB ---")

# ============================================================
# 5. Variasi scaleFactor dan nlevels
# ============================================================

# Mendefinisikan daftar scaleFactor
scale_factors = [1.1, 1.2, 1.5]

# Mendefinisikan daftar nlevels
nlevels_list = [4, 8, 12]

# Membuat figure gabungan untuk semua variasi
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: variasi nfeatures
for i, nf in enumerate(nfeatures_list):
    # Membuat ORB dengan nfeatures tertentu
    orb_nf = cv2.ORB_create(nfeatures=nf)

    # Mendeteksi keypoint dan descriptor
    kp_nf, desc_nf = orb_nf.detectAndCompute(gray_bangunan, None)

    # Menggambar rich keypoints
    img_nf = cv2.drawKeypoints(img_bangunan, kp_nf, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Menampilkan pada subplot
    axes[0, i].imshow(cv2.cvtColor(img_nf, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"nfeatures={nf}\n({len(kp_nf)} detected)", fontsize=10)
    axes[0, i].axis('off')

    # Menampilkan info ke konsol
    print(f"  nfeatures={nf}: {len(kp_nf)} keypoints")

# Memberikan label baris
axes[0, 0].set_ylabel("nfeatures", fontsize=12)

# Baris 2: variasi scaleFactor (dengan nfeatures=500 tetap)
print(f"\n--- Variasi scaleFactor ---")
for i, sf in enumerate(scale_factors):
    # Membuat ORB dengan scaleFactor tertentu
    orb_sf = cv2.ORB_create(nfeatures=500, scaleFactor=sf)

    # Mendeteksi keypoint
    kp_sf, desc_sf = orb_sf.detectAndCompute(gray_bangunan, None)

    # Menggambar keypoints
    img_sf = cv2.drawKeypoints(img_bangunan, kp_sf, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Menampilkan pada subplot baris kedua
    axes[1, i].imshow(cv2.cvtColor(img_sf, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"scaleFactor={sf}\n({len(kp_sf)} detected)", fontsize=10)
    axes[1, i].axis('off')

    # Menampilkan info ke konsol
    print(f"  scaleFactor={sf}: {len(kp_sf)} keypoints")

# Menyembunyikan subplot kosong di baris 2
axes[1, 3].axis('off')

# Memberikan label baris
axes[1, 0].set_ylabel("scaleFactor", fontsize=12)

# Baris 3: variasi nlevels
print(f"\n--- Variasi nlevels ---")
for i, nl in enumerate(nlevels_list):
    # Membuat ORB dengan nlevels tertentu
    orb_nl = cv2.ORB_create(nfeatures=500, nlevels=nl)

    # Mendeteksi keypoint
    kp_nl, desc_nl = orb_nl.detectAndCompute(gray_bangunan, None)

    # Menggambar keypoints
    img_nl = cv2.drawKeypoints(img_bangunan, kp_nl, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Menampilkan pada subplot baris ketiga
    axes[2, i].imshow(cv2.cvtColor(img_nl, cv2.COLOR_BGR2RGB))
    axes[2, i].set_title(f"nlevels={nl}\n({len(kp_nl)} detected)", fontsize=10)
    axes[2, i].axis('off')

    # Menampilkan info ke konsol
    print(f"  nlevels={nl}: {len(kp_nl)} keypoints")

# Menyembunyikan subplot kosong di baris 3
axes[2, 3].axis('off')

# Memberikan label baris
axes[2, 0].set_ylabel("nlevels", fontsize=12)

# Memberikan judul utama
fig.suptitle("Variasi Parameter ORB", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi variasi
plt.savefig(os.path.join(OUTPUT_DIR, "04_orb_variasi.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 04_orb_variasi.png")

# Menutup figure
plt.close()

# ============================================================
# 6. Perbandingan Distribusi Spasial ORB vs SIFT
# ============================================================

# Membuat detektor SIFT dengan 500 fitur untuk perbandingan
sift = cv2.SIFT_create(nfeatures=500)

# Mendeteksi keypoint SIFT
kp_sift, desc_sift = sift.detectAndCompute(gray_bangunan, None)

# Membuat detektor ORB dengan 500 fitur
orb_cmp = cv2.ORB_create(nfeatures=500)

# Mendeteksi keypoint ORB
kp_orb_cmp, desc_orb_cmp = orb_cmp.detectAndCompute(gray_bangunan, None)

# Menampilkan perbandingan jumlah keypoint
print(f"\n--- Perbandingan ORB vs SIFT ---")
print(f"  SIFT keypoints: {len(kp_sift)}")
print(f"  ORB keypoints: {len(kp_orb_cmp)}")

# Membuat figure untuk perbandingan
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Menggambar rich keypoints SIFT
img_sift_vis = cv2.drawKeypoints(img_bangunan, kp_sift, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Menggambar rich keypoints ORB
img_orb_vis = cv2.drawKeypoints(img_bangunan, kp_orb_cmp, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Menampilkan keypoints SIFT
axes[0, 0].imshow(cv2.cvtColor(img_sift_vis, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"SIFT ({len(kp_sift)} keypoints)", fontsize=12)
axes[0, 0].axis('off')

# Menampilkan keypoints ORB
axes[0, 1].imshow(cv2.cvtColor(img_orb_vis, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"ORB ({len(kp_orb_cmp)} keypoints)", fontsize=12)
axes[0, 1].axis('off')

# Menampilkan distribusi spasial SIFT sebagai scatter plot
sift_x = [kp.pt[0] for kp in kp_sift]
sift_y = [kp.pt[1] for kp in kp_sift]
sift_s = [kp.size for kp in kp_sift]

# Menampilkan scatter plot SIFT
axes[1, 0].scatter(sift_x, sift_y, s=[s*2 for s in sift_s], alpha=0.5, c='blue', edgecolors='darkblue')
axes[1, 0].set_title("Distribusi Spasial SIFT", fontsize=12)
axes[1, 0].set_xlabel("X")
axes[1, 0].set_ylabel("Y")

# Membalik sumbu Y agar sesuai dengan koordinat gambar
axes[1, 0].invert_yaxis()

# Mengatur aspek rasio
axes[1, 0].set_aspect('equal')
axes[1, 0].grid(True, alpha=0.3)

# Menampilkan distribusi spasial ORB sebagai scatter plot
orb_x = [kp.pt[0] for kp in kp_orb_cmp]
orb_y = [kp.pt[1] for kp in kp_orb_cmp]
orb_s = [kp.size for kp in kp_orb_cmp]

# Menampilkan scatter plot ORB
axes[1, 1].scatter(orb_x, orb_y, s=[s*2 for s in orb_s], alpha=0.5, c='red', edgecolors='darkred')
axes[1, 1].set_title("Distribusi Spasial ORB", fontsize=12)
axes[1, 1].set_xlabel("X")
axes[1, 1].set_ylabel("Y")

# Membalik sumbu Y
axes[1, 1].invert_yaxis()
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True, alpha=0.3)

# Memberikan judul utama
fig.suptitle("Perbandingan ORB vs SIFT", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi perbandingan
plt.savefig(os.path.join(OUTPUT_DIR, "04_orb_vs_sift.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 04_orb_vs_sift.png")

# Menutup figure
plt.close()

# ============================================================
# 7. Benchmark Waktu: ORB vs SIFT
# ============================================================

# Menentukan jumlah iterasi untuk rata-rata
n_runs = 10

# Menampilkan header benchmark
print(f"\n--- Benchmark Waktu (rata-rata {n_runs} run) ---")

# Mengukur waktu SIFT
sift_times = []
for run in range(n_runs):
    # Membuat SIFT baru setiap run
    sift_bench = cv2.SIFT_create(nfeatures=500)

    # Mencatat waktu mulai
    t_start = time.time()

    # Mendeteksi dan menghitung descriptor
    sift_bench.detectAndCompute(gray_bangunan, None)

    # Menghitung waktu
    sift_times.append(time.time() - t_start)

# Menghitung rata-rata waktu SIFT
avg_sift_time = np.mean(sift_times)

# Menampilkan rata-rata waktu SIFT
print(f"  SIFT: {avg_sift_time*1000:.2f} ms (std: {np.std(sift_times)*1000:.2f} ms)")

# Mengukur waktu ORB
orb_times = []
for run in range(n_runs):
    # Membuat ORB baru setiap run
    orb_bench = cv2.ORB_create(nfeatures=500)

    # Mencatat waktu mulai
    t_start = time.time()

    # Mendeteksi dan menghitung descriptor
    orb_bench.detectAndCompute(gray_bangunan, None)

    # Menghitung waktu
    orb_times.append(time.time() - t_start)

# Menghitung rata-rata waktu ORB
avg_orb_time = np.mean(orb_times)

# Menampilkan rata-rata waktu ORB
print(f"  ORB: {avg_orb_time*1000:.2f} ms (std: {np.std(orb_times)*1000:.2f} ms)")

# Menghitung rasio kecepatan ORB terhadap SIFT
speedup = avg_sift_time / avg_orb_time if avg_orb_time > 0 else 0

# Menampilkan rasio kecepatan
print(f"  ORB {speedup:.1f}x lebih cepat dari SIFT")

# Membuat figure untuk benchmark
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Membuat bar chart perbandingan waktu
methods = ['SIFT', 'ORB']
times_ms = [avg_sift_time * 1000, avg_orb_time * 1000]
stds_ms = [np.std(sift_times) * 1000, np.std(orb_times) * 1000]
colors = ['steelblue', 'coral']

# Menggambar bar chart
bars = axes[0].bar(methods, times_ms, yerr=stds_ms, color=colors,
                    edgecolor='black', capsize=5, alpha=0.8)

# Memberikan judul dan label
axes[0].set_title("Waktu Rata-rata Deteksi + Deskripsi", fontsize=12)
axes[0].set_ylabel("Waktu (ms)", fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Menambahkan label nilai pada bar
for bar, t in zip(bars, times_ms):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{t:.2f} ms', ha='center', va='bottom', fontsize=10)

# Membuat box plot perbandingan distribusi waktu
data_bp = [np.array(sift_times)*1000, np.array(orb_times)*1000]

# Menggambar box plot
bp = axes[1].boxplot(data_bp, labels=methods, patch_artist=True)

# Mewarnai box plot
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Memberikan judul dan label
axes[1].set_title(f"Distribusi Waktu ({n_runs} runs)", fontsize=12)
axes[1].set_ylabel("Waktu (ms)", fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

# Memberikan judul utama
fig.suptitle(f"Benchmark Waktu: ORB vs SIFT (ORB {speedup:.1f}x lebih cepat)",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan benchmark
plt.savefig(os.path.join(OUTPUT_DIR, "04_orb_timing.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 04_orb_timing.png")

# Menutup figure
plt.close()

# ============================================================
# 8. Pengujian pada Gambar Rotasi
# ============================================================

# Mendefinisikan sudut rotasi yang akan diuji
rotation_angles = [0, 45, 90, 180]

# Menampilkan header pengujian rotasi
print(f"\n--- Pengujian pada Gambar Rotasi ---")

# Membuat figure untuk pengujian rotasi
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Mendapatkan dimensi gambar
h, w = gray_bangunan.shape[:2]

# Menentukan pusat rotasi
center = (w // 2, h // 2)

# Melakukan iterasi untuk setiap sudut rotasi
for i, angle in enumerate(rotation_angles):
    # Membuat matriks rotasi
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Merotasi gambar
    img_rot = cv2.warpAffine(img_bangunan, M, (w, h))

    # Mengkonversi ke grayscale
    gray_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)

    # Mendeteksi ORB pada gambar terotasi
    orb_rot = cv2.ORB_create(nfeatures=500)
    kp_rot_orb, desc_rot_orb = orb_rot.detectAndCompute(gray_rot, None)

    # Mendeteksi SIFT pada gambar terotasi
    sift_rot = cv2.SIFT_create(nfeatures=500)
    kp_rot_sift, desc_rot_sift = sift_rot.detectAndCompute(gray_rot, None)

    # Menggambar keypoints ORB
    img_orb_rot = cv2.drawKeypoints(img_rot, kp_rot_orb, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Menggambar keypoints SIFT
    img_sift_rot = cv2.drawKeypoints(img_rot, kp_rot_sift, None,
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Menampilkan ORB pada baris pertama
    axes[0, i].imshow(cv2.cvtColor(img_orb_rot, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"ORB rot={angle}°\n({len(kp_rot_orb)} kp)", fontsize=10)
    axes[0, i].axis('off')

    # Menampilkan SIFT pada baris kedua
    axes[1, i].imshow(cv2.cvtColor(img_sift_rot, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"SIFT rot={angle}°\n({len(kp_rot_sift)} kp)", fontsize=10)
    axes[1, i].axis('off')

    # Menampilkan info ke konsol
    print(f"  Rotasi {angle}°: ORB={len(kp_rot_orb)}, SIFT={len(kp_rot_sift)}")

# Memberikan label baris
axes[0, 0].set_ylabel("ORB", fontsize=12)
axes[1, 0].set_ylabel("SIFT", fontsize=12)

# Memberikan judul utama
fig.suptitle("ORB vs SIFT pada Gambar Rotasi", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi rotasi (disimpan sebagai bagian dari orb_vs_sift)
plt.savefig(os.path.join(OUTPUT_DIR, "04_orb_rotasi.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 04_orb_rotasi.png")

# Menutup figure
plt.close()

# ============================================================
# 9. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 4: ORB FEATURE DETECTION")

# Menampilkan garis pemisah
print("=" * 60)

# Menjelaskan komponen ORB
print("1. ORB menggabungkan FAST detector + oriented BRIEF descriptor")

# Menjelaskan deskriptor biner
print("2. Descriptor ORB bersifat biner (uint8), 32 bytes = 256 bits")
print("   -> Lebih efisien memori dibanding SIFT (128 float32)")

# Menjelaskan perbandingan kecepatan
print(f"3. ORB {speedup:.1f}x lebih cepat dari SIFT")
print(f"   SIFT: {avg_sift_time*1000:.2f} ms, ORB: {avg_orb_time*1000:.2f} ms")

# Menjelaskan parameter
print("4. nfeatures: jumlah keypoint yang dipertahankan")
print("   scaleFactor: faktor penurunan resolusi antar level piramida")
print("   nlevels: jumlah level dalam piramida gambar")

# Menjelaskan rotasi
print("5. Kedua metode relatif stabil terhadap rotasi,")
print("   tetapi ORB menggunakan pendekatan intensitas (FAST)")
print("   yang sensitif terhadap blur akibat rotasi")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 04_orb_keypoints.png")
print("  - 04_orb_variasi.png")
print("  - 04_orb_vs_sift.png")
print("  - 04_orb_timing.png")
print("  - 04_orb_rotasi.png")

# Menampilkan garis penutup
print("=" * 60)
