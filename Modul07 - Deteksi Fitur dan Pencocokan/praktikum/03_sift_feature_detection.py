"""
==========================================================================
PERCOBAAN 3: SIFT FEATURE DETECTION AND DESCRIPTION
==========================================================================
Program ini mempelajari deteksi dan deskripsi fitur menggunakan metode
SIFT (Scale-Invariant Feature Transform). SIFT adalah detektor fitur
yang tahan terhadap perubahan skala, rotasi, dan perubahan iluminasi.

Konsep yang dipelajari:
- Scale-space extrema detection: mencari titik ekstrem di ruang skala
  menggunakan Difference of Gaussians (DoG)
- Keypoint localization: menyaring keypoint yang lemah dan edge
- Orientation assignment: menentukan orientasi dominan keypoint
- Keypoint descriptor: membuat deskriptor 128-dimensi berbasis histogram
  gradien orientasi pada neighbourhood

Fungsi utama yang dipelajari:
- cv2.SIFT_create()             : Membuat objek detektor SIFT
- sift.detectAndCompute()       : Mendeteksi keypoint dan menghitung descriptor
- cv2.drawKeypoints()           : Menggambar keypoint pada gambar
- cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS : Menampilkan ukuran & orientasi

Hasil: Visualisasi fitur SIFT dengan variasi parameter dan analisis distribusi
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
print("PERCOBAAN 3: SIFT FEATURE DETECTION AND DESCRIPTION")
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
# 2. Deteksi SIFT Dasar
# ============================================================

# Membuat objek detektor SIFT dengan parameter default
sift = cv2.SIFT_create()

# Mendeteksi keypoint dan menghitung descriptor
keypoints, descriptors = sift.detectAndCompute(gray_bangunan, None)

# Menampilkan jumlah keypoint yang terdeteksi
print(f"\n[HASIL] Jumlah keypoint SIFT: {len(keypoints)}")

# Menampilkan ukuran dan tipe descriptor
print(f"[HASIL] Ukuran descriptor: {descriptors.shape}")
print(f"[HASIL] Tipe descriptor: {descriptors.dtype}")

# Menghitung rata-rata ukuran keypoint
avg_size = np.mean([kp.size for kp in keypoints])

# Menampilkan rata-rata ukuran keypoint
print(f"[HASIL] Rata-rata ukuran keypoint: {avg_size:.2f}")

# Menghitung rata-rata orientasi keypoint
avg_angle = np.mean([kp.angle for kp in keypoints])

# Menampilkan rata-rata orientasi
print(f"[HASIL] Rata-rata orientasi keypoint: {avg_angle:.2f} derajat")

# Menampilkan respons rata-rata
avg_response = np.mean([kp.response for kp in keypoints])
print(f"[HASIL] Rata-rata respons keypoint: {avg_response:.6f}")

# Menggambar keypoint dengan ukuran dan orientasi (rich keypoints)
img_kp_rich = cv2.drawKeypoints(img_bangunan, keypoints, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Menggambar keypoint sebagai titik sederhana
img_kp_simple = cv2.drawKeypoints(img_bangunan, keypoints, None,
                                   color=(0, 255, 0))

# ============================================================
# 3. Visualisasi Keypoint SIFT Dasar
# ============================================================

# Membuat figure dengan 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Menampilkan gambar asli
axes[0].imshow(cv2.cvtColor(img_bangunan, cv2.COLOR_BGR2RGB))
axes[0].set_title("Gambar Asli", fontsize=12)
axes[0].axis('off')

# Menampilkan keypoint sederhana (titik)
axes[1].imshow(cv2.cvtColor(img_kp_simple, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"SIFT Keypoints ({len(keypoints)})", fontsize=12)
axes[1].axis('off')

# Menampilkan rich keypoints (ukuran + orientasi)
axes[2].imshow(cv2.cvtColor(img_kp_rich, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Rich Keypoints (size + orientation)", fontsize=12)
axes[2].axis('off')

# Memberikan judul utama
fig.suptitle("Percobaan 3: SIFT Feature Detection", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi keypoint SIFT
plt.savefig(os.path.join(OUTPUT_DIR, "03_sift_keypoints.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 03_sift_keypoints.png")

# Menutup figure
plt.close()

# ============================================================
# 4. Variasi Parameter nfeatures
# ============================================================

# Mendefinisikan daftar nilai nfeatures untuk pengujian
nfeatures_list = [100, 500, 1000, 2000]

# Menampilkan header variasi nfeatures
print(f"\n--- Variasi nfeatures ---")

# Membuat figure untuk variasi nfeatures
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Melakukan iterasi untuk setiap nilai nfeatures
for i, nf in enumerate(nfeatures_list):
    # Membuat detektor SIFT dengan nfeatures tertentu
    sift_nf = cv2.SIFT_create(nfeatures=nf)

    # Mendeteksi keypoint dan descriptor
    kp_nf, desc_nf = sift_nf.detectAndCompute(gray_bangunan, None)

    # Menggambar rich keypoints
    img_nf = cv2.drawKeypoints(img_bangunan, kp_nf, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Menampilkan pada subplot
    axes[i].imshow(cv2.cvtColor(img_nf, cv2.COLOR_BGR2RGB))

    # Memberikan judul dengan informasi nfeatures dan jumlah keypoint aktual
    axes[i].set_title(f"nfeatures={nf}\n(actual: {len(kp_nf)})", fontsize=11)

    # Menonaktifkan sumbu
    axes[i].axis('off')

    # Menampilkan info ke konsol
    print(f"  nfeatures={nf}: detected={len(kp_nf)}")

# Memberikan judul utama
fig.suptitle("Variasi Parameter nfeatures pada SIFT", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# ============================================================
# 5. Variasi Parameter contrastThreshold
# ============================================================

# Mendefinisikan daftar nilai contrastThreshold
contrast_thresholds = [0.01, 0.04, 0.08]

# Menampilkan header variasi contrastThreshold
print(f"\n--- Variasi contrastThreshold ---")

# Membuat figure untuk variasi contrastThreshold
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

# Melakukan iterasi untuk setiap nilai contrastThreshold
for i, ct in enumerate(contrast_thresholds):
    # Membuat SIFT dengan contrastThreshold tertentu
    sift_ct = cv2.SIFT_create(contrastThreshold=ct)

    # Mendeteksi keypoint dan descriptor
    kp_ct, desc_ct = sift_ct.detectAndCompute(gray_bangunan, None)

    # Menggambar rich keypoints
    img_ct = cv2.drawKeypoints(img_bangunan, kp_ct, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Menampilkan pada subplot
    axes2[i].imshow(cv2.cvtColor(img_ct, cv2.COLOR_BGR2RGB))

    # Memberikan judul
    axes2[i].set_title(f"contrastThreshold={ct}\n({len(kp_ct)} keypoints)", fontsize=11)

    # Menonaktifkan sumbu
    axes2[i].axis('off')

    # Menampilkan info ke konsol
    print(f"  contrastThreshold={ct}: {len(kp_ct)} keypoints")

# Memberikan judul utama
fig2.suptitle("Variasi contrastThreshold pada SIFT", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan gabungan variasi SIFT
fig_var, axes_var = plt.subplots(2, 4, figsize=(20, 10))

# Baris pertama: variasi nfeatures
for i, nf in enumerate(nfeatures_list):
    sift_v = cv2.SIFT_create(nfeatures=nf)
    kp_v, _ = sift_v.detectAndCompute(gray_bangunan, None)
    img_v = cv2.drawKeypoints(img_bangunan, kp_v, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axes_var[0, i].imshow(cv2.cvtColor(img_v, cv2.COLOR_BGR2RGB))
    axes_var[0, i].set_title(f"nfeatures={nf}\n({len(kp_v)})", fontsize=10)
    axes_var[0, i].axis('off')

# Memberikan label baris pertama
axes_var[0, 0].set_ylabel("nfeatures", fontsize=12)

# Baris kedua: variasi contrastThreshold (3 plot + 1 kosong)
for i, ct in enumerate(contrast_thresholds):
    sift_v = cv2.SIFT_create(contrastThreshold=ct)
    kp_v, _ = sift_v.detectAndCompute(gray_bangunan, None)
    img_v = cv2.drawKeypoints(img_bangunan, kp_v, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axes_var[1, i].imshow(cv2.cvtColor(img_v, cv2.COLOR_BGR2RGB))
    axes_var[1, i].set_title(f"contrastThr={ct}\n({len(kp_v)})", fontsize=10)
    axes_var[1, i].axis('off')

# Menyembunyikan subplot kosong
axes_var[1, 3].axis('off')

# Memberikan label baris kedua
axes_var[1, 0].set_ylabel("contrastThreshold", fontsize=12)

# Memberikan judul utama
fig_var.suptitle("Variasi Parameter SIFT", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi variasi
plt.savefig(os.path.join(OUTPUT_DIR, "03_sift_variasi.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 03_sift_variasi.png")

# Menutup semua figure
plt.close('all')

# ============================================================
# 6. Analisis Distribusi Keypoint (Ukuran dan Orientasi)
# ============================================================

# Mengekstrak ukuran semua keypoint
sizes = [kp.size for kp in keypoints]

# Mengekstrak orientasi semua keypoint (dalam derajat)
angles = [kp.angle for kp in keypoints]

# Mengekstrak respons semua keypoint
responses = [kp.response for kp in keypoints]

# Membuat figure untuk analisis distribusi
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Membuat histogram distribusi ukuran keypoint
axes[0, 0].hist(sizes, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

# Memberikan judul dan label sumbu
axes[0, 0].set_title(f"Distribusi Ukuran Keypoint\n(mean={np.mean(sizes):.2f}, std={np.std(sizes):.2f})", fontsize=11)
axes[0, 0].set_xlabel("Ukuran (Size)", fontsize=10)
axes[0, 0].set_ylabel("Frekuensi", fontsize=10)

# Menambahkan garis rata-rata
axes[0, 0].axvline(np.mean(sizes), color='red', linestyle='--', label=f'Mean: {np.mean(sizes):.2f}')

# Menampilkan legend
axes[0, 0].legend()

# Mengaktifkan grid
axes[0, 0].grid(True, alpha=0.3)

# Membuat histogram distribusi orientasi keypoint
axes[0, 1].hist(angles, bins=36, color='coral', edgecolor='black', alpha=0.7)

# Memberikan judul dan label sumbu
axes[0, 1].set_title(f"Distribusi Orientasi Keypoint\n(mean={np.mean(angles):.2f}°)", fontsize=11)
axes[0, 1].set_xlabel("Orientasi (Derajat)", fontsize=10)
axes[0, 1].set_ylabel("Frekuensi", fontsize=10)

# Mengaktifkan grid
axes[0, 1].grid(True, alpha=0.3)

# Membuat polar/circular plot untuk orientasi
ax_polar = plt.subplot(2, 2, 3, projection='polar')

# Mengkonversi orientasi ke radian
angles_rad = np.deg2rad(angles)

# Membuat histogram circular
n_bins_polar = 36

# Menghitung histogram
counts, bin_edges = np.histogram(angles_rad, bins=n_bins_polar, range=(0, 2*np.pi))

# Menghitung pusat setiap bin
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Menghitung lebar setiap bin
bin_width = 2 * np.pi / n_bins_polar

# Menggambar bar plot polar
ax_polar.bar(bin_centers, counts, width=bin_width, color='lightgreen',
             edgecolor='darkgreen', alpha=0.7)

# Memberikan judul
ax_polar.set_title("Orientasi Keypoint (Polar)", fontsize=11, pad=20)

# Membuat histogram distribusi respons keypoint
axes[1, 1].hist(responses, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)

# Memberikan judul dan label sumbu
axes[1, 1].set_title(f"Distribusi Respons Keypoint\n(mean={np.mean(responses):.6f})", fontsize=11)
axes[1, 1].set_xlabel("Respons", fontsize=10)
axes[1, 1].set_ylabel("Frekuensi", fontsize=10)

# Mengaktifkan grid
axes[1, 1].grid(True, alpha=0.3)

# Memberikan judul utama
fig.suptitle("Analisis Distribusi Keypoint SIFT", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi distribusi
plt.savefig(os.path.join(OUTPUT_DIR, "03_sift_distribusi.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 03_sift_distribusi.png")

# Menutup figure
plt.close()

# ============================================================
# 7. Pengujian pada Gambar dengan Skala Berbeda
# ============================================================

# Mendefinisikan faktor skala yang akan diuji
scale_factors = [0.5, 1.0, 2.0]

# Mendefinisikan label untuk setiap skala
scale_labels = ["50%", "100%", "200%"]

# Menampilkan header pengujian skala
print(f"\n--- Pengujian pada Skala Berbeda ---")

# Membuat figure untuk pengujian skala
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Menyimpan data untuk perbandingan
scale_data = []

# Melakukan iterasi untuk setiap skala
for i, (sf, label) in enumerate(zip(scale_factors, scale_labels)):
    # Mengubah ukuran gambar sesuai faktor skala
    if sf != 1.0:
        # Menghitung dimensi baru
        new_w = int(img_bangunan.shape[1] * sf)
        new_h = int(img_bangunan.shape[0] * sf)

        # Mengubah ukuran gambar
        img_scaled = cv2.resize(img_bangunan, (new_w, new_h))
    else:
        # Menggunakan gambar asli untuk skala 100%
        img_scaled = img_bangunan.copy()

    # Mengkonversi ke grayscale
    gray_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

    # Membuat SIFT dan mendeteksi
    sift_scale = cv2.SIFT_create()

    # Mencatat waktu mulai
    start_time = time.time()

    # Mendeteksi keypoint dan descriptor
    kp_s, desc_s = sift_scale.detectAndCompute(gray_scaled, None)

    # Menghitung waktu proses
    elapsed = time.time() - start_time

    # Menggambar rich keypoints
    img_kp_s = cv2.drawKeypoints(img_scaled, kp_s, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Menghitung rata-rata ukuran keypoint
    avg_s = np.mean([kp.size for kp in kp_s]) if len(kp_s) > 0 else 0

    # Menyimpan data
    scale_data.append({
        'scale': label,
        'keypoints': len(kp_s),
        'avg_size': avg_s,
        'time': elapsed,
        'img_size': img_scaled.shape[:2]
    })

    # Menampilkan keypoint pada subplot baris pertama
    axes[0, i].imshow(cv2.cvtColor(img_kp_s, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"Skala {label}\n({img_scaled.shape[1]}x{img_scaled.shape[0]})", fontsize=11)
    axes[0, i].axis('off')

    # Membuat histogram ukuran keypoint pada baris kedua
    if len(kp_s) > 0:
        sizes_s = [kp.size for kp in kp_s]
        axes[1, i].hist(sizes_s, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, i].set_title(f"{len(kp_s)} keypoints, avg size={avg_s:.2f}", fontsize=10)
    axes[1, i].set_xlabel("Ukuran Keypoint")
    axes[1, i].set_ylabel("Frekuensi")
    axes[1, i].grid(True, alpha=0.3)

    # Menampilkan info ke konsol
    print(f"  Skala {label}: {len(kp_s)} keypoints, avg_size={avg_s:.2f}, waktu={elapsed:.4f}s")

# Memberikan judul utama
fig.suptitle("SIFT pada Skala Gambar Berbeda", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi skala
plt.savefig(os.path.join(OUTPUT_DIR, "03_sift_skala.png"), dpi=150, bbox_inches='tight')

# Menampilkan pesan file tersimpan
print(f"\n[SAVED] 03_sift_skala.png")

# Menutup figure
plt.close()

# ============================================================
# 8. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 3: SIFT FEATURE DETECTION")

# Menampilkan garis pemisah
print("=" * 60)

# Menjelaskan tahapan SIFT
print("1. SIFT terdiri dari 4 tahap utama:")
print("   a. Scale-space extrema detection (DoG pyramid)")
print("   b. Keypoint localization (sub-pixel accuracy)")
print("   c. Orientation assignment (gradient histogram)")
print("   d. Descriptor computation (128-D vector)")

# Menjelaskan parameter nfeatures
print("2. nfeatures mengontrol jumlah keypoint yang dipertahankan")

# Menjelaskan contrastThreshold
print("3. contrastThreshold memfilter keypoint lemah:")
print("   - Threshold rendah: lebih banyak keypoint (termasuk noise)")
print("   - Threshold tinggi: hanya keypoint kontras tinggi")

# Menjelaskan sifat skala-invarian
print("4. SIFT bersifat scale-invariant: mendeteksi fitur yang")
print("   konsisten meskipun ukuran gambar berubah")

# Menampilkan tabel perbandingan skala
print(f"\n{'Skala':<10} {'Keypoints':<12} {'Avg Size':<12} {'Waktu (s)':<12}")
print(f"{'-'*46}")
for d in scale_data:
    print(f"{d['scale']:<10} {d['keypoints']:<12} {d['avg_size']:<12.2f} {d['time']:<12.4f}")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 03_sift_keypoints.png")
print("  - 03_sift_variasi.png")
print("  - 03_sift_distribusi.png")
print("  - 03_sift_skala.png")

# Menampilkan garis penutup
print("=" * 60)
