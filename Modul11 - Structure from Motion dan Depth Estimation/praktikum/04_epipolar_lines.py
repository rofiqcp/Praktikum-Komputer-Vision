"""
==========================================================================
PERCOBAAN 4: VISUALISASI GARIS EPIPOLAR
==========================================================================
Program ini mempelajari cara menghitung dan memvisualisasikan garis
epipolar pada pasangan gambar stereo. Garis epipolar menunjukkan
di mana titik korespondensi mungkin berada pada gambar lain.

Konsep utama:
- Setiap titik pada gambar kiri memiliki garis epipolar di gambar kanan
- Garis epipolar dihitung dari Fundamental Matrix: l' = F * x
- Semua garis epipolar bertemu di titik epipole
- Titik korespondensi pasti terletak pada garis epipolar yang sesuai

Fungsi utama yang dipelajari:
- cv2.findFundamentalMat()         : Estimasi Fundamental Matrix
- cv2.computeCorrespondEpilines()  : Menghitung garis epipolar
- cv2.line()                       : Menggambar garis pada gambar
- cv2.circle()                     : Menggambar titik pada gambar

Hasil: Visualisasi garis epipolar pada kedua gambar stereo
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan aljabar linier
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
print("PERCOBAAN 4: VISUALISASI GARIS EPIPOLAR")
print("=" * 60)

# ============================================================
# 1. Memuat pasangan gambar stereo
# ============================================================

# Mendefinisikan path gambar kiri
path_left = os.path.join(IMAGE_DIR, "stereo_left.png")

# Mendefinisikan path gambar kanan
path_right = os.path.join(IMAGE_DIR, "stereo_right.png")

# Membaca gambar kiri dalam format BGR
img_left = cv2.imread(path_left)

# Membaca gambar kanan dalam format BGR
img_right = cv2.imread(path_right)

# Memeriksa apakah gambar berhasil dimuat
if img_left is None or img_right is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi dimensi gambar
print(f"\n[INFO] Ukuran gambar: {img_left.shape}")

# Mengkonversi gambar kiri ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kanan ke grayscale
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Deteksi fitur dan matching
# ============================================================

# Menampilkan informasi tahap deteksi fitur
print("\n--- Deteksi Fitur dan Matching ---")

# Membuat detektor SIFT
sift = cv2.SIFT_create()

# Mendeteksi keypoint dan deskriptor pada gambar kiri
kp1, des1 = sift.detectAndCompute(gray_left, None)

# Mendeteksi keypoint dan deskriptor pada gambar kanan
kp2, des2 = sift.detectAndCompute(gray_right, None)

# Menampilkan jumlah keypoint
print(f"[SIFT] Keypoint kiri : {len(kp1)}")
print(f"[SIFT] Keypoint kanan: {len(kp2)}")

# Membuat BFMatcher dengan norma L2
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Melakukan knnMatch dengan k=2
matches = bf.knnMatch(des1, des2, k=2)

# Menerapkan Lowe's Ratio Test
good_matches = []
for m, n in matches:
    # Menyaring match ambigu
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Menampilkan jumlah match yang lolos
print(f"[MATCH] Good matches: {len(good_matches)}")

# Mengekstrak koordinat titik korespondensi
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# ============================================================
# 3. Estimasi Fundamental Matrix
# ============================================================

# Menampilkan informasi tahap estimasi F
print("\n--- Estimasi Fundamental Matrix ---")

# Mengestimasi Fundamental Matrix menggunakan RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0)

# Mengambil mask inlier
inlier_mask = mask.ravel().astype(bool)

# Menyaring hanya titik inlier
pts1_inlier = pts1[inlier_mask]
pts2_inlier = pts2[inlier_mask]

# Menampilkan jumlah inlier
print(f"[RANSAC] Inlier: {np.sum(inlier_mask)}")
print(f"[F] Fundamental Matrix:")
print(F)

# ============================================================
# 4. Fungsi untuk menggambar garis epipolar
# ============================================================

def gambar_garis_epipolar(img, lines, pts, colors):
    """
    Menggambar garis epipolar dan titik pada gambar.
    
    Parameters:
        img   : gambar tempat menggambar
        lines : garis epipolar (ax + by + c = 0)
        pts   : titik korespondensi
        colors: warna untuk setiap garis/titik
    """
    # Membuat salinan gambar agar asli tidak berubah
    img_out = img.copy()
    # Mendapatkan dimensi gambar
    h, w = img_out.shape[:2]
    
    # Iterasi setiap garis epipolar
    for line, pt, color in zip(lines, pts, colors):
        # Mengekstrak koefisien garis: ax + by + c = 0
        a, b, c = line[0], line[1], line[2]
        # Menghitung titik awal garis (x=0)
        if abs(b) > 1e-6:
            x0, y0 = 0, int(-c / b)
        else:
            x0, y0 = int(-c / a), 0
        # Menghitung titik akhir garis (x=w)
        if abs(b) > 1e-6:
            x1, y1 = w, int(-(c + a * w) / b)
        else:
            x1, y1 = int(-c / a), h
        # Menggambar garis epipolar
        cv2.line(img_out, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        # Menggambar titik korespondensi
        cv2.circle(img_out, (int(pt[0]), int(pt[1])), 6, color, -1)
    
    # Mengembalikan gambar dengan garis epipolar
    return img_out

# ============================================================
# 5. Menghitung garis epipolar di kedua gambar
# ============================================================

# Menampilkan informasi tahap penghitungan garis
print("\n--- Menghitung Garis Epipolar ---")

# Memilih sejumlah titik untuk visualisasi (15 titik)
n_lines = min(15, len(pts1_inlier))

# Mengambil sampel titik yang tersebar merata
indices = np.linspace(0, len(pts1_inlier) - 1, n_lines, dtype=int)
pts1_sample = pts1_inlier[indices]
pts2_sample = pts2_inlier[indices]

# Menghitung garis epipolar pada gambar KANAN dari titik di gambar KIRI
# whichImage=1 berarti titik pts1 ada di gambar 1, garis ada di gambar 2
lines2 = cv2.computeCorrespondEpilines(pts1_sample.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)

# Menghitung garis epipolar pada gambar KIRI dari titik di gambar KANAN
# whichImage=2 berarti titik pts2 ada di gambar 2, garis ada di gambar 1
lines1 = cv2.computeCorrespondEpilines(pts2_sample.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)

# Menampilkan jumlah garis epipolar
print(f"[EPILINES] Jumlah garis: {n_lines}")

# Membuat warna yang konsisten untuk setiap pasangan titik
np.random.seed(42)
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n_lines)]

# ============================================================
# 6. Verifikasi: jarak titik ke garis epipolar
# ============================================================

# Menampilkan informasi tahap verifikasi
print("\n--- Verifikasi Jarak Titik ke Garis Epipolar ---")

# Menghitung jarak titik ke garis epipolar yang sesuai
distances = []
for i in range(n_lines):
    # Mengekstrak koefisien garis: ax + by + c = 0
    a, b, c = lines2[i]
    # Mengekstrak koordinat titik korespondensi
    x, y = pts2_sample[i]
    # Menghitung jarak titik ke garis: |ax + by + c| / sqrt(a^2 + b^2)
    dist = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
    # Menyimpan jarak ke list
    distances.append(dist)

# Mengkonversi list menjadi array NumPy
distances = np.array(distances)

# Menampilkan statistik jarak
print(f"[DIST] Rata-rata jarak: {np.mean(distances):.4f} pixels")
print(f"[DIST] Maksimum jarak : {np.max(distances):.4f} pixels")
print(f"[DIST] Minimum jarak  : {np.min(distances):.4f} pixels")

# ============================================================
# 7. Menggambar garis epipolar pada kedua gambar
# ============================================================

# Menampilkan informasi tahap penggambaran
print("\n--- Menggambar Garis Epipolar ---")

# Mengkonversi gambar ke RGB untuk matplotlib
img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

# Menggambar garis epipolar pada gambar kiri (dari titik di gambar kanan)
img_left_epi = gambar_garis_epipolar(img_left_rgb, lines1, pts1_sample, colors)

# Menggambar garis epipolar pada gambar kanan (dari titik di gambar kiri)
img_right_epi = gambar_garis_epipolar(img_right_rgb, lines2, pts2_sample, colors)

# ============================================================
# 8. Visualisasi
# ============================================================

# Membuat figure dengan 4 subplot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Menampilkan garis epipolar pada gambar kiri
axes[0, 0].imshow(img_left_epi)
axes[0, 0].set_title("Garis Epipolar pada Gambar Kiri", fontsize=12)
axes[0, 0].axis("off")

# Menampilkan garis epipolar pada gambar kanan
axes[0, 1].imshow(img_right_epi)
axes[0, 1].set_title("Garis Epipolar pada Gambar Kanan", fontsize=12)
axes[0, 1].axis("off")

# Menampilkan kedua gambar berdampingan dengan garis penghubung
img_combined = np.hstack([img_left_epi, img_right_epi])
axes[1, 0].imshow(img_combined)
axes[1, 0].set_title("Korespondensi Epipolar (Gabungan)", fontsize=12)
axes[1, 0].axis("off")

# Menampilkan histogram jarak titik ke garis epipolar
axes[1, 1].bar(range(n_lines), distances, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel("Index Titik", fontsize=10)
axes[1, 1].set_ylabel("Jarak ke Garis Epipolar (pixels)", fontsize=10)
axes[1, 1].set_title("Jarak Titik ke Garis Epipolar", fontsize=12)
axes[1, 1].axhline(np.mean(distances), color='red', linestyle='--',
                    label=f'Mean = {np.mean(distances):.4f} px')
axes[1, 1].legend()

# Mengatur layout dan judul utama
plt.suptitle("Percobaan 4: Visualisasi Garis Epipolar", fontsize=16, fontweight='bold')
plt.tight_layout()

# Menyimpan figure ke file output
output_path = os.path.join(OUTPUT_DIR, "04_epipolar_lines.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[SAVED] Hasil disimpan di: {output_path}")

# Menampilkan figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 4: VISUALISASI GARIS EPIPOLAR")
print("=" * 60)
print(f"1. Garis epipolar menunjukkan lokasi titik korespondensi pada gambar lain")
print(f"2. Dihitung dari Fundamental Matrix: l' = F * x, l = F^T * x'")
print(f"3. {n_lines} pasang garis epipolar divisualisasikan")
print(f"4. Rata-rata jarak titik ke garis epipolar: {np.mean(distances):.4f} px")
print(f"5. Jarak ideal = 0 px (titik tepat di atas garis)")
print(f"6. Warna konsisten: titik dan garis yang sama memiliki warna sama")
print(f"7. Semua garis epipolar bertemu di titik epipole")
print(f"8. Geometri epipolar fundamental untuk rekonstruksi 3D")
print("=" * 60)
