"""
==========================================================================
PERCOBAAN 2: ESTIMASI FUNDAMENTAL MATRIX
==========================================================================
Program ini mempelajari cara mengestimasi Fundamental Matrix (F) dari
pasangan gambar stereo menggunakan titik korespondensi.

Konsep utama:
- Fundamental Matrix (F) menghubungkan titik pada gambar kiri dengan
  garis epipolar pada gambar kanan, dan sebaliknya.
- Persamaan epipolar: x'^T * F * x = 0
- F berukuran 3x3, rank 2 (det(F) ≈ 0), memiliki 7 derajat kebebasan.
- RANSAC digunakan untuk mengestimasi F secara robust terhadap outlier.

Fungsi utama yang dipelajari:
- cv2.SIFT_create()                   : Detektor fitur SIFT
- cv2.BFMatcher()                     : Brute-Force feature matcher
- cv2.findFundamentalMat()            : Estimasi Fundamental Matrix
- cv2.computeCorrespondEpilines()     : Menghitung garis epipolar

Hasil: Fundamental Matrix, verifikasi properti, dan epipolar constraint
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
print("PERCOBAAN 2: ESTIMASI FUNDAMENTAL MATRIX")
print("=" * 60)

# ============================================================
# 1. Memuat pasangan gambar stereo
# ============================================================

# Mendefinisikan path gambar kiri dari pasangan stereo
path_left = os.path.join(IMAGE_DIR, "stereo_left.png")

# Mendefinisikan path gambar kanan dari pasangan stereo
path_right = os.path.join(IMAGE_DIR, "stereo_right.png")

# Membaca gambar kiri dalam format BGR
img_left = cv2.imread(path_left)

# Membaca gambar kanan dalam format BGR
img_right = cv2.imread(path_right)

# Memeriksa apakah kedua gambar berhasil dimuat
if img_left is None or img_right is None:
    print("[ERROR] Gambar stereo tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi dimensi gambar
print(f"\n[INFO] Ukuran gambar kiri : {img_left.shape}")
print(f"[INFO] Ukuran gambar kanan: {img_right.shape}")

# Mengkonversi gambar kiri ke grayscale untuk deteksi fitur
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kanan ke grayscale untuk deteksi fitur
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

# Menampilkan jumlah keypoint yang terdeteksi
print(f"[SIFT] Keypoint kiri : {len(kp1)}")
print(f"[SIFT] Keypoint kanan: {len(kp2)}")

# Membuat BFMatcher dengan norma L2 untuk deskriptor SIFT
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Melakukan knnMatch dengan k=2 untuk Lowe's ratio test
matches = bf.knnMatch(des1, des2, k=2)

# Menerapkan Lowe's Ratio Test dengan threshold 0.7
good_matches = []
for m, n in matches:
    # Menyaring match yang ambigu berdasarkan rasio jarak
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Menampilkan jumlah match yang lolos ratio test
print(f"[MATCH] Match setelah ratio test: {len(good_matches)}")

# ============================================================
# 3. Mengekstrak titik korespondensi
# ============================================================

# Mengekstrak koordinat titik dari keypoint gambar kiri
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])

# Mengekstrak koordinat titik dari keypoint gambar kanan
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Menampilkan jumlah titik korespondensi
print(f"[INFO] Jumlah titik korespondensi: {len(pts1)}")

# ============================================================
# 4. Estimasi Fundamental Matrix dengan RANSAC
# ============================================================

# Menampilkan informasi tahap estimasi
print("\n--- Estimasi Fundamental Matrix ---")

# Mengestimasi Fundamental Matrix menggunakan algoritma RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0)

# Mengambil mask inlier sebagai array 1D
inlier_mask = mask.ravel().astype(bool)

# Menghitung jumlah inlier (titik yang konsisten dengan model F)
num_inliers = np.sum(inlier_mask)

# Menampilkan jumlah inlier dan outlier
print(f"[RANSAC] Inlier : {num_inliers}")
print(f"[RANSAC] Outlier: {len(pts1) - num_inliers}")
print(f"[RANSAC] Rasio inlier: {num_inliers/len(pts1)*100:.1f}%")

# Menampilkan Fundamental Matrix yang diestimasi
print(f"\n[RESULT] Fundamental Matrix F:")
print(F)

# ============================================================
# 5. Verifikasi properti Fundamental Matrix
# ============================================================

# Menampilkan informasi tahap verifikasi
print("\n--- Verifikasi Properti F ---")

# Menghitung determinan F (seharusnya mendekati 0 karena rank 2)
det_F = np.linalg.det(F)
print(f"[PROP] det(F) = {det_F:.10f} (seharusnya ≈ 0)")

# Menghitung rank F menggunakan SVD
U, S, Vt = np.linalg.svd(F)

# Menampilkan singular values (yang ketiga seharusnya ≈ 0)
print(f"[PROP] Singular values: [{S[0]:.6f}, {S[1]:.6f}, {S[2]:.10f}]")
print(f"[PROP] Rank F = {np.sum(S > 1e-6)} (seharusnya 2)")

# Menghitung norma Frobenius dari F
norm_F = np.linalg.norm(F, 'fro')
print(f"[PROP] Norma Frobenius ||F|| = {norm_F:.6f}")

# ============================================================
# 6. Verifikasi epipolar constraint: x'^T F x ≈ 0
# ============================================================

# Menampilkan informasi tahap verifikasi constraint
print("\n--- Verifikasi Epipolar Constraint ---")

# Menyaring hanya titik-titik inlier
pts1_inlier = pts1[inlier_mask]
pts2_inlier = pts2[inlier_mask]

# Menginisialisasi list untuk menyimpan error epipolar
epipolar_errors = []

# Menghitung error epipolar untuk setiap pasangan titik inlier
for i in range(min(len(pts1_inlier), 50)):
    # Membuat vektor homogen x = [x, y, 1]^T untuk titik di gambar kiri
    x1 = np.array([pts1_inlier[i][0], pts1_inlier[i][1], 1.0])
    # Membuat vektor homogen x' = [x', y', 1]^T untuk titik di gambar kanan
    x2 = np.array([pts2_inlier[i][0], pts2_inlier[i][1], 1.0])
    # Menghitung x'^T * F * x (seharusnya ≈ 0)
    error = abs(x2 @ F @ x1)
    # Menyimpan error ke dalam list
    epipolar_errors.append(error)

# Mengkonversi list error menjadi array NumPy
epipolar_errors = np.array(epipolar_errors)

# Menampilkan statistik error epipolar
print(f"[VERIFY] Rata-rata |x'^T F x| = {np.mean(epipolar_errors):.8f}")
print(f"[VERIFY] Maksimum |x'^T F x| = {np.max(epipolar_errors):.8f}")
print(f"[VERIFY] Minimum |x'^T F x| = {np.min(epipolar_errors):.8f}")

# ============================================================
# 7. Visualisasi
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi ---")

# Mengkonversi gambar dari BGR ke RGB untuk matplotlib
img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

# Membuat figure dengan 4 subplot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Menggambar inlier match pada subplot pertama
inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
# Mengambil 30 match inlier terbaik untuk visualisasi
top_inliers = sorted(inlier_matches, key=lambda x: x.distance)[:30]
img_matches = cv2.drawMatches(img_left_rgb, kp1, img_right_rgb, kp2,
                               top_inliers, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Menampilkan hasil inlier match
axes[0, 0].imshow(img_matches)
axes[0, 0].set_title(f"Inlier Matches ({num_inliers} inliers)", fontsize=12)
axes[0, 0].axis("off")

# Membuat visualisasi garis epipolar pada gambar kanan
img_epi_right = img_right_rgb.copy()

# Menghitung garis epipolar di gambar kanan dari titik-titik di gambar kiri
lines2 = cv2.computeCorrespondEpilines(pts1_inlier[:20].reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)

# Menggambar garis epipolar dan titik korespondensi
h, w = img_right_rgb.shape[:2]
for i, (line, pt2) in enumerate(zip(lines2, pts2_inlier[:20])):
    # Menghitung titik awal dan akhir garis epipolar
    x0, y0 = 0, int(-line[2] / line[1]) if abs(line[1]) > 1e-6 else 0
    x1_end, y1_end = w, int(-(line[2] + line[0] * w) / line[1]) if abs(line[1]) > 1e-6 else 0
    # Menentukan warna acak untuk setiap garis
    color = tuple(np.random.randint(0, 255, 3).tolist())
    # Menggambar garis epipolar
    cv2.line(img_epi_right, (x0, y0), (x1_end, y1_end), color, 1)
    # Menggambar titik korespondensi
    cv2.circle(img_epi_right, (int(pt2[0]), int(pt2[1])), 5, color, -1)

# Menampilkan garis epipolar pada subplot kedua
axes[0, 1].imshow(img_epi_right)
axes[0, 1].set_title("Garis Epipolar pada Gambar Kanan", fontsize=12)
axes[0, 1].axis("off")

# Membuat histogram error epipolar pada subplot ketiga
axes[1, 0].hist(epipolar_errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel("Epipolar Error |x'^T F x|", fontsize=10)
axes[1, 0].set_ylabel("Frekuensi", fontsize=10)
axes[1, 0].set_title("Distribusi Error Epipolar", fontsize=12)
axes[1, 0].axvline(np.mean(epipolar_errors), color='red', linestyle='--',
                    label=f'Mean={np.mean(epipolar_errors):.6f}')
axes[1, 0].legend()

# Menampilkan informasi Fundamental Matrix pada subplot keempat
axes[1, 1].axis("off")
info_text = (
    "FUNDAMENTAL MATRIX\n"
    "=" * 35 + "\n\n"
    f"F =\n{np.array2string(F, precision=6, suppress_small=True)}\n\n"
    f"det(F) = {det_F:.2e}\n"
    f"Singular Values:\n  [{S[0]:.4f}, {S[1]:.4f}, {S[2]:.2e}]\n"
    f"Rank = {np.sum(S > 1e-6)}\n\n"
    f"Inlier: {num_inliers}/{len(pts1)}\n"
    f"Mean epipolar error: {np.mean(epipolar_errors):.2e}"
)
# Menampilkan teks ringkasan
axes[1, 1].text(0.05, 0.5, info_text, fontsize=11, fontfamily='monospace',
                verticalalignment='center', transform=axes[1, 1].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 1].set_title("Informasi Fundamental Matrix", fontsize=12)

# Mengatur layout dan judul utama
plt.suptitle("Percobaan 2: Estimasi Fundamental Matrix", fontsize=16, fontweight='bold')
plt.tight_layout()

# Menyimpan figure ke file output
output_path = os.path.join(OUTPUT_DIR, "02_fundamental_matrix.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[SAVED] Hasil disimpan di: {output_path}")

# Menampilkan figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 2: ESTIMASI FUNDAMENTAL MATRIX")
print("=" * 60)
print(f"1. Fundamental Matrix F berukuran 3x3 menghubungkan 2 view")
print(f"2. Epipolar constraint: x'^T * F * x = 0")
print(f"3. Terdeteksi {len(kp1)} keypoint kiri, {len(kp2)} keypoint kanan")
print(f"4. {len(good_matches)} match lolos Lowe's ratio test")
print(f"5. RANSAC menghasilkan {num_inliers} inlier ({num_inliers/len(pts1)*100:.1f}%)")
print(f"6. det(F) = {det_F:.2e} (mendekati 0, rank 2 terpenuhi)")
print(f"7. Rata-rata error epipolar: {np.mean(epipolar_errors):.2e}")
print(f"8. F digunakan untuk menghitung garis epipolar dan geometri stereo")
print("=" * 60)
