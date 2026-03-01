"""
==========================================================================
PERCOBAAN 5: TRIANGULASI TITIK 3D DARI DUA VIEW
==========================================================================
Program ini mempelajari cara melakukan triangulasi titik 3D dari
pasangan titik korespondensi pada dua gambar. Triangulasi menentukan
posisi 3D titik berdasarkan dua proyeksi 2D dan matriks proyeksi kamera.

Konsep utama:
- Matriks proyeksi P = K [R|t] memetakan titik 3D ke 2D
- Triangulasi menggunakan dua matriks proyeksi P1 dan P2
- Koordinat homogen 4D dikonversi ke Euclidean 3D (bagi dengan w)
- Titik 3D divisualisasikan dengan warna berdasarkan kedalaman (Z)

Fungsi utama yang dipelajari:
- cv2.triangulatePoints()        : Triangulasi titik 3D dari 2 view
- cv2.findEssentialMat()         : Estimasi Essential Matrix
- cv2.recoverPose()              : Recovery pose kamera
- matplotlib Axes3D              : Visualisasi scatter plot 3D

Hasil: Titik-titik 3D hasil triangulasi divisualisasikan dalam 3D
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

# Mengimpor modul 3D dari matplotlib untuk scatter plot 3D
from mpl_toolkits.mplot3d import Axes3D

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
print("PERCOBAAN 5: TRIANGULASI TITIK 3D DARI DUA VIEW")
print("=" * 60)

# ============================================================
# 1. Memuat pasangan gambar stereo
# ============================================================

# Mendefinisikan path gambar kiri
path_left = os.path.join(IMAGE_DIR, "stereo_left.png")

# Mendefinisikan path gambar kanan
path_right = os.path.join(IMAGE_DIR, "stereo_right.png")

# Membaca gambar kiri
img_left = cv2.imread(path_left)

# Membaca gambar kanan
img_right = cv2.imread(path_right)

# Memeriksa apakah gambar berhasil dimuat
if img_left is None or img_right is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi dimensi gambar
print(f"\n[INFO] Ukuran gambar: {img_left.shape}")

# Mengkonversi ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Mendefinisikan matriks intrinsik kamera
# ============================================================

# Menampilkan informasi tahap setup kamera
print("\n--- Setup Matriks Kamera ---")

# Mendapatkan dimensi gambar
h, w = gray_left.shape

# Mendefinisikan focal length berdasarkan lebar gambar
focal_length = w * 1.0

# Mendefinisikan principal point di tengah gambar
cx, cy = w / 2.0, h / 2.0

# Membuat matriks intrinsik K
K = np.array([
    [focal_length, 0, cx],
    [0, focal_length, cy],
    [0, 0, 1]
], dtype=np.float64)

# Menampilkan matriks K
print(f"[K] Focal length: {focal_length:.1f}")
print(f"[K] Principal pt: ({cx:.1f}, {cy:.1f})")

# ============================================================
# 3. Deteksi fitur dan matching
# ============================================================

# Menampilkan informasi tahap deteksi fitur
print("\n--- Deteksi Fitur dan Matching ---")

# Membuat detektor SIFT
sift = cv2.SIFT_create()

# Mendeteksi keypoint dan deskriptor pada kedua gambar
kp1, des1 = sift.detectAndCompute(gray_left, None)
kp2, des2 = sift.detectAndCompute(gray_right, None)

# Menampilkan jumlah keypoint
print(f"[SIFT] Keypoint kiri : {len(kp1)}")
print(f"[SIFT] Keypoint kanan: {len(kp2)}")

# Melakukan feature matching dengan BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Menerapkan Lowe's Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Menampilkan jumlah good matches
print(f"[MATCH] Good matches: {len(good_matches)}")

# Mengekstrak koordinat titik korespondensi
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# ============================================================
# 4. Estimasi Essential Matrix dan Recovery Pose
# ============================================================

# Menampilkan informasi tahap estimasi E dan pose
print("\n--- Essential Matrix dan Recovery Pose ---")

# Mengestimasi Essential Matrix
E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Mengambil mask inlier
inlier_mask = mask_e.ravel().astype(bool)

# Merecovery pose kamera (R dan t)
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

# Menampilkan informasi pose
print(f"[POSE] Inlier: {np.sum(inlier_mask)}")
print(f"[POSE] R (rotation):\n{R}")
print(f"[POSE] t (translation): {t.flatten()}")

# Menyaring titik inlier
pts1_inlier = pts1[inlier_mask]
pts2_inlier = pts2[inlier_mask]

# ============================================================
# 5. Membuat matriks proyeksi P1 dan P2
# ============================================================

# Menampilkan informasi tahap pembuatan matriks proyeksi
print("\n--- Matriks Proyeksi ---")

# Matriks proyeksi kamera 1: P1 = K * [I | 0] (kamera di origin)
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

# Matriks proyeksi kamera 2: P2 = K * [R | t]
P2 = K @ np.hstack([R, t])

# Menampilkan matriks proyeksi
print(f"[P1] Dimensi: {P1.shape}")
print(f"[P2] Dimensi: {P2.shape}")

# ============================================================
# 6. Triangulasi titik 3D
# ============================================================

# Menampilkan informasi tahap triangulasi
print("\n--- Triangulasi Titik 3D ---")

# Melakukan triangulasi menggunakan cv2.triangulatePoints
# Input: matriks proyeksi P1, P2 dan titik korespondensi 2D
points_4d = cv2.triangulatePoints(P1, P2, pts1_inlier.T, pts2_inlier.T)

# Mengkonversi dari koordinat homogen 4D ke Euclidean 3D
# Membagi setiap koordinat dengan komponen w (baris ke-4)
points_3d = points_4d[:3, :] / points_4d[3, :]

# Mentranspose agar setiap baris adalah satu titik 3D
points_3d = points_3d.T

# Menampilkan jumlah titik 3D yang dihasilkan
print(f"[TRI] Jumlah titik 3D: {len(points_3d)}")

# Menampilkan statistik koordinat 3D
print(f"[TRI] X: min={points_3d[:, 0].min():.2f}, max={points_3d[:, 0].max():.2f}")
print(f"[TRI] Y: min={points_3d[:, 1].min():.2f}, max={points_3d[:, 1].max():.2f}")
print(f"[TRI] Z: min={points_3d[:, 2].min():.2f}, max={points_3d[:, 2].max():.2f}")

# ============================================================
# 7. Filtering titik outlier
# ============================================================

# Menampilkan informasi tahap filtering
print("\n--- Filtering Outlier ---")

# Menghitung median kedalaman Z
median_z = np.median(points_3d[:, 2])

# Menyaring titik yang kedalaman Z-nya wajar (dalam 5x median)
valid_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < median_z * 5)

# Menerapkan filter
points_3d_filtered = points_3d[valid_mask]

# Menampilkan jumlah titik setelah filtering
print(f"[FILTER] Titik sebelum filter: {len(points_3d)}")
print(f"[FILTER] Titik setelah filter : {len(points_3d_filtered)}")

# ============================================================
# 8. Visualisasi 3D
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi ---")

# Mengekstrak koordinat X, Y, Z
X = points_3d_filtered[:, 0]
Y = points_3d_filtered[:, 1]
Z = points_3d_filtered[:, 2]

# Menormalisasi depth untuk pewarnaan
depth_normalized = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)

# Mengkonversi gambar ke RGB
img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

# Membuat figure dengan 4 subplot
fig = plt.figure(figsize=(18, 14))

# Menampilkan gambar kiri dengan titik yang ditriangulasi pada subplot pertama
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(img_left_rgb)
pts1_valid = pts1_inlier[valid_mask]
sc = ax1.scatter(pts1_valid[:, 0], pts1_valid[:, 1], c=depth_normalized,
                  cmap='jet', s=10, alpha=0.7)
ax1.set_title("Titik pada Gambar Kiri (warna=depth)", fontsize=12)
ax1.axis("off")
plt.colorbar(sc, ax=ax1, fraction=0.046, label='Depth (normalized)')

# Menampilkan gambar kanan dengan titik pada subplot kedua
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(img_right_rgb)
pts2_valid = pts2_inlier[valid_mask]
sc2 = ax2.scatter(pts2_valid[:, 0], pts2_valid[:, 1], c=depth_normalized,
                   cmap='jet', s=10, alpha=0.7)
ax2.set_title("Titik pada Gambar Kanan (warna=depth)", fontsize=12)
ax2.axis("off")
plt.colorbar(sc2, ax=ax2, fraction=0.046, label='Depth (normalized)')

# Membuat scatter plot 3D pada subplot ketiga
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
sc3 = ax3.scatter(X, Y, Z, c=depth_normalized, cmap='jet', s=5, alpha=0.6)
ax3.set_xlabel("X", fontsize=10)
ax3.set_ylabel("Y", fontsize=10)
ax3.set_zlabel("Z (Depth)", fontsize=10)
ax3.set_title("Titik 3D Hasil Triangulasi", fontsize=12)
plt.colorbar(sc3, ax=ax3, fraction=0.046, label='Depth')

# Menampilkan scatter plot 3D dari sudut pandang berbeda pada subplot keempat
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
sc4 = ax4.scatter(X, Z, -Y, c=depth_normalized, cmap='jet', s=5, alpha=0.6)
ax4.set_xlabel("X", fontsize=10)
ax4.set_ylabel("Z (Depth)", fontsize=10)
ax4.set_zlabel("-Y", fontsize=10)
ax4.set_title("Titik 3D (Tampak Atas)", fontsize=12)
ax4.view_init(elev=60, azim=-90)
plt.colorbar(sc4, ax=ax4, fraction=0.046, label='Depth')

# Mengatur layout dan judul utama
plt.suptitle("Percobaan 5: Triangulasi Titik 3D dari Dua View", fontsize=16, fontweight='bold')
plt.tight_layout()

# Menyimpan figure ke file output
output_path = os.path.join(OUTPUT_DIR, "05_triangulasi_titik_3d.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[SAVED] Hasil disimpan di: {output_path}")

# Menampilkan figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 5: TRIANGULASI TITIK 3D")
print("=" * 60)
print(f"1. Triangulasi menghitung posisi 3D dari dua proyeksi 2D")
print(f"2. Membutuhkan matriks proyeksi P1 = K[I|0] dan P2 = K[R|t]")
print(f"3. cv2.triangulatePoints() menghasilkan koordinat homogen 4D")
print(f"4. Konversi ke Euclidean: X/W, Y/W, Z/W")
print(f"5. {len(points_3d_filtered)} titik 3D berhasil ditriangulasi")
print(f"6. Depth (Z): min={Z.min():.2f}, max={Z.max():.2f}")
print(f"7. Warna berdasarkan kedalaman: biru=dekat, merah=jauh")
print(f"8. Filtering menghapus {len(points_3d) - len(points_3d_filtered)} outlier")
print("=" * 60)
