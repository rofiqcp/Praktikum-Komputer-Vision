"""
==========================================================================
PERCOBAAN 3: ESSENTIAL MATRIX DAN RECOVERY POSE KAMERA
==========================================================================
Program ini mempelajari cara mengestimasi Essential Matrix (E) dari
pasangan gambar stereo dan merecovery pose kamera (rotasi R dan
translasi t) menggunakan dekomposisi matriks E.

Konsep utama:
- Essential Matrix E = K'^T * F * K, di mana K adalah matriks intrinsik
- E mengkodekan rotasi dan translasi relatif antara dua kamera
- recoverPose() mendekomposisi E menjadi R dan t (hingga skala)
- Sudut rotasi diekstrak dari matriks rotasi R menggunakan Rodrigues

Fungsi utama yang dipelajari:
- cv2.SIFT_create()                : Detektor fitur SIFT
- cv2.findEssentialMat()           : Estimasi Essential Matrix
- cv2.recoverPose()                : Recovery pose kamera dari E
- cv2.Rodrigues()                  : Konversi rotation vector <-> matrix

Hasil: Essential Matrix, rotation matrix R, translation vector t,
       sudut rotasi dalam derajat
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
print("PERCOBAAN 3: ESSENTIAL MATRIX DAN RECOVERY POSE KAMERA")
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

# Memeriksa apakah kedua gambar berhasil dimuat
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
# 2. Mendefinisikan matriks intrinsik kamera (sintetis)
# ============================================================

# Menampilkan informasi tahap pembuatan matriks intrinsik
print("\n--- Matriks Intrinsik Kamera ---")

# Mendapatkan ukuran gambar
h, w = gray_left.shape

# Mendefinisikan focal length (f) berdasarkan lebar gambar
focal_length = w * 1.0

# Mendefinisikan titik pusat gambar sebagai principal point
cx, cy = w / 2.0, h / 2.0

# Membuat matriks intrinsik kamera K (3x3)
K = np.array([
    [focal_length, 0, cx],
    [0, focal_length, cy],
    [0, 0, 1]
], dtype=np.float64)

# Menampilkan matriks intrinsik K
print(f"[K] Focal length : {focal_length:.1f} pixels")
print(f"[K] Principal pt : ({cx:.1f}, {cy:.1f})")
print(f"[K] Matriks K:")
print(K)

# ============================================================
# 3. Deteksi fitur dan matching
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

# Membuat BFMatcher untuk matching deskriptor SIFT
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Melakukan knnMatch dengan k=2
matches = bf.knnMatch(des1, des2, k=2)

# Menerapkan Lowe's Ratio Test dengan threshold 0.7
good_matches = []
for m, n in matches:
    # Menyaring match berdasarkan rasio jarak
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Menampilkan jumlah match yang lolos filter
print(f"[MATCH] Good matches: {len(good_matches)}")

# Mengekstrak koordinat titik korespondensi
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# ============================================================
# 4. Estimasi Essential Matrix
# ============================================================

# Menampilkan informasi tahap estimasi Essential Matrix
print("\n--- Estimasi Essential Matrix ---")

# Mengestimasi Essential Matrix menggunakan RANSAC dan matriks K
E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Mengambil mask inlier sebagai array 1D
inlier_mask = mask_e.ravel().astype(bool)

# Menghitung jumlah inlier
num_inliers = np.sum(inlier_mask)

# Menampilkan Essential Matrix
print(f"[E] Essential Matrix:")
print(E)

# Menampilkan jumlah inlier
print(f"\n[RANSAC] Inlier: {num_inliers}/{len(pts1)}")

# ============================================================
# 5. Verifikasi properti Essential Matrix
# ============================================================

# Menampilkan informasi tahap verifikasi
print("\n--- Verifikasi Properti E ---")

# Menghitung SVD dari Essential Matrix
U, S, Vt = np.linalg.svd(E)

# Menampilkan singular values (dua pertama harus sama, ketiga ≈ 0)
print(f"[PROP] Singular values: [{S[0]:.4f}, {S[1]:.4f}, {S[2]:.6f}]")
print(f"[PROP] S1/S2 ratio: {S[0]/S[1]:.4f} (seharusnya ≈ 1.0)")

# Menghitung determinan E (seharusnya ≈ 0)
det_E = np.linalg.det(E)
print(f"[PROP] det(E) = {det_E:.10f}")

# Menghitung constraint 2*E*E^T*E - trace(E*E^T)*E = 0
EEt = E @ E.T
constraint = 2 * EEt @ E - np.trace(EEt) * E
constraint_norm = np.linalg.norm(constraint)
print(f"[PROP] ||2*E*E^T*E - tr(E*E^T)*E|| = {constraint_norm:.6f}")

# ============================================================
# 6. Recovery pose kamera (R, t) dari Essential Matrix
# ============================================================

# Menampilkan informasi tahap recovery pose
print("\n--- Recovery Pose Kamera ---")

# Merecovery rotasi R dan translasi t dari Essential Matrix
num_valid, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

# Menampilkan jumlah titik valid yang berada di depan kedua kamera
print(f"[POSE] Titik valid di depan kedua kamera: {num_valid}")

# Menampilkan matriks rotasi R
print(f"\n[POSE] Rotation Matrix R:")
print(R)

# Menampilkan vektor translasi t (unit vector, skala tidak diketahui)
print(f"\n[POSE] Translation Vector t:")
print(t.flatten())

# ============================================================
# 7. Mengekstrak sudut rotasi dari R
# ============================================================

# Menampilkan informasi tahap ekstraksi sudut
print("\n--- Sudut Rotasi ---")

# Mengkonversi matriks rotasi R ke rotation vector menggunakan Rodrigues
rvec, _ = cv2.Rodrigues(R)

# Menghitung besaran rotasi dalam radian
rotation_angle_rad = np.linalg.norm(rvec)

# Mengkonversi ke derajat
rotation_angle_deg = np.degrees(rotation_angle_rad)

# Menampilkan rotation vector
print(f"[ROT] Rotation vector: {rvec.flatten()}")
print(f"[ROT] Besaran rotasi : {rotation_angle_deg:.2f} derajat")

# Menghitung sudut Euler dari matriks rotasi R
def rotation_matrix_to_euler(R):
    """Mengkonversi matriks rotasi ke sudut Euler (dalam derajat)."""
    # Menghitung sudut Y (pitch)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    # Memeriksa apakah kondisi singular
    singular = sy < 1e-6
    if not singular:
        # Menghitung sudut X (roll)
        x = np.arctan2(R[2, 1], R[2, 2])
        # Menghitung sudut Y (pitch)
        y = np.arctan2(-R[2, 0], sy)
        # Menghitung sudut Z (yaw)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Menghitung sudut untuk kondisi gimbal lock
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    # Mengembalikan sudut dalam derajat
    return np.degrees(np.array([x, y, z]))

# Menghitung sudut Euler
euler_angles = rotation_matrix_to_euler(R)

# Menampilkan sudut Euler
print(f"[EULER] Roll (X) : {euler_angles[0]:.2f}°")
print(f"[EULER] Pitch (Y): {euler_angles[1]:.2f}°")
print(f"[EULER] Yaw (Z)  : {euler_angles[2]:.2f}°")

# Menampilkan arah translasi
print(f"\n[TRANS] Arah translasi (unit vector):")
print(f"  X: {t[0, 0]:.4f} ({'kanan' if t[0, 0] > 0 else 'kiri'})")
print(f"  Y: {t[1, 0]:.4f} ({'bawah' if t[1, 0] > 0 else 'atas'})")
print(f"  Z: {t[2, 0]:.4f} ({'depan' if t[2, 0] > 0 else 'belakang'})")

# ============================================================
# 8. Visualisasi
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi ---")

# Mengkonversi gambar ke RGB untuk matplotlib
img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

# Membuat figure dengan 4 subplot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Menampilkan gambar kiri dengan keypoint pada subplot pertama
img_kp_left = cv2.drawKeypoints(img_left_rgb, kp1, None, color=(0, 255, 0))
axes[0, 0].imshow(img_kp_left)
axes[0, 0].set_title(f"Gambar Kiri ({len(kp1)} keypoints)", fontsize=12)
axes[0, 0].axis("off")

# Menampilkan inlier matches pada subplot kedua
inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
top_matches = sorted(inlier_matches, key=lambda x: x.distance)[:30]
img_inlier = cv2.drawMatches(img_left_rgb, kp1, img_right_rgb, kp2,
                              top_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
axes[0, 1].imshow(img_inlier)
axes[0, 1].set_title(f"Inlier Matches ({num_inliers} inliers)", fontsize=12)
axes[0, 1].axis("off")

# Menampilkan visualisasi pose kamera pada subplot ketiga
ax3d = fig.add_subplot(2, 2, 3, projection='3d')
# Menggambar posisi dan orientasi kamera 1 (origin)
origin1 = np.array([0, 0, 0])
# Menggambar sumbu kamera 1
ax3d.quiver(*origin1, 1, 0, 0, color='r', arrow_length_ratio=0.1, label='X')
ax3d.quiver(*origin1, 0, 1, 0, color='g', arrow_length_ratio=0.1, label='Y')
ax3d.quiver(*origin1, 0, 0, 1, color='b', arrow_length_ratio=0.1, label='Z')
# Menggambar posisi kamera 2
origin2 = t.flatten() * 2
# Menggambar sumbu kamera 2 yang dirotasi
scale = 0.8
for i, (color, label) in enumerate(zip(['r', 'g', 'b'], ['X2', 'Y2', 'Z2'])):
    direction = R[:, i] * scale
    ax3d.quiver(*origin2, *direction, color=color, linestyle='dashed', arrow_length_ratio=0.1)
# Menggambar garis baseline antara dua kamera
ax3d.plot([origin1[0], origin2[0]], [origin1[1], origin2[1]], [origin1[2], origin2[2]],
          'k--', linewidth=2, label='Baseline')
# Menambahkan label kamera
ax3d.text(*origin1, "  Cam 1", fontsize=10)
ax3d.text(*origin2, "  Cam 2", fontsize=10)
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.set_title("Pose Kamera Relatif", fontsize=12)
ax3d.legend(fontsize=8)

# Menampilkan informasi ringkasan pada subplot keempat
axes[1, 1].axis("off")
info_text = (
    "ESSENTIAL MATRIX & POSE\n"
    "=" * 40 + "\n\n"
    f"Inlier: {num_inliers}/{len(pts1)}\n\n"
    f"Rotation (Euler):\n"
    f"  Roll  = {euler_angles[0]:.2f}°\n"
    f"  Pitch = {euler_angles[1]:.2f}°\n"
    f"  Yaw   = {euler_angles[2]:.2f}°\n\n"
    f"Translation direction:\n"
    f"  t = [{t[0,0]:.4f}, {t[1,0]:.4f}, {t[2,0]:.4f}]\n\n"
    f"Singular values E:\n"
    f"  [{S[0]:.4f}, {S[1]:.4f}, {S[2]:.6f}]"
)
# Menampilkan teks ringkasan
axes[1, 1].text(0.05, 0.5, info_text, fontsize=11, fontfamily='monospace',
                verticalalignment='center', transform=axes[1, 1].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
axes[1, 1].set_title("Informasi Pose Kamera", fontsize=12)

# Mengatur layout dan judul utama
plt.suptitle("Percobaan 3: Essential Matrix dan Recovery Pose", fontsize=16, fontweight='bold')
plt.tight_layout()

# Menyimpan figure ke file output
output_path = os.path.join(OUTPUT_DIR, "03_essential_matrix_pose.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[SAVED] Hasil disimpan di: {output_path}")

# Menampilkan figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 3: ESSENTIAL MATRIX DAN POSE KAMERA")
print("=" * 60)
print(f"1. Essential Matrix E menghubungkan titik ternormalisasi pada 2 view")
print(f"2. E = K'^T * F * K, membutuhkan kalibrasi kamera (matriks K)")
print(f"3. Focal length: {focal_length:.1f} px, principal point: ({cx:.1f}, {cy:.1f})")
print(f"4. RANSAC menghasilkan {num_inliers} inlier")
print(f"5. Singular values E: [{S[0]:.4f}, {S[1]:.4f}, {S[2]:.6f}]")
print(f"6. Rotasi: Roll={euler_angles[0]:.2f}°, Pitch={euler_angles[1]:.2f}°, Yaw={euler_angles[2]:.2f}°")
print(f"7. Translasi (arah): [{t[0,0]:.4f}, {t[1,0]:.4f}, {t[2,0]:.4f}]")
print(f"8. Translasi dari recoverPose() adalah unit vector (skala tidak diketahui)")
print("=" * 60)
