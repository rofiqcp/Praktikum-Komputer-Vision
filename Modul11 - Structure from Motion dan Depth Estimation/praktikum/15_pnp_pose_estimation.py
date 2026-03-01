"""
==========================================================================
PERCOBAAN 15: POSE ESTIMATION DENGAN PNP (PERSPECTIVE-N-POINT)
==========================================================================
Program ini mempelajari cara mengestimasi pose kamera (posisi dan
orientasi) dari korespondensi titik 3D-2D menggunakan algoritma PnP
(Perspective-N-Point). Beberapa metode PnP dibandingkan untuk melihat
akurasi dan robustness masing-masing.

Konsep utama:
- PnP mengestimasi pose kamera dari minimal 4 pasangan titik 3D-2D
- Pose kamera = rotation vector (rvec) + translation vector (tvec)
- cv2.Rodrigues() mengkonversi antara rotation vector dan rotation matrix
- Reprojection error mengukur seberapa baik pose yang diestimasi
- RANSAC-based PnP lebih robust terhadap outlier pada data noisy
- Berbagai metode: ITERATIVE, P3P, EPNP, SQPNP memiliki trade-off

Fungsi utama yang dipelajari:
- cv2.solvePnP()              : Estimasi pose kamera dari 3D-2D
- cv2.solvePnPRansac()        : PnP dengan RANSAC (robust)
- cv2.Rodrigues()             : Konversi rvec <-> rotation matrix
- cv2.projectPoints()         : Proyeksi titik 3D ke gambar 2D
- cv2.drawFrameAxes()         : Menggambar sumbu 3D pada gambar

Hasil: Perbandingan pose estimation dari berbagai metode PnP, visualisasi
       reprojection error, dan sumbu 3D yang diproyeksikan pada gambar
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
print("PERCOBAAN 15: POSE ESTIMATION DENGAN PNP")
print("=" * 60)

# ============================================================
# 1. Mendefinisikan Titik 3D Objek (Kubus)
# ============================================================

# Menampilkan informasi tahap definisi titik 3D
print("\n--- Mendefinisikan Titik 3D Objek ---")

# Mendefinisikan ukuran kubus dalam satuan dunia (mm)
cube_size = 50.0

# Mendefinisikan 8 titik sudut kubus dalam koordinat dunia 3D
object_points = np.array([
    [0, 0, 0],
    [cube_size, 0, 0],
    [cube_size, cube_size, 0],
    [0, cube_size, 0],
    [0, 0, cube_size],
    [cube_size, 0, cube_size],
    [cube_size, cube_size, cube_size],
    [0, cube_size, cube_size],
], dtype=np.float64)

# Menambahkan titik tambahan pada permukaan kubus untuk akurasi lebih baik
additional_points = np.array([
    [cube_size / 2, 0, 0],
    [cube_size, cube_size / 2, 0],
    [cube_size / 2, cube_size, 0],
    [0, cube_size / 2, 0],
    [cube_size / 2, cube_size / 2, 0],
    [cube_size / 2, cube_size / 2, cube_size],
], dtype=np.float64)

# Menggabungkan semua titik 3D
object_points = np.vstack([object_points, additional_points])

# Menampilkan informasi titik 3D
print(f"[INFO] Jumlah titik 3D: {len(object_points)}")
print(f"[INFO] Ukuran kubus: {cube_size:.0f} mm")
print(f"[INFO] Koordinat X range: [{object_points[:,0].min():.0f}, {object_points[:,0].max():.0f}]")
print(f"[INFO] Koordinat Y range: [{object_points[:,1].min():.0f}, {object_points[:,1].max():.0f}]")
print(f"[INFO] Koordinat Z range: [{object_points[:,2].min():.0f}, {object_points[:,2].max():.0f}]")

# ============================================================
# 2. Mendefinisikan Parameter Kamera dan Pose Asli
# ============================================================

# Menampilkan informasi tahap parameter kamera
print("\n--- Parameter Kamera dan Pose Asli ---")

# Mendefinisikan ukuran gambar
img_w, img_h = 640, 480

# Mendefinisikan focal length (dalam piksel)
fx = 800.0
fy = 800.0

# Mendefinisikan principal point (pusat gambar)
cx_cam = img_w / 2.0
cy_cam = img_h / 2.0

# Membuat matriks intrinsik kamera (3x3)
camera_matrix = np.array([
    [fx, 0, cx_cam],
    [0, fy, cy_cam],
    [0, 0, 1]
], dtype=np.float64)

# Mendefinisikan koefisien distorsi (tanpa distorsi untuk simulasi)
dist_coeffs = np.zeros((4, 1), dtype=np.float64)

# Menampilkan matriks intrinsik
print("[INFO] Matriks Intrinsik Kamera:")
for row in camera_matrix:
    print(f"  [{row[0]:8.1f} {row[1]:8.1f} {row[2]:8.1f}]")

# Mendefinisikan pose kamera asli (ground truth)
# Rotasi: 30 derajat sekitar sumbu X, 20 derajat sekitar sumbu Y
angle_x = np.radians(30)
angle_y = np.radians(20)
angle_z = np.radians(10)

# Membuat rotation matrix dari sudut Euler
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(angle_x), -np.sin(angle_x)],
    [0, np.sin(angle_x), np.cos(angle_x)]
])

# Membuat rotation matrix untuk sumbu Y
Ry = np.array([
    [np.cos(angle_y), 0, np.sin(angle_y)],
    [0, 1, 0],
    [-np.sin(angle_y), 0, np.cos(angle_y)]
])

# Membuat rotation matrix untuk sumbu Z
Rz = np.array([
    [np.cos(angle_z), -np.sin(angle_z), 0],
    [np.sin(angle_z), np.cos(angle_z), 0],
    [0, 0, 1]
])

# Menggabungkan rotasi (R = Rz * Ry * Rx)
R_true = Rz @ Ry @ Rx

# Mengkonversi rotation matrix ke rotation vector menggunakan Rodrigues
rvec_true, _ = cv2.Rodrigues(R_true)

# Mendefinisikan translation vector asli (posisi kubus relatif terhadap kamera)
tvec_true = np.array([[10.0], [5.0], [200.0]], dtype=np.float64)

# Menampilkan pose asli
print(f"\n[GROUND TRUTH] Rotation vector: [{rvec_true[0,0]:.4f}, {rvec_true[1,0]:.4f}, {rvec_true[2,0]:.4f}]")
print(f"[GROUND TRUTH] Translation: [{tvec_true[0,0]:.1f}, {tvec_true[1,0]:.1f}, {tvec_true[2,0]:.1f}]")

# Menghitung sudut rotasi total dalam derajat
angle_total = np.linalg.norm(rvec_true) * 180 / np.pi
print(f"[GROUND TRUTH] Total rotation: {angle_total:.2f} derajat")

# ============================================================
# 3. Memproyeksikan Titik 3D ke 2D (Simulasi Pengamatan)
# ============================================================

# Menampilkan informasi tahap proyeksi
print("\n--- Memproyeksikan Titik 3D ke 2D ---")

# Memproyeksikan titik 3D ke bidang gambar 2D menggunakan pose asli
image_points_perfect, _ = cv2.projectPoints(
    object_points, rvec_true, tvec_true, camera_matrix, dist_coeffs
)

# Mengubah bentuk array menjadi Nx2
image_points_perfect = image_points_perfect.reshape(-1, 2)

# Menampilkan informasi proyeksi
print(f"[INFO] Titik 2D terproyeksi: {len(image_points_perfect)}")
print(f"[INFO] X range: [{image_points_perfect[:,0].min():.1f}, {image_points_perfect[:,0].max():.1f}]")
print(f"[INFO] Y range: [{image_points_perfect[:,1].min():.1f}, {image_points_perfect[:,1].max():.1f}]")

# Menambahkan noise Gaussian ke titik 2D (simulasi deteksi tidak sempurna)
noise_sigma = 1.5

# Menghasilkan noise acak
noise_2d = np.random.normal(0, noise_sigma, image_points_perfect.shape)

# Menambahkan noise ke titik 2D
image_points_noisy = image_points_perfect + noise_2d

# Menampilkan informasi noise
print(f"[NOISE] Sigma: {noise_sigma} pixel")
print(f"[NOISE] Max displacement: {np.max(np.abs(noise_2d)):.2f} pixel")

# ============================================================
# 4. Estimasi Pose dengan Berbagai Metode PnP
# ============================================================

# Menampilkan informasi tahap PnP
print("\n--- Estimasi Pose dengan Berbagai Metode PnP ---")

# Mendefinisikan metode PnP yang akan dicoba
pnp_methods = [
    ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
    ("P3P", cv2.SOLVEPNP_P3P),
    ("EPNP", cv2.SOLVEPNP_EPNP),
    ("SQPNP", cv2.SOLVEPNP_SQPNP),
]

# Menginisialisasi dictionary untuk menyimpan hasil
results = {}

# Menampilkan header tabel hasil
print(f"\n{'Metode':<12} {'Status':<10} {'Rot Error(°)':<14} {'Trans Error':<12} {'Reproj Error':<12}")
print("-" * 60)

# Mencoba setiap metode PnP
for method_name, method_flag in pnp_methods:
    try:
        # Menyesuaikan format titik untuk P3P (memerlukan tepat 4 titik)
        if method_name == "P3P":
            # P3P memerlukan tepat 4 titik
            obj_4 = object_points[:4].reshape(-1, 1, 3)
            img_4 = image_points_noisy[:4].reshape(-1, 1, 2)
            # Menjalankan solvePnP dengan metode P3P
            success, rvec_est, tvec_est = cv2.solvePnP(
                obj_4, img_4, camera_matrix, dist_coeffs, flags=method_flag
            )
        else:
            # Menjalankan solvePnP dengan semua titik
            success, rvec_est, tvec_est = cv2.solvePnP(
                object_points, image_points_noisy, camera_matrix, dist_coeffs,
                flags=method_flag
            )

        if success:
            # Menghitung error rotasi (selisih sudut dalam derajat)
            rot_error = np.linalg.norm(rvec_est - rvec_true) * 180 / np.pi

            # Menghitung error translasi (jarak Euclidean)
            trans_error = np.linalg.norm(tvec_est - tvec_true)

            # Menghitung reprojection error
            reproj_pts, _ = cv2.projectPoints(
                object_points, rvec_est, tvec_est, camera_matrix, dist_coeffs
            )
            reproj_pts = reproj_pts.reshape(-1, 2)
            reproj_error = np.mean(np.linalg.norm(reproj_pts - image_points_perfect, axis=1))

            # Menyimpan hasil
            results[method_name] = {
                "rvec": rvec_est, "tvec": tvec_est,
                "rot_error": rot_error, "trans_error": trans_error,
                "reproj_error": reproj_error, "reproj_pts": reproj_pts
            }

            # Menampilkan hasil metode ini
            print(f"{method_name:<12} {'OK':<10} {rot_error:<14.4f} {trans_error:<12.4f} {reproj_error:<12.4f}")
        else:
            # Menampilkan status gagal
            print(f"{method_name:<12} {'FAIL':<10} {'N/A':<14} {'N/A':<12} {'N/A':<12}")

    except Exception as e:
        # Menampilkan error jika metode gagal
        print(f"{method_name:<12} {'ERROR':<10} {str(e)[:30]}")

# ============================================================
# 5. Estimasi Pose dengan RANSAC
# ============================================================

# Menampilkan informasi tahap RANSAC
print("\n--- PnP dengan RANSAC ---")

# Menambahkan outlier ke titik 2D untuk testing RANSAC
image_points_outlier = image_points_noisy.copy()

# Memilih beberapa indeks untuk dijadikan outlier
n_outliers = 3
outlier_indices = np.random.choice(len(image_points_outlier), n_outliers, replace=False)

# Menggeser titik outlier secara signifikan
for idx in outlier_indices:
    image_points_outlier[idx] += np.random.uniform(20, 50, 2)

# Menampilkan informasi outlier
print(f"[INFO] Jumlah outlier ditambahkan: {n_outliers}")
print(f"[INFO] Indeks outlier: {outlier_indices}")

# Menjalankan solvePnPRansac
success_ransac, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(
    object_points, image_points_outlier, camera_matrix, dist_coeffs,
    reprojectionError=5.0, iterationsCount=200
)

if success_ransac:
    # Menghitung error rotasi RANSAC
    rot_err_ransac = np.linalg.norm(rvec_ransac - rvec_true) * 180 / np.pi

    # Menghitung error translasi RANSAC
    trans_err_ransac = np.linalg.norm(tvec_ransac - tvec_true)

    # Menampilkan hasil RANSAC
    print(f"[RANSAC] Inliers: {len(inliers)}/{len(object_points)}")
    print(f"[RANSAC] Rotation error: {rot_err_ransac:.4f} derajat")
    print(f"[RANSAC] Translation error: {trans_err_ransac:.4f}")

    # Menyimpan hasil RANSAC
    results["RANSAC"] = {
        "rvec": rvec_ransac, "tvec": tvec_ransac,
        "rot_error": rot_err_ransac, "trans_error": trans_err_ransac,
        "reproj_error": 0, "reproj_pts": None
    }

# ============================================================
# 6. Visualisasi Sumbu 3D pada Gambar
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Visualisasi Sumbu 3D ---")

# Membuat gambar latar belakang abu-abu
img_canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 200

# Menambahkan pola grid pada latar belakang
for y in range(0, img_h, 40):
    cv2.line(img_canvas, (0, y), (img_w, y), (180, 180, 180), 1)
for x in range(0, img_w, 40):
    cv2.line(img_canvas, (x, 0), (x, img_h), (180, 180, 180), 1)

# Menggambar titik 2D noisy pada gambar
for pt in image_points_noisy:
    # Menggambar lingkaran kecil untuk setiap titik observasi
    cv2.circle(img_canvas, (int(pt[0]), int(pt[1])), 4, (0, 200, 0), -1)

# Menggambar sumbu 3D menggunakan pose ground truth
axis_length = 40.0

# Mendefinisikan titik-titik sumbu 3D (origin + ujung X, Y, Z)
axis_points = np.array([
    [0, 0, 0],
    [axis_length, 0, 0],
    [0, axis_length, 0],
    [0, 0, axis_length]
], dtype=np.float64)

# Memproyeksikan sumbu 3D menggunakan pose ground truth
axis_proj_true, _ = cv2.projectPoints(
    axis_points, rvec_true, tvec_true, camera_matrix, dist_coeffs
)
axis_proj_true = axis_proj_true.reshape(-1, 2).astype(int)

# Menggambar sumbu X (merah), Y (hijau), Z (biru) ground truth
origin = tuple(axis_proj_true[0])
cv2.line(img_canvas, origin, tuple(axis_proj_true[1]), (0, 0, 255), 3)
cv2.line(img_canvas, origin, tuple(axis_proj_true[2]), (0, 255, 0), 3)
cv2.line(img_canvas, origin, tuple(axis_proj_true[3]), (255, 0, 0), 3)

# Menambahkan label sumbu
cv2.putText(img_canvas, "X", tuple(axis_proj_true[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img_canvas, "Y", tuple(axis_proj_true[2] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(img_canvas, "Z", tuple(axis_proj_true[3] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Menambahkan judul pada gambar
cv2.putText(img_canvas, "Ground Truth (garis tebal) vs Estimasi (garis tipis)",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Menggambar sumbu estimasi dari metode ITERATIVE (jika ada)
if "ITERATIVE" in results:
    # Memproyeksikan sumbu menggunakan pose estimasi
    axis_proj_est, _ = cv2.projectPoints(
        axis_points, results["ITERATIVE"]["rvec"],
        results["ITERATIVE"]["tvec"], camera_matrix, dist_coeffs
    )
    axis_proj_est = axis_proj_est.reshape(-1, 2).astype(int)

    # Menggambar sumbu estimasi dengan garis tipis
    origin_est = tuple(axis_proj_est[0])
    cv2.line(img_canvas, origin_est, tuple(axis_proj_est[1]), (0, 0, 200), 1)
    cv2.line(img_canvas, origin_est, tuple(axis_proj_est[2]), (0, 200, 0), 1)
    cv2.line(img_canvas, origin_est, tuple(axis_proj_est[3]), (200, 0, 0), 1)

# ============================================================
# 7. Membuat Visualisasi Perbandingan Lengkap
# ============================================================

# Menampilkan informasi tahap visualisasi perbandingan
print("\n--- Membuat Visualisasi Perbandingan ---")

# Membuat figure perbandingan
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Menampilkan gambar dengan sumbu 3D
axes[0, 0].imshow(cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Proyeksi Sumbu 3D\n(Tebal=GT, Tipis=Estimasi)", fontsize=10)
axes[0, 0].axis('off')

# Membuat bar chart reprojection error
method_list = [m for m in results if results[m]["reproj_error"] > 0]
reproj_errors = [results[m]["reproj_error"] for m in method_list]

# Menentukan warna untuk setiap metode
bar_colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']

# Menggambar bar chart reprojection error
if len(method_list) > 0:
    axes[0, 1].bar(range(len(method_list)), reproj_errors,
                   color=bar_colors[:len(method_list)], edgecolor='black')
    axes[0, 1].set_xticks(range(len(method_list)))
    axes[0, 1].set_xticklabels(method_list, rotation=20, fontsize=9)
    axes[0, 1].set_title("Reprojection Error per Metode", fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel("Error (pixel)")
    # Menambahkan nilai di atas bar
    for i, val in enumerate(reproj_errors):
        axes[0, 1].text(i, val + 0.01, f"{val:.3f}", ha='center', fontsize=8)

# Membuat bar chart rotation error
rot_errors = [results[m]["rot_error"] for m in method_list]

# Menggambar bar chart rotation error
if len(method_list) > 0:
    axes[1, 0].bar(range(len(method_list)), rot_errors,
                   color=bar_colors[:len(method_list)], edgecolor='black')
    axes[1, 0].set_xticks(range(len(method_list)))
    axes[1, 0].set_xticklabels(method_list, rotation=20, fontsize=9)
    axes[1, 0].set_title("Rotation Error per Metode", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Error (derajat)")
    # Menambahkan nilai di atas bar
    for i, val in enumerate(rot_errors):
        axes[1, 0].text(i, val + 0.001, f"{val:.4f}", ha='center', fontsize=8)

# Membuat bar chart translation error
trans_errors = [results[m]["trans_error"] for m in method_list]

# Menggambar bar chart translation error
if len(method_list) > 0:
    axes[1, 1].bar(range(len(method_list)), trans_errors,
                   color=bar_colors[:len(method_list)], edgecolor='black')
    axes[1, 1].set_xticks(range(len(method_list)))
    axes[1, 1].set_xticklabels(method_list, rotation=20, fontsize=9)
    axes[1, 1].set_title("Translation Error per Metode", fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel("Error (mm)")
    # Menambahkan nilai di atas bar
    for i, val in enumerate(trans_errors):
        axes[1, 1].text(i, val + 0.001, f"{val:.4f}", ha='center', fontsize=8)

# Mengatur judul utama figure
plt.suptitle("Perbandingan Metode PnP untuk Pose Estimation", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
path_comparison = os.path.join(OUTPUT_DIR, "15_pnp_perbandingan.png")

# Menyimpan figure perbandingan
fig.savefig(path_comparison, dpi=150, bbox_inches='tight')
print(f"[SAVE] Perbandingan PnP: {path_comparison}")

# Menyimpan gambar sumbu 3D
path_axes = os.path.join(OUTPUT_DIR, "15_pnp_axes_3d.png")
cv2.imwrite(path_axes, img_canvas)
print(f"[SAVE] Sumbu 3D: {path_axes}")

# Menampilkan semua figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 15: POSE ESTIMATION DENGAN PNP")
print("=" * 60)
print("1. PnP mengestimasi pose (R, t) dari korespondensi 3D-2D")
print("2. Minimal 4 titik diperlukan (P3P butuh tepat 4 titik)")
print("3. ITERATIVE: metode dasar, akurat untuk data bersih")
print("4. EPNP: efisien untuk banyak titik, non-iteratif")
print("5. P3P: minimal 4 titik, cepat tapi sensitif terhadap noise")
print("6. SQPNP: metode terbaru, robust dan akurat")
print("7. RANSAC: mengatasi outlier dengan sampling acak")
print("8. Reprojection error mengukur kualitas estimasi pose")
print("9. Noise pada titik 2D menurunkan akurasi semua metode")
print("10. PnP adalah dasar untuk augmented reality dan navigasi")
print("=" * 60)
