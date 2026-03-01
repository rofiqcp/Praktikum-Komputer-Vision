"""
==========================================================================
PERCOBAAN 6: STEREO CALIBRATION
==========================================================================
Program ini mempelajari cara mengkalibrasi pasangan kamera stereo
menggunakan pola checkerboard. Stereo calibration menentukan parameter
intrinsik dan ekstrinsik kedua kamera untuk estimasi kedalaman (depth).

Konsep utama:
- Kalibrasi kamera menentukan matriks intrinsik K dan koefisien distorsi
- Stereo calibration menentukan relasi geometris (R, T) antar dua kamera
- Baseline adalah jarak antara dua kamera (norma dari vektor translasi T)
- Reprojection error mengukur akurasi kalibrasi (semakin kecil semakin baik)

Fungsi utama yang dipelajari:
- cv2.findChessboardCorners()    : Deteksi sudut checkerboard
- cv2.cornerSubPix()             : Refine posisi sudut ke sub-pixel
- cv2.calibrateCamera()          : Kalibrasi kamera individual
- cv2.stereoCalibrate()          : Kalibrasi pasangan kamera stereo
- cv2.drawChessboardCorners()    : Visualisasi sudut yang terdeteksi

Hasil: Parameter kalibrasi stereo (K1, K2, D1, D2, R, T) disimpan ke NPZ
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
print("PERCOBAAN 6: STEREO CALIBRATION")
print("=" * 60)

# ============================================================
# 1. Mendefinisikan parameter checkerboard
# ============================================================

# Menampilkan informasi tahap setup parameter
print("\n--- Setup Parameter Checkerboard ---")

# Mendefinisikan jumlah sudut internal checkerboard (kolom x baris)
CHECKERBOARD = (5, 8)

# Mendefinisikan ukuran kotak checkerboard dalam satuan milimeter
SQUARE_SIZE = 50.0

# Mendefinisikan kriteria terminasi untuk corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Menampilkan informasi parameter
print(f"[INFO] Pola checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} sudut internal")
print(f"[INFO] Ukuran kotak: {SQUARE_SIZE} mm")

# ============================================================
# 2. Membuat titik objek 3D (world coordinates)
# ============================================================

# Menampilkan informasi tahap pembuatan objek points
print("\n--- Mempersiapkan Object Points ---")

# Membuat array koordinat 3D untuk sudut checkerboard
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# Mengisi koordinat X, Y berdasarkan posisi grid (Z=0 karena planar)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Mengalikan dengan ukuran kotak agar sesuai skala dunia nyata
objp *= SQUARE_SIZE

# Menampilkan informasi objek points
print(f"[INFO] Jumlah titik per gambar: {len(objp)}")
print(f"[INFO] Rentang X: 0 - {objp[:, 0].max():.0f} mm")
print(f"[INFO] Rentang Y: 0 - {objp[:, 1].max():.0f} mm")

# ============================================================
# 3. Generate pasangan gambar checkerboard sintetis
# ============================================================

# Menampilkan informasi tahap generate gambar
print("\n--- Generate Gambar Checkerboard Stereo ---")

def buat_checkerboard_stereo(rows=9, cols=6, square_size=50, baseline=40):
    """Membuat pasangan gambar checkerboard dari dua sudut pandang kamera."""

    # Menghitung dimensi gambar berdasarkan ukuran checkerboard
    height = rows * square_size + 100
    width = cols * square_size + 100

    # Membuat gambar checkerboard dasar
    img = np.ones((height, width, 3), dtype=np.uint8) * 200

    # Mendefinisikan offset agar checkerboard di tengah
    offset_y, offset_x = 50, 50

    # Menggambar pola kotak hitam-putih
    for r in range(rows):
        for c in range(cols):
            # Menghitung posisi kotak
            x = offset_x + c * square_size
            y = offset_y + r * square_size
            # Menentukan warna berdasarkan posisi ganjil/genap
            if (r + c) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + square_size, y + square_size), (255, 255, 255), -1)
            else:
                cv2.rectangle(img, (x, y), (x + square_size, y + square_size), (0, 0, 0), -1)

    # Membuat perspektif transform untuk kamera kiri (sedikit rotasi)
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Menentukan titik tujuan untuk kamera kiri (perspektif ringan)
    dst_left = np.float32([
        [10, 5], [w - 5, 10], [w - 15, h - 10], [5, h - 5]
    ])

    # Menentukan titik tujuan untuk kamera kanan (geser horizontal + perspektif)
    dst_right = np.float32([
        [15 + baseline // 4, 8], [w - 10, 5],
        [w - 5, h - 8], [10 + baseline // 4, h - 12]
    ])

    # Menghitung matriks perspektif untuk kamera kiri
    M_left = cv2.getPerspectiveTransform(src_pts, dst_left)

    # Menghitung matriks perspektif untuk kamera kanan
    M_right = cv2.getPerspectiveTransform(src_pts, dst_right)

    # Mengaplikasikan transformasi perspektif pada kamera kiri
    img_left = cv2.warpPerspective(img, M_left, (w, h), borderValue=(200, 200, 200))

    # Mengaplikasikan transformasi perspektif pada kamera kanan
    img_right = cv2.warpPerspective(img, M_right, (w, h), borderValue=(200, 200, 200))

    return img_left, img_right

# Menyiapkan list untuk menyimpan pasangan gambar
stereo_pairs = []

# Mendefinisikan variasi sudut untuk beberapa pasangan
variasi_params = [
    (9, 6, 50, 30),
    (9, 6, 50, 40),
    (9, 6, 50, 50),
]

# Menggenerate beberapa pasangan gambar stereo checkerboard
for idx, (rows, cols, sq, bl) in enumerate(variasi_params):
    # Membuat pasangan gambar checkerboard stereo
    img_l, img_r = buat_checkerboard_stereo(rows, cols, sq, bl)
    # Menyimpan ke list
    stereo_pairs.append((img_l, img_r))
    # Menampilkan informasi
    print(f"[OK] Pasangan {idx + 1}: rows={rows}, cols={cols}, baseline~{bl}")

# ============================================================
# 4. Deteksi sudut checkerboard pada semua pasangan
# ============================================================

# Menampilkan informasi tahap deteksi sudut
print("\n--- Deteksi Sudut Checkerboard ---")

# Menyiapkan list untuk menyimpan object points dan image points
obj_points = []
img_points_left = []
img_points_right = []

# Menyiapkan list untuk menyimpan gambar dengan sudut yang terdeteksi
images_with_corners = []

# Memproses setiap pasangan gambar
for idx, (img_l, img_r) in enumerate(stereo_pairs):
    # Mengkonversi gambar kiri ke grayscale
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

    # Mengkonversi gambar kanan ke grayscale
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Mendeteksi sudut checkerboard pada gambar kiri
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)

    # Mendeteksi sudut checkerboard pada gambar kanan
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

    # Memeriksa apakah sudut berhasil ditemukan pada kedua gambar
    if ret_l and ret_r:
        # Memperhalus posisi sudut ke akurasi sub-pixel pada gambar kiri
        corners_l_refined = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)

        # Memperhalus posisi sudut ke akurasi sub-pixel pada gambar kanan
        corners_r_refined = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        # Menambahkan object points (sama untuk setiap pasangan)
        obj_points.append(objp)

        # Menambahkan image points kiri
        img_points_left.append(corners_l_refined)

        # Menambahkan image points kanan
        img_points_right.append(corners_r_refined)

        # Menggambar sudut yang terdeteksi pada salinan gambar kiri
        vis_l = img_l.copy()
        cv2.drawChessboardCorners(vis_l, CHECKERBOARD, corners_l_refined, ret_l)

        # Menggambar sudut yang terdeteksi pada salinan gambar kanan
        vis_r = img_r.copy()
        cv2.drawChessboardCorners(vis_r, CHECKERBOARD, corners_r_refined, ret_r)

        # Menyimpan gambar visualisasi
        images_with_corners.append((vis_l, vis_r))

        # Menampilkan informasi berhasil
        print(f"[OK] Pasangan {idx + 1}: Sudut ditemukan (L: {len(corners_l_refined)}, R: {len(corners_r_refined)})")
    else:
        # Menampilkan informasi gagal
        print(f"[SKIP] Pasangan {idx + 1}: Sudut TIDAK ditemukan (L: {ret_l}, R: {ret_r})")

# Memeriksa apakah ada cukup data untuk kalibrasi
if len(obj_points) < 1:
    print("[ERROR] Tidak ada pasangan yang valid! Tidak dapat melakukan kalibrasi.")
    exit()

# Menampilkan jumlah pasangan valid
print(f"\n[INFO] Total pasangan valid: {len(obj_points)}")

# Mendapatkan ukuran gambar
img_size = (stereo_pairs[0][0].shape[1], stereo_pairs[0][0].shape[0])
print(f"[INFO] Ukuran gambar: {img_size}")

# ============================================================
# 5. Kalibrasi kamera individual
# ============================================================

# Menampilkan informasi tahap kalibrasi individual
print("\n--- Kalibrasi Kamera Individual ---")

# Melakukan kalibrasi kamera kiri
ret_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
    obj_points, img_points_left, img_size, None, None
)

# Menampilkan error reproyeksi kamera kiri
print(f"[CAM_L] Reprojection error: {ret_l:.4f} pixel")

# Melakukan kalibrasi kamera kanan
ret_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(
    obj_points, img_points_right, img_size, None, None
)

# Menampilkan error reproyeksi kamera kanan
print(f"[CAM_R] Reprojection error: {ret_r:.4f} pixel")

# Menampilkan matriks intrinsik kamera kiri
print(f"\n[K1] Matriks Intrinsik Kamera Kiri:")
print(f"  Focal length (fx, fy): ({K1[0, 0]:.2f}, {K1[1, 1]:.2f})")
print(f"  Principal point (cx, cy): ({K1[0, 2]:.2f}, {K1[1, 2]:.2f})")

# Menampilkan matriks intrinsik kamera kanan
print(f"\n[K2] Matriks Intrinsik Kamera Kanan:")
print(f"  Focal length (fx, fy): ({K2[0, 0]:.2f}, {K2[1, 1]:.2f})")
print(f"  Principal point (cx, cy): ({K2[0, 2]:.2f}, {K2[1, 2]:.2f})")

# Menampilkan koefisien distorsi
print(f"\n[D1] Distorsi kamera kiri : {D1.flatten()[:5]}")
print(f"[D2] Distorsi kamera kanan: {D2.flatten()[:5]}")

# ============================================================
# 6. Stereo calibration
# ============================================================

# Menampilkan informasi tahap stereo calibration
print("\n--- Stereo Calibration ---")

# Mendefinisikan flag untuk stereo calibration
stereo_flags = cv2.CALIB_FIX_INTRINSIC

# Melakukan stereo calibration menggunakan hasil kalibrasi individual
ret_stereo, K1_s, D1_s, K2_s, D2_s, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left, img_points_right,
    K1, D1, K2, D2, img_size,
    criteria=criteria,
    flags=stereo_flags
)

# Menampilkan error reproyeksi stereo
print(f"[STEREO] Reprojection error: {ret_stereo:.4f} pixel")

# Menampilkan matriks rotasi R antar kamera
print(f"\n[R] Matriks Rotasi antar kamera:")
print(R)

# Menampilkan vektor translasi T antar kamera
print(f"\n[T] Vektor Translasi antar kamera:")
print(T.flatten())

# Menghitung baseline (jarak antar kamera)
baseline = np.linalg.norm(T)
print(f"\n[BASELINE] Jarak antar kamera: {baseline:.2f} mm")

# Menampilkan Essential Matrix
print(f"\n[E] Essential Matrix:")
print(E)

# Menampilkan Fundamental Matrix
print(f"\n[F] Fundamental Matrix:")
print(F)

# ============================================================
# 7. Menghitung reprojection error detail
# ============================================================

# Menampilkan informasi tahap analisis error
print("\n--- Analisis Reprojection Error ---")

# Menghitung error per pasangan gambar
errors_left = []
errors_right = []

# Iterasi setiap pasangan untuk menghitung error
for i in range(len(obj_points)):
    # Memproyeksikan ulang titik 3D ke kamera kiri
    imgpoints_reproj_l, _ = cv2.projectPoints(obj_points[i], rvecs_l[i], tvecs_l[i], K1, D1)

    # Menghitung error rata-rata untuk kamera kiri
    error_l = cv2.norm(img_points_left[i], imgpoints_reproj_l, cv2.NORM_L2) / len(imgpoints_reproj_l)
    errors_left.append(error_l)

    # Memproyeksikan ulang titik 3D ke kamera kanan
    imgpoints_reproj_r, _ = cv2.projectPoints(obj_points[i], rvecs_r[i], tvecs_r[i], K2, D2)

    # Menghitung error rata-rata untuk kamera kanan
    error_r = cv2.norm(img_points_right[i], imgpoints_reproj_r, cv2.NORM_L2) / len(imgpoints_reproj_r)
    errors_right.append(error_r)

    # Menampilkan error per pasangan
    print(f"  Pasangan {i + 1}: Error kiri={error_l:.4f} px, Error kanan={error_r:.4f} px")

# Menghitung error rata-rata keseluruhan
mean_error_l = np.mean(errors_left)
mean_error_r = np.mean(errors_right)
print(f"\n[MEAN] Error rata-rata kiri : {mean_error_l:.4f} pixel")
print(f"[MEAN] Error rata-rata kanan: {mean_error_r:.4f} pixel")

# ============================================================
# 8. Menyimpan parameter kalibrasi ke file NPZ
# ============================================================

# Menampilkan informasi tahap penyimpanan
print("\n--- Menyimpan Parameter Kalibrasi ---")

# Mendefinisikan path file output NPZ
calib_path = os.path.join(OUTPUT_DIR, "06_stereo_calibration.npz")

# Menyimpan semua parameter kalibrasi ke file NPZ
np.savez(calib_path,
         K1=K1, D1=D1, K2=K2, D2=D2,
         R=R, T=T, E=E, F=F,
         img_size=np.array(img_size),
         reprojection_error=ret_stereo)

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] Parameter kalibrasi disimpan di: {calib_path}")

# ============================================================
# 9. Visualisasi hasil
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi ---")

# Menentukan jumlah baris berdasarkan jumlah pasangan valid
n_pairs = len(images_with_corners)

# Membuat figure untuk visualisasi deteksi sudut
fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 5 * n_pairs))

# Menangani kasus satu pasangan (axes tidak 2D)
if n_pairs == 1:
    axes = axes.reshape(1, -1)

# Menampilkan setiap pasangan gambar dengan sudut terdeteksi
for i, (vis_l, vis_r) in enumerate(images_with_corners):
    # Mengkonversi gambar kiri dari BGR ke RGB
    vis_l_rgb = cv2.cvtColor(vis_l, cv2.COLOR_BGR2RGB)
    # Mengkonversi gambar kanan dari BGR ke RGB
    vis_r_rgb = cv2.cvtColor(vis_r, cv2.COLOR_BGR2RGB)

    # Menampilkan gambar kiri dengan sudut pada kolom pertama
    axes[i, 0].imshow(vis_l_rgb)
    axes[i, 0].set_title(f"Kamera Kiri - Pasangan {i + 1}", fontsize=11)
    axes[i, 0].axis("off")

    # Menampilkan gambar kanan dengan sudut pada kolom kedua
    axes[i, 1].imshow(vis_r_rgb)
    axes[i, 1].set_title(f"Kamera Kanan - Pasangan {i + 1}", fontsize=11)
    axes[i, 1].axis("off")

# Mengatur judul utama figure
plt.suptitle("Percobaan 6: Deteksi Sudut Checkerboard Stereo", fontsize=14, fontweight='bold')

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan figure visualisasi sudut ke file
output_corners = os.path.join(OUTPUT_DIR, "06_stereo_calibration_corners.png")
plt.savefig(output_corners, dpi=150, bbox_inches='tight')
print(f"[SAVED] Visualisasi sudut: {output_corners}")

# Membuat figure kedua untuk visualisasi error
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Membuat bar chart error kamera kiri
x_labels = [f"Pair {i + 1}" for i in range(len(errors_left))]
x_pos = np.arange(len(errors_left))

# Menampilkan bar chart error kamera kiri
axes2[0].bar(x_pos, errors_left, color='steelblue', alpha=0.8)
axes2[0].axhline(y=mean_error_l, color='red', linestyle='--', label=f'Mean: {mean_error_l:.4f}')
axes2[0].set_xlabel("Pasangan Gambar")
axes2[0].set_ylabel("Reprojection Error (pixel)")
axes2[0].set_title("Error Kamera Kiri")
axes2[0].set_xticks(x_pos)
axes2[0].set_xticklabels(x_labels)
axes2[0].legend()

# Menampilkan bar chart error kamera kanan
axes2[1].bar(x_pos, errors_right, color='coral', alpha=0.8)
axes2[1].axhline(y=mean_error_r, color='red', linestyle='--', label=f'Mean: {mean_error_r:.4f}')
axes2[1].set_xlabel("Pasangan Gambar")
axes2[1].set_ylabel("Reprojection Error (pixel)")
axes2[1].set_title("Error Kamera Kanan")
axes2[1].set_xticks(x_pos)
axes2[1].set_xticklabels(x_labels)
axes2[1].legend()

# Mengatur judul utama figure error
plt.suptitle("Percobaan 6: Perbandingan Reprojection Error", fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure error ke file
output_error = os.path.join(OUTPUT_DIR, "06_stereo_calibration_error.png")
plt.savefig(output_error, dpi=150, bbox_inches='tight')
print(f"[SAVED] Visualisasi error: {output_error}")

# Menampilkan figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 6: STEREO CALIBRATION")
print("=" * 60)
print(f"1. Stereo calibration menentukan hubungan geometris dua kamera")
print(f"2. Deteksi sudut checkerboard dengan cv2.findChessboardCorners()")
print(f"3. Sub-pixel refinement dengan cv2.cornerSubPix()")
print(f"4. Kalibrasi individual: cv2.calibrateCamera()")
print(f"5. Kalibrasi stereo: cv2.stereoCalibrate()")
print(f"6. Reprojection error kiri : {mean_error_l:.4f} pixel")
print(f"7. Reprojection error kanan: {mean_error_r:.4f} pixel")
print(f"8. Reprojection error stereo: {ret_stereo:.4f} pixel")
print(f"9. Baseline (jarak kamera): {baseline:.2f} mm")
print(f"10. Parameter disimpan ke: {calib_path}")
print("=" * 60)
