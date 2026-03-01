"""
==========================================================================
PERCOBAAN 7: STEREO RECTIFICATION
==========================================================================
Program ini mempelajari cara merektifikasi pasangan gambar stereo
agar garis epipolar menjadi horizontal. Rectification menyederhanakan
proses stereo matching karena pencarian korespondensi hanya dilakukan
pada baris yang sama (scanline).

Konsep utama:
- Rectification memutar kedua gambar sehingga epipolar lines sejajar horizontal
- Setelah rectification, titik korespondensi berada pada baris y yang sama
- cv2.stereoRectify() menghitung matriks rotasi dan proyeksi baru
- cv2.initUndistortRectifyMap() membuat peta remap untuk transformasi
- Valid ROI menunjukkan area gambar yang valid setelah rectification

Fungsi utama yang dipelajari:
- cv2.stereoRectify()              : Menghitung transformasi rectification
- cv2.initUndistortRectifyMap()    : Membuat peta undistort + rectify
- cv2.remap()                      : Mengaplikasikan peta transformasi
- cv2.line()                       : Menggambar garis horizontal verifikasi

Hasil: Pasangan gambar stereo yang sudah direktifikasi (epipolar horizontal)
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
print("PERCOBAAN 7: STEREO RECTIFICATION")
print("=" * 60)

# ============================================================
# 1. Memuat gambar stereo
# ============================================================

# Menampilkan informasi tahap memuat gambar
print("\n--- Memuat Gambar Stereo ---")

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
print(f"[INFO] Ukuran gambar kiri : {img_left.shape}")
print(f"[INFO] Ukuran gambar kanan: {img_right.shape}")

# Mengkonversi gambar ke grayscale untuk pemrosesan
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Mendapatkan ukuran gambar (width, height)
h, w = gray_left.shape
img_size = (w, h)

# ============================================================
# 2. Menghitung parameter kalibrasi stereo
# ============================================================

# Menampilkan informasi tahap kalibrasi
print("\n--- Menghitung Parameter Kalibrasi ---")

# Mendefinisikan focal length berdasarkan lebar gambar
focal_length = w * 1.0

# Mendefinisikan principal point di tengah gambar
cx, cy = w / 2.0, h / 2.0

# Membuat matriks intrinsik K (diasumsikan sama untuk kedua kamera)
K = np.array([
    [focal_length, 0, cx],
    [0, focal_length, cy],
    [0, 0, 1]
], dtype=np.float64)

# Menampilkan matriks intrinsik
print(f"[K] Focal length: {focal_length:.1f}")
print(f"[K] Principal point: ({cx:.1f}, {cy:.1f})")

# Mendefinisikan koefisien distorsi (diasumsikan nol untuk gambar sintetis)
D = np.zeros((5, 1), dtype=np.float64)
print(f"[D] Distorsi: {D.flatten()}")

# Membuat detektor SIFT untuk mencari fitur
sift = cv2.SIFT_create()

# Mendeteksi keypoint dan deskriptor pada gambar kiri
kp1, des1 = sift.detectAndCompute(gray_left, None)

# Mendeteksi keypoint dan deskriptor pada gambar kanan
kp2, des2 = sift.detectAndCompute(gray_right, None)

# Menampilkan jumlah keypoint
print(f"[SIFT] Keypoint kiri : {len(kp1)}")
print(f"[SIFT] Keypoint kanan: {len(kp2)}")

# Melakukan feature matching menggunakan BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Menerapkan Lowe's Ratio Test untuk menyaring good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Menampilkan jumlah good matches
print(f"[MATCH] Good matches: {len(good_matches)}")

# Mengekstrak koordinat titik korespondensi
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Mengestimasi Essential Matrix
E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Merecovery pose kamera (R dan T)
_, R, T, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

# Menampilkan informasi pose
print(f"\n[POSE] R (rotasi):\n{R}")
print(f"[POSE] T (translasi): {T.flatten()}")

# ============================================================
# 3. Menghitung transformasi rectification
# ============================================================

# Menampilkan informasi tahap rectification
print("\n--- Menghitung Rectification ---")

# Menghitung matriks rectification menggunakan cv2.stereoRectify
R1, R2, P1, P2, Q, validROI1, validROI2 = cv2.stereoRectify(
    K, D, K, D, img_size, R, T, alpha=0
)

# Menampilkan matriks rotasi rectification kamera kiri
print(f"[R1] Rotasi rectification kamera kiri:")
print(R1)

# Menampilkan matriks rotasi rectification kamera kanan
print(f"\n[R2] Rotasi rectification kamera kanan:")
print(R2)

# Menampilkan matriks proyeksi baru kamera kiri
print(f"\n[P1] Matriks proyeksi baru kamera kiri:")
print(P1)

# Menampilkan matriks proyeksi baru kamera kanan
print(f"\n[P2] Matriks proyeksi baru kamera kanan:")
print(P2)

# Menampilkan matriks disparity-to-depth Q
print(f"\n[Q] Matriks Disparity-to-Depth:")
print(Q)

# Menampilkan valid ROI untuk kedua gambar
print(f"\n[ROI1] Valid ROI kiri : {validROI1}")
print(f"[ROI2] Valid ROI kanan: {validROI2}")

# ============================================================
# 4. Membuat peta remap dan mengaplikasikannya
# ============================================================

# Menampilkan informasi tahap remap
print("\n--- Mengaplikasikan Rectification ---")

# Membuat peta undistort-rectify untuk kamera kiri
map1_left, map2_left = cv2.initUndistortRectifyMap(
    K, D, R1, P1, img_size, cv2.CV_32FC1
)

# Membuat peta undistort-rectify untuk kamera kanan
map1_right, map2_right = cv2.initUndistortRectifyMap(
    K, D, R2, P2, img_size, cv2.CV_32FC1
)

# Mengaplikasikan remap pada gambar kiri
rect_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)

# Mengaplikasikan remap pada gambar kanan
rect_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

# Menampilkan informasi hasil remap
print(f"[INFO] Ukuran gambar kiri rectified : {rect_left.shape}")
print(f"[INFO] Ukuran gambar kanan rectified: {rect_right.shape}")

# Menyimpan gambar rectified
path_rect_left = os.path.join(OUTPUT_DIR, "07_rectified_left.png")
cv2.imwrite(path_rect_left, rect_left)
print(f"[SAVED] Gambar kiri rectified: {path_rect_left}")

# Menyimpan gambar kanan rectified
path_rect_right = os.path.join(OUTPUT_DIR, "07_rectified_right.png")
cv2.imwrite(path_rect_right, rect_right)
print(f"[SAVED] Gambar kanan rectified: {path_rect_right}")

# ============================================================
# 5. Menggambar garis horizontal untuk verifikasi
# ============================================================

# Menampilkan informasi tahap verifikasi
print("\n--- Verifikasi Alignment Epipolar ---")

# Membuat salinan gambar rectified untuk visualisasi
rect_left_lines = rect_left.copy()
rect_right_lines = rect_right.copy()

# Mendefinisikan jumlah garis horizontal yang akan digambar
n_lines = 20

# Mendefinisikan warna hijau untuk garis
line_color = (0, 255, 0)

# Menggambar garis horizontal pada interval reguler
for i in range(n_lines):
    # Menghitung posisi y untuk garis ke-i
    y_pos = int(h * (i + 1) / (n_lines + 1))

    # Menggambar garis pada gambar kiri rectified
    cv2.line(rect_left_lines, (0, y_pos), (w, y_pos), line_color, 1)

    # Menggambar garis pada gambar kanan rectified
    cv2.line(rect_right_lines, (0, y_pos), (w, y_pos), line_color, 1)

# Menampilkan informasi jumlah garis
print(f"[INFO] {n_lines} garis horizontal digambar untuk verifikasi")

# Membuat gambar gabungan kiri-kanan dengan garis (side by side)
combined_lines = np.hstack([rect_left_lines, rect_right_lines])

# Menggambar garis penuh pada gambar gabungan
for i in range(n_lines):
    y_pos = int(h * (i + 1) / (n_lines + 1))
    cv2.line(combined_lines, (0, y_pos), (w * 2, y_pos), line_color, 1)

# Menyimpan gambar gabungan
path_combined = os.path.join(OUTPUT_DIR, "07_rectified_combined_lines.png")
cv2.imwrite(path_combined, combined_lines)
print(f"[SAVED] Gambar gabungan dengan garis: {path_combined}")

# ============================================================
# 6. Menggambar Valid ROI
# ============================================================

# Menampilkan informasi tahap ROI
print("\n--- Menampilkan Valid ROI ---")

# Membuat salinan untuk menggambar ROI
rect_left_roi = rect_left.copy()
rect_right_roi = rect_right.copy()

# Menggambar ROI pada gambar kiri jika valid
if validROI1 != (0, 0, 0, 0):
    x, y, rw, rh = validROI1
    cv2.rectangle(rect_left_roi, (x, y), (x + rw, y + rh), (0, 0, 255), 2)
    print(f"[ROI1] Valid area kiri : x={x}, y={y}, w={rw}, h={rh}")
else:
    print(f"[ROI1] Valid area kiri : seluruh gambar")

# Menggambar ROI pada gambar kanan jika valid
if validROI2 != (0, 0, 0, 0):
    x, y, rw, rh = validROI2
    cv2.rectangle(rect_right_roi, (x, y), (x + rw, y + rh), (0, 0, 255), 2)
    print(f"[ROI2] Valid area kanan: x={x}, y={y}, w={rw}, h={rh}")
else:
    print(f"[ROI2] Valid area kanan: seluruh gambar")

# ============================================================
# 7. Visualisasi perbandingan sebelum dan sesudah rectification
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi Perbandingan ---")

# Mengkonversi gambar ke RGB untuk matplotlib
img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
rect_left_rgb = cv2.cvtColor(rect_left_lines, cv2.COLOR_BGR2RGB)
rect_right_rgb = cv2.cvtColor(rect_right_lines, cv2.COLOR_BGR2RGB)
rect_left_roi_rgb = cv2.cvtColor(rect_left_roi, cv2.COLOR_BGR2RGB)
rect_right_roi_rgb = cv2.cvtColor(rect_right_roi, cv2.COLOR_BGR2RGB)

# Membuat figure dengan 3 baris dan 2 kolom
fig, axes = plt.subplots(3, 2, figsize=(14, 15))

# Menampilkan gambar kiri original (sebelum rectification)
axes[0, 0].imshow(img_left_rgb)
axes[0, 0].set_title("Gambar Kiri - Original", fontsize=11)
axes[0, 0].axis("off")

# Menampilkan gambar kanan original (sebelum rectification)
axes[0, 1].imshow(img_right_rgb)
axes[0, 1].set_title("Gambar Kanan - Original", fontsize=11)
axes[0, 1].axis("off")

# Menampilkan gambar kiri rectified dengan garis horizontal
axes[1, 0].imshow(rect_left_rgb)
axes[1, 0].set_title("Kiri - Rectified + Garis Epipolar", fontsize=11)
axes[1, 0].axis("off")

# Menampilkan gambar kanan rectified dengan garis horizontal
axes[1, 1].imshow(rect_right_rgb)
axes[1, 1].set_title("Kanan - Rectified + Garis Epipolar", fontsize=11)
axes[1, 1].axis("off")

# Menampilkan gambar kiri rectified dengan ROI
axes[2, 0].imshow(rect_left_roi_rgb)
axes[2, 0].set_title("Kiri - Rectified + Valid ROI (merah)", fontsize=11)
axes[2, 0].axis("off")

# Menampilkan gambar kanan rectified dengan ROI
axes[2, 1].imshow(rect_right_roi_rgb)
axes[2, 1].set_title("Kanan - Rectified + Valid ROI (merah)", fontsize=11)
axes[2, 1].axis("off")

# Mengatur judul utama figure
plt.suptitle("Percobaan 7: Stereo Rectification (Sebelum vs Sesudah)",
             fontsize=14, fontweight='bold')

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan figure ke file output
output_path = os.path.join(OUTPUT_DIR, "07_stereo_rectification.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[SAVED] Hasil disimpan di: {output_path}")

# Menampilkan figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 7: STEREO RECTIFICATION")
print("=" * 60)
print(f"1. Rectification membuat garis epipolar sejajar horizontal")
print(f"2. cv2.stereoRectify() menghitung R1, R2, P1, P2, Q")
print(f"3. cv2.initUndistortRectifyMap() membuat peta transformasi")
print(f"4. cv2.remap() mengaplikasikan transformasi pada gambar")
print(f"5. Garis horizontal memverifikasi alignment epipolar")
print(f"6. Valid ROI kiri : {validROI1}")
print(f"7. Valid ROI kanan: {validROI2}")
print(f"8. Matriks Q digunakan untuk konversi disparity ke depth")
print(f"9. Rectification menyederhanakan stereo matching")
print(f"10. Pencarian korespondensi menjadi 1D (sepanjang scanline)")
print("=" * 60)
