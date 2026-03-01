"""
==========================================================================
PERCOBAAN 20: PIPELINE REKONSTRUKSI 3D MULTI-VIEW
==========================================================================
Program ini mempelajari pipeline lengkap rekonstruksi 3D dari multiple
views: deteksi fitur, feature matching, estimasi Essential matrix,
recovery pose (R, t), triangulasi titik 3D, dan visualisasi point cloud
beserta posisi kamera.

Konsep utama:
- Rekonstruksi 3D memerlukan minimal 2 view dengan overlap yang cukup
- Fitur SIFT robust terhadap perubahan skala dan rotasi
- Essential matrix mengenkode hubungan geometri antar dua kamera
- recoverPose mengekstrak rotasi (R) dan translasi (t) dari E
- Triangulasi mengkonversi korespondensi 2D ke titik 3D
- PnP (Perspective-n-Point) mengestimasi pose kamera baru
- Akumulasi titik dari beberapa pasangan membentuk point cloud padat

Fungsi utama yang dipelajari:
- cv2.SIFT_create()          : Membuat detektor fitur SIFT
- cv2.BFMatcher()            : Membuat Brute-Force matcher
- cv2.findEssentialMat()     : Menghitung Essential matrix
- cv2.recoverPose()          : Mengekstrak R dan t dari Essential
- cv2.triangulatePoints()    : Triangulasi titik 3D dari 2 view
- cv2.solvePnPRansac()       : Estimasi pose kamera via PnP

Hasil: Visualisasi point cloud 3D rekonstruksi dari multiple views,
       posisi kamera pada setiap view, dan ringkasan pipeline
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk membuat dan menyimpan visualisasi
import matplotlib.pyplot as plt

# Mengimpor Axes3D untuk plotting 3D
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
print("PERCOBAAN 20: PIPELINE REKONSTRUKSI 3D MULTI-VIEW")
print("=" * 60)

# ============================================================
# 1. Memuat gambar multi-view real untuk pipeline rekonstruksi
# ============================================================

# Menampilkan informasi tahap memuat gambar
print("\n--- Memuat Gambar Multi-View Real ---")

# Mendefinisikan jumlah view kamera
num_views = 5

# Memeriksa apakah file multi-view tersedia
_mv_path_0 = os.path.join(IMAGE_DIR, "multiview_sfm_00.png")
if not os.path.exists(_mv_path_0):
    print("[WARN] Gambar multi-view tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)

# Mendefinisikan ukuran gambar target
h, w = 300, 400

# Mendefinisikan parameter kamera estimasi
fx, fy = 350.0, 350.0
cx, cy = w / 2.0, h / 2.0

# Membuat matriks intrinsik K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

# Menampilkan parameter kamera
print(f"[INFO] Ukuran gambar: {h}x{w}")
print(f"[INFO] Focal length: ({fx:.1f}, {fy:.1f})")
print(f"[INFO] Principal point: ({cx:.1f}, {cy:.1f})")

# Memuat semua gambar multi-view
views = []
for _i in range(num_views):
    _mv_path = os.path.join(IMAGE_DIR, f"multiview_sfm_{_i:02d}.png")
    _mv_img  = cv2.imread(_mv_path)
    if _mv_img is None:
        raise FileNotFoundError(
            f"[ERROR] multiview_sfm_{_i:02d}.png tidak tersedia.\n"
            "  Jalankan terlebih dahulu: python download_image.py"
        )
    _mv_img = cv2.resize(_mv_img, (w, h))
    views.append(_mv_img)
    print(f"[VIEW {_i}] Gambar dimuat: {_mv_img.shape}")

print(f"\n[INFO] Total {len(views)} view berhasil dimuat.")

# Membuat pose kamera estimasi (translasi horizontal bertahap)
camera_poses = []
for i in range(num_views):
    R = np.eye(3)
    t = np.array([[i * 0.5], [0.0], [0.0]], dtype=np.float64)
    camera_poses.append((R, t))

# Menampilkan informasi pose kamera
print(f"[INFO] Jumlah view kamera: {num_views}")
for i, (R, t) in enumerate(camera_poses):
    print(f"  View {i}: translasi = ({t[0,0]:.2f}, {t[1,0]:.2f}, {t[2,0]:.2f})")

# Fungsi untuk memproyeksikan titik 3D ke gambar 2D
def project_points(pts_3d, K, R, t):
    """Memproyeksikan titik 3D ke koordinat piksel 2D."""
    Rt = np.hstack([R, t])
    P = K @ Rt
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    proj = (P @ pts_h.T).T
    pts_2d = proj[:, :2] / proj[:, 2:3]
    visible = (proj[:, 2] > 0) & (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
              (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    return pts_2d, visible

# ============================================================
# 2. Deteksi fitur dan matching antar pasangan berurutan
# ============================================================

# Menampilkan informasi tahap deteksi fitur
print("\n--- Deteksi Fitur dan Matching ---")

# Membuat detektor SIFT
sift = cv2.SIFT_create(nfeatures=500)

# Membuat matcher Brute-Force dengan norm L2
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Menyiapkan list untuk menyimpan keypoints dan deskriptor setiap view
all_kps = []
all_descs = []

# Mendeteksi fitur SIFT pada setiap view
for i, img in enumerate(views):
    # Mengkonversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Mendeteksi keypoints dan menghitung deskriptor
    kps, descs = sift.detectAndCompute(gray, None)
    # Menyimpan keypoints dan deskriptor
    all_kps.append(kps)
    all_descs.append(descs)
    # Menampilkan jumlah fitur terdeteksi
    print(f"[VIEW {i}] Fitur SIFT terdeteksi: {len(kps)}")

# Menyiapkan list untuk menyimpan hasil matching antar pasangan
pair_matches = []

# Melakukan matching antar pasangan berurutan
for i in range(num_views - 1):
    # Menampilkan informasi pasangan yang sedang diproses
    print(f"\n[PAIR {i}-{i+1}] Matching fitur...")

    # Melakukan kNN matching dengan k=2
    matches = bf.knnMatch(all_descs[i], all_descs[i + 1], k=2)

    # Menerapkan ratio test (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Menyimpan matches yang lolos ratio test
    pair_matches.append(good_matches)

    # Menampilkan jumlah matches
    print(f"  Total matches: {len(matches)}, Good matches: {len(good_matches)}")

# ============================================================
# 3. Estimasi Essential Matrix dan Recovery Pose
# ============================================================

# Menampilkan informasi tahap estimasi pose
print("\n--- Estimasi Essential Matrix dan Recovery Pose ---")

# Menyiapkan list untuk menyimpan pose relatif
relative_poses = [(np.eye(3), np.zeros((3, 1)))]

# Menyiapkan list akumulasi titik 3D dan warna
all_pts_3d_recon = []
all_pts_colors_recon = []

# Iterasi untuk setiap pasangan view
for i in range(num_views - 1):
    # Mengekstrak koordinat keypoint dari matches
    src_pts = np.float32([all_kps[i][m.queryIdx].pt for m in pair_matches[i]])
    dst_pts = np.float32([all_kps[i + 1][m.trainIdx].pt for m in pair_matches[i]])

    # Menampilkan informasi jumlah korespondensi
    print(f"\n[PAIR {i}-{i+1}] Korespondensi: {len(src_pts)}")

    # Memastikan ada cukup korespondensi untuk estimasi
    if len(src_pts) < 8:
        print(f"  [WARN] Korespondensi tidak cukup, skip pair")
        relative_poses.append((np.eye(3), np.zeros((3, 1))))
        continue

    # Menghitung Essential matrix menggunakan RANSAC
    E, mask_e = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Menghitung jumlah inlier
    inliers = mask_e.ravel().sum()
    print(f"  Essential matrix inliers: {inliers}/{len(src_pts)}")

    # Mengekstrak rotasi dan translasi dari Essential matrix
    _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask_e)

    # Menampilkan informasi pose yang direcovery
    print(f"  Rotation angle: {np.degrees(np.arccos(np.clip((np.trace(R_rel)-1)/2, -1, 1))):.2f} deg")
    print(f"  Translation: ({t_rel[0,0]:.3f}, {t_rel[1,0]:.3f}, {t_rel[2,0]:.3f})")

    # Menyimpan pose relatif
    relative_poses.append((R_rel.copy(), t_rel.copy()))

    # ============================================================
    # 4. Triangulasi titik 3D dari pasangan view
    # ============================================================

    # Membuat matriks proyeksi untuk view pertama (identitas)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    # Membuat matriks proyeksi untuk view kedua
    P2 = K @ np.hstack([R_rel, t_rel])

    # Memfilter titik berdasarkan mask inlier
    inlier_mask = mask_e.ravel().astype(bool)
    src_inlier = src_pts[inlier_mask]
    dst_inlier = dst_pts[inlier_mask]

    # Melakukan triangulasi titik 3D
    pts_4d = cv2.triangulatePoints(P1, P2, src_inlier.T, dst_inlier.T)

    # Mengkonversi dari homogen ke kartesian
    pts_3d_tri = (pts_4d[:3] / pts_4d[3]).T

    # Memfilter titik yang berada di depan kedua kamera
    front_mask = pts_3d_tri[:, 2] > 0

    # Memfilter titik outlier (terlalu jauh)
    dist_mask = np.linalg.norm(pts_3d_tri, axis=1) < 50
    valid_tri = front_mask & dist_mask

    # Mengambil titik 3D yang valid
    pts_valid = pts_3d_tri[valid_tri]

    # Menampilkan informasi triangulasi
    print(f"  Titik triangulasi valid: {valid_tri.sum()}/{len(pts_3d_tri)}")

    # Menambahkan titik ke akumulasi global
    all_pts_3d_recon.append(pts_valid)

    # Membuat warna untuk titik (dari gambar sumber)
    colors_valid = []
    src_valid = src_inlier[valid_tri]
    for pt in src_valid:
        px, py = int(np.clip(pt[0], 0, w-1)), int(np.clip(pt[1], 0, h-1))
        bgr = views[i][py, px]
        colors_valid.append([bgr[2]/255.0, bgr[1]/255.0, bgr[0]/255.0])
    all_pts_colors_recon.append(np.array(colors_valid) if colors_valid else np.empty((0, 3)))

# ============================================================
# 5. Menghitung pose kamera global
# ============================================================

# Menampilkan informasi tahap pose global
print("\n--- Menghitung Pose Kamera Global ---")

# Menyiapkan list pose global (kamera pertama di origin)
global_R = [np.eye(3)]
global_t = [np.zeros((3, 1))]

# Mengakumulasi pose relatif menjadi global
for i in range(1, num_views):
    # Mengambil pose relatif
    R_rel, t_rel = relative_poses[i]
    # Menghitung pose global: R_global = R_rel * R_prev, t_global = R_rel * t_prev + t_rel
    R_new = R_rel @ global_R[i - 1]
    t_new = R_rel @ global_t[i - 1] + t_rel
    # Menyimpan pose global
    global_R.append(R_new)
    global_t.append(t_new)

# Menampilkan posisi Global setiap kamera
for i in range(num_views):
    # Menghitung posisi kamera di world frame: C = -R^T * t
    cam_pos = -global_R[i].T @ global_t[i]
    print(f"  Kamera {i}: posisi = ({cam_pos[0,0]:.3f}, {cam_pos[1,0]:.3f}, {cam_pos[2,0]:.3f})")

# ============================================================
# 6. Visualisasi Point Cloud dan Posisi Kamera
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Visualisasi Point Cloud dan Posisi Kamera ---")

# Menggabungkan semua titik 3D hasil rekonstruksi
if all_pts_3d_recon:
    recon_pts = np.vstack(all_pts_3d_recon)
    recon_colors = np.vstack(all_pts_colors_recon) if all_pts_colors_recon else None
else:
    recon_pts = np.empty((0, 3))
    recon_colors = None

# Menampilkan jumlah total titik 3D
print(f"[INFO] Total titik 3D rekonstruksi: {len(recon_pts)}")

# Membuat figure untuk visualisasi 3D
fig = plt.figure(figsize=(16, 12))

# Membuat subplot 3D untuk point cloud tampak perspektif
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# Menampilkan titik 3D rekonstruksi
if len(recon_pts) > 0:
    ax1.scatter(recon_pts[:, 0], recon_pts[:, 1], recon_pts[:, 2],
                c=recon_colors if recon_colors is not None else 'blue',
                s=5, alpha=0.6)

# Menampilkan posisi kamera
cam_positions = []
for i in range(num_views):
    # Menghitung posisi kamera
    cam_pos = -global_R[i].T @ global_t[i]
    cam_positions.append(cam_pos.flatten())
    # Menggambar posisi kamera sebagai titik besar
    ax1.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='red', s=100, marker='^')
    # Menambahkan label kamera
    ax1.text(cam_pos[0, 0], cam_pos[1, 0], cam_pos[2, 0], f'C{i}', fontsize=8)

# Mengatur judul subplot
ax1.set_title("Point Cloud + Kamera (3D View)", fontsize=11)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Membuat subplot 3D tampak atas (top view)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')

# Menampilkan titik 3D dari atas
if len(recon_pts) > 0:
    ax2.scatter(recon_pts[:, 0], recon_pts[:, 2], recon_pts[:, 1],
                c='steelblue', s=5, alpha=0.6)

# Menampilkan posisi kamera dari atas
for i, cam_pos in enumerate(cam_positions):
    ax2.scatter(cam_pos[0], cam_pos[2], cam_pos[1], c='red', s=100, marker='^')

# Mengatur sudut pandang dari atas
ax2.view_init(elev=80, azim=-90)
ax2.set_title("Point Cloud (Top View)", fontsize=11)
ax2.set_xlabel("X")
ax2.set_ylabel("Z")
ax2.set_zlabel("Y")

# Membuat subplot untuk menampilkan sample view
ax3 = fig.add_subplot(2, 2, 3)

# Menampilkan gambar dari view pertama
ax3.imshow(cv2.cvtColor(views[0], cv2.COLOR_BGR2RGB))
ax3.set_title("View 0 (Referensi)", fontsize=11)
ax3.axis('off')

# Membuat subplot untuk menampilkan view terakhir
ax4 = fig.add_subplot(2, 2, 4)

# Menampilkan gambar dari view terakhir
ax4.imshow(cv2.cvtColor(views[-1], cv2.COLOR_BGR2RGB))
ax4.set_title(f"View {num_views-1} (Terakhir)", fontsize=11)
ax4.axis('off')

# Mengatur judul utama
fig.suptitle("Pipeline Rekonstruksi 3D Multi-View", fontsize=16, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi point cloud ke file
output_path1 = os.path.join(OUTPUT_DIR, "20_multiview_reconstruction.png")
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"[SAVE] Visualisasi rekonstruksi disimpan: {output_path1}")

# Menutup figure
plt.close(fig)

# ============================================================
# 7. Visualisasi matching antar view
# ============================================================

# Menampilkan informasi tahap visualisasi matching
print("\n--- Visualisasi Feature Matching ---")

# Membuat figure untuk menampilkan matching result
fig2, axes_match = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle("Feature Matching antar Pasangan View", fontsize=15, fontweight='bold')

# Menampilkan matching untuk setiap pasangan
for i in range(min(4, num_views - 1)):
    # Menentukan posisi subplot
    row = i // 2
    col = i % 2

    # Menggambar matches pada pasangan view
    match_img = cv2.drawMatches(
        views[i], all_kps[i],
        views[i + 1], all_kps[i + 1],
        pair_matches[i][:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Menampilkan gambar matching
    axes_match[row, col].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    axes_match[row, col].set_title(f"View {i} ↔ View {i+1} ({len(pair_matches[i])} matches)", fontsize=10)
    axes_match[row, col].axis('off')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi matching ke file
output_path2 = os.path.join(OUTPUT_DIR, "20_feature_matching.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[SAVE] Visualisasi matching disimpan: {output_path2}")

# Menutup figure
plt.close(fig2)

# ============================================================
# 8. Ringkasan Pipeline Rekonstruksi
# ============================================================

# Menampilkan ringkasan pipeline
print("\n" + "=" * 60)
print("RINGKASAN PIPELINE REKONSTRUKSI 3D")
print("=" * 60)
print(f"  Jumlah view            : {num_views}")
print(f"  Titik 3D di scene      : {num_3d_points}")
print(f"  Titik 3D rekonstruksi  : {len(recon_pts)}")

# Menampilkan statistik matching per pasangan
for i in range(num_views - 1):
    print(f"  Pair {i}-{i+1} matches     : {len(pair_matches[i])}")

# Menampilkan posisi kamera
print(f"\n  Posisi Kamera:")
for i, pos in enumerate(cam_positions):
    print(f"    C{i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

# Menampilkan jumlah file output
print(f"\n  File output tersimpan  : 2 file")
print("=" * 60)
