"""
==========================================================================
SCRIPT DOWNLOAD DAN GENERATE GAMBAR SAMPLE
Modul 11 - Structure from Motion dan Depth Estimation
==========================================================================
Script ini menyiapkan semua gambar yang dibutuhkan untuk 20 percobaan
SfM dan Depth Estimation.
- Membuat pasangan gambar stereo sintetis
- Membuat gambar dengan pola kalibrasi (checkerboard)
- Membuat gambar multi-view untuk SfM
- Membuat depth map sintetis

Jalankan script ini PERTAMA KALI sebelum menjalankan percobaan lainnya.
==========================================================================
"""

# Mengimpor library yang dibutuhkan
import os
import numpy as np
import cv2
import math

# ============================================================
# LANGKAH 1: Membuat struktur folder
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("[INFO] Folder 'image/' dan 'output/' siap.")

# ============================================================
# LANGKAH 2: Generate gambar
# ============================================================

def buat_scene_3d(width=800, height=600, camera_x_offset=0, camera_y_offset=0):
    """
    Membuat gambar scene 3D sintetis dengan perspektif.
    camera_x_offset mensimulasikan pergeseran kamera (stereo baseline).
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 200

    # Langit
    for y in range(height // 2):
        ratio = y / (height // 2)
        img[y, :] = (int(230 - ratio * 30), int(200 - ratio * 40), int(160 - ratio * 30))

    # Lantai dengan perspektif grid
    for z in range(1, 50):
        # Garis horizontal perspektif
        y_pos = int(height // 2 + (height // 2) * (1 - 1.0 / (z * 0.3 + 1)))
        if y_pos < height:
            brightness = max(100, 200 - z * 3)
            cv2.line(img, (0, y_pos), (width, y_pos), (brightness, brightness - 10, brightness - 20), 1)

        # Garis vertikal perspektif
        for vx in range(-10, 11):
            vanish_x = width // 2 - camera_x_offset
            x_pos = int(vanish_x + vx * 40 * (1.0 / (z * 0.3 + 1)))
            if 0 <= x_pos < width and y_pos < height:
                cv2.circle(img, (x_pos, y_pos), 1, (brightness, brightness, brightness), -1)

    # Kotak 3D (kubus sederhana) dengan parallax berdasarkan offset kamera
    def draw_box(cx, cy, size, depth, color):
        """Menggambar kotak 3D dengan parallax."""
        # Parallax: objek dekat bergeser lebih banyak
        parallax = int(camera_x_offset * (50.0 / max(depth, 1)))
        px = cx + parallax

        # Sisi depan
        half = size // 2
        cv2.rectangle(img, (px - half, cy - half), (px + half, cy + half), color, -1)
        cv2.rectangle(img, (px - half, cy - half), (px + half, cy + half),
                       tuple(max(0, c - 50) for c in color), 2)

        # Sisi atas (3D look)
        d = size // 4
        pts = np.array([
            [px - half, cy - half],
            [px - half + d, cy - half - d],
            [px + half + d, cy - half - d],
            [px + half, cy - half]
        ], np.int32)
        cv2.fillPoly(img, [pts], tuple(min(255, c + 30) for c in color))

        # Sisi kanan
        pts2 = np.array([
            [px + half, cy - half],
            [px + half + d, cy - half - d],
            [px + half + d, cy + half - d],
            [px + half, cy + half]
        ], np.int32)
        cv2.fillPoly(img, [pts2], tuple(max(0, c - 30) for c in color))

    # Menggambar beberapa objek 3D di berbagai kedalaman
    draw_box(200, 350, 80, 20, (0, 0, 180))    # Dekat, merah
    draw_box(400, 300, 60, 40, (0, 150, 0))     # Sedang, hijau
    draw_box(600, 280, 40, 60, (180, 0, 0))     # Jauh, biru
    draw_box(150, 280, 50, 35, (0, 150, 150))   # Sedang, cyan
    draw_box(500, 400, 100, 10, (150, 0, 150))   # Sangat dekat, ungu

    # Lingkaran (bola) di berbagai kedalaman
    for (bx, by, br, depth, color) in [
        (300, 200, 25, 50, (100, 200, 100)),
        (550, 350, 40, 15, (200, 100, 100)),
        (100, 400, 35, 8, (100, 100, 200)),
    ]:
        parallax = int(camera_x_offset * (50.0 / max(depth, 1)))
        cv2.circle(img, (bx + parallax, by), br, color, -1)
        cv2.circle(img, (bx + parallax - br//4, by - br//4), br//4,
                   tuple(min(255, c + 50) for c in color), -1)

    return img


def buat_stereo_pair(width=800, height=600, baseline=30):
    """Membuat pasangan gambar stereo (kiri dan kanan)."""
    img_left = buat_scene_3d(width, height, camera_x_offset=0)
    img_right = buat_scene_3d(width, height, camera_x_offset=baseline)
    return img_left, img_right


def buat_checkerboard(rows=9, cols=6, square_size=50):
    """Membuat gambar pola checkerboard untuk kalibrasi kamera."""
    height = rows * square_size + 100
    width = cols * square_size + 100
    img = np.ones((height, width, 3), dtype=np.uint8) * 200

    # Menggambar pola kotak hitam-putih
    offset_y, offset_x = 50, 50
    for r in range(rows):
        for c in range(cols):
            x = offset_x + c * square_size
            y = offset_y + r * square_size
            if (r + c) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + square_size, y + square_size), (255, 255, 255), -1)
            else:
                cv2.rectangle(img, (x, y), (x + square_size, y + square_size), (0, 0, 0), -1)

    return img


def buat_checkerboard_perspektif(rows=9, cols=6, square_size=50, angle=15, axis='y'):
    """Membuat gambar checkerboard dengan perspektif (rotasi)."""
    flat = buat_checkerboard(rows, cols, square_size)
    h, w = flat.shape[:2]
    center = (w // 2, h // 2)

    if axis == 'y':
        # Simulasi rotasi Y dengan perspektif transform
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        offset = int(w * 0.1 * (angle / 30))
        dst = np.float32([
            [offset, offset // 2],
            [w - offset // 2, 0],
            [w, h],
            [offset // 2, h - offset // 2]
        ])
    else:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        offset = int(h * 0.1 * (angle / 30))
        dst = np.float32([
            [offset // 2, offset],
            [w - offset // 2, offset // 2],
            [w, h - offset // 2],
            [0, h]
        ])

    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(flat, M, (w, h), borderValue=(200, 200, 200))
    return result


def buat_multiview_set(n_views=5, width=800, height=600):
    """Membuat set gambar dari berbagai sudut pandang."""
    views = []
    for i in range(n_views):
        offset = int((i - n_views // 2) * 20)
        view = buat_scene_3d(width, height, camera_x_offset=offset)
        # Tambahkan sedikit rotasi
        angle = (i - n_views // 2) * 2
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        view = cv2.warpAffine(view, M, (width, height), borderValue=(200, 200, 200))
        views.append(view)
    return views


def buat_depth_map_sintetis(width=800, height=600):
    """Membuat depth map sintetis yang sesuai dengan scene 3D."""
    depth = np.ones((height, width), dtype=np.float32) * 200  # Background jauh

    # Lantai: gradasi depth (atas=jauh, bawah=dekat)
    for y in range(height // 2, height):
        ratio = (y - height // 2) / (height // 2)
        depth[y, :] = 200 - ratio * 150  # Dekat di bawah

    # Objek di berbagai depth
    cv2.rectangle(depth, (160, 310), (240, 390), 20, -1)    # Kotak dekat
    cv2.rectangle(depth, (370, 270), (430, 330), 40, -1)     # Kotak sedang
    cv2.rectangle(depth, (580, 260), (620, 300), 60, -1)     # Kotak jauh
    cv2.circle(depth, (300, 200), 25, 50, -1)                # Bola sedang
    cv2.circle(depth, (550, 350), 40, 15, -1)                # Bola dekat
    cv2.rectangle(depth, (100, 365), (200, 435), 8, -1)      # Kotak sangat dekat
    cv2.circle(depth, (100, 400), 35, 8, -1)                 # Bola sangat dekat

    # Smooth
    depth = cv2.GaussianBlur(depth, (15, 15), 5)

    return depth


def buat_gambar_fitur(width=800, height=600):
    """Membuat gambar dengan banyak fitur untuk feature matching."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 180

    np.random.seed(42)

    # Menggambar banyak bentuk geometris (fitur yang mudah dideteksi)
    for _ in range(20):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        size = np.random.randint(15, 60)
        color = tuple(np.random.randint(0, 200, 3).tolist())
        shape_type = np.random.randint(0, 4)

        if shape_type == 0:
            cv2.rectangle(img, (x, y), (x + size, y + size), color, -1)
        elif shape_type == 1:
            cv2.circle(img, (x, y), size // 2, color, -1)
        elif shape_type == 2:
            pts = np.array([
                [x, y - size // 2],
                [x - size // 2, y + size // 2],
                [x + size // 2, y + size // 2]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.ellipse(img, (x, y), (size, size // 2),
                        np.random.randint(0, 180), 0, 360, color, -1)

    # Menambahkan tekstur
    for _ in range(100):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        cv2.circle(img, (x, y), 3, (np.random.randint(100, 200),) * 3, -1)

    return img


def buat_pair_dengan_homography(img, angle=10, tx=20, ty=10):
    """Membuat pasangan gambar dengan transformasi homography diketahui."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Membuat transformasi: rotasi + translasi
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty

    # Mengaplikasikan transformasi
    transformed = cv2.warpAffine(img, M_rot, (w, h), borderValue=(180, 180, 180))

    return transformed


# ============================================================
# LANGKAH 3: Generate semua gambar
# ============================================================

print("\n" + "=" * 60)
print("GENERATING ASSETS UNTUK MODUL 11")
print("=" * 60)

# 1. Stereo pair
img_left, img_right = buat_stereo_pair()
cv2.imwrite(os.path.join(IMAGE_DIR, "stereo_left.png"), img_left)
cv2.imwrite(os.path.join(IMAGE_DIR, "stereo_right.png"), img_right)
print("[OK] stereo_left.png + stereo_right.png")

# 2. Stereo pair kedua (baseline lebih besar)
img_left2, img_right2 = buat_stereo_pair(baseline=50)
cv2.imwrite(os.path.join(IMAGE_DIR, "stereo_left_wide.png"), img_left2)
cv2.imwrite(os.path.join(IMAGE_DIR, "stereo_right_wide.png"), img_right2)
print("[OK] stereo_left_wide.png + stereo_right_wide.png")

# 3. Checkerboard images (berbagai sudut)
checker_flat = buat_checkerboard()
cv2.imwrite(os.path.join(IMAGE_DIR, "checkerboard_flat.png"), checker_flat)
print("[OK] checkerboard_flat.png")

for i, angle in enumerate([10, 20, 30]):
    for axis in ['y', 'x']:
        cb = buat_checkerboard_perspektif(angle=angle, axis=axis)
        cv2.imwrite(os.path.join(IMAGE_DIR, f"checkerboard_rot{axis}_{angle}.png"), cb)
        print(f"[OK] checkerboard_rot{axis}_{angle}.png")

# 4. Multi-view images
views = buat_multiview_set(n_views=5)
for i, v in enumerate(views):
    cv2.imwrite(os.path.join(IMAGE_DIR, f"view_{i+1}.png"), v)
    print(f"[OK] view_{i+1}.png")

# 5. Depth map sintetis
depth_map = buat_depth_map_sintetis()
depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(os.path.join(IMAGE_DIR, "depth_map_gt.png"), depth_vis)
# Simpan juga versi float sebagai npy
np.save(os.path.join(IMAGE_DIR, "depth_map_gt.npy"), depth_map)
print("[OK] depth_map_gt.png + depth_map_gt.npy")

# 6. Gambar dengan banyak fitur
img_fitur = buat_gambar_fitur()
cv2.imwrite(os.path.join(IMAGE_DIR, "gambar_fitur.png"), img_fitur)
print("[OK] gambar_fitur.png")

# 7. Pair dengan homography
img_fitur2 = buat_pair_dengan_homography(img_fitur)
cv2.imwrite(os.path.join(IMAGE_DIR, "gambar_fitur_rotasi.png"), img_fitur2)
print("[OK] gambar_fitur_rotasi.png")

# 8. Scene lebih detail untuk SfM
scene_detail = buat_scene_3d(camera_x_offset=0)
cv2.imwrite(os.path.join(IMAGE_DIR, "scene_3d.png"), scene_detail)
print("[OK] scene_3d.png")

# 9. Gambar untuk undistort (sedikit barrel distortion simulasi)
def simulate_distortion(img, k1=0.0003):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = (x - cx) / cx
            dy = (y - cy) / cy
            r2 = dx * dx + dy * dy
            factor = 1 + k1 * r2 * 10000
            map_x[y, x] = cx + dx * factor * cx
            map_y[y, x] = cy + dy * factor * cy
    distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(200, 200, 200))
    return distorted

distorted = simulate_distortion(checker_flat)
cv2.imwrite(os.path.join(IMAGE_DIR, "checkerboard_distorted.png"), distorted)
print("[OK] checkerboard_distorted.png")

# ============================================================
# LANGKAH 4: Verifikasi
# ============================================================
print("\n" + "=" * 60)
print("VERIFIKASI FILE")
print("=" * 60)

for f in sorted(os.listdir(IMAGE_DIR)):
    filepath = os.path.join(IMAGE_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  [✓] {f} ({size_kb:.1f} KB)")

print(f"\n[SELESAI] Semua asset untuk Modul 11 berhasil dibuat!")
print(f"[INFO] Folder image: {IMAGE_DIR}")
print(f"[INFO] Folder output: {OUTPUT_DIR}")
print("[INFO] Silakan jalankan percobaan 01-20.")
