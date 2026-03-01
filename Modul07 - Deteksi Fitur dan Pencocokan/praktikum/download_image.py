"""
==========================================================================
SCRIPT DOWNLOAD DAN GENERATE GAMBAR SAMPLE
Modul 07 - Deteksi Fitur dan Pencocokan
==========================================================================
Script ini menyiapkan semua gambar yang dibutuhkan untuk 20 percobaan
Feature Detection dan Feature Matching.
- Membuat folder 'image/' dan 'output/'
- Men-generate gambar sintetis: checkerboard, bangunan, objek planar
- Membuat pasangan gambar overlapping untuk feature matching
- Membuat gambar objek dari sudut berbeda (simulasi)

Jalankan script ini PERTAMA KALI sebelum menjalankan percobaan lainnya.
==========================================================================
"""

# Mengimpor library os untuk operasi file dan folder
import os

# Mengimpor library numpy untuk operasi array/matriks
import numpy as np

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# ============================================================
# LANGKAH 1: Membuat struktur folder yang dibutuhkan
# ============================================================

# Mendapatkan path direktori tempat script ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder image dan output
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Membuat folder jika belum ada
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Folder 'image/' dan 'output/' siap.")

# ============================================================
# LANGKAH 2: Generate gambar checkerboard
# ============================================================

def buat_checkerboard(rows=8, cols=8, cell_size=60):
    """Membuat gambar pola papan catur untuk deteksi corner."""
    h = rows * cell_size
    w = cols * cell_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                y1, y2 = i * cell_size, (i + 1) * cell_size
                x1, x2 = j * cell_size, (j + 1) * cell_size
                img[y1:y2, x1:x2] = 255
    return img

# ============================================================
# LANGKAH 3: Generate gambar bangunan sintetis (banyak corner)
# ============================================================

def buat_bangunan():
    """Membuat gambar bangunan sintetis dengan banyak sudut dan garis."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200

    # Langit
    img[:200, :] = [230, 180, 130]

    # Tanah
    img[450:, :] = [120, 110, 100]

    # Bangunan utama
    cv2.rectangle(img, (100, 150), (350, 450), (160, 150, 140), -1)
    # Jendela bangunan utama (grid 3x4)
    for row in range(4):
        for col in range(3):
            x = 120 + col * 70
            y = 180 + row * 60
            cv2.rectangle(img, (x, y), (x + 45, y + 35), (200, 200, 220), -1)
            cv2.rectangle(img, (x, y), (x + 45, y + 35), (100, 100, 100), 1)

    # Pintu
    cv2.rectangle(img, (190, 370), (260, 450), (60, 50, 40), -1)

    # Bangunan kedua
    cv2.rectangle(img, (400, 200), (700, 450), (180, 170, 150), -1)
    # Jendela bangunan kedua
    for row in range(3):
        for col in range(4):
            x = 420 + col * 65
            y = 220 + row * 65
            cv2.rectangle(img, (x, y), (x + 40, y + 40), (200, 200, 230), -1)
            cv2.rectangle(img, (x, y), (x + 40, y + 40), (100, 100, 100), 1)

    # Atap segitiga pada bangunan pertama
    pts = np.array([[80, 150], [225, 50], [370, 150]], np.int32)
    cv2.fillPoly(img, [pts], (140, 80, 80))

    # Pagar
    for x in range(50, 780, 30):
        cv2.line(img, (x, 430), (x, 470), (80, 80, 80), 2)
    cv2.line(img, (50, 450), (780, 450), (80, 80, 80), 3)

    return img

# ============================================================
# LANGKAH 4: Generate pasangan gambar overlapping
# ============================================================

def buat_pasangan_overlapping():
    """Membuat 2 gambar yang saling overlap (untuk feature matching/stitching)."""
    # Membuat scene besar
    scene = np.ones((500, 1200, 3), dtype=np.uint8) * 200

    # Langit
    scene[:200, :] = [230, 180, 130]
    # Rumput
    scene[350:, :] = [60, 140, 60]

    # Pohon-pohon
    for tx in [100, 300, 500, 700, 900, 1100]:
        h_pohon = np.random.randint(100, 180)
        cv2.rectangle(scene, (tx - 8, 350 - h_pohon), (tx + 8, 350), (30, 80, 30), -1)
        cv2.circle(scene, (tx, 350 - h_pohon - 30), 45, (20, 120 + (tx % 40), 20), -1)

    # Rumah
    cv2.rectangle(scene, (350, 200), (550, 350), (160, 140, 130), -1)
    pts_atap = np.array([[330, 200], [450, 120], [570, 200]], np.int32)
    cv2.fillPoly(scene, [pts_atap], (120, 60, 60))
    cv2.rectangle(scene, (420, 280), (480, 350), (60, 50, 40), -1)
    for wx, wy in [(370, 230), (370, 280), (500, 230), (500, 280)]:
        cv2.rectangle(scene, (wx, wy), (wx + 30, wy + 25), (200, 200, 220), -1)

    # Matahari
    cv2.circle(scene, (1000, 80), 50, (0, 200, 255), -1)

    # Awan
    for cx in [200, 600, 800]:
        cv2.ellipse(scene, (cx, 60), (60, 25), 0, 0, 360, (255, 255, 255), -1)

    # Potong menjadi 2 gambar overlapping (overlap ~40%)
    w = scene.shape[1]
    mid = w // 2
    overlap = 240  # piksel overlap

    img_left = scene[:, :mid + overlap]
    img_right = scene[:, mid - overlap:]

    return img_left, img_right, scene

# ============================================================
# LANGKAH 5: Generate gambar objek planar (buku/poster)
# ============================================================

def buat_objek_planar(nama="buku"):
    """Membuat gambar objek planar untuk homography detection."""
    img = np.ones((400, 300, 3), dtype=np.uint8) * 240

    if nama == "buku":
        # Cover buku
        cv2.rectangle(img, (20, 20), (280, 380), (180, 50, 50), -1)
        cv2.rectangle(img, (30, 30), (270, 370), (200, 70, 70), 2)
        cv2.putText(img, "COMPUTER", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, "VISION", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, "2024", (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 230, 200), 2)
        cv2.line(img, (50, 260), (250, 260), (255, 200, 200), 2)
        cv2.putText(img, "Szeliski", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 230, 200), 2)

    elif nama == "poster":
        # Poster
        cv2.rectangle(img, (10, 10), (290, 390), (50, 50, 180), -1)
        cv2.circle(img, (150, 150), 80, (0, 200, 255), -1)
        cv2.putText(img, "CV LAB", (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        # Dekorasi
        for i in range(5):
            cv2.circle(img, (30 + i * 60, 350), 15, (200, 200, 50), -1)

    elif nama == "kartu":
        # Kartu ID
        cv2.rectangle(img, (10, 80), (290, 320), (255, 255, 255), -1)
        cv2.rectangle(img, (10, 80), (290, 320), (0, 0, 0), 2)
        cv2.rectangle(img, (10, 80), (290, 120), (50, 100, 200), -1)
        cv2.putText(img, "ID CARD", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Foto placeholder
        cv2.rectangle(img, (30, 140), (110, 240), (200, 200, 200), -1)
        cv2.circle(img, (70, 170), 20, (150, 150, 150), -1)
        cv2.rectangle(img, (50, 195), (90, 235), (150, 150, 150), -1)
        # Info
        cv2.putText(img, "Nama: John", (130, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "NIM: 123456", (130, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Prodi: TI", (130, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # Barcode-like
        for x in range(50, 250, 4):
            h = np.random.randint(15, 30)
            cv2.line(img, (x, 270), (x, 270 + h), (0, 0, 0), 1 + (x % 3))

    return img


def buat_objek_dalam_scene(objek_img):
    """Menempatkan objek planar dalam scene dengan perspektif."""
    scene = np.ones((500, 700, 3), dtype=np.uint8) * 180

    # Background (meja)
    scene[250:, :] = [140, 120, 100]

    # Simpan objek dalam scene (dengan sedikit transformasi perspektif)
    h, w = objek_img.shape[:2]

    # Titik sumber (4 corner dari objek)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Titik tujuan (perspektif di scene)
    dst_pts = np.float32([[200, 100], [480, 80], [500, 380], [180, 400]])

    # Hitung homography
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp objek ke scene
    warped = cv2.warpPerspective(objek_img, M, (700, 500))

    # Buat mask
    mask = cv2.warpPerspective(np.ones_like(objek_img) * 255, M, (700, 500))

    # Gabungkan
    mask_bool = mask > 128
    scene[mask_bool] = warped[mask_bool]

    return scene, dst_pts


# ============================================================
# LANGKAH 6: Generate semua gambar
# ============================================================

print("\n[INFO] Membuat gambar sintetis untuk percobaan feature detection...")

# 1. Checkerboard
img_checker = buat_checkerboard(8, 8, 60)
cv2.imwrite(os.path.join(IMAGE_DIR, "checkerboard.jpg"), img_checker)
print(f"  [OK] checkerboard.jpg")

# 2. Bangunan
img_bangunan = buat_bangunan()
cv2.imwrite(os.path.join(IMAGE_DIR, "bangunan.jpg"), img_bangunan)
print(f"  [OK] bangunan.jpg")

# 3. Pasangan overlapping
img_left, img_right, img_full = buat_pasangan_overlapping()
cv2.imwrite(os.path.join(IMAGE_DIR, "scene_left.jpg"), img_left)
cv2.imwrite(os.path.join(IMAGE_DIR, "scene_right.jpg"), img_right)
cv2.imwrite(os.path.join(IMAGE_DIR, "scene_full.jpg"), img_full)
print(f"  [OK] scene_left.jpg, scene_right.jpg, scene_full.jpg")

# 4. Objek planar
for nama in ["buku", "poster", "kartu"]:
    img_obj = buat_objek_planar(nama)
    cv2.imwrite(os.path.join(IMAGE_DIR, f"objek_{nama}.jpg"), img_obj)

    # Objek dalam scene
    img_scene, pts = buat_objek_dalam_scene(img_obj)
    cv2.imwrite(os.path.join(IMAGE_DIR, f"scene_{nama}.jpg"), img_scene)

print(f"  [OK] objek_buku/poster/kartu.jpg + scene_buku/poster/kartu.jpg")

# 5. Gambar rotated (template + rotated version)
img_template = buat_objek_planar("buku")
for angle in [0, 15, 30, 45, 90]:
    h, w = img_template.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_template, M, (w, h), borderValue=(200, 200, 200))
    cv2.imwrite(os.path.join(IMAGE_DIR, f"buku_rot{angle}.jpg"), rotated)
print(f"  [OK] buku_rot0/15/30/45/90.jpg")

# 6. Gambar scaled
for scale_pct in [50, 75, 100, 150, 200]:
    scale = scale_pct / 100.0
    h, w = img_template.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = cv2.resize(img_template, (new_w, new_h))
    cv2.imwrite(os.path.join(IMAGE_DIR, f"buku_scale{scale_pct}.jpg"), scaled)
print(f"  [OK] buku_scale50/75/100/150/200.jpg")

# 7. Gambar dengan perubahan iluminasi
for brightness in [-80, -40, 0, 40, 80]:
    img_bright = cv2.convertScaleAbs(img_template, alpha=1.0, beta=brightness)
    cv2.imwrite(os.path.join(IMAGE_DIR, f"buku_bright{brightness:+d}.jpg"), img_bright)
print(f"  [OK] buku_bright*.jpg (5 variasi)")

# 8. Gambar dengan tekstur kompleks
img_tekstur = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)
img_tekstur = cv2.GaussianBlur(img_tekstur, (5, 5), 2)
# Tambah pola
for i in range(0, 400, 40):
    cv2.line(img_tekstur, (0, i), (600, i + 20), (100, 100, 200), 2)
for j in range(0, 600, 50):
    cv2.circle(img_tekstur, (j, 200), 20, (200, 100, 100), 2)
cv2.imwrite(os.path.join(IMAGE_DIR, "tekstur_kompleks.jpg"), img_tekstur)
print(f"  [OK] tekstur_kompleks.jpg")

# 9. Gambar AR marker
img_marker = np.ones((300, 300, 3), dtype=np.uint8) * 255
cv2.rectangle(img_marker, (40, 40), (260, 260), (0, 0, 0), -1)
cv2.rectangle(img_marker, (60, 60), (140, 140), (255, 255, 255), -1)
cv2.rectangle(img_marker, (160, 60), (240, 140), (0, 0, 0), -1)
cv2.rectangle(img_marker, (60, 160), (140, 240), (0, 0, 0), -1)
cv2.rectangle(img_marker, (160, 160), (240, 240), (255, 255, 255), -1)
cv2.rectangle(img_marker, (80, 80), (120, 120), (0, 0, 0), -1)
cv2.imwrite(os.path.join(IMAGE_DIR, "ar_marker.jpg"), img_marker)
print(f"  [OK] ar_marker.jpg")

# 10. Gambar panorama scene (3 gambar berurutan)
panorama_scene = np.ones((400, 1800, 3), dtype=np.uint8) * 200
panorama_scene[:150, :] = [230, 180, 130]
panorama_scene[300:, :] = [60, 140, 60]

# Detail scene
for x in range(0, 1800, 200):
    # Pohon
    cv2.rectangle(panorama_scene, (x + 80, 200), (x + 100, 300), (30, 80, 30), -1)
    cv2.circle(panorama_scene, (x + 90, 180), 40, (20, 100 + x % 50, 20), -1)

# Bangunan di tengah
cv2.rectangle(panorama_scene, (700, 150), (1100, 300), (170, 160, 150), -1)
for wy in range(170, 280, 35):
    for wx in range(720, 1080, 45):
        cv2.rectangle(panorama_scene, (wx, wy), (wx + 25, wy + 20), (200, 200, 220), -1)

# Potong menjadi 3 gambar overlapping
overlap = 200
w_each = 700
for i, label in enumerate(["pano_left", "pano_center", "pano_right"]):
    start_x = i * (w_each - overlap)
    end_x = start_x + w_each
    if end_x > 1800:
        end_x = 1800
        start_x = end_x - w_each
    crop = panorama_scene[:, start_x:end_x]
    cv2.imwrite(os.path.join(IMAGE_DIR, f"{label}.jpg"), crop)
print(f"  [OK] pano_left/center/right.jpg")

# ============================================================
# Selesai
# ============================================================

print("\n" + "=" * 60)
print("SEMUA PERSIAPAN SELESAI!")
print("=" * 60)
print(f"\nStruktur folder:")
print(f"  {IMAGE_DIR}/")
print(f"    ├── checkerboard.jpg")
print(f"    ├── bangunan.jpg")
print(f"    ├── scene_left.jpg, scene_right.jpg, scene_full.jpg")
print(f"    ├── objek_buku/poster/kartu.jpg")
print(f"    ├── scene_buku/poster/kartu.jpg")
print(f"    ├── buku_rot*.jpg (5 rotasi)")
print(f"    ├── buku_scale*.jpg (5 skala)")
print(f"    ├── buku_bright*.jpg (5 pencahayaan)")
print(f"    ├── tekstur_kompleks.jpg")
print(f"    ├── ar_marker.jpg")
print(f"    └── pano_left/center/right.jpg")
print(f"  {OUTPUT_DIR}/")
print(f"\nSilakan lanjutkan ke percobaan 01-20!")
