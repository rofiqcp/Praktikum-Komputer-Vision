"""
==========================================================================
SCRIPT DOWNLOAD DAN GENERATE GAMBAR SAMPLE
Modul 10 - Computational Photography
==========================================================================
Script ini menyiapkan semua gambar yang dibutuhkan untuk 20 percobaan
Computational Photography.
- Membuat gambar dengan berbagai exposure (untuk HDR)
- Membuat gambar noisy (untuk denoising)
- Membuat gambar dengan area rusak (untuk inpainting)
- Membuat gambar beresolusi rendah (untuk super resolution)

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

# Mendapatkan path direktori tempat script ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("[INFO] Folder 'image/' dan 'output/' siap.")

# ============================================================
# LANGKAH 2: Membuat scene dasar berkualitas tinggi
# ============================================================

def buat_scene_pemandangan(width=800, height=600):
    """Membuat gambar pemandangan sintetis dengan detail tinggi."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Langit gradasi (biru ke putih)
    for y in range(height // 2):
        ratio = y / (height // 2)
        b = int(255 - ratio * 55)
        g = int(200 - ratio * 60)
        r = int(150 - ratio * 50)
        img[y, :] = (b, g, r)

    # Matahari
    cv2.circle(img, (600, 80), 50, (100, 220, 255), -1)
    cv2.circle(img, (600, 80), 60, (80, 180, 255), 3)

    # Gunung
    pts_gunung1 = np.array([[0, 300], [200, 120], [400, 300]], np.int32)
    pts_gunung2 = np.array([[200, 300], [450, 80], [700, 300]], np.int32)
    pts_gunung3 = np.array([[500, 300], [700, 150], [800, 300]], np.int32)
    cv2.fillPoly(img, [pts_gunung1], (80, 80, 60))
    cv2.fillPoly(img, [pts_gunung2], (100, 100, 80))
    cv2.fillPoly(img, [pts_gunung3], (90, 90, 70))

    # Salju di puncak gunung
    pts_salju = np.array([[400, 100], [450, 80], [500, 110]], np.int32)
    cv2.fillPoly(img, [pts_salju], (255, 255, 255))

    # Padang rumput
    for y in range(height // 2, height):
        ratio = (y - height // 2) / (height // 2)
        g = int(180 - ratio * 50)
        img[y, :] = (50, g, 50)

    # Pohon-pohon
    for tx in [100, 250, 450, 600, 720]:
        # Batang
        cv2.rectangle(img, (tx - 8, 350), (tx + 8, 450), (30, 60, 80), -1)
        # Daun
        cv2.circle(img, (tx, 320), 40, (30, 130 + np.random.randint(-20, 20), 30), -1)
        cv2.circle(img, (tx - 20, 340), 30, (20, 120 + np.random.randint(-20, 20), 20), -1)
        cv2.circle(img, (tx + 20, 340), 30, (25, 125 + np.random.randint(-20, 20), 25), -1)

    # Jalan
    pts_jalan = np.array([[350, height], [370, 350], [430, 350], [450, height]], np.int32)
    cv2.fillPoly(img, [pts_jalan], (100, 100, 100))
    # Garis tengah jalan
    for jy in range(360, height, 30):
        cv2.line(img, (400, jy), (400, jy + 15), (200, 200, 200), 2)

    # Rumah kecil
    cv2.rectangle(img, (540, 370), (620, 440), (60, 80, 180), -1)
    pts_atap = np.array([[530, 370], [580, 320], [630, 370]], np.int32)
    cv2.fillPoly(img, [pts_atap], (40, 40, 140))
    cv2.rectangle(img, (565, 400), (595, 440), (40, 60, 100), -1)
    cv2.rectangle(img, (548, 380), (562, 396), (200, 200, 150), -1)
    cv2.rectangle(img, (598, 380), (612, 396), (200, 200, 150), -1)

    # Awan
    for cx, cy in [(150, 60), (350, 40), (550, 70)]:
        cv2.ellipse(img, (cx, cy), (50, 20), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (cx + 30, cy - 10), (30, 15), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (cx - 20, cy + 5), (25, 12), 0, 0, 360, (250, 250, 250), -1)

    # Bunga di rumput
    for _ in range(30):
        fx = np.random.randint(50, width - 50)
        fy = np.random.randint(height // 2 + 50, height - 30)
        color = [(0, 0, 200), (0, 200, 200), (200, 0, 200), (200, 200, 0)][np.random.randint(4)]
        cv2.circle(img, (fx, fy), 4, color, -1)
        cv2.circle(img, (fx, fy), 2, (0, 255, 255), -1)

    return img


def buat_gambar_exposure(base_img, exposure_factor):
    """Mensimulasikan gambar dengan exposure berbeda."""
    # Mengalikan intensitas piksel dengan faktor exposure
    img_float = base_img.astype(np.float32) * exposure_factor
    # Clip ke range 0-255
    img_clipped = np.clip(img_float, 0, 255).astype(np.uint8)
    return img_clipped


def buat_gambar_noisy(base_img, sigma=30):
    """Menambahkan Gaussian noise ke gambar."""
    noise = np.random.randn(*base_img.shape) * sigma
    noisy = np.clip(base_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def buat_gambar_salt_pepper(base_img, amount=0.02):
    """Menambahkan salt & pepper noise ke gambar."""
    noisy = base_img.copy()
    # Salt (putih)
    num_salt = int(amount * base_img.size / 2)
    coords = [np.random.randint(0, i, num_salt) for i in base_img.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # Pepper (hitam)
    coords = [np.random.randint(0, i, num_salt) for i in base_img.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy


def buat_gambar_dengan_mask(base_img):
    """Membuat gambar dengan area rusak dan mask untuk inpainting."""
    damaged = base_img.copy()
    mask = np.zeros(base_img.shape[:2], np.uint8)

    # Menggambar coretan/garis acak sebagai kerusakan
    for _ in range(8):
        x1 = np.random.randint(50, base_img.shape[1] - 50)
        y1 = np.random.randint(50, base_img.shape[0] - 50)
        x2 = x1 + np.random.randint(-80, 80)
        y2 = y1 + np.random.randint(-80, 80)
        thickness = np.random.randint(5, 15)
        cv2.line(damaged, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    # Menambahkan beberapa lingkaran rusak
    for _ in range(5):
        cx = np.random.randint(80, base_img.shape[1] - 80)
        cy = np.random.randint(80, base_img.shape[0] - 80)
        r = np.random.randint(10, 25)
        cv2.circle(damaged, (cx, cy), r, (0, 255, 0), -1)
        cv2.circle(mask, (cx, cy), r, 255, -1)

    return damaged, mask


def buat_gambar_resolusi_rendah(base_img, scale=4):
    """Membuat gambar resolusi rendah dari gambar asli."""
    h, w = base_img.shape[:2]
    small = cv2.resize(base_img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    return small


def buat_gambar_portrait(width=600, height=800):
    """Membuat gambar portrait sintetis untuk efek bokeh."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 180

    # Background (dinding dengan tekstur)
    for y in range(height):
        for x in range(0, width, 2):
            val = 170 + int(10 * math.sin(x * 0.05) * math.cos(y * 0.05))
            img[y, x:x+2] = (val, val - 10, val - 20)

    # Lantai
    img[600:, :] = (120, 130, 140)

    # Orang (simplified)
    # Kepala
    cv2.ellipse(img, (300, 280), (70, 90), 0, 0, 360, (150, 170, 200), -1)
    # Mata
    cv2.ellipse(img, (275, 265), (12, 8), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (325, 265), (12, 8), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (275, 266), 5, (50, 40, 30), -1)
    cv2.circle(img, (325, 266), 5, (50, 40, 30), -1)
    # Hidung
    cv2.line(img, (300, 275), (295, 300), (120, 140, 170), 2)
    # Mulut
    cv2.ellipse(img, (300, 320), (20, 8), 0, 0, 180, (100, 100, 180), 2)
    # Rambut
    cv2.ellipse(img, (300, 230), (80, 60), 0, 180, 360, (40, 30, 20), -1)
    # Badan
    cv2.rectangle(img, (230, 370), (370, 600), (80, 60, 60), -1)
    # Leher
    cv2.rectangle(img, (280, 360), (320, 400), (150, 170, 200), -1)

    return img


def buat_depth_map_portrait(width=600, height=800):
    """Membuat depth map sintetis untuk efek bokeh."""
    depth = np.ones((height, width), dtype=np.float32) * 200

    # Area orang lebih dekat (depth rendah = dekat)
    cv2.ellipse(depth, (300, 280), (90, 110), 0, 0, 360, 50, -1)
    cv2.rectangle(depth, (220, 370), (380, 600), 50, -1)

    # Smooth depth map
    depth = cv2.GaussianBlur(depth, (31, 31), 15)

    return depth


def buat_gambar_style_reference(width=400, height=400):
    """Membuat gambar referensi style (mirip lukisan)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Membuat pola spiral berwarna-warni (mirip Starry Night)
    for y in range(height):
        for x in range(width):
            # Membuat pola bergelombang
            val_r = int(127 + 127 * math.sin(x * 0.03 + y * 0.02))
            val_g = int(127 + 127 * math.sin(x * 0.02 - y * 0.03 + 2))
            val_b = int(127 + 127 * math.cos(x * 0.025 + y * 0.015 + 1))
            img[y, x] = (val_b, val_g, val_r)

    # Menambahkan swirl effect
    img = cv2.GaussianBlur(img, (7, 7), 3)

    return img


def buat_gambar_wajah_sintetis(width=400, height=400):
    """Membuat gambar wajah sintetis sederhana."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 200

    # Wajah
    cv2.ellipse(img, (200, 200), (100, 130), 0, 0, 360, (160, 180, 210), -1)

    # Mata
    cv2.ellipse(img, (165, 175), (18, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (235, 175), (18, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (165, 176), 7, (60, 50, 40), -1)
    cv2.circle(img, (235, 176), 7, (60, 50, 40), -1)
    cv2.circle(img, (163, 174), 2, (255, 255, 255), -1)
    cv2.circle(img, (233, 174), 2, (255, 255, 255), -1)

    # Alis
    cv2.ellipse(img, (165, 155), (25, 5), -10, 180, 360, (80, 60, 40), 2)
    cv2.ellipse(img, (235, 155), (25, 5), 10, 180, 360, (80, 60, 40), 2)

    # Hidung
    pts = np.array([[200, 195], [190, 225], [210, 225]], np.int32)
    cv2.polylines(img, [pts], False, (130, 150, 180), 2)

    # Mulut
    cv2.ellipse(img, (200, 260), (30, 15), 0, 0, 180, (80, 80, 150), 2)

    # Rambut
    cv2.ellipse(img, (200, 130), (110, 70), 0, 180, 360, (40, 30, 20), -1)

    return img


# ============================================================
# LANGKAH 3: Generate semua gambar
# ============================================================

print("\n" + "=" * 60)
print("GENERATING ASSETS UNTUK MODUL 10")
print("=" * 60)

# 1. Gambar scene utama
scene = buat_scene_pemandangan()
cv2.imwrite(os.path.join(IMAGE_DIR, "scene_pemandangan.png"), scene)
print("[OK] scene_pemandangan.png")

# 2. Gambar dengan berbagai exposure (untuk HDR)
for i, exp in enumerate([0.3, 0.6, 1.0, 1.5, 2.5]):
    img_exp = buat_gambar_exposure(scene, exp)
    cv2.imwrite(os.path.join(IMAGE_DIR, f"exposure_{i+1}.png"), img_exp)
    print(f"[OK] exposure_{i+1}.png (factor={exp})")

# 3. Gambar noisy (untuk denoising)
noisy_gauss = buat_gambar_noisy(scene, sigma=30)
cv2.imwrite(os.path.join(IMAGE_DIR, "noisy_gaussian.png"), noisy_gauss)
print("[OK] noisy_gaussian.png")

noisy_heavy = buat_gambar_noisy(scene, sigma=60)
cv2.imwrite(os.path.join(IMAGE_DIR, "noisy_heavy.png"), noisy_heavy)
print("[OK] noisy_heavy.png")

noisy_sp = buat_gambar_salt_pepper(scene, 0.03)
cv2.imwrite(os.path.join(IMAGE_DIR, "noisy_salt_pepper.png"), noisy_sp)
print("[OK] noisy_salt_pepper.png")

# 4. Gambar rusak + mask (untuk inpainting)
damaged, inpaint_mask = buat_gambar_dengan_mask(scene)
cv2.imwrite(os.path.join(IMAGE_DIR, "damaged_image.png"), damaged)
cv2.imwrite(os.path.join(IMAGE_DIR, "inpaint_mask.png"), inpaint_mask)
print("[OK] damaged_image.png + inpaint_mask.png")

# 5. Gambar resolusi rendah (untuk super resolution)
low_res = buat_gambar_resolusi_rendah(scene, scale=4)
cv2.imwrite(os.path.join(IMAGE_DIR, "low_resolution.png"), low_res)
print("[OK] low_resolution.png")

# 6. Gambar portrait + depth map (untuk bokeh)
portrait = buat_gambar_portrait()
cv2.imwrite(os.path.join(IMAGE_DIR, "portrait.png"), portrait)
depth_map = buat_depth_map_portrait()
cv2.imwrite(os.path.join(IMAGE_DIR, "depth_map_portrait.png"),
            cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
print("[OK] portrait.png + depth_map_portrait.png")

# 7. Gambar style reference (untuk style transfer)
style_ref = buat_gambar_style_reference()
cv2.imwrite(os.path.join(IMAGE_DIR, "style_reference.png"), style_ref)
print("[OK] style_reference.png")

# 8. Gambar wajah (untuk enhancement)
wajah = buat_gambar_wajah_sintetis()
cv2.imwrite(os.path.join(IMAGE_DIR, "wajah_sintetis.png"), wajah)
print("[OK] wajah_sintetis.png")

# 9. Gambar gelap (untuk enhancement)
dark_img = buat_gambar_exposure(scene, 0.25)
cv2.imwrite(os.path.join(IMAGE_DIR, "gambar_gelap.png"), dark_img)
print("[OK] gambar_gelap.png")

# 10. Gambar dengan kontras rendah
low_contrast = np.clip(scene.astype(np.float32) * 0.4 + 80, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(IMAGE_DIR, "low_contrast.png"), low_contrast)
print("[OK] low_contrast.png")

# 11. Gambar tekstur untuk style transfer
tekstur = np.zeros((400, 400, 3), dtype=np.uint8)
for y in range(400):
    for x in range(400):
        tekstur[y, x] = (
            int(127 + 127 * math.sin(x * 0.1) * math.cos(y * 0.08)),
            int(127 + 127 * math.cos(x * 0.08 + y * 0.1)),
            int(127 + 127 * math.sin(x * 0.05 + y * 0.12))
        )
cv2.imwrite(os.path.join(IMAGE_DIR, "tekstur_pattern.png"), tekstur)
print("[OK] tekstur_pattern.png")

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

print(f"\n[SELESAI] Semua asset untuk Modul 10 berhasil dibuat!")
print(f"[INFO] Folder image: {IMAGE_DIR}")
print(f"[INFO] Folder output: {OUTPUT_DIR}")
print("[INFO] Silakan jalankan percobaan 01-20.")
