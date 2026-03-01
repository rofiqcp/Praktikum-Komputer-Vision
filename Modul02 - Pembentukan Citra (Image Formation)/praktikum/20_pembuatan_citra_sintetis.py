"""
==========================================================================
PERCOBAAN 20: PEMBUATAN CITRA SINTETIS
==========================================================================
Membuat gambar sintetis menggunakan operasi matematika dan OpenCV.
Berguna untuk testing algoritma, kalibrasi, dan simulasi.

Fungsi:
- np.zeros/ones/full → canvas kosong
- cv2.line/circle/rectangle/ellipse → primitif geometri
- np.sin/cos → fungsi periodik (pola sinusoidal)
- cv2.getGaborKernel() → kernel Gabor untuk tekstur
- np.random → noise (Gaussian, uniform, salt-pepper)
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 20: PEMBUATAN CITRA SINTETIS")
print("=" * 60)

results = {}

# ============================================================
# 1. Gradient Linear (Horizontal & Vertikal)
# ============================================================
print("\n--- 1. Gradient Linear ---")

# Gradient horizontal: kiri gelap, kanan terang
grad_h = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
results["Gradient H"] = grad_h

# Gradient vertikal: atas gelap, bawah terang
grad_v = np.tile(np.linspace(0, 255, 512, dtype=np.uint8).reshape(-1, 1), (1, 512))
results["Gradient V"] = grad_v
print("  Gradient horizontal dan vertikal dibuat (512×512)")

# ============================================================
# 2. Gradient Radial
# ============================================================
print("\n--- 2. Gradient Radial ---")

# Koordinat berpusat di tengah
y_coords, x_coords = np.mgrid[0:512, 0:512]
cx, cy = 256, 256
# Jarak dari pusat
dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
# Normalisasi ke [0, 255]
grad_r = (255 * dist / dist.max()).astype(np.uint8)
results["Gradient Radial"] = grad_r

# ============================================================
# 3. Pola Sinusoidal
# ============================================================
print("\n--- 3. Pola Sinusoidal ---")

# Sinusoidal horizontal
freq = 10  # siklus
x = np.linspace(0, 2 * np.pi * freq, 512)
sin_h = ((np.sin(x) + 1) / 2 * 255).astype(np.uint8)
sin_img = np.tile(sin_h, (512, 1))
results["Sinusoidal H"] = sin_img

# Sinusoidal 2D (checkerboard halus)
xx, yy = np.meshgrid(x, x)
sin_2d = ((np.sin(xx) * np.sin(yy) + 1) / 2 * 255).astype(np.uint8)
results["Sinusoidal 2D"] = sin_2d
print(f"  Frekuensi: {freq} siklus")

# ============================================================
# 4. Pola Checkerboard
# ============================================================
print("\n--- 4. Pola Checkerboard ---")

cell_size = 32
# Buat pola kotak hitam-putih
board = np.zeros((512, 512), dtype=np.uint8)
for i in range(512):
    for j in range(512):
        if ((i // cell_size) + (j // cell_size)) % 2 == 0:
            board[i, j] = 255
results["Checkerboard"] = board
print(f"  Cell size: {cell_size}px")

# ============================================================
# 5. Pola Zone Plate (Chirp)
# ============================================================
print("\n--- 5. Zone Plate ---")

# Frekuensi meningkat dengan jarak dari pusat
r2 = (x_coords - 256) ** 2 + (y_coords - 256) ** 2
# Zone plate: cos(k * r²)
k_zone = 0.0005
zone = ((np.cos(k_zone * r2) + 1) / 2 * 255).astype(np.uint8)
results["Zone Plate"] = zone

# ============================================================
# 6. Gabor Pattern
# ============================================================
print("\n--- 6. Gabor Pattern ---")

# Buat kernel Gabor besar untuk visualisasi
ksize = 511
sigma = 80
theta = np.pi / 4  # 45 derajat
lambd = 40  # panjang gelombang
gamma = 0.5
psi = 0

# Buat Gabor kernel
gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
# Normalisasi ke [0, 255]
gabor_vis = cv2.normalize(gabor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# Resize ke 512x512
gabor_vis = cv2.resize(gabor_vis, (512, 512))
results["Gabor θ=45°"] = gabor_vis
print(f"  σ={sigma}, θ=45°, λ={lambd}, γ={gamma}")

# ============================================================
# 7. Noise Patterns
# ============================================================
print("\n--- 7. Noise Patterns ---")

# Gaussian noise
gauss_noise = np.random.normal(128, 50, (512, 512))
gauss_noise = np.clip(gauss_noise, 0, 255).astype(np.uint8)
results["Gaussian Noise"] = gauss_noise

# Uniform noise
uniform_noise = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
results["Uniform Noise"] = uniform_noise

# Salt and pepper noise
sp = np.full((512, 512), 128, dtype=np.uint8)
# Probabilitas salt dan pepper
prob = 0.05
salt = np.random.random((512, 512)) < prob
pepper = np.random.random((512, 512)) < prob
sp[salt] = 255
sp[pepper] = 0
results["Salt & Pepper"] = sp
print("  Gaussian, Uniform, Salt&Pepper dibuat")

# ============================================================
# 8. Pola Concentric Rings (Lingkaran Konsentris)
# ============================================================
print("\n--- 8. Concentric Rings ---")

ring_img = np.zeros((512, 512), dtype=np.uint8)
# Gambar lingkaran konsentris
for r in range(10, 300, 15):
    cv2.circle(ring_img, (256, 256), r, 255, 2)
results["Concentric Rings"] = ring_img

# ============================================================
# 9. Pola Bintang (Star / Siemens)
# ============================================================
print("\n--- 9. Pola Siemens Star ---")

star = np.zeros((512, 512), dtype=np.uint8)
n_spokes = 36
# Sudut antara spoke
for i in range(n_spokes):
    angle = 2 * np.pi * i / n_spokes
    x_end = int(256 + 250 * np.cos(angle))
    y_end = int(256 + 250 * np.sin(angle))
    cv2.line(star, (256, 256), (x_end, y_end), 255, 3)
results["Siemens Star"] = star

# ============================================================
# 10. Citra Warna Sintetis (RGB)
# ============================================================
print("\n--- 10. Citra Warna Sintetis ---")

color_img = np.zeros((512, 512, 3), dtype=np.uint8)
# Channel Red: gradient horizontal
color_img[:, :, 2] = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
# Channel Green: gradient vertikal
color_img[:, :, 1] = np.tile(np.linspace(0, 255, 512, dtype=np.uint8).reshape(-1, 1), (1, 512))
# Channel Blue: gradient diagonal
diag = np.linspace(0, 255, 512)
xx_d, yy_d = np.meshgrid(diag, diag)
color_img[:, :, 0] = ((xx_d + yy_d) / 2).astype(np.uint8)

# ============================================================
# 11. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

keys = list(results.keys())
for idx in range(min(12, len(keys))):
    r, c = divmod(idx, 4)
    axes[r, c].imshow(results[keys[idx]], cmap='gray')
    axes[r, c].set_title(keys[idx])
    axes[r, c].axis("off")

# Slot terakhir: gambar warna
if len(keys) < 12:
    r, c = divmod(len(keys), 4)
    axes[r, c].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    axes[r, c].set_title("Warna Sintetis RGB")
    axes[r, c].axis("off")

# Sembunyikan slot kosong
for idx in range(len(keys) + 1, 12):
    r, c = divmod(idx, 4)
    axes[r, c].axis("off")

plt.suptitle("Percobaan 20: Pembuatan Citra Sintetis", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "20_citra_sintetis_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# Simpan juga beberapa gambar individu
cv2.imwrite(os.path.join(OUTPUT_DIR, "20_zone_plate.png"), zone)
cv2.imwrite(os.path.join(OUTPUT_DIR, "20_gabor.png"), gabor_vis)
cv2.imwrite(os.path.join(OUTPUT_DIR, "20_warna_sintetis.png"), color_img)

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 20")
print("=" * 60)
print("""
1. np.linspace/meshgrid → gradient linear, radial
2. np.sin/cos → pola sinusoidal, zone plate
3. cv2.getGaborKernel → tekstur Gabor (orientasi+frekuensi)
4. np.random → noise (Gaussian, uniform, salt&pepper)
5. cv2.line/circle → pola geometris (bintang, ring)
6. Citra sintetis berguna untuk testing tanpa kamera
""")
