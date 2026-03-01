"""
==========================================================================
PERCOBAAN 18: REMAPPING GAMBAR
==========================================================================
Remapping mengubah posisi piksel menggunakan peta (map) custom.
Bisa digunakan untuk efek wave, swirl, mirror, dsb.

Fungsi:
- cv2.remap(src, map_x, map_y, interpolation, borderMode) → mapping custom
- np.meshgrid() → membuat grid koordinat
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()
img = cv2.resize(img, (512, 512))
h, w = img.shape[:2]

print("=" * 60)
print("PERCOBAAN 18: REMAPPING GAMBAR")
print("=" * 60)

# Grid koordinat dasar
ix, iy = np.meshgrid(np.arange(w, dtype=np.float32),
                     np.arange(h, dtype=np.float32))

# ============================================================
# 1. Efek Gelombang Horizontal (Wave)
# ============================================================
print("\n--- 1. Efek Gelombang Horizontal ---")

# Amplitudo dan frekuensi gelombang
amplitude = 15.0
frequency = 20.0

# Geser koordinat y berdasarkan sin dari posisi x
map_x_wave = ix.copy()
map_y_wave = iy + amplitude * np.sin(2 * np.pi * ix / frequency)

# Terapkan remap
wave_h = cv2.remap(img, map_x_wave.astype(np.float32),
                   map_y_wave.astype(np.float32), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT_101)
print(f"  Amplitudo={amplitude}, Frekuensi={frequency}")

# ============================================================
# 2. Efek Gelombang Vertikal
# ============================================================
print("\n--- 2. Efek Gelombang Vertikal ---")

# Geser koordinat x berdasarkan sin dari posisi y
map_x_vwave = ix + amplitude * np.sin(2 * np.pi * iy / frequency)
map_y_vwave = iy.copy()

wave_v = cv2.remap(img, map_x_vwave.astype(np.float32),
                   map_y_vwave.astype(np.float32), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT_101)

# ============================================================
# 3. Efek Gelombang Gabungan (Water Ripple)
# ============================================================
print("\n--- 3. Efek Water Ripple ---")

cx, cy = w / 2, h / 2
# Hitung jarak setiap piksel dari pusat
dx = ix - cx
dy = iy - cy
dist = np.sqrt(dx * dx + dy * dy)

# Gelombang lingkaran dari pusat
ripple_amp = 8.0
ripple_freq = 30.0
offset = ripple_amp * np.sin(2 * np.pi * dist / ripple_freq)

# Arah radial
angle = np.arctan2(dy, dx)
map_x_ripple = ix + offset * np.cos(angle)
map_y_ripple = iy + offset * np.sin(angle)

ripple = cv2.remap(img, map_x_ripple.astype(np.float32),
                   map_y_ripple.astype(np.float32), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT_101)

# ============================================================
# 4. Efek Swirl (Puntiran)
# ============================================================
print("\n--- 4. Efek Swirl ---")

# Parameter swirl
swirl_strength = 2.5
swirl_radius = 250.0

# Jarak dari pusat
dist_norm = dist / swirl_radius

# Sudut putar tergantung jarak dari pusat
theta = swirl_strength * np.exp(-dist_norm * dist_norm)

# Putar koordinat
cos_t = np.cos(theta)
sin_t = np.sin(theta)
map_x_swirl = cx + dx * cos_t - dy * sin_t
map_y_swirl = cy + dx * sin_t + dy * cos_t

swirl = cv2.remap(img, map_x_swirl.astype(np.float32),
                  map_y_swirl.astype(np.float32), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT)
print(f"  Swirl strength={swirl_strength}, radius={swirl_radius}")

# ============================================================
# 5. Efek Spherize (Bola)
# ============================================================
print("\n--- 5. Efek Spherize ---")

# Normalisasi koordinat ke [-1,1]
nx = dx / (w / 2)
ny = dy / (h / 2)
r = np.sqrt(nx * nx + ny * ny)

# Hanya ubah area dalam lingkaran
mask_sphere = r < 1.0
# Transformasi spherize (refraksi melalui bola)
theta_s = np.arcsin(np.clip(r, 0, 1))
# Faktor perbesaran di tengah, normal di tepi
new_r = np.where(mask_sphere, np.sin(theta_s * 0.8) / (r + 1e-7), 1.0)

map_x_sphere = (cx + dx * new_r).astype(np.float32)
map_y_sphere = (cy + dy * new_r).astype(np.float32)

sphere = cv2.remap(img, map_x_sphere, map_y_sphere,
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ============================================================
# 6. Efek Mirror Horizontal (Cermin Tengah)
# ============================================================
print("\n--- 6. Efek Mirror ---")

# Cerminkan setengah kanan dari setengah kiri
map_x_mirror = np.where(ix < w / 2, ix, w - 1 - ix).astype(np.float32)
map_y_mirror = iy.copy()

mirror = cv2.remap(img, map_x_mirror, map_y_mirror, cv2.INTER_LINEAR)

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

titles_imgs = [
    ("Original", img),
    ("Wave Horizontal", wave_h),
    ("Wave Vertikal", wave_v),
    ("Water Ripple", ripple),
    ("Swirl", swirl),
    ("Spherize", sphere),
    ("Mirror", mirror),
]

for idx, (title, im) in enumerate(titles_imgs):
    r, c = divmod(idx, 4)
    axes[r, c].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    axes[r, c].set_title(title)
    axes[r, c].axis("off")

axes[1, 3].axis("off")

plt.suptitle("Percobaan 18: Remapping Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "18_remapping_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 18")
print("=" * 60)
print("""
1. cv2.remap() memetakan piksel dari posisi asal ke posisi baru
2. map_x dan map_y mendefinisikan DARI MANA piksel diambil
3. Efek wave: offset sinusoidal pada satu sumbu
4. Efek ripple: gelombang lingkaran dari pusat
5. Efek swirl: rotasi yang bergantung pada jarak dari pusat
6. Efek spherize: distorsi bola menggunakan arcsin
7. Efek mirror: cerminkan koordinat x di tengah
""")
