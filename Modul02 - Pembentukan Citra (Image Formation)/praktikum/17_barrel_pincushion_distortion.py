"""
==========================================================================
PERCOBAAN 17: BARREL DAN PINCUSHION DISTORTION
==========================================================================
Distorsi barrel menyebabkan garis lurus melengkung ke luar, sedangkan
pincushion melengkung ke dalam. Efek ini umum pada lensa wide-angle.

Fungsi:
- cv2.remap(src, map_x, map_y, interpolation) → distorsi custom
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

# Gunakan grid agar distorsi mudah dilihat
img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()
img = cv2.resize(img, (512, 512))

print("=" * 60)
print("PERCOBAAN 17: BARREL DAN PINCUSHION DISTORTION")
print("=" * 60)

# ============================================================
# 1. Fungsi untuk membuat distorsi radial
# ============================================================
def radial_distortion(img, k1, k2=0.0):
    """
    Membuat distorsi radial barrel/pincushion.
    k1 > 0 → barrel, k1 < 0 → pincushion.
    """
    h, w = img.shape[:2]
    # Koordinat ternormalisasi berpusat di tengah
    cx, cy = w / 2, h / 2
    # Membuat grid koordinat piksel
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for j in range(h):
        for i in range(w):
            # Normalisasi koordinat ke [-1, 1]
            x = (i - cx) / cx
            y = (j - cy) / cy
            # Jarak dari pusat (kuadrat)
            r2 = x * x + y * y
            r4 = r2 * r2
            # Faktor distorsi
            factor = 1 + k1 * r2 + k2 * r4
            # Koordinat setelah distorsi
            map_x[j, i] = cx + x * factor * cx
            map_y[j, i] = cy + y * factor * cy

    # Terapkan remap dengan interpolasi bilinear
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# ============================================================
# 2. Versi cepat menggunakan vectorized NumPy
# ============================================================
def radial_distortion_fast(img, k1, k2=0.0):
    """Versi vectorized (cepat) dari distorsi radial."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    # Meshgrid menghasilkan semua koordinat sekaligus
    ix, iy = np.meshgrid(np.arange(w), np.arange(h))
    # Normalisasi ke [-1, 1]
    x = (ix.astype(np.float32) - cx) / cx
    y = (iy.astype(np.float32) - cy) / cy
    # Hitung radius kuadrat
    r2 = x * x + y * y
    r4 = r2 * r2
    # Faktor distorsi
    factor = 1 + k1 * r2 + k2 * r4
    # Koordinat terdistorsi
    map_x = (cx + x * factor * cx).astype(np.float32)
    map_y = (cy + y * factor * cy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)

# ============================================================
# 3. Barrel Distortion (k1 > 0)
# ============================================================
print("\n--- 3. Barrel Distortion ---")
# Berbagai kekuatan barrel
barrel_results = []
k_values_barrel = [0.1, 0.3, 0.6]
for k1 in k_values_barrel:
    res = radial_distortion_fast(img, k1)
    barrel_results.append(res)
    print(f"  k1={k1}: barrel distortion diterapkan")

# ============================================================
# 4. Pincushion Distortion (k1 < 0)
# ============================================================
print("\n--- 4. Pincushion Distortion ---")
pincushion_results = []
k_values_pin = [-0.1, -0.3, -0.6]
for k1 in k_values_pin:
    res = radial_distortion_fast(img, k1)
    pincushion_results.append(res)
    print(f"  k1={k1}: pincushion distortion diterapkan")

# ============================================================
# 5. Mustache Distortion (k1 > 0, k2 < 0)
# ============================================================
print("\n--- 5. Mustache Distortion ---")
# Kombinasi k1 dan k2 menghasilkan efek mustache
mustache = radial_distortion_fast(img, 0.5, -0.3)
print("  k1=0.5, k2=-0.3: mustache distortion")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Baris 1: barrel
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original (Grid)")
axes[0, 0].axis("off")

for i, (res, k) in enumerate(zip(barrel_results, k_values_barrel)):
    ax = axes[0, i] if i > 0 else axes[0, 1]
    if i == 0:
        ax = axes[0, 1]
    elif i == 1:
        ax = axes[0, 2]
    else:
        ax = axes[1, 0]
    ax.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Barrel k1={k}")
    ax.axis("off")

# Baris 2: pincushion
for i, (res, k) in enumerate(zip(pincushion_results, k_values_pin)):
    ax = axes[1, i + 1] if i < 2 else axes[2, 0]
    if i == 0:
        ax = axes[1, 1]
    elif i == 1:
        ax = axes[1, 2]
    else:
        ax = axes[2, 0]
    ax.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Pincushion k1={k}")
    ax.axis("off")

# Mustache
axes[2, 1].imshow(cv2.cvtColor(mustache, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("Mustache (k1=0.5, k2=-0.3)")
axes[2, 1].axis("off")

axes[2, 2].axis("off")

plt.suptitle("Percobaan 17: Barrel & Pincushion Distortion", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "17_barrel_pincushion_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 17")
print("=" * 60)
print("""
1. Barrel distortion (k1 > 0) → garis lurus melengkung keluar
2. Pincushion distortion (k1 < 0) → garis lurus melengkung ke dalam
3. Mustache distortion → kombinasi k1 dan k2 positif/negatif
4. cv2.remap() menerapkan mapping piksel custom
5. np.meshgrid() mempercepat komputasi (vectorized)
""")
