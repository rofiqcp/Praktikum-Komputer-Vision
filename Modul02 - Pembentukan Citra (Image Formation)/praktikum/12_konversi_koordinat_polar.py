"""
==========================================================================
PERCOBAAN 12: KONVERSI KOORDINAT POLAR
==========================================================================
Mengkonversi gambar antara koordinat Kartesian (x,y) dan Polar (r,θ).
Berguna untuk analisis pola radial, clock reading, dll.

Fungsi:
- cv2.warpPolar(src, dsize, center, maxRadius, flags)
  flags: cv2.WARP_POLAR_LINEAR atau cv2.WARP_POLAR_LOG
  + cv2.WARP_INVERSE_MAP untuk inverse
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

img = cv2.imread(os.path.join(IMAGE_DIR, "baboon.jpg"))
if img is None:
    img = cv2.imread(os.path.join(IMAGE_DIR, "baboon.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()
img = cv2.resize(img, (400, 400))
h, w = img.shape[:2]
center = (w // 2, h // 2)

print("=" * 60)
print("PERCOBAAN 12: KOORDINAT POLAR")
print("=" * 60)

# ============================================================
# 1. Kartesian → Polar (linear)
# ============================================================
print("\n--- 1. Kartesian → Polar Linear ---")

max_radius = min(h, w) // 2
# cv2.warpPolar: konversi koordinat
# dsize: ukuran output (width=radius, height=angle)
# WARP_POLAR_LINEAR: mapping linear r
polar_linear = cv2.warpPolar(img, (400, 400), center, max_radius,
                              cv2.WARP_POLAR_LINEAR)
print(f"  Pusat: {center}, Max radius: {max_radius}")

# ============================================================
# 2. Kartesian → Polar (logaritmic)
# ============================================================
print("\n--- 2. Kartesian → Polar Log ---")

# WARP_POLAR_LOG: mapping logaritmik (detail pusat lebih jelas)
polar_log = cv2.warpPolar(img, (400, 400), center, max_radius,
                           cv2.WARP_POLAR_LOG)
print("  Log-polar: detail pusat diperbesar")

# ============================================================
# 3. Polar → Kartesian (inverse)
# ============================================================
print("\n--- 3. Polar → Kartesian ---")

# WARP_INVERSE_MAP: konversi polar → kartesian
inv_linear = cv2.warpPolar(polar_linear, (w, h), center, max_radius,
                            cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
diff = np.mean(cv2.absdiff(img, inv_linear))
print(f"  Perbedaan roundtrip: {diff:.2f}")

# ============================================================
# 4. Rotasi sebagai translasi di polar
# ============================================================
print("\n--- 4. Rotasi via Polar ---")

# Di koordinat polar, rotasi = pergeseran vertikal (shift θ)
shift = 100  # Shift 100 piksel di sumbu θ
polar_shifted = np.roll(polar_linear, shift, axis=0)
# Konversi balik → gambar sudah terrotasi
rotated_via_polar = cv2.warpPolar(polar_shifted, (w, h), center, max_radius,
                                   cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
print(f"  Rotasi via shift polar: {shift} piksel")

# ============================================================
# 5. Pusat polar berbeda
# ============================================================
print("\n--- 5. Pusat Berbeda ---")

pusat_list = [(100, 100), (200, 200), (300, 300), (200, 100)]
hasil_pusat = {}
for p in pusat_list:
    polar_p = cv2.warpPolar(img, (400, 400), p, max_radius,
                             cv2.WARP_POLAR_LINEAR)
    hasil_pusat[p] = polar_p
    print(f"  Pusat {p}")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 1].imshow(cv2.cvtColor(polar_linear, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Polar Linear")
axes[0, 2].imshow(cv2.cvtColor(polar_log, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Polar Log")
axes[0, 3].imshow(cv2.cvtColor(inv_linear, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Inverse (recovered)")

axes[1, 0].imshow(cv2.cvtColor(rotated_via_polar, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Rotasi via Polar")
for i, (p, im) in enumerate(list(hasil_pusat.items())[:3]):
    axes[1, i+1].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    axes[1, i+1].set_title(f"Pusat {p}")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("Percobaan 12: Koordinat Polar", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "12_koordinat_polar_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
