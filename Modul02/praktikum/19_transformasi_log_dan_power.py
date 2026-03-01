"""
==========================================================================
PERCOBAAN 19: TRANSFORMASI LOG DAN POWER (KONTRAS)
==========================================================================
Transformasi intensitas mengubah distribusi kecerahan piksel.
- Log transform: memperluas detail gelap, kompres terang
- Power-law (gamma): gamma<1 terang, gamma>1 gelap
- Piecewise-linear: penyesuaian kontras linear per segmen

Fungsi:
- cv2.LUT(src, lut) → lookup table untuk mapping intensitas
- np.log() → transformasi logaritma
- cv2.normalize() → normalisasi range
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
# Konversi ke grayscale untuk transformasi intensitas
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 19: TRANSFORMASI LOG DAN POWER")
print("=" * 60)

# ============================================================
# 1. Transformasi Logaritma
# ============================================================
print("\n--- 1. Transformasi Logaritma ---")

# s = c * log(1 + r)
# c adalah konstanta untuk menskala ke range [0, 255]
c_log = 255 / np.log(1 + 255)
# Hitung log transform
log_transform = c_log * np.log(1 + gray.astype(np.float64))
log_img = np.clip(log_transform, 0, 255).astype(np.uint8)
print(f"  Konstanta c = {c_log:.2f}")
print(f"  Original range: [{gray.min()}, {gray.max()}]")
print(f"  Log range: [{log_img.min()}, {log_img.max()}]")

# ============================================================
# 2. Inverse Log Transform
# ============================================================
print("\n--- 2. Inverse Log Transform ---")

# s = c * (exp(r/c_inv) - 1)
c_inv = 255 / (np.exp(255 / 50) - 1) if False else 50
# Normalisasi input ke [0, c_inv]
r_norm = gray.astype(np.float64) / 255 * 5
inv_log = np.exp(r_norm) - 1
inv_log = cv2.normalize(inv_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
print(f"  Ekspansi area terang, kompresi area gelap")

# ============================================================
# 3. Power-Law (Gamma) Transform
# ============================================================
print("\n--- 3. Power-Law (Gamma) Transform ---")

gamma_values = [0.3, 0.5, 1.0, 1.5, 2.5, 4.0]
gamma_results = []

for gamma in gamma_values:
    # Buat LUT: s = 255 * (r/255)^gamma
    lut = np.array([((i / 255.0) ** gamma) * 255
                    for i in range(256)]).astype(np.uint8)
    # Terapkan LUT
    result = cv2.LUT(gray, lut)
    gamma_results.append(result)
    print(f"  gamma={gamma}: mean={result.mean():.1f}")

# ============================================================
# 4. Piecewise-Linear Transform (Contrast Stretching)
# ============================================================
print("\n--- 4. Contrast Stretching ---")

# Definisikan titik-titik breakpoint
# (r1, s1) dan (r2, s2)
r1, s1 = 70, 20
r2, s2 = 180, 240

# Buat LUT piecewise-linear
lut_pw = np.zeros(256, dtype=np.uint8)
for i in range(256):
    if i < r1:
        # Segmen pertama: slope rendah (kompres gelap)
        lut_pw[i] = int(s1 / r1 * i)
    elif i <= r2:
        # Segmen kedua: slope tinggi (stretch mid-range)
        lut_pw[i] = int(s1 + (s2 - s1) / (r2 - r1) * (i - r1))
    else:
        # Segmen ketiga: slope rendah (kompres terang)
        lut_pw[i] = int(s2 + (255 - s2) / (255 - r2) * (i - r2))

pw_result = cv2.LUT(gray, lut_pw)
print(f"  Breakpoints: ({r1},{s1}) dan ({r2},{s2})")

# ============================================================
# 5. Bit-Plane Slicing
# ============================================================
print("\n--- 5. Bit-Plane Slicing ---")

bit_planes = []
for bit in range(8):
    # Ekstrak bit ke-n dari setiap piksel
    plane = ((gray >> bit) & 1) * 255
    bit_planes.append(plane.astype(np.uint8))
    print(f"  Bit {bit}: jumlah piksel=1 → {np.sum(plane > 0)}")

# ============================================================
# 6. Kurva Transformasi
# ============================================================
fig_curve, ax_curve = plt.subplots(1, 1, figsize=(8, 6))

# Plot kurva transformasi
r_vals = np.arange(256)

# Log transform curve
log_curve = c_log * np.log(1 + r_vals.astype(np.float64))
ax_curve.plot(r_vals, np.clip(log_curve, 0, 255), 'b-', label='Log', linewidth=2)

# Gamma curves
for gamma in [0.3, 1.0, 2.5]:
    gamma_curve = 255 * (r_vals / 255.0) ** gamma
    ax_curve.plot(r_vals, gamma_curve, '--', label=f'γ={gamma}', linewidth=1.5)

# Piecewise linear
ax_curve.plot(r_vals, lut_pw, 'r-', label='Piecewise', linewidth=2)

# Garis identitas
ax_curve.plot([0, 255], [0, 255], 'k:', alpha=0.5, label='Identitas')

ax_curve.set_xlabel("Input Intensitas (r)")
ax_curve.set_ylabel("Output Intensitas (s)")
ax_curve.set_title("Kurva Transformasi Intensitas")
ax_curve.legend()
ax_curve.grid(True, alpha=0.3)
ax_curve.set_xlim(0, 255)
ax_curve.set_ylim(0, 255)

path_curve = os.path.join(OUTPUT_DIR, "19_kurva_transformasi.png")
fig_curve.savefig(path_curve, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path_curve}")

# ============================================================
# 7. Visualisasi Hasil Utama
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Baris 1
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(log_img, cmap='gray')
axes[0, 1].set_title("Log Transform")
axes[0, 1].axis("off")

axes[0, 2].imshow(inv_log, cmap='gray')
axes[0, 2].set_title("Inverse Log")
axes[0, 2].axis("off")

axes[0, 3].imshow(pw_result, cmap='gray')
axes[0, 3].set_title("Piecewise Linear")
axes[0, 3].axis("off")

# Baris 2: gamma
for i, (gamma, res) in enumerate(zip(gamma_values[:4], gamma_results[:4])):
    axes[1, i].imshow(res, cmap='gray')
    axes[1, i].set_title(f"Gamma={gamma}")
    axes[1, i].axis("off")

plt.suptitle("Percobaan 19: Transformasi Log dan Power", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "19_log_power_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 19")
print("=" * 60)
print("""
1. Log transform: s = c * log(1+r) → memperluas area gelap
2. Inverse log: ekspansi area terang, kompresi area gelap
3. Power-law (gamma): s = c * r^γ
   - γ < 1 → mencerahkan (ekspansi gelap)
   - γ > 1 → menggelapkan (ekspansi terang)
4. Piecewise-linear: kontrol per segmen intensitas
5. Bit-plane slicing: menunjukkan kontribusi setiap bit
6. cv2.LUT() efisien untuk transformasi intensity mapping
""")
