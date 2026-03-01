"""
==========================================================================
PERCOBAAN 04: GAMMA CORRECTION
==========================================================================
Gamma correction mengubah kecerahan non-linear: g = f^γ
- γ < 1 → mencerahkan area gelap
- γ > 1 → menggelapkan area terang
- γ = 1 → tidak berubah (identitas)

Fungsi:
- cv2.LUT(src, lut) → lookup table mapping intensitas
- np.power() → operasi pangkat per piksel
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

img = cv2.imread(os.path.join(IMAGE_DIR, "nature.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!"); exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 04: GAMMA CORRECTION")
print("=" * 60)

# ============================================================
# 1. Gamma Correction Menggunakan LUT
# ============================================================
print("\n--- 1. Gamma dengan LUT ---")

def apply_gamma(image, gamma):
    """Terapkan gamma correction menggunakan lookup table."""
    # Buat tabel: output = 255 * (input/255)^gamma
    table = np.array([(i / 255.0) ** gamma * 255
                      for i in range(256)], dtype=np.uint8)
    # Terapkan LUT
    return cv2.LUT(image, table)

gamma_values = [0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0]
gamma_results = []

for g in gamma_values:
    result = apply_gamma(gray, g)
    gamma_results.append(result)
    print(f"  γ={g:.1f}: mean={result.mean():.1f}, std={result.std():.1f}")

# ============================================================
# 2. Gamma pada Gambar Berwarna
# ============================================================
print("\n--- 2. Gamma pada Gambar Berwarna ---")

gamma_bright = apply_gamma(img, 0.5)
gamma_dark = apply_gamma(img, 2.0)
print(f"  γ=0.5 (terang): mean={gamma_bright.mean():.1f}")
print(f"  γ=2.0 (gelap): mean={gamma_dark.mean():.1f}")

# ============================================================
# 3. Auto Gamma (berdasarkan mean)
# ============================================================
print("\n--- 3. Auto Gamma ---")

def auto_gamma(image, target_mean=127):
    """Hitung gamma optimal berdasarkan target mean."""
    mean = np.mean(image) / 255.0
    if mean <= 0:
        return image, 1.0
    # Hitung gamma yang membuat mean mendekati target
    # target = mean^gamma → gamma = log(target)/log(mean)
    target = target_mean / 255.0
    gamma = np.log(target) / np.log(mean + 1e-7)
    gamma = np.clip(gamma, 0.1, 5.0)
    return apply_gamma(image, gamma), gamma

# Buat gambar gelap dan terang
dark_img = cv2.convertScaleAbs(gray, alpha=0.3, beta=0)
bright_img = cv2.convertScaleAbs(gray, alpha=1.0, beta=100)

auto_dark, g_dark = auto_gamma(dark_img)
auto_bright, g_bright = auto_gamma(bright_img)
print(f"  Gelap (mean={dark_img.mean():.0f}): auto γ={g_dark:.2f}")
print(f"  Terang (mean={bright_img.mean():.0f}): auto γ={g_bright:.2f}")

# ============================================================
# 4. Kurva Gamma
# ============================================================
print("\n--- 4. Kurva Gamma ---")

fig_curve, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 255, 256)
for g in [0.2, 0.5, 1.0, 1.5, 2.5, 4.0]:
    y = 255 * (x / 255) ** g
    ax.plot(x, y, label=f'γ={g}', linewidth=2)
ax.plot([0, 255], [0, 255], 'k--', alpha=0.3, label='identitas')
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.set_title("Kurva Gamma")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)

path_curve = os.path.join(OUTPUT_DIR, "04_gamma_kurva.png")
fig_curve.savefig(path_curve, dpi=150, bbox_inches="tight")
print(f"  [OUTPUT] {path_curve}")

# ============================================================
# 5. Per-Channel Gamma
# ============================================================
print("\n--- 5. Per-Channel Gamma ---")

b, g_ch, r = cv2.split(img)
# Gamma berbeda per channel
r_gamma = apply_gamma(r, 0.7)  # Red lebih terang
g_gamma = apply_gamma(g_ch, 1.0)  # Green tetap
b_gamma = apply_gamma(b, 1.5)  # Blue lebih gelap
per_ch = cv2.merge([b_gamma, g_gamma, r_gamma])
print("  R: γ=0.7, G: γ=1.0, B: γ=1.5")

# ============================================================
# 6. Visualisasi Utama
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Variasi gamma grayscale
for i, (g, res) in enumerate(zip([0.2, 0.7, 1.0, 2.5],
                                  [gamma_results[0], gamma_results[2],
                                   gamma_results[3], gamma_results[5]])):
    axes[0, i].imshow(res, cmap='gray')
    axes[0, i].set_title(f"γ={g}")
    axes[0, i].axis("off")

# Baris 2: Warna
axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Original Warna")
axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(gamma_bright, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("γ=0.5 (Terang)")
axes[1, 1].axis("off")

axes[1, 2].imshow(cv2.cvtColor(gamma_dark, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("γ=2.0 (Gelap)")
axes[1, 2].axis("off")

axes[1, 3].imshow(cv2.cvtColor(per_ch, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Per-Channel Gamma")
axes[1, 3].axis("off")

# Baris 3: Auto gamma
axes[2, 0].imshow(dark_img, cmap='gray')
axes[2, 0].set_title("Gelap")
axes[2, 0].axis("off")

axes[2, 1].imshow(auto_dark, cmap='gray')
axes[2, 1].set_title(f"Auto γ={g_dark:.2f}")
axes[2, 1].axis("off")

axes[2, 2].imshow(bright_img, cmap='gray')
axes[2, 2].set_title("Terang")
axes[2, 2].axis("off")

axes[2, 3].imshow(auto_bright, cmap='gray')
axes[2, 3].set_title(f"Auto γ={g_bright:.2f}")
axes[2, 3].axis("off")

plt.suptitle("Percobaan 04: Gamma Correction", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "04_gamma_correction_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 04")
print("=" * 60)
print("""
1. Gamma correction: g = 255 * (f/255)^γ
2. γ < 1 → mencerahkan (ekspansi area gelap)
3. γ > 1 → menggelapkan (ekspansi area terang)
4. cv2.LUT() efisien: pre-compute tabel 256 nilai
5. Auto gamma: γ = log(target)/log(mean)
6. Per-channel gamma untuk koreksi warna selektif
""")
