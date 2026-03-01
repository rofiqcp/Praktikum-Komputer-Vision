"""
==========================================================================
PERCOBAAN 10: GAMMA CORRECTION
==========================================================================
Gamma correction memperbaiki brightness non-linear.
Rumus: output = 255 × (input/255)^gamma
- gamma < 1 → gambar lebih terang (shadows diperkuat)
- gamma > 1 → gambar lebih gelap (shadows dipergelap)
- gamma = 1 → tidak berubah

Fungsi: cv2.LUT (Look-Up Table) untuk penerapan cepat
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
img = cv2.resize(img, (300, 300))

print("=" * 60)
print("PERCOBAAN 10: GAMMA CORRECTION")
print("=" * 60)

# ============================================================
# 1. Fungsi gamma correction
# ============================================================
def gamma_correction(img, gamma):
    """Terapkan gamma correction: O = 255×(I/255)^γ"""
    # Membuat look-up table (256 nilai)
    inv_gamma = 1.0 / gamma
    # np.arange(256) → [0, 1, 2, ..., 255]
    # Normalisasi ke [0, 1], pangkat gamma, kembali ke [0, 255]
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    # cv2.LUT menerapkan tabel ke setiap piksel (sangat cepat)
    return cv2.LUT(img, table)

# ============================================================
# 2. Berbagai nilai gamma
# ============================================================
print("\n--- 1. Berbagai Nilai Gamma ---")
gamma_values = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
hasil = {}
for g in gamma_values:
    hasil[g] = gamma_correction(img, g)
    mean_val = hasil[g].mean()
    print(f"  γ={g:.1f} → mean brightness: {mean_val:.1f}")

# ============================================================
# 3. Kurva gamma
# ============================================================
print("\n--- 2. Kurva Gamma ---")
x = np.linspace(0, 1, 256)
kurva = {}
for g in [0.3, 0.5, 1.0, 1.5, 2.5]:
    kurva[g] = x ** (1.0 / g)
    print(f"  γ={g:.1f}: mid-tone {x[128]:.2f} → {kurva[g][128]:.2f}")

# ============================================================
# 4. Auto gamma (berdasarkan brightness rata-rata)
# ============================================================
print("\n--- 3. Auto Gamma ---")

def auto_gamma(img, target_mean=128):
    """Hitung gamma otomatis berdasarkan mean brightness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    if mean_val == 0:
        return img, 1.0
    # Hitung gamma yang diperlukan
    gamma = np.log(target_mean / 255.0) / np.log(mean_val / 255.0)
    gamma = np.clip(gamma, 0.1, 5.0)
    return gamma_correction(img, gamma), gamma

img_auto, g_auto = auto_gamma(img)
print(f"  Auto gamma: γ={g_auto:.2f}")
print(f"  Mean sebelum: {cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean():.1f}")
print(f"  Mean sesudah: {cv2.cvtColor(img_auto, cv2.COLOR_BGR2GRAY).mean():.1f}")

# ============================================================
# 5. Gamma per channel
# ============================================================
print("\n--- 4. Gamma Per Channel ---")
b, g_ch, r = cv2.split(img)
b_g = gamma_correction(cv2.merge([b, b, b]), 0.5)[:,:,0]
g_g = gamma_correction(cv2.merge([g_ch, g_ch, g_ch]), 1.5)[:,:,0]
r_g = gamma_correction(cv2.merge([r, r, r]), 1.0)[:,:,0]
img_per_ch = cv2.merge([b_g, g_g, r_g])
print("  B: γ=0.5, G: γ=1.5, R: γ=1.0")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
for i, g in enumerate([0.3, 0.5, 1.5]):
    axes[0, i+1].imshow(cv2.cvtColor(hasil[g if g in hasil else gamma_values[i]], cv2.COLOR_BGR2RGB))
    axes[0, i+1].set_title(f"γ={g}")

# Kurva gamma
axes[1, 0].set_title("Kurva Gamma")
for g, y in kurva.items():
    axes[1, 0].plot(x * 255, y * 255, label=f'γ={g}')
axes[1, 0].plot([0, 255], [0, 255], 'k--', alpha=0.3)
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_xlabel("Input")
axes[1, 0].set_ylabel("Output")

axes[1, 1].imshow(cv2.cvtColor(hasil[2.0], cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("γ=2.0")
axes[1, 2].imshow(cv2.cvtColor(img_auto, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f"Auto γ={g_auto:.2f}")
axes[1, 3].imshow(cv2.cvtColor(img_per_ch, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Per-Channel γ")

for ax in axes.flat:
    if not ax.has_data():
        ax.axis("off")
    elif ax != axes[1, 0]:
        ax.axis("off")

plt.suptitle("Percobaan 10: Gamma Correction", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "10_gamma_correction_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
