"""
==========================================================================
PERCOBAAN 01: BRIGHTNESS DAN CONTRAST
==========================================================================
Mengubah kecerahan (brightness) dan kontras gambar menggunakan
operasi titik: g(x,y) = α·f(x,y) + β
- α (alpha/gain): mengontrol kontras
- β (beta/bias): mengontrol brightness

Fungsi:
- cv2.convertScaleAbs(src, alpha, beta) → ubah brightness/kontras
- cv2.normalize(src, dst, a, b, norm_type) → normalisasi range
- cv2.addWeighted(src1, alpha, src2, beta, gamma) → blending
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

img = cv2.imread(os.path.join(IMAGE_DIR, "kota.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!"); exit()

print("=" * 60)
print("PERCOBAAN 01: BRIGHTNESS DAN CONTRAST")
print("=" * 60)

# ============================================================
# 1. Mengubah Brightness (β)
# ============================================================
print("\n--- 1. Mengubah Brightness ---")

beta_values = [-80, -40, 0, 40, 80]
bright_results = []

for beta in beta_values:
    # alpha=1.0 (kontras tetap), beta menggeser kecerahan
    result = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
    bright_results.append(result)
    mean_val = result.mean()
    print(f"  β={beta:+4d}: mean intensitas = {mean_val:.1f}")

# ============================================================
# 2. Mengubah Contrast (α)
# ============================================================
print("\n--- 2. Mengubah Contrast ---")

alpha_values = [0.5, 0.8, 1.0, 1.5, 2.0]
contrast_results = []

for alpha in alpha_values:
    # beta=0 (brightness tetap), alpha mengubah kontras
    result = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    contrast_results.append(result)
    std_val = result.std()
    print(f"  α={alpha:.1f}: std deviasi = {std_val:.1f}")

# ============================================================
# 3. Kombinasi Brightness + Contrast
# ============================================================
print("\n--- 3. Kombinasi α dan β ---")

# Kontras tinggi + sedikit lebih terang
combo1 = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
# Kontras rendah + gelap
combo2 = cv2.convertScaleAbs(img, alpha=0.6, beta=-20)
print("  Combo1: α=1.5, β=+30 (kontras tinggi, terang)")
print("  Combo2: α=0.6, β=-20 (kontras rendah, gelap)")

# ============================================================
# 4. Auto Brightness/Contrast (normalisasi)
# ============================================================
print("\n--- 4. Auto Brightness/Contrast ---")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Hitung percentile untuk menghindari outlier
lo = np.percentile(gray, 1)
hi = np.percentile(gray, 99)
# Hitung alpha dan beta optimal
alpha_auto = 255.0 / (hi - lo)
beta_auto = -lo * alpha_auto
auto_result = cv2.convertScaleAbs(img, alpha=alpha_auto, beta=beta_auto)
print(f"  Percentile: [{lo:.0f}, {hi:.0f}]")
print(f"  Auto α={alpha_auto:.2f}, β={beta_auto:.1f}")

# ============================================================
# 5. Menggunakan NumPy manual
# ============================================================
print("\n--- 5. Manual NumPy ---")

alpha_m = 1.3
beta_m = 25
# Konversi ke float, terapkan formula, clip ke [0,255]
manual = img.astype(np.float64) * alpha_m + beta_m
manual = np.clip(manual, 0, 255).astype(np.uint8)
# Bandingkan dengan convertScaleAbs
opencv_res = cv2.convertScaleAbs(img, alpha=alpha_m, beta=beta_m)
diff = cv2.absdiff(manual, opencv_res).mean()
print(f"  Perbedaan manual vs OpenCV: {diff:.4f}")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Brightness
axes[0, 0].imshow(cv2.cvtColor(bright_results[0], cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("β=-80 (Gelap)")
axes[0, 0].axis("off")

axes[0, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Original (β=0)")
axes[0, 1].axis("off")

axes[0, 2].imshow(cv2.cvtColor(bright_results[4], cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("β=+80 (Terang)")
axes[0, 2].axis("off")

# Contrast
axes[1, 0].imshow(cv2.cvtColor(contrast_results[0], cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("α=0.5 (Low)")
axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(contrast_results[2], cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("α=1.0 (Normal)")
axes[1, 1].axis("off")

axes[1, 2].imshow(cv2.cvtColor(contrast_results[4], cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("α=2.0 (High)")
axes[1, 2].axis("off")

# Combo + Auto
axes[2, 0].imshow(cv2.cvtColor(combo1, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("α=1.5, β=+30")
axes[2, 0].axis("off")

axes[2, 1].imshow(cv2.cvtColor(combo2, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("α=0.6, β=-20")
axes[2, 1].axis("off")

axes[2, 2].imshow(cv2.cvtColor(auto_result, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title("Auto Brightness/Contrast")
axes[2, 2].axis("off")

plt.suptitle("Percobaan 01: Brightness dan Contrast", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "01_brightness_contrast_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 01")
print("=" * 60)
print("""
1. g(x,y) = α·f(x,y) + β  →  brightness (β) & kontras (α)
2. cv2.convertScaleAbs() → cara cepat, auto-clip ke [0,255]
3. α > 1 meningkatkan kontras, α < 1 menurunkan
4. β > 0 mencerahkan, β < 0 menggelapkan
5. Auto brightness: gunakan percentile untuk α dan β optimal
""")
