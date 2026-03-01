"""
==========================================================================
PERCOBAAN 17: TOP HAT DAN BLACK HAT
==========================================================================
- Top Hat = Original - Opening → mendeteksi fitur terang kecil
  di background gelap.
- Black Hat = Closing - Original → mendeteksi fitur gelap kecil
  di background terang.

Fungsi:
- cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel) → top hat
- cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel) → black hat
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

img = cv2.imread(os.path.join(IMAGE_DIR, "dokumen.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 17: TOP HAT DAN BLACK HAT")
print("=" * 60)

# ============================================================
# 1. Top Hat (White Top Hat)
# ============================================================
print("\n--- 1. Top Hat ---")

se = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
# Top Hat = Original - Opening
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
# Verifikasi manual
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)
tophat_manual = cv2.subtract(gray, opening)
diff = cv2.absdiff(tophat, tophat_manual).max()
print(f"  Top Hat max: {tophat.max()}")
print(f"  Manual vs OpenCV: diff={diff}")

# ============================================================
# 2. Black Hat
# ============================================================
print("\n--- 2. Black Hat ---")

# Black Hat = Closing - Original
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
blackhat_manual = cv2.subtract(closing, gray)
diff_b = cv2.absdiff(blackhat, blackhat_manual).max()
print(f"  Black Hat max: {blackhat.max()}")
print(f"  Manual vs OpenCV: diff={diff_b}")

# ============================================================
# 3. Variasi Ukuran Kernel
# ============================================================
print("\n--- 3. Variasi Ukuran Kernel ---")

sizes = [5, 15, 25, 41]
th_results = []
bh_results = []

for s in sizes:
    se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_v)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se_v)
    th_results.append(th)
    bh_results.append(bh)
    print(f"  SE {s}×{s}: TH max={th.max()}, BH max={bh.max()}")

# ============================================================
# 4. Contrast Enhancement dengan Top Hat + Black Hat
# ============================================================
print("\n--- 4. Contrast Enhancement ---")

# Teknik: original + tophat - blackhat
# Mencerahkan fitur terang kecil + menggelapkan fitur gelap kecil
se_en = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
th_en = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_en)
bh_en = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se_en)
enhanced = cv2.add(gray, th_en)
enhanced = cv2.subtract(enhanced, bh_en)
print(f"  Original std: {gray.std():.1f}")
print(f"  Enhanced std: {enhanced.std():.1f}")

# ============================================================
# 5. Top Hat untuk Koreksi Pencahayaan
# ============================================================
print("\n--- 5. Koreksi Pencahayaan ---")

# Buat gambar dengan pencahayaan tidak merata
h, w = gray.shape
gradient = np.tile(np.linspace(0.3, 1.0, w).astype(np.float32), (h, 1))
uneven = (gray.astype(np.float32) * gradient).astype(np.uint8)

# Top hat menghilangkan variasi pencahayaan background
se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
corrected = cv2.morphologyEx(uneven, cv2.MORPH_TOPHAT, se_bg)
# Perkuat hasilnya
corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
print("  Top Hat mengoreksi pencahayaan tidak merata")

# ============================================================
# 6. Top Hat pada Gambar Biner
# ============================================================
print("\n--- 6. Top Hat Biner ---")

biner = cv2.imread(os.path.join(IMAGE_DIR, "biner_noise.png"), cv2.IMREAD_GRAYSCALE)
if biner is not None:
    se_bin = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    th_bin = cv2.morphologyEx(biner, cv2.MORPH_TOPHAT, se_bin)
    bh_bin = cv2.morphologyEx(biner, cv2.MORPH_BLACKHAT, se_bin)
    print(f"  Top Hat biner: {np.sum(th_bin > 0)} piksel (noise kecil)")
    print(f"  Black Hat biner: {np.sum(bh_bin > 0)} piksel (lubang kecil)")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Top hat
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(tophat, cmap='gray')
axes[0, 1].set_title("Top Hat")
axes[0, 1].axis("off")

axes[0, 2].imshow(blackhat, cmap='gray')
axes[0, 2].set_title("Black Hat")
axes[0, 2].axis("off")

axes[0, 3].imshow(enhanced, cmap='gray')
axes[0, 3].set_title("Enhanced (TH-BH)")
axes[0, 3].axis("off")

# Baris 2: Variasi ukuran
for i, (s, th) in enumerate(zip(sizes, th_results)):
    axes[1, i].imshow(th, cmap='gray')
    axes[1, i].set_title(f"TH SE={s}")
    axes[1, i].axis("off")

# Baris 3: Koreksi pencahayaan
axes[2, 0].imshow(uneven, cmap='gray')
axes[2, 0].set_title("Tidak Merata")
axes[2, 0].axis("off")

axes[2, 1].imshow(corrected, cmap='gray')
axes[2, 1].set_title("Top Hat Corrected")
axes[2, 1].axis("off")

if biner is not None:
    axes[2, 2].imshow(th_bin, cmap='gray')
    axes[2, 2].set_title("TH Biner (noise)")
    axes[2, 2].axis("off")

    axes[2, 3].imshow(bh_bin, cmap='gray')
    axes[2, 3].set_title("BH Biner (holes)")
    axes[2, 3].axis("off")

plt.suptitle("Percobaan 17: Top Hat dan Black Hat", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "17_tophat_blackhat_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 17")
print("=" * 60)
print("""
1. Top Hat = Original - Opening → fitur terang kecil
2. Black Hat = Closing - Original → fitur gelap kecil
3. Ukuran SE menentukan 'kecil' relatif terhadap apa
4. Enhanced = Original + Top Hat - Black Hat
5. Top Hat efektif untuk koreksi pencahayaan tidak merata
6. Berguna untuk deteksi objek kecil di background bervariasi
""")
