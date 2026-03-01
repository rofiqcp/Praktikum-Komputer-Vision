"""
==========================================================================
PERCOBAAN 08: KONVOLUSI DAN FILTER2D (CUSTOM KERNEL)
==========================================================================
Konvolusi 2D menghitung output setiap piksel sebagai weighted sum
dari piksel tetangganya berdasarkan kernel/filter yang diberikan.

Fungsi:
- cv2.filter2D(src, ddepth, kernel) → konvolusi dengan kernel custom
- cv2.getStructuringElement() → kernel morfologi bawaan
- np.ones() / np.array() → membuat kernel manual
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

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 08: KONVOLUSI DAN FILTER2D")
print("=" * 60)

# ============================================================
# 1. Box Filter (Averaging / Mean)
# ============================================================
print("\n--- 1. Box Filter ---")

# Kernel averaging 5×5: semua elemen = 1/25
kernel_box = np.ones((5, 5), dtype=np.float32) / 25
# Terapkan konvolusi (ddepth=-1 artinya sama dengan input)
box_result = cv2.filter2D(gray, -1, kernel_box)
print(f"  Kernel 5×5, setiap elemen = {1/25:.4f}")

# ============================================================
# 2. Kernel Sharpening (Penajaman)
# ============================================================
print("\n--- 2. Kernel Sharpening ---")

# Kernel sharpening: center tinggi, tetangga negatif
kernel_sharp = np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]], dtype=np.float32)
sharp_result = cv2.filter2D(gray, -1, kernel_sharp)
print("  Kernel 3×3: center=5, cross=-1")

# ============================================================
# 3. Kernel Edge Detection
# ============================================================
print("\n--- 3. Kernel Edge Detection ---")

# Horizontal edge
kernel_h = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]], dtype=np.float32)
edge_h = cv2.filter2D(gray, cv2.CV_64F, kernel_h)
edge_h = np.abs(edge_h).astype(np.uint8)

# Vertical edge
kernel_v = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)
edge_v = cv2.filter2D(gray, cv2.CV_64F, kernel_v)
edge_v = np.abs(edge_v).astype(np.uint8)

# Laplacian kernel
kernel_lap = np.array([[ 0,  1,  0],
                        [ 1, -4,  1],
                        [ 0,  1,  0]], dtype=np.float32)
edge_lap = cv2.filter2D(gray, cv2.CV_64F, kernel_lap)
edge_lap = np.abs(edge_lap).astype(np.uint8)
print("  Kernel: horizontal, vertikal, Laplacian")

# ============================================================
# 4. Kernel Emboss (Relief)
# ============================================================
print("\n--- 4. Kernel Emboss ---")

kernel_emboss = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]], dtype=np.float32)
emboss = cv2.filter2D(gray, -1, kernel_emboss)
# Tambah offset 128 agar terlihat seperti relief
emboss = cv2.add(emboss, 128)
print("  Efek timbul (emboss) dengan offset 128")

# ============================================================
# 5. Gabungan Kernel Berbagai Ukuran
# ============================================================
print("\n--- 5. Kernel Berbagai Ukuran ---")

sizes = [3, 5, 7, 11]
blur_results = []
for s in sizes:
    # Buat kernel averaging dengan ukuran berbeda
    kernel = np.ones((s, s), dtype=np.float32) / (s * s)
    result = cv2.filter2D(img, -1, kernel)
    blur_results.append(result)
    print(f"  Box {s}×{s}: blur level semakin kuat")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Filter dasar
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(box_result, cmap='gray')
axes[0, 1].set_title("Box Filter 5×5")
axes[0, 1].axis("off")

axes[0, 2].imshow(sharp_result, cmap='gray')
axes[0, 2].set_title("Sharpening")
axes[0, 2].axis("off")

axes[0, 3].imshow(emboss, cmap='gray')
axes[0, 3].set_title("Emboss")
axes[0, 3].axis("off")

# Baris 2: Edge detection
axes[1, 0].imshow(edge_h, cmap='gray')
axes[1, 0].set_title("Edge Horizontal")
axes[1, 0].axis("off")

axes[1, 1].imshow(edge_v, cmap='gray')
axes[1, 1].set_title("Edge Vertical")
axes[1, 1].axis("off")

axes[1, 2].imshow(edge_lap, cmap='gray')
axes[1, 2].set_title("Laplacian")
axes[1, 2].axis("off")

# Gabung horizontal + vertikal
edge_combined = cv2.addWeighted(edge_h, 0.5, edge_v, 0.5, 0)
axes[1, 3].imshow(edge_combined, cmap='gray')
axes[1, 3].set_title("H + V Combined")
axes[1, 3].axis("off")

# Baris 3: Berbagai ukuran box
for i, (s, res) in enumerate(zip(sizes, blur_results)):
    axes[2, i].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    axes[2, i].set_title(f"Box {s}×{s}")
    axes[2, i].axis("off")

plt.suptitle("Percobaan 08: Konvolusi dan Filter2D", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "08_filter2d_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 08")
print("=" * 60)
print("""
1. cv2.filter2D(src, ddepth, kernel) → konvolusi 2D
2. Box filter: averaging (blur), semua elemen = 1/n²
3. Sharpening: center positif besar, tetangga negatif
4. Edge detection: selisih piksel tetangga (derivatif)
5. Emboss: asimetris, memberikan efek relief/timbul
6. Ukuran kernel lebih besar → efek lebih kuat
""")
