"""
==========================================================================
PERCOBAAN 15: SPLITTING DAN MERGING CHANNEL WARNA
==========================================================================
Program ini mempelajari cara memisahkan dan menggabungkan kembali
channel warna pada gambar. Berguna untuk pemrosesan per-channel.

Fungsi utama:
- cv2.split(img) → Memisahkan gambar menjadi channel-channel individual
- cv2.merge([ch1, ch2, ch3]) → Menggabungkan channel kembali
- img[:,:,i] → Akses channel ke-i menggunakan indexing NumPy
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

print("=" * 60)
print("PERCOBAAN 15: SPLITTING DAN MERGING CHANNEL WARNA")
print("=" * 60)

# Membaca gambar berwarna
img = cv2.imread(os.path.join(IMAGE_DIR, "foto_bunga2.jpg"))
if img is None:
    img = cv2.imread(os.path.join(IMAGE_DIR, "foto_bunga.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!")
    exit()

img = cv2.resize(img, (300, 300))
print(f"[INFO] Gambar: {img.shape}")

# ============================================================
# 1. Split channel BGR menggunakan cv2.split()
# ============================================================
print("\n--- 1. Split Channel BGR ---")

# cv2.split memisahkan gambar 3-channel menjadi 3 gambar 1-channel
b, g, r = cv2.split(img)
print(f"  Blue  channel shape: {b.shape}, dtype: {b.dtype}")
print(f"  Green channel shape: {g.shape}, dtype: {g.dtype}")
print(f"  Red   channel shape: {r.shape}, dtype: {r.dtype}")

# ============================================================
# 2. Split menggunakan NumPy indexing (lebih cepat)
# ============================================================
print("\n--- 2. Split dengan NumPy ---")

# Mengakses channel menggunakan index ketiga (axis=2)
b_np = img[:, :, 0]  # Channel ke-0 = Blue
g_np = img[:, :, 1]  # Channel ke-1 = Green
r_np = img[:, :, 2]  # Channel ke-2 = Red

# Memverifikasi hasilnya identik
print(f"  cv2.split == NumPy indexing: {np.array_equal(b, b_np)}")

# ============================================================
# 3. Visualisasi channel sebagai grayscale
# ============================================================
print("\n--- 3. Visualisasi Channel Grayscale ---")

# Setiap channel adalah gambar grayscale (intensitas 0-255)
# Area terang = nilai tinggi pada channel tersebut
print("  Setiap channel ditampilkan sebagai grayscale")

# ============================================================
# 4. Visualisasi channel dengan warna aslinya
# ============================================================
print("\n--- 4. Visualisasi Channel Berwarna ---")

# Membuat gambar yang hanya menunjukkan satu channel
# Zero array untuk channel yang tidak aktif
zero = np.zeros_like(b)

# Hanya channel Blue yang aktif
img_only_b = cv2.merge([b, zero, zero])
# Hanya channel Green yang aktif
img_only_g = cv2.merge([zero, g, zero])
# Hanya channel Red yang aktif
img_only_r = cv2.merge([zero, zero, r])

print("  Gambar per-channel dengan warna asli dibuat")

# ============================================================
# 5. Merge kembali channel
# ============================================================
print("\n--- 5. Merge Channel ---")

# cv2.merge menggabungkan channel-channel menjadi satu gambar
img_merged = cv2.merge([b, g, r])
# Verifikasi merge menghasilkan gambar yang sama
print(f"  Merge BGR == asli: {np.array_equal(img, img_merged)}")

# Merge dengan urutan berbeda (swap channel)
# RGB → mengubah urutan warna
img_rgb = cv2.merge([r, g, b])
# RBG → channel hijau dan biru tertukar
img_rbg = cv2.merge([r, b, g])
print("  Swap channel: RGB dan RBG juga dibuat")

# ============================================================
# 6. Manipulasi per-channel
# ============================================================
print("\n--- 6. Manipulasi Per-Channel ---")

# Meningkatkan channel merah (membuat gambar lebih merah)
r_boost = np.clip(r.astype(np.int16) + 80, 0, 255).astype(np.uint8)
img_merah = cv2.merge([b, g, r_boost])
print("  Red channel +80")

# Mengurangi channel biru (menghilangkan tone biru)
b_reduce = np.clip(b.astype(np.int16) - 80, 0, 255).astype(np.uint8)
img_kurang_biru = cv2.merge([b_reduce, g, r])
print("  Blue channel -80")

# Meningkatkan channel hijau
g_boost = np.clip(g.astype(np.int16) + 60, 0, 255).astype(np.uint8)
img_hijau = cv2.merge([b, g_boost, r])
print("  Green channel +60")

# ============================================================
# 7. Split channel HSV
# ============================================================
print("\n--- 7. Split Channel HSV ---")

# Konversi ke ruang warna HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split menjadi Hue, Saturation, Value
h_ch, s_ch, v_ch = cv2.split(img_hsv)
print(f"  Hue range:        [{h_ch.min()}, {h_ch.max()}]")
print(f"  Saturation range: [{s_ch.min()}, {s_ch.max()}]")
print(f"  Value range:      [{v_ch.min()}, {v_ch.max()}]")

# Meningkatkan saturasi
s_boost = np.clip(s_ch.astype(np.int16) + 60, 0, 255).astype(np.uint8)
img_saturasi = cv2.cvtColor(cv2.merge([h_ch, s_boost, v_ch]), cv2.COLOR_HSV2BGR)
print("  Saturasi +60 diterapkan")

# ============================================================
# 8. Visualisasi
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Original + channel grayscale
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 1].imshow(b, cmap="gray")
axes[0, 1].set_title("Blue Channel")
axes[0, 2].imshow(g, cmap="gray")
axes[0, 2].set_title("Green Channel")
axes[0, 3].imshow(r, cmap="gray")
axes[0, 3].set_title("Red Channel")

# Baris 2: Channel berwarna + swap
axes[1, 0].imshow(cv2.cvtColor(img_only_b, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Hanya Blue")
axes[1, 1].imshow(cv2.cvtColor(img_only_g, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Hanya Green")
axes[1, 2].imshow(cv2.cvtColor(img_only_r, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Hanya Red")
axes[1, 3].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Swap BGR→RGB")

# Baris 3: Manipulasi channel
axes[2, 0].imshow(cv2.cvtColor(img_merah, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("Red +80")
axes[2, 1].imshow(cv2.cvtColor(img_kurang_biru, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("Blue -80")
axes[2, 2].imshow(cv2.cvtColor(img_hijau, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title("Green +60")
axes[2, 3].imshow(cv2.cvtColor(img_saturasi, cv2.COLOR_BGR2RGB))
axes[2, 3].set_title("Saturasi +60")

for ax in axes.flat:
    ax.axis("off")

plt.suptitle("Percobaan 15: Splitting & Merging Channel", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "15_split_merge_channel_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 15")
print("=" * 60)
print("  cv2.split()  → Pisahkan channel")
print("  cv2.merge()  → Gabungkan channel")
print("  img[:,:,i]   → Akses channel via NumPy")
print("  Manipulasi per-channel untuk efek warna")
print("  HSV split untuk kontrol Hue/Saturation/Value")
print("=" * 60)
