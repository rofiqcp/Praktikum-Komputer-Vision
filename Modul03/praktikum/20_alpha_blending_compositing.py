"""
==========================================================================
PERCOBAAN 20: ALPHA BLENDING DAN COMPOSITING
==========================================================================
Alpha blending menggabungkan dua gambar dengan bobot transparan.
Compositing merupakan teknik menggabungkan foreground + background
menggunakan alpha matte (mask transparan).

Fungsi:
- cv2.addWeighted(src1, alpha, src2, beta, gamma) → blending linear
- cv2.seamlessClone(src, dst, mask, center, flags) → seamless compositing
- Laplacian Pyramid Blending → blending multi-resolusi
- Alpha compositing formula: out = fg*α + bg*(1-α)
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

# Muat dua gambar untuk blending
img1 = cv2.imread(os.path.join(IMAGE_DIR, "kota.jpg"))
img2 = cv2.imread(os.path.join(IMAGE_DIR, "nature.jpg"))
if img1 is None or img2 is None:
    print("[ERROR] Jalankan download_image.py!"); exit()

# Resize ke ukuran sama
SIZE = (512, 512)
img1 = cv2.resize(img1, SIZE)
img2 = cv2.resize(img2, SIZE)

print("=" * 60)
print("PERCOBAAN 20: ALPHA BLENDING DAN COMPOSITING")
print("=" * 60)

# ============================================================
# 1. Linear Blending (addWeighted)
# ============================================================
print("\n--- 1. Linear Blending ---")

results_blend = []
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    # Rumus: dst = src1 * alpha + src2 * beta + gamma
    beta = 1.0 - alpha
    blended = cv2.addWeighted(img1, alpha, img2, beta, 0)
    results_blend.append((alpha, blended))
    print(f"  alpha={alpha:.2f}, beta={beta:.2f}")

# ============================================================
# 2. Alpha Compositing Manual (Per-Pixel)
# ============================================================
print("\n--- 2. Alpha Compositing Manual ---")

# Buat alpha mask gradien horizontal (kiri=img1, kanan=img2)
alpha_mask = np.zeros((SIZE[1], SIZE[0]), dtype=np.float32)
for c in range(SIZE[0]):
    # Gradien dari 1.0 (kiri) ke 0.0 (kanan)
    alpha_mask[:, c] = 1.0 - c / (SIZE[0] - 1)

# Expand alpha ke 3 channel
alpha_3ch = np.stack([alpha_mask] * 3, axis=-1)
# Compositing: out = fg * alpha + bg * (1 - alpha)
composite_grad = (img1.astype(np.float32) * alpha_3ch +
                  img2.astype(np.float32) * (1 - alpha_3ch))
composite_grad = composite_grad.astype(np.uint8)
print("  Gradient compositing: transisi halus kiri→kanan")

# ============================================================
# 3. Alpha Mask Lingkaran
# ============================================================
print("\n--- 3. Alpha Mask Lingkaran ---")

# Buat mask lingkaran dengan feathering (gradien halus)
alpha_circle = np.zeros((SIZE[1], SIZE[0]), dtype=np.float32)
center = (SIZE[0] // 2, SIZE[1] // 2)
radius = 150
# Hitung jarak setiap pixel dari pusat
Y, X = np.ogrid[:SIZE[1], :SIZE[0]]
dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
# Feathered circle: 1 di dalam, gradien di tepi, 0 di luar
feather = 50
alpha_circle = np.clip((radius + feather - dist) / (2 * feather), 0, 1).astype(np.float32)

# Compositing
alpha_c_3ch = np.stack([alpha_circle] * 3, axis=-1)
composite_circle = (img1.astype(np.float32) * alpha_c_3ch +
                    img2.astype(np.float32) * (1 - alpha_c_3ch))
composite_circle = composite_circle.astype(np.uint8)
print(f"  Lingkaran r={radius}, feather={feather}px")

# ============================================================
# 4. Laplacian Pyramid Blending
# ============================================================
print("\n--- 4. Laplacian Pyramid Blending ---")

def build_gaussian_pyramid(img, levels):
    """Bangun Gaussian Pyramid."""
    G = img.copy()
    gp = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

def build_laplacian_pyramid(gp):
    """Bangun Laplacian Pyramid dari Gaussian."""
    lp = [gp[-1]]
    for i in range(len(gp) - 1, 0, -1):
        # Upsample level bawah
        GE = cv2.pyrUp(gp[i])
        # Sesuaikan ukuran (bisa beda 1 pixel)
        GE = cv2.resize(GE, (gp[i-1].shape[1], gp[i-1].shape[0]))
        # Laplacian = current - upsampled(next)
        L = cv2.subtract(gp[i-1], GE)
        lp.append(L)
    return lp[::-1]  # balik urutan (finest first)

levels = 5

# Bangun Gaussian pyramid masing-masing
gp1 = build_gaussian_pyramid(img1, levels)
gp2 = build_gaussian_pyramid(img2, levels)

# Bangun Laplacian pyramid masing-masing
lp1 = build_laplacian_pyramid(gp1)
lp2 = build_laplacian_pyramid(gp2)

# Buat mask Gaussian pyramid (setengah kiri = 1, kanan = 0)
mask = np.zeros(img1.shape, dtype=np.float32)
mask[:, :SIZE[0]//2] = 1.0
gp_mask = build_gaussian_pyramid(mask, levels)

# Blending setiap level Laplacian
lp_blend = []
for la, lb, gm in zip(lp1, lp2, [gp_mask[i] for i in range(len(lp1))]):
    # Resize mask jika perlu
    gm = cv2.resize(gm, (la.shape[1], la.shape[0]))
    lb = cv2.resize(lb, (la.shape[1], la.shape[0]))
    # Blend: la * mask + lb * (1 - mask)
    blended_level = (la.astype(np.float32) * gm +
                     lb.astype(np.float32) * (1 - gm))
    lp_blend.append(blended_level.astype(np.int16))

# Rekonstruksi dari Laplacian blend
result = lp_blend[-1].astype(np.float32)
for i in range(len(lp_blend) - 2, -1, -1):
    result = cv2.pyrUp(result)
    result = cv2.resize(result, (lp_blend[i].shape[1], lp_blend[i].shape[0]))
    result = result + lp_blend[i].astype(np.float32)

lap_blend = np.clip(result, 0, 255).astype(np.uint8)

# Bandingkan dengan direct cut
direct_cut = img1.copy()
direct_cut[:, SIZE[0]//2:] = img2[:, SIZE[0]//2:]
print("  Laplacian blending → transisi halus tanpa seam")

# ============================================================
# 5. Overlay Teks/Logo
# ============================================================
print("\n--- 5. Overlay Teks/Logo ---")

overlay = img1.copy()
# Buat area semi-transparan untuk teks
sub_region = overlay[20:80, 20:350]
# Buat warna overlay hitam semi-transparent
black_bar = np.zeros_like(sub_region)
cv2.addWeighted(sub_region, 0.4, black_bar, 0.6, 0, sub_region)
overlay[20:80] = overlay[20:80]
overlay[20:80, 20:350] = sub_region
# Tulis teks di atas overlay
cv2.putText(overlay, "Computer Vision", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
print("  Teks overlay dengan background semi-transparan")

# ============================================================
# 6. Seamless Cloning (cv2.seamlessClone)
# ============================================================
print("\n--- 6. Seamless Cloning ---")

# Ambil patch dari img1 dan tempel ke img2
src_patch = img1[100:300, 100:300]
# Buat mask putih (semua area digunakan)
mask_sc = 255 * np.ones(src_patch.shape[:2], dtype=np.uint8)
# Titik penempatan di gambar tujuan
center_sc = (256, 256)

# NORMAL_CLONE: tekstur dst dipertahankan, warna src disesuaikan
clone_normal = cv2.seamlessClone(src_patch, img2, mask_sc, center_sc, cv2.NORMAL_CLONE)
# MIXED_CLONE: kombinasi gradient src dan dst
clone_mixed = cv2.seamlessClone(src_patch, img2, mask_sc, center_sc, cv2.MIXED_CLONE)
print("  NORMAL_CLONE vs MIXED_CLONE")

# ============================================================
# 7. Visualisasi 1: Linear Blending
# ============================================================
fig1, axes1 = plt.subplots(1, 5, figsize=(25, 5))
for idx, (alpha, blended) in enumerate(results_blend):
    axes1[idx].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes1[idx].set_title(f"alpha={alpha:.2f}")
    axes1[idx].axis("off")
plt.suptitle("Linear Blending: addWeighted", fontsize=14, fontweight="bold")
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "20_linear_blending.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path1}")

# ============================================================
# 8. Visualisasi 2: Alpha Compositing & Pyramid
# ============================================================
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))

# Baris 1: alpha compositing
axes2[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title("Image 1"); axes2[0, 0].axis("off")
axes2[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes2[0, 1].set_title("Image 2"); axes2[0, 1].axis("off")
axes2[0, 2].imshow(cv2.cvtColor(composite_grad, cv2.COLOR_BGR2RGB))
axes2[0, 2].set_title("Gradient Blend"); axes2[0, 2].axis("off")
axes2[0, 3].imshow(cv2.cvtColor(composite_circle, cv2.COLOR_BGR2RGB))
axes2[0, 3].set_title("Circle Blend"); axes2[0, 3].axis("off")

# Baris 2: Laplacian blending vs direct cut
axes2[1, 0].imshow(cv2.cvtColor(direct_cut, cv2.COLOR_BGR2RGB))
axes2[1, 0].set_title("Direct Cut"); axes2[1, 0].axis("off")
axes2[1, 1].imshow(cv2.cvtColor(lap_blend, cv2.COLOR_BGR2RGB))
axes2[1, 1].set_title("Laplacian Blend"); axes2[1, 1].axis("off")
axes2[1, 2].imshow(cv2.cvtColor(clone_normal, cv2.COLOR_BGR2RGB))
axes2[1, 2].set_title("Seamless Normal"); axes2[1, 2].axis("off")
axes2[1, 3].imshow(cv2.cvtColor(clone_mixed, cv2.COLOR_BGR2RGB))
axes2[1, 3].set_title("Seamless Mixed"); axes2[1, 3].axis("off")

plt.suptitle("Alpha Compositing & Pyramid Blending", fontsize=16, fontweight="bold")
plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "20_compositing_hasil.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] {path2}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 20")
print("=" * 60)
print("""
1. addWeighted(): linear blending sederhana (alpha * A + beta * B)
2. Alpha compositing manual: fg*α + bg*(1-α) per-pixel
3. Alpha mask: gradient, lingkaran, feathering untuk transisi halus
4. Laplacian Pyramid Blending: multi-resolusi, seamless transition
5. Direct cut → seam terlihat, Laplacian blend → transisi halus
6. seamlessClone(): Poisson blending otomatis dari OpenCV
   - NORMAL_CLONE: warna src disesuaikan ke dst
   - MIXED_CLONE: ambil gradient terkuat dari src atau dst
7. Overlay: teknik menu/HUD dengan semi-transparent background
""")
