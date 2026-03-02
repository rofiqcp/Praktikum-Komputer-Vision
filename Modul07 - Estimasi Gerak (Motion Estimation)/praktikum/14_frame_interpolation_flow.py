"""
==========================================================================
PERCOBAAN 14: INTERPOLASI FRAME BERBASIS OPTICAL FLOW
==========================================================================
Program ini mempelajari teknik interpolasi frame menggunakan optical flow.
Berbeda dengan blending linear, metode ini menggunakan informasi gerakan
(flow) untuk melakukan warping sehingga menghasilkan interpolasi yang
lebih natural tanpa efek ghosting.

Pipeline Flow-Based Interpolation:
1. Hitung optical flow dari Frame A ke Frame B
2. Skala flow sesuai posisi temporal (alpha)
3. Warp Frame A dan Frame B ke posisi intermediate
4. Blend hasil warping untuk frame akhir

Fungsi utama yang dipelajari:
- cv2.calcOpticalFlowFarneback()  : Dense optical flow Farneback
- cv2.remap()                     : Warping gambar berdasarkan flow
- cv2.addWeighted()               : Blending dua gambar
- np.meshgrid()                   : Grid koordinat untuk remap

Hasil: Perbandingan interpolasi flow vs linear
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file
import os

# Mengimpor matplotlib untuk menyimpan visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input dan output
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 14: INTERPOLASI FRAME BERBASIS OPTICAL FLOW")
print("=" * 60)

# ============================================================
# 1. Membaca dua frame yang akan diinterpolasi
# ============================================================

# Membaca frame pertama (Frame A)
frame_a_path = os.path.join(IMAGE_DIR, "frame_t0.png")
frame_a = cv2.imread(frame_a_path)

# Membaca frame kedua (Frame B)
frame_b_path = os.path.join(IMAGE_DIR, "frame_t1.png")
frame_b = cv2.imread(frame_b_path)

# Memeriksa apakah kedua gambar berhasil dibaca
if frame_a is None or frame_b is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Memastikan ukuran sama
if frame_a.shape != frame_b.shape:
    frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))

# Mendapatkan dimensi gambar
h, w = frame_a.shape[:2]
print(f"[INFO] Frame A: {frame_a.shape}, Frame B: {frame_b.shape}")

# ============================================================
# 2. Mengkonversi ke grayscale dan menghitung optical flow
# ============================================================

# Mengkonversi kedua frame ke grayscale untuk optical flow
gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

print("[INFO] Menghitung optical flow Farneback (A → B)...")

# Menghitung dense optical flow dari Frame A ke Frame B
# flow_ab[y, x] = (dx, dy) menunjukkan ke mana piksel (x,y) bergerak
flow_ab = cv2.calcOpticalFlowFarneback(
    gray_a, gray_b,
    None,                # flow awal (None = mulai dari nol)
    pyr_scale=0.5,       # skala pyramid (0.5 = klasik)
    levels=3,            # jumlah level pyramid
    winsize=15,          # ukuran window averaging
    iterations=3,        # iterasi per level
    poly_n=5,            # ukuran neighbourhood polinomial
    poly_sigma=1.2,      # sigma Gaussian untuk polinomial
    flags=0
)

print("[INFO] Menghitung optical flow Farneback (B → A)...")

# Menghitung flow dari Frame B ke Frame A (untuk bidirectional warping)
flow_ba = cv2.calcOpticalFlowFarneback(
    gray_b, gray_a,
    None, pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

print(f"[INFO] Flow shape: {flow_ab.shape}")

# ============================================================
# 3. Fungsi warping menggunakan cv2.remap()
# ============================================================

def warp_frame(frame, flow):
    """
    Melakukan warping pada frame berdasarkan optical flow.
    cv2.remap() memindahkan piksel dari posisi asli ke posisi baru
    berdasarkan map koordinat yang diberikan.
    """
    # Membuat grid koordinat (x, y) untuk setiap piksel
    # meshgrid menghasilkan array 2D dari koordinat 1D
    h, w = frame.shape[:2]
    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(x_coords, y_coords)
    
    # Menambahkan flow ke koordinat asli
    # Posisi baru = posisi asli + flow
    map_x_warped = (map_x + flow[:, :, 0]).astype(np.float32)
    map_y_warped = (map_y + flow[:, :, 1]).astype(np.float32)
    
    # Melakukan remap: mengambil piksel dari posisi yang ditentukan map
    warped = cv2.remap(frame, map_x_warped, map_y_warped,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
    
    return warped

# ============================================================
# 4. Melakukan interpolasi berbasis flow untuk berbagai alpha
# ============================================================

# Nilai alpha untuk interpolasi
alpha_values = np.linspace(0.0, 1.0, 7)

# List untuk menyimpan hasil interpolasi flow-based
flow_interpolated = []

# List untuk menyimpan hasil interpolasi linear (pembanding)
linear_interpolated = []

print(f"\n[INFO] Menginterpolasi {len(alpha_values)} frame...")

for alpha in alpha_values:
    # ============================================================
    # Flow-based interpolation:
    # 1. Skala flow_ab dengan alpha → warp Frame A maju
    # 2. Skala flow_ba dengan (1-alpha) → warp Frame B mundur
    # 3. Blend kedua hasil warping
    # ============================================================
    
    # Menskalakan optical flow sesuai posisi temporal
    scaled_flow_ab = flow_ab * alpha          # Flow A→B dikalikan alpha
    scaled_flow_ba = flow_ba * (1.0 - alpha)  # Flow B→A dikalikan (1-alpha)
    
    # Melakukan warping Frame A ke posisi intermediate
    warped_a = warp_frame(frame_a, scaled_flow_ab)
    
    # Melakukan warping Frame B ke posisi intermediate
    warped_b = warp_frame(frame_b, scaled_flow_ba)
    
    # Melakukan blending dari kedua hasil warping
    # Bobot: semakin dekat ke A → bobot A lebih besar, dan sebaliknya
    flow_result = cv2.addWeighted(warped_a, 1.0 - alpha, warped_b, alpha, 0)
    flow_interpolated.append(flow_result)
    
    # Interpolasi linear untuk perbandingan
    linear_result = cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0)
    linear_interpolated.append(linear_result)
    
    print(f"  Alpha={alpha:.2f}: Flow warp + blend selesai")

# ============================================================
# 5. Visualisasi perbandingan flow vs linear interpolation
# ============================================================

# Membuat figure: perbandingan flow-based vs linear
fig1, axes1 = plt.subplots(2, len(alpha_values), figsize=(3 * len(alpha_values), 7))

for idx, alpha in enumerate(alpha_values):
    # Baris atas: Interpolasi berbasis flow
    rgb_flow = cv2.cvtColor(flow_interpolated[idx], cv2.COLOR_BGR2RGB)
    axes1[0, idx].imshow(rgb_flow)
    axes1[0, idx].set_title(f"α={alpha:.1f}", fontsize=9)
    axes1[0, idx].axis("off")
    
    # Baris bawah: Interpolasi linear
    rgb_linear = cv2.cvtColor(linear_interpolated[idx], cv2.COLOR_BGR2RGB)
    axes1[1, idx].imshow(rgb_linear)
    axes1[1, idx].axis("off")

# Menambahkan label baris
axes1[0, 0].set_ylabel("Flow-Based", fontsize=11, fontweight="bold")
axes1[1, 0].set_ylabel("Linear", fontsize=11, fontweight="bold")

# Menambahkan judul utama
plt.suptitle("Percobaan 14: Interpolasi Frame\nAtas = Flow-Based, Bawah = Linear (Alpha Blending)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan figure pertama
output_path_1 = os.path.join(OUTPUT_DIR, "14_flow_vs_linear_interpolation.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[OUTPUT] Perbandingan disimpan di: {output_path_1}")

# ============================================================
# 6. Visualisasi optical flow yang digunakan
# ============================================================

# Membuat figure: visualisasi flow
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# Menghitung magnitude dan arah flow A→B
mag_ab, ang_ab = cv2.cartToPolar(flow_ab[:, :, 0], flow_ab[:, :, 1])

# Subplot 1: Magnitude flow A→B
im1 = axes2[0, 0].imshow(mag_ab, cmap='hot')
axes2[0, 0].set_title("Magnitude Flow (A → B)", fontsize=11)
axes2[0, 0].axis("off")
plt.colorbar(im1, ax=axes2[0, 0], fraction=0.046)

# Subplot 2: Visualisasi flow sebagai HSV
hsv = np.zeros((h, w, 3), dtype=np.uint8)
hsv[..., 0] = ang_ab * 180 / np.pi / 2  # Hue = arah
hsv[..., 1] = 255                         # Saturasi penuh
hsv[..., 2] = cv2.normalize(mag_ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
axes2[0, 1].imshow(flow_rgb)
axes2[0, 1].set_title("Flow HSV (warna=arah, terang=kecepatan)", fontsize=10)
axes2[0, 1].axis("off")

# Subplot 3: Perbedaan antara flow dan linear pada alpha=0.5
mid = len(alpha_values) // 2
diff_methods = cv2.absdiff(flow_interpolated[mid], linear_interpolated[mid])
diff_gray = cv2.cvtColor(diff_methods, cv2.COLOR_BGR2GRAY)
im3 = axes2[1, 0].imshow(diff_gray, cmap='hot')
axes2[1, 0].set_title("Perbedaan Flow vs Linear (α=0.5)", fontsize=10)
axes2[1, 0].axis("off")
plt.colorbar(im3, ax=axes2[1, 0], fraction=0.046)

# Subplot 4: Perbandingan kualitas
# Menghitung SSIM-like metric per alpha
quality_flow = []
quality_linear = []
for idx in range(len(alpha_values)):
    # Perbedaan flow result dengan target (weighted combination of A and B diff)
    qf = cv2.absdiff(flow_interpolated[idx], flow_interpolated[idx]).mean()
    ql = cv2.absdiff(flow_interpolated[idx], linear_interpolated[idx]).mean()
    quality_flow.append(qf)
    quality_linear.append(ql)

axes2[1, 1].bar(range(len(alpha_values)), quality_linear, color='coral', alpha=0.7,
                label='Perbedaan Flow vs Linear')
axes2[1, 1].set_xlabel("Index Alpha", fontsize=10)
axes2[1, 1].set_ylabel("Rata-rata Perbedaan", fontsize=10)
axes2[1, 1].set_title("Jarak antara Metode Flow dan Linear", fontsize=10)
axes2[1, 1].set_xticks(range(len(alpha_values)))
axes2[1, 1].set_xticklabels([f"{a:.1f}" for a in alpha_values], fontsize=8)
axes2[1, 1].grid(True, alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 14: Analisis Optical Flow untuk Interpolasi",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan figure kedua
output_path_2 = os.path.join(OUTPUT_DIR, "14_flow_analysis.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
plt.show()
print(f"[OUTPUT] Analisis flow disimpan di: {output_path_2}")

# Menampilkan statistik
print(f"\n[INFO] Statistik Optical Flow:")
print(f"  - Magnitude rata-rata (A→B): {mag_ab.mean():.2f} piksel")
print(f"  - Magnitude maksimum (A→B) : {mag_ab.max():.2f} piksel")
print(f"  - Perbedaan max flow vs linear: {max(quality_linear):.2f}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 14")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.calcOpticalFlowFarneback()  → Dense optical flow")
print("  2. cv2.remap(src, map_x, map_y)    → Warping gambar")
print("     - map_x, map_y = koordinat sumber untuk setiap piksel output")
print("     - Interpolasi bilinear untuk sub-piksel accuracy")
print("  3. np.meshgrid()                   → Grid koordinat 2D")
print("  4. cv2.addWeighted()               → Blending frame hasil warp")
print("Konsep Flow-Based Interpolation:")
print("  - Hitung flow A→B dan B→A (bidirectional)")
print("  - Skala flow sesuai alpha → warp kedua frame ke posisi tengah")
print("  - Blend hasil warp: lebih natural daripada linear blending")
print("  - Kelebihan: tidak ada ghosting pada area bergerak")
print("  - Kekurangan: lebih lambat, bergantung kualitas flow")
print("=" * 60)
