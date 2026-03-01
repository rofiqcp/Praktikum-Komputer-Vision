"""
==========================================================================
PERCOBAAN 13: INTERPOLASI FRAME LINEAR (ALPHA BLENDING)
==========================================================================
Program ini mempelajari teknik interpolasi frame linear menggunakan
alpha blending. Dua frame dicampur dengan bobot yang berbeda untuk
menghasilkan frame-frame antara (intermediate frames).

Interpolasi Linear:
- Frame_interp = (1 - alpha) * Frame_A + alpha * Frame_B
- alpha = 0.0 → Frame A murni
- alpha = 0.5 → Campuran 50%-50%
- alpha = 1.0 → Frame B murni

Fungsi utama yang dipelajari:
- cv2.addWeighted()               : Blending dua gambar dengan bobot
- cv2.imread()                    : Membaca gambar dari file
- np.linspace()                   : Membuat nilai alpha yang merata

Hasil: Sekuens frame interpolasi dari Frame A ke Frame B
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
print("PERCOBAAN 13: INTERPOLASI FRAME LINEAR (ALPHA BLENDING)")
print("=" * 60)

# ============================================================
# 1. Membaca dua frame yang akan diinterpolasi
# ============================================================

# Membaca frame pertama (Frame A / frame awal)
frame_a_path = os.path.join(IMAGE_DIR, "frame_t0.png")
frame_a = cv2.imread(frame_a_path)

# Membaca frame kedua (Frame B / frame akhir)
frame_b_path = os.path.join(IMAGE_DIR, "frame_t1.png")
frame_b = cv2.imread(frame_b_path)

# Memeriksa apakah kedua gambar berhasil dibaca
if frame_a is None or frame_b is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    print(f"  Frame A: {frame_a_path} → {'OK' if frame_a is not None else 'TIDAK ADA'}")
    print(f"  Frame B: {frame_b_path} → {'OK' if frame_b is not None else 'TIDAK ADA'}")
    exit()

# Menampilkan informasi gambar
print(f"[INFO] Frame A berhasil dibaca: {frame_a.shape}")
print(f"[INFO] Frame B berhasil dibaca: {frame_b.shape}")

# Memastikan kedua frame memiliki ukuran yang sama
if frame_a.shape != frame_b.shape:
    print("[WARN] Ukuran frame berbeda, melakukan resize Frame B sesuai Frame A.")
    frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))

# ============================================================
# 2. Melakukan interpolasi linear dengan berbagai nilai alpha
# ============================================================

# Mendefinisikan jumlah frame interpolasi yang diinginkan
NUM_INTERPOLATED = 9  # 9 frame antara, total 11 (termasuk A dan B)

# Membuat array nilai alpha dari 0.0 sampai 1.0
# np.linspace(start, stop, num) → array teratur
alpha_values = np.linspace(0.0, 1.0, NUM_INTERPOLATED + 2)

print(f"\n[INFO] Menginterpolasi {NUM_INTERPOLATED + 2} frame (termasuk frame asli)")
print(f"[INFO] Nilai alpha: {[f'{a:.2f}' for a in alpha_values]}")

# List untuk menyimpan semua frame hasil interpolasi
interpolated_frames = []

# Melakukan interpolasi untuk setiap nilai alpha
for alpha in alpha_values:
    # ============================================================
    # cv2.addWeighted(src1, alpha1, src2, alpha2, gamma)
    # Rumus: output = alpha1 * src1 + alpha2 * src2 + gamma
    # Untuk interpolasi linear:
    #   output = (1 - alpha) * frame_a + alpha * frame_b
    # ============================================================
    
    # Menghitung bobot untuk Frame A (beta = 1 - alpha)
    beta = 1.0 - alpha
    
    # Melakukan blending menggunakan addWeighted
    blended = cv2.addWeighted(frame_a, beta, frame_b, alpha, 0)
    
    # Menyimpan frame hasil interpolasi
    interpolated_frames.append(blended)
    
    print(f"  Alpha={alpha:.2f}: Frame A × {beta:.2f} + Frame B × {alpha:.2f}")

# ============================================================
# 3. Visualisasi sekuens frame interpolasi
# ============================================================

# Menghitung layout grid untuk visualisasi
total_vis = len(interpolated_frames)
ncols = min(4, total_vis)
nrows = (total_vis + ncols - 1) // ncols

# Membuat figure untuk menampilkan semua frame
fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

# Memastikan axes adalah array 2D
if nrows == 1:
    axes1 = axes1.reshape(1, -1)

for idx in range(total_vis):
    row = idx // ncols
    col = idx % ncols
    
    # Mengkonversi BGR ke RGB untuk matplotlib
    rgb = cv2.cvtColor(interpolated_frames[idx], cv2.COLOR_BGR2RGB)
    axes1[row, col].imshow(rgb)
    
    # Menampilkan label alpha pada setiap subplot
    alpha_val = alpha_values[idx]
    if alpha_val == 0.0:
        label = f"Frame A (α=0.0)"
    elif alpha_val == 1.0:
        label = f"Frame B (α=1.0)"
    else:
        label = f"Interpolasi α={alpha_val:.2f}"
    
    axes1[row, col].set_title(label, fontsize=9)
    axes1[row, col].axis("off")

# Menyembunyikan subplot kosong
for idx in range(total_vis, nrows * ncols):
    row = idx // ncols
    col = idx % ncols
    axes1[row, col].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 13: Interpolasi Frame Linear (Alpha Blending)\n"
             "Frame = (1-α) × A + α × B", fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan figure pertama
output_path_1 = os.path.join(OUTPUT_DIR, "13_frame_interpolation_linear.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil interpolasi disimpan di: {output_path_1}")

# ============================================================
# 4. Analisis perbedaan antara frame interpolasi
# ============================================================

# Menghitung perbedaan antar frame berturutan
diffs = []
for i in range(1, len(interpolated_frames)):
    # Menghitung perbedaan absolut antar frame berturutan
    diff = cv2.absdiff(interpolated_frames[i], interpolated_frames[i-1])
    # Menghitung rata-rata perbedaan sebagai ukuran "jarak"
    mean_diff = diff.mean()
    diffs.append(mean_diff)

# Membuat figure: analisis perbedaan
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Grafik perbedaan antar frame
axes2[0].bar(range(len(diffs)), diffs, color='steelblue', alpha=0.7)
axes2[0].set_xlabel("Pasangan Frame (i → i+1)", fontsize=10)
axes2[0].set_ylabel("Rata-rata Perbedaan Piksel", fontsize=10)
axes2[0].set_title("Perbedaan Antar Frame Berturutan", fontsize=11)
axes2[0].grid(True, alpha=0.3)

# Subplot 2: Visualisasi perbedaan Frame A vs Frame B
diff_ab = cv2.absdiff(frame_a, frame_b)
diff_gray = cv2.cvtColor(diff_ab, cv2.COLOR_BGR2GRAY)
axes2[1].imshow(diff_gray, cmap='hot')
axes2[1].set_title("Perbedaan Absolut: Frame A vs Frame B", fontsize=11)
axes2[1].axis("off")
axes2[1].set_xlabel("")

# Menambahkan colorbar
plt.colorbar(axes2[1].images[0], ax=axes2[1], fraction=0.046, pad=0.04)

# Menambahkan judul utama
plt.suptitle("Percobaan 13: Analisis Interpolasi Linear", fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan figure kedua
output_path_2 = os.path.join(OUTPUT_DIR, "13_interpolation_analysis.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Hasil analisis disimpan di: {output_path_2}")

# ============================================================
# 5. Perbandingan detail: Frame A, tengah, dan Frame B
# ============================================================

# Membuat figure perbandingan detail 3 frame
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

# Frame A
axes3[0].imshow(cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB))
axes3[0].set_title("Frame A (α = 0.0)", fontsize=11)
axes3[0].axis("off")

# Frame tengah (alpha = 0.5)
mid_idx = len(interpolated_frames) // 2
axes3[1].imshow(cv2.cvtColor(interpolated_frames[mid_idx], cv2.COLOR_BGR2RGB))
axes3[1].set_title(f"Interpolasi (α = {alpha_values[mid_idx]:.2f})", fontsize=11)
axes3[1].axis("off")

# Frame B
axes3[2].imshow(cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB))
axes3[2].set_title("Frame B (α = 1.0)", fontsize=11)
axes3[2].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 13: Perbandingan Frame A → Tengah → Frame B",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# Menyimpan figure ketiga
output_path_3 = os.path.join(OUTPUT_DIR, "13_interpolation_detail.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Hasil detail disimpan di: {output_path_3}")

# Menampilkan statistik
print(f"\n[INFO] Statistik Interpolasi:")
print(f"  - Ukuran frame       : {frame_a.shape[1]}x{frame_a.shape[0]} piksel")
print(f"  - Jumlah frame total : {len(interpolated_frames)}")
print(f"  - Perbedaan A-B      : {cv2.absdiff(frame_a, frame_b).mean():.2f} (rata-rata)")
print(f"  - Perbedaan per step : {np.mean(diffs):.2f} (rata-rata)")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 13")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.addWeighted(src1, α1, src2, α2, γ)  → Blending dua gambar")
print("     - Output = α1 * src1 + α2 * src2 + γ")
print("     - Untuk interpolasi: (1-α)*A + α*B")
print("  2. cv2.absdiff()                            → Perbedaan absolut")
print("  3. np.linspace()                            → Membuat nilai alpha merata")
print("Konsep Interpolasi Linear:")
print("  - α=0.0 → Frame A, α=0.5 → campuran, α=1.0 → Frame B")
print("  - Kelebihan: mudah, cepat, tidak butuh optical flow")
print("  - Kekurangan: ghosting pada objek yang bergerak")
print("  - Cocok untuk transisi halus (dissolve effect)")
print("=" * 60)
