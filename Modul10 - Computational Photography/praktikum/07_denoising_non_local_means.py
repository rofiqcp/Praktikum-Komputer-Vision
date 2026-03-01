"""
==========================================================================
PERCOBAAN 7: DENOISING NON-LOCAL MEANS (NLM)
==========================================================================
Program ini mempelajari teknik denoising Non-Local Means (NLM) yang
merupakan salah satu metode denoising terbaik secara klasik. NLM mencari
patch-patch mirip di SELURUH gambar (non-local), bukan hanya tetangga dekat.

Prinsip NLM:
- Untuk setiap piksel, cari semua patch mirip di search window
- Rata-ratakan patch-patch mirip dengan bobot berdasarkan kemiripan
- Hasilnya: noise berkurang drastis, detail tetap tajam

Fungsi utama yang dipelajari:
- cv2.fastNlMeansDenoisingColored(src, None, h, hColor, templateWinSize, searchWinSize)
- Perbandingan NLM vs Bilateral vs Gaussian
- Tabel PSNR komprehensif

Hasil: Perbandingan 3 metode denoising dengan metrik PSNR
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk manajemen path file
import os

# Mengimpor matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 7: DENOISING NON-LOCAL MEANS (NLM)")
print("=" * 60)

# ============================================================
# 1. Membaca gambar noisy dan ground truth
# ============================================================

# Membaca gambar noisy Gaussian
noisy_path = os.path.join(IMAGE_DIR, "noisy_gaussian.png")
noisy_img = cv2.imread(noisy_path)

# Validasi pembacaan
if noisy_img is None:
    print(f"[ERROR] Gagal membaca: {noisy_path}")
    print("[INFO] Jalankan download_image.py terlebih dahulu!")
    exit()

# Membaca gambar asli (ground truth)
gt_path = os.path.join(IMAGE_DIR, "scene_pemandangan.png")
gt_img = cv2.imread(gt_path)

# Validasi ground truth
if gt_img is None:
    print(f"[ERROR] Gagal membaca: {gt_path}")
    exit()

# Menyamakan ukuran jika berbeda
if noisy_img.shape != gt_img.shape:
    gt_img = cv2.resize(gt_img, (noisy_img.shape[1], noisy_img.shape[0]))

# Membaca juga gambar noisy heavy untuk pengujian
heavy_path = os.path.join(IMAGE_DIR, "noisy_heavy.png")
noisy_heavy = cv2.imread(heavy_path)

# Jika noisy_heavy tidak ada, buat versi dengan noise lebih berat
if noisy_heavy is None:
    print("[INFO] noisy_heavy.png tidak ditemukan, menggunakan noise sintetis")
    noise = np.random.normal(0, 50, gt_img.shape).astype(np.float32)
    noisy_heavy = np.clip(gt_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
else:
    # Menyamakan ukuran
    if noisy_heavy.shape != gt_img.shape:
        noisy_heavy = cv2.resize(noisy_heavy, (gt_img.shape[1], gt_img.shape[0]))

# Menampilkan info
print(f"[INFO] Gambar noisy  : {noisy_img.shape[1]}x{noisy_img.shape[0]}")
print(f"[INFO] Ground truth  : {gt_img.shape[1]}x{gt_img.shape[0]}")

# PSNR baseline
psnr_noisy = cv2.PSNR(gt_img, noisy_img)
psnr_heavy = cv2.PSNR(gt_img, noisy_heavy)
print(f"[INFO] PSNR noisy_gaussian: {psnr_noisy:.2f} dB")
print(f"[INFO] PSNR noisy_heavy   : {psnr_heavy:.2f} dB")

# ============================================================
# 2. Non-Local Means Denoising dengan variasi parameter h
# cv2.fastNlMeansDenoisingColored(src, None, h, hForColorComponents,
#                                  templateWindowSize, searchWindowSize)
# - h                  : Kekuatan filter (besar = lebih halus, tapi blur)
# - hForColorComponents: Sama seperti h tapi untuk channel warna
# - templateWindowSize : Ukuran patch template (ganjil, umumnya 7)
# - searchWindowSize   : Ukuran area pencarian (ganjil, umumnya 21)
# ============================================================

# Daftar parameter h yang akan diuji
h_values = [3, 5, 7, 10, 15, 20, 30, 40]

# Menyimpan hasil NLM untuk setiap h
nlm_results = []
nlm_psnr = []

print(f"\n--- NLM: Variasi parameter h (template=7, search=21) ---")
for h in h_values:
    # Menerapkan Non-Local Means denoising untuk gambar berwarna
    denoised = cv2.fastNlMeansDenoisingColored(
        noisy_img,      # Gambar input (noisy)
        None,            # Gambar output (None = buat baru)
        h=h,             # Kekuatan filter luminance
        hForColorComponents=h,  # Kekuatan filter warna
        templateWindowSize=7,   # Ukuran patch template
        searchWindowSize=21     # Ukuran area pencarian
    )

    # Menghitung PSNR
    psnr_val = cv2.PSNR(gt_img, denoised)

    # Menyimpan hasil
    nlm_results.append(denoised)
    nlm_psnr.append(psnr_val)

    print(f"  h={h:2d} — PSNR: {psnr_val:.2f} dB")

# Menemukan h terbaik
best_h_idx = np.argmax(nlm_psnr)
best_h = h_values[best_h_idx]
best_psnr_nlm = nlm_psnr[best_h_idx]
print(f"\n[INFO] Parameter h terbaik: {best_h} (PSNR: {best_psnr_nlm:.2f} dB)")

# ============================================================
# 3. Perbandingan NLM vs Bilateral vs Gaussian
# ============================================================

print("\n--- Perbandingan 3 Metode Denoising ---")

# Gaussian Blur (baseline sederhana)
gauss_result = cv2.GaussianBlur(noisy_img, (7, 7), sigmaX=0)
psnr_gauss = cv2.PSNR(gt_img, gauss_result)
print(f"  Gaussian Blur (7x7)     — PSNR: {psnr_gauss:.2f} dB")

# Bilateral Filter (edge-preserving)
bilat_result = cv2.bilateralFilter(noisy_img, d=9, sigmaColor=75, sigmaSpace=75)
psnr_bilat = cv2.PSNR(gt_img, bilat_result)
print(f"  Bilateral (d=9,sc=75)   — PSNR: {psnr_bilat:.2f} dB")

# NLM dengan parameter terbaik
nlm_best = nlm_results[best_h_idx]
print(f"  NLM (h={best_h})             — PSNR: {best_psnr_nlm:.2f} dB")

# ============================================================
# 4. Uji pada noisy_heavy (noise lebih berat)
# ============================================================

print(f"\n--- Perbandingan pada Noisy Heavy ---")

# Gaussian pada heavy noise
gauss_heavy = cv2.GaussianBlur(noisy_heavy, (7, 7), sigmaX=0)
psnr_gauss_h = cv2.PSNR(gt_img, gauss_heavy)
print(f"  Gaussian (7x7)     — PSNR: {psnr_gauss_h:.2f} dB")

# Bilateral pada heavy noise
bilat_heavy = cv2.bilateralFilter(noisy_heavy, d=9, sigmaColor=75, sigmaSpace=75)
psnr_bilat_h = cv2.PSNR(gt_img, bilat_heavy)
print(f"  Bilateral (d=9)    — PSNR: {psnr_bilat_h:.2f} dB")

# NLM pada heavy noise (h lebih besar untuk noise berat)
nlm_heavy = cv2.fastNlMeansDenoisingColored(noisy_heavy, None, h=20,
                                             hForColorComponents=20,
                                             templateWindowSize=7,
                                             searchWindowSize=21)
psnr_nlm_h = cv2.PSNR(gt_img, nlm_heavy)
print(f"  NLM (h=20)         — PSNR: {psnr_nlm_h:.2f} dB")

# ============================================================
# 5. Visualisasi perbandingan
# ============================================================

# Membuat figure 3 baris x 4 kolom
fig, axes = plt.subplots(3, 4, figsize=(22, 15))

# --- Baris 1: Variasi parameter h (4 contoh) ---
display_h_idx = [0, 2, 4, 7]  # h=3, 7, 15, 40
for i, idx in enumerate(display_h_idx):
    rgb = cv2.cvtColor(nlm_results[idx], cv2.COLOR_BGR2RGB)
    axes[0, i].imshow(rgb)
    axes[0, i].set_title(f"NLM h={h_values[idx]}\n"
                         f"PSNR: {nlm_psnr[idx]:.2f} dB", fontsize=9)
    axes[0, i].axis("off")

# --- Baris 2: Perbandingan 3 metode ---
# Ground truth
gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
axes[1, 0].imshow(gt_rgb)
axes[1, 0].set_title("Ground Truth", fontsize=9)
axes[1, 0].axis("off")

# Gaussian
gauss_rgb = cv2.cvtColor(gauss_result, cv2.COLOR_BGR2RGB)
axes[1, 1].imshow(gauss_rgb)
axes[1, 1].set_title(f"Gaussian Blur\nPSNR: {psnr_gauss:.2f} dB", fontsize=9)
axes[1, 1].axis("off")

# Bilateral
bilat_rgb = cv2.cvtColor(bilat_result, cv2.COLOR_BGR2RGB)
axes[1, 2].imshow(bilat_rgb)
axes[1, 2].set_title(f"Bilateral Filter\nPSNR: {psnr_bilat:.2f} dB", fontsize=9)
axes[1, 2].axis("off")

# NLM best
nlm_rgb = cv2.cvtColor(nlm_best, cv2.COLOR_BGR2RGB)
axes[1, 3].imshow(nlm_rgb)
axes[1, 3].set_title(f"NLM (h={best_h})\nPSNR: {best_psnr_nlm:.2f} dB", fontsize=9)
axes[1, 3].axis("off")

# --- Baris 3: PSNR chart dan tabel ---
# Grafik PSNR vs h
axes[2, 0].plot(h_values, nlm_psnr, 'ro-', linewidth=2, markersize=6, label='NLM')
axes[2, 0].axhline(y=psnr_gauss, color='b', linestyle='--',
                    label=f'Gaussian: {psnr_gauss:.1f} dB')
axes[2, 0].axhline(y=psnr_bilat, color='g', linestyle='--',
                    label=f'Bilateral: {psnr_bilat:.1f} dB')
axes[2, 0].axhline(y=psnr_noisy, color='gray', linestyle=':',
                    label=f'Noisy: {psnr_noisy:.1f} dB')
axes[2, 0].set_title("PSNR vs parameter h", fontsize=9)
axes[2, 0].set_xlabel("h (filter strength)")
axes[2, 0].set_ylabel("PSNR (dB)")
axes[2, 0].legend(fontsize=7)
axes[2, 0].grid(True, alpha=0.3)

# Bar chart perbandingan 3 metode (noise ringan)
methods = ['Noisy', 'Gaussian', 'Bilateral', 'NLM']
psnr_vals = [psnr_noisy, psnr_gauss, psnr_bilat, best_psnr_nlm]
bar_colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800']
bars = axes[2, 1].bar(methods, psnr_vals, color=bar_colors)
axes[2, 1].set_title("PSNR Noise Ringan", fontsize=9)
axes[2, 1].set_ylabel("PSNR (dB)")
axes[2, 1].grid(axis='y', alpha=0.3)
# Menambahkan label nilai di atas bar
for bar, val in zip(bars, psnr_vals):
    axes[2, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', fontsize=8)

# Bar chart perbandingan 3 metode (noise berat)
psnr_heavy_vals = [psnr_heavy, psnr_gauss_h, psnr_bilat_h, psnr_nlm_h]
bars_h = axes[2, 2].bar(methods, psnr_heavy_vals, color=bar_colors)
axes[2, 2].set_title("PSNR Noise Berat", fontsize=9)
axes[2, 2].set_ylabel("PSNR (dB)")
axes[2, 2].grid(axis='y', alpha=0.3)
for bar, val in zip(bars_h, psnr_heavy_vals):
    axes[2, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', fontsize=8)

# Tabel PSNR ringkasan
table_data = [
    ['Noisy', f'{psnr_noisy:.2f}', f'{psnr_heavy:.2f}'],
    ['Gaussian', f'{psnr_gauss:.2f}', f'{psnr_gauss_h:.2f}'],
    ['Bilateral', f'{psnr_bilat:.2f}', f'{psnr_bilat_h:.2f}'],
    ['NLM', f'{best_psnr_nlm:.2f}', f'{psnr_nlm_h:.2f}'],
]
axes[2, 3].axis('off')
table = axes[2, 3].table(
    cellText=table_data,
    colLabels=['Metode', 'Noise Ringan\n(dB)', 'Noise Berat\n(dB)'],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.8)
axes[2, 3].set_title("Tabel PSNR Perbandingan", fontsize=9)

# Menambahkan judul utama
plt.suptitle("Percobaan 7: Non-Local Means Denoising\n"
             "NLM vs Bilateral vs Gaussian — PSNR Comparison",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil
output_path = os.path.join(OUTPUT_DIR, "07_denoising_non_local_means.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 7")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.fastNlMeansDenoisingColored(src, None, h, hColor, tWin, sWin)")
print("     - h               : Kekuatan filter (semakin besar = halus)")
print("     - hForColorComponents: Kekuatan filter untuk channel warna")
print("     - templateWindowSize : Ukuran patch (default 7)")
print("     - searchWindowSize   : Area pencarian (default 21)")
print("  2. Tabel PSNR perbandingan:")
print(f"     Noise Ringan: Gaussian={psnr_gauss:.1f}, "
      f"Bilateral={psnr_bilat:.1f}, NLM={best_psnr_nlm:.1f} dB")
print(f"     Noise Berat : Gaussian={psnr_gauss_h:.1f}, "
      f"Bilateral={psnr_bilat_h:.1f}, NLM={psnr_nlm_h:.1f} dB")
print("  3. NLM unggul karena mencari patch mirip di area luas")
print("  4. Trade-off: NLM lebih lambat dibanding Gaussian/Bilateral")
print("=" * 60)
