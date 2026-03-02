

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 6: DENOISING DENGAN BILATERAL FILTER
    ==========================================================================
    Program ini mempelajari teknik denoising edge-preserving menggunakan
    Bilateral Filter. Berbeda dengan Gaussian blur yang mengaburkan segalanya,
    Bilateral Filter mempertahankan tepi (edge) sambil menghaluskan area datar.

    Prinsip Bilateral Filter:
    - Menggabungkan DUA kernel: spatial dan range (intensitas)
    - Piksel dekat AND mirip intensitasnya → diberi bobot tinggi
    - Piksel dekat TETAPI beda intensitas (edge) → bobot rendah
    - Hasilnya: area datar halus, tepi tetap tajam

    Fungsi utama yang dipelajari:
    - cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
    - Perbandingan Bilateral vs Gaussian
    - Demonstrasi edge preservation

    Hasil: Perbandingan visual edge preservation bilateral vs Gaussian
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
    print("PERCOBAAN 6: DENOISING DENGAN BILATERAL FILTER")
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

    # Menampilkan info
    print(f"[INFO] Gambar noisy  : {noisy_img.shape[1]}x{noisy_img.shape[0]}")
    print(f"[INFO] Ground truth  : {gt_img.shape[1]}x{gt_img.shape[0]}")

    # PSNR gambar noisy (baseline)
    psnr_noisy = cv2.PSNR(gt_img, noisy_img)
    print(f"[INFO] PSNR noisy: {psnr_noisy:.2f} dB")

    # ============================================================
    # 2. Bilateral Filter dengan variasi sigmaColor
    # cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
    # - d          : Diameter neighborhood (pixel); -1 = otomatis dari sigmaSpace
    # - sigmaColor : Filter sigma di ruang warna (intensity domain)
    #                Besar → piksel lebih berbeda juga dianggap mirip
    # - sigmaSpace : Filter sigma di ruang spasial (spatial domain)
    #                Besar → piksel lebih jauh juga ikut dihitung
    # ============================================================

    # Daftar sigma color yang akan diuji (sigmaSpace tetap)
    sigma_color_values = [10, 25, 50, 75, 100, 150, 200, 250]
    fixed_sigma_space = 75
    fixed_d = 9

    # Menyimpan hasil
    color_results = []
    color_psnr = []

    print(f"\n--- Variasi sigmaColor (d={fixed_d}, sigmaSpace={fixed_sigma_space}) ---")
    for sc in sigma_color_values:
        # Menerapkan bilateral filter dengan sigmaColor tertentu
        denoised = cv2.bilateralFilter(
            noisy_img,
            d=fixed_d,
            sigmaColor=sc,
            sigmaSpace=fixed_sigma_space
        )

        # Menghitung PSNR
        psnr_val = cv2.PSNR(gt_img, denoised)

        # Menyimpan hasil
        color_results.append(denoised)
        color_psnr.append(psnr_val)

        print(f"  sigmaColor={sc:3d} — PSNR: {psnr_val:.2f} dB")

    # ============================================================
    # 3. Bilateral Filter dengan variasi sigmaSpace
    # ============================================================

    # Daftar sigma space yang akan diuji (sigmaColor tetap)
    sigma_space_values = [10, 25, 50, 75, 100, 150, 200, 250]
    fixed_sigma_color = 75

    # Menyimpan hasil
    space_results = []
    space_psnr = []

    print(f"\n--- Variasi sigmaSpace (d={fixed_d}, sigmaColor={fixed_sigma_color}) ---")
    for ss in sigma_space_values:
        # Menerapkan bilateral filter dengan sigmaSpace tertentu
        denoised = cv2.bilateralFilter(
            noisy_img,
            d=fixed_d,
            sigmaColor=fixed_sigma_color,
            sigmaSpace=ss
        )

        # Menghitung PSNR
        psnr_val = cv2.PSNR(gt_img, denoised)

        # Menyimpan hasil
        space_results.append(denoised)
        space_psnr.append(psnr_val)

        print(f"  sigmaSpace={ss:3d} — PSNR: {psnr_val:.2f} dB")

    # ============================================================
    # 4. Perbandingan Bilateral vs Gaussian (edge preservation)
    # ============================================================

    # Menerapkan Gaussian blur sebagai pembanding
    gauss_denoised = cv2.GaussianBlur(noisy_img, (9, 9), sigmaX=0)
    psnr_gauss = cv2.PSNR(gt_img, gauss_denoised)

    # Menerapkan bilateral filter dengan parameter optimal
    best_color_idx = np.argmax(color_psnr)
    best_sc = sigma_color_values[best_color_idx]
    bilat_denoised = cv2.bilateralFilter(noisy_img, d=fixed_d,
                                          sigmaColor=best_sc,
                                          sigmaSpace=fixed_sigma_space)
    psnr_bilat = cv2.PSNR(gt_img, bilat_denoised)

    print(f"\n--- Perbandingan Edge Preservation ---")
    print(f"  Gaussian Blur (9x9)         — PSNR: {psnr_gauss:.2f} dB")
    print(f"  Bilateral (sc={best_sc}, ss={fixed_sigma_space}) — PSNR: {psnr_bilat:.2f} dB")

    # Mendeteksi edge menggunakan Canny pada masing-masing hasil
    edges_gt = cv2.Canny(cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY), 50, 150)
    edges_noisy = cv2.Canny(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY), 50, 150)
    edges_gauss = cv2.Canny(cv2.cvtColor(gauss_denoised, cv2.COLOR_BGR2GRAY), 50, 150)
    edges_bilat = cv2.Canny(cv2.cvtColor(bilat_denoised, cv2.COLOR_BGR2GRAY), 50, 150)

    # Menghitung jumlah piksel edge yang terdeteksi
    print(f"\n--- Jumlah Edge Pixels (Canny) ---")
    print(f"  Ground truth : {np.sum(edges_gt > 0):6d} pixels")
    print(f"  Noisy        : {np.sum(edges_noisy > 0):6d} pixels")
    print(f"  Gaussian     : {np.sum(edges_gauss > 0):6d} pixels")
    print(f"  Bilateral    : {np.sum(edges_bilat > 0):6d} pixels")

    # ============================================================
    # 5. Visualisasi perbandingan
    # ============================================================

    # Membuat figure 3 baris x 4 kolom
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))

    # --- Baris 1: Variasi sigmaColor ---
    display_color_idx = [0, 2, 4, 7]  # sigmaColor 10, 50, 100, 250
    for i, idx in enumerate(display_color_idx):
        rgb = cv2.cvtColor(color_results[idx], cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"sigmaColor={sigma_color_values[idx]}\n"
                             f"PSNR: {color_psnr[idx]:.2f} dB", fontsize=9)
        axes[0, i].axis("off")

    # --- Baris 2: Perbandingan denoising dan edge ---
    # Ground truth
    gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(gt_rgb)
    axes[1, 0].set_title("Ground Truth", fontsize=9)
    axes[1, 0].axis("off")

    # Noisy
    noisy_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(noisy_rgb)
    axes[1, 1].set_title(f"Noisy\nPSNR: {psnr_noisy:.2f} dB", fontsize=9)
    axes[1, 1].axis("off")

    # Gaussian denoised
    gauss_rgb = cv2.cvtColor(gauss_denoised, cv2.COLOR_BGR2RGB)
    axes[1, 2].imshow(gauss_rgb)
    axes[1, 2].set_title(f"Gaussian Blur\nPSNR: {psnr_gauss:.2f} dB", fontsize=9)
    axes[1, 2].axis("off")

    # Bilateral denoised
    bilat_rgb = cv2.cvtColor(bilat_denoised, cv2.COLOR_BGR2RGB)
    axes[1, 3].imshow(bilat_rgb)
    axes[1, 3].set_title(f"Bilateral Filter\nPSNR: {psnr_bilat:.2f} dB", fontsize=9)
    axes[1, 3].axis("off")

    # --- Baris 3: Edge maps dan grafik PSNR ---
    # Edge map Gaussian
    axes[2, 0].imshow(edges_gauss, cmap='gray')
    axes[2, 0].set_title(f"Edge — Gaussian\n{np.sum(edges_gauss > 0)} pixels", fontsize=9)
    axes[2, 0].axis("off")

    # Edge map Bilateral
    axes[2, 1].imshow(edges_bilat, cmap='gray')
    axes[2, 1].set_title(f"Edge — Bilateral\n{np.sum(edges_bilat > 0)} pixels", fontsize=9)
    axes[2, 1].axis("off")

    # Grafik PSNR vs sigmaColor
    axes[2, 2].plot(sigma_color_values, color_psnr, 'bo-', linewidth=2, markersize=5)
    axes[2, 2].axhline(y=psnr_noisy, color='r', linestyle='--',
                        label=f'Noisy: {psnr_noisy:.1f} dB')
    axes[2, 2].set_title("PSNR vs sigmaColor", fontsize=9)
    axes[2, 2].set_xlabel("sigmaColor")
    axes[2, 2].set_ylabel("PSNR (dB)")
    axes[2, 2].legend(fontsize=8)
    axes[2, 2].grid(True, alpha=0.3)

    # Grafik PSNR vs sigmaSpace
    axes[2, 3].plot(sigma_space_values, space_psnr, 'go-', linewidth=2, markersize=5)
    axes[2, 3].axhline(y=psnr_noisy, color='r', linestyle='--',
                        label=f'Noisy: {psnr_noisy:.1f} dB')
    axes[2, 3].set_title("PSNR vs sigmaSpace", fontsize=9)
    axes[2, 3].set_xlabel("sigmaSpace")
    axes[2, 3].set_ylabel("PSNR (dB)")
    axes[2, 3].legend(fontsize=8)
    axes[2, 3].grid(True, alpha=0.3)

    # Menambahkan judul utama
    plt.suptitle("Percobaan 6: Denoising dengan Bilateral Filter\n"
                 "Edge-Preserving Smoothing — Perbandingan dengan Gaussian",
                 fontsize=14, fontweight="bold")

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan hasil
    output_path = os.path.join(OUTPUT_DIR, "06_denoising_bilateral_filter.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 6")
    print("=" * 60)
    print("Fungsi yang dipelajari:")
    print("  1. cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)")
    print("     - d          : Diameter neighborhood (-1 = otomatis)")
    print("     - sigmaColor : Filter range (intensitas) — besar = toleran")
    print("     - sigmaSpace : Filter spatial (jarak) — besar = area luas")
    print("  2. Bilateral vs Gaussian:")
    print(f"     - Gaussian PSNR: {psnr_gauss:.2f} dB")
    print(f"     - Bilateral PSNR: {psnr_bilat:.2f} dB")
    print(f"  3. Edge pixels (Canny):")
    print(f"     - Ground truth: {np.sum(edges_gt > 0)}")
    print(f"     - Gaussian    : {np.sum(edges_gauss > 0)} (lebih sedikit = blur)")
    print(f"     - Bilateral   : {np.sum(edges_bilat > 0)} (lebih dekat GT)")
    print("  4. Bilateral filter mempertahankan edge lebih baik")
    print("     karena memperhitungkan perbedaan intensitas")
    print("=" * 60)



if __name__ == "__main__":
    main()
