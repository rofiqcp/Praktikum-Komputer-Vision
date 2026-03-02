

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 10: SUPER RESOLUTION — INTERPOLASI KLASIK
    ==========================================================================
    Program ini mempelajari teknik super resolution menggunakan metode
    interpolasi klasik. Super resolution bertujuan meningkatkan resolusi gambar
    (upscaling) dengan kualitas sebaik mungkin.

    Metode interpolasi yang dibandingkan:
    - INTER_NEAREST  : Tetangga terdekat (cepat, blok-blok/pixelated)
    - INTER_LINEAR   : Bilinear (default, halus tapi blur)
    - INTER_CUBIC    : Bicubic (lebih tajam dari bilinear)
    - INTER_LANCZOS4 : Lanczos (kualitas tinggi, paling lambat)

    Fungsi utama yang dipelajari:
    - cv2.resize(src, dsize, interpolation=...)
    - cv2.PSNR() untuk perbandingan kualitas
    - Perbandingan 4 metode interpolasi side-by-side

    Hasil: Perbandingan visual dan PSNR upscaling 4x dengan 4 metode
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

    # Mengimpor time untuk mengukur waktu eksekusi
    import time

    # Mendapatkan direktori tempat script ini berada
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Mendefinisikan path folder gambar input
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

    # Mendefinisikan path folder output
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    # Membuat folder output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PERCOBAAN 10: SUPER RESOLUTION — INTERPOLASI KLASIK")
    print("=" * 60)

    # ============================================================
    # 1. Membaca gambar low resolution dan ground truth
    # ============================================================

    # Membaca gambar resolusi rendah
    lr_path = os.path.join(IMAGE_DIR, "low_resolution.png")
    lr_img = cv2.imread(lr_path)

    # Validasi pembacaan
    if lr_img is None:
        print(f"[ERROR] Gagal membaca: {lr_path}")
        print("[INFO] Jalankan download_image.py terlebih dahulu!")
        exit()

    # Membaca gambar asli (ground truth / high resolution)
    gt_path = os.path.join(IMAGE_DIR, "scene_pemandangan.png")
    gt_img = cv2.imread(gt_path)

    # Validasi ground truth
    if gt_img is None:
        print(f"[ERROR] Gagal membaca: {gt_path}")
        exit()

    # Menampilkan info dimensi
    print(f"[INFO] Low resolution  : {lr_img.shape[1]}x{lr_img.shape[0]}")
    print(f"[INFO] Ground truth    : {gt_img.shape[1]}x{gt_img.shape[0]}")

    # ============================================================
    # 2. Menentukan faktor upscaling
    # ============================================================

    # Faktor upscaling 4x
    scale_factor = 4

    # Menghitung ukuran target (4x dari low resolution)
    target_width = lr_img.shape[1] * scale_factor
    target_height = lr_img.shape[0] * scale_factor
    target_size = (target_width, target_height)

    print(f"\n[INFO] Faktor upscaling: {scale_factor}x")
    print(f"[INFO] Target ukuran   : {target_width}x{target_height}")

    # Menyesuaikan ground truth ke ukuran target untuk PSNR
    gt_resized = cv2.resize(gt_img, target_size, interpolation=cv2.INTER_LANCZOS4)

    # ============================================================
    # 3. Upscaling dengan 4 metode interpolasi
    # cv2.resize(src, (width, height), interpolation=method)
    # ============================================================

    # Mendefinisikan metode interpolasi yang akan dibandingkan
    methods = [
        (cv2.INTER_NEAREST, "INTER_NEAREST",
         "Tetangga terdekat\nPiksel terdekat langsung disalin\nBlok-blok/pixelated"),
        (cv2.INTER_LINEAR, "INTER_LINEAR",
         "Bilinear interpolation\nRata-rata 4 tetangga\nHalus tapi agak blur"),
        (cv2.INTER_CUBIC, "INTER_CUBIC",
         "Bicubic interpolation\nPolinom kubik 16 tetangga\nLebih tajam dari bilinear"),
        (cv2.INTER_LANCZOS4, "INTER_LANCZOS4",
         "Lanczos resampling\nKernel sinc 64 piksel\nKualitas tertinggi"),
    ]

    # Menyimpan hasil upscaling
    results = []
    psnr_values = []
    exec_times = []

    print(f"\n--- Upscaling {scale_factor}x dengan 4 metode ---")
    for method_flag, method_name, method_desc in methods:
        # Mengukur waktu eksekusi
        start = time.time()

        # Melakukan upscaling dengan metode tertentu
        upscaled = cv2.resize(
            lr_img,                    # Gambar input (low resolution)
            target_size,               # Ukuran target (width, height)
            interpolation=method_flag  # Metode interpolasi
        )

        # Menghitung waktu eksekusi
        elapsed = time.time() - start

        # Menghitung PSNR terhadap ground truth
        psnr_val = cv2.PSNR(gt_resized, upscaled)

        # Menyimpan hasil
        results.append((upscaled, method_name, method_desc))
        psnr_values.append(psnr_val)
        exec_times.append(elapsed)

        print(f"  {method_name:18s} — PSNR: {psnr_val:.2f} dB, Waktu: {elapsed:.4f}s")

    # ============================================================
    # 4. Analisis detail: Zoom pada area tertentu
    # ============================================================

    # Menentukan area zoom (bagian tengah gambar)
    zoom_y1 = target_height // 3
    zoom_y2 = zoom_y1 + target_height // 4
    zoom_x1 = target_width // 3
    zoom_x2 = zoom_x1 + target_width // 4

    print(f"\n[INFO] Area zoom: ({zoom_x1},{zoom_y1}) - ({zoom_x2},{zoom_y2})")

    # Crop area zoom dari setiap hasil
    zoom_results = []
    for upscaled, name, desc in results:
        # Memotong area zoom
        crop = upscaled[zoom_y1:zoom_y2, zoom_x1:zoom_x2]
        zoom_results.append(crop)

    # Crop area zoom dari ground truth
    zoom_gt = gt_resized[zoom_y1:zoom_y2, zoom_x1:zoom_x2]

    # Crop area zoom dari LR (juga di-upscale dulu)
    lr_upscaled_nearest = cv2.resize(lr_img, target_size, interpolation=cv2.INTER_NEAREST)
    zoom_lr = lr_upscaled_nearest[zoom_y1:zoom_y2, zoom_x1:zoom_x2]

    # ============================================================
    # 5. Menghitung perbedaan absolut (error map)
    # ============================================================

    # Menghitung error map untuk setiap metode (grayscale)
    error_maps = []
    for upscaled, name, desc in results:
        # Menghitung perbedaan absolut per piksel
        diff = cv2.absdiff(gt_resized, upscaled)

        # Konversi ke grayscale untuk visualisasi
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Normalisasi untuk visualisasi yang lebih jelas
        diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)

        error_maps.append(diff_norm)

    # ============================================================
    # 6. Visualisasi hasil
    # ============================================================

    # Membuat figure 4 baris x 4 kolom
    fig, axes = plt.subplots(4, 4, figsize=(22, 22))

    # --- Baris 1: Hasil upscaling full image ---
    for i, (upscaled, name, desc) in enumerate(results):
        rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"{name}\nPSNR: {psnr_values[i]:.2f} dB", fontsize=9)
        axes[0, i].axis("off")

    # --- Baris 2: Zoom detail ---
    for i in range(4):
        zoom_rgb = cv2.cvtColor(zoom_results[i], cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(zoom_rgb)
        axes[1, i].set_title(f"Zoom: {results[i][1]}", fontsize=9)
        axes[1, i].axis("off")

    # --- Baris 3: Error maps dan perbandingan ---
    for i in range(4):
        axes[2, i].imshow(error_maps[i], cmap='hot')
        axes[2, i].set_title(f"Error: {results[i][1]}\n"
                             f"Mean err: {np.mean(error_maps[i]):.1f}", fontsize=9)
        axes[2, i].axis("off")

    # --- Baris 4: LR asli, GT, grafik PSNR, grafik waktu ---
    # Low resolution (asli)
    lr_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
    axes[3, 0].imshow(lr_rgb)
    axes[3, 0].set_title(f"Low Resolution\n{lr_img.shape[1]}x{lr_img.shape[0]}", fontsize=9)
    axes[3, 0].axis("off")

    # Ground truth
    gt_rgb = cv2.cvtColor(gt_resized, cv2.COLOR_BGR2RGB)
    axes[3, 1].imshow(gt_rgb)
    axes[3, 1].set_title(f"Ground Truth\n{target_width}x{target_height}", fontsize=9)
    axes[3, 1].axis("off")

    # Bar chart PSNR
    method_names = [r[1].replace("INTER_", "") for r in results]
    colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800']
    bars = axes[3, 2].bar(method_names, psnr_values, color=colors)
    axes[3, 2].set_title("PSNR per Metode (dB)", fontsize=9)
    axes[3, 2].set_ylabel("PSNR (dB)")
    axes[3, 2].grid(axis='y', alpha=0.3)
    axes[3, 2].tick_params(labelsize=8)
    # Menambahkan label PSNR di atas bar
    for bar, val in zip(bars, psnr_values):
        axes[3, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', fontsize=8)

    # Bar chart waktu eksekusi
    bars_t = axes[3, 3].bar(method_names, [t*1000 for t in exec_times], color=colors)
    axes[3, 3].set_title("Waktu Eksekusi (ms)", fontsize=9)
    axes[3, 3].set_ylabel("Waktu (ms)")
    axes[3, 3].grid(axis='y', alpha=0.3)
    axes[3, 3].tick_params(labelsize=8)
    for bar, val in zip(bars_t, exec_times):
        axes[3, 3].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val*1000:.2f}', ha='center', fontsize=8)

    # Menambahkan judul utama
    plt.suptitle(f"Percobaan 10: Super Resolution — Interpolasi Klasik ({scale_factor}x Upscale)\n"
                 "NEAREST vs LINEAR vs CUBIC vs LANCZOS4",
                 fontsize=14, fontweight="bold")

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan hasil
    output_path = os.path.join(OUTPUT_DIR, "10_super_resolution_interpolasi.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 10")
    print("=" * 60)
    print("Fungsi yang dipelajari:")
    print("  1. cv2.resize(src, (w, h), interpolation=method)")
    print(f"     - Upscaling {scale_factor}x: {lr_img.shape[1]}x{lr_img.shape[0]} → "
          f"{target_width}x{target_height}")
    print("  2. Metode interpolasi:")
    print(f"     - INTER_NEAREST  : PSNR={psnr_values[0]:.2f} dB, "
          f"Waktu={exec_times[0]*1000:.2f}ms")
    print(f"     - INTER_LINEAR   : PSNR={psnr_values[1]:.2f} dB, "
          f"Waktu={exec_times[1]*1000:.2f}ms")
    print(f"     - INTER_CUBIC    : PSNR={psnr_values[2]:.2f} dB, "
          f"Waktu={exec_times[2]*1000:.2f}ms")
    print(f"     - INTER_LANCZOS4 : PSNR={psnr_values[3]:.2f} dB, "
          f"Waktu={exec_times[3]*1000:.2f}ms")
    print("  3. Kesimpulan:")
    print("     - NEAREST: Paling cepat, kualitas terendah (pixelated)")
    print("     - LINEAR : Default, keseimbangan kecepatan dan kualitas")
    print("     - CUBIC  : Kualitas baik, sedikit lebih lambat")
    print("     - LANCZOS4: Kualitas terbaik, paling lambat")
    print("  4. Untuk upscaling besar, metode deep learning (EDSR, ESRGAN)")
    print("     jauh lebih baik dari interpolasi klasik")
    print("=" * 60)



if __name__ == "__main__":
    main()
