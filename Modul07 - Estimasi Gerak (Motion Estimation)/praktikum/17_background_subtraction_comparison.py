

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 17: PERBANDINGAN METODE BACKGROUND SUBTRACTION
    ==========================================================================
    Program ini membandingkan semua metode background subtraction yang umum
    digunakan dalam computer vision:
    1. Frame Differencing (sederhana)
    2. Running Average (adaptif)
    3. MOG2 (Mixture of Gaussians v2)
    4. KNN (K-Nearest Neighbors)

    Setiap metode dijalankan pada video yang sama dan dibandingkan dari segi:
    - Kualitas mask foreground (visual)
    - Jumlah piksel foreground yang terdeteksi
    - Kecepatan pemrosesan (waktu per frame)

    Fungsi utama yang dipelajari:
    - cv2.absdiff()                       : Frame differencing
    - cv2.accumulateWeighted()            : Running average
    - cv2.createBackgroundSubtractorMOG2(): Background subtractor MOG2
    - cv2.createBackgroundSubtractorKNN() : Background subtractor KNN
    - time.perf_counter()                 : Pengukuran waktu akurat

    Hasil: Perbandingan side-by-side semua metode
    ==========================================================================
    """

    # Mengimpor library OpenCV untuk pemrosesan video
    import cv2

    # Mengimpor NumPy untuk operasi array numerik
    import numpy as np

    # Mengimpor os untuk operasi path file
    import os

    # Mengimpor matplotlib untuk visualisasi
    import matplotlib.pyplot as plt

    # Mengimpor time untuk pengukuran waktu
    import time

    # Mendapatkan direktori script saat ini
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Mendefinisikan path folder gambar input dan output
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    # Membuat folder output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PERCOBAAN 17: PERBANDINGAN METODE BACKGROUND SUBTRACTION")
    print("=" * 60)

    # ============================================================
    # 1. Membuka video
    # ============================================================

    # Menggunakan video multi-objek untuk perbandingan
    video_path = os.path.join(IMAGE_DIR, "video_multi_objek.avi")
    cap = cv2.VideoCapture(video_path)

    # Memeriksa apakah video berhasil dibuka
    if not cap.isOpened():
        print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
        exit()

    # Membaca properti video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Video: {width}x{height}, FPS={fps}, Total={total_frames} frame")

    # ============================================================
    # 2. Inisialisasi semua metode background subtraction
    # ============================================================

    print("[INFO] Menginisialisasi 4 metode background subtraction...")

    # --- Metode 1: Frame Differencing ---
    # Keterangan: Membandingkan frame saat ini dengan frame sebelumnya
    # Kelebihan: Paling sederhana dan cepat
    # Kekurangan: Sensitif terhadap noise, hanya deteksi tepi gerakan
    DIFF_THRESHOLD = 30

    # --- Metode 2: Running Average ---
    # Keterangan: Memelihara model background sebagai rata-rata berjalan
    # cv2.accumulateWeighted() mengupdate: bg = α*frame + (1-α)*bg
    ALPHA_RUNNING = 0.05  # Learning rate (0.05 = lambat adaptasi)
    running_bg = None      # Model background (diinisialisasi nanti)

    # --- Metode 3: MOG2 (Mixture of Gaussians v2) ---
    # Keterangan: Memodelkan setiap piksel background sebagai campuran Gaussian
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500,            # Jumlah frame untuk training
        varThreshold=16,        # Threshold varian (sensitivitas)
        detectShadows=True      # Juga deteksi bayangan
    )

    # --- Metode 4: KNN (K-Nearest Neighbors) ---
    # Keterangan: Mengklasifikasi piksel foreground/background menggunakan KNN
    knn = cv2.createBackgroundSubtractorKNN(
        history=500,            # Jumlah frame untuk training
        dist2Threshold=400.0,   # Threshold jarak kuadrat
        detectShadows=True      # Deteksi bayangan
    )

    print("  [OK] Frame Differencing (threshold={})".format(DIFF_THRESHOLD))
    print("  [OK] Running Average (alpha={})".format(ALPHA_RUNNING))
    print("  [OK] MOG2 (history=500, varThreshold=16)")
    print("  [OK] KNN (history=500, dist2Threshold=400)")

    # ============================================================
    # 3. Memproses video dan menerapkan semua metode
    # ============================================================

    # Menyimpan waktu pemrosesan untuk setiap metode
    times = {'Frame Diff': [], 'Running Avg': [], 'MOG2': [], 'KNN': []}

    # Menyimpan jumlah piksel foreground per frame
    fg_counts = {'Frame Diff': [], 'Running Avg': [], 'MOG2': [], 'KNN': []}

    # Menyimpan beberapa frame untuk visualisasi
    vis_frames = []
    vis_masks = {'Frame Diff': [], 'Running Avg': [], 'MOG2': [], 'KNN': []}
    vis_indices = []
    target_vis = [10, 30, 60, 90]

    # Membaca frame pertama untuk inisialisasi
    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame pertama.")
        exit()

    # Mengkonversi frame pertama ke grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Inisialisasi running average background
    running_bg = prev_gray.astype(np.float32)

    print("[INFO] Memproses video dengan 4 metode...")

    frame_count = 0

    while True:
        # Membaca frame berikutnya
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Mengkonversi ke grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Metode 1: Frame Differencing ---
        t_start = time.perf_counter()
        # Menghitung perbedaan absolut antara frame sekarang dan sebelumnya
        diff = cv2.absdiff(curr_gray, prev_gray)
        # Menerapkan threshold untuk membuat binary mask
        _, mask_diff = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        t_diff = time.perf_counter() - t_start
        times['Frame Diff'].append(t_diff)
        fg_counts['Frame Diff'].append(np.count_nonzero(mask_diff))

        # --- Metode 2: Running Average ---
        t_start = time.perf_counter()
        # Mengupdate model background: bg = α * frame + (1-α) * bg
        cv2.accumulateWeighted(curr_gray, running_bg, ALPHA_RUNNING)
        # Menghitung perbedaan antara frame dan model background
        bg_diff = cv2.absdiff(curr_gray, running_bg.astype(np.uint8))
        # Menerapkan threshold
        _, mask_running = cv2.threshold(bg_diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        t_running = time.perf_counter() - t_start
        times['Running Avg'].append(t_running)
        fg_counts['Running Avg'].append(np.count_nonzero(mask_running))

        # --- Metode 3: MOG2 ---
        t_start = time.perf_counter()
        # Menerapkan background subtractor MOG2
        mask_mog2 = mog2.apply(frame)
        # Membersihkan bayangan (nilai 127 → 0)
        _, mask_mog2_clean = cv2.threshold(mask_mog2, 200, 255, cv2.THRESH_BINARY)
        t_mog2 = time.perf_counter() - t_start
        times['MOG2'].append(t_mog2)
        fg_counts['MOG2'].append(np.count_nonzero(mask_mog2_clean))

        # --- Metode 4: KNN ---
        t_start = time.perf_counter()
        # Menerapkan background subtractor KNN
        mask_knn = knn.apply(frame)
        # Membersihkan bayangan
        _, mask_knn_clean = cv2.threshold(mask_knn, 200, 255, cv2.THRESH_BINARY)
        t_knn = time.perf_counter() - t_start
        times['KNN'].append(t_knn)
        fg_counts['KNN'].append(np.count_nonzero(mask_knn_clean))

        # Menyimpan snapshot untuk visualisasi
        if frame_count in target_vis:
            vis_frames.append(frame.copy())
            vis_masks['Frame Diff'].append(mask_diff.copy())
            vis_masks['Running Avg'].append(mask_running.copy())
            vis_masks['MOG2'].append(mask_mog2_clean.copy())
            vis_masks['KNN'].append(mask_knn_clean.copy())
            vis_indices.append(frame_count)

        # Memperbarui frame sebelumnya
        prev_gray = curr_gray.copy()

    # Menutup video capture
    cap.release()

    print(f"[INFO] Selesai memproses {frame_count} frame.")

    # ============================================================
    # 4. Visualisasi perbandingan side-by-side
    # ============================================================

    # Membuat figure: perbandingan mask pada beberapa frame
    num_vis = min(4, len(vis_frames))
    fig1, axes1 = plt.subplots(5, num_vis, figsize=(4 * num_vis, 18))

    method_names = ['Frame Diff', 'Running Avg', 'MOG2', 'KNN']

    for col in range(num_vis):
        # Baris 0: Frame asli
        axes1[0, col].imshow(cv2.cvtColor(vis_frames[col], cv2.COLOR_BGR2RGB))
        axes1[0, col].set_title(f"Frame #{vis_indices[col]}", fontsize=10)
        axes1[0, col].axis("off")

        # Baris 1-4: Mask dari setiap metode
        for row, method in enumerate(method_names):
            axes1[row + 1, col].imshow(vis_masks[method][col], cmap='gray')
            fg_pct = (np.count_nonzero(vis_masks[method][col]) / (width * height)) * 100
            axes1[row + 1, col].set_title(f"{method}\nFG: {fg_pct:.1f}%", fontsize=9)
            axes1[row + 1, col].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Percobaan 17: Perbandingan Metode Background Subtraction\n"
                 "Putih = Foreground (objek bergerak)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Menyimpan figure pertama
    output_path_1 = os.path.join(OUTPUT_DIR, "17_bg_subtraction_comparison.png")
    plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Perbandingan visual disimpan di: {output_path_1}")

    # ============================================================
    # 5. Visualisasi metrik: foreground pixels dan timing
    # ============================================================

    # Membuat figure: grafik metrik
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))

    # Warna untuk setiap metode
    colors = {'Frame Diff': 'blue', 'Running Avg': 'green', 'MOG2': 'red', 'KNN': 'orange'}

    # Subplot 1: Jumlah piksel foreground per frame
    for method in method_names:
        axes2[0].plot(fg_counts[method], color=colors[method], label=method, alpha=0.7, linewidth=1.5)
    axes2[0].set_xlabel("Nomor Frame", fontsize=10)
    axes2[0].set_ylabel("Jumlah Piksel Foreground", fontsize=10)
    axes2[0].set_title("Piksel Foreground per Frame", fontsize=12)
    axes2[0].legend(fontsize=9)
    axes2[0].grid(True, alpha=0.3)

    # Subplot 2: Waktu pemrosesan per frame
    for method in method_names:
        # Mengkonversi ke milidetik
        times_ms = [t * 1000 for t in times[method]]
        axes2[1].plot(times_ms, color=colors[method], label=method, alpha=0.7, linewidth=1.5)
    axes2[1].set_xlabel("Nomor Frame", fontsize=10)
    axes2[1].set_ylabel("Waktu (milidetik)", fontsize=10)
    axes2[1].set_title("Waktu Pemrosesan per Frame", fontsize=12)
    axes2[1].legend(fontsize=9)
    axes2[1].grid(True, alpha=0.3)

    # Menambahkan judul utama
    plt.suptitle("Percobaan 17: Metrik Perbandingan Background Subtraction",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Menyimpan figure kedua
    output_path_2 = os.path.join(OUTPUT_DIR, "17_bg_subtraction_metrics.png")
    plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[OUTPUT] Metrik perbandingan disimpan di: {output_path_2}")

    # ============================================================
    # 6. Menampilkan tabel ringkasan
    # ============================================================

    print(f"\n[INFO] ============ TABEL PERBANDINGAN ============")
    print(f"{'Metode':<15} {'Rata-rata FG':<15} {'Rata-rata Waktu':<18} {'Maks Waktu':<15}")
    print("-" * 63)

    for method in method_names:
        avg_fg = np.mean(fg_counts[method])
        avg_time = np.mean(times[method]) * 1000  # Milidetik
        max_time = np.max(times[method]) * 1000
        print(f"{method:<15} {avg_fg:<15.0f} {avg_time:<18.3f} ms {max_time:<15.3f} ms")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 17")
    print("=" * 60)
    print("Metode yang dibandingkan:")
    print("  1. Frame Differencing:")
    print("     - cv2.absdiff() + cv2.threshold()")
    print("     - Paling sederhana, deteksi tepi gerakan saja")
    print("  2. Running Average:")
    print("     - cv2.accumulateWeighted() → model background adaptif")
    print("     - bg = α*frame + (1-α)*bg")
    print("  3. MOG2:")
    print("     - cv2.createBackgroundSubtractorMOG2()")
    print("     - Model campuran Gaussian, deteksi bayangan")
    print("  4. KNN:")
    print("     - cv2.createBackgroundSubtractorKNN()")
    print("     - Klasifikasi berbasis K-Nearest Neighbors")
    print("Kesimpulan:")
    print("  - MOG2/KNN lebih akurat tapi lebih lambat")
    print("  - Frame diff paling cepat tapi paling kasar")
    print("  - Running average = keseimbangan sederhana-akurat")
    print("=" * 60)



if __name__ == "__main__":
    main()
