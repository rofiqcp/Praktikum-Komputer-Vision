

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 19: SIMULASI OPTICAL FLOW REAL-TIME
    ==========================================================================
    Program ini mensimulasikan pemrosesan optical flow secara real-time.
    Video dibaca frame per frame dengan pengukuran waktu yang akurat.
    Perbandingan kecepatan antara sparse (Lucas-Kanade) dan dense (Farneback)
    optical flow dilakukan untuk memahami trade-off masing-masing metode.

    Fitur yang disimulasikan:
    1. Pemrosesan frame-by-frame dengan timing
    2. Penghitungan dan tampilan FPS
    3. Perbandingan waktu: sparse LK vs dense Farneback
    4. Visualisasi flow secara real-time-like

    Fungsi utama yang dipelajari:
    - cv2.calcOpticalFlowPyrLK()       : Sparse optical flow (cepat)
    - cv2.calcOpticalFlowFarneback()   : Dense optical flow (lambat)
    - cv2.goodFeaturesToTrack()        : Deteksi fitur untuk sparse flow
    - time.perf_counter()              : Pengukuran waktu presisi tinggi

    Hasil: Perbandingan kecepatan dan kualitas sparse vs dense flow
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

    # Mengimpor time untuk pengukuran waktu presisi
    import time

    # Mendapatkan direktori script saat ini
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Mendefinisikan path folder gambar input dan output
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    # Membuat folder output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PERCOBAAN 19: SIMULASI OPTICAL FLOW REAL-TIME")
    print("=" * 60)

    # ============================================================
    # 1. Membuka video
    # ============================================================

    # Menggunakan video bola untuk simulasi
    video_path = os.path.join(IMAGE_DIR, "video_bola.avi")
    cap = cv2.VideoCapture(video_path)

    # Memeriksa apakah video berhasil dibuka
    if not cap.isOpened():
        print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
        exit()

    # Membaca properti video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Video: {width}x{height}, FPS video={fps_video}, Total={total_frames} frame")

    # ============================================================
    # 2. Inisialisasi parameter
    # ============================================================

    # Parameter untuk deteksi fitur (sparse flow)
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # Parameter untuk Lucas-Kanade
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Membaca frame pertama
    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame pertama.")
        exit()

    # Mengkonversi ke grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi fitur awal untuk sparse flow
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    print(f"[INFO] Fitur awal terdeteksi: {len(prev_pts) if prev_pts is not None else 0}")

    # ============================================================
    # 3. Pemrosesan real-time: Sparse LK
    # ============================================================

    print("\n[INFO] === Fase 1: Sparse Lucas-Kanade ===")

    # Variabel penyimpan waktu dan FPS untuk sparse
    times_sparse = []
    fps_sparse_list = []
    flow_magnitude_sparse = []

    # Variabel untuk snapshot
    sparse_snapshots = []
    sparse_frame_nums = []

    # Inisial titik untuk re-tracking
    re_detect_interval = 30  # Re-detect fitur setiap 30 frame
    frame_count = 0

    # Membuat mask untuk visualisasi trajectory
    mask_draw = np.zeros_like(prev_frame)
    colors = np.random.randint(0, 255, (100, 3))

    while True:
        # Membaca frame berikutnya
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Mengkonversi ke grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mengukur waktu pemrosesan sparse flow
        t_start = time.perf_counter()

        # Re-detect fitur setiap interval tertentu
        if prev_pts is None or len(prev_pts) < 10 or frame_count % re_detect_interval == 0:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            mask_draw = np.zeros_like(prev_frame)  # Reset trajectory

        if prev_pts is not None and len(prev_pts) > 0:
            # Menghitung sparse optical flow LK
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **lk_params
            )

            # Memfilter titik yang berhasil di-track
            if next_pts is not None:
                good_new = next_pts[status == 1]
                good_old = prev_pts[status == 1]

                # Menghitung magnitude rata-rata flow
                if len(good_new) > 0 and len(good_old) > 0:
                    diff = good_new - good_old
                    mag = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
                    flow_magnitude_sparse.append(mag.mean())
                else:
                    flow_magnitude_sparse.append(0)

                # Menggambar trajectory
                vis_frame = frame.copy()
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    mask_draw = cv2.line(mask_draw, (a, b), (c, d),
                                          colors[i % 100].tolist(), 2)
                    vis_frame = cv2.circle(vis_frame, (a, b), 3,
                                            colors[i % 100].tolist(), -1)

                vis_frame = cv2.add(vis_frame, mask_draw)
                prev_pts = good_new.reshape(-1, 1, 2)
            else:
                vis_frame = frame.copy()
                flow_magnitude_sparse.append(0)
        else:
            vis_frame = frame.copy()
            flow_magnitude_sparse.append(0)

        # Mengukur waktu selesai
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        times_sparse.append(elapsed)

        # Menghitung FPS
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_sparse_list.append(current_fps)

        # Menambahkan FPS counter pada gambar
        cv2.putText(vis_frame, f"Sparse LK | FPS: {current_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Frame: {frame_count}/{total_frames}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Menyimpan snapshot
        if frame_count in [10, 30, 60]:
            sparse_snapshots.append(vis_frame.copy())
            sparse_frame_nums.append(frame_count)

        # Memperbarui frame sebelumnya
        prev_gray = curr_gray.copy()

    print(f"[INFO] Sparse LK selesai: {frame_count} frame")
    print(f"[INFO] FPS rata-rata Sparse: {np.mean(fps_sparse_list):.1f}")

    # ============================================================
    # 4. Pemrosesan real-time: Dense Farneback
    # ============================================================

    print("\n[INFO] === Fase 2: Dense Farneback ===")

    # Reset video ke awal
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Variabel penyimpan waktu dan FPS untuk dense
    times_dense = []
    fps_dense_list = []
    flow_magnitude_dense = []

    # Variabel untuk snapshot
    dense_snapshots = []
    dense_frame_nums = []

    frame_count = 0

    while True:
        # Membaca frame berikutnya
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Mengkonversi ke grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mengukur waktu pemrosesan dense flow
        t_start = time.perf_counter()

        # Menghitung dense optical flow Farneback
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Menghitung magnitude dan arah flow
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

        # Mengukur waktu selesai
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        times_dense.append(elapsed)

        # Meyimpan rata-rata magnitude
        flow_magnitude_dense.append(mag.mean())

        # Menghitung FPS
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_dense_list.append(current_fps)

        # Membuat visualisasi flow sebagai HSV
        hsv = np.zeros_like(prev_frame)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Menambahkan FPS counter
        cv2.putText(flow_vis, f"Dense Farneback | FPS: {current_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(flow_vis, f"Frame: {frame_count}/{total_frames}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Menyimpan snapshot
        if frame_count in [10, 30, 60]:
            dense_snapshots.append(flow_vis.copy())
            dense_frame_nums.append(frame_count)

        # Memperbarui frame sebelumnya
        prev_gray = curr_gray.copy()

    # Menutup video capture
    cap.release()

    print(f"[INFO] Dense Farneback selesai: {frame_count} frame")
    print(f"[INFO] FPS rata-rata Dense: {np.mean(fps_dense_list):.1f}")

    # ============================================================
    # 5. Visualisasi perbandingan snapshot
    # ============================================================

    # Menentukan jumlah snapshot yang tersedia
    num_snap = min(len(sparse_snapshots), len(dense_snapshots), 3)

    fig1, axes1 = plt.subplots(2, num_snap, figsize=(6 * num_snap, 8))

    if num_snap == 1:
        axes1 = axes1.reshape(2, 1)

    for idx in range(num_snap):
        # Baris atas: Sparse LK
        axes1[0, idx].imshow(cv2.cvtColor(sparse_snapshots[idx], cv2.COLOR_BGR2RGB))
        axes1[0, idx].set_title(f"Sparse LK - Frame #{sparse_frame_nums[idx]}", fontsize=10)
        axes1[0, idx].axis("off")

        # Baris bawah: Dense Farneback
        axes1[1, idx].imshow(cv2.cvtColor(dense_snapshots[idx], cv2.COLOR_BGR2RGB))
        axes1[1, idx].set_title(f"Dense Farneback - Frame #{dense_frame_nums[idx]}", fontsize=10)
        axes1[1, idx].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Percobaan 19: Simulasi Real-Time Optical Flow\n"
                 "Atas = Sparse LK (trajectory), Bawah = Dense Farneback (HSV)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Menyimpan figure pertama
    output_path_1 = os.path.join(OUTPUT_DIR, "19_realtime_snapshots.png")
    plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Snapshot disimpan di: {output_path_1}")

    # ============================================================
    # 6. Visualisasi perbandingan kecepatan
    # ============================================================

    # Membuat figure: perbandingan metrik
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: FPS per frame
    min_len = min(len(fps_sparse_list), len(fps_dense_list))
    axes2[0, 0].plot(fps_sparse_list[:min_len], color='blue', alpha=0.7, label='Sparse LK', linewidth=1)
    axes2[0, 0].plot(fps_dense_list[:min_len], color='red', alpha=0.7, label='Dense Farneback', linewidth=1)
    axes2[0, 0].set_xlabel("Nomor Frame", fontsize=10)
    axes2[0, 0].set_ylabel("FPS", fontsize=10)
    axes2[0, 0].set_title("FPS per Frame (semakin tinggi = lebih cepat)", fontsize=11)
    axes2[0, 0].legend(fontsize=9)
    axes2[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Waktu pemrosesan per frame (milidetik)
    times_sparse_ms = [t * 1000 for t in times_sparse[:min_len]]
    times_dense_ms = [t * 1000 for t in times_dense[:min_len]]
    axes2[0, 1].plot(times_sparse_ms, color='blue', alpha=0.7, label='Sparse LK', linewidth=1)
    axes2[0, 1].plot(times_dense_ms, color='red', alpha=0.7, label='Dense Farneback', linewidth=1)
    axes2[0, 1].set_xlabel("Nomor Frame", fontsize=10)
    axes2[0, 1].set_ylabel("Waktu (ms)", fontsize=10)
    axes2[0, 1].set_title("Waktu Pemrosesan per Frame", fontsize=11)
    axes2[0, 1].legend(fontsize=9)
    axes2[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Perbandingan FPS rata-rata (bar chart)
    avg_fps = [np.mean(fps_sparse_list), np.mean(fps_dense_list)]
    bar_colors = ['steelblue', 'coral']
    bars = axes2[1, 0].bar(['Sparse LK', 'Dense Farneback'], avg_fps, color=bar_colors, alpha=0.7)
    axes2[1, 0].set_ylabel("FPS Rata-rata", fontsize=10)
    axes2[1, 0].set_title("Perbandingan FPS Rata-rata", fontsize=11)
    axes2[1, 0].grid(True, alpha=0.3, axis='y')
    # Menambahkan label di atas bar
    for bar, val in zip(bars, avg_fps):
        axes2[1, 0].text(bar.get_x() + bar.get_width()/2, val + 1,
                          f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

    # Subplot 4: Magnitude flow rata-rata
    min_mag = min(len(flow_magnitude_sparse), len(flow_magnitude_dense))
    axes2[1, 1].plot(flow_magnitude_sparse[:min_mag], color='blue', alpha=0.7,
                      label='Sparse LK', linewidth=1)
    axes2[1, 1].plot(flow_magnitude_dense[:min_mag], color='red', alpha=0.7,
                      label='Dense Farneback', linewidth=1)
    axes2[1, 1].set_xlabel("Nomor Frame", fontsize=10)
    axes2[1, 1].set_ylabel("Magnitude Rata-rata", fontsize=10)
    axes2[1, 1].set_title("Flow Magnitude Rata-rata per Frame", fontsize=11)
    axes2[1, 1].legend(fontsize=9)
    axes2[1, 1].grid(True, alpha=0.3)

    # Menambahkan judul utama
    plt.suptitle("Percobaan 19: Perbandingan Kecepatan — Sparse LK vs Dense Farneback",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Menyimpan figure kedua
    output_path_2 = os.path.join(OUTPUT_DIR, "19_realtime_performance.png")
    plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[OUTPUT] Perbandingan kecepatan disimpan di: {output_path_2}")

    # Menampilkan ringkasan kecepatan
    print(f"\n[INFO] ============ PERBANDINGAN KECEPATAN ============")
    print(f"{'Metode':<20} {'FPS Rata²':<12} {'Waktu Rata²':<15} {'Speedup':<10}")
    print("-" * 57)
    avg_sparse = np.mean(times_sparse) * 1000
    avg_dense = np.mean(times_dense) * 1000
    speedup = avg_dense / avg_sparse if avg_sparse > 0 else 0
    print(f"{'Sparse LK':<20} {np.mean(fps_sparse_list):<12.1f} {avg_sparse:<15.2f} ms {'1.0x':<10}")
    print(f"{'Dense Farneback':<20} {np.mean(fps_dense_list):<12.1f} {avg_dense:<15.2f} ms {f'{speedup:.1f}x lebih lambat':<10}")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 19")
    print("=" * 60)
    print("Fungsi yang dipelajari:")
    print("  1. cv2.calcOpticalFlowPyrLK()     → Sparse flow (cepat)")
    print("     - Hanya menghitung di titik fitur")
    print("     - Cocok untuk real-time tracking")
    print("  2. cv2.calcOpticalFlowFarneback()  → Dense flow (lambat)")
    print("     - Menghitung flow untuk SEMUA piksel")
    print("     - Lebih detail tapi lebih berat")
    print("  3. time.perf_counter()             → Pengukuran waktu presisi")
    print("Perbandingan:")
    print("  Sparse LK  : Cepat, cocok real-time, hanya titik fitur")
    print("  Dense Farnb : Lambat, detail penuh, semua piksel")
    print("  Trade-off   : Kecepatan vs Detail")
    print(f"  Speedup LK  : ~{speedup:.1f}x lebih cepat dari Farneback")
    print("=" * 60)



if __name__ == "__main__":
    main()
