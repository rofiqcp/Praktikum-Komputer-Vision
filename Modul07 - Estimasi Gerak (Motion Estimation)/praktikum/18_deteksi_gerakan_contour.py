

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 18: DETEKSI GERAKAN DENGAN ANALISIS KONTUR
    ==========================================================================
    Program ini mempelajari cara mendeteksi objek bergerak menggunakan
    kombinasi background subtraction dan analisis kontur. Setelah mendapatkan
    binary mask dari background subtraction, kontur diekstrak dan difilter
    untuk mendeteksi objek bergerak sesungguhnya.

    Pipeline Deteksi Gerakan:
    1. Background subtraction → binary mask foreground
    2. Morphological operations → membersihkan noise
    3. cv2.findContours() → menemukan kontur objek
    4. Filter kontur berdasarkan area (membuang noise kecil)
    5. Menggambar bounding box dan label pada setiap objek

    Fungsi utama yang dipelajari:
    - cv2.createBackgroundSubtractorMOG2()  : Background subtraction
    - cv2.morphologyEx()                    : Operasi morfologi (bersihkan noise)
    - cv2.findContours()                    : Menemukan kontur pada mask
    - cv2.contourArea()                     : Menghitung luas kontur
    - cv2.boundingRect()                    : Bounding box kontur
    - cv2.rectangle()                       : Menggambar kotak
    - cv2.putText()                         : Menambahkan teks label

    Hasil: Deteksi objek bergerak dengan bounding box dan label
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

    # Mendapatkan direktori script saat ini
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Mendefinisikan path folder gambar input dan output
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    # Membuat folder output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PERCOBAAN 18: DETEKSI GERAKAN DENGAN ANALISIS KONTUR")
    print("=" * 60)

    # ============================================================
    # 1. Membuka video dan inisialisasi background subtractor
    # ============================================================

    # Menggunakan video multi-objek bergerak
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

    # Membuat background subtractor MOG2
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,           # Jumlah frame untuk belajar background
        varThreshold=50,       # Threshold varian (lebih tinggi = kurang sensitif)
        detectShadows=True     # Mendeteksi bayangan
    )

    # ============================================================
    # 2. Mendefinisikan parameter deteksi
    # ============================================================

    # Area minimum kontur untuk dianggap sebagai objek (dalam piksel²)
    MIN_CONTOUR_AREA = 500

    # Kernel untuk operasi morfologi (membersihkan noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Kernel besar untuk closing (menutup celah)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    print(f"[INFO] Area kontur minimum: {MIN_CONTOUR_AREA} piksel²")
    print(f"[INFO] Kernel morfologi: ellipse 5x5 dan 15x15")

    # ============================================================
    # 3. Memproses video frame per frame
    # ============================================================

    print("[INFO] Memproses video untuk deteksi gerakan...")

    # Menyimpan statistik per frame
    objects_per_frame = []

    # Menyimpan frame untuk visualisasi
    vis_data = []  # list of (frame_num, original, mask, detected)
    target_frames = [15, 30, 60, 90]

    frame_count = 0

    while True:
        # Membaca frame berikutnya
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ============================================================
        # Langkah 3a: Background subtraction
        # Menghasilkan mask foreground (255 = foreground, 127 = bayangan, 0 = bg)
        # ============================================================
        fg_mask = bg_subtractor.apply(frame)

        # Menghapus bayangan (nilai 127) agar hanya foreground murni
        _, fg_mask_clean = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # ============================================================
        # Langkah 3b: Operasi morfologi untuk membersihkan mask
        # ============================================================

        # Opening = erode + dilate → menghilangkan noise kecil
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_OPEN, kernel)

        # Closing = dilate + erode → menutup celah di dalam objek
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, kernel_close)

        # Dilasi tambahan untuk memperbesar area objek
        fg_mask_clean = cv2.dilate(fg_mask_clean, kernel, iterations=2)

        # ============================================================
        # Langkah 3c: Menemukan kontur pada mask
        # cv2.findContours() menemukan semua kontur (batas objek)
        # RETR_EXTERNAL → hanya kontur terluar
        # CHAIN_APPROX_SIMPLE → kompres titik kontur
        # ============================================================
        contours, _ = cv2.findContours(
            fg_mask_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ============================================================
        # Langkah 3d: Filter kontur dan gambar bounding box
        # ============================================================

        # Membuat salinan frame untuk menggambar deteksi
        detected_frame = frame.copy()

        # Counter objek terdeteksi di frame ini
        obj_count = 0

        for contour in contours:
            # Menghitung area kontur
            area = cv2.contourArea(contour)

            # Memfilter kontur yang terlalu kecil (noise)
            if area < MIN_CONTOUR_AREA:
                continue

            # Objek terdeteksi: menaikkan counter
            obj_count += 1

            # Menghitung bounding box (kotak pembatas)
            # cv2.boundingRect() → (x, y, width, height)
            x, y, w, h = cv2.boundingRect(contour)

            # Menggambar bounding box (kotak hijau) pada frame
            cv2.rectangle(detected_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Menambahkan label dengan nomor objek dan area
            label = f"Obj {obj_count} ({area:.0f}px²)"
            cv2.putText(detected_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Menggambar kontur (garis merah) pada frame
            cv2.drawContours(detected_frame, [contour], -1, (0, 0, 255), 1)

            # Menggambar centroid (titik tengah) objek
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(detected_frame, (cx, cy), 5, (255, 0, 0), -1)

        # Menambahkan info jumlah objek pada frame
        info_text = f"Frame {frame_count}: {obj_count} objek terdeteksi"
        cv2.putText(detected_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Menyimpan jumlah objek per frame
        objects_per_frame.append(obj_count)

        # Menyimpan frame untuk visualisasi
        if frame_count in target_frames:
            vis_data.append((frame_count, frame.copy(), fg_mask_clean.copy(),
                            detected_frame.copy()))

    # Menutup video capture
    cap.release()

    print(f"[INFO] Selesai memproses {frame_count} frame.")
    print(f"[INFO] Rata-rata objek terdeteksi: {np.mean(objects_per_frame):.1f} per frame")
    print(f"[INFO] Maksimum objek terdeteksi : {np.max(objects_per_frame)} di satu frame")

    # ============================================================
    # 4. Visualisasi hasil deteksi
    # ============================================================

    # Membuat figure: pipeline deteksi
    num_vis = min(4, len(vis_data))
    fig1, axes1 = plt.subplots(3, num_vis, figsize=(4.5 * num_vis, 12))

    if num_vis == 1:
        axes1 = axes1.reshape(3, 1)

    for col in range(num_vis):
        fnum, orig, mask, detected = vis_data[col]

        # Baris 1: Frame asli
        axes1[0, col].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes1[0, col].set_title(f"Frame Asli #{fnum}", fontsize=10)
        axes1[0, col].axis("off")

        # Baris 2: Foreground mask
        axes1[1, col].imshow(mask, cmap='gray')
        axes1[1, col].set_title(f"Foreground Mask #{fnum}", fontsize=10)
        axes1[1, col].axis("off")

        # Baris 3: Frame dengan bounding box
        axes1[2, col].imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
        axes1[2, col].set_title(f"Deteksi #{fnum}", fontsize=10)
        axes1[2, col].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Percobaan 18: Deteksi Gerakan dengan Analisis Kontur\n"
                 "Hijau=BBox, Merah=Kontur, Biru●=Centroid",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Menyimpan figure pertama
    output_path_1 = os.path.join(OUTPUT_DIR, "18_deteksi_gerakan_contour.png")
    plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Deteksi gerakan disimpan di: {output_path_1}")

    # ============================================================
    # 5. Grafik statistik deteksi
    # ============================================================

    # Membuat figure: statistik
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Jumlah objek per frame
    axes2[0].plot(objects_per_frame, color='steelblue', linewidth=1.5)
    axes2[0].fill_between(range(len(objects_per_frame)), objects_per_frame,
                           alpha=0.3, color='steelblue')
    axes2[0].set_xlabel("Nomor Frame", fontsize=10)
    axes2[0].set_ylabel("Jumlah Objek Terdeteksi", fontsize=10)
    axes2[0].set_title("Jumlah Objek per Frame", fontsize=12)
    axes2[0].grid(True, alpha=0.3)

    # Subplot 2: Histogram jumlah objek
    max_obj = max(objects_per_frame) if objects_per_frame else 1
    axes2[1].hist(objects_per_frame, bins=range(0, max_obj + 2),
                  color='coral', alpha=0.7, edgecolor='darkred', align='left')
    axes2[1].set_xlabel("Jumlah Objek", fontsize=10)
    axes2[1].set_ylabel("Jumlah Frame", fontsize=10)
    axes2[1].set_title("Distribusi Jumlah Objek per Frame", fontsize=12)
    axes2[1].grid(True, alpha=0.3)

    # Menambahkan judul utama
    plt.suptitle("Percobaan 18: Statistik Deteksi Gerakan", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Menyimpan figure kedua
    output_path_2 = os.path.join(OUTPUT_DIR, "18_deteksi_statistik.png")
    plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[OUTPUT] Statistik deteksi disimpan di: {output_path_2}")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 18")
    print("=" * 60)
    print("Fungsi yang dipelajari:")
    print("  1. cv2.createBackgroundSubtractorMOG2() → Background subtraction")
    print("  2. cv2.morphologyEx(MORPH_OPEN)         → Hilangkan noise kecil")
    print("  3. cv2.morphologyEx(MORPH_CLOSE)        → Tutup celah objek")
    print("  4. cv2.findContours()                   → Temukan kontur objek")
    print("     - RETR_EXTERNAL: hanya kontur terluar")
    print("     - CHAIN_APPROX_SIMPLE: kompres titik")
    print("  5. cv2.contourArea()                    → Hitung luas kontur")
    print("  6. cv2.boundingRect()                   → Bounding box (x,y,w,h)")
    print("  7. cv2.moments()                        → Hitung centroid objek")
    print("  8. cv2.putText()                        → Label teks pada frame")
    print("Pipeline:")
    print("  BG Subtract → Threshold → Morfologi → Contour → Filter → BBox")
    print("=" * 60)



if __name__ == "__main__":
    main()
