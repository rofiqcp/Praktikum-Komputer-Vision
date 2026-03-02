

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 15: ANALISIS MAGNITUDE DAN ARAH OPTICAL FLOW
    ==========================================================================
    Program ini mempelajari analisis detail komponen optical flow, yaitu
    magnitude (kecepatan) dan direction (arah) dari vektor flow.

    Komponen Optical Flow:
    - Magnitude = sqrt(dx² + dy²) → kecepatan gerakan
    - Direction = atan2(dy, dx)   → arah gerakan (dalam radian)
    - Threshold magnitude untuk memisahkan area bergerak vs statis
    - Histogram arah untuk analisis pola gerakan

    Fungsi utama yang dipelajari:
    - cv2.calcOpticalFlowFarneback()  : Dense optical flow
    - cv2.cartToPolar()               : Konversi (dx,dy) ke (magnitude, angle)
    - cv2.threshold()                 : Threshold magnitude
    - np.histogram()                  : Histogram arah flow
    - matplotlib polar plot            : Rose diagram arah

    Hasil: Analisis komprehensif magnitude dan arah optical flow
    ==========================================================================
    """

    # Mengimpor library OpenCV untuk pemrosesan gambar dan video
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
    print("PERCOBAAN 15: ANALISIS MAGNITUDE DAN ARAH OPTICAL FLOW")
    print("=" * 60)

    # ============================================================
    # 1. Membuka video dan membaca dua frame berturut-turut
    # ============================================================

    # Membuka file video yang berisi multi-objek bergerak
    video_path = os.path.join(IMAGE_DIR, "video_multi_objek.avi")
    cap = cv2.VideoCapture(video_path)

    # Memeriksa apakah video berhasil dibuka
    if not cap.isOpened():
        print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
        exit()

    # Membaca informasi video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video berhasil dibuka: {width}x{height}")
    print(f"[INFO] Total frame: {total_frames}")

    # Membaca frame pertama
    ret, frame1 = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame pertama.")
        exit()

    # Skip beberapa frame untuk mendapatkan pergerakan yang jelas
    for _ in range(10):
        ret, frame2 = cap.read()

    # Memeriksa keberhasilan membaca
    if not ret:
        print("[ERROR] Gagal membaca frame kedua.")
        exit()

    # Mengkonversi kedua frame ke grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    print("[INFO] Frame 1 dan Frame 11 berhasil diekstrak.")

    # Melepaskan video capture
    cap.release()

    # ============================================================
    # 2. Menghitung dense optical flow
    # ============================================================

    print("[INFO] Menghitung dense optical flow Farneback...")

    # Menghitung optical flow menggunakan metode Farneback
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=0.5,    # Skala pyramid
        levels=3,          # 3 level pyramid
        winsize=15,        # Ukuran window
        iterations=3,      # Iterasi per level
        poly_n=5,          # Neighbourhood polinomial
        poly_sigma=1.2,    # Sigma Gaussian
        flags=0
    )

    # Memisahkan komponen horizontal (dx) dan vertikal (dy)
    flow_x = flow[:, :, 0]  # Komponen horizontal (positif = ke kanan)
    flow_y = flow[:, :, 1]  # Komponen vertikal (positif = ke bawah)

    # ============================================================
    # 3. Menghitung magnitude dan arah menggunakan cartToPolar
    # ============================================================

    # cv2.cartToPolar(x, y) mengonversi dari kartesian ke polar
    # Output: magnitude = sqrt(x² + y²), angle = atan2(y, x)
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)

    print(f"[INFO] Flow shape: {flow.shape}")
    print(f"[INFO] Magnitude: min={magnitude.min():.4f}, max={magnitude.max():.4f}, mean={magnitude.mean():.4f}")
    print(f"[INFO] Arah (derajat): min={angle.min():.1f}, max={angle.max():.1f}")

    # ============================================================
    # 4. Threshold magnitude untuk memisahkan area bergerak vs statis
    # ============================================================

    # Mendefinisikan threshold magnitude
    MAG_THRESHOLD = 1.0  # Piksel dengan magnitude > 1.0 dianggap bergerak

    # Menormalisasi magnitude ke range 0-255 untuk threshold
    mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Menerapkan threshold pada magnitude
    _, motion_mask = cv2.threshold(mag_normalized, int(255 * MAG_THRESHOLD / magnitude.max()),
                                    255, cv2.THRESH_BINARY)

    # Menghitung persentase area bergerak
    total_pixels = width * height
    moving_pixels = np.count_nonzero(motion_mask)
    moving_percentage = (moving_pixels / total_pixels) * 100

    print(f"\n[INFO] Threshold magnitude: {MAG_THRESHOLD}")
    print(f"[INFO] Piksel bergerak: {moving_pixels} ({moving_percentage:.1f}%)")
    print(f"[INFO] Piksel statis  : {total_pixels - moving_pixels} ({100-moving_percentage:.1f}%)")

    # ============================================================
    # 5. Visualisasi magnitude dan komponen flow
    # ============================================================

    # Membuat figure utama: komponen flow
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 11))

    # Subplot 1: Frame asli
    axes1[0, 0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    axes1[0, 0].set_title("Frame 1 (Asli)", fontsize=11)
    axes1[0, 0].axis("off")

    # Subplot 2: Komponen flow horizontal (dx)
    im_fx = axes1[0, 1].imshow(flow_x, cmap='coolwarm', vmin=-np.abs(flow_x).max(),
                                vmax=np.abs(flow_x).max())
    axes1[0, 1].set_title("Flow Horizontal (dx)\nMerah=kanan, Biru=kiri", fontsize=10)
    axes1[0, 1].axis("off")
    plt.colorbar(im_fx, ax=axes1[0, 1], fraction=0.046)

    # Subplot 3: Komponen flow vertikal (dy)
    im_fy = axes1[0, 2].imshow(flow_y, cmap='coolwarm', vmin=-np.abs(flow_y).max(),
                                vmax=np.abs(flow_y).max())
    axes1[0, 2].set_title("Flow Vertikal (dy)\nMerah=bawah, Biru=atas", fontsize=10)
    axes1[0, 2].axis("off")
    plt.colorbar(im_fy, ax=axes1[0, 2], fraction=0.046)

    # Subplot 4: Magnitude flow
    im_mag = axes1[1, 0].imshow(magnitude, cmap='hot')
    axes1[1, 0].set_title("Magnitude (kecepatan gerakan)", fontsize=11)
    axes1[1, 0].axis("off")
    plt.colorbar(im_mag, ax=axes1[1, 0], fraction=0.046)

    # Subplot 5: Arah flow
    im_ang = axes1[1, 1].imshow(angle, cmap='hsv', vmin=0, vmax=360)
    axes1[1, 1].set_title("Arah (direction) dalam derajat", fontsize=11)
    axes1[1, 1].axis("off")
    plt.colorbar(im_ang, ax=axes1[1, 1], fraction=0.046)

    # Subplot 6: Motion mask (threshold)
    axes1[1, 2].imshow(motion_mask, cmap='gray')
    axes1[1, 2].set_title(f"Motion Mask (threshold={MAG_THRESHOLD})\n"
                           f"Bergerak: {moving_percentage:.1f}%", fontsize=10)
    axes1[1, 2].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Percobaan 15: Analisis Komponen Optical Flow",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Menyimpan figure pertama
    output_path_1 = os.path.join(OUTPUT_DIR, "15_flow_components.png")
    plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Komponen flow disimpan di: {output_path_1}")

    # ============================================================
    # 6. Rose diagram (histogram arah)
    # ============================================================

    # Mengambil arah hanya dari piksel yang bergerak (di atas threshold)
    moving_angles = angle[motion_mask > 0].flatten()
    moving_magnitudes = magnitude[motion_mask > 0].flatten()

    # Menghitung histogram arah
    num_bins = 36  # 36 bin = setiap 10 derajat
    bin_edges = np.linspace(0, 360, num_bins + 1)
    angle_hist, _ = np.histogram(moving_angles, bins=bin_edges)

    # Menghitung histogram arah yang di-weight oleh magnitude
    angle_hist_weighted, _ = np.histogram(moving_angles, bins=bin_edges, weights=moving_magnitudes)

    # Membuat figure: rose diagram dan histogram
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6),
                                subplot_kw={'projection': None})

    # Membuat subplot polar untuk rose diagram (tidak di-weight)
    ax_polar1 = fig2.add_subplot(131, polar=True)
    theta = np.deg2rad(bin_edges[:-1] + 5)  # Posisi tengah setiap bin
    bars1 = ax_polar1.bar(theta, angle_hist, width=np.deg2rad(10),
                           color='steelblue', alpha=0.7, edgecolor='navy')
    ax_polar1.set_title("Rose Diagram Arah\n(jumlah piksel)", fontsize=11, pad=20)
    ax_polar1.set_theta_zero_location('E')  # 0° di kanan (East)

    # Rose diagram (di-weight oleh magnitude)
    ax_polar2 = fig2.add_subplot(132, polar=True)
    bars2 = ax_polar2.bar(theta, angle_hist_weighted, width=np.deg2rad(10),
                           color='coral', alpha=0.7, edgecolor='darkred')
    ax_polar2.set_title("Rose Diagram Arah\n(weighted by magnitude)", fontsize=11, pad=20)
    ax_polar2.set_theta_zero_location('E')

    # Histogram magnitude
    ax_hist = fig2.add_subplot(133)
    ax_hist.hist(magnitude.flatten(), bins=50, color='green', alpha=0.7,
                 edgecolor='darkgreen', range=(0, magnitude.max()))
    ax_hist.axvline(x=MAG_THRESHOLD, color='red', linestyle='--', linewidth=2,
                    label=f'Threshold={MAG_THRESHOLD}')
    ax_hist.set_xlabel("Magnitude (piksel/frame)", fontsize=10)
    ax_hist.set_ylabel("Jumlah Piksel", fontsize=10)
    ax_hist.set_title("Histogram Magnitude", fontsize=11)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    # Menambahkan judul utama
    plt.suptitle("Percobaan 15: Distribusi Arah dan Magnitude Optical Flow",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Menyimpan figure kedua
    output_path_2 = os.path.join(OUTPUT_DIR, "15_flow_distribution.png")
    plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[OUTPUT] Distribusi flow disimpan di: {output_path_2}")

    # ============================================================
    # 7. Statistik detail flow field
    # ============================================================

    print(f"\n[INFO] ============ STATISTIK FLOW FIELD ============")
    print(f"  Komponen Horizontal (dx):")
    print(f"    - Min  : {flow_x.min():.4f} piksel")
    print(f"    - Max  : {flow_x.max():.4f} piksel")
    print(f"    - Mean : {flow_x.mean():.4f} piksel")
    print(f"    - Std  : {flow_x.std():.4f} piksel")
    print(f"  Komponen Vertikal (dy):")
    print(f"    - Min  : {flow_y.min():.4f} piksel")
    print(f"    - Max  : {flow_y.max():.4f} piksel")
    print(f"    - Mean : {flow_y.mean():.4f} piksel")
    print(f"    - Std  : {flow_y.std():.4f} piksel")
    print(f"  Magnitude:")
    print(f"    - Min  : {magnitude.min():.4f}")
    print(f"    - Max  : {magnitude.max():.4f}")
    print(f"    - Mean : {magnitude.mean():.4f}")
    print(f"    - Median: {np.median(magnitude):.4f}")
    print(f"  Arah dominan: {bin_edges[np.argmax(angle_hist)]:.0f}° - "
          f"{bin_edges[np.argmax(angle_hist)+1]:.0f}°")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 15")
    print("=" * 60)
    print("Fungsi yang dipelajari:")
    print("  1. cv2.cartToPolar(dx, dy)  → Konversi (dx,dy) ke (magnitude, angle)")
    print("     - Magnitude = sqrt(dx² + dy²)")
    print("     - Angle = atan2(dy, dx)")
    print("  2. cv2.threshold()          → Threshold magnitude (bergerak vs statis)")
    print("  3. np.histogram()           → Histogram distribusi arah")
    print("  4. matplotlib polar plot    → Rose diagram arah gerakan")
    print("Konsep:")
    print("  - Magnitude menunjukkan KECEPATAN gerakan (piksel/frame)")
    print("  - Direction menunjukkan ARAH gerakan (0-360°)")
    print("  - Threshold magnitude memisahkan area bergerak dari background")
    print("  - Rose diagram memvisualisasikan distribusi arah secara intuitif")
    print("  - Statistik flow berguna untuk analisis pola gerakan")
    print("=" * 60)



if __name__ == "__main__":
    main()
