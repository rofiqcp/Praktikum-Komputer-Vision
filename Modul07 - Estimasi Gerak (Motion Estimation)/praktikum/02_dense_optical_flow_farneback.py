

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 2: DENSE OPTICAL FLOW - FARNEBACK
    ==========================================================================
    Program ini mempelajari cara menghitung dense optical flow menggunakan
    metode Farneback. Berbeda dengan Lucas-Kanade yang sparse (hanya di titik
    fitur), Farneback menghitung flow untuk SELURUH PIKSEL dalam gambar.

    Metode Farneback:
    - Mengaproksimasi setiap piksel dengan polinomial kuadratik
    - Menghitung perpindahan dari perubahan koefisien polinomial
    - Menghasilkan vektor (u, v) untuk setiap piksel

    Fungsi utama yang dipelajari:
    - cv2.calcOpticalFlowFarneback()  : Menghitung dense optical flow
    - cv2.cartToPolar()               : Konversi kartesian ke polar (mag, angle)
    - cv2.normalize()                 : Normalisasi nilai array

    Hasil: Visualisasi dense flow field menggunakan HSV color coding
    ==========================================================================
    """

    # Mengimpor library yang dibutuhkan
    import cv2
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # Mendapatkan direktori script saat ini
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PERCOBAAN 2: DENSE OPTICAL FLOW - FARNEBACK")
    print("=" * 60)

    # ============================================================
    # 1. Membaca sepasang frame dari video
    # ============================================================

    # Membuka video
    video_path = os.path.join(IMAGE_DIR, "video_bola.avi")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
        exit()

    # Membaca frame pertama dan mengkonversi ke grayscale
    ret, frame1 = cap.read()
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Membaca frame kedua (beberapa frame kemudian untuk gerakan lebih jelas)
    for _ in range(5):
        ret, frame2 = cap.read()
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    print(f"[INFO] Ukuran frame: {prev_gray.shape}")

    # ============================================================
    # 2. Menghitung Dense Optical Flow dengan Farneback
    # cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels,
    #                                winsize, iterations, poly_n, poly_sigma, flags)
    # Parameter:
    #   - pyr_scale: skala pyramid (0.5 = 2x di setiap level)
    #   - levels: jumlah level pyramid
    #   - winsize: ukuran window averaging
    #   - iterations: jumlah iterasi di setiap level
    #   - poly_n: ukuran neighborhood polinomial (5 atau 7)
    #   - poly_sigma: standar deviasi Gaussian untuk poly_n
    # ============================================================

    # Menghitung dense optical flow
    # flow memiliki shape (H, W, 2) - berisi (u, v) untuk setiap piksel
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,          # Frame sebelumnya (grayscale)
        next_gray,          # Frame saat ini (grayscale)
        None,               # Output flow (None = buat baru)
        pyr_scale=0.5,      # Reduksi 50% di setiap level pyramid
        levels=3,           # 3 level pyramid
        winsize=15,         # Window averaging 15x15
        iterations=3,       # 3 iterasi per level
        poly_n=5,           # Neighborhood polinomial 5x5
        poly_sigma=1.2,     # Sigma Gaussian untuk smoothing polinomial
        flags=0             # Tidak ada flag tambahan
    )

    print(f"[INFO] Shape flow field: {flow.shape}")
    print(f"[INFO] Flow berisi (u, v) perpindahan untuk setiap piksel")

    # ============================================================
    # 3. Visualisasi menggunakan HSV color coding
    # Hue = arah gerakan, Value = kecepatan gerakan
    # ============================================================

    # Mengekstrak komponen horizontal (u) dan vertikal (v)
    flow_u = flow[..., 0]  # Perpindahan horizontal
    flow_v = flow[..., 1]  # Perpindahan vertikal

    # Mengkonversi dari koordinat kartesian (u, v) ke polar (magnitude, angle)
    # cv2.cartToPolar() menghitung:
    #   - magnitude = sqrt(u² + v²) → kecepatan gerakan
    #   - angle = atan2(v, u) → arah gerakan (dalam radian)
    magnitude, angle = cv2.cartToPolar(flow_u, flow_v)

    print(f"[INFO] Magnitude range: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
    print(f"[INFO] Angle range: [{angle.min():.2f}, {angle.max():.2f}] radian")

    # Membuat gambar HSV untuk visualisasi
    # H (Hue) = arah gerakan (0-180)
    # S (Saturation) = 255 (penuh)
    # V (Value) = kecepatan gerakan (0-255)
    hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)

    # Mengatur Hue berdasarkan arah gerakan (konversi radian ke derajat / 2)
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Mengatur Saturation ke maksimum (warna penuh)
    hsv[..., 1] = 255

    # Mengatur Value berdasarkan magnitude (kecepatan)
    # cv2.normalize() mengubah range ke 0-255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Mengkonversi HSV ke BGR untuk ditampilkan
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    print("[INFO] Visualisasi HSV:")
    print("  - Warna MERAH → gerakan ke KANAN")
    print("  - Warna HIJAU → gerakan ke BAWAH")
    print("  - Warna BIRU  → gerakan ke KIRI")
    print("  - TERANG = gerakan cepat, GELAP = gerakan lambat/diam")

    # ============================================================
    # 4. Visualisasi menggunakan vektor panah (quiver plot)
    # ============================================================

    # Membuat grid untuk menampilkan panah (tidak semua piksel, terlalu padat)
    step = 20  # Tampilkan panah setiap 20 piksel

    # Membuat koordinat grid
    y_coords, x_coords = np.mgrid[step//2:prev_gray.shape[0]:step,
                                    step//2:prev_gray.shape[1]:step]

    # Mengambil flow di titik-titik grid
    fx = flow_u[step//2::step, step//2::step]
    fy = flow_v[step//2::step, step//2::step]

    # ============================================================
    # 5. Menghitung flow untuk beberapa frame (akumulasi)
    # ============================================================

    # Menyimpan hasil flow untuk beberapa frame
    flow_frames = [(frame1.copy(), flow_rgb.copy(), "Frame 1→6")]

    # Membaca beberapa frame lagi dan hitung flow
    for frame_idx in range(3):
        prev_g = next_gray.copy()
        for _ in range(10):
            ret, frame_next = cap.read()
            if not ret:
                break
        if not ret:
            break
        next_g = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)

        # Hitung dense flow
        fl = cv2.calcOpticalFlowFarneback(prev_g, next_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mg, ag = cv2.cartToPolar(fl[..., 0], fl[..., 1])
        h = np.zeros_like(hsv)
        h[..., 0] = ag * 180 / np.pi / 2
        h[..., 1] = 255
        h[..., 2] = cv2.normalize(mg, None, 0, 255, cv2.NORM_MINMAX)
        flow_vis = cv2.cvtColor(h, cv2.COLOR_HSV2BGR)

        flow_frames.append((frame_next.copy(), flow_vis.copy(), f"Frame {(frame_idx+1)*10+6}→{(frame_idx+1)*10+16}"))
        next_gray = next_g

    cap.release()

    # ============================================================
    # 6. Menyimpan visualisasi
    # ============================================================

    n_results = min(len(flow_frames), 4)
    fig, axes = plt.subplots(n_results, 2, figsize=(14, 5 * n_results))
    if n_results == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_results):
        frame_orig, flow_vis, label = flow_frames[idx]

        # Kolom 1: Frame asli
        axes[idx, 0].imshow(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f"Frame Asli - {label}", fontsize=11)
        axes[idx, 0].axis("off")

        # Kolom 2: Dense Flow (HSV)
        axes[idx, 1].imshow(cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f"Dense Optical Flow - {label}", fontsize=11)
        axes[idx, 1].axis("off")

    plt.suptitle("Percobaan 2: Dense Optical Flow Farneback\nWarna=arah, Kecerahan=kecepatan",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "02_dense_optical_flow_farneback.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 2")
    print("=" * 60)
    print("Fungsi yang dipelajari:")
    print("  1. cv2.calcOpticalFlowFarneback() → Dense optical flow")
    print("     - Menghitung flow untuk SETIAP piksel")
    print("     - Output: array (H, W, 2) berisi (u, v)")
    print("     - pyr_scale, levels: kontrol pyramid")
    print("     - winsize: ukuran window rata-rata")
    print("  2. cv2.cartToPolar()  → Konversi (u,v) ke (magnitude, angle)")
    print("  3. cv2.normalize()    → Normalisasi nilai ke range tertentu")
    print("  4. Visualisasi HSV: Hue=arah, Value=kecepatan")
    print("  5. Dense flow ≠ Sparse flow:")
    print("     - Dense: semua piksel, lambat tapi lengkap")
    print("     - Sparse: hanya fitur, cepat tapi terbatas")
    print("=" * 60)



if __name__ == "__main__":
    main()
