

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 5: BACKGROUND SUBTRACTION DENGAN KNN
    ==========================================================================
    Program ini mempelajari cara melakukan background subtraction menggunakan
    metode KNN (K-Nearest Neighbors). Metode KNN mengklasifikasikan setiap
    piksel sebagai foreground atau background berdasarkan jarak ke sampel
    background terdekat.

    Program juga membandingkan hasil KNN vs MOG2 secara side-by-side untuk
    melihat perbedaan performa kedua metode.

    Fungsi utama yang dipelajari:
    - cv2.createBackgroundSubtractorKNN()   : Membuat model background KNN
      - history: jumlah frame untuk membangun model
      - dist2Threshold: threshold jarak kuadrat untuk klasifikasi
      - detectShadows: aktifkan deteksi bayangan
    - subtractor.apply()                    : Menerapkan model ke frame baru
    - Perbandingan KNN vs MOG2

    Hasil: Perbandingan foreground mask KNN vs MOG2 dan efek deteksi bayangan
    ==========================================================================
    """

    # Mengimpor library OpenCV untuk pemrosesan video dan gambar
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
    print("PERCOBAAN 5: BACKGROUND SUBTRACTION DENGAN KNN")
    print("=" * 60)

    # ============================================================
    # 1. Membuka video dan menyiapkan model KNN dan MOG2
    # ============================================================

    # Membuka file video yang berisi objek bergerak
    video_path = os.path.join(IMAGE_DIR, "video_orang.avi")
    cap = cv2.VideoCapture(video_path)

    # Memeriksa apakah video berhasil dibuka
    if not cap.isOpened():
        print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
        exit()

    # Membaca informasi video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video berhasil dibuka: {frame_width}x{frame_height}")
    print(f"[INFO] Total frame: {total_frames}")

    # ============================================================
    # 2. Membuat Background Subtractor KNN
    # cv2.createBackgroundSubtractorKNN():
    #   - history: jumlah frame terakhir untuk model (default 500)
    #   - dist2Threshold: threshold jarak kuadrat (default 400.0)
    #     Semakin kecil = semakin sensitif
    #   - detectShadows: deteksi bayangan (True/False)
    # ============================================================

    # Membuat model background subtractor KNN
    knn = cv2.createBackgroundSubtractorKNN(
        history=500,            # Menggunakan 500 frame terakhir
        dist2Threshold=400.0,   # Threshold jarak kuadrat standar
        detectShadows=True      # Mengaktifkan deteksi bayangan
    )

    print("[INFO] Model KNN dibuat: history=500, dist2Threshold=400.0, detectShadows=True")

    # Membuat model MOG2 sebagai pembanding
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )

    print("[INFO] Model MOG2 dibuat sebagai pembanding")

    # ============================================================
    # 3. Memproses video: KNN vs MOG2 side-by-side
    # ============================================================

    # Menyimpan frame hasil untuk visualisasi
    original_frames = []
    knn_masks = []
    mog2_masks = []
    frame_indices = []
    frame_count = 0

    # Frame yang ingin divisualisasikan
    target_frames = [10, 30, 60, 100]

    # Membuat kernel untuk pembersihan morfologi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    print("[INFO] Memproses video dengan KNN dan MOG2...")

    while True:
        # Membaca frame berikutnya
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Menerapkan KNN background subtractor ke frame
        fg_knn = knn.apply(frame)

        # Menerapkan MOG2 background subtractor ke frame yang sama
        fg_mog2 = mog2.apply(frame)

        # Menyimpan frame target untuk visualisasi
        if frame_count in target_frames:
            original_frames.append(frame.copy())
            knn_masks.append(fg_knn.copy())
            mog2_masks.append(fg_mog2.copy())
            frame_indices.append(frame_count)

        # Berhenti setelah melewati target terakhir
        if frame_count > max(target_frames) + 10:
            break

    # Melepaskan video capture
    cap.release()

    print(f"[INFO] Selesai memproses {frame_count} frame.")

    # ============================================================
    # 4. Visualisasi perbandingan KNN vs MOG2
    # ============================================================

    # Membuat figure dengan 4 baris x 3 kolom (Original, KNN, MOG2)
    fig, axes = plt.subplots(len(frame_indices), 3, figsize=(15, 4 * len(frame_indices)))

    # Memastikan axes selalu 2D
    if len(frame_indices) == 1:
        axes = axes.reshape(1, -1)

    for idx in range(len(frame_indices)):
        # Kolom 1: Frame asli
        rgb_frame = cv2.cvtColor(original_frames[idx], cv2.COLOR_BGR2RGB)
        axes[idx, 0].imshow(rgb_frame)
        axes[idx, 0].set_title(f"Frame {frame_indices[idx]} - Original", fontsize=10)
        axes[idx, 0].axis("off")

        # Kolom 2: Foreground mask KNN
        # Putih (255) = foreground, abu-abu (127) = bayangan, hitam (0) = background
        axes[idx, 1].imshow(knn_masks[idx], cmap="gray")
        axes[idx, 1].set_title(f"Frame {frame_indices[idx]} - KNN", fontsize=10)
        axes[idx, 1].axis("off")

        # Kolom 3: Foreground mask MOG2
        axes[idx, 2].imshow(mog2_masks[idx], cmap="gray")
        axes[idx, 2].set_title(f"Frame {frame_indices[idx]} - MOG2", fontsize=10)
        axes[idx, 2].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Percobaan 5: Perbandingan Background Subtraction KNN vs MOG2\nPutih=Foreground, Abu-abu=Bayangan, Hitam=Background",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Menyimpan hasil visualisasi perbandingan
    output_path = os.path.join(OUTPUT_DIR, "05_background_subtraction_knn_vs_mog2.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Perbandingan disimpan di: {output_path}")

    # ============================================================
    # 5. Visualisasi efek deteksi bayangan (shadow detection)
    # ============================================================

    # Membuka kembali video untuk membandingkan shadow on/off
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Gagal membuka ulang video.")
        exit()

    # Membuat KNN tanpa deteksi bayangan sebagai pembanding
    knn_no_shadow = cv2.createBackgroundSubtractorKNN(
        history=500, dist2Threshold=400.0, detectShadows=False
    )

    # Membuat KNN dengan deteksi bayangan
    knn_with_shadow = cv2.createBackgroundSubtractorKNN(
        history=500, dist2Threshold=400.0, detectShadows=True
    )

    # Memproses frame hingga frame target
    shadow_frame = None
    mask_no_shadow = None
    mask_with_shadow = None
    mask_shadow_only = None
    frame_count2 = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count2 += 1

        # Menerapkan kedua subtractor ke frame yang sama
        m_no = knn_no_shadow.apply(frame)
        m_yes = knn_with_shadow.apply(frame)

        # Mengambil hasil di frame ke-60
        if frame_count2 == 60:
            shadow_frame = frame.copy()
            mask_no_shadow = m_no.copy()
            mask_with_shadow = m_yes.copy()

            # Mengekstrak hanya area bayangan (nilai 127 di mask)
            mask_shadow_only = np.zeros_like(m_yes)
            mask_shadow_only[m_yes == 127] = 255
            break

    # Melepaskan video capture
    cap.release()

    # Membuat figure untuk perbandingan bayangan
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))

    # Frame asli
    rgb_shadow = cv2.cvtColor(shadow_frame, cv2.COLOR_BGR2RGB)
    axes2[0].imshow(rgb_shadow)
    axes2[0].set_title("Frame Asli (ke-60)", fontsize=10)
    axes2[0].axis("off")

    # Mask tanpa deteksi bayangan
    axes2[1].imshow(mask_no_shadow, cmap="gray")
    axes2[1].set_title("KNN tanpa Shadow", fontsize=10)
    axes2[1].axis("off")

    # Mask dengan deteksi bayangan
    axes2[2].imshow(mask_with_shadow, cmap="gray")
    axes2[2].set_title("KNN dengan Shadow\n(abu-abu=bayangan)", fontsize=10)
    axes2[2].axis("off")

    # Hanya area bayangan
    axes2[3].imshow(mask_shadow_only, cmap="gray")
    axes2[3].set_title("Area Bayangan Saja", fontsize=10)
    axes2[3].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Efek Shadow Detection pada KNN Background Subtractor",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Menyimpan hasil visualisasi bayangan
    output_path2 = os.path.join(OUTPUT_DIR, "05_knn_shadow_detection.png")
    plt.savefig(output_path2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[OUTPUT] Shadow detection disimpan di: {output_path2}")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 5")
    print("=" * 60)
    print("Fungsi yang dipelajari:")
    print("  1. cv2.createBackgroundSubtractorKNN()   → Membuat model BG KNN")
    print("     - history: jumlah frame untuk model (default 500)")
    print("     - dist2Threshold: threshold jarak kuadrat (default 400)")
    print("     - detectShadows: deteksi bayangan (True/False)")
    print("  2. Perbandingan KNN vs MOG2:")
    print("     - KNN: berbasis K-nearest neighbors, lebih tahan noise")
    print("     - MOG2: berbasis Gaussian mixture, lebih cepat")
    print("     - Keduanya adaptif dan mendukung shadow detection")
    print("  3. Shadow Detection:")
    print("     - Bayangan ditandai abu-abu (127) di foreground mask")
    print("     - Berguna untuk membedakan objek asli dari bayangannya")
    print("     - Tanpa shadow detection, bayangan dianggap foreground")
    print("  4. Pilihan metode tergantung aplikasi dan skenario")
    print("=" * 60)



if __name__ == "__main__":
    main()
