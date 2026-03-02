

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 1: OPTICAL FLOW LUCAS-KANADE (SPARSE)
    ==========================================================================
    Program ini mempelajari cara menghitung sparse optical flow menggunakan
    metode Lucas-Kanade. Optical flow menunjukkan pergerakan piksel antara
    dua frame berturut-turut dalam video.

    Metode Lucas-Kanade:
    - Menghitung flow HANYA di titik-titik fitur (corner) yang kuat
    - Menggunakan image pyramid untuk menangani gerakan besar
    - Cocok untuk tracking titik-titik tertentu

    Fungsi utama yang dipelajari:
    - cv2.goodFeaturesToTrack()       : Mendeteksi corner/fitur yang bagus
    - cv2.calcOpticalFlowPyrLK()      : Menghitung sparse optical flow LK
    - cv2.line()                      : Menggambar garis trajectory
    - cv2.circle()                    : Menggambar titik posisi fitur

    Hasil: Visualisasi pergerakan titik-titik fitur antar frame
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
    print("PERCOBAAN 1: OPTICAL FLOW LUCAS-KANADE (SPARSE)")
    print("=" * 60)

    # ============================================================
    # 1. Membuka video dan membaca frame pertama
    # ============================================================

    # Membuka file video yang berisi objek bergerak
    video_path = os.path.join(IMAGE_DIR, "video_bola.avi")
    cap = cv2.VideoCapture(video_path)

    # Memeriksa apakah video berhasil dibuka
    if not cap.isOpened():
        print("[ERROR] Video tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
        exit()

    # Membaca frame pertama dari video
    # ret = True/False (berhasil/gagal), frame = data gambar
    ret, old_frame = cap.read()

    # Mengkonversi frame pertama ke grayscale (optical flow butuh 1 channel)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    print(f"[INFO] Video berhasil dibuka: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"[INFO] Total frame: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    # ============================================================
    # 2. Mendeteksi fitur (corner) di frame pertama
    # cv2.goodFeaturesToTrack() - Shi-Tomasi corner detector
    # Parameter:
    #   - maxCorners: jumlah maksimum corner yang dideteksi
    #   - qualityLevel: threshold kualitas minimum (0-1)
    #   - minDistance: jarak minimum antar corner (piksel)
    # ============================================================

    # Mendefinisikan parameter untuk deteksi fitur
    feature_params = dict(
        maxCorners=100,      # Maksimum 100 titik fitur
        qualityLevel=0.3,    # Kualitas minimum 30% dari corner terkuat
        minDistance=7,        # Jarak minimum 7 piksel antar fitur
        blockSize=7          # Ukuran blok untuk komputasi corner
    )

    # Mendeteksi fitur/corner pada frame pertama
    # Output: array berisi koordinat (x, y) setiap corner
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    print(f"[INFO] Jumlah fitur terdeteksi: {len(p0)}")

    # ============================================================
    # 3. Mendefinisikan parameter Lucas-Kanade
    # ============================================================

    # Parameter untuk calcOpticalFlowPyrLK:
    # - winSize: ukuran window pencarian (semakin besar = gerakan lebih besar)
    # - maxLevel: level pyramid (0 = tanpa pyramid, 2 = 3 level)
    # - criteria: kriteria berhenti iterasi
    lk_params = dict(
        winSize=(15, 15),    # Window pencarian 15x15 piksel
        maxLevel=2,          # Menggunakan 3 level pyramid (0, 1, 2)
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        # Berhenti setelah 10 iterasi ATAU error < 0.03
    )

    # ============================================================
    # 4. Menghitung Optical Flow untuk beberapa frame
    # ============================================================

    # Membuat mask kosong untuk menggambar trajectory (jejak gerakan)
    mask = np.zeros_like(old_frame)

    # Mendefinisikan warna acak untuk setiap titik fitur
    colors = np.random.randint(0, 255, (100, 3))

    # Menyimpan beberapa frame hasil untuk visualisasi
    frame_results = []
    frame_indices = []
    frame_count = 0

    print("[INFO] Memproses optical flow frame per frame...")

    while True:
        # Membaca frame berikutnya
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Mengkonversi frame saat ini ke grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ============================================================
        # Menghitung optical flow menggunakan metode pyramidal Lucas-Kanade
        # cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, ...)
        # Output:
        #   - p1: posisi baru titik fitur di frame saat ini
        #   - status: 1 jika fitur ditemukan, 0 jika hilang
        #   - err: error untuk setiap titik
        # ============================================================
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        # Memfilter hanya titik yang berhasil di-track (status == 1)
        if p1 is not None:
            # Mengambil titik baru dan lama yang berhasil dilacak
            good_new = p1[status == 1]
            good_old = p0[status == 1]

        # Menggambar garis dan lingkaran untuk setiap titik yang berhasil di-track
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Mengekstrak koordinat (x, y)
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)

            # Menggambar garis dari posisi lama ke posisi baru (trajectory)
            mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)

            # Menggambar lingkaran di posisi baru (posisi saat ini)
            frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)

        # Menggabungkan frame asli dengan mask trajectory
        img = cv2.add(frame, mask)

        # Menyimpan beberapa frame untuk visualisasi
        if frame_count in [1, 30, 60, 90]:
            frame_results.append(img.copy())
            frame_indices.append(frame_count)

        # Memperbarui frame dan titik lama untuk iterasi berikutnya
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Melepaskan video capture
    cap.release()

    print(f"[INFO] Selesai memproses {frame_count} frame.")
    print(f"[INFO] Jumlah fitur yang masih ter-track: {len(good_new)}")

    # ============================================================
    # 5. Visualisasi hasil optical flow
    # ============================================================

    # Membuat figure dengan beberapa subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (frame_img, frame_num) in enumerate(zip(frame_results, frame_indices)):
        # Konversi BGR ke RGB untuk matplotlib
        rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(rgb)
        axes[idx].set_title(f"Frame {frame_num}", fontsize=12)
        axes[idx].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Percobaan 1: Optical Flow Lucas-Kanade (Sparse)\nJejak pergerakan titik-titik fitur",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Menyimpan hasil visualisasi
    output_path = os.path.join(OUTPUT_DIR, "01_optical_flow_lucas_kanade.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 1")
    print("=" * 60)
    print("Fungsi yang dipelajari:")
    print("  1. cv2.goodFeaturesToTrack()    → Deteksi corner (Shi-Tomasi)")
    print("     - maxCorners: jumlah maksimum fitur")
    print("     - qualityLevel: threshold kualitas")
    print("     - minDistance: jarak minimum antar fitur")
    print("  2. cv2.calcOpticalFlowPyrLK()   → Hitung sparse optical flow")
    print("     - Pyramidal Lucas-Kanade method")
    print("     - winSize: ukuran window pencarian")
    print("     - maxLevel: jumlah level pyramid")
    print("     - Return: posisi baru, status, error")
    print("  3. Optical flow menunjukkan PERPINDAHAN piksel antar frame")
    print("  4. Sparse = hanya di titik fitur, bukan seluruh gambar")
    print("=" * 60)



if __name__ == "__main__":
    main()
