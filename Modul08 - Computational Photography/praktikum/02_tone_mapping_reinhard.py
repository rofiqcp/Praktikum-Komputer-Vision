

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 2: TONE MAPPING REINHARD
    ==========================================================================
    Program ini mempelajari teknik tone mapping menggunakan operator Reinhard.
    Tone mapping mengompresi rentang dinamis tinggi (HDR) agar dapat
    ditampilkan pada monitor standar (LDR, 0-255).

    Operator Reinhard terinspirasi dari adaptasi visual manusia:
    - gamma      : Koreksi gamma (kecerahan keseluruhan)
    - intensity  : Kontrol intensitas global (-8 s/d 8)
    - light_adapt: Adaptasi cahaya (0=global, 1=lokal)
    - color_adapt: Adaptasi warna (0=tanpa, 1=penuh)

    Fungsi utama yang dipelajari:
    - cv2.createTonemapReinhard()  : Tone mapping Reinhard
    - cv2.createMergeDebevec()     : Merge exposure ke HDR
    - cv2.createCalibrateDebevec() : Kalibrasi CRF
    - cv2.createMergeMertens()     : Exposure fusion (alternatif)

    Hasil: Perbandingan parameter Reinhard dan exposure fusion
    ==========================================================================
    """

    # Mengimpor library OpenCV untuk pemrosesan HDR
    import cv2

    # Mengimpor NumPy untuk manipulasi array numerik
    import numpy as np

    # Mengimpor os untuk manajemen path file
    import os

    # Mengimpor matplotlib untuk visualisasi perbandingan
    import matplotlib.pyplot as plt

    # Mendapatkan direktori tempat script ini berada
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Mendefinisikan path folder gambar input
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

    # Mendefinisikan path folder output hasil percobaan
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    # Membuat folder output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PERCOBAAN 2: TONE MAPPING REINHARD")
    print("=" * 60)

    # ============================================================
    # 1. Membaca gambar multi-exposure dan membuat HDR
    # ============================================================

    # Daftar file gambar exposure
    exposure_files = [
        "exposure_1.png", "exposure_2.png", "exposure_3.png",
        "exposure_4.png", "exposure_5.png"
    ]

    # Waktu exposure setiap gambar dalam detik
    exposure_times = np.array([1/30, 1/15, 1/8, 1/4, 1/2], dtype=np.float32)

    # Membuat list untuk menampung gambar
    images = []

    # Membaca semua gambar exposure
    for fname in exposure_files:
        # Membentuk path lengkap file gambar
        img_path = os.path.join(IMAGE_DIR, fname)

        # Membaca gambar
        img = cv2.imread(img_path)

        # Validasi pembacaan gambar
        if img is None:
            print(f"[ERROR] Gagal membaca: {img_path}")
            print("[INFO] Jalankan download_image.py terlebih dahulu!")
            exit()

        # Menambahkan ke list
        images.append(img)

    print(f"[INFO] {len(images)} gambar exposure berhasil dimuat")

    # Menghitung CRF menggunakan metode Debevec
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, exposure_times)

    # Menggabungkan ke HDR radiance map
    merge = cv2.createMergeDebevec()
    hdr = merge.process(images, exposure_times, response)

    print(f"[INFO] HDR radiance map dibuat — rentang: [{hdr.min():.3f}, {hdr.max():.3f}]")

    # ============================================================
    # 2. Tone Mapping Reinhard dengan variasi parameter
    # cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)
    # ============================================================

    # Mendefinisikan konfigurasi parameter yang akan dibandingkan
    # Setiap tuple berisi: (gamma, intensity, light_adapt, color_adapt, label)
    configs = [
        (1.0, 0.0, 0.0, 0.0, "Default\ng=1.0, int=0, la=0, ca=0"),
        (1.5, 0.0, 0.8, 0.6, "Balanced\ng=1.5, int=0, la=0.8, ca=0.6"),
        (2.2, 0.0, 1.0, 1.0, "High Adapt\ng=2.2, int=0, la=1.0, ca=1.0"),
        (1.0, 4.0, 0.5, 0.5, "High Intensity\ng=1.0, int=4, la=0.5, ca=0.5"),
        (0.8, -2.0, 0.5, 0.0, "Low Gamma+Int\ng=0.8, int=-2, la=0.5, ca=0"),
        (2.5, 0.0, 0.0, 1.0, "High Gamma\ng=2.5, int=0, la=0, ca=1.0"),
    ]

    # Membuat list untuk menyimpan hasil tone mapping
    results = []

    # Menjalankan tone mapping Reinhard dengan setiap konfigurasi
    for gamma, intensity, light_adapt, color_adapt, label in configs:
        # Membuat objek TonemapReinhard dengan parameter tertentu
        tonemap = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt
        )

        # Melakukan tone mapping pada HDR image
        ldr = tonemap.process(hdr)

        # Memastikan nilai dalam rentang 0-1
        ldr = np.clip(ldr, 0, 1)

        # Mengkonversi ke uint8 untuk visualisasi
        ldr_8bit = (ldr * 255).astype(np.uint8)

        # Menyimpan hasil dan label
        results.append((ldr_8bit, label))

        # Menghitung statistik kecerahan rata-rata
        brightness = np.mean(ldr_8bit)
        print(f"[INFO] {label.split(chr(10))[0]:20s} — Kecerahan rata-rata: {brightness:.1f}")

    # ============================================================
    # 3. Exposure Fusion sebagai alternatif (tanpa HDR)
    # cv2.createMergeMertens() — langsung dari gambar LDR
    # ============================================================

    # Membuat objek MergeMertens dengan bobot default
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,    # Bobot kontras
        saturation_weight=1.0,  # Bobot saturasi
        exposure_weight=1.0     # Bobot exposure
    )

    # Melakukan exposure fusion (input: list gambar uint8)
    fusion = merge_mertens.process(images)

    # Clipping dan konversi ke uint8
    fusion = np.clip(fusion, 0, 1)
    fusion_8bit = (fusion * 255).astype(np.uint8)

    print(f"\n[INFO] Exposure Fusion Mertens selesai — "
          f"Kecerahan rata-rata: {np.mean(fusion_8bit):.1f}")

    # ============================================================
    # 4. Visualisasi perbandingan
    # ============================================================

    # Membuat figure dengan 2 baris x 4 kolom
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    # Menampilkan 6 variasi parameter Reinhard
    for idx, (ldr_img, label) in enumerate(results):
        # Menentukan posisi baris dan kolom
        row = idx // 4
        col = idx % 4

        # Konversi BGR ke RGB untuk matplotlib
        rgb = cv2.cvtColor(ldr_img, cv2.COLOR_BGR2RGB)

        # Menampilkan gambar pada subplot
        axes[row, col].imshow(rgb)

        # Menambahkan judul
        axes[row, col].set_title(f"Reinhard #{idx+1}\n{label}", fontsize=8)

        # Menyembunyikan sumbu
        axes[row, col].axis("off")

    # Menampilkan hasil exposure fusion
    fusion_rgb = cv2.cvtColor(fusion_8bit, cv2.COLOR_BGR2RGB)
    axes[1, 2].imshow(fusion_rgb)
    axes[1, 2].set_title("Exposure Fusion\n(Mertens — tanpa HDR)", fontsize=8)
    axes[1, 2].axis("off")

    # Menampilkan gambar exposure tengah sebagai referensi
    ref_rgb = cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB)
    axes[1, 3].imshow(ref_rgb)
    axes[1, 3].set_title("Referensi\nExposure Normal (1.0x)", fontsize=8)
    axes[1, 3].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Percobaan 2: Tone Mapping Reinhard — Variasi Parameter\n"
                 "gamma, intensity, light_adapt, color_adapt",
                 fontsize=14, fontweight="bold")

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan hasil visualisasi ke file
    output_path = os.path.join(OUTPUT_DIR, "02_tone_mapping_reinhard.png")
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
    print("  1. cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)")
    print("     - gamma       : Koreksi gamma (< 1 terang, > 1 gelap)")
    print("     - intensity   : Intensitas global (-8 s/d 8)")
    print("     - light_adapt : 0 = global, 1 = adaptasi lokal")
    print("     - color_adapt : 0 = tanpa adaptasi warna, 1 = penuh")
    print("  2. Operator Reinhard memodelkan adaptasi visual manusia")
    print("  3. Exposure Fusion (Mertens) sebagai alternatif tanpa HDR:")
    print("     - Tidak perlu exposure times / CRF")
    print("     - Langsung menggabungkan gambar LDR")
    print("  4. Parameter berbeda menghasilkan tone mapping berbeda")
    print("=" * 60)



if __name__ == "__main__":
    main()
