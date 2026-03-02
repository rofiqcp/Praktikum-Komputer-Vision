

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 3: TONE MAPPING DRAGO & PERBANDINGAN OPERATOR
    ==========================================================================
    Program ini mempelajari tone mapping menggunakan operator Drago dan Mantiuk,
    lalu membandingkannya dengan Reinhard. Setiap operator memiliki karakteristik
    berbeda dalam memetakan HDR ke LDR.

    Operator yang digunakan:
    - Drago   : Logaritmik adaptif, menghasilkan warna natural
    - Mantiuk : Berbasis kontras, mempertahankan detail
    - Reinhard: Berbasis adaptasi visual manusia

    Fungsi utama yang dipelajari:
    - cv2.createTonemapDrago(gamma, saturation, bias)
    - cv2.createTonemapMantiuk(gamma, scale, saturation)
    - cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)
    - cv2.createTonemap(gamma) — tone mapping global sederhana

    Hasil: Perbandingan visual tiga operator tone mapping
    ==========================================================================
    """

    # Mengimpor library OpenCV untuk pemrosesan HDR
    import cv2

    # Mengimpor NumPy untuk manipulasi array
    import numpy as np

    # Mengimpor os untuk manajemen path
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
    print("PERCOBAAN 3: TONE MAPPING DRAGO & PERBANDINGAN OPERATOR")
    print("=" * 60)

    # ============================================================
    # 1. Membaca gambar multi-exposure dan membuat HDR
    # ============================================================

    # Daftar file exposure
    exposure_files = [
        "exposure_1.png", "exposure_2.png", "exposure_3.png",
        "exposure_4.png", "exposure_5.png"
    ]

    # Waktu exposure dalam detik
    exposure_times = np.array([1/30, 1/15, 1/8, 1/4, 1/2], dtype=np.float32)

    # Membaca semua gambar exposure
    images = []
    for fname in exposure_files:
        # Membangun path lengkap
        img_path = os.path.join(IMAGE_DIR, fname)

        # Membaca gambar
        img = cv2.imread(img_path)

        # Validasi
        if img is None:
            print(f"[ERROR] Gagal membaca: {img_path}")
            print("[INFO] Jalankan download_image.py terlebih dahulu!")
            exit()

        # Menambahkan ke list
        images.append(img)

    print(f"[INFO] {len(images)} gambar exposure dimuat")

    # Membuat HDR radiance map menggunakan pipeline Debevec
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, exposure_times)

    merge = cv2.createMergeDebevec()
    hdr = merge.process(images, exposure_times, response)

    print(f"[INFO] HDR radiance map — rentang: [{hdr.min():.3f}, {hdr.max():.3f}]")

    # ============================================================
    # 2. Tone Mapping Drago
    # cv2.createTonemapDrago(gamma, saturation, bias)
    # - gamma     : Koreksi gamma (default 1.0)
    # - saturation: Tingkat saturasi warna (default 1.0)
    # - bias      : Bias fungsi log adaptif (0.7-0.9 ideal)
    # ============================================================

    # Konfigurasi variasi parameter Drago
    # Format: (gamma, saturation, bias, label)
    drago_configs = [
        (1.0, 1.0, 0.85, "Default\ng=1.0, sat=1.0, bias=0.85"),
        (2.2, 1.0, 0.85, "Gamma 2.2\ng=2.2, sat=1.0, bias=0.85"),
        (1.0, 0.5, 0.85, "Low Saturation\ng=1.0, sat=0.5, bias=0.85"),
        (1.0, 1.0, 0.5,  "Low Bias\ng=1.0, sat=1.0, bias=0.5"),
    ]

    # Menyimpan hasil Drago
    drago_results = []

    print("\n--- Tone Mapping DRAGO ---")
    for gamma, saturation, bias, label in drago_configs:
        # Membuat objek TonemapDrago dengan parameter tertentu
        tonemap_drago = cv2.createTonemapDrago(
            gamma=gamma,
            saturation=saturation,
            bias=bias
        )

        # Menjalankan tone mapping
        ldr = tonemap_drago.process(hdr)

        # Clipping dan konversi ke 8-bit
        ldr = np.clip(ldr, 0, 1)
        ldr_8bit = (ldr * 255).astype(np.uint8)

        # Menyimpan hasil
        drago_results.append((ldr_8bit, label))

        # Menghitung dan menampilkan kecerahan rata-rata
        brightness = np.mean(ldr_8bit)
        print(f"  {label.split(chr(10))[0]:20s} — Kecerahan: {brightness:.1f}")

    # ============================================================
    # 3. Tone Mapping Mantiuk
    # cv2.createTonemapMantiuk(gamma, scale, saturation)
    # - gamma     : Koreksi gamma (default 1.0)
    # - scale     : Faktor skala kontras (0.6-0.9)
    # - saturation: Tingkat saturasi warna
    # ============================================================

    # Konfigurasi variasi parameter Mantiuk
    mantiuk_configs = [
        (1.0, 0.7, 1.0, "Default\ng=1.0, sc=0.7, sat=1.0"),
        (2.2, 0.7, 1.0, "Gamma 2.2\ng=2.2, sc=0.7, sat=1.0"),
        (1.0, 0.85, 1.0, "High Scale\ng=1.0, sc=0.85, sat=1.0"),
        (1.0, 0.7, 0.5,  "Low Saturation\ng=1.0, sc=0.7, sat=0.5"),
    ]

    # Menyimpan hasil Mantiuk
    mantiuk_results = []

    print("\n--- Tone Mapping MANTIUK ---")
    for gamma, scale, saturation, label in mantiuk_configs:
        # Membuat objek TonemapMantiuk
        tonemap_mantiuk = cv2.createTonemapMantiuk(
            gamma=gamma,
            scale=scale,
            saturation=saturation
        )

        # Menjalankan tone mapping
        ldr = tonemap_mantiuk.process(hdr)

        # Clipping dan konversi
        ldr = np.clip(ldr, 0, 1)
        ldr_8bit = (ldr * 255).astype(np.uint8)

        # Menyimpan hasil
        mantiuk_results.append((ldr_8bit, label))

        # Statistik
        brightness = np.mean(ldr_8bit)
        print(f"  {label.split(chr(10))[0]:20s} — Kecerahan: {brightness:.1f}")

    # ============================================================
    # 4. Tone Mapping Reinhard (untuk perbandingan)
    # ============================================================

    # Membuat tone mapping Reinhard dengan parameter umum
    tonemap_reinhard = cv2.createTonemapReinhard(
        gamma=1.5, intensity=0.0, light_adapt=0.8, color_adapt=0.6
    )

    # Menjalankan tone mapping Reinhard
    ldr_reinhard = tonemap_reinhard.process(hdr)
    ldr_reinhard = np.clip(ldr_reinhard, 0, 1)
    reinhard_8bit = (ldr_reinhard * 255).astype(np.uint8)

    print(f"\n--- Tone Mapping REINHARD (referensi) ---")
    print(f"  Kecerahan rata-rata: {np.mean(reinhard_8bit):.1f}")

    # ============================================================
    # 5. Visualisasi perbandingan 3 operator
    # ============================================================

    # Membuat figure 3 baris x 4 kolom
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))

    # --- Baris 1: Drago ---
    for idx, (img, label) in enumerate(drago_results):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, idx].imshow(rgb)
        axes[0, idx].set_title(f"Drago #{idx+1}\n{label}", fontsize=8)
        axes[0, idx].axis("off")

    # --- Baris 2: Mantiuk ---
    for idx, (img, label) in enumerate(mantiuk_results):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, idx].imshow(rgb)
        axes[1, idx].set_title(f"Mantiuk #{idx+1}\n{label}", fontsize=8)
        axes[1, idx].axis("off")

    # --- Baris 3: Perbandingan terbaik dari setiap operator ---
    # Drago terbaik (default)
    best_drago_rgb = cv2.cvtColor(drago_results[0][0], cv2.COLOR_BGR2RGB)
    axes[2, 0].imshow(best_drago_rgb)
    axes[2, 0].set_title("Best Drago\n(Default)", fontsize=9)
    axes[2, 0].axis("off")

    # Mantiuk terbaik (default)
    best_mantiuk_rgb = cv2.cvtColor(mantiuk_results[0][0], cv2.COLOR_BGR2RGB)
    axes[2, 1].imshow(best_mantiuk_rgb)
    axes[2, 1].set_title("Best Mantiuk\n(Default)", fontsize=9)
    axes[2, 1].axis("off")

    # Reinhard
    reinhard_rgb = cv2.cvtColor(reinhard_8bit, cv2.COLOR_BGR2RGB)
    axes[2, 2].imshow(reinhard_rgb)
    axes[2, 2].set_title("Reinhard\n(g=1.5, la=0.8, ca=0.6)", fontsize=9)
    axes[2, 2].axis("off")

    # Histogram perbandingan kecerahan
    brightness_data = {
        "Drago": np.mean(drago_results[0][0]),
        "Mantiuk": np.mean(mantiuk_results[0][0]),
        "Reinhard": np.mean(reinhard_8bit),
    }
    # Menampilkan bar chart perbandingan kecerahan
    bar_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    axes[2, 3].bar(brightness_data.keys(), brightness_data.values(), color=bar_colors)
    axes[2, 3].set_title("Perbandingan Kecerahan\nRata-rata", fontsize=9)
    axes[2, 3].set_ylabel("Mean Brightness")
    axes[2, 3].set_ylim(0, 255)
    axes[2, 3].grid(axis="y", alpha=0.3)

    # Menambahkan judul utama
    plt.suptitle("Percobaan 3: Perbandingan Tone Mapping — Drago vs Mantiuk vs Reinhard",
                 fontsize=14, fontweight="bold")

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan hasil ke file
    output_path = os.path.join(OUTPUT_DIR, "03_tone_mapping_drago.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 3")
    print("=" * 60)
    print("Operator tone mapping yang dipelajari:")
    print("  1. cv2.createTonemapDrago(gamma, saturation, bias)")
    print("     - Berbasis logaritmik adaptif")
    print("     - bias 0.7-0.9 menghasilkan tampilan natural")
    print("     - saturation mengontrol kekuatan warna")
    print("  2. cv2.createTonemapMantiuk(gamma, scale, saturation)")
    print("     - Berbasis kontras (contrast mapping)")
    print("     - scale mengontrol tingkat kompresi kontras")
    print("     - Cenderung menghasilkan gambar lebih tajam")
    print("  3. cv2.createTonemapReinhard(gamma, intensity, la, ca)")
    print("     - Berbasis adaptasi visual manusia")
    print("     - light_adapt mengontrol adaptasi lokal vs global")
    print("  4. Setiap operator cocok untuk kebutuhan berbeda")
    print("=" * 60)



if __name__ == "__main__":
    main()
