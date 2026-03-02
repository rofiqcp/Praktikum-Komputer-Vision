

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 9: IMAGE INPAINTING — TELEA (FAST MARCHING METHOD)
    ==========================================================================
    Program ini mempelajari teknik image inpainting menggunakan metode Telea
    (Fast Marching Method). Metode ini mengisi area rusak berdasarkan estimasi
    cepat dari piksel terdekat — lebih cepat dari Navier-Stokes.

    Metode Telea (Alexandru Telea, 2004):
    - Menggunakan Fast Marching Method (FMM)
    - Mengisi piksel dari batas area rusak ke dalam (inward)
    - Mempertimbangkan gradien, jarak, dan arah propagasi
    - Lebih cepat dan seringkali lebih halus dari NS

    Fungsi utama yang dipelajari:
    - cv2.inpaint(src, mask, inpaintRadius, cv2.INPAINT_TELEA)
    - Perbandingan NS vs Telea side-by-side
    - Pembuatan custom mask (lingkaran, persegi panjang)

    Hasil: Perbandingan visual NS vs Telea dan custom inpainting mask
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
    print("PERCOBAAN 9: IMAGE INPAINTING — TELEA (FMM)")
    print("=" * 60)

    # ============================================================
    # 1. Membaca gambar rusak dan mask
    # ============================================================

    # Membaca gambar yang memiliki area rusak
    damaged_path = os.path.join(IMAGE_DIR, "damaged_image.png")
    damaged_img = cv2.imread(damaged_path)

    # Validasi pembacaan
    if damaged_img is None:
        print(f"[ERROR] Gagal membaca: {damaged_path}")
        print("[INFO] Jalankan download_image.py terlebih dahulu!")
        exit()

    # Membaca mask inpainting
    mask_path = os.path.join(IMAGE_DIR, "inpaint_mask.png")
    inpaint_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Validasi mask
    if inpaint_mask is None:
        print(f"[ERROR] Gagal membaca: {mask_path}")
        exit()

    # Memastikan mask binary
    _, inpaint_mask = cv2.threshold(inpaint_mask, 127, 255, cv2.THRESH_BINARY)

    # Menyamakan ukuran jika berbeda
    if inpaint_mask.shape[:2] != damaged_img.shape[:2]:
        inpaint_mask = cv2.resize(inpaint_mask, (damaged_img.shape[1], damaged_img.shape[0]))
        _, inpaint_mask = cv2.threshold(inpaint_mask, 127, 255, cv2.THRESH_BINARY)

    # Menampilkan info
    h, w = damaged_img.shape[:2]
    print(f"[INFO] Gambar    : {w}x{h}")
    print(f"[INFO] Mask      : {inpaint_mask.shape[1]}x{inpaint_mask.shape[0]}")

    # ============================================================
    # 2. Inpainting Telea (Fast Marching Method)
    # cv2.inpaint(src, mask, inpaintRadius, cv2.INPAINT_TELEA)
    # ============================================================

    # Daftar radius untuk percobaan
    radii = [1, 3, 5, 7, 10, 15]

    # Menyimpan hasil Telea
    telea_results = []
    telea_times = []

    print(f"\n--- Inpainting TELEA (FMM) ---")
    for radius in radii:
        # Mengukur waktu eksekusi
        start = time.time()

        # Menerapkan inpainting Telea
        restored = cv2.inpaint(
            damaged_img,
            inpaint_mask,
            inpaintRadius=radius,
            flags=cv2.INPAINT_TELEA  # Metode Telea (Fast Marching)
        )

        # Menghitung waktu
        elapsed = time.time() - start

        # Menyimpan hasil
        telea_results.append(restored)
        telea_times.append(elapsed)

        print(f"  Radius={radius:2d} — Waktu: {elapsed:.4f}s")

    # ============================================================
    # 3. Inpainting Navier-Stokes untuk perbandingan
    # ============================================================

    # Menyimpan hasil NS
    ns_results = []
    ns_times = []

    print(f"\n--- Inpainting NAVIER-STOKES (perbandingan) ---")
    for radius in radii:
        # Mengukur waktu eksekusi
        start = time.time()

        # Menerapkan inpainting NS
        restored = cv2.inpaint(
            damaged_img,
            inpaint_mask,
            inpaintRadius=radius,
            flags=cv2.INPAINT_NS  # Metode Navier-Stokes
        )

        # Menghitung waktu
        elapsed = time.time() - start

        # Menyimpan hasil
        ns_results.append(restored)
        ns_times.append(elapsed)

        print(f"  Radius={radius:2d} — Waktu: {elapsed:.4f}s")

    # ============================================================
    # 4. Membuat custom mask dan uji inpainting
    # ============================================================

    # Membaca gambar scene untuk custom inpainting demo
    scene_path = os.path.join(IMAGE_DIR, "scene_pemandangan.png")
    scene_img = cv2.imread(scene_path)

    # Jika scene tidak tersedia, gunakan damaged image
    if scene_img is None:
        scene_img = damaged_img.copy()

    # Mendapatkan dimensi gambar scene
    sh, sw = scene_img.shape[:2]

    # --- Membuat custom mask: lingkaran ---
    custom_mask_circle = np.zeros((sh, sw), dtype=np.uint8)

    # Menggambar lingkaran putih di tengah gambar
    center_x, center_y = sw // 2, sh // 2
    radius_circle = min(sh, sw) // 8
    cv2.circle(custom_mask_circle, (center_x, center_y), radius_circle, 255, -1)

    print(f"\n--- Custom Mask: Lingkaran ---")
    print(f"  Pusat: ({center_x}, {center_y}), Radius: {radius_circle}")

    # --- Membuat custom mask: persegi panjang ---
    custom_mask_rect = np.zeros((sh, sw), dtype=np.uint8)

    # Menggambar persegi panjang putih
    rect_x1 = sw // 4
    rect_y1 = sh // 4
    rect_x2 = 3 * sw // 4
    rect_y2 = sh // 4 + sh // 8
    cv2.rectangle(custom_mask_rect, (rect_x1, rect_y1), (rect_x2, rect_y2), 255, -1)

    print(f"\n--- Custom Mask: Persegi Panjang ---")
    print(f"  Dari ({rect_x1},{rect_y1}) ke ({rect_x2},{rect_y2})")

    # Membuat gambar "rusak" dengan area yang ditutup warna hitam
    damaged_circle = scene_img.copy()
    damaged_circle[custom_mask_circle > 0] = 0  # Area lingkaran jadi hitam

    damaged_rect = scene_img.copy()
    damaged_rect[custom_mask_rect > 0] = 0  # Area persegi jadi hitam

    # Inpainting custom mask — lingkaran
    restored_circle_telea = cv2.inpaint(damaged_circle, custom_mask_circle, 5, cv2.INPAINT_TELEA)
    restored_circle_ns = cv2.inpaint(damaged_circle, custom_mask_circle, 5, cv2.INPAINT_NS)

    # Inpainting custom mask — persegi panjang
    restored_rect_telea = cv2.inpaint(damaged_rect, custom_mask_rect, 5, cv2.INPAINT_TELEA)
    restored_rect_ns = cv2.inpaint(damaged_rect, custom_mask_rect, 5, cv2.INPAINT_NS)

    print(f"\n[INFO] Custom inpainting selesai (lingkaran dan persegi)")

    # ============================================================
    # 5. Visualisasi hasil
    # ============================================================

    # Membuat figure 4 baris x 4 kolom
    fig, axes = plt.subplots(4, 4, figsize=(22, 22))

    # --- Baris 1: NS vs Telea side-by-side (original mask) ---
    # Gambar rusak
    damaged_rgb = cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(damaged_rgb)
    axes[0, 0].set_title("Gambar Rusak", fontsize=9)
    axes[0, 0].axis("off")

    # Mask
    axes[0, 1].imshow(inpaint_mask, cmap='gray')
    axes[0, 1].set_title("Inpainting Mask", fontsize=9)
    axes[0, 1].axis("off")

    # Telea radius=5
    telea_rgb = cv2.cvtColor(telea_results[2], cv2.COLOR_BGR2RGB)
    axes[0, 2].imshow(telea_rgb)
    axes[0, 2].set_title(f"TELEA (r=5)\n{telea_times[2]:.4f}s", fontsize=9)
    axes[0, 2].axis("off")

    # NS radius=5
    ns_rgb = cv2.cvtColor(ns_results[2], cv2.COLOR_BGR2RGB)
    axes[0, 3].imshow(ns_rgb)
    axes[0, 3].set_title(f"NS (r=5)\n{ns_times[2]:.4f}s", fontsize=9)
    axes[0, 3].axis("off")

    # --- Baris 2: Variasi radius Telea ---
    display_idx = [0, 1, 3, 5]  # radius 1, 3, 7, 15
    for i, idx in enumerate(display_idx):
        rgb = cv2.cvtColor(telea_results[idx], cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(rgb)
        axes[1, i].set_title(f"Telea r={radii[idx]}\n{telea_times[idx]:.4f}s", fontsize=9)
        axes[1, i].axis("off")

    # --- Baris 3: Custom mask — lingkaran ---
    # Gambar rusak (lingkaran)
    d_circ_rgb = cv2.cvtColor(damaged_circle, cv2.COLOR_BGR2RGB)
    axes[2, 0].imshow(d_circ_rgb)
    axes[2, 0].set_title("Rusak (Lingkaran)", fontsize=9)
    axes[2, 0].axis("off")

    # Mask lingkaran
    axes[2, 1].imshow(custom_mask_circle, cmap='gray')
    axes[2, 1].set_title("Mask Lingkaran", fontsize=9)
    axes[2, 1].axis("off")

    # Telea lingkaran
    rc_telea_rgb = cv2.cvtColor(restored_circle_telea, cv2.COLOR_BGR2RGB)
    axes[2, 2].imshow(rc_telea_rgb)
    axes[2, 2].set_title("Telea (Lingkaran)", fontsize=9)
    axes[2, 2].axis("off")

    # NS lingkaran
    rc_ns_rgb = cv2.cvtColor(restored_circle_ns, cv2.COLOR_BGR2RGB)
    axes[2, 3].imshow(rc_ns_rgb)
    axes[2, 3].set_title("NS (Lingkaran)", fontsize=9)
    axes[2, 3].axis("off")

    # --- Baris 4: Custom mask — persegi dan bar chart waktu ---
    # Gambar rusak (persegi)
    d_rect_rgb = cv2.cvtColor(damaged_rect, cv2.COLOR_BGR2RGB)
    axes[3, 0].imshow(d_rect_rgb)
    axes[3, 0].set_title("Rusak (Persegi)", fontsize=9)
    axes[3, 0].axis("off")

    # Telea persegi
    rr_telea_rgb = cv2.cvtColor(restored_rect_telea, cv2.COLOR_BGR2RGB)
    axes[3, 1].imshow(rr_telea_rgb)
    axes[3, 1].set_title("Telea (Persegi)", fontsize=9)
    axes[3, 1].axis("off")

    # NS persegi
    rr_ns_rgb = cv2.cvtColor(restored_rect_ns, cv2.COLOR_BGR2RGB)
    axes[3, 2].imshow(rr_ns_rgb)
    axes[3, 2].set_title("NS (Persegi)", fontsize=9)
    axes[3, 2].axis("off")

    # Bar chart perbandingan waktu
    x_pos = np.arange(len(radii))
    bar_width = 0.35
    bars1 = axes[3, 3].bar(x_pos - bar_width/2, telea_times, bar_width,
                             label='Telea', color='#2196F3')
    bars2 = axes[3, 3].bar(x_pos + bar_width/2, ns_times, bar_width,
                             label='NS', color='#FF9800')
    axes[3, 3].set_xlabel("Radius")
    axes[3, 3].set_ylabel("Waktu (detik)")
    axes[3, 3].set_title("Perbandingan Waktu\nTelea vs NS", fontsize=9)
    axes[3, 3].set_xticks(x_pos)
    axes[3, 3].set_xticklabels([str(r) for r in radii])
    axes[3, 3].legend(fontsize=8)
    axes[3, 3].grid(axis='y', alpha=0.3)

    # Menambahkan judul utama
    plt.suptitle("Percobaan 9: Image Inpainting — Telea (FMM) vs Navier-Stokes\n"
                 "Perbandingan metode dan custom mask",
                 fontsize=14, fontweight="bold")

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan hasil
    output_path = os.path.join(OUTPUT_DIR, "09_image_inpainting_telea.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 9")
    print("=" * 60)
    print("Fungsi yang dipelajari:")
    print("  1. cv2.inpaint(src, mask, radius, cv2.INPAINT_TELEA)")
    print("     - Metode Fast Marching (Telea 2004)")
    print("     - Mengisi dari batas ke dalam (inward propagation)")
    print("     - Umumnya lebih cepat dari NS")
    print("  2. Perbandingan NS vs Telea:")
    print("     Radius | Telea (s) | NS (s)")
    for i, r in enumerate(radii):
        print(f"       {r:2d}   |  {telea_times[i]:.4f}   | {ns_times[i]:.4f}")
    print("  3. Custom mask dibuat dengan:")
    print("     - cv2.circle() → mask lingkaran")
    print("     - cv2.rectangle() → mask persegi panjang")
    print("  4. Telea umumnya lebih halus, NS lebih baik untuk struktur")
    print("=" * 60)



if __name__ == "__main__":
    main()
