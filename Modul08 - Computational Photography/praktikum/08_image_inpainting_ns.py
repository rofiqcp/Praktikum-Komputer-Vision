

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 8: IMAGE INPAINTING — NAVIER-STOKES
    ==========================================================================
    Program ini mempelajari teknik image inpainting menggunakan metode
    Navier-Stokes (NS). Inpainting adalah proses mengisi area rusak/hilang
    pada gambar berdasarkan informasi piksel di sekitarnya.

    Metode Navier-Stokes (Bertalmio et al., 2001):
    - Menggunakan persamaan dinamika fluida untuk menyebarkan informasi
    - Mempropagasi isophote (garis intensitas konstan) ke dalam area rusak
    - Cocok untuk area rusak kecil dengan struktur linier

    Fungsi utama yang dipelajari:
    - cv2.inpaint(src, mask, inpaintRadius, cv2.INPAINT_NS)
    - Variasi inpaint radius dan efeknya
    - Pembuatan dan penggunaan inpainting mask

    Hasil: Visualisasi proses restorasi gambar rusak dengan variasi radius
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

    # Mendapatkan direktori tempat script ini berada
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Mendefinisikan path folder gambar input
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

    # Mendefinisikan path folder output
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    # Membuat folder output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PERCOBAAN 8: IMAGE INPAINTING — NAVIER-STOKES")
    print("=" * 60)

    # ============================================================
    # 1. Membaca gambar rusak dan mask
    # ============================================================

    # Membaca gambar yang memiliki area rusak (teks, goresan, dll)
    damaged_path = os.path.join(IMAGE_DIR, "damaged_image.png")
    damaged_img = cv2.imread(damaged_path)

    # Validasi pembacaan gambar rusak
    if damaged_img is None:
        print(f"[ERROR] Gagal membaca: {damaged_path}")
        print("[INFO] Jalankan download_image.py terlebih dahulu!")
        exit()

    # Membaca mask inpainting (putih = area yang perlu diperbaiki)
    mask_path = os.path.join(IMAGE_DIR, "inpaint_mask.png")
    inpaint_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Validasi pembacaan mask
    if inpaint_mask is None:
        print(f"[ERROR] Gagal membaca: {mask_path}")
        exit()

    # Memastikan mask adalah binary (0 atau 255)
    _, inpaint_mask = cv2.threshold(inpaint_mask, 127, 255, cv2.THRESH_BINARY)

    # Menyamakan ukuran mask dengan gambar jika berbeda
    if inpaint_mask.shape[:2] != damaged_img.shape[:2]:
        inpaint_mask = cv2.resize(inpaint_mask, (damaged_img.shape[1], damaged_img.shape[0]))
        _, inpaint_mask = cv2.threshold(inpaint_mask, 127, 255, cv2.THRESH_BINARY)

    # Menampilkan info gambar
    print(f"[INFO] Gambar rusak : {damaged_img.shape[1]}x{damaged_img.shape[0]}")
    print(f"[INFO] Mask         : {inpaint_mask.shape[1]}x{inpaint_mask.shape[0]}")

    # Menghitung persentase area rusak
    damaged_pixels = np.sum(inpaint_mask > 0)
    total_pixels = inpaint_mask.shape[0] * inpaint_mask.shape[1]
    damage_percent = (damaged_pixels / total_pixels) * 100
    print(f"[INFO] Area rusak   : {damaged_pixels} piksel ({damage_percent:.2f}%)")

    # ============================================================
    # 2. Inpainting Navier-Stokes dengan variasi radius
    # cv2.inpaint(src, mask, inpaintRadius, flags)
    # - src           : Gambar input yang rusak
    # - mask          : Mask (putih = area rusak)
    # - inpaintRadius : Radius area referensi (piksel)
    #                   Besar radius = info lebih luas tapi lebih lambat
    # - flags         : cv2.INPAINT_NS (Navier-Stokes)
    # ============================================================

    # Daftar radius inpainting yang akan diuji
    radii = [1, 2, 3, 5, 7, 10, 15, 20]

    # Menyimpan hasil untuk setiap radius
    ns_results = []

    print(f"\n--- Inpainting Navier-Stokes dengan variasi radius ---")
    for radius in radii:
        # Menerapkan inpainting NS dengan radius tertentu
        restored = cv2.inpaint(
            damaged_img,          # Gambar rusak
            inpaint_mask,         # Mask area yang rusak
            inpaintRadius=radius, # Radius referensi
            flags=cv2.INPAINT_NS # Metode Navier-Stokes
        )

        # Menyimpan hasil
        ns_results.append(restored)

        # Menghitung perbedaan rata-rata pada area yang diperbaiki
        # Hanya menghitung di area mask (piksel yang direstorasi)
        mask_bool = inpaint_mask > 0
        diff = np.abs(restored.astype(np.float32) - damaged_img.astype(np.float32))
        avg_change = np.mean(diff[mask_bool])

        print(f"  Radius={radius:2d} — Perubahan rata-rata pada area rusak: {avg_change:.2f}")

    # ============================================================
    # 3. Visualisasi overlay mask pada gambar
    # ============================================================

    # Membuat overlay gambar rusak dengan mask berwarna merah
    overlay = damaged_img.copy()

    # Membuat mask berwarna merah (BGR format)
    red_mask = np.zeros_like(damaged_img)
    red_mask[:, :, 2] = inpaint_mask  # Channel merah = mask

    # Menggabungkan overlay
    overlay_display = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)

    print(f"\n[INFO] Overlay mask dibuat (merah = area rusak)")

    # ============================================================
    # 4. Analisis detail: Zoom pada area yang diperbaiki
    # ============================================================

    # Mencari bounding box dari area mask terbesar
    contours, _ = cv2.findContours(inpaint_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Mengambil area terbesar jika ada konttur
    if len(contours) > 0:
        # Mengurutkan konttur berdasarkan area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Menambahkan padding untuk crop
        pad = 20
        y1 = max(0, y - pad)
        y2 = min(damaged_img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(damaged_img.shape[1], x + w + pad)

        # Crop area yang diperbaiki
        crop_damaged = damaged_img[y1:y2, x1:x2]
        crop_restored_3 = ns_results[2][y1:y2, x1:x2]   # radius=3
        crop_restored_10 = ns_results[5][y1:y2, x1:x2]  # radius=10

        print(f"[INFO] Area zoom: ({x1},{y1}) - ({x2},{y2})")
    else:
        # Fallback: ambil area tengah
        h, w = damaged_img.shape[:2]
        crop_damaged = damaged_img[h//4:3*h//4, w//4:3*w//4]
        crop_restored_3 = ns_results[2][h//4:3*h//4, w//4:3*w//4]
        crop_restored_10 = ns_results[5][h//4:3*h//4, w//4:3*w//4]

    # ============================================================
    # 5. Visualisasi hasil
    # ============================================================

    # Membuat figure 3 baris x 4 kolom
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))

    # --- Baris 1: Gambar damaged, mask, overlay, dan best result ---
    # Gambar rusak
    damaged_rgb = cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(damaged_rgb)
    axes[0, 0].set_title("Gambar Rusak\n(Input)", fontsize=9)
    axes[0, 0].axis("off")

    # Mask inpainting
    axes[0, 1].imshow(inpaint_mask, cmap='gray')
    axes[0, 1].set_title(f"Inpainting Mask\n({damage_percent:.1f}% area)", fontsize=9)
    axes[0, 1].axis("off")

    # Overlay (merah = area rusak)
    overlay_rgb = cv2.cvtColor(overlay_display, cv2.COLOR_BGR2RGB)
    axes[0, 2].imshow(overlay_rgb)
    axes[0, 2].set_title("Overlay\n(Merah = area rusak)", fontsize=9)
    axes[0, 2].axis("off")

    # Hasil inpainting terbaik (radius=5)
    restored_rgb = cv2.cvtColor(ns_results[3], cv2.COLOR_BGR2RGB)
    axes[0, 3].imshow(restored_rgb)
    axes[0, 3].set_title("Restored (NS)\nRadius=5", fontsize=9)
    axes[0, 3].axis("off")

    # --- Baris 2: Variasi radius (4 contoh) ---
    display_radii_idx = [0, 2, 4, 7]  # radius 1, 3, 7, 20
    for i, idx in enumerate(display_radii_idx):
        rgb = cv2.cvtColor(ns_results[idx], cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(rgb)
        axes[1, i].set_title(f"NS Radius={radii[idx]}", fontsize=9)
        axes[1, i].axis("off")

    # --- Baris 3: Detail zoom dan perbandingan ---
    # Zoom damaged
    crop_d_rgb = cv2.cvtColor(crop_damaged, cv2.COLOR_BGR2RGB)
    axes[2, 0].imshow(crop_d_rgb)
    axes[2, 0].set_title("Zoom: Damaged", fontsize=9)
    axes[2, 0].axis("off")

    # Zoom restored radius=3
    crop_r3_rgb = cv2.cvtColor(crop_restored_3, cv2.COLOR_BGR2RGB)
    axes[2, 1].imshow(crop_r3_rgb)
    axes[2, 1].set_title("Zoom: NS Radius=3", fontsize=9)
    axes[2, 1].axis("off")

    # Zoom restored radius=10
    crop_r10_rgb = cv2.cvtColor(crop_restored_10, cv2.COLOR_BGR2RGB)
    axes[2, 2].imshow(crop_r10_rgb)
    axes[2, 2].set_title("Zoom: NS Radius=10", fontsize=9)
    axes[2, 2].axis("off")

    # Info text tentang metode NS
    info_text = (
        "Metode Navier-Stokes:\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "• Berdasarkan PDE fluida\n"
        "• Propagasi isophote\n"
        f"• Radii diuji: {radii}\n"
        f"• Area rusak: {damage_percent:.1f}%\n\n"
        "Radius kecil:\n"
        "  → Lebih tajam, lokal\n"
        "Radius besar:\n"
        "  → Lebih halus, global"
    )
    axes[2, 3].text(0.1, 0.5, info_text, transform=axes[2, 3].transAxes,
                    fontsize=9, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[2, 3].set_title("Info Metode", fontsize=9)
    axes[2, 3].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Percobaan 8: Image Inpainting — Navier-Stokes (NS)\n"
                 "Restorasi area rusak menggunakan persamaan fluida",
                 fontsize=14, fontweight="bold")

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan hasil
    output_path = os.path.join(OUTPUT_DIR, "08_image_inpainting_ns.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

    # ============================================================
    # RINGKASAN
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 8")
    print("=" * 60)
    print("Fungsi yang dipelajari:")
    print("  1. cv2.inpaint(src, mask, inpaintRadius, cv2.INPAINT_NS)")
    print("     - src         : Gambar input yang rusak")
    print("     - mask        : Binary mask (putih = area rusak)")
    print("     - inpaintRadius: Radius area referensi (1-20)")
    print("     - INPAINT_NS  : Metode Navier-Stokes")
    print("  2. Metode Navier-Stokes:")
    print("     - Berdasarkan Partial Differential Equations (PDE)")
    print("     - Mempropagasi isophote (garis intensitas konstan)")
    print("     - Cocok untuk goresan/retakan tipis")
    print(f"  3. Area rusak: {damaged_pixels} piksel ({damage_percent:.1f}%)")
    print(f"  4. Radius diuji: {radii}")
    print("     - Radius kecil → restorasi lokal, lebih tajam")
    print("     - Radius besar → restorasi luas, lebih halus")
    print("=" * 60)



if __name__ == "__main__":
    main()
