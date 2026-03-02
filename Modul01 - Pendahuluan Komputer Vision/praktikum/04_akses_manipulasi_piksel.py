"""
==========================================================================
 PERCOBAAN 4 — AKSES DAN MANIPULASI PIKSEL
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Memahami cara mengakses piksel tunggal, memodifikasi area
           piksel, serta memahami sistem koordinat gambar OpenCV.
 Konsep  : img[y, x], slicing img[y1:y2, x1:x2], copy vs reference,
           koordinat: (0,0) = kiri atas, y=baris, x=kolom.
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def akses_piksel_tunggal(img):
    """
    Mengakses nilai piksel pada posisi tertentu.
    Koordinat OpenCV: img[y, x]  (baris, kolom).
    Untuk gambar BGR → mengembalikan (B, G, R).
    """
    h, w = img.shape[:2]
    # Piksel pojok kiri atas
    px_00 = img[0, 0]
    # Piksel tengah
    px_mid = img[h // 2, w // 2]
    # Piksel pojok kanan bawah
    px_end = img[h - 1, w - 1]

    print(f"  Posisi (0,0)       → BGR = {px_00}")
    print(f"  Posisi tengah      → BGR = {px_mid}")
    print(f"  Posisi ({h-1},{w-1}) → BGR = {px_end}")
    return px_00, px_mid, px_end


def modifikasi_area_piksel(img):
    """
    Mengubah warna blok piksel menggunakan slicing NumPy.
    img[y1:y2, x1:x2] = nilai_baru
    PENTING: gunakan .copy() agar gambar asli tidak berubah.
    """
    hasil = img.copy()  # copy agar original aman

    h, w = hasil.shape[:2]
    bh, bw = h // 5, w // 5  # ukuran blok 1/5 gambar

    # Blok merah di kiri atas
    hasil[0:bh, 0:bw] = (0, 0, 255)
    # Blok hijau di kanan atas
    hasil[0:bh, w - bw:w] = (0, 255, 0)
    # Blok biru di kiri bawah
    hasil[h - bh:h, 0:bw] = (255, 0, 0)
    # Blok kuning di kanan bawah
    hasil[h - bh:h, w - bw:w] = (0, 255, 255)

    print("  Empat blok warna ditambahkan di keempat pojok.")
    return hasil


def demo_copy_vs_reference(img):
    """
    Menunjukkan perbedaan copy (independen) vs reference (shared).
    Tanpa copy, perubahan akan mempengaruhi gambar asli!
    """
    # Reference (alias) — modifikasi mempengaruhi asli
    ref = img
    ref_asli_sebelum = img[0, 0].copy()
    ref[0, 0] = [0, 0, 0]
    ref_asli_sesudah = img[0, 0].copy()
    img[0, 0] = ref_asli_sebelum  # kembalikan

    # Copy — modifikasi TIDAK mempengaruhi asli
    salinan = img.copy()
    salinan[0, 0] = [0, 0, 0]
    asli_setelah_copy = img[0, 0].copy()

    print(f"  Reference: sebelum={ref_asli_sebelum} → setelah={ref_asli_sesudah} (BERUBAH)")
    print(f"  Copy:      asli setelah modify copy = {asli_setelah_copy} (TIDAK berubah)")


def tampilkan_hasil(img_original, img_modified):
    """Visualisasi perbandingan sebelum dan sesudah modifikasi."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img_modified, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Setelah Modifikasi Piksel")
    axes[1].axis("off")

    plt.suptitle("Percobaan 4 — Akses & Manipulasi Piksel", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "04_akses_manipulasi_piksel.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 4: AKSES DAN MANIPULASI PIKSEL")
    print("=" * 60)

    path = os.path.join(IMAGE_DIR, "foto_kucing.jpg")
    img = cv2.imread(path)

    print("\n[1] Akses piksel tunggal:")
    akses_piksel_tunggal(img)

    print("\n[2] Copy vs Reference:")
    demo_copy_vs_reference(img)

    print("\n[3] Modifikasi area piksel:")
    img_mod = modifikasi_area_piksel(img)
    tampilkan_hasil(img, img_mod)

    print("\nRINGKASAN:")
    print("  img[y, x]            → akses piksel (0,0)=kiri-atas")
    print("  img[y1:y2, x1:x2]   → slicing area piksel")
    print("  img.copy()           → salin agar original aman")


if __name__ == "__main__":
    main()
