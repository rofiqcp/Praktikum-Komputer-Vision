"""
==========================================================================
 PERCOBAAN 20 — MENYIMPAN BERBAGAI FORMAT GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Menyimpan gambar dalam berbagai format (JPEG, PNG, BMP, TIFF)
           dan memahami perbedaan kualitas serta ukuran file.
 Konsep  : cv2.imwrite(path, img, [params])
           JPEG → lossy, quality 0-100 (cv2.IMWRITE_JPEG_QUALITY)
           PNG  → lossless, compression 0-9 (cv2.IMWRITE_PNG_COMPRESSION)
           BMP  → tanpa kompresi, ukuran besar
           TIFF → lossless, mendukung banyak mode warna
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


def simpan_jpeg(img, quality, nama):
    """
    Menyimpan gambar sebagai JPEG dengan kualitas tertentu.
    IMWRITE_JPEG_QUALITY: 0 (terburuk) - 100 (terbaik)
    """
    path = os.path.join(OUTPUT_DIR, nama)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    size = os.path.getsize(path) / 1024  # KB
    print(f"  JPEG q={quality:3d} → {size:8.1f} KB  ({path})")
    return path, size


def simpan_png(img, compression, nama):
    """
    Menyimpan gambar sebagai PNG dengan level kompresi.
    IMWRITE_PNG_COMPRESSION: 0 (tanpa kompresi) - 9 (kompresi maks)
    PNG selalu lossless (kualitas tidak berubah).
    """
    path = os.path.join(OUTPUT_DIR, nama)
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    size = os.path.getsize(path) / 1024
    print(f"  PNG  c={compression} → {size:8.1f} KB  ({path})")
    return path, size


def simpan_format_lain(img, nama, ext):
    """Menyimpan gambar dalam format BMP/TIFF."""
    path = os.path.join(OUTPUT_DIR, nama)
    cv2.imwrite(path, img)
    size = os.path.getsize(path) / 1024
    print(f"  {ext.upper():4s}   → {size:8.1f} KB  ({path})")
    return path, size


def bandingkan_kualitas_jpeg(img):
    """Membandingkan kualitas JPEG: 10, 50, 90."""
    hasil = []
    for q in [10, 50, 90]:
        path, size = simpan_jpeg(img, q, f"20_jpeg_q{q}.jpg")
        img_baca = cv2.imread(path)
        hasil.append((q, img_baca, size))
    return hasil


def tampilkan_hasil(img, jpeg_results, sizes_dict):
    """Visualisasi perbandingan format dan kualitas."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Baris 1: Perbandingan JPEG
    for i, (q, im, sz) in enumerate(jpeg_results):
        axes[0, i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"JPEG q={q}\n({sz:.1f} KB)")
        axes[0, i].axis("off")

    # Baris 2 kiri: Original
    axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")

    # Baris 2 tengah: Perbandingan ukuran
    formats = list(sizes_dict.keys())
    sizes   = list(sizes_dict.values())
    colors  = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    bars = axes[1, 1].barh(formats, sizes, color=colors[:len(formats)])
    axes[1, 1].set_xlabel("Ukuran (KB)")
    axes[1, 1].set_title("Perbandingan Ukuran File")
    for bar, s in zip(bars, sizes):
        axes[1, 1].text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                        f'{s:.0f} KB', va='center', fontsize=9)

    # Baris 2 kanan: Tabel info
    axes[1, 2].axis("off")
    info = (
        "FORMAT  | KOMPRESI  | KUALITAS\n"
        "--------|-----------|----------\n"
        "JPEG    | Lossy     | Bervariasi\n"
        "PNG     | Lossless  | Sempurna\n"
        "BMP     | Tidak ada | Sempurna\n"
        "TIFF    | Lossless  | Sempurna"
    )
    axes[1, 2].text(0.1, 0.5, info, fontfamily='monospace',
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle("Percobaan 20 — Menyimpan Berbagai Format Gambar",
                 fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "20_berbagai_format.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 20: MENYIMPAN BERBAGAI FORMAT GAMBAR")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "lena.jpg"))
    h, w = img.shape[:2]
    print(f"\n  Ukuran gambar: {w}x{h}")
    print(f"  Ukuran memori: {img.nbytes / 1024:.1f} KB")

    # 1) JPEG dengan berbagai kualitas
    print("\n[1] JPEG — Berbagai Kualitas:")
    jpeg_results = bandingkan_kualitas_jpeg(img)

    # 2) PNG dengan berbagai kompresi
    print("\n[2] PNG — Berbagai Level Kompresi:")
    _, sz_png0 = simpan_png(img, 0, "20_png_c0.png")
    _, sz_png9 = simpan_png(img, 9, "20_png_c9.png")

    # 3) Format lain
    print("\n[3] Format Lainnya:")
    _, sz_bmp  = simpan_format_lain(img, "20_gambar.bmp", "BMP")
    _, sz_tiff = simpan_format_lain(img, "20_gambar.tiff", "TIFF")

    # Kumpulkan ukuran untuk perbandingan
    sizes_dict = {
        "JPEG q=10": jpeg_results[0][2],
        "JPEG q=90": jpeg_results[2][2],
        "PNG c=9":   sz_png9,
        "BMP":       sz_bmp,
        "TIFF":      sz_tiff,
    }

    tampilkan_hasil(img, jpeg_results, sizes_dict)

    print("\nRINGKASAN:")
    print("  JPEG → lossy, ukuran kecil, quality 0-100")
    print("  PNG  → lossless, kompresi 0-9, mendukung transparansi")
    print("  BMP  → tanpa kompresi, ukuran paling besar")
    print("  TIFF → lossless, cocok untuk pengolahan citra profesional")
    print("  cv2.imwrite(path, img, [PARAM, value]) untuk menyimpan")


if __name__ == "__main__":
    main()
