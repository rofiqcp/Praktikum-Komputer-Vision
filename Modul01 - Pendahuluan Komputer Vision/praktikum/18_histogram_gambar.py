"""
==========================================================================
 PERCOBAAN 18 — HISTOGRAM GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Memahami histogram intensitas piksel dan ekualisasi histogram.
 Konsep  : cv2.calcHist()  → menghitung distribusi intensitas
           cv2.equalizeHist() → meratakan distribusi agar kontras meningkat
           Histogram = grafik frekuensi setiap nilai intensitas (0–255).
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


def hitung_histogram(img_gray):
    """
    cv2.calcHist([images], [channels], mask, [histSize], [ranges])
    Mengembalikan array 256 elemen (frekuensi per intensitas).
    """
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    print(f"  Histogram shape: {hist.shape}")
    print(f"  Total piksel : {int(hist.sum())}")
    return hist


def hitung_histogram_warna(img_bgr):
    """Menghitung histogram untuk setiap channel BGR."""
    warna = ('b', 'g', 'r')
    labels = ('Biru', 'Hijau', 'Merah')
    hists = {}
    for ch, w, lbl in zip(range(3), warna, labels):
        hists[lbl] = cv2.calcHist([img_bgr], [ch], None, [256], [0, 256])
    return hists


def ekualisasi_histogram(img_gray):
    """
    cv2.equalizeHist() meratakan distribusi intensitas
    sehingga rentang kontras menjadi lebih lebar/merata.
    """
    eq = cv2.equalizeHist(img_gray)
    print("  Ekualisasi histogram selesai")
    return eq


def hitung_cdf(hist):
    """Menghitung Cumulative Distribution Function (CDF)."""
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    return cdf_normalized


def tampilkan_hasil(img_gray, hist_gray, eq_img, hist_eq,
                    img_bgr, hists_warna):
    """Visualisasi histogram dan ekualisasi."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Baris 1: Grayscale original + histogram + CDF
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title("Grayscale"); axes[0, 0].axis("off")

    axes[0, 1].plot(hist_gray, color='black')
    axes[0, 1].set_title("Histogram Gray")
    axes[0, 1].set_xlim([0, 256])

    cdf = hitung_cdf(hist_gray)
    axes[0, 2].plot(cdf, color='blue')
    axes[0, 2].set_title("CDF Original")
    axes[0, 2].set_xlim([0, 256])

    # Baris 2: Ekualisasi + histogram + CDF
    axes[1, 0].imshow(eq_img, cmap='gray')
    axes[1, 0].set_title("Setelah Equalize"); axes[1, 0].axis("off")

    axes[1, 1].plot(hist_eq, color='black')
    axes[1, 1].set_title("Histogram Equalized")
    axes[1, 1].set_xlim([0, 256])

    cdf_eq = hitung_cdf(hist_eq)
    axes[1, 2].plot(cdf_eq, color='blue')
    axes[1, 2].set_title("CDF Equalized")
    axes[1, 2].set_xlim([0, 256])

    # Baris 3: Warna + histogram warna + perbandingan
    axes[2, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title("Gambar Warna"); axes[2, 0].axis("off")

    colors_map = {'Merah': 'red', 'Hijau': 'green', 'Biru': 'blue'}
    for lbl, h in hists_warna.items():
        axes[2, 1].plot(h, color=colors_map[lbl], label=lbl)
    axes[2, 1].set_title("Histogram BGR")
    axes[2, 1].set_xlim([0, 256])
    axes[2, 1].legend()

    # Perbandingan berdampingan
    perbandingan = np.hstack([img_gray, eq_img])
    axes[2, 2].imshow(perbandingan, cmap='gray')
    axes[2, 2].set_title("Original vs Equalized"); axes[2, 2].axis("off")

    plt.suptitle("Percobaan 18 — Histogram Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "18_histogram_gambar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 18: HISTOGRAM GAMBAR")
    print("=" * 60)

    img_bgr = cv2.imread(os.path.join(IMAGE_DIR, "lena.jpg"))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Histogram grayscale
    print("\n[1] Histogram Grayscale:")
    hist_gray = hitung_histogram(img_gray)

    # 2) Ekualisasi
    print("\n[2] Ekualisasi Histogram:")
    eq_img = ekualisasi_histogram(img_gray)
    hist_eq = hitung_histogram(eq_img)

    # 3) Histogram warna
    print("\n[3] Histogram Warna (BGR):")
    hists_warna = hitung_histogram_warna(img_bgr)
    for lbl, h in hists_warna.items():
        print(f"  Channel {lbl}: mean={h.mean():.1f}")

    tampilkan_hasil(img_gray, hist_gray, eq_img, hist_eq,
                    img_bgr, hists_warna)

    print("\nRINGKASAN:")
    print("  Histogram  → distribusi intensitas piksel (0–255)")
    print("  calcHist   → menghitung frekuensi tiap intensitas")
    print("  equalizeHist → meratakan distribusi (peningkatan kontras)")
    print("  CDF → kurva kumulatif, setelah equalize CDF menjadi linear")


if __name__ == "__main__":
    main()
