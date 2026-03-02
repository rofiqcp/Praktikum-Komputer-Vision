"""
==========================================================================
 PERCOBAAN 8 — MENULIS TEKS PADA GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Menulis teks / anotasi pada gambar menggunakan OpenCV.
 Konsep  : cv2.putText(), cv2.getTextSize(), font face, skala, warna.
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


def tulis_teks_dasar(img):
    """
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
    - org     : posisi pojok kiri bawah teks (x, y)
    - fontFace: FONT_HERSHEY_SIMPLEX, _COMPLEX, _SCRIPT, dll.
    - fontScale: skala ukuran huruf (1.0 = default)
    """
    hasil = img.copy()
    fonts = [
        (cv2.FONT_HERSHEY_SIMPLEX,  "SIMPLEX"),
        (cv2.FONT_HERSHEY_COMPLEX,  "COMPLEX"),
        (cv2.FONT_HERSHEY_DUPLEX,   "DUPLEX"),
        (cv2.FONT_HERSHEY_TRIPLEX,  "TRIPLEX"),
        (cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, "SCRIPT"),
    ]
    y = 50
    for font, nama in fonts:
        cv2.putText(hasil, f"Font: {nama}", (30, y), font, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        y += 50

    print(f"  {len(fonts)} jenis font ditampilkan.")
    return hasil


def tulis_teks_dengan_background(img, teks, posisi, skala=1.0):
    """
    Menulis teks dengan kotak latar belakang untuk kontras yang baik.
    Menggunakan getTextSize() untuk menghitung ukuran teks.
    """
    hasil = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # Hitung ukuran teks
    (tw, th), baseline = cv2.getTextSize(teks, font, skala, thickness)
    x, y = posisi

    # Gambar background kotak
    cv2.rectangle(hasil, (x - 5, y - th - 5), (x + tw + 5, y + baseline + 5),
                  (0, 0, 0), -1)
    # Tulis teks di atas background
    cv2.putText(hasil, teks, (x, y), font, skala, (255, 255, 255),
                thickness, cv2.LINE_AA)

    print(f"  Teks '{teks}' ukuran: {tw}x{th} px, baseline={baseline}")
    return hasil


def anotasi_gambar(img):
    """Menambahkan anotasi informatif pada gambar (label + panah)."""
    hasil = img.copy()
    h, w = hasil.shape[:2]

    # Label pojok
    cv2.putText(hasil, f"{w}x{h}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 1, cv2.LINE_AA)

    # Anotasi dengan panah
    cv2.arrowedLine(hasil, (w // 2, h // 2 - 30), (w // 2, h // 2 + 30),
                    (0, 0, 255), 2)
    cv2.putText(hasil, "Titik Tengah", (w // 2 + 10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    print("  Anotasi ditambahkan: ukuran + titik tengah")
    return hasil


def tampilkan_hasil(img_fonts, img_bg, img_anotasi):
    """Visualisasi semua demo teks."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(cv2.cvtColor(img_fonts, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Berbagai Font"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Teks + Background"); axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(img_anotasi, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Anotasi Gambar"); axes[2].axis("off")

    plt.suptitle("Percobaan 8 — Menulis Teks pada Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "08_menulis_teks.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 8: MENULIS TEKS PADA GAMBAR")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "foto_kucing.jpg"))

    print("\n[1] Teks dengan berbagai font:")
    img_fonts = tulis_teks_dasar(img)

    print("\n[2] Teks dengan background:")
    img_bg = tulis_teks_dengan_background(img, "Hello OpenCV!", (30, 350), 1.2)

    print("\n[3] Anotasi gambar:")
    img_anotasi = anotasi_gambar(img)

    tampilkan_hasil(img_fonts, img_bg, img_anotasi)

    print("\nRINGKASAN:")
    print("  cv2.putText(img, text, org, font, scale, color, thick)")
    print("  cv2.getTextSize() → hitung lebar & tinggi teks")


if __name__ == "__main__":
    main()
