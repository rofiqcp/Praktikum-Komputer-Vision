"""
==========================================================================
 PERCOBAAN 10 — RESIZE DAN SCALING
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Mengubah ukuran gambar menggunakan cv2.resize() dengan
           berbagai metode interpolasi.
 Konsep  : INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA,
           INTER_LANCZOS4 — kapan dan mengapa pakai masing-masing.
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


def resize_ukuran_absolut(img, lebar_baru, tinggi_baru):
    """
    Resize ke ukuran absolut (lebar, tinggi) tertentu.
    cv2.resize(img, (width, height)) — perhatikan urutan: (W, H)!
    """
    hasil = cv2.resize(img, (lebar_baru, tinggi_baru))
    print(f"  {img.shape[:2]} → ({tinggi_baru}, {lebar_baru})")
    return hasil


def resize_dengan_skala(img, fx=0.5, fy=0.5):
    """
    Resize berdasarkan faktor skala (fx=horizontal, fy=vertikal).
    fx=0.5 → gambar menjadi setengah lebar.
    """
    hasil = cv2.resize(img, None, fx=fx, fy=fy)
    print(f"  fx={fx}, fy={fy} → {img.shape[:2]} → {hasil.shape[:2]}")
    return hasil


def bandingkan_interpolasi(img, skala=3.0):
    """
    Membandingkan 5 metode interpolasi saat ZOOM IN (upscale).
    - NEAREST  : piksel terdekat (blocky, cepat)
    - LINEAR   : bilinear (default, smooth)
    - CUBIC    : bicubic (lebih smooth, lebih lambat)
    - AREA     : terbaik untuk downscale
    - LANCZOS4 : paling tajam untuk upscale
    """
    metode = [
        (cv2.INTER_NEAREST,  "NEAREST"),
        (cv2.INTER_LINEAR,   "LINEAR"),
        (cv2.INTER_CUBIC,    "CUBIC"),
        (cv2.INTER_AREA,     "AREA"),
        (cv2.INTER_LANCZOS4, "LANCZOS4"),
    ]
    # Crop kecil dulu agar zoom terlihat jelas
    roi = img[100:150, 100:150]
    hasil = {}
    for flag, nama in metode:
        resized = cv2.resize(roi, None, fx=skala, fy=skala, interpolation=flag)
        hasil[nama] = resized
        print(f"  {nama:10s} → {resized.shape[:2]}")
    return roi, hasil


def tampilkan_hasil(img, img_kecil, img_skala, roi, interpolasi):
    """Visualisasi perbandingan resize."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Baris 1: resize dasar
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Original {img.shape[1]}x{img.shape[0]}")
    axes[0, 1].imshow(cv2.cvtColor(img_kecil, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Resize 320x240")
    axes[0, 2].imshow(cv2.cvtColor(img_skala, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Skala 0.5x")
    axes[0, 3].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title("ROI (zoom source)")

    # Baris 2: perbandingan interpolasi
    interp_items = list(interpolasi.items())[:4]
    for i, (nama, resized) in enumerate(interp_items):
        axes[1, i].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(nama)

    for ax in axes.flatten():
        ax.axis("off")

    plt.suptitle("Percobaan 10 — Resize & Scaling", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "10_resize_scaling.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 10: RESIZE DAN SCALING")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "foto_tekstur.jpg"))

    print("\n[1] Resize ukuran absolut:")
    img_kecil = resize_ukuran_absolut(img, 320, 240)

    print("\n[2] Resize dengan skala:")
    img_skala = resize_dengan_skala(img, 0.5, 0.5)

    print("\n[3] Perbandingan interpolasi (zoom 3x):")
    roi, interpolasi = bandingkan_interpolasi(img, 3.0)

    tampilkan_hasil(img, img_kecil, img_skala, roi, interpolasi)

    print("\nRINGKASAN:")
    print("  cv2.resize(img, (W,H))       → resize absolut")
    print("  cv2.resize(img, None, fx, fy) → resize relatif")
    print("  Upscale: LANCZOS4/CUBIC  |  Downscale: AREA")


if __name__ == "__main__":
    main()
