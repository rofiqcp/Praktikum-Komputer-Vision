"""
==========================================================================
 PERCOBAAN 12 — ROTASI GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Memutar gambar pada sudut tertentu menggunakan OpenCV.
 Konsep  : cv2.getRotationMatrix2D(), cv2.warpAffine(),
           rotasi dengan/tanpa cropping pinggir.
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


def rotasi_sederhana(img, sudut, skala=1.0):
    """
    Rotasi gambar sebesar 'sudut' derajat dari titik tengah.
    cv2.getRotationMatrix2D(center, angle, scale) → matriks 2x3.
    cv2.warpAffine(img, M, (W, H)) → terapkan transformasi.
    Catatan: bagian luar frame menjadi hitam jika tidak diatur.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, sudut, skala)
    hasil = cv2.warpAffine(img, M, (w, h))
    print(f"  Rotasi {sudut}° (skala {skala}) dari center {center}")
    return hasil


def rotasi_tanpa_crop(img, sudut):
    """
    Rotasi gambar dengan memperbesar kanvas agar tidak ada bagian
    gambar yang terpotong. Menghitung ukuran baru secara geometris.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, sudut, 1.0)

    # Hitung ukuran bounding box setelah rotasi
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Sesuaikan translation agar gambar tetap di tengah
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    hasil = cv2.warpAffine(img, M, (new_w, new_h))
    print(f"  Rotasi {sudut}° tanpa crop: {w}x{h} → {new_w}x{new_h}")
    return hasil


def rotasi_multi_sudut(img, sudut_list=[0, 45, 90, 135, 180, 270]):
    """Rotasi gambar ke beberapa sudut sekaligus."""
    hasil = []
    for s in sudut_list:
        r = rotasi_sederhana(img, s)
        hasil.append((s, r))
    return hasil


def tampilkan_hasil(img, rotasi_list, rot_nocrop):
    """Visualisasi rotasi berbagai sudut."""
    n = len(rotasi_list)
    fig, axes = plt.subplots(2, max(3, (n + 1) // 2), figsize=(18, 10))

    for i, (sudut, r) in enumerate(rotasi_list):
        row, col = divmod(i, axes.shape[1])
        axes[row, col].imshow(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f"Rotasi {sudut}°")
        axes[row, col].axis("off")

    # Rotasi tanpa crop di slot terakhir
    last_row = (n) // axes.shape[1]
    last_col = (n) % axes.shape[1]
    if last_row < axes.shape[0] and last_col < axes.shape[1]:
        axes[last_row, last_col].imshow(cv2.cvtColor(rot_nocrop, cv2.COLOR_BGR2RGB))
        axes[last_row, last_col].set_title("45° Tanpa Crop")
        axes[last_row, last_col].axis("off")

    # Sembunyikan subplot kosong
    for ax in axes.flatten():
        if not ax.has_data():
            ax.axis("off")

    plt.suptitle("Percobaan 12 — Rotasi Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "12_rotasi_gambar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 12: ROTASI GAMBAR")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "foto_burung.jpg"))

    print("\n[1] Rotasi berbagai sudut:")
    rotasi_list = rotasi_multi_sudut(img)

    print("\n[2] Rotasi 45° tanpa crop:")
    rot_nocrop = rotasi_tanpa_crop(img, 45)

    tampilkan_hasil(img, rotasi_list, rot_nocrop)

    print("\nRINGKASAN:")
    print("  cv2.getRotationMatrix2D(center, angle, scale)")
    print("  cv2.warpAffine(img, M, (W, H))")
    print("  Tanpa crop → perbesar kanvas sesuai bounding box")


if __name__ == "__main__":
    main()
