"""
==========================================================================
 PERCOBAAN 2 — PROPERTI GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Menampilkan dan menganalisis properti dasar sebuah gambar
           digital: dimensi, jumlah channel, tipe data, statistik piksel.
 Konsep  : shape, dtype, size, ndim, min/max/mean/std dari NumPy.
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


def analisis_properti(img, label="Gambar"):
    """
    Mencetak semua properti penting dari sebuah gambar.
    - shape   : (tinggi, lebar, channel)
    - dtype   : tipe data piksel (uint8, float32, dll.)
    - size    : total elemen piksel
    - ndim    : jumlah dimensi array
    """
    print(f"\n--- {label} ---")
    print(f"  Shape (H,W,C) : {img.shape}")
    print(f"  Dimensi (ndim): {img.ndim}")
    print(f"  Dtype          : {img.dtype}")
    print(f"  Total piksel   : {img.size:,}")
    print(f"  Ukuran memori  : {img.nbytes:,} bytes ({img.nbytes/1024:.1f} KB)")
    print(f"  Min / Max      : {img.min()} / {img.max()}")
    print(f"  Mean ± Std     : {img.mean():.2f} ± {img.std():.2f}")
    return {
        "label": label, "shape": img.shape, "dtype": str(img.dtype),
        "size": img.size, "nbytes": img.nbytes,
        "min": int(img.min()), "max": int(img.max()),
        "mean": float(img.mean()), "std": float(img.std()),
    }


def visualisasi_properti(img_color, img_gray, props):
    """Buat figure yang menunjukkan gambar beserta propertinya."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Color {img_color.shape}\n"
                      f"dtype={img_color.dtype}  mem={img_color.nbytes/1024:.0f}KB")
    axes[0].axis("off")

    axes[1].imshow(img_gray, cmap="gray")
    axes[1].set_title(f"Grayscale {img_gray.shape}\n"
                      f"dtype={img_gray.dtype}  mem={img_gray.nbytes/1024:.0f}KB")
    axes[1].axis("off")

    plt.suptitle("Percobaan 2 — Properti Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "02_properti_gambar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 2: PROPERTI GAMBAR")
    print("=" * 60)

    path = os.path.join(IMAGE_DIR, "foto_kucing.jpg")
    img_color = cv2.imread(path, cv2.IMREAD_COLOR)
    img_gray  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    p1 = analisis_properti(img_color, "Gambar Berwarna (BGR)")
    p2 = analisis_properti(img_gray, "Gambar Grayscale")

    visualisasi_properti(img_color, img_gray, [p1, p2])

    print("\nRINGKASAN:")
    print("  img.shape → (H, W, C)  |  img.dtype → uint8/float32")
    print("  img.size  → total elemen |  img.nbytes → ukuran memori")


if __name__ == "__main__":
    main()
