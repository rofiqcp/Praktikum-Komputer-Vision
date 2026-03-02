"""
==========================================================================
 PERCOBAAN 17 — BRIGHTNESS DAN CONTRAST
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Mengatur kecerahan (brightness) dan kontras (contrast) gambar.
 Konsep  : g(x) = α·f(x) + β
           α (alpha/gain) → kontras (>1 naik, <1 turun)
           β (beta/bias)  → kecerahan (+positif terang, -negatif gelap)
           cv2.convertScaleAbs(src, alpha=, beta=)
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


def atur_brightness(img, beta):
    """
    Menambah/mengurangi kecerahan.
    beta > 0  → lebih terang
    beta < 0  → lebih gelap
    """
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def atur_contrast(img, alpha):
    """
    Mengubah kontras gambar.
    alpha > 1.0  → kontras naik
    alpha < 1.0  → kontras turun
    alpha = 1.0  → tidak berubah
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def atur_brightness_contrast(img, alpha, beta):
    """
    Mengatur kontras DAN kecerahan sekaligus.
    g(x) = alpha * f(x) + beta
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def tampilkan_hasil(img, hasil_brightness, hasil_contrast, hasil_gabung):
    """Visualisasi perubahan brightness & contrast."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # Baris 1: Brightness
    labels_b = [("Gelap (β=-80)", hasil_brightness[0]),
                ("Original", img),
                ("Terang (β=+80)", hasil_brightness[1])]
    for i, (lbl, im) in enumerate(labels_b):
        axes[0, i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(lbl)
        axes[0, i].axis("off")

    # Baris 2: Contrast
    labels_c = [("Rendah (α=0.5)", hasil_contrast[0]),
                ("Original", img),
                ("Tinggi (α=2.0)", hasil_contrast[1])]
    for i, (lbl, im) in enumerate(labels_c):
        axes[1, i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(lbl)
        axes[1, i].axis("off")

    # Baris 3: Gabungan
    labels_g = [("α=0.5, β=-50", hasil_gabung[0]),
                ("Original", img),
                ("α=1.5, β=+30", hasil_gabung[1])]
    for i, (lbl, im) in enumerate(labels_g):
        axes[2, i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[2, i].set_title(lbl)
        axes[2, i].axis("off")

    plt.suptitle("Percobaan 17 — Brightness dan Contrast", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "17_brightness_contrast.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 17: BRIGHTNESS DAN CONTRAST")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "foto_gelap.jpg"))
    print(f"\n  Ukuran gambar: {img.shape}")

    # 1) Brightness
    print("\n[1] Mengatur Brightness (β):")
    gelap  = atur_brightness(img, -80)
    terang = atur_brightness(img, +80)
    print("  β = -80 → lebih gelap")
    print("  β = +80 → lebih terang")

    # 2) Contrast
    print("\n[2] Mengatur Contrast (α):")
    rendah = atur_contrast(img, 0.5)
    tinggi = atur_contrast(img, 2.0)
    print("  α = 0.5 → kontras rendah (pudar)")
    print("  α = 2.0 → kontras tinggi (tajam)")

    # 3) Gabungan
    print("\n[3] Gabungan Brightness & Contrast:")
    gab1 = atur_brightness_contrast(img, 0.5, -50)
    gab2 = atur_brightness_contrast(img, 1.5, +30)
    print("  α=0.5, β=-50 → gelap dan pudar")
    print("  α=1.5, β=+30 → terang dan tajam")

    tampilkan_hasil(
        img,
        [gelap, terang],
        [rendah, tinggi],
        [gab1, gab2]
    )

    print("\nRINGKASAN:")
    print("  g(x) = α · f(x) + β")
    print("  α → kontras,  β → kecerahan")
    print("  cv2.convertScaleAbs(src, alpha=α, beta=β)")


if __name__ == "__main__":
    main()
