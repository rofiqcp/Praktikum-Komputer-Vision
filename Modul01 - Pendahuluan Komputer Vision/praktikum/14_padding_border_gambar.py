"""
==========================================================================
 PERCOBAAN 14 — PADDING DAN BORDER GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Menambahkan border/padding di sekeliling gambar.
 Konsep  : cv2.copyMakeBorder() dengan mode CONSTANT, REFLECT,
           REPLICATE, WRAP. Berguna untuk konvolusi dan estetika.
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


def padding_constant(img, top=50, bottom=50, left=50, right=50, warna=(255, 0, 0)):
    """
    BORDER_CONSTANT: isi border dengan warna solid tertentu.
    Berguna untuk menambahkan bingkai foto.
    """
    hasil = cv2.copyMakeBorder(img, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=warna)
    print(f"  CONSTANT (warna={warna}): {img.shape[:2]} → {hasil.shape[:2]}")
    return hasil


def padding_reflect(img, size=50):
    """
    BORDER_REFLECT: piksel dipantulkan seperti cermin.
    Berguna untuk konvolusi agar tepi tidak menghasilkan artefak.
    """
    hasil = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_REFLECT)
    print(f"  REFLECT: {img.shape[:2]} → {hasil.shape[:2]}")
    return hasil


def padding_replicate(img, size=50):
    """
    BORDER_REPLICATE: piksel terakhir di tepi diulang.
    """
    hasil = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_REPLICATE)
    print(f"  REPLICATE: {img.shape[:2]} → {hasil.shape[:2]}")
    return hasil


def padding_wrap(img, size=50):
    """
    BORDER_WRAP: gambar dibungkus (wrap around).
    Piksel kanan muncul di kiri, atas muncul di bawah.
    """
    hasil = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_WRAP)
    print(f"  WRAP: {img.shape[:2]} → {hasil.shape[:2]}")
    return hasil


def tampilkan_hasil(img, p_const, p_reflect, p_replicate, p_wrap):
    """Visualisasi 4 mode padding."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
    axes[0, 1].imshow(cv2.cvtColor(p_const, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("CONSTANT (biru)"); axes[0, 1].axis("off")
    axes[0, 2].imshow(cv2.cvtColor(p_reflect, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("REFLECT"); axes[0, 2].axis("off")
    axes[1, 0].imshow(cv2.cvtColor(p_replicate, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("REPLICATE"); axes[1, 0].axis("off")
    axes[1, 1].imshow(cv2.cvtColor(p_wrap, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("WRAP"); axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    plt.suptitle("Percobaan 14 — Padding & Border", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "14_padding_border.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 14: PADDING DAN BORDER GAMBAR")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "fruits.jpg"))
    img = cv2.resize(img, (300, 300))  # kecilkan agar border terlihat jelas

    print("\n[1] Mode padding:")
    p_const     = padding_constant(img, warna=(255, 0, 0))
    p_reflect   = padding_reflect(img)
    p_replicate = padding_replicate(img)
    p_wrap      = padding_wrap(img)

    tampilkan_hasil(img, p_const, p_reflect, p_replicate, p_wrap)

    print("\nRINGKASAN:")
    print("  cv2.copyMakeBorder(img, top, bottom, left, right, mode)")
    print("  CONSTANT  → warna solid  |  REFLECT → cermin")
    print("  REPLICATE → ulang tepi   |  WRAP    → bungkus")


if __name__ == "__main__":
    main()
