"""
==========================================================================
 PERCOBAAN 13 — FLIP GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Membalik gambar secara horizontal, vertikal, dan keduanya.
 Konsep  : cv2.flip(img, flipCode)
           flipCode=0 → vertikal, 1 → horizontal, -1 → keduanya.
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


def flip_horizontal(img):
    """
    cv2.flip(img, 1) → cermin horizontal (kiri ↔ kanan).
    Berguna untuk augmentasi data dan efek mirror.
    """
    return cv2.flip(img, 1)


def flip_vertikal(img):
    """
    cv2.flip(img, 0) → cermin vertikal (atas ↔ bawah).
    """
    return cv2.flip(img, 0)


def flip_keduanya(img):
    """
    cv2.flip(img, -1) → cermin horizontal + vertikal sekaligus.
    Sama dengan rotasi 180°.
    """
    return cv2.flip(img, -1)


def tampilkan_hasil(img, h_flip, v_flip, hv_flip):
    """Visualisasi semua hasil flip."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original"); axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(h_flip, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Flip Horizontal (code=1)"); axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(v_flip, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Flip Vertikal (code=0)"); axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(hv_flip, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Flip Keduanya (code=-1)"); axes[1, 1].axis("off")

    plt.suptitle("Percobaan 13 — Flip Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "13_flip_gambar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 13: FLIP GAMBAR")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "foto_arsitektur.jpg"))

    print("\n[1] Flip horizontal (kiri ↔ kanan):")
    h_flip = flip_horizontal(img)
    print("  ✓ flipCode=1")

    print("[2] Flip vertikal (atas ↔ bawah):")
    v_flip = flip_vertikal(img)
    print("  ✓ flipCode=0")

    print("[3] Flip keduanya (= rotasi 180°):")
    hv_flip = flip_keduanya(img)
    print("  ✓ flipCode=-1")

    tampilkan_hasil(img, h_flip, v_flip, hv_flip)

    print("\nRINGKASAN:")
    print("  cv2.flip(img, 1)  → horizontal")
    print("  cv2.flip(img, 0)  → vertikal")
    print("  cv2.flip(img, -1) → keduanya (rotasi 180°)")


if __name__ == "__main__":
    main()
