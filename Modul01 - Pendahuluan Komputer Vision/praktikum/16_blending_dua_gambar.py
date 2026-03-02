"""
==========================================================================
 PERCOBAAN 16 — BLENDING DUA GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Mencampur (blend) dua gambar dengan bobot berbeda dan
           membuat efek transisi/fade.
 Konsep  : cv2.addWeighted(img1, alpha, img2, beta, gamma),
           alpha blending: dst = α·src1 + (1-α)·src2.
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


def samakan_ukuran(img1, img2):
    """Resize kedua gambar ke ukuran minimum bersama."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return img1[:h, :w], img2[:h, :w]


def blend_dua_gambar(img1, img2, alpha=0.5):
    """
    cv2.addWeighted(src1, alpha, src2, beta, gamma) → dst
    dst = alpha * src1 + beta * src2 + gamma
    Dengan beta = 1-alpha, gamma = 0 → interpolasi linear.
    """
    a, b = samakan_ukuran(img1, img2)
    beta = 1.0 - alpha
    hasil = cv2.addWeighted(a, alpha, b, beta, 0)
    print(f"  Blend: alpha={alpha:.1f}, beta={beta:.1f}")
    return hasil


def buat_transisi(img1, img2, steps=5):
    """
    Membuat efek fade transisi dari gambar 1 ke gambar 2.
    Menghasilkan serangkaian gambar dengan alpha bertahap.
    """
    a, b = samakan_ukuran(img1, img2)
    transisi = []
    for i in range(steps + 1):
        alpha = i / steps
        blended = cv2.addWeighted(a, 1 - alpha, b, alpha, 0)
        transisi.append((alpha, blended))
    print(f"  Transisi: {steps + 1} frame (alpha 0.0 → 1.0)")
    return transisi


def tampilkan_hasil(img1, img2, blend, transisi):
    """Visualisasi blending dan transisi."""
    n_trans = len(transisi)
    fig, axes = plt.subplots(2, max(3, n_trans), figsize=(3 * n_trans, 8))

    # Baris 1: gambar asli + blend
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Gambar 1"); axes[0, 0].axis("off")
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Gambar 2"); axes[0, 1].axis("off")
    axes[0, 2].imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Blend α=0.5"); axes[0, 2].axis("off")
    for i in range(3, axes.shape[1]):
        axes[0, i].axis("off")

    # Baris 2: transisi
    for i, (alpha, img) in enumerate(transisi):
        if i < axes.shape[1]:
            axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f"α={alpha:.1f}")
            axes[1, i].axis("off")

    plt.suptitle("Percobaan 16 — Blending Dua Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "16_blending_gambar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 16: BLENDING DUA GAMBAR")
    print("=" * 60)

    img1 = cv2.imread(os.path.join(IMAGE_DIR, "foto_malam.jpg"))
    img2 = cv2.imread(os.path.join(IMAGE_DIR, "foto_siang.jpg"))

    print("\n[1] Blend dengan alpha=0.5:")
    blend = blend_dua_gambar(img1, img2, 0.5)

    print("\n[2] Transisi fade (6 langkah):")
    transisi = buat_transisi(img1, img2, 5)

    img1s, img2s = samakan_ukuran(img1, img2)
    tampilkan_hasil(img1s, img2s, blend, transisi)

    print("\nRINGKASAN:")
    print("  cv2.addWeighted(img1, α, img2, 1-α, 0)")
    print("  α mendekati 1 → lebih banyak img1")
    print("  α mendekati 0 → lebih banyak img2")


if __name__ == "__main__":
    main()
