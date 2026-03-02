"""
==========================================================================
 PERCOBAAN 15 — SPLITTING DAN MERGING CHANNEL
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Memisahkan (split) dan menggabungkan (merge) channel warna
           dari gambar BGR. Melihat kontribusi masing-masing channel.
 Konsep  : cv2.split(), cv2.merge(), visualisasi per-channel.
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


def split_channel(img):
    """
    cv2.split(img) → memisahkan gambar 3-channel menjadi 3 gambar 1-channel.
    Return: (B, G, R) masing-masing berupa array 2-D.
    """
    b, g, r = cv2.split(img)
    print(f"  Split: B={b.shape}, G={g.shape}, R={r.shape}")
    return b, g, r


def merge_channel(b, g, r):
    """
    cv2.merge([b, g, r]) → menggabungkan 3 channel menjadi 1 gambar BGR.
    """
    merged = cv2.merge([b, g, r])
    print(f"  Merge: {merged.shape}")
    return merged


def visualisasi_channel_berwarna(img, b, g, r):
    """
    Menampilkan setiap channel DENGAN warna aslinya:
    - Channel B: gambar biru (B, 0, 0)
    - Channel G: gambar hijau (0, G, 0)
    - Channel R: gambar merah (0, 0, R)
    """
    zeros = np.zeros_like(b)
    img_b = cv2.merge([b, zeros, zeros])     # hanya biru
    img_g = cv2.merge([zeros, g, zeros])     # hanya hijau
    img_r = cv2.merge([zeros, zeros, r])     # hanya merah
    return img_b, img_g, img_r


def swap_channel(img):
    """Menukar channel: BGR → RGB (tanpa cvtColor)."""
    b, g, r = cv2.split(img)
    swapped = cv2.merge([r, g, b])  # sekarang RGB jika ditampilkan sebagai BGR
    print("  Channel di-swap: BGR → RGB")
    return swapped


def tampilkan_hasil(img, b, g, r, img_b, img_g, img_r, swapped):
    """Visualisasi semua channel."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original BGR"); axes[0, 0].axis("off")
    axes[0, 1].imshow(b, cmap="gray")
    axes[0, 1].set_title("Channel B (gray)"); axes[0, 1].axis("off")
    axes[0, 2].imshow(g, cmap="gray")
    axes[0, 2].set_title("Channel G (gray)"); axes[0, 2].axis("off")
    axes[0, 3].imshow(r, cmap="gray")
    axes[0, 3].set_title("Channel R (gray)"); axes[0, 3].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Channel B (warna)"); axes[1, 0].axis("off")
    axes[1, 1].imshow(cv2.cvtColor(img_g, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Channel G (warna)"); axes[1, 1].axis("off")
    axes[1, 2].imshow(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Channel R (warna)"); axes[1, 2].axis("off")
    axes[1, 3].imshow(cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title("Channel Swap"); axes[1, 3].axis("off")

    plt.suptitle("Percobaan 15 — Splitting & Merging Channel", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "15_splitting_merging.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 15: SPLITTING DAN MERGING CHANNEL")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "foto_bunga2.jpg"))

    print("\n[1] Split BGR:")
    b, g, r = split_channel(img)

    print("\n[2] Visualisasi channel berwarna:")
    img_b, img_g, img_r = visualisasi_channel_berwarna(img, b, g, r)

    print("\n[3] Swap channel:")
    swapped = swap_channel(img)

    print("\n[4] Merge kembali:")
    merged = merge_channel(b, g, r)

    tampilkan_hasil(img, b, g, r, img_b, img_g, img_r, swapped)

    print("\nRINGKASAN:")
    print("  cv2.split(img)       → pisahkan B, G, R")
    print("  cv2.merge([b, g, r]) → gabungkan kembali")


if __name__ == "__main__":
    main()
