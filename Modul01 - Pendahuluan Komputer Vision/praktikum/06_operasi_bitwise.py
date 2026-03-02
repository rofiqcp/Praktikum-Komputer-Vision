"""
==========================================================================
 PERCOBAAN 6 — OPERASI BITWISE
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Memahami operasi bitwise (AND, OR, XOR, NOT) pada gambar
           dan penggunaannya untuk masking dan kompositing.
 Konsep  : cv2.bitwise_and/or/xor/not(), mask binary untuk isolasi area.
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


def buat_bentuk_uji():
    """
    Membuat dua gambar biner hitam-putih untuk demo bitwise.
    - Kotak putih di kiri
    - Lingkaran putih di kanan (sedikit overlap)
    """
    img1 = np.zeros((300, 300), dtype=np.uint8)
    img2 = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (200, 250), 255, -1)
    cv2.circle(img2, (200, 150), 120, 255, -1)
    return img1, img2


def operasi_bitwise_dasar(img1, img2):
    """
    Empat operasi bitwise pada gambar:
    - AND : hanya area overlap (irisan)
    - OR  : gabungan kedua area (union)
    - XOR : area yang TIDAK overlap (symmetric difference)
    - NOT : inversi (kebalikan hitam↔putih)
    """
    bw_and = cv2.bitwise_and(img1, img2)
    bw_or  = cv2.bitwise_or(img1, img2)
    bw_xor = cv2.bitwise_xor(img1, img2)
    bw_not = cv2.bitwise_not(img1)

    print("  AND → area irisan saja")
    print("  OR  → gabungan semua area")
    print("  XOR → area non-overlap")
    print("  NOT → inversi gambar")
    return bw_and, bw_or, bw_xor, bw_not


def demo_masking_dengan_bitwise(img_color):
    """
    Menggunakan bitwise_and + mask untuk mengisolasi area tertentu.
    Contoh: buat mask lingkaran, lalu terapkan ke gambar berwarna.
    """
    h, w = img_color.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 3, 255, -1)

    # bitwise_and dengan mask → hanya area dalam lingkaran yg terlihat
    hasil = cv2.bitwise_and(img_color, img_color, mask=mask)
    print(f"  Mask lingkaran diterapkan: radius={min(h,w)//3}")
    return mask, hasil


def tampilkan_hasil(img1, img2, bw_and, bw_or, bw_xor, bw_not,
                    img_color, mask, masked):
    """Visualisasi semua operasi bitwise."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Baris 1: operasi dasar
    axes[0, 0].imshow(img1, cmap="gray"); axes[0, 0].set_title("Kotak")
    axes[0, 1].imshow(img2, cmap="gray"); axes[0, 1].set_title("Lingkaran")
    axes[0, 2].imshow(bw_and, cmap="gray"); axes[0, 2].set_title("AND (Irisan)")
    axes[0, 3].imshow(bw_or, cmap="gray"); axes[0, 3].set_title("OR (Gabungan)")

    # Baris 2: xor, not, masking
    axes[1, 0].imshow(bw_xor, cmap="gray"); axes[1, 0].set_title("XOR")
    axes[1, 1].imshow(bw_not, cmap="gray"); axes[1, 1].set_title("NOT Kotak")
    axes[1, 2].imshow(mask, cmap="gray"); axes[1, 2].set_title("Mask Lingkaran")
    axes[1, 3].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title("Hasil Masking")

    for ax in axes.flatten():
        ax.axis("off")

    plt.suptitle("Percobaan 6 — Operasi Bitwise", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "06_operasi_bitwise.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 6: OPERASI BITWISE")
    print("=" * 60)

    print("\n[1] Operasi bitwise pada bentuk biner:")
    img1, img2 = buat_bentuk_uji()
    bw_and, bw_or, bw_xor, bw_not = operasi_bitwise_dasar(img1, img2)

    print("\n[2] Masking gambar berwarna:")
    img_color = cv2.imread(os.path.join(IMAGE_DIR, "foto_alam.jpg"))
    mask, masked = demo_masking_dengan_bitwise(img_color)

    tampilkan_hasil(img1, img2, bw_and, bw_or, bw_xor, bw_not,
                    img_color, mask, masked)

    print("\nRINGKASAN:")
    print("  cv2.bitwise_and/or/xor/not() → operasi bit per piksel")
    print("  Parameter mask= untuk isolasi area tertentu")


if __name__ == "__main__":
    main()
