"""
==========================================================================
 PERCOBAAN 19 — MASKING GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Menerapkan mask/topeng pada gambar untuk memilih area tertentu.
 Konsep  : Mask = gambar biner 0/255 (hitam/putih).
           Area putih → piksel ditampilkan, area hitam → piksel disembunyikan.
           cv2.bitwise_and(src, src, mask=mask) untuk menerapkan mask.
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


def buat_mask_persegi(h, w, x1, y1, x2, y2):
    """Membuat mask persegi panjang (area putih di dalam kotak)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    print(f"  Mask persegi: ({x1},{y1}) → ({x2},{y2})")
    return mask


def buat_mask_lingkaran(h, w, cx, cy, radius):
    """Membuat mask lingkaran dengan pusat (cx,cy) dan jari-jari radius."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    print(f"  Mask lingkaran: pusat=({cx},{cy}), r={radius}")
    return mask


def buat_mask_poligon(h, w, points):
    """
    Membuat mask berbentuk poligon.
    points = array titik-titik sudut.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    print(f"  Mask poligon: {len(points)} titik sudut")
    return mask


def terapkan_mask(img, mask):
    """
    cv2.bitwise_and(src1, src2, mask=mask)
    Menerapkan mask ke gambar. Piksel di luar mask menjadi hitam.
    """
    return cv2.bitwise_and(img, img, mask=mask)


def mask_invert(mask):
    """Membalik mask: area putih → hitam, hitam → putih."""
    return cv2.bitwise_not(mask)


def tampilkan_hasil(img, masks, hasil, labels):
    """Visualisasi mask dan hasilnya."""
    n = len(masks)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 12))

    for i in range(n):
        # Baris 1: Gambar asli
        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        # Baris 2: Mask
        axes[1, i].imshow(masks[i], cmap='gray')
        axes[1, i].set_title(f"Mask: {labels[i]}")
        axes[1, i].axis("off")

        # Baris 3: Hasil
        axes[2, i].imshow(cv2.cvtColor(hasil[i], cv2.COLOR_BGR2RGB))
        axes[2, i].set_title(f"Hasil: {labels[i]}")
        axes[2, i].axis("off")

    plt.suptitle("Percobaan 19 — Masking Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "19_masking_gambar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 19: MASKING GAMBAR")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "foto_orang2.jpg"))
    h, w = img.shape[:2]
    print(f"\n  Ukuran gambar: {w}x{h}")

    masks, hasil, labels = [], [], []

    # 1) Mask persegi
    print("\n[1] Mask Persegi Panjang:")
    m1 = buat_mask_persegi(h, w, w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    r1 = terapkan_mask(img, m1)
    masks.append(m1); hasil.append(r1); labels.append("Persegi")

    # 2) Mask lingkaran
    print("\n[2] Mask Lingkaran:")
    radius = min(h, w) // 3
    m2 = buat_mask_lingkaran(h, w, w // 2, h // 2, radius)
    r2 = terapkan_mask(img, m2)
    masks.append(m2); hasil.append(r2); labels.append("Lingkaran")

    # 3) Mask poligon (segitiga)
    print("\n[3] Mask Poligon (Segitiga):")
    pts = [(w // 2, h // 6), (w // 6, 5 * h // 6), (5 * w // 6, 5 * h // 6)]
    m3 = buat_mask_poligon(h, w, pts)
    r3 = terapkan_mask(img, m3)
    masks.append(m3); hasil.append(r3); labels.append("Poligon")

    # 4) Mask invert
    print("\n[4] Mask Invert (dari lingkaran):")
    m4 = mask_invert(m2)
    r4 = terapkan_mask(img, m4)
    masks.append(m4); hasil.append(r4); labels.append("Invert")

    tampilkan_hasil(img, masks, hasil, labels)

    print("\nRINGKASAN:")
    print("  Mask = gambar biner (0=hitam, 255=putih)")
    print("  Putih → area ditampilkan, hitam → disembunyikan")
    print("  bitwise_and(src, src, mask=mask) → menerapkan mask")
    print("  bitwise_not(mask) → membalik mask")


if __name__ == "__main__":
    main()
