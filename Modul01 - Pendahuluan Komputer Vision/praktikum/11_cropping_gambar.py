"""
==========================================================================
 PERCOBAAN 11 — CROPPING GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Memotong (crop) area tertentu dari gambar dan menyimpannya.
 Konsep  : Cropping = slicing NumPy: img[y1:y2, x1:x2].
           Center crop, aspect-ratio crop, multi-crop grid.
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


def crop_manual(img, y1, y2, x1, x2):
    """Crop area [y1:y2, x1:x2] dari gambar."""
    crop = img[y1:y2, x1:x2].copy()
    print(f"  Crop [{y1}:{y2}, {x1}:{x2}] → {crop.shape}")
    return crop


def crop_tengah(img, crop_w, crop_h):
    """
    Center crop: potong area tengah gambar sesuai ukuran (crop_w, crop_h).
    Berguna untuk menstandardkan ukuran input klasifikasi.
    """
    h, w = img.shape[:2]
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    crop = img[y1:y1 + crop_h, x1:x1 + crop_w].copy()
    print(f"  Center crop {crop_w}x{crop_h} → {crop.shape}")
    return crop


def crop_grid(img, rows=2, cols=3):
    """
    Membagi gambar menjadi grid rows x cols dan mengkrop setiap sel.
    Berguna untuk analisis bagian-bagian gambar secara terpisah.
    """
    h, w = img.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    crops = []
    for r in range(rows):
        for c in range(cols):
            y1, x1 = r * cell_h, c * cell_w
            crop = img[y1:y1 + cell_h, x1:x1 + cell_w].copy()
            crops.append(crop)
    print(f"  Grid {rows}x{cols} → {len(crops)} potongan, ukuran {cell_w}x{cell_h}")
    return crops


def tampilkan_hasil(img, crop_manual_img, crop_center, crops_grid):
    """Visualisasi semua hasil crop."""
    fig = plt.figure(figsize=(16, 10))

    # Original + manual crop
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original"); ax1.axis("off")

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(cv2.cvtColor(crop_manual_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Crop Manual"); ax2.axis("off")

    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(cv2.cvtColor(crop_center, cv2.COLOR_BGR2RGB))
    ax3.set_title("Center Crop"); ax3.axis("off")

    # Grid crops
    for i, crop in enumerate(crops_grid[:5]):
        ax = fig.add_subplot(2, 4, 4 + i + 1)
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Grid [{i}]"); ax.axis("off")

    plt.suptitle("Percobaan 11 — Cropping Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "11_cropping_gambar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 11: CROPPING GAMBAR")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "foto_alam2.jpg"))
    h, w = img.shape[:2]

    print("\n[1] Crop manual (kiri atas, 1/3 gambar):")
    crop_m = crop_manual(img, 0, h // 2, 0, w // 2)

    print("\n[2] Center crop (250x250):")
    crop_c = crop_tengah(img, 250, 250)

    print("\n[3] Grid crop (2x3):")
    crops_g = crop_grid(img, 2, 3)

    tampilkan_hasil(img, crop_m, crop_c, crops_g)

    print("\nRINGKASAN:")
    print("  img[y1:y2, x1:x2] → crop area manapun")
    print("  Center crop → standarisasi input CNN")
    print("  Grid crop   → analisis per-region")


if __name__ == "__main__":
    main()
