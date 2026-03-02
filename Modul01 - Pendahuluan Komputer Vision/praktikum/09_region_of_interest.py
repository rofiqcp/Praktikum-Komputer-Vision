"""
==========================================================================
 PERCOBAAN 9 — REGION OF INTEREST (ROI)
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Memahami cara memilih dan memanipulasi sub-area (ROI) dari
           gambar menggunakan slicing NumPy.
 Konsep  : roi = img[y1:y2, x1:x2], copy ROI, paste ROI, visualisasi.
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


def ekstrak_roi(img, y1, y2, x1, x2):
    """
    Mengekstrak Region of Interest dari gambar.
    ROI = img[y1:y2, x1:x2] → mengambil area dari baris y1..y2-1,
    kolom x1..x2-1.
    PENTING: selalu gunakan .copy() agar ROI independen.
    """
    roi = img[y1:y2, x1:x2].copy()
    print(f"  ROI diekstrak: [{y1}:{y2}, {x1}:{x2}] → shape={roi.shape}")
    return roi


def tempel_roi(img, roi, y, x):
    """Menempelkan ROI ke posisi (y, x) pada gambar."""
    hasil = img.copy()
    rh, rw = roi.shape[:2]
    hasil[y:y + rh, x:x + rw] = roi
    print(f"  ROI ditempelkan di posisi ({y}, {x})")
    return hasil


def highlight_roi(img, y1, y2, x1, x2, warna=(0, 255, 0)):
    """Menggambar kotak di sekeliling area ROI untuk visualisasi."""
    hasil = img.copy()
    cv2.rectangle(hasil, (x1, y1), (x2, y2), warna, 3)
    cv2.putText(hasil, "ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, warna, 2)
    return hasil


def tampilkan_hasil(img_ori, img_highlight, roi, img_paste):
    """Visualisasi proses ROI."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Gambar Original"); axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(img_highlight, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Area ROI (hijau)"); axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"ROI Diekstrak ({roi.shape[1]}x{roi.shape[0]})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(img_paste, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("ROI Ditempelkan"); axes[1, 1].axis("off")

    plt.suptitle("Percobaan 9 — Region of Interest (ROI)", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "09_region_of_interest.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 9: REGION OF INTEREST (ROI)")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "foto_pasar.jpg"))
    h, w = img.shape[:2]

    # Definisikan area ROI (1/4 tengah gambar)
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4

    print("\n[1] Ekstrak ROI:")
    roi = ekstrak_roi(img, y1, y2, x1, x2)

    print("\n[2] Highlight ROI:")
    img_highlight = highlight_roi(img, y1, y2, x1, x2)

    print("\n[3] Tempel ROI di pojok kiri atas:")
    roi_kecil = cv2.resize(roi, (w // 4, h // 4))
    img_paste = tempel_roi(img, roi_kecil, 10, 10)

    tampilkan_hasil(img, img_highlight, roi, img_paste)

    print("\nRINGKASAN:")
    print("  roi = img[y1:y2, x1:x2]  → ekstrak area")
    print("  img[y:y+h, x:x+w] = roi  → tempel area")
    print("  Selalu .copy() agar independen dari original")


if __name__ == "__main__":
    main()
