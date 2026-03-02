"""
==========================================================================
 PERCOBAAN 3 — KONVERSI RUANG WARNA
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Mengenal berbagai ruang warna (BGR, RGB, Grayscale, HSV, LAB,
           YCrCb, HLS) dan konversinya menggunakan cv2.cvtColor().
 Konsep  : Setiap ruang warna punya kegunaan berbeda — HSV untuk
           segmentasi warna, LAB untuk perceptual distance, dsb.
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


def konversi_ruang_warna(img_bgr):
    """
    Konversi gambar BGR ke beberapa ruang warna sekaligus.
    Return dict nama_ruang_warna → gambar_hasil.
    """
    konversi = {
        "RGB":   cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "GRAY":  cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        "HSV":   cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV),
        "LAB":   cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB),
        "YCrCb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb),
        "HLS":   cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS),
    }
    for nama, res in konversi.items():
        print(f"  {nama:6s} → shape={res.shape}  range=[{res.min()}, {res.max()}]")
    return konversi


def visualisasi_channel(img_bgr, nama_cs, img_cs, channel_names):
    """
    Tampilkan gambar hasil konversi beserta channel-channelnya.
    """
    if img_cs.ndim == 2:
        # Grayscale — hanya 1 channel
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original (RGB)"); axes[0].axis("off")
        axes[1].imshow(img_cs, cmap="gray")
        axes[1].set_title("Grayscale"); axes[1].axis("off")
    else:
        ch = cv2.split(img_cs)
        fig, axes = plt.subplots(1, len(ch) + 1, figsize=(4 * (len(ch) + 1), 4))
        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original (RGB)"); axes[0].axis("off")
        for i, (c, name) in enumerate(zip(ch, channel_names)):
            axes[i + 1].imshow(c, cmap="gray")
            axes[i + 1].set_title(f"{nama_cs} — {name}")
            axes[i + 1].axis("off")

    plt.suptitle(f"Percobaan 3 — Ruang Warna {nama_cs}", fontweight="bold")
    plt.tight_layout()
    return fig


def simpan_semua(img_bgr, hasil):
    """Simpan visualisasi untuk beberapa ruang warna utama."""
    configs = {
        "HSV":   ["H (Hue)", "S (Saturation)", "V (Value)"],
        "LAB":   ["L (Lightness)", "A (Green-Red)", "B (Blue-Yellow)"],
        "YCrCb": ["Y (Luma)", "Cr (Red-diff)", "Cb (Blue-diff)"],
        "HLS":   ["H (Hue)", "L (Lightness)", "S (Saturation)"],
    }
    figs = []
    for nama, ch_names in configs.items():
        fig = visualisasi_channel(img_bgr, nama, hasil[nama], ch_names)
        figs.append((nama, fig))

    # Gabung semua ke satu figure besar
    fig_all, axes = plt.subplots(2, 3, figsize=(15, 10))
    items = [("RGB", hasil["RGB"]), ("GRAY", hasil["GRAY"]),
             ("HSV", hasil["HSV"]), ("LAB", hasil["LAB"]),
             ("YCrCb", hasil["YCrCb"]), ("HLS", hasil["HLS"])]
    for ax, (nama, img) in zip(axes.flatten(), items):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        elif nama == "RGB":
            ax.imshow(img)
        else:
            ax.imshow(img[:, :, 0], cmap="gray")  # channel pertama saja
        ax.set_title(nama); ax.axis("off")

    plt.suptitle("Percobaan 3 — Semua Ruang Warna", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "03_konversi_ruang_warna.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")

    for nama, fig in figs:
        plt.close(fig)


def main():
    print("=" * 60)
    print(" PERCOBAAN 3: KONVERSI RUANG WARNA")
    print("=" * 60)

    path = os.path.join(IMAGE_DIR, "foto_bunga2.jpg")
    img_bgr = cv2.imread(path)

    hasil = konversi_ruang_warna(img_bgr)
    simpan_semua(img_bgr, hasil)

    print("\nRINGKASAN:")
    print("  cv2.cvtColor(img, code)  → konversi ruang warna")
    print("  BGR→RGB, BGR→GRAY, BGR→HSV, BGR→LAB, BGR→YCrCb, BGR→HLS")
    print("  HSV: deteksi warna  |  LAB: jarak perseptual  |  YCrCb: video")


if __name__ == "__main__":
    main()
