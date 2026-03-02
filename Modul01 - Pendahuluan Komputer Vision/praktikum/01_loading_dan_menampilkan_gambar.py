"""
==========================================================================
 PERCOBAAN 1 — LOADING DAN MENAMPILKAN GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Memahami cara memuat gambar dari file menggunakan OpenCV,
           menampilkannya, serta menyimpan hasil ke folder output.
 Konsep  : cv2.imread() dengan tiga mode flag, cv2.cvtColor(), matplotlib.
 Catatan : OpenCV membaca gambar dalam format BGR (bukan RGB).
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# --- Konfigurasi path ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_gambar_berwarna(path_gambar: str):
    """
    Membaca gambar dalam mode WARNA (3 channel BGR).
    IMREAD_COLOR (flag=1) mengabaikan alpha channel.
    Return: array NumPy (H, W, 3) uint8.
    """
    img = cv2.imread(path_gambar, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Gagal membaca: {path_gambar}")
    print(f"[COLOR]     Shape={img.shape}  Dtype={img.dtype}")
    return img


def load_gambar_grayscale(path_gambar: str):
    """
    Membaca gambar dalam mode GRAYSCALE (1 channel abu-abu).
    IMREAD_GRAYSCALE (flag=0) langsung konversi.
    Return: array 2-D (H, W).
    """
    img = cv2.imread(path_gambar, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gagal membaca: {path_gambar}")
    print(f"[GRAY]      Shape={img.shape}  Dtype={img.dtype}")
    return img


def load_gambar_unchanged(path_gambar: str):
    """
    Membaca gambar APA ADANYA termasuk alpha channel (jika ada).
    IMREAD_UNCHANGED (flag=-1) berguna untuk PNG transparan.
    """
    img = cv2.imread(path_gambar, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Gagal membaca: {path_gambar}")
    print(f"[UNCHANGED] Shape={img.shape}  Dtype={img.dtype}")
    return img


def tampilkan_dan_simpan(img_color, img_gray, img_unchanged):
    """Visualisasi tiga mode pembacaan, simpan ke output/."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Mode COLOR (BGR→RGB)")
    axes[0].axis("off")

    axes[1].imshow(img_gray, cmap="gray")
    axes[1].set_title("Mode GRAYSCALE")
    axes[1].axis("off")

    if img_unchanged.ndim == 3 and img_unchanged.shape[2] == 4:
        axes[2].imshow(cv2.cvtColor(img_unchanged, cv2.COLOR_BGRA2RGBA))
    else:
        axes[2].imshow(cv2.cvtColor(img_unchanged, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Mode UNCHANGED")
    axes[2].axis("off")

    plt.suptitle("Percobaan 1 — Tiga Mode Pembacaan Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "01_loading_gambar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 1: LOADING DAN MENAMPILKAN GAMBAR")
    print("=" * 60)

    path = os.path.join(IMAGE_DIR, "foto_kucing.jpg")
    img_color     = load_gambar_berwarna(path)
    img_gray      = load_gambar_grayscale(path)
    img_unchanged = load_gambar_unchanged(path)
    tampilkan_dan_simpan(img_color, img_gray, img_unchanged)

    print("\nRINGKASAN:")
    print("  cv2.imread(path, flag)  → baca gambar")
    print("    IMREAD_COLOR=1  |  IMREAD_GRAYSCALE=0  |  IMREAD_UNCHANGED=-1")
    print("  cv2.cvtColor()  → konversi BGR↔RGB")


if __name__ == "__main__":
    main()
