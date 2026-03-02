"""
==========================================================================
 PERCOBAAN 5 — OPERASI ARITMATIKA GAMBAR
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Memahami perbedaan operasi aritmatika OpenCV (saturasi)
           vs NumPy (wrap-around/overflow) pada gambar.
 Konsep  : cv2.add(), cv2.subtract(), cv2.addWeighted(), overflow.
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


def demo_saturasi_vs_overflow():
    """
    Menunjukkan perbedaan KRITIS antara:
    - cv2.add()  → saturasi: 200 + 100 = 255 (dipotong di 255)
    - numpy +    → overflow: 200 + 100 = 44  (wrap 300 % 256)
    """
    a = np.array([[200]], dtype=np.uint8)
    b = np.array([[100]], dtype=np.uint8)

    cv_result = cv2.add(a, b)[0, 0]       # 255 (saturasi)
    np_result = (a + b)[0, 0]              # 44  (overflow)

    print(f"  200 + 100 via cv2.add()  = {cv_result}  (saturasi)")
    print(f"  200 + 100 via numpy +    = {np_result}  (overflow/wrap)")
    return cv_result, np_result


def penjumlahan_gambar(img1, img2):
    """
    cv2.add(img1, img2) → penjumlahan dengan saturasi per piksel.
    Kedua gambar harus ukuran dan tipe sama.
    """
    # Pastikan ukuran sama
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    a = img1[:h, :w]
    b = img2[:h, :w]

    hasil = cv2.add(a, b)
    print(f"  cv2.add: shape={hasil.shape}")
    return a, b, hasil


def pengurangan_gambar(img1, img2):
    """cv2.subtract(img1, img2) → pengurangan dengan saturasi (min 0)."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    a, b = img1[:h, :w], img2[:h, :w]
    hasil = cv2.subtract(a, b)
    print(f"  cv2.subtract: shape={hasil.shape}")
    return hasil


def blending_tertimbang(img1, img2, alpha=0.7):
    """
    cv2.addWeighted(img1, alpha, img2, beta, gamma)
    → dst = alpha*img1 + beta*img2 + gamma
    Berguna untuk blending/overlay dua gambar.
    """
    beta = 1.0 - alpha
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    a, b = img1[:h, :w], img2[:h, :w]
    hasil = cv2.addWeighted(a, alpha, b, beta, 0)
    print(f"  addWeighted: alpha={alpha}, beta={beta:.1f}")
    return hasil


def tampilkan_hasil(img1, img2, tambah, kurang, blend):
    """Visualisasi semua operasi aritmatika."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Gambar 1"); axes[0, 0].axis("off")
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Gambar 2"); axes[0, 1].axis("off")
    axes[0, 2].imshow(cv2.cvtColor(tambah, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("cv2.add (Saturasi)"); axes[0, 2].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(kurang, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("cv2.subtract"); axes[1, 0].axis("off")
    axes[1, 1].imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("addWeighted (α=0.7)"); axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    plt.suptitle("Percobaan 5 — Operasi Aritmatika Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "05_operasi_aritmatika.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 5: OPERASI ARITMATIKA GAMBAR")
    print("=" * 60)

    print("\n[1] Saturasi (OpenCV) vs Overflow (NumPy):")
    demo_saturasi_vs_overflow()

    img1 = cv2.imread(os.path.join(IMAGE_DIR, "foto_bunga.jpg"))
    img2 = cv2.imread(os.path.join(IMAGE_DIR, "foto_hewan.jpg"))

    print("\n[2] Penjumlahan gambar:")
    a, b, tambah = penjumlahan_gambar(img1, img2)

    print("\n[3] Pengurangan gambar:")
    kurang = pengurangan_gambar(img1, img2)

    print("\n[4] Blending tertimbang:")
    blend = blending_tertimbang(img1, img2, alpha=0.7)

    tampilkan_hasil(a, b, tambah, kurang, blend)

    print("\nRINGKASAN:")
    print("  cv2.add()          → penjumlahan saturasi (maks 255)")
    print("  cv2.subtract()     → pengurangan saturasi (min 0)")
    print("  cv2.addWeighted()  → blending: α·img1 + β·img2 + γ")


if __name__ == "__main__":
    main()
