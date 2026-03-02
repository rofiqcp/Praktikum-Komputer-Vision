"""
==========================================================================
 PERCOBAAN 7 — MENGGAMBAR BENTUK GEOMETRIS
 Modul 1: Pendahuluan Komputer Vision

 Tujuan  : Menggambar garis, kotak, lingkaran, elips, poligon, dan
           panah pada gambar menggunakan primitive drawing OpenCV.
 Konsep  : cv2.line(), rectangle(), circle(), ellipse(), polylines(),
           arrowedLine(), fillPoly().
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def buat_kanvas(h=500, w=700, warna=(40, 40, 40)):
    """Membuat kanvas kosong berwarna gelap untuk menggambar."""
    return np.full((h, w, 3), warna, dtype=np.uint8)


def gambar_garis(canvas):
    """cv2.line(img, pt1, pt2, color, thickness, lineType)"""
    cv2.line(canvas, (50, 50), (300, 50), (0, 255, 0), 2)
    cv2.line(canvas, (50, 80), (300, 80), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.arrowedLine(canvas, (50, 110), (300, 110), (255, 0, 255), 2)
    print("  Garis, garis anti-aliased, dan panah digambar.")
    return canvas


def gambar_kotak(canvas):
    """cv2.rectangle(img, pt1, pt2, color, thickness) — -1 = filled"""
    cv2.rectangle(canvas, (350, 30), (500, 130), (0, 0, 255), 2)
    cv2.rectangle(canvas, (520, 30), (670, 130), (255, 100, 0), -1)
    print("  Kotak outline dan kotak filled digambar.")
    return canvas


def gambar_lingkaran(canvas):
    """cv2.circle(img, center, radius, color, thickness)"""
    cv2.circle(canvas, (100, 250), 60, (255, 255, 0), 2)
    cv2.circle(canvas, (250, 250), 50, (0, 180, 255), -1)
    print("  Lingkaran outline dan filled digambar.")
    return canvas


def gambar_elips(canvas):
    """cv2.ellipse(img, center, axes, angle, startAngle, endAngle, ...)"""
    cv2.ellipse(canvas, (450, 250), (100, 50), 0, 0, 360, (200, 200, 0), 2)
    cv2.ellipse(canvas, (450, 250), (100, 50), 30, 0, 270, (0, 200, 200), -1)
    print("  Elips penuh dan parsial (270°) digambar.")
    return canvas


def gambar_poligon(canvas):
    """cv2.polylines() untuk outline, cv2.fillPoly() untuk filled."""
    pts_segitiga = np.array([[100, 380], [200, 480], [50, 480]], np.int32)
    pts_bintang  = np.array([[350, 380], [370, 430], [420, 430],
                             [380, 460], [400, 490], [350, 470],
                             [300, 490], [320, 460], [280, 430],
                             [330, 430]], np.int32)

    cv2.polylines(canvas, [pts_segitiga], True, (255, 200, 100), 2)
    cv2.fillPoly(canvas, [pts_bintang], (100, 100, 255))
    print("  Segitiga outline dan bintang filled digambar.")
    return canvas


def tampilkan_dan_simpan(canvas):
    """Menampilkan dan menyimpan kanvas hasil gambar."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title("Percobaan 7 — Bentuk Geometris di OpenCV", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "07_bentuk_geometris.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 7: MENGGAMBAR BENTUK GEOMETRIS")
    print("=" * 60)

    canvas = buat_kanvas()

    print("\n[1] Garis & Panah:")
    gambar_garis(canvas)
    print("[2] Kotak:")
    gambar_kotak(canvas)
    print("[3] Lingkaran:")
    gambar_lingkaran(canvas)
    print("[4] Elips:")
    gambar_elips(canvas)
    print("[5] Poligon:")
    gambar_poligon(canvas)

    tampilkan_dan_simpan(canvas)

    print("\nRINGKASAN:")
    print("  cv2.line/rectangle/circle/ellipse/polylines/fillPoly()")
    print("  thickness=-1 → filled  |  LINE_AA → anti-aliased")


if __name__ == "__main__":
    main()
