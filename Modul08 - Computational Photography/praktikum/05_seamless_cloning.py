"""
==========================================================================
PERCOBAAN 05: SEAMLESS CLONING (POISSON BLENDING)
==========================================================================
Program ini mempelajari seamless cloning (poisson blending).
Praktikum 05 - Seamless Cloning (Poisson Blending)
Modul 08: Computational Photography

Topik: cv2.seamlessClone(), NORMAL_CLONE, MIXED_CLONE, MONOCHROME_TRANSFER
Referensi: Mastering OpenCV 4 with Python Ch.12 (Fernández Villán),
           Learning OpenCV (Bradski & Kaehler), OpenCV Tutorial

Hasil: Visualisasi dan analisis disimpan ke folder output/
==========================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)



def demo_normal_clone():
    """NORMAL_CLONE: tempel objek ke background dengan seamless blending."""
    # Background: gradasi langit
    dst = np.zeros((400, 600, 3), dtype=np.uint8)
    for y in range(400):
        ratio = y / 400
        dst[y, :] = [int(255 * (1 - ratio * 0.6)), int(200 * (1 - ratio * 0.4)), int(100 * ratio + 200)]

    # Source: objek lingkaran (simulasi bulan/objek bulat)
    src = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(src, (300, 200), 80, (220, 210, 180), -1)
    cv2.circle(src, (270, 170), 20, (200, 195, 160), -1)   # kawah
    cv2.circle(src, (320, 220), 15, (210, 205, 170), -1)   # kawah

    # Mask: area objek yang akan dipindahkan
    mask = np.zeros(src.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (300, 200), 82, 255, -1)

    # Center: tempat objek akan ditempatkan di dst
    center = (300, 200)

    # Naive copy-paste biasa
    naive = dst.copy()
    roi = src[118:282, 218:382]
    naive[118:282, 218:382] = roi

    # Seamless clone
    output_normal = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1); plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)); plt.title("Source"); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)); plt.title("Destination"); plt.axis('off')
    plt.subplot(1, 4, 3); plt.imshow(cv2.cvtColor(naive, cv2.COLOR_BGR2RGB)); plt.title("Naive Copy-Paste\n(tepi terlihat)"); plt.axis('off')
    plt.subplot(1, 4, 4); plt.imshow(cv2.cvtColor(output_normal, cv2.COLOR_BGR2RGB)); plt.title("Seamless Clone\n(NORMAL_CLONE)"); plt.axis('off')
    plt.suptitle("NORMAL_CLONE vs Naive Copy-Paste")
    plt.tight_layout(); plt.savefig("output_05_seamless_normal.png", dpi=100); plt.show()
    print("[OK] NORMAL_CLONE berhasil — tepi objek menyatu dengan background")


def demo_mixed_clone():
    """MIXED_CLONE: mempertahankan tekstur background di area overlap."""
    # Background dengan tekstur menarik
    dst = np.zeros((400, 600, 3), dtype=np.uint8)
    # Pattern checkerboard di background
    for y in range(400):
        for x in range(600):
            if ((x // 40) + (y // 40)) % 2 == 0:
                dst[y, x] = [200, 200, 200]
            else:
                dst[y, x] = [100, 100, 100]

    # Source: lingkaran solid berwarna
    src = dst.copy()
    cv2.circle(src, (300, 200), 100, (0, 100, 220), -1)
    cv2.circle(src, (300, 200), 60, (0, 150, 255), -1)

    mask = np.zeros(src.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (300, 200), 102, 255, -1)
    center = (300, 200)

    out_normal = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    out_mixed = cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)); plt.title("Background (Checkerboard)"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(out_normal, cv2.COLOR_BGR2RGB)); plt.title("NORMAL_CLONE\n(tekstur terhapus)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(out_mixed, cv2.COLOR_BGR2RGB)); plt.title("MIXED_CLONE\n(tekstur background dipertahankan)"); plt.axis('off')
    plt.suptitle("NORMAL_CLONE vs MIXED_CLONE")
    plt.tight_layout(); plt.savefig("output_05_mixed_clone.png", dpi=100); plt.show()
    print("[OK] MIXED_CLONE mempertahankan tekstur background di area clone")


def demo_monochrome_transfer():
    """MONOCHROME_TRANSFER: pindahkan tekstur dari grayscale source ke warna tujuan."""
    # Source: gambar grayscale dengan tekstur (simulasi lukisan hitam putih)
    src = np.zeros((400, 600, 3), dtype=np.uint8)
    # Buat pola tekstur di source
    for y in range(400):
        for x in range(600):
            val = int(128 + 127 * np.sin(x * 0.1) * np.cos(y * 0.1))
            src[y, x] = [val, val, val]
    cv2.circle(src, (300, 200), 120, (200, 200, 200), -1)
    cv2.circle(src, (300, 200), 80, (150, 150, 150), -1)
    cv2.circle(src, (300, 200), 40, (100, 100, 100), -1)

    # Background: gradasi warna
    dst = np.zeros((400, 600, 3), dtype=np.uint8)
    for x in range(600):
        dst[:, x] = [int(x * 255 / 600), int((600 - x) * 200 / 600), 100]

    mask = np.zeros(src.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (300, 200), 122, 255, -1)
    center = (300, 200)

    out = cv2.seamlessClone(src, dst, mask, center, cv2.MONOCHROME_TRANSFER)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)); plt.title("Source (Tekstur)"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)); plt.title("Destination (Warna)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); plt.title("MONOCHROME_TRANSFER\n(tekstur + warna dst)"); plt.axis('off')
    plt.suptitle("MONOCHROME_TRANSFER: Tekstur dari Source + Warna dari Destination")
    plt.tight_layout(); plt.savefig("output_05_monochrome_transfer.png", dpi=100); plt.show()


def demo_face_swap_concept():
    """Konsep face swap sederhana menggunakan seamless cloning."""
    # Simulasi: dua wajah oval
    face1 = np.ones((300, 300, 3), dtype=np.uint8) * [200, 170, 150]
    face2 = np.ones((300, 300, 3), dtype=np.uint8) * [160, 130, 110]

    # Face 1: fitur lebih terang
    cv2.ellipse(face1, (150, 130), (90, 110), 0, 0, 360, (220, 190, 170), -1)
    cv2.ellipse(face1, (120, 110), (15, 10), 0, 0, 360, (50, 40, 80), -1)   # mata kiri
    cv2.ellipse(face1, (180, 110), (15, 10), 0, 0, 360, (50, 40, 80), -1)   # mata kanan
    cv2.ellipse(face1, (150, 160), (30, 8), 0, 0, 180, (150, 80, 80), 2)    # mulut

    # Face 2: fitur lebih gelap
    cv2.ellipse(face2, (150, 130), (90, 110), 0, 0, 360, (180, 150, 130), -1)
    cv2.ellipse(face2, (115, 115), (18, 12), 0, 0, 360, (30, 25, 60), -1)
    cv2.ellipse(face2, (185, 115), (18, 12), 0, 0, 360, (30, 25, 60), -1)
    cv2.ellipse(face2, (150, 165), (35, 10), 0, 0, 180, (120, 60, 60), 2)

    # Swap fitur face1 ke background face2
    mask = np.zeros(face1.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (150, 130), (85, 105), 0, 0, 360, 255, -1)
    center = (150, 130)

    swapped = cv2.seamlessClone(face1, face2, mask, center, cv2.NORMAL_CLONE)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)); plt.title("Wajah 1 (Source)"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)); plt.title("Wajah 2 (Destination)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)); plt.title("Face Swap (Seamless)"); plt.axis('off')
    plt.suptitle("Konsep Face Swap dengan Seamless Cloning")
    plt.tight_layout(); plt.savefig("output_05_face_swap.png", dpi=100); plt.show()
    print("[OK] Face swap konseptual berhasil — warna kulit menyatu secara natural")


if __name__ == "__main__":
    print("=" * 55)
    print("PRAKTIKUM 05: SEAMLESS CLONING (POISSON BLENDING)")
    print("=" * 55)

    print("\n[1] NORMAL_CLONE — Tempel Objek ke Background")
    demo_normal_clone()

    print("\n[2] MIXED_CLONE — Pertahankan Tekstur Background")
    demo_mixed_clone()

    print("\n[3] MONOCHROME_TRANSFER — Transfer Tekstur")
    demo_monochrome_transfer()

    print("\n[4] Konsep Face Swap dengan Seamless Cloning")
    demo_face_swap_concept()

    print("\n[SELESAI] Semua demo seamless cloning berhasil dijalankan.")
