"""
==========================================================================
PERCOBAAN 7: DATA AUGMENTASI DASAR
==========================================================================
Program ini mendemonstrasikan teknik data augmentasi untuk memperbanyak
data training secara artifisial. Augmentasi penting agar model belajar
fitur yang invariant terhadap transformasi geometri dan fotometri.

Teknik yang dipelajari:
- Geometric: flip, rotasi, crop, resize, translate
- Photometric: brightness, contrast, noise, blur
- Kombinasi augmentasi secara random

Referensi: Deep Learning for CV (Rosebrock), Géron Ch.10
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_gambar(nama_file):
    """Memuat gambar dari folder image/."""
    path = os.path.join(IMAGE_DIR, nama_file)
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] {nama_file} tidak ditemukan!")
    return img


def augmentasi_flip(img):
    """Flip gambar secara horizontal, vertikal, dan keduanya."""
    flip_h = cv2.flip(img, 1)      # Horizontal
    flip_v = cv2.flip(img, 0)      # Vertikal
    flip_hv = cv2.flip(img, -1)    # Kedua arah
    return {"Horizontal Flip": flip_h, "Vertical Flip": flip_v, "Both Flip": flip_hv}


def augmentasi_rotasi(img, sudut_list=[15, 45, 90, 180]):
    """Rotasi gambar pada berbagai sudut (derajat)."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    hasil = {}
    for sudut in sudut_list:
        M = cv2.getRotationMatrix2D(center, sudut, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        hasil[f"Rotasi {sudut}°"] = rotated
    return hasil


def augmentasi_brightness_contrast(img):
    """Mengubah brightness dan contrast gambar."""
    hasil = {}
    # Brightness
    bright = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
    dark = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)
    hasil["Terang (+50)"] = bright
    hasil["Gelap (-50)"] = dark
    # Contrast
    high_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    low_contrast = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
    hasil["High Contrast"] = high_contrast
    hasil["Low Contrast"] = low_contrast
    return hasil


def augmentasi_noise(img):
    """Menambahkan noise Gaussian dan Salt-Pepper pada gambar."""
    hasil = {}
    # Gaussian noise
    noise = np.random.normal(0, 25, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    hasil["Gaussian Noise"] = noisy

    # Salt & Pepper noise
    sp = img.copy()
    n_salt = int(0.02 * img.size / 3)
    coords = [np.random.randint(0, i, n_salt) for i in img.shape[:2]]
    sp[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, n_salt) for i in img.shape[:2]]
    sp[coords[0], coords[1]] = 0
    hasil["Salt & Pepper"] = sp

    # Blur
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    hasil["Gaussian Blur"] = blurred
    return hasil


def augmentasi_crop_random(img, n_crops=4):
    """Mengambil random crop dari gambar asli."""
    h, w = img.shape[:2]
    crop_h, crop_w = int(h * 0.7), int(w * 0.7)
    hasil = {}
    for i in range(n_crops):
        y = np.random.randint(0, h - crop_h)
        x = np.random.randint(0, w - crop_w)
        crop = img[y:y + crop_h, x:x + crop_w]
        crop = cv2.resize(crop, (w, h))
        hasil[f"Crop {i + 1}"] = crop
    return hasil


def visualisasi_augmentasi(img, semua_hasil, judul, nama_output):
    """Menampilkan semua hasil augmentasi dalam satu grid."""
    n = len(semua_hasil) + 1
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Asli")
    axes[0, 0].axis('off')

    for i, (nama, aug_img) in enumerate(semua_hasil.items()):
        r, c = (i + 1) // cols, (i + 1) % cols
        axes[r, c].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        axes[r, c].set_title(nama, fontsize=9)
        axes[r, c].axis('off')

    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle(judul, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, nama_output), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] output/{nama_output}")


def main():
    """Fungsi utama: demonstrasi data augmentasi."""
    print("=" * 60)
    print("PERCOBAAN 7: DATA AUGMENTASI DASAR")
    print("=" * 60)

    img = load_gambar("kucing.jpg")
    if img is None:
        return
    img = cv2.resize(img, (224, 224))

    print("\n--- 1. Augmentasi Flip ---")
    flip_hasil = augmentasi_flip(img)
    visualisasi_augmentasi(img, flip_hasil, "Augmentasi: Flip", "07_augmentasi_flip.png")

    print("\n--- 2. Augmentasi Rotasi ---")
    rot_hasil = augmentasi_rotasi(img)
    visualisasi_augmentasi(img, rot_hasil, "Augmentasi: Rotasi", "07_augmentasi_rotasi.png")

    print("\n--- 3. Augmentasi Brightness/Contrast ---")
    bc_hasil = augmentasi_brightness_contrast(img)
    visualisasi_augmentasi(img, bc_hasil, "Augmentasi: Brightness/Contrast", "07_augmentasi_bc.png")

    print("\n--- 4. Augmentasi Noise ---")
    noise_hasil = augmentasi_noise(img)
    visualisasi_augmentasi(img, noise_hasil, "Augmentasi: Noise", "07_augmentasi_noise.png")

    print("\n--- 5. Random Crop ---")
    crop_hasil = augmentasi_crop_random(img)
    visualisasi_augmentasi(img, crop_hasil, "Augmentasi: Random Crop", "07_augmentasi_crop.png")

    cv2.imshow("Gambar Asli - Percobaan 7", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 7")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. Augmentasi geometri (flip, rotasi, crop) mengubah posisi/orientasi
2. Augmentasi fotometri (brightness, contrast, noise) mengubah tampilan
3. Augmentasi meningkatkan variasi data training
4. Model menjadi lebih robust terhadap variasi di dunia nyata
5. Kombinasi augmentasi memberikan hasil terbaik

Output: output/07_augmentasi_flip.png, output/07_augmentasi_rotasi.png,
        output/07_augmentasi_bc.png, output/07_augmentasi_noise.png,
        output/07_augmentasi_crop.png
""")


if __name__ == "__main__":
    main()
