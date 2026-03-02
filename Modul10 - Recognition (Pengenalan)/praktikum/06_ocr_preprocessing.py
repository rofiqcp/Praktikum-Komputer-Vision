"""
==========================================================================
PERCOBAAN 6: OCR PREPROCESSING
==========================================================================
Program ini mendemonstrasikan teknik preprocessing gambar
untuk meningkatkan akurasi OCR (Optical Character Recognition).

Konsep: binarisasi, denoising, deskew, resize untuk OCR.

Referensi: Mastering OpenCV 4, Tesseract docs
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
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
        print(f"  [WARN] Gambar {nama_file} tidak ditemukan.")
    return img


def binarisasi_adaptif(gray):
    """
    Binarisasi adaptif: threshold yang berbeda per region.
    Cocok untuk dokumen dengan pencahayaan tidak merata.
    """
    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Adaptive mean
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    # Adaptive Gaussian
    gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return otsu, adaptive, gauss


def denoise_untuk_ocr(gray):
    """Menghilangkan noise pada gambar teks."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = cv2.medianBlur(gray, 5)
    nlm = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return blur, median, nlm


def deskew(gray):
    """
    Memperbaiki kemiringan teks menggunakan momen gambar.
    Menghitung sudut skew dari momen → rotasi balik.
    """
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 10:
        return gray, 0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def visualisasi_preprocessing(gray):
    """Menampilkan semua tahap preprocessing OCR."""
    otsu, adaptive, gauss = binarisasi_adaptif(gray)
    _, median, nlm = denoise_untuk_ocr(gray)
    deskewed, angle = deskew(otsu)

    images = [gray, otsu, adaptive, gauss, median, nlm]
    titles = ["Asli", "Otsu", "Adaptive Mean", "Adaptive Gaussian",
              "Median Blur", "NLM Denoise"]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.suptitle("Preprocessing OCR: Binarisasi & Denoising", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_ocr_preprocessing.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/06_ocr_preprocessing.png")


def main():
    """Fungsi utama: OCR preprocessing."""
    print("=" * 60)
    print("PERCOBAAN 6: OCR PREPROCESSING")
    print("=" * 60)

    img = load_gambar("teks_printed.jpg")
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.ones((200, 400), dtype=np.uint8) * 240
        cv2.putText(gray, "Hello World OCR", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, 30, 2)
        cv2.putText(gray, "Preprocessing Test", (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, 50, 2)
        noise = np.random.normal(0, 15, gray.shape)
        gray = np.clip(gray.astype(float) + noise, 0, 255).astype(np.uint8)

    print("\n--- 1. Binarisasi ---")
    otsu, adaptive, gauss = binarisasi_adaptif(gray)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "06_otsu.png"), otsu)
    print("  [SAVED] output/06_otsu.png")

    print("\n--- 2. Denoising ---")
    _, median, nlm = denoise_untuk_ocr(gray)

    print("\n--- 3. Deskew ---")
    deskewed, angle = deskew(otsu)
    print(f"  Sudut kemiringan: {angle:.2f} derajat")

    print("\n--- 4. Visualisasi ---")
    visualisasi_preprocessing(gray)

    cv2.imshow("OCR Preprocessing", np.hstack([gray, otsu]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nRINGKASAN: Preprocessing yang baik (binarisasi, denoise,")
    print("deskew) sangat meningkatkan akurasi OCR.")


if __name__ == "__main__":
    main()
