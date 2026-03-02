"""
==========================================================================
PERCOBAAN 7: OCR TESSERACT (KONSEP)
==========================================================================
Program ini mendemonstrasikan konsep OCR menggunakan
Tesseract dan preprocessing OpenCV.

Konsep: pipeline OCR lengkap, preprocessing + recognition.

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


def buat_gambar_teks():
    """Membuat gambar berisi teks untuk demo OCR."""
    img = np.ones((300, 600, 3), dtype=np.uint8) * 245
    texts = [
        (30, 50, "KOMPUTER VISION", 1.0, 2),
        (30, 100, "Praktikum Modul 10", 0.7, 2),
        (30, 150, "Recognition & OCR", 0.7, 1),
        (30, 200, "OpenCV + Tesseract", 0.6, 1),
        (30, 250, "2024 - Indonesia", 0.6, 1),
    ]
    for x, y, text, scale, thick in texts:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                   scale, (20, 20, 20), thick)
    return img


def simulasi_ocr(img):
    """
    Simulasi proses OCR:
    1. Preprocessing (grayscale, threshold)
    2. Deteksi region teks (contour)
    3. Recognition (simulasi jika tesseract tidak tersedia)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Deteksi baris teks via projection profile
    row_sum = np.sum(binary, axis=1)
    in_text = False
    text_regions = []
    for i, val in enumerate(row_sum):
        if val > 0 and not in_text:
            start = i
            in_text = True
        elif val == 0 and in_text:
            text_regions.append((start, i))
            in_text = False
    if in_text:
        text_regions.append((start, len(row_sum)))
    
    # Gambar bounding box per baris
    result = img.copy()
    for i, (y1, y2) in enumerate(text_regions):
        cv2.rectangle(result, (5, y1-2), (img.shape[1]-5, y2+2), (0, 200, 0), 2)
        cv2.putText(result, f"Baris {i+1}", (img.shape[1]-100, y1+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
    
    # Coba tesseract jika tersedia
    ocr_text = ""
    try:
        import pytesseract
        ocr_text = pytesseract.image_to_string(gray, lang='eng')
        print(f"  Tesseract output:\n{ocr_text}")
    except ImportError:
        print("  [INFO] pytesseract tidak tersedia. Install: pip install pytesseract")
        print("  [INFO] Simulasi OCR berdasarkan deteksi region teks")
    
    return result, text_regions, ocr_text


def visualisasi_pipeline_ocr(img):
    """Menampilkan pipeline OCR step-by-step."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(gray, 3)
    _, bin_clean = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 15), np.uint8)
    dilated = cv2.dilate(bin_clean, kernel, iterations=1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    steps = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "1. Input"),
        (gray, "2. Grayscale"),
        (binary, "3. Binarisasi"),
        (bin_clean, "4. Denoise + Threshold"),
        (dilated, "5. Dilasi (grup teks)"),
    ]
    for ax, (im, title) in zip(axes.flat[:5], steps):
        cmap = 'gray' if len(im.shape) == 2 else None
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, "6. OCR Recognition\n(Tesseract/CRNN)\n\n→ Teks Output",
                    ha='center', va='center', fontsize=12,
                    bbox=dict(facecolor='lightyellow', edgecolor='gray'))
    
    plt.suptitle("Pipeline OCR: Input → Preprocessing → Recognition", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_ocr_pipeline.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/07_ocr_pipeline.png")


def main():
    """Fungsi utama: OCR Tesseract konsep."""
    print("=" * 60)
    print("PERCOBAAN 7: OCR TESSERACT (KONSEP)")
    print("=" * 60)

    img = load_gambar("teks_printed.jpg")
    if img is None:
        img = buat_gambar_teks()

    print("\n--- 1. Simulasi OCR ---")
    result, regions, text = simulasi_ocr(img)
    print(f"  Baris teks terdeteksi: {len(regions)}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "07_ocr_result.png"), result)
    print("  [SAVED] output/07_ocr_result.png")

    print("\n--- 2. Pipeline OCR ---")
    visualisasi_pipeline_ocr(img)

    cv2.imshow("OCR Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nRINGKASAN: OCR pipeline: preprocessing (binarisasi,")
    print("denoise) → text detection → character recognition.")


if __name__ == "__main__":
    main()
