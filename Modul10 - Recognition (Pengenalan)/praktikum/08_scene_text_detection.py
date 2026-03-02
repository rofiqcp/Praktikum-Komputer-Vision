"""
==========================================================================
PERCOBAAN 8: SCENE TEXT DETECTION
==========================================================================
Deteksi teks pada gambar pemandangan/scene menggunakan teknik MSER dan morphology.

Referensi: Mastering OpenCV 4
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


def deteksi_teks_mser(img):
    """
    Mendeteksi region teks menggunakan MSER (Maximally Stable Extremal Regions).
    MSER mendeteksi region dengan intensitas stabil pada berbagai threshold.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    mser.setMinArea(100)
    mser.setMaxArea(5000)
    regions, boxes = mser.detectRegions(gray)
    
    result = img.copy()
    for box in boxes:
        x, y, w, h = box
        aspect_ratio = w / max(h, 1)
        if 0.1 < aspect_ratio < 10 and w > 10 and h > 10:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return result, len(boxes)


def deteksi_teks_morphology(img):
    """
    Mendeteksi region teks menggunakan gradien + morphology.
    1. Sobel gradient → threshold → dilasi horizontal → contour.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad = np.absolute(grad_x).astype(np.uint8)
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = img.copy()
    text_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 10 and w/h > 1.5:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            text_boxes.append((x, y, w, h))
    return result, text_boxes


def main():
    """Fungsi utama: scene text detection."""
    print("=" * 60)
    print("PERCOBAAN 8: SCENE TEXT DETECTION")
    print("=" * 60)
    
    img = load_gambar("teks_scene.jpg")
    if img is None:
        img = np.ones((300, 500, 3), dtype=np.uint8) * 200
        cv2.putText(img, "OPEN 24H", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30,30,30), 3)
        cv2.putText(img, "Coffee Shop", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,50,50), 2)
        cv2.putText(img, "Free WiFi", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 2)
    
    print("\n--- 1. Deteksi MSER ---")
    result_mser, n_regions = deteksi_teks_mser(img)
    print(f"  Region MSER: {n_regions}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "08_mser.png"), result_mser)
    print("  [SAVED] output/08_mser.png")
    
    print("\n--- 2. Deteksi Morphology ---")
    result_morph, boxes = deteksi_teks_morphology(img)
    print(f"  Text boxes: {len(boxes)}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "08_morphology.png"), result_morph)
    print("  [SAVED] output/08_morphology.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("Asli"); axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(result_mser, cv2.COLOR_BGR2RGB)); axes[1].set_title("MSER"); axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(result_morph, cv2.COLOR_BGR2RGB)); axes[2].set_title("Morphology"); axes[2].axis('off')
    plt.suptitle("Scene Text Detection", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_text_detection.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/08_text_detection.png")
    
    cv2.imshow("Text Detection", result_morph)
    cv2.waitKey(0); cv2.destroyAllWindows()
    print("\nRINGKASAN: MSER dan morphology-based methods untuk deteksi teks scene.")


if __name__ == "__main__":
    main()
