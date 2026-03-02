"""
==========================================================================
PERCOBAAN 9: PEDESTRIAN DETECTION DENGAN HOG
==========================================================================
Deteksi pejalan kaki menggunakan HOG descriptor + SVM classifier bawaan OpenCV.

Referensi: Learning OpenCV, ML for OpenCV
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


def deteksi_pedestrian(img):
    """
    Mendeteksi pejalan kaki menggunakan HOG + SVM bawaan OpenCV.
    HOG: histogram gradien terarah, invariant terhadap pencahayaan.
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(img, winStride=(8,8),
                                           padding=(4,4), scale=1.05)
    return boxes, weights


def visualisasi_hog(gray):
    """Menghitung dan memvisualisasikan gradien HOG."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gray, cmap='gray'); axes[0].set_title("Grayscale")
    axes[1].imshow(np.abs(gx), cmap='hot'); axes[1].set_title("Gradient X")
    axes[2].imshow(np.abs(gy), cmap='hot'); axes[2].set_title("Gradient Y")
    axes[3].imshow(mag, cmap='hot'); axes[3].set_title("Magnitude")
    for ax in axes: ax.axis('off')
    plt.suptitle("HOG: Gradient Visualization", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_hog_gradient.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/09_hog_gradient.png")


def main():
    """Fungsi utama: pedestrian detection HOG."""
    print("=" * 60)
    print("PERCOBAAN 9: PEDESTRIAN DETECTION DENGAN HOG")
    print("=" * 60)
    
    img = load_gambar("pedestrian.jpg")
    if img is None:
        img = np.ones((400, 600, 3), dtype=np.uint8) * 180
        for x in [150, 300, 450]:
            cv2.rectangle(img, (x-20, 100), (x+20, 350), (100, 80, 60), -1)
            cv2.circle(img, (x, 80), 20, (200, 180, 160), -1)
    
    print("\n--- 1. Deteksi Pedestrian ---")
    boxes, weights = deteksi_pedestrian(img)
    result = img.copy()
    for (x,y,w,h), weight in zip(boxes, weights):
        cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(result, f"{weight[0]:.2f}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    print(f"  Pedestrian terdeteksi: {len(boxes)}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "09_pedestrian.png"), result)
    print("  [SAVED] output/09_pedestrian.png")
    
    print("\n--- 2. HOG Gradient ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    visualisasi_hog(gray)
    
    cv2.imshow("Pedestrian Detection", result)
    cv2.waitKey(0); cv2.destroyAllWindows()
    print("\nRINGKASAN: HOG+SVM mendeteksi pejalan kaki berdasarkan gradient orientasi.")


if __name__ == "__main__":
    main()
