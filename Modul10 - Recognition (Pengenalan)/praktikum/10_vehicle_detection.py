"""
==========================================================================
PERCOBAAN 10: VEHICLE DETECTION
==========================================================================
Deteksi kendaraan menggunakan background subtraction dan contour analysis.

Referensi: Learning OpenCV
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


def deteksi_kendaraan_contour(img):
    """
    Mendeteksi kendaraan menggunakan background model sederhana + contour.
    Pada gambar statis: edge detection + contour filtering.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = img.copy()
    vehicles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if 0.5 < aspect < 4.0:
                cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(result, f"Vehicle", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                vehicles.append((x,y,w,h))
    return result, vehicles


def deteksi_kendaraan_haar(img):
    """Deteksi menggunakan Haar Cascade untuk mobil (jika tersedia)."""
    car_cascade_path = cv2.data.haarcascades + 'haarcascade_car.xml'
    if not os.path.exists(car_cascade_path):
        # Gunakan default file
        car_cascade_path = os.path.join(IMAGE_DIR, "haarcascade_car.xml")
    
    result = img.copy()
    try:
        cascade = cv2.CascadeClassifier(car_cascade_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
        for (x,y,w,h) in cars:
            cv2.rectangle(result, (x,y), (x+w,y+h), (0,0,255), 2)
        return result, len(cars)
    except:
        return result, 0


def main():
    """Fungsi utama: vehicle detection."""
    print("=" * 60)
    print("PERCOBAAN 10: VEHICLE DETECTION")
    print("=" * 60)
    
    img = load_gambar("kendaraan.jpg")
    if img is None:
        img = load_gambar("mobil.jpg")
    if img is None:
        img = np.ones((400, 600, 3), dtype=np.uint8) * 180
        cv2.rectangle(img, (100,200), (250,320), (50,50,150), -1)  # mobil 1
        cv2.rectangle(img, (350,180), (520,310), (150,50,50), -1)  # mobil 2
    
    print("\n--- 1. Deteksi Contour ---")
    result_cnt, vehicles = deteksi_kendaraan_contour(img)
    print(f"  Kendaraan (contour): {len(vehicles)}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "10_vehicle_contour.png"), result_cnt)
    print("  [SAVED] output/10_vehicle_contour.png")
    
    print("\n--- 2. Deteksi Haar ---")
    result_haar, n_cars = deteksi_kendaraan_haar(img)
    print(f"  Kendaraan (Haar): {n_cars}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(result_cnt, cv2.COLOR_BGR2RGB)); axes[0].set_title("Contour-based")
    axes[1].imshow(cv2.cvtColor(result_haar, cv2.COLOR_BGR2RGB)); axes[1].set_title("Haar Cascade")
    for ax in axes: ax.axis('off')
    plt.suptitle("Vehicle Detection", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_vehicle_detection.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/10_vehicle_detection.png")
    
    cv2.imshow("Vehicle Detection", result_cnt)
    cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
