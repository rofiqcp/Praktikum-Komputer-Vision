"""
==========================================================================
PERCOBAAN 1: FACE DETECTION DENGAN HAAR CASCADE
==========================================================================
Program ini mendemonstrasikan deteksi wajah menggunakan algoritma
Viola-Jones dengan Haar Cascade Classifier bawaan OpenCV.

Konsep yang dipelajari:
- Haar-like features (edge, line, four-rectangle)
- Integral image untuk perhitungan cepat
- AdaBoost cascade classifier
- Parameter detectMultiScale: scaleFactor, minNeighbors, minSize

Referensi: Szeliski Ch.6, Viola & Jones (2001), OpenCV docs
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


def deteksi_wajah_haar(img, cascade_path=None, scale=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Mendeteksi wajah menggunakan Haar Cascade Classifier.
    scaleFactor: faktor pengecilan gambar di setiap level pyramid.
    minNeighbors: jumlah minimal tetangga untuk filter deteksi.
    """
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale,
                                           minNeighbors=min_neighbors, minSize=min_size)
    return faces


def gambar_bounding_box(img, faces, color=(0, 255, 0), thickness=2):
    """Menggambar bounding box di sekitar wajah yang terdeteksi."""
    result = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(result, f"Wajah", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result


def variasi_parameter(img):
    """
    Mendeteksi wajah dengan berbagai parameter untuk melihat pengaruhnya.
    scaleFactor kecil = lebih banyak deteksi, lebih lambat.
    minNeighbors besar = filter ketat, deteksi lebih sedikit.
    """
    params = [
        (1.05, 3, "scale=1.05, neighbors=3"),
        (1.1, 5, "scale=1.10, neighbors=5"),
        (1.2, 5, "scale=1.20, neighbors=5"),
        (1.3, 8, "scale=1.30, neighbors=8"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (scale, nn, label) in zip(axes.flat, params):
        faces = deteksi_wajah_haar(img, scale=scale, min_neighbors=nn)
        result = gambar_bounding_box(img, faces)
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{label}\nDeteksi: {len(faces)} wajah", fontsize=10)
        ax.axis('off')

    plt.suptitle("Pengaruh Parameter pada Haar Cascade", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_variasi_parameter.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/01_variasi_parameter.png")


def deteksi_multi_cascade(img):
    """
    Mendeteksi wajah dengan berbagai Haar Cascade:
    - frontal face (default)
    - frontal face (alt)
    - profile face
    - eye detection
    """
    cascades = {
        "Frontal Default": cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        "Frontal Alt": cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
        "Frontal Alt2": cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
    }

    fig, axes = plt.subplots(1, len(cascades), figsize=(15, 5))
    for ax, (name, path) in zip(axes, cascades.items()):
        faces = deteksi_wajah_haar(img, cascade_path=path)
        result = gambar_bounding_box(img, faces)
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}\n({len(faces)} wajah)")
        ax.axis('off')

    plt.suptitle("Perbandingan Haar Cascade Classifier", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_multi_cascade.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/01_multi_cascade.png")


def main():
    """Fungsi utama: face detection dengan Haar Cascade."""
    print("=" * 60)
    print("PERCOBAAN 1: FACE DETECTION DENGAN HAAR CASCADE")
    print("=" * 60)

    # Load gambar
    img = load_gambar("wajah_single.jpg")
    if img is None:
        img = load_gambar("wajah_grup.jpg")
    if img is None:
        print("  [INFO] Membuat gambar sintetis untuk demo")
        img = np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (150, 120), 60, (200, 180, 160), -1)
        cv2.circle(img, (130, 105), 8, (80, 60, 50), -1)
        cv2.circle(img, (170, 105), 8, (80, 60, 50), -1)

    # --- 1. Deteksi dasar ---
    print("\n--- 1. Deteksi Wajah Dasar ---")
    faces = deteksi_wajah_haar(img)
    result = gambar_bounding_box(img, faces)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_face_detection.png"), result)
    print(f"  Wajah terdeteksi: {len(faces)}")
    print("  [SAVED] output/01_face_detection.png")

    # --- 2. Variasi parameter ---
    print("\n--- 2. Variasi Parameter ---")
    variasi_parameter(img)

    # --- 3. Multi cascade ---
    print("\n--- 3. Perbandingan Cascade ---")
    deteksi_multi_cascade(img)

    # Tampilkan gambar
    cv2.imshow("Face Detection - Haar Cascade", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN: Haar Cascade mendeteksi wajah menggunakan")
    print("fitur Haar-like + AdaBoost cascade. Parameter scaleFactor")
    print("dan minNeighbors mengontrol sensitivitas deteksi.")
    print("=" * 60)


if __name__ == "__main__":
    main()
