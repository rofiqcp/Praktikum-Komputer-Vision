"""
==========================================================================
PERCOBAAN 2: FACE DETECTION DENGAN DNN (KONSEP)
==========================================================================
Program ini mendemonstrasikan konsep deteksi wajah menggunakan
Deep Neural Network melalui modul OpenCV DNN.

Konsep yang dipelajari:
- OpenCV DNN module untuk face detection
- Perbandingan Haar Cascade vs DNN
- Confidence threshold dan filtering
- Preprocessing blob untuk face detection

Referensi: Mastering OpenCV 4, OpenCV DNN docs
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


def deteksi_wajah_dnn(img, confidence_threshold=0.5):
    """
    Deteksi wajah menggunakan DNN (Caffe model).
    Model: res10_300x300_ssd_iter_140000.caffemodel
    Jika model tidak tersedia, gunakan simulasi.
    """
    model_path = os.path.join(IMAGE_DIR, "deploy.prototxt")
    weights_path = os.path.join(IMAGE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    if os.path.exists(model_path) and os.path.exists(weights_path):
        net = cv2.dnn.readNetFromCaffe(model_path, weights_path)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1, conf))
        return faces, "DNN"
    else:
        # Fallback: Haar Cascade
        print("  [INFO] Model DNN tidak ditemukan, menggunakan Haar Cascade")
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        faces = [(x, y, w, h, 0.95) for (x, y, w, h) in rects]
        return faces, "Haar (fallback)"


def visualisasi_perbandingan(img):
    """
    Membandingkan deteksi Haar Cascade vs DNN:
    - Akurasi (false positive/negative)
    - Kecepatan
    - Kemampuan menangani pose/rotasi
    """
    import time

    # Haar Cascade
    t0 = time.time()
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar_faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    t_haar = (time.time() - t0) * 1000

    # DNN (atau fallback)
    t0 = time.time()
    dnn_faces, method = deteksi_wajah_dnn(img)
    t_dnn = (time.time() - t0) * 1000

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Haar result
    result_haar = img.copy()
    for (x, y, w, h) in haar_faces:
        cv2.rectangle(result_haar, (x, y), (x + w, y + h), (0, 255, 0), 2)
    axes[0].imshow(cv2.cvtColor(result_haar, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Haar Cascade\n{len(haar_faces)} wajah, {t_haar:.1f}ms")
    axes[0].axis('off')

    # DNN result
    result_dnn = img.copy()
    for face in dnn_faces:
        x, y, w, h = face[:4]
        conf = face[4]
        cv2.rectangle(result_dnn, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(result_dnn, f"{conf:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    axes[1].imshow(cv2.cvtColor(result_dnn, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"{method}\n{len(dnn_faces)} wajah, {t_dnn:.1f}ms")
    axes[1].axis('off')

    # Tabel perbandingan
    axes[2].axis('off')
    table_data = [
        ["Aspek", "Haar Cascade", "DNN"],
        ["Deteksi", str(len(haar_faces)), str(len(dnn_faces))],
        ["Waktu", f"{t_haar:.1f}ms", f"{t_dnn:.1f}ms"],
        ["Confidence", "Tidak", "Ya"],
        ["Rotasi", "Terbatas", "Lebih baik"],
        ["Model Size", "~1MB", "~10MB"],
    ]
    table = axes[2].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axes[2].set_title("Perbandingan Metode", fontsize=12)

    plt.suptitle("Haar Cascade vs DNN Face Detection", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_haar_vs_dnn.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/02_haar_vs_dnn.png")


def main():
    """Fungsi utama: face detection DNN konsep."""
    print("=" * 60)
    print("PERCOBAAN 2: FACE DETECTION DENGAN DNN (KONSEP)")
    print("=" * 60)

    img = load_gambar("wajah_grup.jpg")
    if img is None:
        img = load_gambar("wajah_single.jpg")
    if img is None:
        img = np.ones((300, 400, 3), dtype=np.uint8) * 180
        cv2.circle(img, (100, 150), 50, (200, 180, 160), -1)
        cv2.circle(img, (300, 150), 50, (200, 180, 160), -1)

    print("\n--- 1. Deteksi DNN ---")
    faces, method = deteksi_wajah_dnn(img)
    print(f"  Metode: {method}")
    print(f"  Wajah terdeteksi: {len(faces)}")

    result = img.copy()
    for face in faces:
        x, y, w, h = face[:4]
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_dnn_detection.png"), result)
    print("  [SAVED] output/02_dnn_detection.png")

    print("\n--- 2. Perbandingan Haar vs DNN ---")
    visualisasi_perbandingan(img)

    cv2.imshow("Face Detection DNN", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN: DNN face detection lebih akurat dari Haar,")
    print("mendukung confidence score, dan lebih robust terhadap")
    print("variasi pose dan pencahayaan.")
    print("=" * 60)


if __name__ == "__main__":
    main()
