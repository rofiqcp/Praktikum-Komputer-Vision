"""
==========================================================================
PERCOBAAN 5: FACE LANDMARK DETECTION
==========================================================================
Program ini mendemonstrasikan deteksi landmark (titik-titik fitur)
pada wajah menggunakan berbagai metode.

Konsep yang dipelajari:
- Face landmark: 68 titik, 5 titik
- Aplikasi: face alignment, expression analysis
- cv2.face.createFacemarkLBF() atau dlib

Referensi: Mastering OpenCV 4, dlib docs
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


def deteksi_landmark_manual(img):
    """
    Mendeteksi landmark wajah secara sederhana.
    Menggunakan Haar Cascade + estimasi posisi fitur.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    result = img.copy()
    landmarks_all = []
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Estimasi landmark berdasarkan proporsi wajah
        landmarks = {
            "mata_kiri": (x + int(w*0.3), y + int(h*0.35)),
            "mata_kanan": (x + int(w*0.7), y + int(h*0.35)),
            "hidung": (x + int(w*0.5), y + int(h*0.55)),
            "mulut_kiri": (x + int(w*0.35), y + int(h*0.75)),
            "mulut_kanan": (x + int(w*0.65), y + int(h*0.75)),
        }
        for name, (px, py) in landmarks.items():
            cv2.circle(result, (px, py), 3, (0, 0, 255), -1)
            cv2.putText(result, name.split('_')[0], (px+5, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        landmarks_all.append(landmarks)
    return result, landmarks_all


def visualisasi_68_titik():
    """Membuat diagram 68 titik landmark standar pada wajah."""
    fig, ax = plt.subplots(figsize=(6, 8))
    # Gambar outline wajah
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(150 + 120*np.cos(theta), 200 + 160*np.sin(theta), 'b-', alpha=0.3)
    
    regions = {
        "Jawline (0-16)": [(50+i*18, 100+abs(i-8)*12) for i in range(17)],
        "Alis Kiri (17-21)": [(80+i*12, 120) for i in range(5)],
        "Alis Kanan (22-26)": [(180+i*12, 120) for i in range(5)],
        "Hidung (27-35)": [(150, 150+i*10) for i in range(9)],
        "Mata Kiri (36-41)": [(90+i*8, 150) for i in range(6)],
        "Mata Kanan (42-47)": [(190+i*8, 150) for i in range(6)],
        "Mulut (48-67)": [(110+i*8, 260) for i in range(20)],
    }
    colors = ['red', 'blue', 'blue', 'green', 'purple', 'purple', 'orange']
    for (name, pts), color in zip(regions.items(), colors):
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=color, s=20, zorder=5)
        ax.annotate(name, xy=(xs[0], ys[0]-10), fontsize=7, color=color)
    
    ax.set_xlim(0, 300)
    ax.set_ylim(350, 50)
    ax.set_title("68 Titik Landmark Wajah (Standar dlib)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_landmark_68.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/05_landmark_68.png")


def main():
    """Fungsi utama: face landmark detection."""
    print("=" * 60)
    print("PERCOBAAN 5: FACE LANDMARK DETECTION")
    print("=" * 60)

    print("\n--- 1. Deteksi Landmark ---")
    img = load_gambar("wajah_single.jpg")
    if img is None:
        img = np.ones((300, 300, 3), dtype=np.uint8) * 200
        cv2.circle(img, (150, 130), 80, (180, 160, 140), -1)
        cv2.circle(img, (120, 110), 10, (60, 50, 40), -1)
        cv2.circle(img, (180, 110), 10, (60, 50, 40), -1)
    
    result, landmarks = deteksi_landmark_manual(img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "05_landmarks.png"), result)
    print(f"  Wajah ditemukan: {len(landmarks)}")
    print("  [SAVED] output/05_landmarks.png")

    print("\n--- 2. Diagram 68 Titik ---")
    visualisasi_68_titik()

    cv2.imshow("Face Landmarks", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nRINGKASAN: Landmark wajah penting untuk alignment,")
    print("expression analysis, dan face morphing.")


if __name__ == "__main__":
    main()
