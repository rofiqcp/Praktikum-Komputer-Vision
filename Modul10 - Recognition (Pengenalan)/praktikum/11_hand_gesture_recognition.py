"""
==========================================================================
PERCOBAAN 11: HAND GESTURE RECOGNITION
==========================================================================
Pengenalan gestur tangan menggunakan convexity defects dan hull analysis.

Referensi: OpenCV CV Projects Python
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


def deteksi_tangan_hsv(img):
    """
    Mendeteksi tangan menggunakan segmentasi warna kulit di HSV.
    Range warna kulit: H(0-20), S(30-170), V(60-255).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60])
    upper = np.array([20, 170, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def hitung_jari(contour):
    """
    Menghitung jumlah jari menggunakan convexity defects.
    Defect = titik terjauh dari convex hull ke kontur.
    """
    if len(contour) < 5:
        return 0, None, None
    
    hull = cv2.convexHull(contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour, hull)
    except:
        return 0, None, None
    
    if defects is None:
        return 0, cv2.convexHull(contour), None
    
    n_fingers = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        # Hitung sudut
        a = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        b = np.sqrt((far[0]-start[0])**2 + (far[1]-start[1])**2)
        c = np.sqrt((end[0]-far[0])**2 + (end[1]-far[1])**2)
        angle = np.arccos((b**2+c**2-a**2) / (2*b*c+1e-5))
        if angle < np.pi/2 and d > 5000:
            n_fingers += 1
    return n_fingers + 1, cv2.convexHull(contour), defects


def main():
    """Fungsi utama: hand gesture recognition."""
    print("=" * 60)
    print("PERCOBAAN 11: HAND GESTURE RECOGNITION")
    print("=" * 60)
    
    img = load_gambar("wajah_single.jpg")  # placeholder, gunakan gambar tangan jika ada
    if img is None:
        img = np.ones((400, 400, 3), dtype=np.uint8) * 200
    
    # Buat gambar sintetis tangan (5 jari)
    hand = np.zeros((400, 400, 3), dtype=np.uint8)
    pts = np.array([[200,350],[150,200],[120,100],[150,80],[170,150],
                    [190,60],[210,150],[230,70],[250,150],[280,100],
                    [270,200],[250,350]], np.int32)
    cv2.fillPoly(hand, [pts], (180, 160, 140))
    
    print("\n--- 1. Segmentasi Tangan ---")
    mask = deteksi_tangan_hsv(hand)
    mask_bin = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask_bin, 50, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        n_fingers, hull, defects = hitung_jari(cnt)
        print(f"  Jumlah jari: {n_fingers}")
        
        result = hand.copy()
        cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
        if hull is not None:
            cv2.drawContours(result, [cv2.convexHull(cnt)], -1, (255, 0, 0), 2)
        cv2.putText(result, f"Jari: {n_fingers}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "11_gesture.png"), result)
        print("  [SAVED] output/11_gesture.png")
        
        cv2.imshow("Gesture Recognition", result)
        cv2.waitKey(0); cv2.destroyAllWindows()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)); axes[0].set_title("Input")
    axes[1].imshow(mask_bin, cmap='gray'); axes[1].set_title("Mask")
    if contours:
        axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); axes[2].set_title(f"Jari: {n_fingers}")
    for ax in axes: ax.axis('off')
    plt.suptitle("Hand Gesture Recognition", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_hand_gesture.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/11_hand_gesture.png")
    print("\nRINGKASAN: Gestur tangan dikenali via convexity defects pada convex hull.")


if __name__ == "__main__":
    main()
