"""
==========================================================================
PERCOBAAN 10: DETEKSI OBJEK SLIDING WINDOW
==========================================================================
Program ini mendemonstrasikan konsep deteksi objek menggunakan teknik
sliding window. Window (jendela) bergeser di seluruh gambar dan pada
setiap posisi dilakukan klasifikasi apakah ada objek atau tidak.

Konsep yang dipelajari:
- Sliding window: jendela bergeser pixel demi pixel
- Image pyramid: multi-scale detection
- Non-Maximum Suppression (NMS) untuk menghilangkan duplikat
- Perbandingan dengan metode modern (region proposal)

Referensi: Szeliski Ch.5, Deep Learning for CV (Rosebrock)
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


def sliding_window(img, window_size, step_size):
    """
    Generator yang menghasilkan setiap posisi sliding window.
    Yield: (x, y, window_crop)
    window_size: (width, height) ukuran jendela
    step_size: jumlah pixel geser per langkah
    """
    for y in range(0, img.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, img.shape[1] - window_size[0] + 1, step_size):
            window = img[y:y + window_size[1], x:x + window_size[0]]
            yield (x, y, window)


def image_pyramid(img, scale_factor=0.7, min_size=(64, 64)):
    """
    Membuat image pyramid: versi gambar pada berbagai skala.
    Setiap level lebih kecil dari sebelumnya (downscale).
    Berguna untuk deteksi objek di berbagai ukuran.
    """
    pyramid = [img.copy()]
    current = img.copy()
    while True:
        h, w = current.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        if new_w < min_size[0] or new_h < min_size[1]:
            break
        current = cv2.resize(current, (new_w, new_h))
        pyramid.append(current)
    return pyramid


def klasifikasi_sederhana(window):
    """
    Klasifikasi sederhana berdasarkan statistik warna.
    Mengembalikan skor 'objek' dan 'background'.
    Di dunia nyata, ini diganti dengan CNN/SVM classifier.
    """
    hsv = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
    mean_sat = np.mean(hsv[:, :, 1])
    mean_val = np.mean(hsv[:, :, 2])
    std_val = np.std(window)
    skor = (mean_sat / 255.0) * 0.4 + (std_val / 128.0) * 0.6
    return min(1.0, skor)


def non_maximum_suppression(boxes, scores, threshold=0.3):
    """
    Non-Maximum Suppression: menghilangkan bounding box yang overlap.
    Hanya menyimpan box dengan skor tertinggi di setiap area.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < threshold]

    return keep


def visualisasi_sliding_window(img, window_size=(64, 64), step=32):
    """
    Menampilkan proses sliding window pada gambar.
    Menggambar kotak merah pada beberapa posisi window.
    """
    img_vis = img.copy()
    count = 0
    boxes = []
    scores = []

    for (x, y, window) in sliding_window(img, window_size, step):
        skor = klasifikasi_sederhana(window)
        if skor > 0.5:
            boxes.append([x, y, x + window_size[0], y + window_size[1]])
            scores.append(skor)
        count += 1

    print(f"  Total windows diperiksa: {count}")
    print(f"  Deteksi sebelum NMS: {len(boxes)}")

    # Terapkan NMS
    keep = non_maximum_suppression(boxes, scores, threshold=0.3)
    print(f"  Deteksi setelah NMS: {len(keep)}")

    # Gambar semua deteksi (merah tipis)
    for box in boxes:
        cv2.rectangle(img_vis, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

    # Gambar deteksi setelah NMS (hijau tebal)
    img_nms = img.copy()
    for i in keep:
        box = boxes[i]
        cv2.rectangle(img_nms, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(img_nms, f"{scores[i]:.2f}", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Semua Deteksi ({len(boxes)})")
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(img_nms, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Setelah NMS ({len(keep)})")
    ax2.axis('off')

    plt.suptitle("Sliding Window + Non-Maximum Suppression", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_sliding_window.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/10_sliding_window.png")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "10_deteksi_nms.jpg"), img_nms)
    print("  [SAVED] output/10_deteksi_nms.jpg")
    return img_nms


def visualisasi_pyramid(img):
    """Menampilkan image pyramid pada berbagai skala."""
    pyramid = image_pyramid(img, scale_factor=0.7)
    n = min(5, len(pyramid))
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    for i in range(n):
        axes[i].imshow(cv2.cvtColor(pyramid[i], cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Level {i}\n{pyramid[i].shape[1]}x{pyramid[i].shape[0]}", fontsize=9)
        axes[i].axis('off')

    plt.suptitle("Image Pyramid untuk Multi-scale Detection", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_image_pyramid.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/10_image_pyramid.png")


def main():
    """Fungsi utama: deteksi objek sliding window."""
    print("=" * 60)
    print("PERCOBAAN 10: DETEKSI OBJEK SLIDING WINDOW")
    print("=" * 60)

    img = load_gambar("scene_outdoor.jpg")
    if img is None:
        img = load_gambar("kucing.jpg")
    if img is None:
        return
    img = cv2.resize(img, (320, 240))

    print("\n--- 1. Sliding Window Detection ---")
    img_nms = visualisasi_sliding_window(img)

    print("\n--- 2. Image Pyramid ---")
    visualisasi_pyramid(img)

    cv2.imshow("Deteksi Sliding Window", img_nms)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 10")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. Sliding window memeriksa setiap posisi dalam gambar
2. Image pyramid mendeteksi objek di berbagai skala
3. NMS menghilangkan deteksi duplikat yang overlap
4. Metode ini lambat tapi fundamental untuk understanding
5. Metode modern (YOLO, SSD) lebih efisien

Output: output/10_sliding_window.png, output/10_image_pyramid.png,
        output/10_deteksi_nms.jpg
""")


if __name__ == "__main__":
    main()
