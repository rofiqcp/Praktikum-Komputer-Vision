"""
==========================================================================
PERCOBAAN 12: YOLO - KONSEP GRID DAN PREDIKSI
==========================================================================
Program ini mendemonstrasikan konsep dasar arsitektur YOLO (You Only Look
Once) untuk deteksi objek. YOLO membagi gambar menjadi grid SxS dan
setiap sel grid memprediksi bounding box + kelas objek.

Konsep yang dipelajari:
- Grid cell dan prediksi per sel
- Format bounding box (x, y, w, h, confidence)
- Anchor boxes untuk berbagai aspect ratio
- IoU (Intersection over Union) untuk evaluasi
- NMS untuk menghilangkan duplikat

Referensi: Deep Learning for CV (Rosebrock), Mastering OpenCV 4
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


def gambar_grid_yolo(img, grid_size=7):
    """
    Menggambar grid YOLO pada gambar.
    Setiap sel grid bertanggung jawab mendeteksi objek yang center-nya
    jatuh di dalam sel tersebut.
    """
    h, w = img.shape[:2]
    img_grid = img.copy()
    cell_h, cell_w = h // grid_size, w // grid_size

    for i in range(1, grid_size):
        cv2.line(img_grid, (i * cell_w, 0), (i * cell_w, h), (0, 255, 255), 1)
        cv2.line(img_grid, (0, i * cell_h), (w, i * cell_h), (0, 255, 255), 1)
    return img_grid, cell_h, cell_w


def hitung_iou(box1, box2):
    """
    Menghitung Intersection over Union (IoU) antara dua bounding box.
    IoU = Area_Intersection / Area_Union
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def visualisasi_prediksi_grid(img, grid_size=7):
    """
    Mensimulasikan prediksi YOLO pada grid gambar.
    Setiap sel memiliki confidence score dan bounding box prediksi.
    """
    h, w = img.shape[:2]
    img_grid, cell_h, cell_w = gambar_grid_yolo(img, grid_size)

    # Simulasi: buat beberapa prediksi random
    np.random.seed(42)
    predictions = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            conf = np.random.random()
            if conf > 0.7:
                cx = (gx + 0.5) * cell_w
                cy = (gy + 0.5) * cell_h
                bw = np.random.uniform(30, 100)
                bh = np.random.uniform(30, 100)
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                predictions.append({
                    'box': [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                    'confidence': conf,
                    'grid': (gx, gy)
                })

    # Gambar prediksi
    for pred in predictions:
        box = pred['box']
        conf = pred['confidence']
        color = (0, int(255 * conf), int(255 * (1 - conf)))
        cv2.rectangle(img_grid, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(img_grid, f"{conf:.2f}", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img_grid, predictions


def visualisasi_iou():
    """
    Membuat visualisasi IoU dengan berbagai tingkat overlap.
    Menunjukkan arti IoU=0.1, 0.5, 0.9 secara visual.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    cases = [
        ("IoU ≈ 0.0", [10, 10, 60, 60], [70, 70, 120, 120]),
        ("IoU ≈ 0.25", [10, 10, 70, 70], [40, 40, 100, 100]),
        ("IoU ≈ 0.5", [10, 10, 80, 80], [30, 30, 100, 100]),
        ("IoU ≈ 0.9", [10, 10, 90, 90], [15, 15, 95, 95]),
    ]

    for i, (title, box1, box2) in enumerate(cases):
        canvas = np.ones((130, 130, 3), dtype=np.uint8) * 255
        cv2.rectangle(canvas, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
        cv2.rectangle(canvas, (box2[0], box2[1]), (box2[2], box2[3]), (0, 0, 255), 2)
        iou = hitung_iou(box1, box2)
        axes[i].imshow(canvas)
        axes[i].set_title(f"{title}\nIoU = {iou:.3f}")
        axes[i].axis('off')

    plt.suptitle("Visualisasi IoU (Intersection over Union)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_iou_visualisasi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/12_iou_visualisasi.png")


def visualisasi_anchor_boxes():
    """
    Menampilkan anchor boxes dengan berbagai aspect ratio.
    Anchor boxes membantu YOLO mendeteksi objek berbagai bentuk.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    canvas = np.ones((300, 300, 3), dtype=np.uint8) * 240

    # Anchor boxes dengan berbagai aspect ratio
    anchors = [(60, 60), (80, 40), (40, 80), (100, 50), (50, 100)]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0), (128, 0, 255)]
    center = (150, 150)

    for (aw, ah), color in zip(anchors, colors):
        x1, y1 = center[0] - aw // 2, center[1] - ah // 2
        x2, y2 = center[0] + aw // 2, center[1] + ah // 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    ax.imshow(canvas)
    ax.set_title("Anchor Boxes (berbagai aspect ratio)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_anchor_boxes.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/12_anchor_boxes.png")


def main():
    """Fungsi utama: konsep YOLO grid dan prediksi."""
    print("=" * 60)
    print("PERCOBAAN 12: YOLO - KONSEP GRID DAN PREDIKSI")
    print("=" * 60)

    img = load_gambar("scene_outdoor.jpg")
    if img is None:
        img = load_gambar("kucing.jpg")
    if img is None:
        return
    img = cv2.resize(img, (448, 448))

    print("\n--- 1. Grid YOLO dan Prediksi ---")
    img_pred, predictions = visualisasi_prediksi_grid(img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "12_yolo_grid.jpg"), img_pred)
    print(f"  Jumlah prediksi: {len(predictions)}")
    print("  [SAVED] output/12_yolo_grid.jpg")

    print("\n--- 2. Visualisasi IoU ---")
    visualisasi_iou()

    print("\n--- 3. Anchor Boxes ---")
    visualisasi_anchor_boxes()

    cv2.imshow("YOLO Grid - Percobaan 12", img_pred)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 12")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. YOLO membagi gambar ke grid SxS
2. Setiap sel memprediksi B boxes + C class probabilities
3. IoU mengukur overlap antara prediksi dan ground truth
4. Anchor boxes menangani objek dengan aspect ratio berbeda
5. NMS menghilangkan deteksi duplikat

Output: output/12_yolo_grid.jpg, output/12_iou_visualisasi.png,
        output/12_anchor_boxes.png
""")


if __name__ == "__main__":
    main()
