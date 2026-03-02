"""
==========================================================================
PERCOBAAN 16: EVALUASI DETECTION METRICS
==========================================================================
Metrik evaluasi deteksi objek: IoU, mAP, Precision-Recall per threshold.

Referensi: Szeliski, ML for OpenCV
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


def hitung_iou(box1, box2):
    """
    Menghitung IoU (Intersection over Union) antara dua bounding box.
    box = [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def hitung_ap(precisions, recalls):
    """Menghitung Average Precision dari precision-recall curve."""
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    ap = 0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i-1]) * mpre[i]
    return ap


def main():
    """Fungsi utama: evaluasi detection metrics."""
    print("=" * 60)
    print("PERCOBAAN 16: EVALUASI DETECTION METRICS")
    print("=" * 60)
    
    np.random.seed(42)
    # Simulasi GT dan prediksi
    gt_boxes = [[50,50,150,150], [200,100,350,250], [400,50,500,200]]
    pred_boxes = [[55,45,160,155], [190,90,340,245], [405,55,510,195], [100,200,200,300]]
    pred_scores = [0.95, 0.88, 0.75, 0.45]
    
    print("\n--- 1. IoU per Deteksi ---")
    for i, pb in enumerate(pred_boxes):
        ious = [hitung_iou(pb, gt) for gt in gt_boxes]
        best_iou = max(ious)
        print(f"  Pred {i+1}: best IoU = {best_iou:.3f} (conf={pred_scores[i]:.2f})")
    
    print("\n--- 2. Visualisasi IoU ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Gambar boxes
    canvas = np.ones((350, 550, 3), dtype=np.uint8) * 240
    for gt in gt_boxes:
        cv2.rectangle(canvas, (gt[0],gt[1]), (gt[2],gt[3]), (0,200,0), 2)
    for pb, sc in zip(pred_boxes, pred_scores):
        cv2.rectangle(canvas, (pb[0],pb[1]), (pb[2],pb[3]), (200,0,0), 2)
    axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Hijau=GT, Merah=Pred"); axes[0].axis('off')
    
    # IoU vs threshold
    thresholds = np.arange(0.1, 1.0, 0.05)
    tp_counts = []
    for th in thresholds:
        tp = 0
        for pb in pred_boxes:
            if any(hitung_iou(pb, gt) >= th for gt in gt_boxes):
                tp += 1
        tp_counts.append(tp)
    axes[1].plot(thresholds, tp_counts, 'b-o', linewidth=2)
    axes[1].set_xlabel("IoU Threshold"); axes[1].set_ylabel("True Positives")
    axes[1].set_title("TP vs IoU Threshold"); axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Detection Evaluation: IoU dan mAP", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_detection_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/16_detection_metrics.png")
    print("\nRINGKASAN: IoU mengukur overlap GT vs prediksi. mAP = rata-rata AP per kelas.")


if __name__ == "__main__":
    main()
