"""
==========================================================================
PERCOBAAN 15: EVALUASI CLASSIFICATION METRICS
==========================================================================
Metrik evaluasi klasifikasi: accuracy, precision, recall, F1, confusion matrix.

Referensi: ML for OpenCV, Geron
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


def hitung_confusion_matrix(y_true, y_pred, n_classes):
    """Menghitung confusion matrix: CM[i][j] = true i, pred j."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def hitung_metrik(cm):
    """Menghitung Accuracy, Precision, Recall, F1 dari CM."""
    n = cm.shape[0]
    precision, recall, f1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        precision[i] = tp/(tp+fp) if tp+fp>0 else 0
        recall[i] = tp/(tp+fn) if tp+fn>0 else 0
        f1[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i]) if precision[i]+recall[i]>0 else 0
    return np.trace(cm)/cm.sum(), precision, recall, f1


def main():
    """Fungsi utama: evaluasi classification metrics."""
    print("=" * 60)
    print("PERCOBAAN 15: EVALUASI CLASSIFICATION METRICS")
    print("=" * 60)
    
    np.random.seed(42)
    classes = ["Wajah A", "Wajah B", "Wajah C", "Unknown"]
    n_cls = len(classes)
    n_samples = 200
    y_true = np.random.randint(0, n_cls, n_samples)
    y_pred = y_true.copy()
    wrong = np.random.choice(n_samples, 40, replace=False)
    y_pred[wrong] = np.random.randint(0, n_cls, 40)
    
    cm = hitung_confusion_matrix(y_true, y_pred, n_cls)
    acc, prec, rec, f1 = hitung_metrik(cm)
    
    print(f"\n  Accuracy: {acc:.2%}")
    for i, c in enumerate(classes):
        print(f"  {c}: P={prec[i]:.2f} R={rec[i]:.2f} F1={f1[i]:.2f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # CM heatmap
    im = axes[0].imshow(cm, cmap='YlOrRd')
    axes[0].set_xticks(range(n_cls)); axes[0].set_xticklabels(classes, rotation=30)
    axes[0].set_yticks(range(n_cls)); axes[0].set_yticklabels(classes)
    for i in range(n_cls):
        for j in range(n_cls):
            axes[0].text(j, i, str(cm[i,j]), ha='center', va='center',
                        color='white' if cm[i,j]>cm.max()/2 else 'black')
    axes[0].set_title(f"Confusion Matrix (Acc={acc:.1%})")
    plt.colorbar(im, ax=axes[0])
    
    # Bar chart metrics
    x = np.arange(n_cls); w = 0.25
    axes[1].bar(x-w, prec, w, label='Precision', color='steelblue')
    axes[1].bar(x, rec, w, label='Recall', color='coral')
    axes[1].bar(x+w, f1, w, label='F1', color='seagreen')
    axes[1].set_xticks(x); axes[1].set_xticklabels(classes, rotation=30)
    axes[1].legend(); axes[1].set_ylim(0, 1.1); axes[1].grid(alpha=0.3, axis='y')
    axes[1].set_title("Metrik per Kelas")
    
    plt.suptitle("Classification Evaluation Metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_classification_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/15_classification_metrics.png")
    print("\nRINGKASAN: CM, Precision, Recall, F1 mengukur performa klasifikasi.")


if __name__ == "__main__":
    main()
