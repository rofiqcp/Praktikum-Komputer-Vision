"""
==========================================================================
PERCOBAAN 17: ROC CURVE DAN AUC
==========================================================================
Analisis performa recognition menggunakan ROC curve dan AUC.

Referensi: Geron, ML for OpenCV
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


def simulasi_score(n_pos=100, n_neg=200, quality="good"):
    """Simulasi skor recognition untuk ROC curve."""
    np.random.seed(42)
    if quality == "good":
        pos_scores = np.random.normal(0.75, 0.1, n_pos)
        neg_scores = np.random.normal(0.3, 0.15, n_neg)
    elif quality == "medium":
        pos_scores = np.random.normal(0.6, 0.15, n_pos)
        neg_scores = np.random.normal(0.4, 0.15, n_neg)
    else:
        pos_scores = np.random.normal(0.55, 0.2, n_pos)
        neg_scores = np.random.normal(0.45, 0.2, n_neg)
    return np.clip(pos_scores, 0, 1), np.clip(neg_scores, 0, 1)


def hitung_roc(pos_scores, neg_scores):
    """Menghitung ROC curve: TPR vs FPR untuk berbagai threshold."""
    all_scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    thresholds = np.linspace(0, 1, 200)
    tpr_list, fpr_list = [], []
    for th in thresholds:
        preds = (all_scores >= th).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        tpr_list.append(tp / max(tp+fn, 1))
        fpr_list.append(fp / max(fp+tn, 1))
    auc = -np.trapz(tpr_list, fpr_list)
    return fpr_list, tpr_list, auc


def main():
    """Fungsi utama: ROC curve dan AUC."""
    print("=" * 60)
    print("PERCOBAAN 17: ROC CURVE DAN AUC")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for quality, color, label in [("good","green","Baik"), ("medium","orange","Sedang"), ("bad","red","Buruk")]:
        pos, neg = simulasi_score(quality=quality)
        fpr, tpr, auc = hitung_roc(pos, neg)
        axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC={auc:.2f})")
        print(f"  Model {label}: AUC = {auc:.3f}")
    
    axes[0].plot([0,1], [0,1], 'k--', alpha=0.5)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve - Recognition"); axes[0].legend(); axes[0].grid(alpha=0.3)
    
    # Score distribution
    pos_good, neg_good = simulasi_score(quality="good")
    axes[1].hist(pos_good, bins=30, alpha=0.6, color='green', label='Genuine')
    axes[1].hist(neg_good, bins=30, alpha=0.6, color='red', label='Impostor')
    axes[1].axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    axes[1].set_xlabel("Score"); axes[1].set_ylabel("Count")
    axes[1].set_title("Score Distribution"); axes[1].legend()
    
    plt.suptitle("ROC Curve dan Score Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_roc_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/17_roc_curve.png")
    print("\nRINGKASAN: ROC-AUC mengukur performa recognition pada semua threshold.")


if __name__ == "__main__":
    main()
