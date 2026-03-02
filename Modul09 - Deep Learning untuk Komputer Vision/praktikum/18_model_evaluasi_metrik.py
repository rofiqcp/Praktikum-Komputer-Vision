"""
==========================================================================
PERCOBAAN 18: MODEL EVALUASI DAN METRIK
==========================================================================
Program ini mendemonstrasikan berbagai metrik evaluasi model deep learning
untuk klasifikasi dan deteksi objek.

Konsep yang dipelajari:
- Confusion Matrix: TP, TN, FP, FN
- Accuracy, Precision, Recall, F1-Score
- Kurva ROC dan AUC
- Precision-Recall Curve
- mAP (mean Average Precision) untuk deteksi objek
- Kapan menggunakan metrik tertentu

Referensi: Szeliski Ch.5, Géron Ch.10
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


def hitung_confusion_matrix(y_true, y_pred, n_classes):
    """
    Menghitung Confusion Matrix dari prediksi dan ground truth.
    CM[i][j] = jumlah sampel yang sebenarnya kelas i, diprediksi kelas j.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def hitung_metrik(cm):
    """
    Menghitung Precision, Recall, F1 dari confusion matrix.
    Precision = TP / (TP + FP) -> seberapa tepat prediksi positif
    Recall = TP / (TP + FN) -> seberapa lengkap deteksi positif
    F1 = 2 * P * R / (P + R) -> harmonik rata-rata P dan R
    """
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)

    for i in range(n):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) \
            if (precision[i] + recall[i]) > 0 else 0

    accuracy = np.trace(cm) / np.sum(cm)
    return accuracy, precision, recall, f1


def visualisasi_confusion_matrix(cm, classes):
    """
    Menampilkan Confusion Matrix sebagai heatmap berwarna.
    Diagonal = prediksi benar, off-diagonal = error.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd')
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Annotasi angka
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=12)

    ax.set_ylabel("Label Sebenarnya")
    ax.set_xlabel("Label Prediksi")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/18_confusion_matrix.png")
    return cm


def visualisasi_roc_curve():
    """
    Menampilkan kurva ROC (Receiver Operating Characteristic).
    ROC plot TPR vs FPR untuk berbagai threshold.
    AUC = luas di bawah kurva, semakin besar semakin baik (max 1.0).
    """
    np.random.seed(42)
    n = 500

    # Simulasi 3 model dengan performa berbeda
    models = {
        "Model Bagus (AUC~0.92)": (2.0, 1.5),
        "Model Sedang (AUC~0.78)": (1.0, 1.5),
        "Model Buruk (AUC~0.55)": (0.2, 2.0),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, (shift, noise) in models.items():
        y_true = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        scores = np.concatenate([
            np.random.normal(0, noise, n // 2),
            np.random.normal(shift, noise, n // 2)
        ])

        # Hitung ROC
        thresholds = np.linspace(scores.min(), scores.max(), 200)
        tpr_list, fpr_list = [], []
        prec_list, rec_list = [], []

        for th in thresholds:
            y_pred = (scores >= th).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1
            rec = tpr

            tpr_list.append(tpr)
            fpr_list.append(fpr)
            prec_list.append(prec)
            rec_list.append(rec)

        auc = np.abs(np.trapz(tpr_list, fpr_list))
        ax1.plot(fpr_list, tpr_list, linewidth=2, label=f"{name} AUC={auc:.2f}")
        ax2.plot(rec_list, prec_list, linewidth=2, label=name)

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.set_xlabel("FPR (False Positive Rate)")
    ax1.set_ylabel("TPR (True Positive Rate)")
    ax1.set_title("Kurva ROC")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Kurva Precision-Recall")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Evaluasi Model: ROC dan Precision-Recall", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_roc_pr_curve.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/18_roc_pr_curve.png")


def visualisasi_metrik_per_kelas(classes, precision, recall, f1):
    """
    Menampilkan perbandingan metrik (Precision, Recall, F1) per kelas
    dalam bentuk bar chart berdampingan.
    """
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    ax.bar(x, recall, width, label='Recall', color='coral')
    ax.bar(x + width, f1, width, label='F1-Score', color='seagreen')

    ax.set_xlabel("Kelas")
    ax.set_ylabel("Skor")
    ax.set_title("Metrik Evaluasi per Kelas")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Anotasi nilai
    for i in range(len(classes)):
        ax.text(i - width, precision[i] + 0.02, f"{precision[i]:.2f}", ha='center', fontsize=7)
        ax.text(i, recall[i] + 0.02, f"{recall[i]:.2f}", ha='center', fontsize=7)
        ax.text(i + width, f1[i] + 0.02, f"{f1[i]:.2f}", ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_metrik_per_kelas.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/18_metrik_per_kelas.png")


def main():
    """Fungsi utama: evaluasi model dan metrik."""
    print("=" * 60)
    print("PERCOBAAN 18: MODEL EVALUASI DAN METRIK")
    print("=" * 60)

    # --- 1. Confusion Matrix ---
    print("\n--- 1. Confusion Matrix ---")
    classes = ["Kucing", "Anjing", "Burung", "Ikan", "Kuda"]
    np.random.seed(42)

    # Simulasi prediksi model
    n_samples = 200
    y_true = np.random.randint(0, len(classes), n_samples)
    y_pred = y_true.copy()
    # Tambah noise: 20% salah prediksi
    wrong = np.random.choice(n_samples, n_samples // 5, replace=False)
    y_pred[wrong] = np.random.randint(0, len(classes), len(wrong))

    cm = hitung_confusion_matrix(y_true, y_pred, len(classes))
    visualisasi_confusion_matrix(cm, classes)

    # --- 2. Hitung metrik ---
    print("\n--- 2. Metrik Evaluasi ---")
    accuracy, precision, recall, f1 = hitung_metrik(cm)
    print(f"  Accuracy keseluruhan: {accuracy:.2%}")
    for i, cls in enumerate(classes):
        print(f"  {cls}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1={f1[i]:.2f}")

    visualisasi_metrik_per_kelas(classes, precision, recall, f1)

    # --- 3. Kurva ROC dan Precision-Recall ---
    print("\n--- 3. Kurva ROC dan Precision-Recall ---")
    visualisasi_roc_curve()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 18")
    print("=" * 60)
    print("""
Metrik evaluasi model deep learning:
1. Confusion Matrix: visualisasi error per kelas
2. Accuracy: proporsi prediksi benar (bisa misleading pada data imbalanced)
3. Precision: ketepatan prediksi positif
4. Recall: kelengkapan deteksi positif
5. F1-Score: harmonic mean dari Precision dan Recall
6. ROC-AUC: performa model pada berbagai threshold
7. Precision-Recall Curve: fokus pada kelas positif

Output: output/18_confusion_matrix.png, output/18_roc_pr_curve.png,
        output/18_metrik_per_kelas.png
""")


if __name__ == "__main__":
    main()
