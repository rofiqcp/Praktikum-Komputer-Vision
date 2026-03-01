"""
==========================================================================
PERCOBAAN 18: MODEL EVALUASI METRIK
==========================================================================
Program ini mengimplementasikan dan memvisualisasikan berbagai metrik
evaluasi yang digunakan untuk mengukur performa model deep learning.
Metrik evaluasi sangat penting untuk memahami seberapa baik model
bekerja pada data yang belum pernah dilihat, dan untuk membandingkan
performa antar model secara objektif.

Fungsi utama yang dipelajari:
- Accuracy, Precision, Recall, F1-Score : Metrik klasifikasi
- Confusion Matrix                       : Tabel prediksi vs aktual
- ROC Curve dan AUC                      : Kurva trade-off TPR vs FPR
- Precision-Recall Curve                 : Trade-off precision-recall
- Operasi NumPy: np.sum(), np.unique(), np.argsort()
- Matplotlib: plt.imshow(), plt.plot(), plt.bar()
- sklearn.metrics (opsional, dengan fallback manual)

Konsep yang dipelajari:
- Definisi dan formula setiap metrik evaluasi
- Confusion matrix dan interpretasinya
- ROC curve, AUC, dan threshold selection
- Precision-recall trade-off
- Per-class metrics untuk evaluasi multi-kelas
- Top-K accuracy untuk klasifikasi
==========================================================================
"""

# Mengimpor NumPy untuk operasi komputasi numerik
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi grafik
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mencoba mengimpor sklearn untuk metrik evaluasi
try:
    # Mengimpor metrik dari scikit-learn
    from sklearn.metrics import (confusion_matrix as sk_confusion_matrix,
                                  roc_curve as sk_roc_curve,
                                  auc as sk_auc,
                                  precision_recall_curve as sk_pr_curve)
    SKLEARN_AVAILABLE = True
    print("  [INFO] scikit-learn tersedia, menggunakan sklearn.metrics")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("  [INFO] scikit-learn tidak tersedia, menggunakan implementasi manual")

# Menampilkan header percobaan
print("=" * 60)
print("PERCOBAAN 18: MODEL EVALUASI METRIK")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep metrik evaluasi
# ============================================================
print("\n--- 1. Konsep Metrik Evaluasi ---")

# Menjelaskan konsep metrik evaluasi
print("""
  Metrik evaluasi mengukur performa model pada data test:

  Confusion Matrix:
                    Predicted Pos   Predicted Neg
  Actual Pos          TP                FN
  Actual Neg          FP                TN

  Metrik dasar:
  - Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  - Precision = TP / (TP + FP)    -> "Seberapa tepat prediksi positif"
  - Recall    = TP / (TP + FN)    -> "Seberapa lengkap positif terdeteksi"
  - F1-Score  = 2 * P * R / (P + R) -> Harmonic mean P dan R

  | Metrik     | Kapan penting?                          |
  |------------|-----------------------------------------|
  | Accuracy   | Kelas seimbang                          |
  | Precision  | Minimize false positive (spam filter)   |
  | Recall     | Minimize false negative (diagnosa medis)|
  | F1-Score   | Keseimbangan precision dan recall       |
""")

# ============================================================
# 2. Membuat data prediksi sintetis
# ============================================================
print("\n--- 2. Membuat Data Prediksi Sintetis ---")

# Mengatur random seed untuk reproducibility
np.random.seed(42)

# Mendefinisikan nama kelas (5 kelas)
nama_kelas = ['Kucing', 'Anjing', 'Mobil', 'Bunga', 'Gedung']
n_kelas = len(nama_kelas)

# Membuat label ground truth (200 sampel)
n_sampel = 200
y_true = np.random.randint(0, n_kelas, n_sampel)

# Membuat prediksi untuk Model A (akurasi ~75%)
y_pred_A = y_true.copy()
# Menambahkan noise pada 25% prediksi
n_salah_A = int(0.25 * n_sampel)
idx_salah_A = np.random.choice(n_sampel, n_salah_A, replace=False)
for idx in idx_salah_A:
    # Memilih kelas yang salah secara acak
    kelas_salah = np.random.choice([c for c in range(n_kelas) if c != y_true[idx]])
    y_pred_A[idx] = kelas_salah

# Membuat prediksi untuk Model B (akurasi ~60%)
y_pred_B = y_true.copy()
n_salah_B = int(0.40 * n_sampel)
idx_salah_B = np.random.choice(n_sampel, n_salah_B, replace=False)
for idx in idx_salah_B:
    # Memilih kelas yang salah
    kelas_salah = np.random.choice([c for c in range(n_kelas) if c != y_true[idx]])
    y_pred_B[idx] = kelas_salah

# Membuat probabilitas prediksi untuk ROC/PR curve (kasus biner)
# Mengambil kelas 0 vs rest
y_true_biner = (y_true == 0).astype(int)

# Membuat skor probabilitas untuk Model A
scores_A = np.random.rand(n_sampel)
# Meningkatkan skor untuk sampel positif (model yang cukup baik)
scores_A[y_true_biner == 1] += np.random.uniform(0.2, 0.5, np.sum(y_true_biner == 1))
scores_A = np.clip(scores_A, 0, 1)

# Membuat skor probabilitas untuk Model B (model yang lebih buruk)
scores_B = np.random.rand(n_sampel)
scores_B[y_true_biner == 1] += np.random.uniform(0.0, 0.3, np.sum(y_true_biner == 1))
scores_B = np.clip(scores_B, 0, 1)

# Menampilkan informasi data
print(f"  Total sampel       : {n_sampel}")
print(f"  Jumlah kelas       : {n_kelas}")
print(f"  Distribusi kelas   : {[int(np.sum(y_true == c)) for c in range(n_kelas)]}")
print(f"  Model A salah      : {n_salah_A} sampel")
print(f"  Model B salah      : {n_salah_B} sampel")

# ============================================================
# 3. Implementasi metrik evaluasi dari nol
# ============================================================
print("\n--- 3. Implementasi Metrik Evaluasi ---")


def hitung_confusion_matrix(y_true, y_pred, n_classes):
    """
    Menghitung confusion matrix secara manual.
    """
    # Menginisialisasi matriks nol
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Mengisi confusion matrix
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Mengembalikan confusion matrix
    return cm


def hitung_accuracy(y_true, y_pred):
    """
    Menghitung accuracy: proporsi prediksi yang benar.
    """
    # Menghitung jumlah prediksi benar
    benar = np.sum(y_true == y_pred)

    # Menghitung total sampel
    total = len(y_true)

    # Mengembalikan accuracy
    return benar / total


def hitung_precision_recall_f1(cm, kelas):
    """
    Menghitung precision, recall, dan F1-score untuk kelas tertentu.
    """
    # Menghitung True Positive
    tp = cm[kelas, kelas]

    # Menghitung False Positive (prediksi kelas ini tapi sebenarnya bukan)
    fp = np.sum(cm[:, kelas]) - tp

    # Menghitung False Negative (sebenarnya kelas ini tapi prediksi salah)
    fn = np.sum(cm[kelas, :]) - tp

    # Menghitung precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Menghitung recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Menghitung F1-score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Mengembalikan metrik
    return precision, recall, f1


def hitung_roc_curve(y_true_bin, scores, n_thresholds=200):
    """
    Menghitung ROC curve secara manual.
    """
    # Membuat range threshold
    thresholds = np.linspace(0, 1, n_thresholds)

    # Menyiapkan list untuk TPR dan FPR
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        # Membuat prediksi biner berdasarkan threshold
        y_pred_bin = (scores >= thresh).astype(int)

        # Menghitung TP, FP, TN, FN
        tp = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
        fp = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
        tn = np.sum((y_pred_bin == 0) & (y_true_bin == 0))
        fn = np.sum((y_pred_bin == 0) & (y_true_bin == 1))

        # Menghitung TPR (True Positive Rate / Recall)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Menghitung FPR (False Positive Rate)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Menyimpan nilai
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Mengembalikan FPR, TPR, dan thresholds
    return np.array(fpr_list), np.array(tpr_list), thresholds


def hitung_auc(fpr, tpr):
    """
    Menghitung Area Under Curve (AUC) menggunakan trapezoidal rule.
    """
    # Mengurutkan berdasarkan FPR
    sorted_idx = np.argsort(fpr)
    fpr_sorted = fpr[sorted_idx]
    tpr_sorted = tpr[sorted_idx]

    # Menghitung AUC dengan trapezoidal rule
    auc_val = np.trapz(tpr_sorted, fpr_sorted)

    # Mengembalikan nilai AUC
    return abs(auc_val)


def hitung_pr_curve(y_true_bin, scores, n_thresholds=200):
    """
    Menghitung Precision-Recall curve secara manual.
    """
    # Membuat range threshold
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    # Menyiapkan list
    prec_list = []
    rec_list = []

    for thresh in thresholds:
        # Membuat prediksi biner
        y_pred_bin = (scores >= thresh).astype(int)

        # Menghitung TP dan FP
        tp = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
        fp = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
        fn = np.sum((y_pred_bin == 0) & (y_true_bin == 1))

        # Menghitung precision dan recall
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Menyimpan nilai
        prec_list.append(prec)
        rec_list.append(rec)

    # Mengembalikan precision, recall, thresholds
    return np.array(prec_list), np.array(rec_list), thresholds


# Menghitung confusion matrix untuk kedua model
cm_A = hitung_confusion_matrix(y_true, y_pred_A, n_kelas)
cm_B = hitung_confusion_matrix(y_true, y_pred_B, n_kelas)

# Menghitung accuracy
acc_A = hitung_accuracy(y_true, y_pred_A)
acc_B = hitung_accuracy(y_true, y_pred_B)

# Menampilkan hasil
print(f"  Accuracy Model A: {acc_A:.4f} ({acc_A*100:.1f}%)")
print(f"  Accuracy Model B: {acc_B:.4f} ({acc_B*100:.1f}%)")

# ============================================================
# 4. Menghitung per-class metrics
# ============================================================
print("\n--- 4. Per-Class Metrics ---")

# Menampilkan header tabel
print(f"\n  Model A:")
print(f"  {'Kelas':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print(f"  {'-'*48}")

# Menyiapkan list untuk menyimpan metrik per kelas
prec_A_list = []
rec_A_list = []
f1_A_list = []

# Menghitung metrik untuk setiap kelas Model A
for c in range(n_kelas):
    # Menghitung precision, recall, F1 untuk kelas c
    p, r, f1 = hitung_precision_recall_f1(cm_A, c)

    # Menyimpan metrik
    prec_A_list.append(p)
    rec_A_list.append(r)
    f1_A_list.append(f1)

    # Menampilkan metrik
    print(f"  {nama_kelas[c]:<12} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")

# Menghitung rata-rata (macro average)
macro_prec_A = np.mean(prec_A_list)
macro_rec_A = np.mean(rec_A_list)
macro_f1_A = np.mean(f1_A_list)
print(f"  {'Macro Avg':<12} {macro_prec_A:<12.4f} {macro_rec_A:<12.4f} {macro_f1_A:<12.4f}")

# Menampilkan metrik Model B
print(f"\n  Model B:")
print(f"  {'Kelas':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print(f"  {'-'*48}")

# Menyiapkan list untuk Model B
prec_B_list = []
rec_B_list = []
f1_B_list = []

# Menghitung metrik untuk setiap kelas Model B
for c in range(n_kelas):
    # Menghitung precision, recall, F1 untuk kelas c
    p, r, f1 = hitung_precision_recall_f1(cm_B, c)

    # Menyimpan metrik
    prec_B_list.append(p)
    rec_B_list.append(r)
    f1_B_list.append(f1)

    # Menampilkan metrik
    print(f"  {nama_kelas[c]:<12} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")

# Menghitung macro average Model B
macro_prec_B = np.mean(prec_B_list)
macro_rec_B = np.mean(rec_B_list)
macro_f1_B = np.mean(f1_B_list)
print(f"  {'Macro Avg':<12} {macro_prec_B:<12.4f} {macro_rec_B:<12.4f} {macro_f1_B:<12.4f}")

# ============================================================
# 5. Top-K Accuracy
# ============================================================
print("\n--- 5. Top-K Accuracy ---")


def hitung_topk_accuracy(y_true, pred_probs, k=1):
    """
    Menghitung Top-K accuracy.
    Prediksi dianggap benar jika kelas benar ada di K prediksi teratas.
    """
    # Menghitung jumlah benar
    benar = 0
    total = len(y_true)

    for i in range(total):
        # Mendapatkan top-K indeks prediksi
        topk_idx = np.argsort(pred_probs[i])[-k:]

        # Mengecek apakah kelas benar ada di top-K
        if y_true[i] in topk_idx:
            benar += 1

    # Mengembalikan Top-K accuracy
    return benar / total


# Membuat probabilitas prediksi sintetis untuk Top-K
np.random.seed(55)
pred_probs_full = np.random.dirichlet(np.ones(n_kelas), n_sampel)

# Meningkatkan probabilitas kelas yang benar
for i in range(n_sampel):
    pred_probs_full[i, y_true[i]] += np.random.uniform(0.3, 0.8)

# Menormalisasi ulang agar total = 1
pred_probs_full = pred_probs_full / pred_probs_full.sum(axis=1, keepdims=True)

# Menghitung Top-K accuracy untuk K = 1, 2, 3, 5
for k in [1, 2, 3, 5]:
    # Menghitung Top-K accuracy
    topk_acc = hitung_topk_accuracy(y_true, pred_probs_full, k=k)
    print(f"  Top-{k} Accuracy: {topk_acc:.4f} ({topk_acc*100:.1f}%)")

# ============================================================
# 6. Visualisasi Confusion Matrix (Gambar 1)
# ============================================================
print("\n--- 6. Visualisasi Confusion Matrix ---")

# Membuat figure untuk confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Memberikan judul utama
fig.suptitle("Confusion Matrix: Model A vs Model B",
             fontsize=16, fontweight='bold')

# --- Subplot 1: Confusion Matrix Model A ---
ax1 = axes[0]

# Menampilkan confusion matrix sebagai heatmap
im1 = ax1.imshow(cm_A, interpolation='nearest', cmap='Blues')

# Menambahkan colorbar
plt.colorbar(im1, ax=ax1)

# Menambahkan teks angka di setiap sel
for i in range(n_kelas):
    for j in range(n_kelas):
        # Menentukan warna teks berdasarkan intensitas
        warna_teks = 'white' if cm_A[i, j] > cm_A.max() / 2 else 'black'

        # Menambahkan teks
        ax1.text(j, i, str(cm_A[i, j]), ha='center', va='center',
                 color=warna_teks, fontsize=12, fontweight='bold')

# Mengatur label tick
ax1.set_xticks(range(n_kelas))
ax1.set_yticks(range(n_kelas))
ax1.set_xticklabels(nama_kelas, rotation=45, ha='right', fontsize=9)
ax1.set_yticklabels(nama_kelas, fontsize=9)

# Mengatur judul dan label
ax1.set_title(f"Model A (Acc={acc_A:.1%})", fontsize=12, fontweight='bold')
ax1.set_xlabel("Prediksi")
ax1.set_ylabel("Aktual")

# --- Subplot 2: Confusion Matrix Model B ---
ax2 = axes[1]

# Menampilkan confusion matrix Model B
im2 = ax2.imshow(cm_B, interpolation='nearest', cmap='Oranges')

# Menambahkan colorbar
plt.colorbar(im2, ax=ax2)

# Menambahkan teks angka di setiap sel
for i in range(n_kelas):
    for j in range(n_kelas):
        # Menentukan warna teks
        warna_teks = 'white' if cm_B[i, j] > cm_B.max() / 2 else 'black'

        # Menambahkan teks
        ax2.text(j, i, str(cm_B[i, j]), ha='center', va='center',
                 color=warna_teks, fontsize=12, fontweight='bold')

# Mengatur label tick
ax2.set_xticks(range(n_kelas))
ax2.set_yticks(range(n_kelas))
ax2.set_xticklabels(nama_kelas, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(nama_kelas, fontsize=9)

# Mengatur judul dan label
ax2.set_title(f"Model B (Acc={acc_B:.1%})", fontsize=12, fontweight='bold')
ax2.set_xlabel("Prediksi")
ax2.set_ylabel("Aktual")

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_1 = os.path.join(OUTPUT_DIR, "18_confusion_matrix.png")

# Menyimpan figure
plt.savefig(output_path_1, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_1}")

# ============================================================
# 7. Visualisasi ROC Curve (Gambar 2)
# ============================================================
print("\n--- 7. Visualisasi ROC Curve ---")

# Menghitung ROC curve untuk Model A
fpr_A, tpr_A, _ = hitung_roc_curve(y_true_biner, scores_A)

# Menghitung AUC untuk Model A
auc_A = hitung_auc(fpr_A, tpr_A)

# Menghitung ROC curve untuk Model B
fpr_B, tpr_B, _ = hitung_roc_curve(y_true_biner, scores_B)

# Menghitung AUC untuk Model B
auc_B = hitung_auc(fpr_B, tpr_B)

# Membuat figure untuk ROC curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Memberikan judul utama
fig.suptitle("ROC Curve dan Area Under Curve (AUC)",
             fontsize=16, fontweight='bold')

# --- Subplot 1: ROC curves ---
ax1 = axes[0]

# Menggambar ROC curve Model A
ax1.plot(fpr_A, tpr_A, 'b-', linewidth=2, label=f'Model A (AUC={auc_A:.3f})')

# Menggambar ROC curve Model B
ax1.plot(fpr_B, tpr_B, 'r-', linewidth=2, label=f'Model B (AUC={auc_B:.3f})')

# Menggambar garis diagonal (random classifier)
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.5)')

# Mengatur judul dan label
ax1.set_title("ROC Curve", fontsize=12, fontweight='bold')
ax1.set_xlabel("False Positive Rate (FPR)")
ax1.set_ylabel("True Positive Rate (TPR)")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)

# --- Subplot 2: AUC comparison bar chart ---
ax2 = axes[1]

# Menggambar bar chart AUC
models = ['Model A', 'Model B', 'Random']
auc_vals = [auc_A, auc_B, 0.5]
colors = ['steelblue', 'indianred', 'gray']

# Menggambar bar
bars = ax2.bar(models, auc_vals, color=colors)

# Menambahkan label nilai
for bar, val in zip(bars, auc_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Menambahkan garis horizontal untuk AUC ideal
ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (1.0)')
ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')

# Mengatur judul dan label
ax2.set_title("Perbandingan AUC", fontsize=12, fontweight='bold')
ax2.set_ylabel("AUC Score")
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Menampilkan informasi AUC
print(f"  AUC Model A: {auc_A:.4f}")
print(f"  AUC Model B: {auc_B:.4f}")

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_2 = os.path.join(OUTPUT_DIR, "18_roc_curve.png")

# Menyimpan figure
plt.savefig(output_path_2, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_2}")

# ============================================================
# 8. Visualisasi Precision-Recall Curve (Gambar 3)
# ============================================================
print("\n--- 8. Visualisasi Precision-Recall Curve ---")

# Menghitung PR curve untuk Model A
prec_A_curve, rec_A_curve, _ = hitung_pr_curve(y_true_biner, scores_A)

# Menghitung PR curve untuk Model B
prec_B_curve, rec_B_curve, _ = hitung_pr_curve(y_true_biner, scores_B)

# Menghitung AP (Average Precision) secara sederhana
ap_A = np.mean(prec_A_curve)
ap_B = np.mean(prec_B_curve)

# Membuat figure untuk PR curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Memberikan judul utama
fig.suptitle("Precision-Recall Curve", fontsize=16, fontweight='bold')

# --- Subplot 1: PR curves ---
ax1 = axes[0]

# Menggambar PR curve Model A
ax1.plot(rec_A_curve, prec_A_curve, 'b-', linewidth=2,
         label=f'Model A (AP={ap_A:.3f})')

# Menggambar PR curve Model B
ax1.plot(rec_B_curve, prec_B_curve, 'r-', linewidth=2,
         label=f'Model B (AP={ap_B:.3f})')

# Menggambar garis baseline (proporsi positif)
baseline = np.mean(y_true_biner)
ax1.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5,
            label=f'Baseline ({baseline:.3f})')

# Mengatur judul dan label
ax1.set_title("Precision-Recall Curve", fontsize=12, fontweight='bold')
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.05)

# --- Subplot 2: Precision dan Recall vs Threshold ---
ax2 = axes[1]

# Membuat range threshold
thresh_range = np.linspace(0.01, 0.99, 100)

# Menghitung precision dan recall untuk setiap threshold (Model A)
prec_vs_thresh = []
rec_vs_thresh = []

for thresh in thresh_range:
    # Prediksi biner
    y_pred_th = (scores_A >= thresh).astype(int)

    # Menghitung TP, FP, FN
    tp = np.sum((y_pred_th == 1) & (y_true_biner == 1))
    fp = np.sum((y_pred_th == 1) & (y_true_biner == 0))
    fn = np.sum((y_pred_th == 0) & (y_true_biner == 1))

    # Menghitung precision dan recall
    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    prec_vs_thresh.append(p)
    rec_vs_thresh.append(r)

# Menggambar precision vs threshold
ax2.plot(thresh_range, prec_vs_thresh, 'b-', linewidth=2, label='Precision')

# Menggambar recall vs threshold
ax2.plot(thresh_range, rec_vs_thresh, 'r-', linewidth=2, label='Recall')

# Mencari threshold optimal (F1 tertinggi)
f1_by_thresh = []
for p, r in zip(prec_vs_thresh, rec_vs_thresh):
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    f1_by_thresh.append(f1)

# Menggambar F1 vs threshold
ax2.plot(thresh_range, f1_by_thresh, 'g--', linewidth=2, label='F1-Score')

# Menandai threshold optimal
idx_best = np.argmax(f1_by_thresh)
ax2.axvline(x=thresh_range[idx_best], color='purple', linestyle=':',
            linewidth=2, label=f'Best thresh={thresh_range[idx_best]:.2f}')

# Mengatur judul dan label
ax2.set_title("Metrik vs Threshold (Model A)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Score")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_3 = os.path.join(OUTPUT_DIR, "18_precision_recall.png")

# Menyimpan figure
plt.savefig(output_path_3, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_3}")

# ============================================================
# 9. Perbandingan metrik antar model (Gambar 4)
# ============================================================
print("\n--- 9. Perbandingan Metrik Antar Model ---")

# Membuat figure untuk perbandingan
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Memberikan judul utama
fig.suptitle("Perbandingan Metrik Evaluasi: Model A vs Model B",
             fontsize=16, fontweight='bold')

# --- Subplot 1: Per-class F1 comparison ---
ax1 = axes[0, 0]

# Mendefinisikan posisi bar
x_pos = np.arange(n_kelas)
bar_width = 0.35

# Menggambar bar F1 Model A
ax1.bar(x_pos - bar_width / 2, f1_A_list, bar_width,
        label='Model A', color='steelblue')

# Menggambar bar F1 Model B
ax1.bar(x_pos + bar_width / 2, f1_B_list, bar_width,
        label='Model B', color='indianred')

# Mengatur label tick
ax1.set_xticks(x_pos)
ax1.set_xticklabels(nama_kelas, fontsize=9)

# Mengatur judul dan label
ax1.set_title("F1-Score per Kelas", fontsize=11, fontweight='bold')
ax1.set_ylabel("F1-Score")
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1.1)

# --- Subplot 2: All metrics comparison (macro) ---
ax2 = axes[0, 1]

# Mendefinisikan metrik yang dibandingkan
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
model_A_metrics = [acc_A, macro_prec_A, macro_rec_A, macro_f1_A]
model_B_metrics = [acc_B, macro_prec_B, macro_rec_B, macro_f1_B]

# Mendefinisikan posisi bar
x_met = np.arange(len(metric_names))

# Menggambar bar Model A
ax2.bar(x_met - bar_width / 2, model_A_metrics, bar_width,
        label='Model A', color='steelblue')

# Menggambar bar Model B
ax2.bar(x_met + bar_width / 2, model_B_metrics, bar_width,
        label='Model B', color='indianred')

# Menambahkan nilai di atas bar
for i in range(len(metric_names)):
    ax2.text(x_met[i] - bar_width / 2, model_A_metrics[i] + 0.02,
             f'{model_A_metrics[i]:.2f}', ha='center', fontsize=8)
    ax2.text(x_met[i] + bar_width / 2, model_B_metrics[i] + 0.02,
             f'{model_B_metrics[i]:.2f}', ha='center', fontsize=8)

# Mengatur label tick
ax2.set_xticks(x_met)
ax2.set_xticklabels(metric_names, fontsize=9)

# Mengatur judul dan label
ax2.set_title("Macro-Average Metrics", fontsize=11, fontweight='bold')
ax2.set_ylabel("Score")
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.15)

# --- Subplot 3: Per-class Precision vs Recall ---
ax3 = axes[1, 0]

# Menggambar scatter plot precision vs recall per kelas (Model A)
for c in range(n_kelas):
    ax3.scatter(rec_A_list[c], prec_A_list[c], s=120, zorder=5,
                label=f'{nama_kelas[c]} (A)')

# Menggambar scatter plot untuk Model B dengan marker berbeda
for c in range(n_kelas):
    ax3.scatter(rec_B_list[c], prec_B_list[c], s=80, marker='x',
                zorder=5, linewidths=2)

# Menggambar garis iso-F1
f1_levels = [0.3, 0.5, 0.7, 0.9]
for f1_val in f1_levels:
    # Menghitung kurva iso-F1
    recall_range = np.linspace(0.01, 1.0, 100)
    precision_iso = f1_val * recall_range / (2 * recall_range - f1_val)
    valid = (precision_iso > 0) & (precision_iso <= 1)
    ax3.plot(recall_range[valid], precision_iso[valid], '--', alpha=0.3,
             color='gray')
    # Menambahkan label F1
    if np.any(valid):
        mid_idx = np.sum(valid) // 2
        ax3.annotate(f'F1={f1_val}', xy=(recall_range[valid][mid_idx],
                     precision_iso[valid][mid_idx]), fontsize=7, alpha=0.5)

# Mengatur judul dan label
ax3.set_title("Precision vs Recall per Kelas", fontsize=11, fontweight='bold')
ax3.set_xlabel("Recall")
ax3.set_ylabel("Precision")
ax3.legend(fontsize=7, loc='lower left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.05, 1.1)
ax3.set_ylim(-0.05, 1.1)

# --- Subplot 4: Tabel ringkasan ---
ax4 = axes[1, 1]

# Menyembunyikan axes
ax4.axis('off')

# Membuat data tabel
tabel_data = [
    ['Accuracy', f'{acc_A:.3f}', f'{acc_B:.3f}'],
    ['Precision (macro)', f'{macro_prec_A:.3f}', f'{macro_prec_B:.3f}'],
    ['Recall (macro)', f'{macro_rec_A:.3f}', f'{macro_rec_B:.3f}'],
    ['F1-Score (macro)', f'{macro_f1_A:.3f}', f'{macro_f1_B:.3f}'],
    ['AUC (biner)', f'{auc_A:.3f}', f'{auc_B:.3f}'],
]

# Mendefinisikan header
kolom_header = ['Metrik', 'Model A', 'Model B']

# Menggambar tabel
tabel = ax4.table(cellText=tabel_data, colLabels=kolom_header,
                  loc='center', cellLoc='center')

# Mengatur font
tabel.auto_set_font_size(False)
tabel.set_fontsize(11)

# Mengatur skala tabel
tabel.scale(1.0, 2.0)

# Mewarnai header
for j in range(len(kolom_header)):
    tabel[0, j].set_facecolor('#2196F3')
    tabel[0, j].set_text_props(color='white', fontweight='bold')

# Mewarnai sel yang lebih baik
for i in range(1, len(tabel_data) + 1):
    val_A = float(tabel_data[i - 1][1])
    val_B = float(tabel_data[i - 1][2])
    if val_A > val_B:
        tabel[i, 1].set_facecolor('#C8E6C9')
    elif val_B > val_A:
        tabel[i, 2].set_facecolor('#C8E6C9')

# Mengatur judul
ax4.set_title("Ringkasan Perbandingan Model", fontsize=11, fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_4 = os.path.join(OUTPUT_DIR, "18_metrik_perbandingan.png")

# Menyimpan figure
plt.savefig(output_path_4, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_4}")

# ============================================================
# 10. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 18")
print("=" * 60)
print(f"""
Konsep yang telah dipelajari:
1. Accuracy mengukur proporsi prediksi benar secara keseluruhan
2. Precision mengukur ketepatan prediksi positif
3. Recall mengukur kelengkapan deteksi positif
4. F1-Score adalah harmonic mean dari precision dan recall
5. Confusion matrix menunjukkan detail prediksi vs aktual per kelas
6. ROC curve menggambarkan trade-off TPR vs FPR di berbagai threshold
7. AUC meringkas performa keseluruhan model (ROC) dalam satu angka
8. Top-K accuracy berguna saat K prediksi teratas dipertimbangkan

Hasil Evaluasi:
- Model A: Acc={acc_A:.3f}, F1={macro_f1_A:.3f}, AUC={auc_A:.3f}
- Model B: Acc={acc_B:.3f}, F1={macro_f1_B:.3f}, AUC={auc_B:.3f}

Output disimpan di folder: output/
- 18_confusion_matrix.png     : Confusion matrix Model A dan B
- 18_roc_curve.png            : ROC curve dan AUC
- 18_precision_recall.png     : Precision-recall curve
- 18_metrik_perbandingan.png  : Perbandingan semua metrik
""")
