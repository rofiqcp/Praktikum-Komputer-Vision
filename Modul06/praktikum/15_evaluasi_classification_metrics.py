"""
==========================================================================
PERCOBAAN 15: EVALUASI CLASSIFICATION METRICS
==========================================================================
Program ini mempelajari cara mengevaluasi kinerja sistem klasifikasi
menggunakan berbagai metrik evaluasi. Data prediksi sintetis digunakan
untuk memahami konsep accuracy, precision, recall, F1-score, confusion
matrix, dan berbagai metode averaging (macro, micro, weighted).

Konsep yang dipelajari:
- Confusion matrix: tabel prediksi benar vs salah per kelas
- Accuracy: rasio total prediksi benar terhadap total sampel
- Precision: rasio true positive terhadap semua prediksi positif
- Recall: rasio true positive terhadap semua data positif sebenarnya
- F1-score: harmonic mean dari precision dan recall
- Macro averaging: rata-rata metrik tanpa bobot (equal weight per class)
- Micro averaging: agregasi global TP, FP, FN
- Weighted averaging: rata-rata berbobot berdasarkan jumlah sampel per kelas

Fungsi utama yang dipelajari:
- np.random.seed()                : Menetapkan seed random untuk reprodusibilitas
- np.random.choice()              : Membuat data prediksi sintetis
- np.zeros()                      : Membuat confusion matrix kosong
- plt.imshow() + plt.colorbar()   : Visualisasi heatmap confusion matrix
- plt.bar()                       : Membuat bar chart per-class metrics
- plt.table()                     : Membuat tabel classification report

Hasil: confusion matrix heatmap, classification report, dan per-class bar chart
==========================================================================
"""

# Mengimpor NumPy untuk operasi array dan statistik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor matplotlib gridspec untuk layout subplot yang fleksibel
import matplotlib.gridspec as gridspec

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 15: EVALUASI CLASSIFICATION METRICS")
print("=" * 60)

# ============================================================
# 1. Membuat Data Prediksi Sintetis (5 Kelas, 200 Sampel)
# ============================================================

print("\n[INFO] Membuat data prediksi sintetis...")
print("-" * 50)

# Menetapkan seed random agar hasil dapat direproduksi
np.random.seed(42)

# Mendefinisikan nama kelas klasifikasi (5 kelas)
class_names = ["Wajah", "Kendaraan", "Teks", "Tangan", "Pemandangan"]

# Mendefinisikan jumlah kelas
n_classes = len(class_names)

# Mendefinisikan jumlah total sampel
n_samples = 200

# Membuat label ground truth sintetis dengan distribusi tidak seimbang
# Kelas 0,1 lebih banyak; kelas 3,4 lebih sedikit
gt_probs = [0.30, 0.25, 0.20, 0.15, 0.10]

# Membuat label ground truth berdasarkan distribusi probabilitas
y_true = np.random.choice(n_classes, size=n_samples, p=gt_probs)

# Membuat label prediksi dengan tingkat akurasi yang bervariasi per kelas
y_pred = np.copy(y_true)

# Mendefinisikan probabilitas kesalahan per kelas (semakin besar = semakin sulit)
error_rates = [0.10, 0.15, 0.20, 0.30, 0.35]

# Menambahkan kesalahan prediksi pada setiap kelas
for cls_idx in range(n_classes):
    # Mendapatkan indeks semua sampel yang berasal dari kelas ini
    cls_mask = np.where(y_true == cls_idx)[0]

    # Menghitung jumlah sampel yang akan salah diklasifikasikan
    n_errors = int(len(cls_mask) * error_rates[cls_idx])

    # Memilih sampel secara acak untuk dibuat salah
    error_indices = np.random.choice(cls_mask, size=n_errors, replace=False)

    # Membuat prediksi salah ke kelas lain secara acak
    for idx in error_indices:
        # Membuat daftar kelas selain kelas asli
        other_classes = [c for c in range(n_classes) if c != cls_idx]

        # Mengubah prediksi ke kelas lain secara acak
        y_pred[idx] = np.random.choice(other_classes)

# Menghitung akurasi keseluruhan
overall_accuracy = np.sum(y_true == y_pred) / n_samples

# Menampilkan informasi data sintetis
print(f"  Total sampel: {n_samples}")
print(f"  Jumlah kelas: {n_classes}")
print(f"  Kelas: {class_names}")
print(f"  Distribusi ground truth: {[np.sum(y_true == i) for i in range(n_classes)]}")
print(f"  Distribusi prediksi:     {[np.sum(y_pred == i) for i in range(n_classes)]}")
print(f"  Akurasi keseluruhan: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")

# ============================================================
# 2. Menghitung Confusion Matrix
# ============================================================

print("\n[INFO] Menghitung confusion matrix...")
print("-" * 50)

# Membuat matriks kosong berukuran n_classes x n_classes
confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

# Mengisi confusion matrix dengan menghitung setiap pasangan (true, pred)
for true_label, pred_label in zip(y_true, y_pred):
    # Menambah hitungan pada posisi [true_label, pred_label]
    confusion_matrix[true_label, pred_label] += 1

# Menampilkan confusion matrix
print("  Confusion Matrix:")
print(f"  {'':>12s}", end="")
for name in class_names:
    print(f"  {name[:5]:>6s}", end="")
print()

# Menampilkan setiap baris confusion matrix
for i in range(n_classes):
    print(f"  {class_names[i]:>12s}", end="")
    for j in range(n_classes):
        print(f"  {confusion_matrix[i, j]:>6d}", end="")
    print()

# ============================================================
# 3. Menghitung Precision, Recall, F1-Score per Kelas
# ============================================================

print("\n[INFO] Menghitung metrik per kelas...")
print("-" * 50)

# Membuat array untuk menyimpan metrik per kelas
precision_per_class = np.zeros(n_classes)
recall_per_class = np.zeros(n_classes)
f1_per_class = np.zeros(n_classes)
support_per_class = np.zeros(n_classes, dtype=int)

# Menghitung metrik untuk setiap kelas
for cls_idx in range(n_classes):
    # Menghitung True Positives: prediksi benar untuk kelas ini
    tp = confusion_matrix[cls_idx, cls_idx]

    # Menghitung False Positives: prediksi kelas ini yang sebenarnya bukan
    fp = np.sum(confusion_matrix[:, cls_idx]) - tp

    # Menghitung False Negatives: sebenarnya kelas ini tapi diprediksi lain
    fn = np.sum(confusion_matrix[cls_idx, :]) - tp

    # Menghitung jumlah sampel aktual (support) untuk kelas ini
    support_per_class[cls_idx] = np.sum(confusion_matrix[cls_idx, :])

    # Menghitung precision: TP / (TP + FP)
    if (tp + fp) > 0:
        precision_per_class[cls_idx] = tp / (tp + fp)
    else:
        precision_per_class[cls_idx] = 0.0

    # Menghitung recall: TP / (TP + FN)
    if (tp + fn) > 0:
        recall_per_class[cls_idx] = tp / (tp + fn)
    else:
        recall_per_class[cls_idx] = 0.0

    # Menghitung F1-score: 2 * (precision * recall) / (precision + recall)
    p = precision_per_class[cls_idx]
    r = recall_per_class[cls_idx]
    if (p + r) > 0:
        f1_per_class[cls_idx] = 2 * p * r / (p + r)
    else:
        f1_per_class[cls_idx] = 0.0

    # Menampilkan metrik untuk kelas ini
    print(f"  {class_names[cls_idx]:>12s}: "
          f"Precision={p:.4f}, Recall={r:.4f}, F1={f1_per_class[cls_idx]:.4f}, "
          f"Support={support_per_class[cls_idx]}")

# ============================================================
# 4. Menghitung Macro, Micro, Weighted Averaging
# ============================================================

print("\n[INFO] Menghitung macro, micro, dan weighted averaging...")
print("-" * 50)

# --- Macro averaging: rata-rata sederhana tanpa bobot ---
# Menghitung macro precision (rata-rata precision semua kelas)
macro_precision = np.mean(precision_per_class)

# Menghitung macro recall (rata-rata recall semua kelas)
macro_recall = np.mean(recall_per_class)

# Menghitung macro F1 (rata-rata F1 semua kelas)
macro_f1 = np.mean(f1_per_class)

# Menampilkan hasil macro averaging
print(f"  Macro Averaging:")
print(f"    Precision = {macro_precision:.4f}")
print(f"    Recall    = {macro_recall:.4f}")
print(f"    F1-Score  = {macro_f1:.4f}")

# --- Micro averaging: agregasi global TP, FP, FN ---
# Menghitung total TP, FP, FN secara global
total_tp = 0
total_fp = 0
total_fn = 0

# Mengakumulasi TP, FP, FN dari semua kelas
for cls_idx in range(n_classes):
    # Mengambil TP untuk kelas ini
    tp = confusion_matrix[cls_idx, cls_idx]

    # Menambahkan ke total TP
    total_tp += tp

    # Menghitung dan menambahkan FP untuk kelas ini
    total_fp += np.sum(confusion_matrix[:, cls_idx]) - tp

    # Menghitung dan menambahkan FN untuk kelas ini
    total_fn += np.sum(confusion_matrix[cls_idx, :]) - tp

# Menghitung micro precision: total_TP / (total_TP + total_FP)
micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

# Menghitung micro recall: total_TP / (total_TP + total_FN)
micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

# Menghitung micro F1
micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0 else 0.0)

# Menampilkan hasil micro averaging
print(f"  Micro Averaging:")
print(f"    Precision = {micro_precision:.4f}")
print(f"    Recall    = {micro_recall:.4f}")
print(f"    F1-Score  = {micro_f1:.4f}")

# --- Weighted averaging: rata-rata berbobot berdasarkan support ---
# Menghitung total support
total_support = np.sum(support_per_class)

# Menghitung bobot setiap kelas berdasarkan proporsi sampel
weights = support_per_class / total_support

# Menghitung weighted precision
weighted_precision = np.sum(precision_per_class * weights)

# Menghitung weighted recall
weighted_recall = np.sum(recall_per_class * weights)

# Menghitung weighted F1
weighted_f1 = np.sum(f1_per_class * weights)

# Menampilkan hasil weighted averaging
print(f"  Weighted Averaging:")
print(f"    Precision = {weighted_precision:.4f}")
print(f"    Recall    = {weighted_recall:.4f}")
print(f"    F1-Score  = {weighted_f1:.4f}")

# ============================================================
# 5. Visualisasi 1: Confusion Matrix Heatmap
# ============================================================

print("\n[INFO] Membuat visualisasi confusion matrix...")
print("-" * 50)

# Membuat figure untuk confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel kiri: Confusion matrix (angka mentah) ---
# Menampilkan confusion matrix sebagai heatmap
im1 = axes[0].imshow(confusion_matrix, interpolation='nearest', cmap='Blues')

# Menambahkan colorbar untuk skala warna
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Menambahkan label angka pada setiap sel
for i in range(n_classes):
    for j in range(n_classes):
        # Menentukan warna teks berdasarkan nilai sel
        color = "white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black"

        # Menambahkan teks angka pada posisi (j, i)
        axes[0].text(j, i, str(confusion_matrix[i, j]),
                     ha="center", va="center", color=color, fontsize=12)

# Mengatur label sumbu x (kelas prediksi)
axes[0].set_xticks(range(n_classes))
axes[0].set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)

# Mengatur label sumbu y (kelas sebenarnya)
axes[0].set_yticks(range(n_classes))
axes[0].set_yticklabels(class_names, fontsize=9)

# Menambahkan label sumbu
axes[0].set_xlabel("Prediksi", fontsize=11)
axes[0].set_ylabel("Aktual (Ground Truth)", fontsize=11)
axes[0].set_title("Confusion Matrix (Angka Mentah)", fontsize=12, fontweight='bold')

# --- Panel kanan: Confusion matrix ternormalisasi per baris ---
# Menghitung normalisasi: setiap baris dibagi total baris
row_sums = confusion_matrix.sum(axis=1, keepdims=True)

# Menghindari pembagian dengan nol
row_sums_safe = np.where(row_sums == 0, 1, row_sums)

# Menghitung confusion matrix ternormalisasi
cm_normalized = confusion_matrix.astype(float) / row_sums_safe

# Menampilkan confusion matrix ternormalisasi sebagai heatmap
im2 = axes[1].imshow(cm_normalized, interpolation='nearest', cmap='Oranges',
                     vmin=0, vmax=1)

# Menambahkan colorbar
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# Menambahkan label persentase pada setiap sel
for i in range(n_classes):
    for j in range(n_classes):
        # Menentukan warna teks berdasarkan nilai sel
        color = "white" if cm_normalized[i, j] > 0.5 else "black"

        # Menambahkan teks persentase pada posisi (j, i)
        axes[1].text(j, i, f"{cm_normalized[i, j]:.2f}",
                     ha="center", va="center", color=color, fontsize=11)

# Mengatur label sumbu x dan y
axes[1].set_xticks(range(n_classes))
axes[1].set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
axes[1].set_yticks(range(n_classes))
axes[1].set_yticklabels(class_names, fontsize=9)
axes[1].set_xlabel("Prediksi", fontsize=11)
axes[1].set_ylabel("Aktual (Ground Truth)", fontsize=11)
axes[1].set_title("Confusion Matrix (Ternormalisasi)", fontsize=12, fontweight='bold')

# Menambahkan judul utama
plt.suptitle("Percobaan 15: Confusion Matrix - Evaluasi Klasifikasi",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi confusion matrix
output_path_1 = os.path.join(OUTPUT_DIR, "15_confusion_matrix.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 6. Visualisasi 2: Classification Report Tabel
# ============================================================

print("\n[INFO] Membuat visualisasi classification report...")
print("-" * 50)

# Membuat figure untuk classification report
fig, ax = plt.subplots(figsize=(10, 6))

# Menyembunyikan sumbu karena akan menampilkan tabel
ax.axis('off')

# Menyiapkan data tabel classification report
table_data = []

# Menambahkan baris header
header = ["Kelas", "Precision", "Recall", "F1-Score", "Support"]

# Menambahkan metrik per kelas ke tabel
for cls_idx in range(n_classes):
    # Membuat baris data untuk setiap kelas
    row = [
        class_names[cls_idx],
        f"{precision_per_class[cls_idx]:.4f}",
        f"{recall_per_class[cls_idx]:.4f}",
        f"{f1_per_class[cls_idx]:.4f}",
        str(support_per_class[cls_idx])
    ]
    # Menambahkan baris ke tabel
    table_data.append(row)

# Menambahkan baris separator kosong
table_data.append(["", "", "", "", ""])

# Menambahkan baris macro average
table_data.append([
    "Macro Avg",
    f"{macro_precision:.4f}",
    f"{macro_recall:.4f}",
    f"{macro_f1:.4f}",
    str(total_support)
])

# Menambahkan baris micro average
table_data.append([
    "Micro Avg",
    f"{micro_precision:.4f}",
    f"{micro_recall:.4f}",
    f"{micro_f1:.4f}",
    str(total_support)
])

# Menambahkan baris weighted average
table_data.append([
    "Weighted Avg",
    f"{weighted_precision:.4f}",
    f"{weighted_recall:.4f}",
    f"{weighted_f1:.4f}",
    str(total_support)
])

# Membuat tabel pada axes
table = ax.table(cellText=table_data,
                 colLabels=header,
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.22, 0.18, 0.18, 0.18, 0.14])

# Mengatur ukuran font tabel
table.auto_set_font_size(False)
table.set_fontsize(11)

# Mengatur tinggi baris tabel
table.scale(1.0, 1.5)

# Mengatur warna header tabel
for j in range(len(header)):
    # Mengatur warna background header menjadi biru muda
    table[0, j].set_facecolor('#4472C4')
    # Mengatur warna teks header menjadi putih
    table[0, j].set_text_props(color='white', fontweight='bold')

# Mengatur warna baris data per kelas
for i in range(1, n_classes + 1):
    for j in range(len(header)):
        # Mengatur warna latar bergantian untuk keterbacaan
        if i % 2 == 0:
            table[i, j].set_facecolor('#D6E4F0')
        else:
            table[i, j].set_facecolor('#EDF2F9')

# Mengatur warna baris averaging
for i in range(n_classes + 2, n_classes + 5):
    for j in range(len(header)):
        # Mengatur warna latar kuning muda untuk baris averaging
        table[i, j].set_facecolor('#FFF2CC')

# Menambahkan judul tabel
ax.set_title("Classification Report\n(Evaluasi Metrik Klasifikasi per Kelas)",
             fontsize=14, fontweight='bold', pad=20)

# Menambahkan keterangan accuracy di bawah
ax.text(0.5, -0.02, f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)",
        transform=ax.transAxes, ha='center', fontsize=12, fontstyle='italic')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi classification report
output_path_2 = os.path.join(OUTPUT_DIR, "15_classification_report.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 7. Visualisasi 3: Per-Class Performance Bar Charts
# ============================================================

print("\n[INFO] Membuat visualisasi per-class performance...")
print("-" * 50)

# Membuat figure dengan 2 baris subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Menentukan posisi bar untuk setiap kelas
x_pos = np.arange(n_classes)

# Mendefinisikan lebar bar
bar_width = 0.25

# --- Panel kiri atas: Precision, Recall, F1 bar chart ---
# Menggambar bar precision
bars_p = axes[0, 0].bar(x_pos - bar_width, precision_per_class, bar_width,
                         label='Precision', color='#4472C4', alpha=0.85)

# Menggambar bar recall
bars_r = axes[0, 0].bar(x_pos, recall_per_class, bar_width,
                         label='Recall', color='#ED7D31', alpha=0.85)

# Menggambar bar F1-score
bars_f = axes[0, 0].bar(x_pos + bar_width, f1_per_class, bar_width,
                         label='F1-Score', color='#70AD47', alpha=0.85)

# Menambahkan label nilai di atas setiap bar precision
for bar in bars_p:
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

# Menambahkan label nilai di atas setiap bar recall
for bar in bars_r:
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

# Menambahkan label nilai di atas setiap bar F1
for bar in bars_f:
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

# Mengatur sumbu dan label
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(class_names, fontsize=9)
axes[0, 0].set_ylabel("Skor", fontsize=10)
axes[0, 0].set_title("Precision / Recall / F1 per Kelas", fontsize=11, fontweight='bold')
axes[0, 0].set_ylim(0, 1.15)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(axis='y', alpha=0.3)

# --- Panel kanan atas: Support (jumlah sampel) per kelas ---
# Menggambar bar support
colors_support = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5']
bars_s = axes[0, 1].bar(x_pos, support_per_class, color=colors_support, alpha=0.85)

# Menambahkan label nilai di atas setiap bar support
for bar in bars_s:
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)

# Mengatur sumbu dan label
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(class_names, fontsize=9)
axes[0, 1].set_ylabel("Jumlah Sampel", fontsize=10)
axes[0, 1].set_title("Distribusi Sampel per Kelas (Support)", fontsize=11, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# --- Panel kiri bawah: Perbandingan averaging methods ---
# Mendefinisikan metode averaging
avg_methods = ["Macro", "Micro", "Weighted"]

# Mendefinisikan nilai metrik untuk setiap averaging
avg_precision = [macro_precision, micro_precision, weighted_precision]
avg_recall = [macro_recall, micro_recall, weighted_recall]
avg_f1 = [macro_f1, micro_f1, weighted_f1]

# Menentukan posisi bar untuk averaging methods
x_avg = np.arange(len(avg_methods))

# Menggambar bar precision untuk averaging
axes[1, 0].bar(x_avg - bar_width, avg_precision, bar_width,
               label='Precision', color='#4472C4', alpha=0.85)

# Menggambar bar recall untuk averaging
axes[1, 0].bar(x_avg, avg_recall, bar_width,
               label='Recall', color='#ED7D31', alpha=0.85)

# Menggambar bar F1 untuk averaging
axes[1, 0].bar(x_avg + bar_width, avg_f1, bar_width,
               label='F1-Score', color='#70AD47', alpha=0.85)

# Mengatur sumbu dan label
axes[1, 0].set_xticks(x_avg)
axes[1, 0].set_xticklabels(avg_methods, fontsize=10)
axes[1, 0].set_ylabel("Skor", fontsize=10)
axes[1, 0].set_title("Perbandingan Metode Averaging", fontsize=11, fontweight='bold')
axes[1, 0].set_ylim(0, 1.15)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(axis='y', alpha=0.3)

# --- Panel kanan bawah: Precision-Recall trade-off per kelas ---
# Mensimulasikan precision-recall trade-off dengan variasi threshold
# Kita buat data sintetis: confidence score per prediksi
confidence_scores = np.random.beta(2, 1, size=n_samples)

# Menambahkan noise sedikit agar lebih realistis
confidence_scores = np.clip(confidence_scores, 0.01, 0.99)

# Mendefinisikan threshold yang akan diuji
thresholds = np.linspace(0.1, 0.9, 20)

# Menampilkan precision-recall trade-off per kelas
for cls_idx in range(n_classes):
    # Membuat array untuk menyimpan precision dan recall pada setiap threshold
    prec_at_thresh = []
    rec_at_thresh = []

    # Menghitung precision dan recall pada setiap threshold
    for thresh in thresholds:
        # Menentukan prediksi aktif (confidence >= threshold)
        active_mask = confidence_scores >= thresh

        # Menghitung TP: prediksi aktif, prediksi benar, dan kelas ini
        tp_mask = active_mask & (y_pred == cls_idx) & (y_true == cls_idx)
        tp = np.sum(tp_mask)

        # Menghitung FP: prediksi aktif, prediksi kelas ini, tapi aslinya bukan
        fp_mask = active_mask & (y_pred == cls_idx) & (y_true != cls_idx)
        fp = np.sum(fp_mask)

        # Menghitung FN: kelas ini aslinya, tapi tidak diprediksi kelas ini atau tidak aktif
        fn_mask = (y_true == cls_idx) & (~(active_mask & (y_pred == cls_idx)))
        fn = np.sum(fn_mask)

        # Menghitung precision pada threshold ini
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0

        # Menghitung recall pada threshold ini
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Menyimpan hasil
        prec_at_thresh.append(p)
        rec_at_thresh.append(r)

    # Menggambar kurva precision-recall untuk kelas ini
    axes[1, 1].plot(rec_at_thresh, prec_at_thresh, 'o-', markersize=3,
                    label=class_names[cls_idx], alpha=0.8)

# Mengatur sumbu dan label
axes[1, 1].set_xlabel("Recall", fontsize=10)
axes[1, 1].set_ylabel("Precision", fontsize=10)
axes[1, 1].set_title("Precision-Recall Trade-off per Kelas", fontsize=11, fontweight='bold')
axes[1, 1].set_xlim(-0.05, 1.05)
axes[1, 1].set_ylim(-0.05, 1.15)
axes[1, 1].legend(fontsize=8, loc='lower left')
axes[1, 1].grid(True, alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 15: Evaluasi Classification Metrics",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi per-class metrics
output_path_3 = os.path.join(OUTPUT_DIR, "15_per_class_metrics.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 15")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Confusion Matrix:")
print("     - Baris = kelas aktual (ground truth)")
print("     - Kolom = kelas prediksi")
print("     - Diagonal = prediksi benar (True Positive per kelas)")
print("  2. Metrik per Kelas:")
print("     - Precision = TP / (TP + FP) → keakuratan prediksi positif")
print("     - Recall    = TP / (TP + FN) → kelengkapan deteksi")
print("     - F1-Score  = 2·P·R / (P+R) → harmonic mean P dan R")
print("  3. Metode Averaging:")
print(f"     - Macro   : P={macro_precision:.4f}, R={macro_recall:.4f}, F1={macro_f1:.4f}")
print(f"     - Micro   : P={micro_precision:.4f}, R={micro_recall:.4f}, F1={micro_f1:.4f}")
print(f"     - Weighted: P={weighted_precision:.4f}, R={weighted_recall:.4f}, F1={weighted_f1:.4f}")
print("  4. Macro = rata-rata tanpa bobot (equal per class)")
print("     Micro = agregasi global (TP+FP+FN total)")
print("     Weighted = rata-rata berbobot oleh support")
print(f"\nData sintetis: {n_samples} sampel, {n_classes} kelas")
print(f"Akurasi keseluruhan: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")
print("=" * 60)
