"""
==========================================================================
PERCOBAAN 16: EVALUASI DETECTION METRICS (IoU, mAP)
==========================================================================
Program ini mempelajari cara mengevaluasi kinerja sistem deteksi objek
menggunakan metrik IoU (Intersection over Union), Precision-Recall curve,
Average Precision (AP), dan mean Average Precision (mAP). Data bounding
box sintetis digunakan untuk memahami konsep evaluasi deteksi objek.

Konsep yang dipelajari:
- IoU (Intersection over Union): ukuran overlap antara 2 bounding box
- True Positive / False Positive pada deteksi (berdasarkan IoU threshold)
- Precision-Recall curve pada berbagai confidence threshold
- AP (Average Precision): area di bawah kurva PR
- mAP (mean Average Precision): rata-rata AP semua kelas
- Pengaruh IoU threshold terhadap hasil evaluasi

Fungsi utama yang dipelajari:
- Implementasi IoU dari awal (manual calculation)
- np.maximum() / np.minimum()    : Kalkulasi intersection area
- np.argsort()                    : Mengurutkan prediksi berdasarkan confidence
- np.cumsum()                     : Kumulatif sum untuk PR curve
- np.trapz()                      : Integrasi numerik untuk AP
- plt.fill_between()              : Visualisasi area di bawah kurva

Hasil: Visualisasi IoU, precision-recall curve, dan mAP chart
==========================================================================
"""

# Mengimpor NumPy untuk operasi array dan komputasi numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor matplotlib patches untuk menggambar bounding box
import matplotlib.patches as patches

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 16: EVALUASI DETECTION METRICS (IoU, mAP)")
print("=" * 60)

# ============================================================
# 1. Implementasi Fungsi IoU (Intersection over Union)
# ============================================================

print("\n[INFO] Mengimplementasikan fungsi IoU dari awal...")
print("-" * 50)


# Mendefinisikan fungsi untuk menghitung IoU antara dua bounding box
def hitung_iou(box_a, box_b):
    """
    Menghitung Intersection over Union antara dua bounding box.
    Format box: [x1, y1, x2, y2] (koordinat kiri-atas dan kanan-bawah).
    """
    # Menghitung koordinat intersection (area overlap)
    # Koordinat x kiri dari intersection = max dari kedua x1
    x_inter_left = max(box_a[0], box_b[0])

    # Koordinat y atas dari intersection = max dari kedua y1
    y_inter_top = max(box_a[1], box_b[1])

    # Koordinat x kanan dari intersection = min dari kedua x2
    x_inter_right = min(box_a[2], box_b[2])

    # Koordinat y bawah dari intersection = min dari kedua y2
    y_inter_bottom = min(box_a[3], box_b[3])

    # Menghitung lebar intersection (jika positif berarti ada overlap)
    inter_width = max(0, x_inter_right - x_inter_left)

    # Menghitung tinggi intersection
    inter_height = max(0, y_inter_bottom - y_inter_top)

    # Menghitung luas area intersection
    inter_area = inter_width * inter_height

    # Menghitung luas box A = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])

    # Menghitung luas box B
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Menghitung luas union = area_a + area_b - intersection
    union_area = area_a + area_b - inter_area

    # Menghitung IoU = intersection / union (hindari pembagian nol)
    iou = inter_area / union_area if union_area > 0 else 0.0

    # Mengembalikan nilai IoU, area intersection, dan area union
    return iou, inter_area, union_area


# Membuat contoh bounding box untuk demonstrasi IoU
print("  Contoh perhitungan IoU:")

# Mendefinisikan pasangan bounding box dengan berbagai tingkat overlap
iou_examples = [
    {"name": "Overlap tinggi", "box_a": [50, 50, 200, 200],
     "box_b": [70, 60, 210, 210]},
    {"name": "Overlap sedang", "box_a": [50, 50, 200, 200],
     "box_b": [130, 120, 300, 300]},
    {"name": "Overlap rendah", "box_a": [50, 50, 200, 200],
     "box_b": [180, 170, 350, 350]},
    {"name": "Tidak overlap", "box_a": [50, 50, 200, 200],
     "box_b": [250, 250, 400, 400]},
    {"name": "Box identik", "box_a": [100, 100, 250, 250],
     "box_b": [100, 100, 250, 250]},
    {"name": "Box di dalam box", "box_a": [50, 50, 300, 300],
     "box_b": [100, 100, 200, 200]},
]

# Menghitung IoU untuk setiap pasangan contoh
for ex in iou_examples:
    # Menghitung IoU menggunakan fungsi yang dibuat
    iou_val, inter_a, union_a = hitung_iou(ex["box_a"], ex["box_b"])

    # Menyimpan hasil IoU ke dalam dictionary
    ex["iou"] = iou_val
    ex["inter"] = inter_a
    ex["union"] = union_a

    # Menampilkan hasil perhitungan
    print(f"    {ex['name']:>20s}: IoU = {iou_val:.4f} "
          f"(intersection={inter_a}, union={union_a})")

# ============================================================
# 2. Visualisasi Contoh IoU
# ============================================================

print("\n[INFO] Membuat visualisasi IoU examples...")
print("-" * 50)

# Membuat figure dengan 2x3 subplot untuk 6 contoh IoU
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Meratakan array axes untuk iterasi mudah
axes_flat = axes.flatten()

# Mendefinisikan warna untuk bounding box
color_a = '#4472C4'  # Biru untuk ground truth
color_b = '#ED7D31'  # Oranye untuk prediksi

# Menampilkan setiap contoh IoU
for idx, ex in enumerate(iou_examples):
    # Mengambil axes untuk subplot ini
    ax = axes_flat[idx]

    # Mengatur batas axes
    ax.set_xlim(0, 420)
    ax.set_ylim(420, 0)

    # Mengambil bounding box A dan B
    ba = ex["box_a"]
    bb = ex["box_b"]

    # Menggambar bounding box A (ground truth) sebagai rectangle
    rect_a = patches.Rectangle((ba[0], ba[1]), ba[2] - ba[0], ba[3] - ba[1],
                                linewidth=2, edgecolor=color_a,
                                facecolor=color_a, alpha=0.25)
    ax.add_patch(rect_a)

    # Menggambar border bounding box A
    rect_a_border = patches.Rectangle((ba[0], ba[1]), ba[2] - ba[0], ba[3] - ba[1],
                                       linewidth=2, edgecolor=color_a,
                                       facecolor='none')
    ax.add_patch(rect_a_border)

    # Menggambar bounding box B (prediksi) sebagai rectangle
    rect_b = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                                linewidth=2, edgecolor=color_b,
                                facecolor=color_b, alpha=0.25)
    ax.add_patch(rect_b)

    # Menggambar border bounding box B
    rect_b_border = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                                       linewidth=2, edgecolor=color_b,
                                       facecolor='none', linestyle='--')
    ax.add_patch(rect_b_border)

    # Menambahkan teks IoU di tengah gambar
    ax.text(210, 380, f"IoU = {ex['iou']:.4f}", ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    # Mengatur judul subplot
    ax.set_title(f"{ex['name']}", fontsize=11, fontweight='bold')

    # Mengatur grid
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

# Menambahkan legend manual
axes_flat[0].plot([], [], color=color_a, linewidth=3, label='Ground Truth (A)')
axes_flat[0].plot([], [], color=color_b, linewidth=3, linestyle='--', label='Prediksi (B)')
axes_flat[0].legend(loc='upper right', fontsize=8)

# Menambahkan judul utama
plt.suptitle("Percobaan 16: Visualisasi IoU (Intersection over Union)",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi IoU
output_path_1 = os.path.join(OUTPUT_DIR, "16_iou_visualisasi.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 3. Membuat Data Deteksi Sintetis (Ground Truth + Prediksi)
# ============================================================

print("\n[INFO] Membuat data deteksi sintetis...")
print("-" * 50)

# Menetapkan seed random untuk reprodusibilitas
np.random.seed(42)

# Mendefinisikan nama kelas untuk deteksi
det_class_names = ["Wajah", "Kendaraan", "Tangan"]

# Mendefinisikan jumlah kelas deteksi
n_det_classes = len(det_class_names)

# Mendefinisikan ukuran gambar sintetis
img_w, img_h = 640, 480

# Membuat ground truth bounding boxes untuk setiap kelas
ground_truths = {}

# Membuat prediksi bounding boxes untuk setiap kelas
predictions = {}

# Menghasilkan data ground truth dan prediksi untuk setiap kelas
for cls_idx, cls_name in enumerate(det_class_names):
    # Menentukan jumlah ground truth untuk kelas ini (5-15)
    n_gt = np.random.randint(8, 16)

    # Membuat bounding box ground truth secara acak
    gt_boxes = []
    for _ in range(n_gt):
        # Menghasilkan ukuran bounding box acak
        w = np.random.randint(40, 120)
        h = np.random.randint(40, 120)

        # Menghasilkan posisi acak
        x1 = np.random.randint(0, img_w - w)
        y1 = np.random.randint(0, img_h - h)

        # Menghitung koordinat kanan bawah
        x2 = x1 + w
        y2 = y1 + h

        # Menambahkan bounding box ke list
        gt_boxes.append([x1, y1, x2, y2])

    # Menyimpan ground truth
    ground_truths[cls_name] = np.array(gt_boxes)

    # Menentukan jumlah prediksi (lebih banyak dari GT untuk ada FP)
    n_pred = n_gt + np.random.randint(2, 8)

    # Membuat bounding box prediksi
    pred_boxes = []
    pred_scores = []

    # Membuat prediksi yang mendekati ground truth (True Positives)
    for gt_idx in range(min(n_gt, n_pred)):
        # Mengambil ground truth box sebagai referensi
        gt_box = gt_boxes[gt_idx]

        # Menambahkan offset acak (noise) ke prediksi
        offset = np.random.randint(-15, 16, size=4)

        # Membuat prediksi box dengan offset
        pred_box = [
            max(0, gt_box[0] + offset[0]),
            max(0, gt_box[1] + offset[1]),
            min(img_w, gt_box[2] + offset[2]),
            min(img_h, gt_box[3] + offset[3])
        ]

        # Menambahkan prediksi box ke list
        pred_boxes.append(pred_box)

        # Menghasilkan confidence score tinggi untuk prediksi yang mendekati GT
        score = np.random.uniform(0.5, 0.99)
        pred_scores.append(score)

    # Menambahkan beberapa prediksi palsu (False Positives)
    n_false = n_pred - len(pred_boxes)
    for _ in range(n_false):
        # Menghasilkan bounding box acak yang tidak overlap dengan GT
        w = np.random.randint(30, 100)
        h = np.random.randint(30, 100)
        x1 = np.random.randint(0, img_w - w)
        y1 = np.random.randint(0, img_h - h)

        # Menambahkan false positive box
        pred_boxes.append([x1, y1, x1 + w, y1 + h])

        # Menghasilkan confidence score rendah untuk false positives
        score = np.random.uniform(0.1, 0.6)
        pred_scores.append(score)

    # Menyimpan prediksi beserta confidence score
    predictions[cls_name] = {
        "boxes": np.array(pred_boxes),
        "scores": np.array(pred_scores)
    }

    # Menampilkan info data sintetis per kelas
    print(f"  {cls_name}: {n_gt} GT boxes, {n_pred} prediksi")

# ============================================================
# 4. Menghitung IoU untuk Setiap Prediksi terhadap GT
# ============================================================

print("\n[INFO] Menghitung IoU prediksi vs ground truth...")
print("-" * 50)


# Mendefinisikan fungsi untuk menghitung AP pada satu kelas
def hitung_ap(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """
    Menghitung Average Precision untuk satu kelas pada IoU threshold tertentu.
    """
    # Mengurutkan prediksi berdasarkan confidence score (tertinggi dahulu)
    sorted_indices = np.argsort(-pred_scores)

    # Menyusun ulang prediksi berdasarkan urutan score
    sorted_boxes = pred_boxes[sorted_indices]
    sorted_scores = pred_scores[sorted_indices]

    # Mendapatkan jumlah ground truth
    n_gt = len(gt_boxes)

    # Membuat array untuk melacak GT yang sudah di-match
    gt_matched = np.zeros(n_gt, dtype=bool)

    # Membuat list untuk menyimpan TP dan FP
    tp_list = []
    fp_list = []
    score_list = []

    # Mengevaluasi setiap prediksi secara berurutan
    for pred_idx in range(len(sorted_boxes)):
        # Mengambil bounding box prediksi
        pred_box = sorted_boxes[pred_idx]

        # Menginisialisasi IoU terbaik
        best_iou = 0.0
        best_gt_idx = -1

        # Menghitung IoU terhadap semua ground truth
        for gt_idx in range(n_gt):
            # Menghitung IoU antara prediksi dan GT
            iou_val, _, _ = hitung_iou(pred_box.tolist(), gt_boxes[gt_idx].tolist())

            # Memperbarui IoU terbaik jika lebih besar
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gt_idx

        # Menentukan apakah TP atau FP berdasarkan IoU threshold
        if best_iou >= iou_threshold and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
            # True Positive: IoU cukup dan GT belum di-match
            tp_list.append(1)
            fp_list.append(0)

            # Menandai GT sebagai sudah di-match
            gt_matched[best_gt_idx] = True
        else:
            # False Positive: IoU kurang atau GT sudah di-match
            tp_list.append(0)
            fp_list.append(1)

        # Menyimpan score
        score_list.append(sorted_scores[pred_idx])

    # Mengkonversi ke numpy array
    tp_array = np.array(tp_list)
    fp_array = np.array(fp_list)

    # Menghitung kumulatif TP dan FP
    cum_tp = np.cumsum(tp_array)
    cum_fp = np.cumsum(fp_array)

    # Menghitung precision pada setiap titik
    precision_curve = cum_tp / (cum_tp + cum_fp)

    # Menghitung recall pada setiap titik
    recall_curve = cum_tp / n_gt if n_gt > 0 else cum_tp

    # Menghitung AP menggunakan integrasi numerik (metode trapezoid)
    # Menambahkan titik awal (recall=0, precision=1)
    recall_with_start = np.concatenate([[0], recall_curve])
    precision_with_start = np.concatenate([[1], precision_curve])

    # Menghitung AP dengan np.trapz (trapezoid integration)
    ap = np.trapz(precision_with_start, recall_with_start)

    # Mengembalikan AP, precision curve, recall curve, dan scores
    return ap, precision_curve, recall_curve, np.array(score_list)


# Menghitung AP untuk setiap kelas pada IoU=0.5
print("  AP (Average Precision) pada IoU=0.5:")
ap_per_class = {}
pr_curves = {}

# Menghitung AP untuk setiap kelas
for cls_name in det_class_names:
    # Mengambil ground truth dan prediksi untuk kelas ini
    gt = ground_truths[cls_name]
    pred = predictions[cls_name]

    # Menghitung AP pada IoU=0.5
    ap_val, prec, rec, scores = hitung_ap(
        gt, pred["boxes"], pred["scores"], iou_threshold=0.5
    )

    # Menyimpan hasil AP
    ap_per_class[cls_name] = ap_val

    # Menyimpan kurva PR
    pr_curves[cls_name] = {"precision": prec, "recall": rec, "scores": scores}

    # Menampilkan AP untuk kelas ini
    print(f"    {cls_name:>12s}: AP@0.5 = {ap_val:.4f}")

# Menghitung mAP (mean Average Precision)
mAP_05 = np.mean(list(ap_per_class.values()))
print(f"\n    {'mAP@0.5':>12s}: {mAP_05:.4f}")

# ============================================================
# 5. Menghitung AP pada Berbagai IoU Thresholds
# ============================================================

print("\n[INFO] Menghitung AP pada berbagai IoU thresholds...")
print("-" * 50)

# Mendefinisikan range IoU threshold untuk evaluasi
iou_thresholds = np.arange(0.3, 0.95, 0.05)

# Membuat dictionary untuk menyimpan AP pada setiap threshold
ap_at_thresholds = {cls: [] for cls in det_class_names}

# Menghitung AP untuk setiap kelas pada setiap IoU threshold
for iou_thresh in iou_thresholds:
    for cls_name in det_class_names:
        # Mengambil data ground truth dan prediksi
        gt = ground_truths[cls_name]
        pred = predictions[cls_name]

        # Menghitung AP pada IoU threshold ini
        ap_val, _, _, _ = hitung_ap(gt, pred["boxes"], pred["scores"],
                                    iou_threshold=iou_thresh)

        # Menyimpan AP
        ap_at_thresholds[cls_name].append(ap_val)

# Menghitung mAP pada setiap IoU threshold
mAP_at_thresholds = []
for i in range(len(iou_thresholds)):
    # Menghitung rata-rata AP semua kelas pada threshold ini
    avg_ap = np.mean([ap_at_thresholds[cls][i] for cls in det_class_names])
    mAP_at_thresholds.append(avg_ap)

# Menghitung mAP@[0.5:0.95] (COCO-style)
# Mengambil indeks threshold 0.5 sampai 0.95 dengan step 0.05
coco_indices = [i for i, t in enumerate(iou_thresholds) if 0.5 <= t <= 0.95]
mAP_coco = np.mean([mAP_at_thresholds[i] for i in coco_indices])

# Menampilkan mAP pada berbagai threshold
print(f"  mAP@[0.5:0.95] (COCO-style): {mAP_coco:.4f}")
for cls_name in det_class_names:
    # Menghitung rata-rata AP COCO-style per kelas
    cls_ap_coco = np.mean([ap_at_thresholds[cls_name][i] for i in coco_indices])
    print(f"    {cls_name:>12s}: AP@[0.5:0.95] = {cls_ap_coco:.4f}")

# ============================================================
# 6. Visualisasi 2: Precision-Recall Curve
# ============================================================

print("\n[INFO] Membuat visualisasi Precision-Recall curve...")
print("-" * 50)

# Membuat figure dengan 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mendefinisikan warna untuk setiap kelas
class_colors = ['#4472C4', '#ED7D31', '#70AD47']

# --- Panel kiri: PR curve per kelas pada IoU=0.5 ---
for cls_idx, cls_name in enumerate(det_class_names):
    # Mengambil precision dan recall curve
    prec = pr_curves[cls_name]["precision"]
    rec = pr_curves[cls_name]["recall"]

    # Menggambar PR curve
    axes[0].plot(rec, prec, '-', color=class_colors[cls_idx], linewidth=2,
                 label=f'{cls_name} (AP={ap_per_class[cls_name]:.3f})')

    # Mengisi area di bawah kurva untuk visualisasi AP
    axes[0].fill_between(rec, prec, alpha=0.1, color=class_colors[cls_idx])

# Mengatur sumbu dan label
axes[0].set_xlabel("Recall", fontsize=11)
axes[0].set_ylabel("Precision", fontsize=11)
axes[0].set_title(f"Precision-Recall Curve (IoU≥0.5)\nmAP@0.5 = {mAP_05:.4f}",
                  fontsize=11, fontweight='bold')
axes[0].set_xlim(-0.05, 1.05)
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# --- Panel kanan: AP vs IoU threshold per kelas ---
for cls_idx, cls_name in enumerate(det_class_names):
    # Menggambar AP pada berbagai IoU threshold
    axes[1].plot(iou_thresholds, ap_at_thresholds[cls_name], 'o-',
                 color=class_colors[cls_idx], linewidth=2, markersize=4,
                 label=cls_name)

# Menggambar mAP pada berbagai IoU threshold
axes[1].plot(iou_thresholds, mAP_at_thresholds, 'k--', linewidth=2,
             markersize=4, label=f'mAP (avg={mAP_coco:.3f})', alpha=0.7)

# Menambahkan garis vertikal pada IoU=0.5 (standar PASCAL VOC)
axes[1].axvline(x=0.5, color='red', linestyle=':', alpha=0.5, label='IoU=0.5 (VOC)')

# Mengatur sumbu dan label
axes[1].set_xlabel("IoU Threshold", fontsize=11)
axes[1].set_ylabel("Average Precision", fontsize=11)
axes[1].set_title("AP vs IoU Threshold per Kelas", fontsize=11, fontweight='bold')
axes[1].set_xlim(0.25, 0.95)
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(fontsize=8, loc='lower left')
axes[1].grid(True, alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 16: Precision-Recall Curve dan AP vs IoU",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi PR curve
output_path_2 = os.path.join(OUTPUT_DIR, "16_precision_recall_curve.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 7. Visualisasi 3: mAP Comparison Chart
# ============================================================

print("\n[INFO] Membuat visualisasi mAP comparison...")
print("-" * 50)

# Membuat figure untuk mAP comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel kiri: Bar chart AP@0.5 per kelas ---
# Mendefinisikan posisi bar
x_pos = np.arange(n_det_classes)

# Mengambil AP values per kelas
ap_values = [ap_per_class[cls] for cls in det_class_names]

# Menggambar bar chart AP per kelas
bars = axes[0].bar(x_pos, ap_values, color=class_colors, alpha=0.85,
                   edgecolor='black', linewidth=0.5)

# Menambahkan label nilai di atas setiap bar
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{bar.get_height():.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

# Menambahkan garis horizontal untuk mAP
axes[0].axhline(y=mAP_05, color='red', linestyle='--', linewidth=1.5,
                label=f'mAP@0.5 = {mAP_05:.3f}')

# Mengatur sumbu dan label
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(det_class_names, fontsize=10)
axes[0].set_ylabel("Average Precision (AP)", fontsize=11)
axes[0].set_title("AP@0.5 per Kelas", fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 1.15)
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# --- Panel kanan: Comparison table ---
# Menyembunyikan sumbu
axes[1].axis('off')

# Menyiapkan data tabel perbandingan
comparison_data = []

# Menambahkan baris per kelas
for cls_name in det_class_names:
    # Menghitung AP pada berbagai threshold
    ap_05 = ap_per_class[cls_name]

    # Mengambil AP pada IoU=0.75 (ketat)
    idx_075 = list(iou_thresholds).index(
        min(iou_thresholds, key=lambda x: abs(x - 0.75)))
    ap_075 = ap_at_thresholds[cls_name][idx_075]

    # Menghitung AP COCO-style
    cls_coco = np.mean([ap_at_thresholds[cls_name][i] for i in coco_indices])

    # Menambahkan baris ke tabel
    comparison_data.append([
        cls_name, f"{ap_05:.4f}", f"{ap_075:.4f}", f"{cls_coco:.4f}",
        str(len(ground_truths[cls_name]))
    ])

# Menambahkan baris rata-rata (mAP)
mAP_075_val = np.mean([ap_at_thresholds[cls][idx_075] for cls in det_class_names])
comparison_data.append([
    "mAP (rata-rata)", f"{mAP_05:.4f}", f"{mAP_075_val:.4f}",
    f"{mAP_coco:.4f}", "-"
])

# Mendefinisikan header tabel
comp_header = ["Kelas", "AP@0.5", "AP@0.75", "AP@[.5:.95]", "#GT"]

# Membuat tabel
table = axes[1].table(cellText=comparison_data, colLabels=comp_header,
                      cellLoc='center', loc='center',
                      colWidths=[0.24, 0.16, 0.16, 0.22, 0.10])

# Mengatur ukuran font tabel
table.auto_set_font_size(False)
table.set_fontsize(11)

# Mengatur skala tabel
table.scale(1.0, 1.6)

# Mengatur warna header
for j in range(len(comp_header)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Mengatur warna baris data
for i in range(1, len(comparison_data) + 1):
    for j in range(len(comp_header)):
        if i == len(comparison_data):
            # Mengatur warna kuning untuk baris mAP
            table[i, j].set_facecolor('#FFF2CC')
        elif i % 2 == 0:
            table[i, j].set_facecolor('#D6E4F0')
        else:
            table[i, j].set_facecolor('#EDF2F9')

# Menambahkan judul tabel
axes[1].set_title("Tabel Perbandingan AP per Kelas", fontsize=12, fontweight='bold')

# Menambahkan judul utama
plt.suptitle("Percobaan 16: mAP Comparison dan Tabel Evaluasi",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi mAP comparison
output_path_3 = os.path.join(OUTPUT_DIR, "16_mAP_chart.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 16")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. IoU (Intersection over Union):")
print("     - Mengukur overlap antara prediksi dan ground truth")
print("     - IoU = Area_intersection / Area_union")
print("     - Range: 0 (tidak overlap) sampai 1 (identik)")
print("  2. Average Precision (AP):")
print("     - Area di bawah kurva Precision-Recall")
print("     - Prediksi diurutkan berdasarkan confidence score")
print("     - TP ditentukan oleh IoU >= threshold")
print(f"     - AP@0.5: {[f'{cls}={ap_per_class[cls]:.4f}' for cls in det_class_names]}")
print("  3. mean Average Precision (mAP):")
print(f"     - mAP@0.5 = {mAP_05:.4f}")
print(f"     - mAP@[0.5:0.95] = {mAP_coco:.4f}")
print("  4. IoU Threshold:")
print("     - Rendah (0.3): lebih banyak TP, AP tinggi")
print("     - Tinggi (0.9): hanya prediksi sangat akurat, AP rendah")
print("     - PASCAL VOC  : IoU=0.5")
print("     - COCO        : IoU=[0.5:0.05:0.95]")
print(f"\nData sintetis: {n_det_classes} kelas, {img_w}x{img_h} gambar")
for cls_name in det_class_names:
    print(f"  - {cls_name}: {len(ground_truths[cls_name])} GT, "
          f"{len(predictions[cls_name]['boxes'])} prediksi")
print("=" * 60)
