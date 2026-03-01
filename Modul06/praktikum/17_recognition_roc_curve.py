"""
==========================================================================
PERCOBAAN 17: ROC CURVE DAN EER UNTUK RECOGNITION
==========================================================================
Program ini mempelajari cara mengevaluasi sistem recognition menggunakan
ROC (Receiver Operating Characteristic) curve, AUC (Area Under Curve),
EER (Equal Error Rate), dan DET (Detection Error Trade-off) curve.
Data similarity score sintetis digunakan untuk pasangan genuine/impostor.

Konsep yang dipelajari:
- Distribusi skor genuine (same person) vs impostor (different person)
- ROC curve: True Positive Rate vs False Positive Rate
- AUC: Area di bawah ROC curve (ukuran performa keseluruhan)
- EER: titik di mana FAR = FRR (False Acceptance = False Rejection)
- DET curve: FRR vs FAR (Detection Error Trade-off)
- Pengaruh threshold terhadap FAR dan FRR
- Perbandingan model berdasarkan distribusi skor

Fungsi utama yang dipelajari:
- np.random.normal()             : Distribusi skor sintetis
- np.sort() / np.searchsorted()  : Operasi threshold sweeping
- np.trapz()                     : Integrasi numerik untuk AUC
- np.interp()                    : Interpolasi untuk mencari EER
- plt.fill_between()             : Visualisasi area AUC
- plt.axvline() / plt.axhline()  : Penanda titik EER

Hasil: ROC curve, DET curve, distribusi skor, dan analisis threshold
==========================================================================
"""

# Mengimpor NumPy untuk operasi array dan statistik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 17: ROC CURVE DAN EER UNTUK RECOGNITION")
print("=" * 60)

# ============================================================
# 1. Membuat Data Similarity Score Sintetis
# ============================================================

print("\n[INFO] Membuat data similarity score sintetis...")
print("-" * 50)

# Menetapkan seed random untuk reprodusibilitas
np.random.seed(42)

# Mendefinisikan jumlah pasangan genuine (orang sama)
n_genuine = 500

# Mendefinisikan jumlah pasangan impostor (orang berbeda)
n_impostor = 500

# Mendefinisikan konfigurasi 3 model dengan distribusi skor berbeda
model_configs = {
    "Model A (Bagus)": {
        "genuine_mean": 0.75, "genuine_std": 0.10,
        "impostor_mean": 0.25, "impostor_std": 0.10
    },
    "Model B (Sedang)": {
        "genuine_mean": 0.65, "genuine_std": 0.15,
        "impostor_mean": 0.35, "impostor_std": 0.15
    },
    "Model C (Buruk)": {
        "genuine_mean": 0.55, "genuine_std": 0.18,
        "impostor_mean": 0.42, "impostor_std": 0.18
    }
}

# Membuat dictionary untuk menyimpan skor setiap model
model_scores = {}

# Membuat skor sintetis untuk setiap model
for model_name, config in model_configs.items():
    # Menghasilkan skor genuine (similarity tinggi)
    genuine_scores = np.random.normal(
        config["genuine_mean"], config["genuine_std"], n_genuine
    )

    # Menghasilkan skor impostor (similarity rendah)
    impostor_scores = np.random.normal(
        config["impostor_mean"], config["impostor_std"], n_impostor
    )

    # Membatasi skor dalam range [0, 1]
    genuine_scores = np.clip(genuine_scores, 0, 1)
    impostor_scores = np.clip(impostor_scores, 0, 1)

    # Menyimpan skor ke dictionary
    model_scores[model_name] = {
        "genuine": genuine_scores,
        "impostor": impostor_scores
    }

    # Menampilkan statistik skor
    print(f"  {model_name}:")
    print(f"    Genuine  : mean={np.mean(genuine_scores):.4f}, "
          f"std={np.std(genuine_scores):.4f}")
    print(f"    Impostor : mean={np.mean(impostor_scores):.4f}, "
          f"std={np.std(impostor_scores):.4f}")

# ============================================================
# 2. Menghitung ROC Curve (TPR vs FPR)
# ============================================================

print("\n[INFO] Menghitung ROC curve untuk setiap model...")
print("-" * 50)


# Mendefinisikan fungsi untuk menghitung ROC curve
def hitung_roc(genuine_scores, impostor_scores, n_thresholds=200):
    """
    Menghitung ROC curve dari distribusi skor genuine dan impostor.
    TPR (True Positive Rate) = genuine yang diterima / total genuine
    FPR (False Positive Rate) = impostor yang diterima / total impostor
    """
    # Mendefinisikan range threshold dari 0 sampai 1
    thresholds = np.linspace(0, 1, n_thresholds)

    # Membuat array untuk menyimpan TPR dan FPR
    tpr_array = np.zeros(n_thresholds)
    fpr_array = np.zeros(n_thresholds)

    # Membuat array untuk FRR (False Rejection Rate) dan FAR (False Acceptance Rate)
    frr_array = np.zeros(n_thresholds)
    far_array = np.zeros(n_thresholds)

    # Menghitung TPR dan FPR pada setiap threshold
    for i, thresh in enumerate(thresholds):
        # TPR: proporsi genuine yang skornya >= threshold (diterima benar)
        tpr_array[i] = np.sum(genuine_scores >= thresh) / len(genuine_scores)

        # FPR: proporsi impostor yang skornya >= threshold (diterima salah)
        fpr_array[i] = np.sum(impostor_scores >= thresh) / len(impostor_scores)

        # FAR = FPR (impostor diterima = false acceptance)
        far_array[i] = fpr_array[i]

        # FRR = 1 - TPR (genuine ditolak = false rejection)
        frr_array[i] = 1 - tpr_array[i]

    # Mengembalikan hasil ROC beserta threshold
    return thresholds, tpr_array, fpr_array, far_array, frr_array


# Menghitung ROC curve dan metrik untuk setiap model
roc_data = {}

for model_name, scores in model_scores.items():
    # Menghitung ROC curve
    thresholds, tpr, fpr, far, frr = hitung_roc(
        scores["genuine"], scores["impostor"]
    )

    # Menghitung AUC menggunakan integrasi trapezoid
    # Mengurutkan FPR dan TPR untuk integrasi yang benar
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    # Menghitung AUC = area di bawah ROC curve
    auc_value = np.trapz(tpr_sorted, fpr_sorted)

    # Mencari EER: titik di mana FAR = FRR
    # Menghitung selisih absolut antara FAR dan FRR
    diff_far_frr = np.abs(far - frr)

    # Menemukan indeks dengan selisih minimum (titik EER)
    eer_index = np.argmin(diff_far_frr)

    # Mengambil nilai EER
    eer_value = (far[eer_index] + frr[eer_index]) / 2.0
    eer_threshold = thresholds[eer_index]

    # Menyimpan semua data ROC
    roc_data[model_name] = {
        "thresholds": thresholds,
        "tpr": tpr, "fpr": fpr,
        "far": far, "frr": frr,
        "auc": auc_value,
        "eer": eer_value,
        "eer_threshold": eer_threshold
    }

    # Menampilkan metrik untuk model ini
    print(f"  {model_name}:")
    print(f"    AUC = {auc_value:.4f}")
    print(f"    EER = {eer_value:.4f} (threshold = {eer_threshold:.4f})")

# ============================================================
# 3. Visualisasi 1: ROC Curve
# ============================================================

print("\n[INFO] Membuat visualisasi ROC curve...")
print("-" * 50)

# Membuat figure dengan 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mendefinisikan warna untuk setiap model
model_colors = ['#4472C4', '#ED7D31', '#70AD47']

# --- Panel kiri: ROC Curve (FPR vs TPR) ---
# Menggambar garis diagonal (random classifier)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')

# Menggambar ROC curve untuk setiap model
for idx, (model_name, data) in enumerate(roc_data.items()):
    # Mengurutkan berdasarkan FPR untuk plot yang rapi
    sort_idx = np.argsort(data["fpr"])

    # Menggambar ROC curve
    axes[0].plot(data["fpr"][sort_idx], data["tpr"][sort_idx],
                 color=model_colors[idx], linewidth=2,
                 label=f'{model_name} (AUC={data["auc"]:.3f})')

    # Mengisi area di bawah kurva
    axes[0].fill_between(data["fpr"][sort_idx], data["tpr"][sort_idx],
                         alpha=0.08, color=model_colors[idx])

    # Menandai titik EER pada ROC curve
    eer_fpr = data["far"][np.argmin(np.abs(data["far"] - data["frr"]))]
    eer_tpr = data["tpr"][np.argmin(np.abs(data["far"] - data["frr"]))]
    axes[0].plot(eer_fpr, eer_tpr, 'o', color=model_colors[idx],
                 markersize=8, markeredgecolor='black', markeredgewidth=1)

# Mengatur sumbu dan label
axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=11)
axes[0].set_ylabel("True Positive Rate (TPR)", fontsize=11)
axes[0].set_title("ROC Curve - Perbandingan Model", fontsize=12, fontweight='bold')
axes[0].set_xlim(-0.02, 1.02)
axes[0].set_ylim(-0.02, 1.02)
axes[0].legend(fontsize=9, loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# --- Panel kanan: ROC Curve (skala logaritmik untuk detail FPR rendah) ---
# Menggambar ROC curve pada skala semi-log
for idx, (model_name, data) in enumerate(roc_data.items()):
    # Mengurutkan berdasarkan FPR
    sort_idx = np.argsort(data["fpr"])

    # Mengambil FPR dan TPR yang sudah diurutkan
    fpr_s = data["fpr"][sort_idx]
    tpr_s = data["tpr"][sort_idx]

    # Mengganti FPR=0 dengan nilai kecil untuk skala log
    fpr_log = np.where(fpr_s > 0, fpr_s, 1e-4)

    # Menggambar ROC curve pada skala semi-log
    axes[1].semilogx(fpr_log, tpr_s, color=model_colors[idx], linewidth=2,
                     label=f'{model_name}')

# Mengatur sumbu dan label
axes[1].set_xlabel("False Positive Rate (FPR) - skala log", fontsize=11)
axes[1].set_ylabel("True Positive Rate (TPR)", fontsize=11)
axes[1].set_title("ROC Curve (Skala Logaritmik)", fontsize=12, fontweight='bold')
axes[1].set_ylim(-0.02, 1.02)
axes[1].legend(fontsize=9, loc='lower right')
axes[1].grid(True, alpha=0.3, which='both')

# Menambahkan judul utama
plt.suptitle("Percobaan 17: ROC Curve untuk Recognition System",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ROC curve
output_path_1 = os.path.join(OUTPUT_DIR, "17_roc_curve.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 4. Visualisasi 2: DET Curve (Detection Error Trade-off)
# ============================================================

print("\n[INFO] Membuat visualisasi DET curve...")
print("-" * 50)

# Membuat figure untuk DET curve
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel kiri: DET Curve (FRR vs FAR) ---
# Menggambar garis diagonal (EER line)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='EER line (FAR=FRR)')

# Menggambar DET curve untuk setiap model
for idx, (model_name, data) in enumerate(roc_data.items()):
    # Mengurutkan berdasarkan FAR
    sort_idx = np.argsort(data["far"])

    # Menggambar DET curve (FAR vs FRR)
    axes[0].plot(data["far"][sort_idx], data["frr"][sort_idx],
                 color=model_colors[idx], linewidth=2,
                 label=f'{model_name} (EER={data["eer"]:.3f})')

    # Menandai titik EER
    axes[0].plot(data["eer"], data["eer"], 'o', color=model_colors[idx],
                 markersize=10, markeredgecolor='black', markeredgewidth=1.5)

# Mengatur sumbu dan label
axes[0].set_xlabel("False Acceptance Rate (FAR)", fontsize=11)
axes[0].set_ylabel("False Rejection Rate (FRR)", fontsize=11)
axes[0].set_title("DET Curve - Detection Error Trade-off", fontsize=12,
                  fontweight='bold')
axes[0].set_xlim(-0.02, 0.6)
axes[0].set_ylim(-0.02, 0.6)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# --- Panel kanan: FAR dan FRR vs Threshold ---
# Memilih model pertama untuk analisis detail
first_model = list(roc_data.keys())[0]
data = roc_data[first_model]

# Menggambar FAR vs threshold
axes[1].plot(data["thresholds"], data["far"], 'b-', linewidth=2,
             label='FAR (False Acceptance Rate)')

# Menggambar FRR vs threshold
axes[1].plot(data["thresholds"], data["frr"], 'r-', linewidth=2,
             label='FRR (False Rejection Rate)')

# Menandai titik EER
axes[1].axvline(x=data["eer_threshold"], color='green', linestyle='--',
                linewidth=1.5, alpha=0.7,
                label=f'EER threshold = {data["eer_threshold"]:.3f}')

# Menandai titik perpotongan FAR=FRR
axes[1].axhline(y=data["eer"], color='gray', linestyle=':', alpha=0.5)

# Menambahkan anotasi EER
axes[1].annotate(f'EER = {data["eer"]:.3f}',
                 xy=(data["eer_threshold"], data["eer"]),
                 xytext=(data["eer_threshold"] + 0.1, data["eer"] + 0.1),
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black'),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

# Mengatur sumbu dan label
axes[1].set_xlabel("Threshold", fontsize=11)
axes[1].set_ylabel("Error Rate", fontsize=11)
axes[1].set_title(f"FAR & FRR vs Threshold ({first_model})", fontsize=12,
                  fontweight='bold')
axes[1].set_xlim(-0.02, 1.02)
axes[1].set_ylim(-0.02, 1.02)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 17: DET Curve dan Analisis FAR/FRR",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi DET curve
output_path_2 = os.path.join(OUTPUT_DIR, "17_det_curve.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 5. Visualisasi 3: Score Distribution
# ============================================================

print("\n[INFO] Membuat visualisasi score distribution...")
print("-" * 50)

# Membuat figure dengan 1x3 subplot untuk setiap model
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Menampilkan distribusi skor untuk setiap model
for idx, (model_name, scores) in enumerate(model_scores.items()):
    # Mendefinisikan bins untuk histogram
    bins = np.linspace(0, 1, 50)

    # Menggambar histogram skor genuine
    axes[idx].hist(scores["genuine"], bins=bins, alpha=0.6, color='#4472C4',
                   label=f'Genuine (n={n_genuine})', density=True,
                   edgecolor='white', linewidth=0.5)

    # Menggambar histogram skor impostor
    axes[idx].hist(scores["impostor"], bins=bins, alpha=0.6, color='#ED7D31',
                   label=f'Impostor (n={n_impostor})', density=True,
                   edgecolor='white', linewidth=0.5)

    # Menandai threshold EER
    eer_thresh = roc_data[model_name]["eer_threshold"]
    axes[idx].axvline(x=eer_thresh, color='red', linestyle='--', linewidth=2,
                      label=f'EER thresh = {eer_thresh:.3f}')

    # Menambahkan area shading untuk overlap (error zone)
    axes[idx].axvspan(
        min(scores["impostor"].max(), eer_thresh),
        max(scores["genuine"].min(), eer_thresh),
        alpha=0.05, color='red'
    )

    # Mengatur sumbu dan label
    axes[idx].set_xlabel("Similarity Score", fontsize=10)
    axes[idx].set_ylabel("Density", fontsize=10)
    axes[idx].set_title(f"{model_name}\nEER={roc_data[model_name]['eer']:.3f}, "
                        f"AUC={roc_data[model_name]['auc']:.3f}",
                        fontsize=10, fontweight='bold')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

# Menambahkan judul utama
plt.suptitle("Percobaan 17: Distribusi Skor Genuine vs Impostor",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi distribusi skor
output_path_3 = os.path.join(OUTPUT_DIR, "17_score_distribution.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# 6. Visualisasi 4: Analisis Threshold Lengkap
# ============================================================

print("\n[INFO] Membuat visualisasi analisis threshold...")
print("-" * 50)

# Membuat figure dengan 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Panel kiri atas: Perbandingan AUC ---
# Mengambil nilai AUC per model
model_names_short = [name.split("(")[1].rstrip(")") for name in roc_data.keys()]
auc_values = [data["auc"] for data in roc_data.values()]

# Menggambar bar chart AUC
bars_auc = axes[0, 0].bar(model_names_short, auc_values, color=model_colors,
                           alpha=0.85, edgecolor='black', linewidth=0.5)

# Menambahkan label nilai di atas setiap bar
for bar in bars_auc:
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.4f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

# Mengatur sumbu dan label
axes[0, 0].set_ylabel("AUC", fontsize=11)
axes[0, 0].set_title("Perbandingan AUC antar Model", fontsize=11, fontweight='bold')
axes[0, 0].set_ylim(0, 1.15)
axes[0, 0].grid(axis='y', alpha=0.3)

# --- Panel kanan atas: Perbandingan EER ---
# Mengambil nilai EER per model
eer_values = [data["eer"] for data in roc_data.values()]

# Menggambar bar chart EER
bars_eer = axes[0, 1].bar(model_names_short, eer_values, color=model_colors,
                           alpha=0.85, edgecolor='black', linewidth=0.5)

# Menambahkan label nilai di atas setiap bar
for bar in bars_eer:
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{bar.get_height():.4f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

# Mengatur sumbu dan label
axes[0, 1].set_ylabel("EER (Equal Error Rate)", fontsize=11)
axes[0, 1].set_title("Perbandingan EER antar Model\n(lebih rendah = lebih baik)",
                      fontsize=11, fontweight='bold')
axes[0, 1].set_ylim(0, max(eer_values) * 1.5)
axes[0, 1].grid(axis='y', alpha=0.3)

# --- Panel kiri bawah: FAR pada FRR tetap (operating points) ---
# Mendefinisikan target FRR (operating points)
target_frrs = [0.01, 0.05, 0.10, 0.20]

# Menghitung FAR pada setiap target FRR untuk setiap model
x_op = np.arange(len(target_frrs))
bar_w = 0.25

# Menggambar bar chart FAR pada berbagai operating points
for idx, (model_name, data) in enumerate(roc_data.items()):
    # Menghitung FAR pada setiap target FRR
    far_at_target = []
    for target_frr in target_frrs:
        # Mencari threshold yang memberikan FRR mendekati target
        frr_diff = np.abs(data["frr"] - target_frr)
        best_idx = np.argmin(frr_diff)

        # Mengambil FAR pada threshold tersebut
        far_at_target.append(data["far"][best_idx])

    # Menggambar bar chart
    axes[1, 0].bar(x_op + idx * bar_w, far_at_target, bar_w,
                   color=model_colors[idx], alpha=0.85,
                   label=model_names_short[idx])

# Mengatur sumbu dan label
axes[1, 0].set_xticks(x_op + bar_w)
axes[1, 0].set_xticklabels([f'FRR={frr}' for frr in target_frrs], fontsize=9)
axes[1, 0].set_ylabel("FAR (False Acceptance Rate)", fontsize=10)
axes[1, 0].set_title("FAR pada Operating Points (FRR tetap)", fontsize=11,
                      fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(axis='y', alpha=0.3)

# --- Panel kanan bawah: Tabel ringkasan ---
# Menyembunyikan sumbu
axes[1, 1].axis('off')

# Membuat data tabel ringkasan
summary_data = []
for model_name, data in roc_data.items():
    # Mengambil nama model singkat
    short_name = model_name.split("(")[1].rstrip(")")

    # Menambahkan baris ke tabel
    summary_data.append([
        short_name,
        f"{data['auc']:.4f}",
        f"{data['eer']:.4f}",
        f"{data['eer_threshold']:.4f}",
        f"{n_genuine}",
        f"{n_impostor}"
    ])

# Mendefinisikan header tabel
summary_header = ["Model", "AUC", "EER", "EER Thresh", "#Genuine", "#Impostor"]

# Membuat tabel
table = axes[1, 1].table(cellText=summary_data, colLabels=summary_header,
                          cellLoc='center', loc='center',
                          colWidths=[0.14, 0.14, 0.14, 0.18, 0.16, 0.16])

# Mengatur ukuran font tabel
table.auto_set_font_size(False)
table.set_fontsize(11)

# Mengatur skala tabel
table.scale(1.0, 1.8)

# Mengatur warna header
for j in range(len(summary_header)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Mengatur warna baris data
for i in range(1, len(summary_data) + 1):
    for j in range(len(summary_header)):
        if i % 2 == 0:
            table[i, j].set_facecolor('#D6E4F0')
        else:
            table[i, j].set_facecolor('#EDF2F9')

# Menambahkan judul tabel
axes[1, 1].set_title("Tabel Ringkasan Performa Model", fontsize=11, fontweight='bold')

# Menambahkan judul utama
plt.suptitle("Percobaan 17: Analisis Threshold dan Perbandingan Model",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi analisis threshold
output_path_4 = os.path.join(OUTPUT_DIR, "17_threshold_analysis.png")
plt.savefig(output_path_4, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_4}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 17")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. ROC Curve (Receiver Operating Characteristic):")
print("     - Sumbu X = FPR (False Positive Rate)")
print("     - Sumbu Y = TPR (True Positive Rate)")
print("     - Semakin dekat ke pojok kiri atas = semakin baik")
print("  2. AUC (Area Under Curve):")
print("     - Range: 0.5 (random) sampai 1.0 (sempurna)")
for model_name, data in roc_data.items():
    print(f"     - {model_name}: AUC = {data['auc']:.4f}")
print("  3. EER (Equal Error Rate):")
print("     - Titik di mana FAR = FRR")
print("     - Semakin rendah = semakin baik")
for model_name, data in roc_data.items():
    print(f"     - {model_name}: EER = {data['eer']:.4f}")
print("  4. DET Curve (Detection Error Trade-off):")
print("     - FRR vs FAR")
print("     - Semakin dekat ke origin = semakin baik")
print("  5. Threshold Analysis:")
print("     - FAR & FRR berubah berlawanan terhadap threshold")
print("     - Threshold tinggi → FRR naik, FAR turun (ketat)")
print("     - Threshold rendah → FRR turun, FAR naik (longgar)")
print(f"\nData sintetis: {n_genuine} genuine + {n_impostor} impostor pairs")
print("=" * 60)
