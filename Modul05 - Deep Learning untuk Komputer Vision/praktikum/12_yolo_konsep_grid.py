"""
==========================================================================
PERCOBAAN 12: YOLO KONSEP - DETEKSI BERBASIS GRID
==========================================================================
Program ini mempelajari konsep dasar YOLO (You Only Look Once), yaitu
pendekatan deteksi objek yang membagi gambar menjadi grid SxS dan
setiap sel grid memprediksi bounding box beserta confidence score.
Implementasi mensimulasikan pipeline deteksi YOLO secara konseptual,
mulai dari pembagian grid, prediksi per sel, hingga NMS.

Fungsi utama yang dipelajari:
- cv2.rectangle()          : Menggambar bounding box dan grid
- cv2.putText()            : Menuliskan teks confidence score
- cv2.line()               : Menggambar garis grid
- Manual grid division     : Membagi gambar menjadi grid SxS
- Manual NMS               : Non-Maximum Suppression pada prediksi grid
- Manual confidence score  : Simulasi skor deteksi per sel

Konsep yang dipelajari:
- YOLO: deteksi satu-pass (You Only Look Once)
- Grid division: membagi gambar menjadi SxS sel
- Anchor boxes: prediksi bounding box relatif terhadap sel
- Confidence score: P(objek) * IoU(pred, truth)
- Class probability per sel
- NMS pada output grid
- Pipeline deteksi YOLO end-to-end
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan komputasi numerik
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi hasil
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan header percobaan
print("=" * 60)
print("PERCOBAAN 12: YOLO KONSEP - DETEKSI BERBASIS GRID")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep YOLO
# ============================================================
print("\n--- 1. Konsep YOLO (You Only Look Once) ---")

# Menjelaskan konsep YOLO
print("""
  YOLO (You Only Look Once) adalah metode deteksi objek real-time:

  1. Bagi gambar menjadi grid SxS (misal 7x7 atau 13x13)
  2. Setiap sel grid bertanggung jawab mendeteksi objek
     yang pusatnya jatuh di dalam sel tersebut
  3. Setiap sel memprediksi B bounding boxes, masing-masing terdiri dari:
     - (x, y): pusat box relatif terhadap sel
     - (w, h): ukuran box relatif terhadap gambar
     - confidence: P(objek) * IoU(pred, truth)
  4. Setiap sel juga memprediksi C class probabilities
  5. Total prediksi: S x S x (B * 5 + C) tensor

  Keunggulan YOLO vs Sliding Window:
  - Satu pass: melihat seluruh gambar sekaligus
  - Sangat cepat (real-time)
  - Memahami konteks global gambar
""")

# ============================================================
# 2. Memuat gambar untuk deteksi
# ============================================================
print("\n--- 2. Memuat Gambar ---")

# Memuat gambar scene untuk demonstrasi
img = cv2.imread(os.path.join(IMAGE_DIR, "scene_traffic.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img is None:
    # Mencoba alternatif
    img = cv2.imread(os.path.join(IMAGE_DIR, "scene_outdoor.jpg"))
    if img is None:
        img = cv2.imread(os.path.join(IMAGE_DIR, "mobil.jpg"))
        if img is None:
            # Membuat gambar sintetis dengan objek-objek
            print("  [INFO] Membuat gambar sintetis dengan objek...")
            img = np.ones((448, 448, 3), dtype=np.uint8) * 200

            # Menggambar sky (langit biru muda)
            img[:150, :] = [230, 200, 150]

            # Menggambar road (jalan abu-abu)
            img[300:, :] = [100, 100, 100]

            # Menggambar "mobil" merah
            cv2.rectangle(img, (50, 280), (150, 340), (0, 0, 200), -1)
            cv2.rectangle(img, (70, 260), (130, 280), (0, 0, 180), -1)

            # Menggambar "mobil" biru
            cv2.rectangle(img, (250, 290), (370, 360), (200, 50, 0), -1)
            cv2.rectangle(img, (280, 270), (340, 290), (180, 40, 0), -1)

            # Menggambar "orang"
            cv2.rectangle(img, (180, 200), (210, 240), (60, 60, 60), -1)
            cv2.rectangle(img, (175, 240), (215, 320), (80, 80, 80), -1)

            # Menggambar "pohon"
            cv2.rectangle(img, (380, 150), (410, 300), (30, 100, 30), -1)
            cv2.circle(img, (395, 130), 40, (20, 150, 20), -1)

            # Menambahkan noise
            noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)

# Meresize gambar ke ukuran YOLO standar (448x448)
img = cv2.resize(img, (448, 448))

# Menampilkan informasi gambar
print(f"  Ukuran gambar: {img.shape}")

# ============================================================
# 3. Membagi gambar menjadi grid
# ============================================================
print("\n--- 3. Membagi Gambar Menjadi Grid ---")

# Mendefinisikan ukuran grid
S = 7  # Grid 7x7 (seperti YOLO v1)

# Menghitung ukuran sel dalam piksel
h_img, w_img = img.shape[:2]
h_sel = h_img // S
w_sel = w_img // S

# Menampilkan informasi grid
print(f"  Grid size    : {S} x {S}")
print(f"  Ukuran sel   : {w_sel} x {h_sel} piksel")
print(f"  Total sel    : {S * S}")

# Menggambar grid pada gambar
img_grid = img.copy()

# Menggambar garis vertikal grid
for i in range(1, S):
    # Menghitung posisi X garis vertikal
    x = i * w_sel

    # Menggambar garis vertikal
    cv2.line(img_grid, (x, 0), (x, h_img), (0, 255, 0), 1)

# Menggambar garis horizontal grid
for j in range(1, S):
    # Menghitung posisi Y garis horizontal
    y = j * h_sel

    # Menggambar garis horizontal
    cv2.line(img_grid, (0, y), (w_img, y), (0, 255, 0), 1)

# Menambahkan nomor sel
for sy in range(S):
    for sx in range(S):
        # Menghitung posisi teks (pojok kiri atas sel)
        teks_x = sx * w_sel + 2
        teks_y = sy * h_sel + 12

        # Menuliskan nomor sel
        cv2.putText(img_grid, f"{sy},{sx}", (teks_x, teks_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

# ============================================================
# 4. Simulasi output YOLO per sel
# ============================================================
print("\n--- 4. Simulasi Output YOLO per Sel ---")

# Mendefinisikan kelas objek yang bisa dideteksi
kelas_objek = ["mobil", "orang", "pohon", "background"]
jumlah_kelas = len(kelas_objek)

# Mendefinisikan jumlah bounding box per sel
B = 2  # Setiap sel memprediksi 2 bounding box

# Menjelaskan format output
print(f"  Jumlah kelas (C)          : {jumlah_kelas}")
print(f"  Bounding box per sel (B)  : {B}")
print(f"  Output per sel            : {B}x5 + {jumlah_kelas} = {B*5 + jumlah_kelas}")
print(f"  Total output tensor       : {S}x{S}x{B*5 + jumlah_kelas} = "
      f"{S*S*(B*5 + jumlah_kelas)}")

# Membuat simulasi prediksi YOLO
# Menyiapkan array untuk confidence score setiap sel
confidence_map = np.zeros((S, S))

# Menyiapkan list untuk bounding box predictions
prediksi_boxes = []

# Mendefinisikan objek "ground truth" yang ada dalam gambar (simulasi)
objek_gt = [
    {"kelas": 0, "nama": "mobil", "cx": 100, "cy": 310, "w": 100, "h": 60},
    {"kelas": 0, "nama": "mobil", "cx": 310, "cy": 325, "w": 120, "h": 70},
    {"kelas": 1, "nama": "orang", "cx": 195, "cy": 260, "w": 40, "h": 120},
    {"kelas": 2, "nama": "pohon", "cx": 395, "cy": 220, "w": 50, "h": 170},
]

# Menampilkan objek yang ada dalam gambar
print("\n  Objek dalam gambar (ground truth):")
for obj in objek_gt:
    print(f"    {obj['nama']:10s} - pusat:({obj['cx']},{obj['cy']}) "
          f"ukuran:({obj['w']}x{obj['h']})")

# Mensimulasikan prediksi YOLO
for obj in objek_gt:
    # Menentukan sel grid mana yang bertanggung jawab
    sel_x = min(obj['cx'] // w_sel, S - 1)
    sel_y = min(obj['cy'] // h_sel, S - 1)

    # Mengatur confidence score tinggi untuk sel ini
    confidence_map[sel_y, sel_x] = np.random.uniform(0.7, 0.98)

    # Menghitung posisi relatif terhadap sel
    x_rel = (obj['cx'] - sel_x * w_sel) / w_sel
    y_rel = (obj['cy'] - sel_y * h_sel) / h_sel

    # Menghitung ukuran relatif terhadap gambar
    w_rel = obj['w'] / w_img
    h_rel = obj['h'] / h_img

    # Menyimpan prediksi bounding box
    prediksi_boxes.append({
        'sel_x': sel_x, 'sel_y': sel_y,
        'x_rel': x_rel, 'y_rel': y_rel,
        'w_rel': w_rel, 'h_rel': h_rel,
        'confidence': confidence_map[sel_y, sel_x],
        'kelas': obj['kelas'],
        'nama': obj['nama'],
        # Menghitung koordinat absolut bounding box
        'x1': obj['cx'] - obj['w'] // 2,
        'y1': obj['cy'] - obj['h'] // 2,
        'x2': obj['cx'] + obj['w'] // 2,
        'y2': obj['cy'] + obj['h'] // 2,
    })

    # Menampilkan prediksi
    print(f"\n    Objek '{obj['nama']}' -> Sel ({sel_x}, {sel_y})")
    print(f"      Posisi relatif : ({x_rel:.2f}, {y_rel:.2f})")
    print(f"      Ukuran relatif : ({w_rel:.3f}, {h_rel:.3f})")
    print(f"      Confidence     : {confidence_map[sel_y, sel_x]:.2f}")

# Menambahkan noise confidence untuk sel-sel tanpa objek
for sy in range(S):
    for sx in range(S):
        if confidence_map[sy, sx] == 0:
            # Memberikan confidence rendah (noise) untuk sel kosong
            confidence_map[sy, sx] = np.random.uniform(0.01, 0.15)

# ============================================================
# 5. Visualisasi grid dan confidence map
# ============================================================
print("\n--- 5. Visualisasi Grid dan Confidence Map ---")

# Membuat figure untuk visualisasi grid
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# --- Subplot 1: Gambar dengan grid ---
axes[0].imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Gambar dengan Grid {S}x{S}", fontsize=12, fontweight='bold')
axes[0].axis('off')

# --- Subplot 2: Confidence score map ---
# Menampilkan confidence map sebagai heatmap
im = axes[1].imshow(confidence_map, cmap='hot', interpolation='nearest',
                     vmin=0, vmax=1)

# Menambahkan nilai confidence di setiap sel
for sy in range(S):
    for sx in range(S):
        # Memilih warna teks berdasarkan nilai
        warna_teks = 'white' if confidence_map[sy, sx] > 0.5 else 'black'
        axes[1].text(sx, sy, f"{confidence_map[sy, sx]:.2f}",
                     ha='center', va='center', fontsize=7, color=warna_teks,
                     fontweight='bold')

# Menambahkan colorbar
plt.colorbar(im, ax=axes[1], fraction=0.046)

# Mengatur judul
axes[1].set_title("Confidence Score per Sel", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Sel X")
axes[1].set_ylabel("Sel Y")

# --- Subplot 3: Grid 13x13 untuk perbandingan ---
S2 = 13  # Grid lebih detail
img_grid2 = img.copy()
h_sel2 = h_img // S2
w_sel2 = w_img // S2

# Menggambar garis vertikal grid 13x13
for i in range(1, S2):
    x = i * w_sel2
    cv2.line(img_grid2, (x, 0), (x, h_img), (255, 255, 0), 1)

# Menggambar garis horizontal grid 13x13
for j in range(1, S2):
    y = j * h_sel2
    cv2.line(img_grid2, (0, y), (w_img, y), (255, 255, 0), 1)

# Menampilkan grid 13x13
axes[2].imshow(cv2.cvtColor(img_grid2, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Grid {S2}x{S2} (lebih detail)", fontsize=12, fontweight='bold')
axes[2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 12: YOLO - Pembagian Grid dan Confidence Score",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi grid
plt.savefig(os.path.join(OUTPUT_DIR, "12_yolo_grid.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/12_yolo_grid.png")

# ============================================================
# 6. Visualisasi prediksi bounding box dari grid
# ============================================================
print("\n--- 6. Visualisasi Prediksi Bounding Box ---")

# Mendefinisikan warna per kelas objek
warna_kelas = {
    0: (0, 0, 255),   # mobil: merah
    1: (0, 255, 0),   # orang: hijau
    2: (255, 0, 0),   # pohon: biru
    3: (128, 128, 0), # background: teal
}

# Mendefinisikan warna kelas untuk matplotlib (RGB float)
warna_kelas_plt = {
    0: 'red',
    1: 'lime',
    2: 'blue',
    3: 'cyan',
}

# Membuat figure untuk prediksi
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# --- Subplot 1: Ground truth boxes ---
img_gt = img.copy()

# Menggambar grid tipis
for i in range(1, S):
    cv2.line(img_gt, (i * w_sel, 0), (i * w_sel, h_img), (128, 128, 128), 1)
for j in range(1, S):
    cv2.line(img_gt, (0, j * h_sel), (w_img, j * h_sel), (128, 128, 128), 1)

# Menggambar ground truth bounding box
for obj in objek_gt:
    # Menghitung koordinat box
    x1 = obj['cx'] - obj['w'] // 2
    y1 = obj['cy'] - obj['h'] // 2
    x2 = obj['cx'] + obj['w'] // 2
    y2 = obj['cy'] + obj['h'] // 2

    # Menggambar rectangle
    warna = warna_kelas[obj['kelas']]
    cv2.rectangle(img_gt, (x1, y1), (x2, y2), warna, 2)

    # Menuliskan nama kelas
    cv2.putText(img_gt, obj['nama'], (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)

    # Menandai pusat objek
    cv2.circle(img_gt, (obj['cx'], obj['cy']), 4, warna, -1)

# Menampilkan ground truth
axes[0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
axes[0].set_title("Ground Truth + Grid", fontsize=12, fontweight='bold')
axes[0].axis('off')

# --- Subplot 2: Prediksi YOLO (semua boxes) ---
img_pred = img.copy()

# Menggambar grid
for i in range(1, S):
    cv2.line(img_pred, (i * w_sel, 0), (i * w_sel, h_img), (128, 128, 128), 1)
for j in range(1, S):
    cv2.line(img_pred, (0, j * h_sel), (w_img, j * h_sel), (128, 128, 128), 1)

# Menambahkan beberapa prediksi "noise" (false positives) untuk NMS
prediksi_semua = list(prediksi_boxes)

# Menambahkan prediksi duplikat/noise
for pred in prediksi_boxes:
    # Membuat prediksi geser sedikit (simulasi bounding box kedua per sel)
    pred_noise = pred.copy()
    pred_noise['x1'] = pred['x1'] + np.random.randint(-15, 15)
    pred_noise['y1'] = pred['y1'] + np.random.randint(-15, 15)
    pred_noise['x2'] = pred['x2'] + np.random.randint(-15, 15)
    pred_noise['y2'] = pred['y2'] + np.random.randint(-15, 15)
    pred_noise['confidence'] = pred['confidence'] * np.random.uniform(0.6, 0.9)
    prediksi_semua.append(pred_noise)

# Menggambar semua prediksi bounding box
for pred in prediksi_semua:
    # Menggambar rectangle
    warna = warna_kelas[pred['kelas']]
    cv2.rectangle(img_pred, (pred['x1'], pred['y1']),
                  (pred['x2'], pred['y2']), warna, 2)

    # Menuliskan confidence
    cv2.putText(img_pred, f"{pred['confidence']:.2f}",
                (pred['x1'], pred['y1'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, warna, 1)

# Menampilkan prediksi
axes[1].imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Semua Prediksi ({len(prediksi_semua)} boxes)",
                  fontsize=12, fontweight='bold')
axes[1].axis('off')

# --- Subplot 3: Prediksi setelah threshold + marking sel ---
img_thresh = img.copy()

# Menggambar grid
for i in range(1, S):
    cv2.line(img_thresh, (i * w_sel, 0), (i * w_sel, h_img), (128, 128, 128), 1)
for j in range(1, S):
    cv2.line(img_thresh, (0, j * h_sel), (w_img, j * h_sel), (128, 128, 128), 1)

# Menandai sel yang bertanggung jawab untuk deteksi
for pred in prediksi_boxes:
    # Menghitung posisi sel
    sx = pred['sel_x']
    sy = pred['sel_y']

    # Membuat overlay semi-transparan untuk sel ini
    overlay_sel = img_thresh.copy()
    warna = warna_kelas[pred['kelas']]
    cv2.rectangle(overlay_sel, (sx * w_sel, sy * h_sel),
                  ((sx + 1) * w_sel, (sy + 1) * h_sel), warna, -1)
    cv2.addWeighted(overlay_sel, 0.3, img_thresh, 0.7, 0, img_thresh)

    # Menggambar bounding box prediksi
    cv2.rectangle(img_thresh, (pred['x1'], pred['y1']),
                  (pred['x2'], pred['y2']), warna, 3)

    # Menuliskan kelas dan confidence
    label = f"{pred['nama']} {pred['confidence']:.2f}"
    cv2.putText(img_thresh, label, (pred['x1'], pred['y1'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)

# Menampilkan deteksi setelah threshold
axes[2].imshow(cv2.cvtColor(img_thresh, cv2.COLOR_BGR2RGB))
axes[2].set_title("Sel Responsible + Prediksi Final",
                  fontsize=12, fontweight='bold')
axes[2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 12: YOLO - Prediksi Bounding Box dari Grid",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi prediksi
plt.savefig(os.path.join(OUTPUT_DIR, "12_yolo_prediksi.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/12_yolo_prediksi.png")

# ============================================================
# 7. Implementasi NMS pada prediksi grid
# ============================================================
print("\n--- 7. NMS pada Prediksi Grid ---")

def hitung_iou(box1, box2):
    """
    Menghitung IoU antara dua box dalam format [x1, y1, x2, y2].
    """
    # Menghitung koordinat intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    # Menghitung area intersection
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Menghitung area masing-masing box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Menghitung dan mengembalikan IoU
    return inter / (area1 + area2 - inter + 1e-7)


def nms_yolo(prediksi_list, iou_threshold=0.4):
    """
    Menerapkan NMS pada prediksi YOLO.
    Melakukan NMS per kelas objek.
    """
    # Menyiapkan list untuk hasil akhir
    hasil_nms = []

    # Mendapatkan semua kelas unik
    kelas_unik = set([p['kelas'] for p in prediksi_list])

    # Melakukan NMS per kelas
    for kelas in kelas_unik:
        # Mengambil prediksi untuk kelas ini saja
        pred_kelas = [p for p in prediksi_list if p['kelas'] == kelas]

        # Mengurutkan berdasarkan confidence (tinggi ke rendah)
        pred_kelas.sort(key=lambda x: x['confidence'], reverse=True)

        # Menyiapkan list untuk prediksi yang dipertahankan
        kept = []

        # Melakukan iterasi NMS
        while len(pred_kelas) > 0:
            # Mengambil prediksi dengan confidence tertinggi
            best = pred_kelas.pop(0)
            kept.append(best)

            # Membuang prediksi yang overlap tinggi
            pred_kelas = [
                p for p in pred_kelas
                if hitung_iou(
                    [best['x1'], best['y1'], best['x2'], best['y2']],
                    [p['x1'], p['y1'], p['x2'], p['y2']]
                ) < iou_threshold
            ]

        # Menambahkan hasil NMS kelas ini
        hasil_nms.extend(kept)

    # Mengembalikan hasil NMS
    return hasil_nms


# Menerapkan NMS pada semua prediksi
prediksi_nms = nms_yolo(prediksi_semua, iou_threshold=0.4)

# Menampilkan hasil NMS
print(f"  Prediksi sebelum NMS: {len(prediksi_semua)}")
print(f"  Prediksi setelah NMS: {len(prediksi_nms)}")

# Menampilkan detail prediksi final
for pred in prediksi_nms:
    print(f"    {pred['nama']:10s} - conf:{pred['confidence']:.2f} "
          f"box:({pred['x1']},{pred['y1']})-({pred['x2']},{pred['y2']})")

# ============================================================
# 8. Visualisasi pipeline YOLO end-to-end
# ============================================================
print("\n--- 8. Visualisasi Pipeline YOLO End-to-End ---")

# Membuat figure 2x2 untuk pipeline langkah per langkah
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# --- Step 1: Input gambar ---
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Step 1: Input Gambar", fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# --- Step 2: Grid division + confidence ---
img_step2 = img.copy()

# Menggambar grid dengan warna sesuai confidence
for sy in range(S):
    for sx in range(S):
        # Menghitung warna berdasarkan confidence
        conf = confidence_map[sy, sx]
        if conf > 0.5:
            # Membuat overlay untuk sel dengan confidence tinggi
            overlay_s2 = img_step2.copy()
            intensitas = int(255 * conf)
            cv2.rectangle(overlay_s2, (sx * w_sel, sy * h_sel),
                          ((sx + 1) * w_sel, (sy + 1) * h_sel),
                          (0, intensitas, 0), -1)
            cv2.addWeighted(overlay_s2, 0.4, img_step2, 0.6, 0, img_step2)

# Menggambar grid lines
for i in range(1, S):
    cv2.line(img_step2, (i * w_sel, 0), (i * w_sel, h_img), (255, 255, 255), 1)
for j in range(1, S):
    cv2.line(img_step2, (0, j * h_sel), (w_img, j * h_sel), (255, 255, 255), 1)

# Menampilkan step 2
axes[0, 1].imshow(cv2.cvtColor(img_step2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Step 2: Grid + Confidence (hijau=tinggi)",
                      fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# --- Step 3: Semua prediksi bounding box ---
img_step3 = img.copy()

# Menggambar semua prediksi
for pred in prediksi_semua:
    warna = warna_kelas[pred['kelas']]
    cv2.rectangle(img_step3, (pred['x1'], pred['y1']),
                  (pred['x2'], pred['y2']), warna, 1)

# Menampilkan step 3
axes[1, 0].imshow(cv2.cvtColor(img_step3, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f"Step 3: Semua Prediksi ({len(prediksi_semua)} boxes)",
                      fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# --- Step 4: Hasil setelah NMS ---
img_step4 = img.copy()

# Menggambar prediksi setelah NMS dengan label
for pred in prediksi_nms:
    warna = warna_kelas[pred['kelas']]
    cv2.rectangle(img_step4, (pred['x1'], pred['y1']),
                  (pred['x2'], pred['y2']), warna, 3)
    label = f"{pred['nama']} {pred['confidence']:.2f}"

    # Menggambar background untuk teks
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img_step4, (pred['x1'], pred['y1'] - th - 8),
                  (pred['x1'] + tw + 4, pred['y1']), warna, -1)
    cv2.putText(img_step4, label, (pred['x1'] + 2, pred['y1'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Menampilkan step 4
axes[1, 1].imshow(cv2.cvtColor(img_step4, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"Step 4: Setelah NMS ({len(prediksi_nms)} deteksi final)",
                      fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 12: Pipeline Deteksi YOLO - Grid -> Prediksi -> NMS",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi NMS YOLO
plt.savefig(os.path.join(OUTPUT_DIR, "12_yolo_nms.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/12_yolo_nms.png")

# ============================================================
# 9. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 12")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. YOLO membagi gambar menjadi grid SxS untuk deteksi satu-pass
2. Setiap sel grid memprediksi B bounding box + confidence
3. Confidence = P(objek) x IoU(prediksi, ground truth)
4. Setiap sel juga memprediksi probabilitas kelas objek
5. Grid yang lebih detail (13x13 vs 7x7) mendeteksi objek lebih kecil
6. NMS per kelas menghilangkan prediksi duplikat
7. Pipeline YOLO: Input -> Grid -> Prediksi -> NMS -> Deteksi Final
8. YOLO sangat cepat karena hanya satu forward pass (You Only Look Once)

Output disimpan di folder: output/
- 12_yolo_grid.png     : Grid overlay dan confidence map
- 12_yolo_prediksi.png : Prediksi bounding box dari grid
- 12_yolo_nms.png      : Pipeline YOLO end-to-end
""")
