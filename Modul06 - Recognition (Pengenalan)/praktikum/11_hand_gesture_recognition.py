"""
==========================================================================
PERCOBAAN 11: HAND GESTURE RECOGNITION (PENGENALAN GESTUR TANGAN)
==========================================================================
Program ini mempelajari cara mendeteksi dan mengenali gestur tangan
menggunakan segmentasi warna kulit, analisis kontur, convex hull, dan
convexity defects. Jumlah jari ditentukan dari analisis sudut defects.

Konsep yang dipelajari:
- Segmentasi warna kulit menggunakan ruang warna YCrCb
- Convex hull: poligon terkecil yang melingkupi kontur
- Convexity defects: titik lekukan antara kontur dan hull
- Analisis sudut untuk menghitung jumlah jari
- Klasifikasi gestur berdasarkan jumlah jari

Fungsi utama yang dipelajari:
- cv2.cvtColor(COLOR_BGR2YCrCb)  : Konversi ke ruang warna YCrCb
- cv2.inRange()                   : Segmentasi rentang warna kulit
- cv2.convexHull()                : Menghitung convex hull
- cv2.convexityDefects()          : Menghitung convexity defects
- cv2.findContours()              : Menemukan kontur tangan
- np.arccos()                     : Menghitung sudut untuk deteksi jari

Hasil: Visualisasi segmentasi kulit, convex hull, defects, dan gestur
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan trigonometri
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor math untuk perhitungan sudut
import math

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 11: HAND GESTURE RECOGNITION")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Tangan
# ============================================================

print("\n[INFO] Memuat gambar gestur tangan...")
print("-" * 50)

# Mendefinisikan nama file untuk setiap gestur tangan
gesture_files = {
    "open": "tangan_open.jpg",
    "fist": "tangan_fist.jpg",
    "peace": "tangan_peace.jpg",
    "thumbsup": "tangan_thumbsup.jpg",
    "pointing": "tangan_pointing.jpg"
}

# Mendefinisikan jumlah jari yang diharapkan untuk setiap gestur
expected_fingers = {
    "open": 5,
    "fist": 0,
    "peace": 2,
    "thumbsup": 1,
    "pointing": 1
}

# Membuat dictionary untuk menyimpan gambar yang dimuat
gesture_images = {}

# Memuat setiap gambar gestur
for gesture_name, filename in gesture_files.items():
    # Membaca gambar gestur dari file
    img = cv2.imread(os.path.join(IMAGE_DIR, filename))

    # Memeriksa apakah gambar berhasil dimuat
    if img is not None:
        # Menyimpan gambar ke dictionary
        gesture_images[gesture_name] = img
        print(f"  [{gesture_name}] Dimuat: {img.shape[1]}x{img.shape[0]}")
    else:
        # Menampilkan pesan error jika gambar tidak ditemukan
        print(f"  [{gesture_name}] TIDAK DITEMUKAN: {filename}")

# Memeriksa apakah ada gambar yang berhasil dimuat
if len(gesture_images) == 0:
    print("[ERROR] Tidak ada gambar gestur yang ditemukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()

# ============================================================
# 2. Segmentasi Warna Kulit (Skin Detection)
# ============================================================

print("\n[INFO] Melakukan segmentasi warna kulit...")
print("-" * 50)

# Mendefinisikan fungsi untuk mendeteksi warna kulit menggunakan YCrCb
def detect_skin(image):
    """Mendeteksi region kulit menggunakan ruang warna YCrCb."""
    # Mengkonversi gambar ke ruang warna YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Mendefinisikan range warna kulit dalam YCrCb
    # Y: luminance, Cr: red chroma, Cb: blue chroma
    lower_skin = np.array([0, 133, 77])
    upper_skin = np.array([255, 173, 127])

    # Membuat mask berdasarkan range warna kulit
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Membuat kernel untuk operasi morfologi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Menerapkan opening untuk menghapus noise kecil
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Menerapkan closing untuk mengisi lubang pada mask
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Menerapkan Gaussian blur untuk menghaluskan mask
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

    # Menerapkan threshold untuk memastikan mask biner
    _, skin_mask = cv2.threshold(skin_mask, 127, 255, cv2.THRESH_BINARY)

    # Mengembalikan mask kulit
    return skin_mask

# Mengambil gambar pertama yang tersedia untuk demo segmentasi kulit
first_gesture = list(gesture_images.keys())[0]
demo_img = gesture_images[first_gesture]

# Menerapkan deteksi kulit pada gambar demo
skin_mask_demo = detect_skin(demo_img)

# Membuat gambar kulit yang diisolasi (bitwise AND dengan mask)
skin_isolated = cv2.bitwise_and(demo_img, demo_img, mask=skin_mask_demo)

# Mengkonversi gambar untuk tampilan
demo_rgb = cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB)
skin_iso_rgb = cv2.cvtColor(skin_isolated, cv2.COLOR_BGR2RGB)

# Mengkonversi gambar YCrCb untuk visualisasi
ycrcb_demo = cv2.cvtColor(demo_img, cv2.COLOR_BGR2YCrCb)

# Membuat figure untuk visualisasi segmentasi kulit
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Menampilkan gambar asli
axes[0, 0].imshow(demo_rgb)
axes[0, 0].set_title(f"1. Gambar Asli ({first_gesture})", fontsize=11, fontweight="bold")
axes[0, 0].axis("off")

# Menampilkan channel Cr (red chroma)
axes[0, 1].imshow(ycrcb_demo[:, :, 1], cmap="RdYlGn_r")
axes[0, 1].set_title("2. Channel Cr (Red Chroma)", fontsize=11, fontweight="bold")
axes[0, 1].axis("off")

# Menampilkan mask kulit
axes[1, 0].imshow(skin_mask_demo, cmap="gray")
axes[1, 0].set_title("3. Skin Mask (YCrCb)", fontsize=11, fontweight="bold")
axes[1, 0].axis("off")

# Menampilkan region kulit yang diisolasi
axes[1, 1].imshow(skin_iso_rgb)
axes[1, 1].set_title("4. Kulit Terisolasi", fontsize=11, fontweight="bold")
axes[1, 1].axis("off")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 11: Segmentasi Warna Kulit (YCrCb Color Space)",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi segmentasi kulit
output_path_1 = os.path.join(OUTPUT_DIR, "11_skin_detection.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 3. Convex Hull dan Convexity Defects
# ============================================================

print("\n[INFO] Menghitung convex hull dan convexity defects...")
print("-" * 50)

# Mendefinisikan fungsi untuk menemukan kontur tangan terbesar
def find_hand_contour(skin_mask):
    """Menemukan kontur tangan terbesar dari mask kulit."""
    # Menemukan semua kontur pada mask
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Memeriksa apakah ada kontur yang ditemukan
    if len(contours) == 0:
        return None

    # Mengurutkan kontur berdasarkan area (terbesar pertama)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    # Mengambil kontur terbesar (diasumsikan sebagai tangan)
    hand_contour = contours_sorted[0]

    # Mengembalikan kontur tangan jika areanya cukup besar
    if cv2.contourArea(hand_contour) > 1000:
        return hand_contour
    return None

# Mendefinisikan fungsi untuk menghitung jumlah jari dari convexity defects
def count_fingers(contour, img_shape):
    """Menghitung jumlah jari dari convexity defects."""
    # Menghitung convex hull dengan returnPoints=False untuk mendapatkan indices
    hull_indices = cv2.convexHull(contour, returnPoints=False)

    # Memeriksa apakah hull memiliki cukup titik
    if hull_indices is None or len(hull_indices) < 3:
        return 0, [], [], []

    # Menghitung convexity defects
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return 0, [], [], []

    # Memeriksa apakah defects ditemukan
    if defects is None:
        return 0, [], [], []

    # Menghitung convex hull dengan returnPoints=True untuk menggambar
    hull_points = cv2.convexHull(contour, returnPoints=True)

    # Menginisialisasi counter jari
    finger_count = 0

    # Membuat list untuk menyimpan titik ujung jari dan defects
    finger_tips = []
    defect_points = []
    valid_defects = []

    # Menghitung area kontur untuk referensi
    contour_area = cv2.contourArea(contour)

    # Menganalisis setiap convexity defect
    for i in range(defects.shape[0]):
        # Mengambil data defect: start, end, far (titik paling jauh), distance
        s, e, f, d = defects[i, 0]

        # Mendapatkan koordinat titik start, end, dan far
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Menghitung panjang sisi segitiga menggunakan Euclidean distance
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # Menghitung sudut di titik far menggunakan hukum cosinus
        # cos(angle) = (b^2 + c^2 - a^2) / (2*b*c)
        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c + 1e-7)

        # Membatasi cos_angle ke range [-1, 1] untuk menghindari error
        cos_angle = max(-1, min(1, cos_angle))

        # Menghitung sudut dalam derajat
        angle = math.degrees(math.acos(cos_angle))

        # Menghitung jarak defect dari hull (dalam piksel)
        depth = d / 256.0

        # Memeriksa apakah defect menunjukkan celah antar jari
        # Sudut < 90 derajat dan depth cukup besar
        if angle < 90 and depth > 20:
            # Menambahkan counter jari
            finger_count += 1

            # Menyimpan titik ujung jari
            finger_tips.append(start)

            # Menyimpan titik defect
            defect_points.append(far)

            # Menyimpan data defect untuk visualisasi
            valid_defects.append((start, end, far, angle, depth))

    # Jumlah jari = jumlah celah + 1 (jika ada celah)
    if finger_count > 0:
        finger_count += 1

    # Membatasi jumlah jari maksimum 5
    finger_count = min(finger_count, 5)

    # Mengembalikan jumlah jari, titik ujung, titik defect, hull
    return finger_count, finger_tips, defect_points, valid_defects

# Mengambil gambar demo untuk visualisasi convex hull
demo_skin = detect_skin(demo_img)
demo_contour = find_hand_contour(demo_skin)

# Memeriksa apakah kontur tangan ditemukan
if demo_contour is not None:
    # Menghitung convex hull untuk visualisasi
    hull_vis = cv2.convexHull(demo_contour, returnPoints=True)

    # Membuat gambar untuk menampilkan kontur dan hull
    img_hull = demo_img.copy()

    # Menggambar kontur tangan dengan warna hijau
    cv2.drawContours(img_hull, [demo_contour], -1, (0, 255, 0), 2)

    # Menggambar convex hull dengan warna merah
    cv2.drawContours(img_hull, [hull_vis], -1, (0, 0, 255), 2)

    # Menghitung jumlah jari
    n_fingers, tips, defects_pts, valid_defs = count_fingers(demo_contour, demo_img.shape)

    # Membuat gambar untuk menampilkan defects
    img_defects = demo_img.copy()

    # Menggambar kontur
    cv2.drawContours(img_defects, [demo_contour], -1, (0, 255, 0), 2)

    # Menggambar convex hull
    cv2.drawContours(img_defects, [hull_vis], -1, (0, 0, 255), 2)

    # Menggambar titik defect dan garis ke ujung jari
    for (start, end, far, angle, depth) in valid_defs:
        # Menggambar lingkaran biru pada titik far (lembah antar jari)
        cv2.circle(img_defects, far, 8, (255, 0, 0), -1)

        # Menggambar lingkaran kuning pada titik start (ujung jari)
        cv2.circle(img_defects, start, 6, (0, 255, 255), -1)

        # Menuliskan sudut di dekat titik defect
        cv2.putText(img_defects, f"{angle:.0f}°", (far[0] + 10, far[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Menuliskan jumlah jari pada gambar
    cv2.putText(img_defects, f"Jari: {n_fingers}", (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Mengkonversi gambar ke RGB
    img_hull_rgb = cv2.cvtColor(img_hull, cv2.COLOR_BGR2RGB)
    img_defects_rgb = cv2.cvtColor(img_defects, cv2.COLOR_BGR2RGB)

    # Membuat figure untuk visualisasi hull dan defects
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Menampilkan gambar asli
    axes[0, 0].imshow(cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Gambar Asli", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Menampilkan mask kulit
    axes[0, 1].imshow(demo_skin, cmap="gray")
    axes[0, 1].set_title("2. Skin Mask", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Menampilkan kontur + convex hull
    axes[1, 0].imshow(img_hull_rgb)
    axes[1, 0].set_title("3. Kontur (hijau) + Hull (merah)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Menampilkan defects dan finger counting
    axes[1, 1].imshow(img_defects_rgb)
    axes[1, 1].set_title(f"4. Defects + Jari: {n_fingers}", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # Menambahkan judul utama figure
    plt.suptitle("Percobaan 11: Convex Hull dan Convexity Defects",
                 fontsize=13, fontweight="bold")

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan visualisasi convex hull
    output_path_2 = os.path.join(OUTPUT_DIR, "11_convex_hull.png")
    plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
    print(f"[OUTPUT] Disimpan: {output_path_2}")

    # Menutup figure
    plt.close()
else:
    # Menampilkan pesan jika kontur tidak ditemukan
    print("  [WARNING] Kontur tangan tidak ditemukan pada gambar demo.")

# ============================================================
# 4. Pengenalan Gestur pada Semua Gambar Tangan
# ============================================================

print("\n[INFO] Mengenali gestur pada semua gambar tangan...")
print("-" * 50)

# Membuat dictionary untuk menyimpan hasil pengenalan
recognition_results = {}

# Memproses setiap gambar gestur
for gesture_name, img in gesture_images.items():
    # Melakukan segmentasi kulit
    skin_mask = detect_skin(img)

    # Menemukan kontur tangan
    hand_contour = find_hand_contour(skin_mask)

    # Memeriksa apakah kontur ditemukan
    if hand_contour is not None:
        # Menghitung jumlah jari
        n_fingers, tips, defs, valid = count_fingers(hand_contour, img.shape)

        # Menentukan gestur berdasarkan jumlah jari
        if n_fingers == 0:
            predicted_gesture = "fist"
        elif n_fingers == 1:
            predicted_gesture = "thumbsup/pointing"
        elif n_fingers == 2:
            predicted_gesture = "peace"
        elif n_fingers >= 4:
            predicted_gesture = "open"
        else:
            predicted_gesture = f"{n_fingers} jari"

        # Menyimpan hasil pengenalan
        recognition_results[gesture_name] = {
            "fingers": n_fingers,
            "predicted": predicted_gesture,
            "expected": expected_fingers[gesture_name],
            "contour": hand_contour,
            "skin_mask": skin_mask,
            "valid_defects": valid
        }

        # Menampilkan hasil per gestur
        match = "✓" if n_fingers == expected_fingers[gesture_name] else "✗"
        print(f"  [{gesture_name}] Jari: {n_fingers} (harapan: {expected_fingers[gesture_name]}) "
              f"→ {predicted_gesture} {match}")
    else:
        # Menyimpan hasil jika kontur tidak ditemukan
        recognition_results[gesture_name] = {
            "fingers": -1,
            "predicted": "tidak terdeteksi",
            "expected": expected_fingers[gesture_name],
            "contour": None,
            "skin_mask": skin_mask,
            "valid_defects": []
        }
        print(f"  [{gesture_name}] Kontur tidak ditemukan")

# ============================================================
# 5. Visualisasi Hasil Gestur pada Semua Gambar
# ============================================================

print("\n[INFO] Menyimpan visualisasi hasil gestur...")

# Menentukan jumlah gambar yang tersedia
n_gestures = len(gesture_images)

# Membuat figure untuk menampilkan semua hasil gestur
fig, axes = plt.subplots(2, n_gestures, figsize=(4 * n_gestures, 8))

# Memastikan axes 2D
if n_gestures == 1:
    axes = axes.reshape(-1, 1)

# Menampilkan setiap gestur
for col, (gesture_name, img) in enumerate(gesture_images.items()):
    # Mengambil hasil pengenalan
    result = recognition_results[gesture_name]

    # --- Baris 1: Gambar asli dengan anotasi ---

    # Membuat salinan gambar untuk anotasi
    img_annotated = img.copy()

    # Menggambar kontur dan hull jika tersedia
    if result["contour"] is not None:
        # Menghitung convex hull
        hull = cv2.convexHull(result["contour"], returnPoints=True)

        # Menggambar kontur hijau
        cv2.drawContours(img_annotated, [result["contour"]], -1, (0, 255, 0), 2)

        # Menggambar hull merah
        cv2.drawContours(img_annotated, [hull], -1, (0, 0, 255), 2)

        # Menggambar defects
        for (start, end, far, angle, depth) in result["valid_defects"]:
            # Menggambar titik defect biru
            cv2.circle(img_annotated, far, 5, (255, 0, 0), -1)

    # Menuliskan hasil jari
    label = f"Jari: {result['fingers']}"
    cv2.putText(img_annotated, label, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mengkonversi ke RGB
    img_ann_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)

    # Menampilkan gambar dengan anotasi
    axes[0, col].imshow(img_ann_rgb)
    axes[0, col].set_title(f"{gesture_name}\nJari: {result['fingers']} (harapan: {result['expected']})",
                           fontsize=9, fontweight="bold")
    axes[0, col].axis("off")

    # --- Baris 2: Skin mask ---

    # Menampilkan skin mask
    axes[1, col].imshow(result["skin_mask"], cmap="gray")
    axes[1, col].set_title(f"Skin Mask\nPrediksi: {result['predicted']}", fontsize=9)
    axes[1, col].axis("off")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 11: Pengenalan Gestur Tangan",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi hasil gestur
output_path_3 = os.path.join(OUTPUT_DIR, "11_gesture_result.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# 6. Laporan Finger Counting dan Klasifikasi
# ============================================================

print("\n[INFO] Membuat laporan finger counting...")
print("-" * 50)

# Menghitung akurasi keseluruhan
correct_count = 0
total_count = 0

# Membuat data untuk visualisasi laporan
gesture_names_list = []
expected_list = []
detected_list = []

# Mengumpulkan data dari semua hasil
for gesture_name, result in recognition_results.items():
    # Menambahkan nama gestur ke list
    gesture_names_list.append(gesture_name)

    # Menambahkan jumlah jari yang diharapkan
    expected_list.append(result["expected"])

    # Menambahkan jumlah jari yang terdeteksi
    detected_val = result["fingers"] if result["fingers"] >= 0 else 0
    detected_list.append(detected_val)

    # Menghitung akurasi
    total_count += 1
    if result["fingers"] == result["expected"]:
        correct_count += 1

# Menghitung akurasi keseluruhan
accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

# Menampilkan akurasi
print(f"  Akurasi: {correct_count}/{total_count} = {accuracy:.1f}%")

# Membuat figure untuk laporan finger counting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Membuat grouped bar chart perbandingan expected vs detected
x_pos = np.arange(len(gesture_names_list))
bar_width = 0.35

# Menggambar bar chart jumlah jari expected
bars1 = axes[0].bar(x_pos - bar_width/2, expected_list, bar_width,
                    label="Expected", color="steelblue", edgecolor="black")

# Menggambar bar chart jumlah jari detected
bars2 = axes[0].bar(x_pos + bar_width/2, detected_list, bar_width,
                    label="Detected", color="coral", edgecolor="black")

# Mengatur label sumbu x
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(gesture_names_list, rotation=30, ha="right")
axes[0].set_ylabel("Jumlah Jari")
axes[0].set_title("Perbandingan Jumlah Jari", fontsize=11, fontweight="bold")
axes[0].legend()
axes[0].set_ylim(0, 6)

# Menambahkan label angka pada setiap bar
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f"{int(bar.get_height())}", ha='center', fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f"{int(bar.get_height())}", ha='center', fontsize=9)

# Membuat tabel laporan klasifikasi
table_data = []

# Menyusun data tabel
for gesture_name, result in recognition_results.items():
    # Menentukan status benar/salah
    status = "Benar" if result["fingers"] == result["expected"] else "Salah"

    # Menambahkan baris ke tabel
    table_data.append([gesture_name, result["expected"],
                      result["fingers"], result["predicted"], status])

# Menampilkan tabel pada subplot
axes[1].axis("off")

# Mendefinisikan header kolom
col_labels = ["Gestur", "Jari\nExpected", "Jari\nDetected", "Prediksi", "Status"]

# Membuat tabel
table = axes[1].table(cellText=table_data, colLabels=col_labels,
                      loc="center", cellLoc="center")

# Mengatur ukuran font tabel
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.6)

# Mewarnai header tabel
for j, label in enumerate(col_labels):
    table[0, j].set_facecolor("#4472C4")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Mewarnai baris berdasarkan status
for i, row in enumerate(table_data):
    # Menentukan warna berdasarkan status
    if row[4] == "Benar":
        bg_color = "#C6EFCE"
    else:
        bg_color = "#FFC7CE"

    # Menerapkan warna pada setiap kolom
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(bg_color)

# Menambahkan judul tabel
axes[1].set_title(f"Laporan Klasifikasi (Akurasi: {accuracy:.1f}%)",
                  fontsize=11, fontweight="bold")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 11: Finger Counting Report",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi finger counting
output_path_4 = os.path.join(OUTPUT_DIR, "11_finger_counting.png")
plt.savefig(output_path_4, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_4}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 11")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Segmentasi Warna Kulit (Skin Detection):")
print("     - cv2.cvtColor(COLOR_BGR2YCrCb): ruang warna yang memisahkan")
print("       luminance dari chroma, baik untuk deteksi kulit")
print("     - cv2.inRange(): segmentasi range Cr=[133,173], Cb=[77,127]")
print("  2. Analisis Kontur Tangan:")
print("     - cv2.findContours(): menemukan kontur tangan")
print("     - cv2.contourArea(): menghitung area kontur")
print("  3. Convex Hull dan Convexity Defects:")
print("     - cv2.convexHull(): poligon cembung terkecil yang melingkupi")
print("     - cv2.convexityDefects(): titik lekukan antara kontur & hull")
print("       → (start_idx, end_idx, far_idx, distance)")
print("  4. Finger Counting:")
print("     - Hukum cosinus: cos(θ) = (b²+c²-a²)/(2bc)")
print("     - Sudut < 90° + depth > threshold → celah antar jari")
print("     - Jumlah jari = celah + 1 (maks 5)")
print(f"\nHasil pengenalan gestur:")
for gesture_name, result in recognition_results.items():
    match = "✓" if result["fingers"] == result["expected"] else "✗"
    print(f"  - {gesture_name:10s}: {result['fingers']} jari → {result['predicted']} {match}")
print(f"\n  Akurasi keseluruhan: {accuracy:.1f}%")
print("=" * 60)
