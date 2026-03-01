"""
==========================================================================
PERCOBAAN 19: COMPLETE RECOGNITION PIPELINE
==========================================================================
Program ini membangun pipeline recognition end-to-end yang lengkap,
mulai dari deteksi wajah, preprocessing, ekstraksi fitur, klasifikasi,
hingga visualisasi hasil. Setiap langkah pipeline diukur waktunya dan
divisualisasikan secara individual.

Konsep yang dipelajari:
- Pipeline recognition: rangkaian langkah dari input ke output
- Step 1: Face detection (Haar Cascade Classifier)
- Step 2: Face preprocessing (resize, grayscale, histogram equalization)
- Step 3: Feature extraction (LBP histogram)
- Step 4: Classification (nearest neighbor dengan thresholding)
- Step 5: Visualization (bounding box, label, confidence score)
- Pipeline timing: mengukur waktu setiap langkah
- Pipeline flow diagram: visualisasi alur proses

Fungsi utama yang dipelajari:
- cv2.CascadeClassifier.detectMultiScale() : Deteksi wajah
- cv2.equalizeHist()                       : Histogram equalization
- cv2.resize()                             : Normalisasi ukuran wajah
- LBP histogram (manual)                   : Ekstraksi fitur tekstur
- np.linalg.norm()                         : Jarak nearest neighbor
- cv2.rectangle() / cv2.putText()          : Visualisasi bounding box

Hasil: Visualisasi setiap step pipeline dan hasil final
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan deteksi wajah
import cv2

# Mengimpor NumPy untuk operasi array dan perhitungan fitur
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu setiap langkah pipeline
import time

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 19: COMPLETE RECOGNITION PIPELINE")
print("=" * 60)

# ============================================================
# 1. Setup Pipeline Components
# ============================================================

print("\n[INFO] Menyiapkan komponen pipeline...")
print("-" * 50)

# Memuat Haar Cascade Classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Memeriksa apakah cascade berhasil dimuat
print(f"  Face cascade dimuat: {not face_cascade.empty()}")

# Mendefinisikan ukuran standar untuk normalisasi wajah
FACE_SIZE = (100, 100)

# Mendefinisikan jumlah bins untuk LBP histogram
LBP_BINS = 256

# Mendefinisikan threshold jarak untuk klasifikasi (reject jika terlalu jauh)
DISTANCE_THRESHOLD = 150.0

# Mendefinisikan folder wajah untuk database
face_folders = {
    "andi": os.path.join(IMAGE_DIR, "faces", "andi"),
    "budi": os.path.join(IMAGE_DIR, "faces", "budi"),
    "citra": os.path.join(IMAGE_DIR, "faces", "citra")
}

# Membuat dictionary untuk menyimpan waktu setiap step
pipeline_times = {}

print(f"  Ukuran normalisasi: {FACE_SIZE}")
print(f"  LBP bins: {LBP_BINS}")
print(f"  Distance threshold: {DISTANCE_THRESHOLD}")


# ============================================================
# 2. Step 1: Face Detection
# ============================================================

print("\n[STEP 1] Face Detection - Mendeteksi wajah pada gambar...")
print("-" * 50)


# Mendefinisikan fungsi untuk deteksi wajah
def step1_face_detection(image, cascade):
    """
    Step 1 Pipeline: Mendeteksi wajah pada gambar menggunakan Haar Cascade.
    Mengembalikan list bounding box wajah yang terdeteksi.
    """
    # Mencatat waktu mulai
    t_start = time.time()

    # Mengkonversi ke grayscale jika belum
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Mendeteksi wajah menggunakan Haar Cascade
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Menghitung waktu proses
    elapsed = time.time() - t_start

    # Mengembalikan hasil deteksi dan waktu
    return faces, gray, elapsed


# Memuat gambar wajah dari database untuk enrollment
print("  Memuat gambar wajah dari database...")

# Membuat database wajah
face_database = {}
enrollment_results = {}

# Memuat dan mendeteksi wajah dari setiap folder
for person_name, folder_path in face_folders.items():
    # Memeriksa apakah folder ada
    if not os.path.exists(folder_path):
        print(f"    [{person_name}] Folder tidak ditemukan")
        continue

    # Mendapatkan daftar file gambar
    image_files = sorted([f for f in os.listdir(folder_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # Memuat gambar pertama sebagai representative
    if len(image_files) > 0:
        # Membaca gambar
        img_path = os.path.join(folder_path, image_files[0])
        img = cv2.imread(img_path)

        if img is not None:
            # Mendeteksi wajah pada gambar enrollment
            faces_det, gray, det_time = step1_face_detection(img, face_cascade)

            # Menyimpan hasil enrollment
            enrollment_results[person_name] = {
                "image": img,
                "gray": gray,
                "faces": faces_det,
                "det_time": det_time,
                "n_files": len(image_files)
            }

            # Menampilkan info
            print(f"    [{person_name}] {len(faces_det)} wajah terdeteksi "
                  f"({det_time*1000:.1f}ms), {len(image_files)} gambar tersedia")

# Memuat gambar test (wajah tunggal)
test_image_path = os.path.join(IMAGE_DIR, "wajah_single.jpg")
test_image = cv2.imread(test_image_path)

# Jika gambar test tidak ada, buat gambar sintetis
if test_image is None:
    print("  [INFO] Gambar test tidak ditemukan, membuat gambar sintetis...")

    # Membuat gambar sintetis dengan wajah sederhana
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 200

    # Menggambar elips wajah
    cv2.ellipse(test_image, (200, 150), (60, 80), 0, 0, 360, (180, 160, 140), -1)

    # Menggambar mata
    cv2.circle(test_image, (175, 130), 8, (40, 30, 20), -1)
    cv2.circle(test_image, (225, 130), 8, (40, 30, 20), -1)

    # Menggambar mulut
    cv2.ellipse(test_image, (200, 170), (20, 8), 0, 0, 180, (80, 50, 40), 2)

# Mendeteksi wajah pada gambar test
test_faces, test_gray, test_det_time = step1_face_detection(
    test_image, face_cascade
)

# Menyimpan waktu Step 1
pipeline_times["Step 1: Detection"] = test_det_time

print(f"\n  Gambar test: {test_image.shape[:2]}")
print(f"  Wajah terdeteksi: {len(test_faces)}")
print(f"  Waktu deteksi: {test_det_time*1000:.2f} ms")

# ============================================================
# 3. Step 2: Face Preprocessing
# ============================================================

print("\n[STEP 2] Face Preprocessing - Normalisasi dan enhancement...")
print("-" * 50)


# Mendefinisikan fungsi untuk preprocessing wajah
def step2_preprocessing(image_gray, face_bbox, target_size=(100, 100)):
    """
    Step 2 Pipeline: Preprocessing wajah - crop, resize, equalize.
    """
    # Mencatat waktu mulai
    t_start = time.time()

    # Mengekstrak region wajah dari gambar
    x, y, w, h = face_bbox
    face_roi = image_gray[y:y+h, x:x+w]

    # Meresize wajah ke ukuran standar
    face_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_LINEAR)

    # Menerapkan histogram equalization untuk normalisasi kontras
    face_equalized = cv2.equalizeHist(face_resized)

    # Menerapkan Gaussian blur ringan untuk mengurangi noise
    face_smoothed = cv2.GaussianBlur(face_equalized, (3, 3), 0)

    # Menghitung waktu proses
    elapsed = time.time() - t_start

    # Mengembalikan hasil preprocessing
    return {
        "roi": face_roi,
        "resized": face_resized,
        "equalized": face_equalized,
        "final": face_smoothed
    }, elapsed


# Memproses wajah yang terdeteksi pada gambar test
preprocessed_faces = []
preprocess_time_total = 0

# Memproses setiap wajah terdeteksi
if len(test_faces) > 0:
    for face_bbox in test_faces:
        # Melakukan preprocessing
        result, prep_time = step2_preprocessing(test_gray, face_bbox, FACE_SIZE)
        preprocessed_faces.append(result)
        preprocess_time_total += prep_time
else:
    # Jika tidak ada deteksi, gunakan seluruh gambar sebagai fallback
    print("  [WARN] Tidak ada wajah terdeteksi, menggunakan seluruh gambar")

    # Meresize seluruh gambar test
    face_resized = cv2.resize(test_gray, FACE_SIZE)
    face_equalized = cv2.equalizeHist(face_resized)
    face_smoothed = cv2.GaussianBlur(face_equalized, (3, 3), 0)

    # Mengukur waktu
    t_start = time.time()
    preprocess_time_total = time.time() - t_start

    # Menyimpan hasil
    preprocessed_faces.append({
        "roi": test_gray,
        "resized": face_resized,
        "equalized": face_equalized,
        "final": face_smoothed
    })

# Menyimpan waktu Step 2
pipeline_times["Step 2: Preprocessing"] = preprocess_time_total

print(f"  Wajah dipreproses: {len(preprocessed_faces)}")
print(f"  Target size: {FACE_SIZE}")
print(f"  Steps: crop → resize → equalize → smooth")
print(f"  Waktu preprocessing: {preprocess_time_total*1000:.2f} ms")

# Memproses juga wajah dari database enrollment
db_preprocessed = {}
for person_name, er in enrollment_results.items():
    if len(er["faces"]) > 0:
        # Memproses wajah pertama dari enrollment
        result, _ = step2_preprocessing(er["gray"], er["faces"][0], FACE_SIZE)
        db_preprocessed[person_name] = result

# ============================================================
# 4. Step 3: Feature Extraction (LBP Histogram)
# ============================================================

print("\n[STEP 3] Feature Extraction - LBP Histogram...")
print("-" * 50)


# Mendefinisikan fungsi untuk menghitung LBP (Local Binary Pattern)
def hitung_lbp(image):
    """
    Menghitung LBP image dari gambar grayscale.
    Setiap piksel dibandingkan dengan 8 tetangganya.
    """
    # Mendapatkan ukuran gambar
    rows, cols = image.shape

    # Membuat array LBP kosong
    lbp = np.zeros((rows - 2, cols - 2), dtype=np.uint8)

    # Menghitung LBP untuk setiap piksel (kecuali border)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Mengambil nilai piksel pusat
            center = image[i, j]

            # Menghitung kode LBP 8-bit
            code = 0

            # Membandingkan dengan 8 tetangga (searah jarum jam)
            code |= (1 << 7) if image[i-1, j-1] >= center else 0
            code |= (1 << 6) if image[i-1, j] >= center else 0
            code |= (1 << 5) if image[i-1, j+1] >= center else 0
            code |= (1 << 4) if image[i, j+1] >= center else 0
            code |= (1 << 3) if image[i+1, j+1] >= center else 0
            code |= (1 << 2) if image[i+1, j] >= center else 0
            code |= (1 << 1) if image[i+1, j-1] >= center else 0
            code |= (1 << 0) if image[i, j-1] >= center else 0

            # Menyimpan kode LBP
            lbp[i-1, j-1] = code

    # Mengembalikan gambar LBP
    return lbp


# Mendefinisikan fungsi untuk mengekstrak fitur LBP histogram
def step3_feature_extraction(face_image, n_bins=256):
    """
    Step 3 Pipeline: Mengekstrak fitur LBP histogram dari gambar wajah.
    """
    # Mencatat waktu mulai
    t_start = time.time()

    # Menghitung LBP image
    lbp_image = hitung_lbp(face_image)

    # Menghitung histogram LBP
    hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, 256))

    # Menormalisasi histogram
    hist = hist.astype(float)
    hist_sum = np.sum(hist)
    if hist_sum > 0:
        hist = hist / hist_sum

    # Menghitung waktu proses
    elapsed = time.time() - t_start

    # Mengembalikan fitur dan LBP image
    return hist, lbp_image, elapsed


# Mengekstrak fitur dari wajah test
test_features = []
feature_time_total = 0

# Mengekstrak fitur dari setiap wajah yang telah dipreproses
for prep_face in preprocessed_faces:
    # Mengekstrak fitur LBP
    feat, lbp_img, feat_time = step3_feature_extraction(prep_face["final"], LBP_BINS)
    test_features.append({"histogram": feat, "lbp_image": lbp_img})
    feature_time_total += feat_time

# Menyimpan waktu Step 3
pipeline_times["Step 3: Feature Extraction"] = feature_time_total

print(f"  Metode: Local Binary Pattern (LBP)")
print(f"  Dimensi fitur: {LBP_BINS} bins")
print(f"  Waktu ekstraksi: {feature_time_total*1000:.2f} ms")

# Mengekstrak fitur dari database enrollment
db_features = {}
for person_name, prep in db_preprocessed.items():
    # Mengekstrak fitur LBP dari wajah enrollment
    feat, _, _ = step3_feature_extraction(prep["final"], LBP_BINS)
    db_features[person_name] = feat

print(f"  Database enrollment: {len(db_features)} orang")

# ============================================================
# 5. Step 4: Classification (Nearest Neighbor)
# ============================================================

print("\n[STEP 4] Classification - Nearest Neighbor...")
print("-" * 50)


# Mendefinisikan fungsi untuk klasifikasi nearest neighbor
def step4_classification(test_feature, db_features, threshold):
    """
    Step 4 Pipeline: Klasifikasi menggunakan nearest neighbor.
    Mengembalikan identitas terdekat dan confidence score.
    """
    # Mencatat waktu mulai
    t_start = time.time()

    # Menginisialisasi variabel untuk tracking jarak minimum
    min_distance = float('inf')
    best_match = "Unknown"
    all_distances = {}

    # Menghitung jarak ke setiap orang dalam database
    for person_name, db_feat in db_features.items():
        # Menghitung jarak Euclidean antara fitur test dan database
        distance = np.linalg.norm(test_feature - db_feat)

        # Menyimpan jarak
        all_distances[person_name] = distance

        # Memperbarui jarak minimum
        if distance < min_distance:
            min_distance = distance
            best_match = person_name

    # Menentukan apakah hasil di atas threshold (reject)
    if min_distance > threshold:
        best_match = "Unknown"

    # Menghitung confidence score (inverse of distance, normalized)
    if min_distance > 0:
        confidence = max(0, 1 - min_distance / threshold)
    else:
        confidence = 1.0

    # Menghitung waktu proses
    elapsed = time.time() - t_start

    # Mengembalikan hasil klasifikasi
    return {
        "identity": best_match,
        "distance": min_distance,
        "confidence": confidence,
        "all_distances": all_distances
    }, elapsed


# Mengklasifikasikan setiap wajah test
classification_results = []
classify_time_total = 0

# Mengklasifikasikan setiap wajah
for feat_data in test_features:
    # Melakukan klasifikasi
    result, cls_time = step4_classification(
        feat_data["histogram"], db_features, DISTANCE_THRESHOLD
    )
    classification_results.append(result)
    classify_time_total += cls_time

    # Menampilkan hasil
    print(f"  Hasil: {result['identity']} "
          f"(distance={result['distance']:.4f}, "
          f"confidence={result['confidence']:.4f})")

    # Menampilkan jarak ke semua orang
    for name, dist in result['all_distances'].items():
        print(f"    → {name}: {dist:.4f}")

# Menyimpan waktu Step 4
pipeline_times["Step 4: Classification"] = classify_time_total

print(f"\n  Threshold: {DISTANCE_THRESHOLD}")
print(f"  Waktu klasifikasi: {classify_time_total*1000:.2f} ms")

# ============================================================
# 6. Step 5: Visualization
# ============================================================

print("\n[STEP 5] Visualization - Menggambar hasil pada gambar...")
print("-" * 50)

# Mencatat waktu mulai visualisasi
t_viz_start = time.time()

# Membuat salinan gambar test untuk visualisasi
result_image = test_image.copy()

# Menggambar bounding box dan label pada setiap wajah
if len(test_faces) > 0:
    for i, (x, y, w, h) in enumerate(test_faces):
        # Mengambil hasil klasifikasi
        if i < len(classification_results):
            cls_result = classification_results[i]
            identity = cls_result["identity"]
            confidence = cls_result["confidence"]
        else:
            identity = "Unknown"
            confidence = 0.0

        # Menentukan warna berdasarkan identitas
        if identity != "Unknown":
            color = (0, 255, 0)  # Hijau untuk recognized
        else:
            color = (0, 0, 255)  # Merah untuk unknown

        # Menggambar bounding box
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

        # Membuat label teks
        label = f"{identity} ({confidence:.2f})"

        # Menggambar background untuk teks
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_image, (x, y - 25),
                      (x + text_size[0], y), color, -1)

        # Menggambar teks label
        cv2.putText(result_image, label, (x, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Menghitung waktu visualisasi
viz_time = time.time() - t_viz_start
pipeline_times["Step 5: Visualization"] = viz_time

print(f"  Waktu visualisasi: {viz_time*1000:.2f} ms")

# Menghitung total waktu pipeline
total_pipeline_time = sum(pipeline_times.values())
pipeline_times["Total Pipeline"] = total_pipeline_time
print(f"\n  Total waktu pipeline: {total_pipeline_time*1000:.2f} ms")

# ============================================================
# 7. Visualisasi 1: Step 1 - Detection dan Step 2 - Preprocessing
# ============================================================

print("\n[INFO] Membuat visualisasi Step 1 dan Step 2...")
print("-" * 50)

# Membuat figure untuk step 1 dan 2
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# --- Baris atas: Step 1 - Face Detection ---
# Panel 1: Gambar original
axes[0, 0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Input Image", fontsize=10, fontweight='bold')
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])

# Panel 2: Gambar grayscale
axes[0, 1].imshow(test_gray, cmap='gray')
axes[0, 1].set_title("Grayscale", fontsize=10, fontweight='bold')
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])

# Panel 3: Deteksi wajah
det_viz = cv2.cvtColor(test_image.copy(), cv2.COLOR_BGR2RGB)
if len(test_faces) > 0:
    for (x, y, w, h) in test_faces:
        cv2.rectangle(det_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
axes[0, 2].imshow(det_viz)
axes[0, 2].set_title(f"Detection ({len(test_faces)} faces)", fontsize=10,
                     fontweight='bold')
axes[0, 2].set_xticks([])
axes[0, 2].set_yticks([])

# Panel 4: Timeline/info deteksi
axes[0, 3].axis('off')
det_info = (f"Step 1: Face Detection\n\n"
            f"Metode: Haar Cascade\n"
            f"scaleFactor: 1.1\n"
            f"minNeighbors: 5\n"
            f"minSize: (30, 30)\n\n"
            f"Wajah terdeteksi: {len(test_faces)}\n"
            f"Waktu: {pipeline_times['Step 1: Detection']*1000:.2f} ms")
axes[0, 3].text(0.1, 0.9, det_info, transform=axes[0, 3].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# --- Baris bawah: Step 2 - Preprocessing ---
if len(preprocessed_faces) > 0:
    prep = preprocessed_faces[0]

    # Panel 1: ROI (crop)
    axes[1, 0].imshow(prep["roi"], cmap='gray')
    axes[1, 0].set_title("Face ROI (Crop)", fontsize=10, fontweight='bold')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # Panel 2: Resized
    axes[1, 1].imshow(prep["resized"], cmap='gray')
    axes[1, 1].set_title(f"Resized ({FACE_SIZE})", fontsize=10, fontweight='bold')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    # Panel 3: Equalized
    axes[1, 2].imshow(prep["equalized"], cmap='gray')
    axes[1, 2].set_title("Hist. Equalized", fontsize=10, fontweight='bold')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    # Panel 4: Final (smoothed)
    axes[1, 3].imshow(prep["final"], cmap='gray')
    axes[1, 3].set_title("Final (Smoothed)", fontsize=10, fontweight='bold')
    axes[1, 3].set_xticks([])
    axes[1, 3].set_yticks([])
else:
    # Jika tidak ada wajah, tampilkan placeholder
    for j in range(4):
        axes[1, j].axis('off')
        axes[1, j].text(0.5, 0.5, "No Face", ha='center', va='center', fontsize=12)

# Menambahkan judul utama
plt.suptitle("Percobaan 19: Pipeline Step 1 (Detection) & Step 2 (Preprocessing)",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi step 1 dan 2
output_path_1 = os.path.join(OUTPUT_DIR, "19_pipeline_step1.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 8. Visualisasi 2: Step 3 - Feature Extraction & Step 4 - Classification
# ============================================================

print("\n[INFO] Membuat visualisasi Step 3 dan Step 4...")
print("-" * 50)

# Membuat figure untuk step 3 dan 4
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# --- Baris atas: Step 3 - Feature Extraction ---
if len(test_features) > 0:
    feat_data = test_features[0]

    # Panel 1: LBP Image
    axes[0, 0].imshow(feat_data["lbp_image"], cmap='gray')
    axes[0, 0].set_title("LBP Image", fontsize=10, fontweight='bold')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # Panel 2: LBP Histogram
    axes[0, 1].bar(range(LBP_BINS), feat_data["histogram"], color='#4472C4',
                   alpha=0.7, width=1.0)
    axes[0, 1].set_xlabel("LBP Bin", fontsize=9)
    axes[0, 1].set_ylabel("Frekuensi (Norm.)", fontsize=9)
    axes[0, 1].set_title(f"LBP Histogram ({LBP_BINS} bins)", fontsize=10,
                         fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Panel 3: Perbandingan histogram test vs database
    if len(db_features) > 0:
        colors_db = ['#ED7D31', '#70AD47', '#FFC000']
        for idx, (name, db_feat) in enumerate(db_features.items()):
            axes[0, 2].plot(db_feat, color=colors_db[idx % len(colors_db)],
                            alpha=0.7, linewidth=1, label=f'DB: {name}')

        # Menambahkan histogram test
        axes[0, 2].plot(feat_data["histogram"], 'b-', linewidth=1.5,
                        alpha=0.9, label='Test')

        axes[0, 2].set_xlabel("LBP Bin", fontsize=9)
        axes[0, 2].set_ylabel("Frekuensi", fontsize=9)
        axes[0, 2].set_title("Perbandingan Histogram", fontsize=10,
                             fontweight='bold')
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].grid(axis='y', alpha=0.3)
    else:
        axes[0, 2].axis('off')
        axes[0, 2].text(0.5, 0.5, "No DB Features", ha='center', fontsize=10)
else:
    for j in range(3):
        axes[0, j].axis('off')

# --- Baris bawah: Step 4 - Classification ---
# Panel 1: Bar chart jarak ke setiap orang
if len(classification_results) > 0:
    cls_result = classification_results[0]
    names = list(cls_result["all_distances"].keys())
    distances = list(cls_result["all_distances"].values())

    # Menentukan warna bar
    bar_colors = []
    for name in names:
        if name == cls_result["identity"]:
            bar_colors.append('#70AD47')
        else:
            bar_colors.append('#4472C4')

    # Menggambar bar chart
    axes[1, 0].barh(names, distances, color=bar_colors, alpha=0.85,
                    edgecolor='black', linewidth=0.5)

    # Menambahkan garis threshold
    axes[1, 0].axvline(x=DISTANCE_THRESHOLD, color='red', linestyle='--',
                       linewidth=1.5, label=f'Threshold = {DISTANCE_THRESHOLD}')

    # Menambahkan label jarak
    for i, (name, dist) in enumerate(zip(names, distances)):
        axes[1, 0].text(dist + 1, i, f'{dist:.2f}', va='center', fontsize=9)

    axes[1, 0].set_xlabel("Jarak Euclidean", fontsize=10)
    axes[1, 0].set_title("Jarak ke Database", fontsize=10, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(axis='x', alpha=0.3)

    # Panel 2: Info klasifikasi
    axes[1, 1].axis('off')
    cls_info = (f"Step 4: Classification\n\n"
                f"Metode: Nearest Neighbor\n"
                f"Metrik: Euclidean Distance\n"
                f"Threshold: {DISTANCE_THRESHOLD}\n\n"
                f"Hasil: {cls_result['identity']}\n"
                f"Jarak: {cls_result['distance']:.4f}\n"
                f"Confidence: {cls_result['confidence']:.4f}\n\n"
                f"Waktu: {pipeline_times['Step 4: Classification']*1000:.2f} ms")
    axes[1, 1].text(0.1, 0.9, cls_info, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 3: Pipeline timing breakdown
    step_names = [k for k in pipeline_times.keys() if k != "Total Pipeline"]
    step_times_ms = [pipeline_times[k] * 1000 for k in step_names]

    # Menggambar pie chart timing
    wedges, texts, autotexts = axes[1, 2].pie(
        step_times_ms, labels=None, autopct='%1.1f%%',
        colors=['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5'],
        startangle=90
    )

    # Menambahkan legend
    axes[1, 2].legend(
        [f'{name.split(":")[1].strip()} ({t:.1f}ms)' for name, t in zip(step_names, step_times_ms)],
        loc='center left', bbox_to_anchor=(-0.2, 0.5), fontsize=7
    )
    axes[1, 2].set_title(f"Pipeline Timing\n(Total: {total_pipeline_time*1000:.1f} ms)",
                         fontsize=10, fontweight='bold')
else:
    for j in range(3):
        axes[1, j].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 19: Pipeline Step 3 (Features) & Step 4 (Classification)",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi step 3 dan 4
output_path_2 = os.path.join(OUTPUT_DIR, "19_pipeline_step2.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 9. Visualisasi 3: Pipeline Final - Flow Diagram dan Hasil
# ============================================================

print("\n[INFO] Membuat visualisasi pipeline final...")
print("-" * 50)

# Membuat figure untuk pipeline final
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# --- Panel kiri atas: Gambar input ---
axes[0, 0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Input Image", fontsize=11, fontweight='bold')
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])

# --- Panel tengah atas: Gambar hasil recognition ---
axes[0, 1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Recognition Result", fontsize=11, fontweight='bold')
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])

# --- Panel kanan atas: Pipeline flow diagram ---
axes[0, 2].axis('off')

# Menggambar flow diagram sebagai teks terstruktur
flow_text = (
    "PIPELINE FLOW\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "┌─────────────────────┐\n"
    "│  Input Image        │\n"
    "└──────────┬──────────┘\n"
    "           ▼\n"
    "┌─────────────────────┐\n"
    "│ Step 1: Detection   │\n"
    "│ (Haar Cascade)      │\n"
    "└──────────┬──────────┘\n"
    "           ▼\n"
    "┌─────────────────────┐\n"
    "│ Step 2: Preprocess  │\n"
    "│ (Resize+EqualHist)  │\n"
    "└──────────┬──────────┘\n"
    "           ▼\n"
    "┌─────────────────────┐\n"
    "│ Step 3: Features    │\n"
    "│ (LBP Histogram)     │\n"
    "└──────────┬──────────┘\n"
    "           ▼\n"
    "┌─────────────────────┐\n"
    "│ Step 4: Classify    │\n"
    "│ (Nearest Neighbor)  │\n"
    "└──────────┬──────────┘\n"
    "           ▼\n"
    "┌─────────────────────┐\n"
    "│ Step 5: Visualize   │\n"
    "│ (Box + Label)       │\n"
    "└─────────────────────┘"
)
axes[0, 2].text(0.05, 0.95, flow_text, transform=axes[0, 2].transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
axes[0, 2].set_title("Pipeline Flow Diagram", fontsize=11, fontweight='bold')

# --- Panel kiri bawah: Bar chart timing ---
step_names_short = ["Detection", "Preprocess", "Features", "Classify", "Visualize"]
step_times_ms = [pipeline_times[k] * 1000 for k in pipeline_times
                 if k != "Total Pipeline"]

# Membatasi jumlah step jika kurang
while len(step_times_ms) < len(step_names_short):
    step_times_ms.append(0.0)

# Menggambar horizontal bar chart
colors_bar = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5']
bars = axes[1, 0].barh(step_names_short, step_times_ms[:len(step_names_short)],
                        color=colors_bar, alpha=0.85,
                        edgecolor='black', linewidth=0.5)

# Menambahkan label waktu
for bar in bars:
    width = bar.get_width()
    axes[1, 0].text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f} ms', va='center', fontsize=9)

# Mengatur sumbu
axes[1, 0].set_xlabel("Waktu (ms)", fontsize=10)
axes[1, 0].set_title("Pipeline Timing Breakdown", fontsize=11, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# --- Panel tengah bawah: Tabel detail pipeline ---
axes[1, 1].axis('off')

# Membuat data tabel
table_data = []
for step_name, step_time in pipeline_times.items():
    table_data.append([step_name, f"{step_time*1000:.2f} ms"])

# Membuat tabel
table = axes[1, 1].table(cellText=table_data,
                          colLabels=["Pipeline Step", "Waktu"],
                          cellLoc='center', loc='center',
                          colWidths=[0.65, 0.25])

# Mengatur font dan skala tabel
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.5)

# Mengatur warna header
for j in range(2):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Mengatur warna baris data
for i in range(1, len(table_data) + 1):
    for j in range(2):
        if i == len(table_data):
            table[i, j].set_facecolor('#FFF2CC')
        elif i % 2 == 0:
            table[i, j].set_facecolor('#D6E4F0')
        else:
            table[i, j].set_facecolor('#EDF2F9')

axes[1, 1].set_title("Pipeline Timing Table", fontsize=11, fontweight='bold')

# --- Panel kanan bawah: Database enrollment faces ---
if len(db_preprocessed) > 0:
    n_db = len(db_preprocessed)
    for idx, (name, prep) in enumerate(db_preprocessed.items()):
        # Membuat sub-plot area
        ax_sub = axes[1, 2].inset_axes([idx / n_db, 0, 1 / n_db, 1])
        ax_sub.imshow(prep["final"], cmap='gray')
        ax_sub.set_title(name, fontsize=9)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])

    axes[1, 2].set_title("Database Wajah Terdaftar", fontsize=11, fontweight='bold')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
else:
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, "No Database Faces", ha='center', fontsize=12)
    axes[1, 2].set_title("Database Wajah", fontsize=11, fontweight='bold')

# Menambahkan judul utama
plt.suptitle("Percobaan 19: Complete Recognition Pipeline - Hasil Final",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi pipeline final
output_path_3 = os.path.join(OUTPUT_DIR, "19_pipeline_final.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 19")
print("=" * 60)
print("Pipeline Recognition End-to-End:")
print("  Step 1: Face Detection (Haar Cascade)")
print(f"          → {len(test_faces)} wajah terdeteksi, "
      f"{pipeline_times['Step 1: Detection']*1000:.2f} ms")
print("  Step 2: Preprocessing (Crop → Resize → Equalize → Smooth)")
print(f"          → ukuran {FACE_SIZE}, "
      f"{pipeline_times['Step 2: Preprocessing']*1000:.2f} ms")
print("  Step 3: Feature Extraction (LBP Histogram)")
print(f"          → {LBP_BINS} dimensi fitur, "
      f"{pipeline_times['Step 3: Feature Extraction']*1000:.2f} ms")
print("  Step 4: Classification (Nearest Neighbor)")
print(f"          → threshold={DISTANCE_THRESHOLD}, "
      f"{pipeline_times['Step 4: Classification']*1000:.2f} ms")
print("  Step 5: Visualization (Bounding Box + Label)")
print(f"          → {pipeline_times['Step 5: Visualization']*1000:.2f} ms")
print(f"\n  Total Pipeline: {total_pipeline_time*1000:.2f} ms")
if len(classification_results) > 0:
    print(f"  Hasil: {classification_results[0]['identity']} "
          f"(confidence={classification_results[0]['confidence']:.4f})")
print(f"  Database: {len(db_features)} orang terdaftar")
print("=" * 60)
