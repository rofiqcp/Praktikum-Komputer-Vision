"""
==========================================================================
PERCOBAAN 1: FACE DETECTION DENGAN HAAR CASCADE
==========================================================================
Program ini mempelajari cara mendeteksi wajah pada gambar menggunakan
metode Haar Cascade Classifier. Haar Cascade adalah metode deteksi objek
klasik yang menggunakan fitur Haar-like dan algoritma AdaBoost.

Konsep yang dipelajari:
- Haar-like features: fitur berbasis perbedaan intensitas piksel
- Integral image: percepatan komputasi fitur Haar
- Cascade classifier: rangkaian classifier bertingkat (cascade)
- Parameter scaleFactor dan minNeighbors untuk tuning deteksi

Fungsi utama yang dipelajari:
- cv2.CascadeClassifier()         : Membuat objek cascade classifier
- cv2.CascadeClassifier.detectMultiScale() : Mendeteksi objek multi-skala
- cv2.rectangle()                 : Menggambar kotak pembatas (bounding box)
- cv2.putText()                   : Menulis teks pada gambar

Hasil: Visualisasi deteksi wajah dengan variasi parameter
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan deteksi wajah
import cv2

# Mengimpor NumPy untuk operasi array numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu pemrosesan
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
print("PERCOBAAN 1: FACE DETECTION DENGAN HAAR CASCADE")
print("=" * 60)

# ============================================================
# 1. Memuat Haar Cascade Classifier
# ============================================================

# Memuat cascade classifier untuk deteksi wajah frontal
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Memuat cascade classifier untuk deteksi mata
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# Memuat cascade classifier untuk deteksi senyuman
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)

# Memeriksa apakah cascade berhasil dimuat
print(f"[INFO] Face cascade dimuat: {not face_cascade.empty()}")
print(f"[INFO] Eye cascade dimuat: {not eye_cascade.empty()}")
print(f"[INFO] Smile cascade dimuat: {not smile_cascade.empty()}")

# ============================================================
# 2. Deteksi Wajah pada Gambar Tunggal
# ============================================================

# Membaca gambar wajah tunggal dari file
img_single = cv2.imread(os.path.join(IMAGE_DIR, "wajah_single.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_single is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Mengkonversi gambar ke grayscale karena Haar cascade membutuhkan input grayscale
gray_single = cv2.cvtColor(img_single, cv2.COLOR_BGR2GRAY)

# Mencatat waktu mulai deteksi untuk mengukur performa
start_time = time.time()

# Mendeteksi wajah menggunakan detectMultiScale
# Parameter: scaleFactor=1.1, minNeighbors=5, minSize=(30,30)
faces_single = face_cascade.detectMultiScale(
    gray_single,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Menghitung waktu pemrosesan deteksi wajah
elapsed_single = time.time() - start_time

# Menampilkan jumlah wajah yang terdeteksi dan waktu proses
print(f"\n[INFO] Deteksi wajah tunggal:")
print(f"  Jumlah wajah terdeteksi: {len(faces_single)}")
print(f"  Waktu proses: {elapsed_single*1000:.2f} ms")

# Membuat salinan gambar untuk digambar bounding box
img_result_single = img_single.copy()

# Menggambar kotak pembatas pada setiap wajah yang terdeteksi
for (x, y, w, h) in faces_single:
    # Menggambar kotak biru untuk wajah
    cv2.rectangle(img_result_single, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mengambil region of interest (ROI) wajah untuk deteksi mata
    roi_gray = gray_single[y:y+h, x:x+w]

    # Mengambil ROI wajah dari gambar berwarna
    roi_color = img_result_single[y:y+h, x:x+w]

    # Mendeteksi mata di dalam area wajah
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)

    # Menggambar kotak hijau untuk setiap mata yang terdeteksi
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Mendeteksi senyuman di dalam area wajah
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=15)

    # Menggambar kotak merah untuk setiap senyuman yang terdeteksi
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

# Menambahkan teks informasi pada gambar hasil
cv2.putText(img_result_single, f"Wajah: {len(faces_single)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# ============================================================
# 3. Deteksi Wajah pada Gambar dengan Aksesoris
# ============================================================

# Membaca gambar wajah dengan kacamata
img_kacamata = cv2.imread(os.path.join(IMAGE_DIR, "wajah_kacamata.jpg"))

# Membaca gambar wajah dengan topi
img_topi = cv2.imread(os.path.join(IMAGE_DIR, "wajah_topi.jpg"))

# Mengkonversi gambar kacamata ke grayscale
gray_kacamata = cv2.cvtColor(img_kacamata, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar topi ke grayscale
gray_topi = cv2.cvtColor(img_topi, cv2.COLOR_BGR2GRAY)

# Mendeteksi wajah pada gambar dengan kacamata
faces_kacamata = face_cascade.detectMultiScale(gray_kacamata, 1.1, 5, minSize=(30, 30))

# Mendeteksi wajah pada gambar dengan topi
faces_topi = face_cascade.detectMultiScale(gray_topi, 1.1, 5, minSize=(30, 30))

# Membuat salinan gambar untuk visualisasi
img_res_kacamata = img_kacamata.copy()
img_res_topi = img_topi.copy()

# Menggambar bounding box pada gambar kacamata
for (x, y, w, h) in faces_kacamata:
    cv2.rectangle(img_res_kacamata, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Menggambar bounding box pada gambar topi
for (x, y, w, h) in faces_topi:
    cv2.rectangle(img_res_topi, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Menampilkan hasil deteksi untuk gambar dengan aksesoris
print(f"\n[INFO] Deteksi wajah dengan kacamata: {len(faces_kacamata)} wajah")
print(f"[INFO] Deteksi wajah dengan topi: {len(faces_topi)} wajah")

# ============================================================
# 4. Menyimpan Hasil Deteksi Tunggal
# ============================================================

# Membuat figure untuk menampilkan hasil deteksi tunggal
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Mengkonversi BGR ke RGB untuk matplotlib dan menampilkan gambar tunggal
axes[0].imshow(cv2.cvtColor(img_result_single, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Wajah Tunggal\n({len(faces_single)} wajah, {elapsed_single*1000:.1f}ms)", fontsize=11)
axes[0].axis("off")

# Menampilkan gambar wajah dengan kacamata
axes[1].imshow(cv2.cvtColor(img_res_kacamata, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Dengan Kacamata\n({len(faces_kacamata)} wajah)", fontsize=11)
axes[1].axis("off")

# Menampilkan gambar wajah dengan topi
axes[2].imshow(cv2.cvtColor(img_res_topi, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Dengan Topi\n({len(faces_topi)} wajah)", fontsize=11)
axes[2].axis("off")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 1: Haar Cascade - Deteksi Wajah Tunggal & Aksesoris",
             fontsize=13, fontweight="bold")

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan hasil visualisasi ke file
output_path_1 = os.path.join(OUTPUT_DIR, "01_haar_deteksi_tunggal.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure untuk membebaskan memori
plt.close()

# ============================================================
# 5. Variasi Parameter scaleFactor
# ============================================================

# Mendefinisikan nilai scaleFactor yang akan diuji
scale_factors = [1.05, 1.1, 1.2, 1.3]

# Membuat figure untuk menampilkan perbandingan scaleFactor
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Menyiapkan list untuk menyimpan jumlah deteksi dan waktu
deteksi_scale = []
waktu_scale = []

# Menguji setiap nilai scaleFactor
for i, sf in enumerate(scale_factors):
    # Membuat salinan gambar untuk setiap percobaan
    img_sf = img_single.copy()

    # Mencatat waktu mulai
    t_start = time.time()

    # Mendeteksi wajah dengan scaleFactor yang berbeda-beda
    faces_sf = face_cascade.detectMultiScale(gray_single, scaleFactor=sf, minNeighbors=5, minSize=(30, 30))

    # Menghitung waktu proses
    t_elapsed = time.time() - t_start

    # Menyimpan jumlah deteksi dan waktu
    deteksi_scale.append(len(faces_sf))
    waktu_scale.append(t_elapsed * 1000)

    # Menggambar bounding box pada setiap deteksi
    for (x, y, w, h) in faces_sf:
        cv2.rectangle(img_sf, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Menampilkan gambar hasil pada subplot baris atas
    axes[0, i].imshow(cv2.cvtColor(img_sf, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"scaleFactor={sf}\n{len(faces_sf)} wajah, {t_elapsed*1000:.1f}ms", fontsize=10)
    axes[0, i].axis("off")

    # Menampilkan informasi parameter
    print(f"  scaleFactor={sf}: {len(faces_sf)} wajah, waktu={t_elapsed*1000:.1f}ms")

# ============================================================
# 6. Variasi Parameter minNeighbors
# ============================================================

# Mendefinisikan nilai minNeighbors yang akan diuji
min_neighbors_list = [1, 3, 5, 7]

# Menyiapkan list untuk menyimpan hasil
deteksi_neighbors = []
waktu_neighbors = []

# Menguji setiap nilai minNeighbors
for i, mn in enumerate(min_neighbors_list):
    # Membuat salinan gambar untuk setiap percobaan
    img_mn = img_single.copy()

    # Mencatat waktu mulai
    t_start = time.time()

    # Mendeteksi wajah dengan minNeighbors yang berbeda-beda
    faces_mn = face_cascade.detectMultiScale(gray_single, scaleFactor=1.1, minNeighbors=mn, minSize=(30, 30))

    # Menghitung waktu proses
    t_elapsed = time.time() - t_start

    # Menyimpan jumlah deteksi dan waktu
    deteksi_neighbors.append(len(faces_mn))
    waktu_neighbors.append(t_elapsed * 1000)

    # Menggambar bounding box pada setiap deteksi
    for (x, y, w, h) in faces_mn:
        cv2.rectangle(img_mn, (x, y), (x + w, y + h), (0, 200, 0), 2)

    # Menampilkan gambar hasil pada subplot baris bawah
    axes[1, i].imshow(cv2.cvtColor(img_mn, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"minNeighbors={mn}\n{len(faces_mn)} wajah, {t_elapsed*1000:.1f}ms", fontsize=10)
    axes[1, i].axis("off")

    # Menampilkan informasi parameter
    print(f"  minNeighbors={mn}: {len(faces_mn)} wajah, waktu={t_elapsed*1000:.1f}ms")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 1: Variasi Parameter Haar Cascade\n"
             "Baris atas: scaleFactor | Baris bawah: minNeighbors",
             fontsize=13, fontweight="bold")

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan hasil visualisasi variasi parameter
output_path_2 = os.path.join(OUTPUT_DIR, "01_haar_variasi_parameter.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 7. Deteksi Wajah pada Gambar Grup
# ============================================================

# Membaca gambar grup yang berisi beberapa orang
img_grup = cv2.imread(os.path.join(IMAGE_DIR, "wajah_grup.jpg"))

# Mengkonversi gambar grup ke grayscale
gray_grup = cv2.cvtColor(img_grup, cv2.COLOR_BGR2GRAY)

# Mencatat waktu mulai deteksi
start_grup = time.time()

# Mendeteksi wajah pada gambar grup
faces_grup = face_cascade.detectMultiScale(gray_grup, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

# Menghitung waktu proses
elapsed_grup = time.time() - start_grup

# Menampilkan hasil deteksi pada gambar grup
print(f"\n[INFO] Deteksi wajah grup:")
print(f"  Jumlah wajah terdeteksi: {len(faces_grup)}")
print(f"  Waktu proses: {elapsed_grup*1000:.2f} ms")

# Membuat salinan gambar grup untuk digambar bounding box
img_result_grup = img_grup.copy()

# Mendefinisikan warna berbeda untuk setiap wajah yang terdeteksi
warna_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)]

# Menggambar bounding box berwarna pada setiap wajah
for idx, (x, y, w, h) in enumerate(faces_grup):
    # Memilih warna dari daftar warna (berputar jika lebih dari 8)
    warna = warna_list[idx % len(warna_list)]

    # Menggambar kotak pembatas pada wajah
    cv2.rectangle(img_result_grup, (x, y), (x + w, y + h), warna, 2)

    # Menambahkan label nomor wajah
    cv2.putText(img_result_grup, f"#{idx+1}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2)

# Membuat figure untuk menampilkan hasil deteksi grup
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Menampilkan gambar grup asli
axes[0].imshow(cv2.cvtColor(img_grup, cv2.COLOR_BGR2RGB))
axes[0].set_title("Gambar Grup Asli", fontsize=12)
axes[0].axis("off")

# Menampilkan gambar grup dengan deteksi wajah
axes[1].imshow(cv2.cvtColor(img_result_grup, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Deteksi: {len(faces_grup)} wajah ({elapsed_grup*1000:.1f}ms)", fontsize=12)
axes[1].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 1: Haar Cascade - Deteksi Wajah pada Gambar Grup",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi deteksi grup
output_path_3 = os.path.join(OUTPUT_DIR, "01_haar_deteksi_grup.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 1")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.CascadeClassifier()           → Memuat model Haar Cascade")
print("     - haarcascade_frontalface_default.xml : deteksi wajah")
print("     - haarcascade_eye.xml               : deteksi mata")
print("     - haarcascade_smile.xml             : deteksi senyum")
print("  2. detectMultiScale()                 → Deteksi objek multi-skala")
print("     - scaleFactor: faktor skala piramida gambar")
print("       Kecil (1.05) = lebih teliti tapi lambat")
print("       Besar (1.3)  = lebih cepat tapi bisa miss")
print("     - minNeighbors: jumlah deteksi overlap minimum")
print("       Kecil (1) = banyak deteksi (termasuk false positive)")
print("       Besar (7) = sedikit deteksi (lebih ketat)")
print("  3. Haar Cascade bekerja cepat namun kurang akurat")
print("     untuk kondisi pencahayaan/sudut yang bervariasi")
print(f"\nHasil deteksi:")
print(f"  - Wajah tunggal  : {len(faces_single)} terdeteksi")
print(f"  - Dengan kacamata: {len(faces_kacamata)} terdeteksi")
print(f"  - Dengan topi    : {len(faces_topi)} terdeteksi")
print(f"  - Gambar grup    : {len(faces_grup)} terdeteksi")
print("=" * 60)
