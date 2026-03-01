"""
==========================================================================
PERCOBAAN 5: FACE LANDMARK DETECTION CONCEPTS
==========================================================================
Program ini mempelajari konsep deteksi landmark wajah (facial landmarks).
Landmark wajah adalah titik-titik kunci pada wajah seperti mata, hidung,
mulut, dan kontur rahang. Karena dlib/mediapipe mungkin tidak tersedia,
program mengimplementasikan estimasi landmark berdasarkan geometri wajah.

Konsep yang dipelajari:
- Facial landmarks: 68 titik / 5 titik kunci pada wajah
- Estimasi posisi mata, hidung, dan mulut dari deteksi wajah
- Face alignment: meluruskan wajah berdasarkan posisi mata
- Normalisasi wajah: crop dan resize ke ukuran standar

Fungsi utama yang dipelajari:
- cv2.CascadeClassifier.detectMultiScale() : Deteksi wajah
- cv2.getRotationMatrix2D()        : Membuat matriks rotasi 2D
- cv2.warpAffine()                 : Transformasi affine (rotasi)
- cv2.circle() / cv2.line()        : Menggambar landmark
- np.arctan2()                     : Menghitung sudut rotasi

Hasil: Visualisasi landmark, face alignment, dan normalized faces
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan matematika
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor math untuk fungsi matematika (atan2, degrees)
import math

# Mengimpor glob untuk mencari file dengan pola tertentu
import glob

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 5: FACE LANDMARK DETECTION CONCEPTS")
print("=" * 60)

# ============================================================
# 1. Deteksi Wajah dan Estimasi Landmark Geometris
# ============================================================

print("\n[INFO] Mendeteksi wajah dan mengestimasi landmark...")
print("-" * 50)

# Memuat Haar cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Memuat Haar cascade untuk deteksi mata
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# Mendefinisikan fungsi untuk mengestimasi landmark wajah secara geometris
def estimasi_landmark(face_rect, img_gray):
    """
    Mengestimasi posisi landmark wajah berdasarkan proporsi geometris.
    Input: face_rect = (x, y, w, h) dari deteksi wajah
    Output: dictionary berisi posisi landmark (mata, hidung, mulut, dll.)
    """
    # Mengekstrak koordinat bounding box wajah
    x, y, w, h = face_rect

    # Mendefinisikan landmark berdasarkan proporsi wajah standar
    landmarks = {}

    # Posisi mata kiri (sekitar 30% dari kiri, 35% dari atas area wajah)
    landmarks['mata_kiri'] = (int(x + 0.30 * w), int(y + 0.35 * h))

    # Posisi mata kanan (sekitar 70% dari kiri, 35% dari atas)
    landmarks['mata_kanan'] = (int(x + 0.70 * w), int(y + 0.35 * h))

    # Posisi hidung (sekitar 50% horizontal, 55% vertikal)
    landmarks['hidung'] = (int(x + 0.50 * w), int(y + 0.55 * h))

    # Posisi mulut kiri (sekitar 35% horizontal, 75% vertikal)
    landmarks['mulut_kiri'] = (int(x + 0.35 * w), int(y + 0.75 * h))

    # Posisi mulut kanan (sekitar 65% horizontal, 75% vertikal)
    landmarks['mulut_kanan'] = (int(x + 0.65 * w), int(y + 0.75 * h))

    # Posisi mulut tengah (sekitar 50% horizontal, 75% vertikal)
    landmarks['mulut_tengah'] = (int(x + 0.50 * w), int(y + 0.75 * h))

    # Posisi alis kiri
    landmarks['alis_kiri'] = (int(x + 0.30 * w), int(y + 0.28 * h))

    # Posisi alis kanan
    landmarks['alis_kanan'] = (int(x + 0.70 * w), int(y + 0.28 * h))

    # Posisi dagu
    landmarks['dagu'] = (int(x + 0.50 * w), int(y + 0.95 * h))

    # Posisi kontur rahang kiri
    landmarks['rahang_kiri'] = (int(x + 0.05 * w), int(y + 0.65 * h))

    # Posisi kontur rahang kanan
    landmarks['rahang_kanan'] = (int(x + 0.95 * w), int(y + 0.65 * h))

    # Mendeteksi mata menggunakan Haar cascade untuk posisi lebih akurat
    roi_gray = img_gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20))

    # Jika mata terdeteksi, gunakan posisi yang lebih akurat
    if len(eyes) >= 2:
        # Mengurutkan mata berdasarkan posisi x (kiri ke kanan)
        eyes_sorted = sorted(eyes, key=lambda e: e[0])

        # Memperbarui posisi mata kiri
        ex, ey, ew, eh = eyes_sorted[0]
        landmarks['mata_kiri'] = (x + ex + ew // 2, y + ey + eh // 2)

        # Memperbarui posisi mata kanan
        ex, ey, ew, eh = eyes_sorted[1]
        landmarks['mata_kanan'] = (x + ex + ew // 2, y + ey + eh // 2)

    # Mengembalikan dictionary landmark
    return landmarks

# Mendefinisikan fungsi untuk menggambar landmark pada gambar
def gambar_landmark(img, landmarks, warna_titik=(0, 255, 0), warna_garis=(255, 200, 0)):
    """Menggambar landmark pada gambar."""
    # Membuat salinan gambar
    hasil = img.copy()

    # Menggambar titik untuk setiap landmark
    for nama, (px, py) in landmarks.items():
        # Menggambar lingkaran kecil di posisi landmark
        cv2.circle(hasil, (px, py), 3, warna_titik, -1)

        # Menuliskan nama landmark di samping titik
        cv2.putText(hasil, nama.split('_')[-1][:4], (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Menggambar garis penghubung antar landmark
    # Garis antara kedua mata
    cv2.line(hasil, landmarks['mata_kiri'], landmarks['mata_kanan'], warna_garis, 1)

    # Garis dari mata kiri ke hidung
    cv2.line(hasil, landmarks['mata_kiri'], landmarks['hidung'], warna_garis, 1)

    # Garis dari mata kanan ke hidung
    cv2.line(hasil, landmarks['mata_kanan'], landmarks['hidung'], warna_garis, 1)

    # Garis dari hidung ke mulut tengah
    cv2.line(hasil, landmarks['hidung'], landmarks['mulut_tengah'], warna_garis, 1)

    # Garis mulut (kiri ke kanan)
    cv2.line(hasil, landmarks['mulut_kiri'], landmarks['mulut_kanan'], warna_garis, 1)

    # Garis kontur rahang
    cv2.line(hasil, landmarks['rahang_kiri'], landmarks['dagu'], warna_garis, 1)
    cv2.line(hasil, landmarks['dagu'], landmarks['rahang_kanan'], warna_garis, 1)

    # Mengembalikan gambar dengan landmark
    return hasil

# ============================================================
# 2. Mendeteksi dan Menggambar Landmark pada Berbagai Gambar
# ============================================================

# Mendefinisikan daftar gambar yang akan diproses
gambar_files = [
    ("Wajah Tunggal", "wajah_single.jpg"),
    ("Wajah Kacamata", "wajah_kacamata.jpg"),
    ("Wajah Topi", "wajah_topi.jpg"),
]

# Membuat figure untuk visualisasi landmark
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Menyiapkan list untuk menyimpan hasil landmark
semua_landmarks = []

# Memproses setiap gambar
for col, (nama, filename) in enumerate(gambar_files):
    # Membaca gambar
    img = cv2.imread(os.path.join(IMAGE_DIR, filename))

    # Memeriksa apakah gambar berhasil dimuat
    if img is None:
        print(f"[WARNING] {filename} tidak ditemukan!")
        continue

    # Mengkonversi ke grayscale untuk deteksi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

    # Menampilkan jumlah wajah terdeteksi
    print(f"  {nama}: {len(faces)} wajah terdeteksi")

    # Menampilkan gambar asli pada baris atas
    axes[0, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, col].set_title(f"{nama}\nAsli", fontsize=11)
    axes[0, col].axis("off")

    # Membuat salinan untuk menggambar landmark
    img_landmark = img.copy()

    # Memproses setiap wajah yang terdeteksi
    for face_rect in faces:
        # Mengestimasi landmark untuk wajah ini
        landmarks = estimasi_landmark(face_rect, gray)

        # Menggambar landmark pada gambar
        img_landmark = gambar_landmark(img_landmark, landmarks)

        # Menggambar bounding box wajah
        x, y, w, h = face_rect
        cv2.rectangle(img_landmark, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Menyimpan landmark untuk proses selanjutnya
        semua_landmarks.append((nama, face_rect, landmarks, img.copy()))

    # Menampilkan gambar dengan landmark pada baris bawah
    axes[1, col].imshow(cv2.cvtColor(img_landmark, cv2.COLOR_BGR2RGB))
    axes[1, col].set_title(f"{nama}\nDengan Landmark", fontsize=11)
    axes[1, col].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 5: Estimasi Facial Landmarks\n"
             "Landmark dihitung berdasarkan proporsi geometris wajah",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi landmark
output_path_1 = os.path.join(OUTPUT_DIR, "05_landmark_deteksi.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 3. Face Alignment (Meluruskan Wajah)
# ============================================================

print("\n[INFO] Melakukan face alignment...")
print("-" * 50)

# Mendefinisikan fungsi untuk menghitung sudut rotasi dari posisi mata
def hitung_sudut_rotasi(mata_kiri, mata_kanan):
    """
    Menghitung sudut rotasi berdasarkan posisi kedua mata.
    Tujuan: membuat garis antara kedua mata horizontal (0 derajat).
    """
    # Menghitung selisih koordinat
    dx = mata_kanan[0] - mata_kiri[0]
    dy = mata_kanan[1] - mata_kiri[1]

    # Menghitung sudut menggunakan arctan2
    sudut = math.degrees(math.atan2(dy, dx))

    # Mengembalikan sudut rotasi (negatif untuk meluruskan)
    return sudut

# Mendefinisikan fungsi untuk melakukan face alignment
def align_face(img, landmarks, output_size=(200, 200)):
    """
    Meluruskan wajah berdasarkan posisi mata.
    Langkah: rotasi gambar agar garis mata horizontal, lalu crop.
    """
    # Mengambil posisi mata kiri dan kanan
    mata_kiri = landmarks['mata_kiri']
    mata_kanan = landmarks['mata_kanan']

    # Menghitung sudut rotasi
    sudut = hitung_sudut_rotasi(mata_kiri, mata_kanan)

    # Menghitung titik pusat antara kedua mata
    pusat_x = (mata_kiri[0] + mata_kanan[0]) // 2
    pusat_y = (mata_kiri[1] + mata_kanan[1]) // 2

    # Membuat matriks rotasi menggunakan cv2.getRotationMatrix2D
    # Parameter: center, angle, scale
    M = cv2.getRotationMatrix2D((pusat_x, pusat_y), sudut, 1.0)

    # Menerapkan rotasi pada gambar menggunakan cv2.warpAffine
    h, w = img.shape[:2]
    img_rotated = cv2.warpAffine(img, M, (w, h))

    # Menghitung jarak antara kedua mata
    jarak_mata = np.sqrt((mata_kanan[0] - mata_kiri[0]) ** 2 +
                         (mata_kanan[1] - mata_kiri[1]) ** 2)

    # Menghitung area crop berdasarkan jarak mata
    # Wajah biasanya sekitar 2.5x jarak mata lebarnya
    margin = jarak_mata * 1.2

    # Menghitung koordinat crop
    crop_x1 = max(0, int(pusat_x - margin))
    crop_y1 = max(0, int(pusat_y - margin * 0.8))
    crop_x2 = min(w, int(pusat_x + margin))
    crop_y2 = min(h, int(pusat_y + margin * 1.2))

    # Melakukan crop pada gambar yang sudah dirotasi
    face_cropped = img_rotated[crop_y1:crop_y2, crop_x1:crop_x2]

    # Memeriksa apakah crop valid
    if face_cropped.size == 0:
        return cv2.resize(img_rotated, output_size)

    # Meresize ke ukuran output standar
    face_aligned = cv2.resize(face_cropped, output_size)

    # Mengembalikan gambar wajah yang sudah di-align
    return face_aligned, img_rotated, sudut

# Membuat figure untuk visualisasi face alignment
n_faces = min(len(semua_landmarks), 3)
fig, axes = plt.subplots(n_faces, 4, figsize=(16, 4 * n_faces))

# Memastikan axes selalu 2D
if n_faces == 1:
    axes = axes.reshape(1, -1)

# Memproses setiap wajah untuk alignment
for row in range(n_faces):
    # Mengambil data wajah
    nama, face_rect, landmarks, img_original = semua_landmarks[row]

    # Menghitung sudut rotasi
    sudut = hitung_sudut_rotasi(landmarks['mata_kiri'], landmarks['mata_kanan'])

    # Menampilkan gambar asli
    axes[row, 0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    axes[row, 0].set_title(f"{nama}\nAsli", fontsize=10)
    axes[row, 0].axis("off")

    # Menampilkan gambar dengan landmark dan garis mata
    img_with_line = img_original.copy()
    cv2.line(img_with_line, landmarks['mata_kiri'], landmarks['mata_kanan'], (0, 0, 255), 2)
    cv2.circle(img_with_line, landmarks['mata_kiri'], 5, (0, 255, 0), -1)
    cv2.circle(img_with_line, landmarks['mata_kanan'], 5, (0, 255, 0), -1)
    cv2.putText(img_with_line, f"Sudut: {sudut:.1f} deg",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    axes[row, 1].imshow(cv2.cvtColor(img_with_line, cv2.COLOR_BGR2RGB))
    axes[row, 1].set_title(f"Garis Mata\nSudut: {sudut:.1f}°", fontsize=10)
    axes[row, 1].axis("off")

    # Melakukan face alignment
    result = align_face(img_original, landmarks)
    if isinstance(result, tuple):
        face_aligned, img_rotated, sudut_rotasi = result
    else:
        face_aligned = result
        img_rotated = img_original

    # Menampilkan gambar setelah rotasi
    axes[row, 2].imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    axes[row, 2].set_title(f"Setelah Rotasi\n(-{sudut:.1f}°)", fontsize=10)
    axes[row, 2].axis("off")

    # Menampilkan wajah yang sudah di-align dan crop
    axes[row, 3].imshow(cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB))
    axes[row, 3].set_title("Aligned & Cropped\n200x200", fontsize=10)
    axes[row, 3].axis("off")

    # Menampilkan informasi alignment
    print(f"  {nama}: sudut={sudut:.2f}°, aligned ke 200x200")

# Menambahkan judul utama
plt.suptitle("Percobaan 5: Face Alignment\n"
             "Meluruskan wajah berdasarkan posisi mata (rotasi + crop)",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi alignment
output_path_2 = os.path.join(OUTPUT_DIR, "05_face_alignment.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 4. Normalisasi Wajah dari Dataset
# ============================================================

print("\n[INFO] Menormalisasi wajah dari dataset...")
print("-" * 50)

# Mendefinisikan path folder faces
faces_dir = os.path.join(IMAGE_DIR, "faces")

# Mendefinisikan nama orang
nama_orang = ["andi", "budi", "citra"]

# Menyiapkan list untuk menyimpan wajah yang dinormalisasi
normalized_faces = {nama: [] for nama in nama_orang}

# Memproses setiap orang
for nama in nama_orang:
    # Mendapatkan path folder
    person_dir = os.path.join(faces_dir, nama)

    # Mencari file gambar
    foto_list = sorted(glob.glob(os.path.join(person_dir, "*.jpg")))

    # Memproses setiap foto (ambil 3 pertama)
    for foto_path in foto_list[:3]:
        # Membaca gambar
        img = cv2.imread(foto_path)

        # Memeriksa apakah berhasil dimuat
        if img is None:
            continue

        # Mengkonversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mendeteksi wajah
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

        # Jika wajah terdeteksi, lakukan alignment
        if len(faces) > 0:
            # Mengambil wajah pertama
            face_rect = faces[0]

            # Mengestimasi landmark
            landmarks = estimasi_landmark(face_rect, gray)

            # Melakukan alignment
            result = align_face(img, landmarks, output_size=(100, 100))
            if isinstance(result, tuple):
                face_normalized = result[0]
            else:
                face_normalized = result

            # Menyimpan wajah yang dinormalisasi
            normalized_faces[nama].append(face_normalized)
        else:
            # Jika tidak terdeteksi, crop dan resize langsung
            face_cropped = cv2.resize(img, (100, 100))
            normalized_faces[nama].append(face_cropped)

    # Menampilkan jumlah wajah yang dinormalisasi
    print(f"  {nama}: {len(normalized_faces[nama])} wajah dinormalisasi")

# Membuat figure untuk visualisasi wajah yang dinormalisasi
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# Menampilkan wajah yang dinormalisasi untuk setiap orang
for row, nama in enumerate(nama_orang):
    for col in range(3):
        if col < len(normalized_faces[nama]):
            # Menampilkan wajah yang dinormalisasi
            face_img = normalized_faces[nama][col]
            axes[row, col].imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f"{nama.capitalize()} #{col+1}", fontsize=10)
        else:
            axes[row, col].set_title("N/A", fontsize=10)

        # Menyembunyikan sumbu
        axes[row, col].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 5: Wajah yang Sudah Dinormalisasi\n"
             "Aligned, cropped, dan resized ke 100x100",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil normalisasi
output_path_3 = os.path.join(OUTPUT_DIR, "05_face_normalized.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 5")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. Facial Landmarks")
print("     - Titik kunci: mata, hidung, mulut, alis, rahang, dagu")
print("     - 68 titik (dlib) atau 5 titik (sederhana)")
print("     - Estimasi geometris berdasarkan proporsi wajah")
print("  2. cv2.getRotationMatrix2D(center, angle, scale)")
print("     - Membuat matriks transformasi rotasi 2D")
print("     - center: titik pusat rotasi")
print("     - angle: sudut rotasi (derajat)")
print("  3. cv2.warpAffine(img, M, dsize)")
print("     - Menerapkan transformasi affine pada gambar")
print("     - Digunakan untuk merotasi gambar")
print("  4. Face Alignment Pipeline:")
print("     a. Deteksi wajah (Haar Cascade)")
print("     b. Estimasi/deteksi posisi mata")
print("     c. Hitung sudut antara kedua mata")
print("     d. Rotasi gambar agar mata horizontal")
print("     e. Crop area wajah")
print("     f. Resize ke ukuran standar")
print("  5. Normalisasi penting untuk face recognition yang konsisten")
print("=" * 60)
