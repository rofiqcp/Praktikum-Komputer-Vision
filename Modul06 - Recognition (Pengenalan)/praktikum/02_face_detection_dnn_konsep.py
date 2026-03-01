"""
==========================================================================
PERCOBAAN 2: FACE DETECTION DENGAN DNN (DEEP NEURAL NETWORK) KONSEP
==========================================================================
Program ini mempelajari konsep deteksi wajah berbasis Deep Neural Network
(DNN) dan membandingkannya dengan metode Haar Cascade. Karena model DNN
(Caffe/TensorFlow) mungkin tidak tersedia, program ini mendemonstrasikan
proses pembuatan blob dan deteksi alternatif menggunakan warna kulit.

Konsep yang dipelajari:
- DNN-based face detection vs Haar Cascade
- Blob creation: preprocessing gambar untuk input neural network
- Skin color detection sebagai alternatif deteksi wajah sederhana
- Perbandingan robustness deteksi pada kondisi berbeda

Fungsi utama yang dipelajari:
- cv2.dnn.blobFromImage()         : Membuat blob dari gambar untuk DNN
- cv2.inRange()                   : Segmentasi warna (skin detection)
- cv2.findContours()              : Mencari kontur pada mask
- cv2.CascadeClassifier.detectMultiScale() : Deteksi Haar sebagai baseline

Hasil: Visualisasi blob, skin detection, dan perbandingan metode
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan DNN
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
print("PERCOBAAN 2: FACE DETECTION DENGAN DNN KONSEP")
print("=" * 60)

# ============================================================
# 1. Konsep Blob Creation untuk DNN
# ============================================================

# Membaca gambar wajah tunggal untuk demonstrasi blob
img_single = cv2.imread(os.path.join(IMAGE_DIR, "wajah_single.jpg"))

# Membaca gambar wajah dengan kacamata
img_kacamata = cv2.imread(os.path.join(IMAGE_DIR, "wajah_kacamata.jpg"))

# Membaca gambar wajah dengan topi
img_topi = cv2.imread(os.path.join(IMAGE_DIR, "wajah_topi.jpg"))

# Membaca gambar grup
img_grup = cv2.imread(os.path.join(IMAGE_DIR, "wajah_grup.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_single is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

print("\n[INFO] Demonstrasi Blob Creation untuk DNN Face Detection")
print("-" * 50)

# Mendefinisikan daftar gambar dan nama untuk diproses
gambar_list = [
    ("Wajah Tunggal", img_single),
    ("Wajah Kacamata", img_kacamata),
    ("Wajah Topi", img_topi),
]

# Membuat figure untuk visualisasi blob pada setiap gambar
fig, axes = plt.subplots(3, 4, figsize=(18, 12))

# Memproses setiap gambar untuk mendemonstrasikan blob creation
for row, (nama, img) in enumerate(gambar_list):
    # Menampilkan gambar asli pada kolom pertama
    axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[row, 0].set_title(f"{nama}\nAsli ({img.shape[1]}x{img.shape[0]})", fontsize=9)
    axes[row, 0].axis("off")

    # Membuat blob dari gambar menggunakan cv2.dnn.blobFromImage
    # Parameter: image, scalefactor, size, mean, swapRB, crop
    # scalefactor=1.0: tidak ada penskalaan intensitas
    # size=(300,300): ukuran input standar untuk DNN face detector
    # mean=(104,177,123): nilai rata-rata yang dikurangi (ImageNet mean)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300),
                                  mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Menampilkan informasi blob
    print(f"  {nama}: blob shape = {blob.shape}")
    print(f"    - Dimensi: (batch, channels, height, width)")
    print(f"    - Range nilai: [{blob.min():.1f}, {blob.max():.1f}]")

    # Mengekstrak channel blob untuk visualisasi (batch=0)
    # Blob memiliki shape (1, 3, 300, 300) -> ambil (3, 300, 300)
    blob_channels = blob[0]

    # Menormalisasi channel 0 (Blue) untuk visualisasi
    ch0 = blob_channels[0]
    ch0_norm = ((ch0 - ch0.min()) / (ch0.max() - ch0.min() + 1e-8) * 255).astype(np.uint8)

    # Menampilkan channel Blue dari blob
    axes[row, 1].imshow(ch0_norm, cmap='Blues')
    axes[row, 1].set_title(f"Blob Ch-B\nshape={blob.shape}", fontsize=9)
    axes[row, 1].axis("off")

    # Menormalisasi channel 1 (Green) untuk visualisasi
    ch1 = blob_channels[1]
    ch1_norm = ((ch1 - ch1.min()) / (ch1.max() - ch1.min() + 1e-8) * 255).astype(np.uint8)

    # Menampilkan channel Green dari blob
    axes[row, 2].imshow(ch1_norm, cmap='Greens')
    axes[row, 2].set_title(f"Blob Ch-G", fontsize=9)
    axes[row, 2].axis("off")

    # Menormalisasi channel 2 (Red) untuk visualisasi
    ch2 = blob_channels[2]
    ch2_norm = ((ch2 - ch2.min()) / (ch2.max() - ch2.min() + 1e-8) * 255).astype(np.uint8)

    # Menampilkan channel Red dari blob
    axes[row, 3].imshow(ch2_norm, cmap='Reds')
    axes[row, 3].set_title(f"Blob Ch-R", fontsize=9)
    axes[row, 3].axis("off")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 2: Blob Creation untuk DNN Face Detection\n"
             "cv2.dnn.blobFromImage() - Preprocessing gambar menjadi tensor 4D",
             fontsize=13, fontweight="bold")

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan hasil visualisasi blob creation
output_path_1 = os.path.join(OUTPUT_DIR, "02_dnn_blob_creation.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 2. Skin Color Detection sebagai Alternatif Sederhana
# ============================================================

print("\n[INFO] Skin Color Detection - Alternatif Deteksi Wajah Sederhana")
print("-" * 50)

# Membuat figure untuk visualisasi skin detection
fig, axes = plt.subplots(3, 4, figsize=(18, 12))

# Memproses setiap gambar untuk skin detection
for row, (nama, img) in enumerate(gambar_list):
    # Mengkonversi gambar dari BGR ke ruang warna YCrCb
    # YCrCb lebih robust untuk deteksi kulit dibanding BGR
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Mendefinisikan range warna kulit di ruang warna YCrCb
    # Cr: 133-173 (komponen merah), Cb: 77-127 (komponen biru)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # Membuat mask berdasarkan range warna kulit
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Menerapkan operasi morfologi untuk membersihkan mask
    # Kernel untuk operasi morphological opening dan closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Opening: menghilangkan noise kecil (erosi lalu dilasi)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # Closing: mengisi lubang kecil (dilasi lalu erosi)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Menerapkan Gaussian blur untuk menghaluskan mask
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

    # Mencari kontur pada skin mask untuk menemukan area wajah
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Membuat salinan gambar untuk menggambar hasil deteksi
    img_skin_result = img.copy()

    # Menghitung jumlah wajah yang terdeteksi berdasarkan ukuran kontur
    face_count = 0

    # Memfilter kontur berdasarkan ukuran area (hanya area besar = wajah)
    for cnt in contours:
        # Menghitung area kontur
        area = cv2.contourArea(cnt)

        # Memfilter kontur yang cukup besar untuk menjadi wajah
        if area > 1000:
            # Mendapatkan bounding rectangle dari kontur
            x, y, w, h = cv2.boundingRect(cnt)

            # Menghitung rasio aspek (wajah biasanya mendekati persegi)
            aspect_ratio = w / float(h) if h > 0 else 0

            # Memfilter berdasarkan rasio aspek wajah (0.5 - 2.0)
            if 0.3 < aspect_ratio < 2.5:
                # Menggambar bounding box pada area kulit yang terdeteksi
                cv2.rectangle(img_skin_result, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Menambahkan label confidence berdasarkan area
                confidence = min(area / 10000.0, 1.0)
                cv2.putText(img_skin_result, f"{confidence:.2f}",
                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Menambah hitungan wajah terdeteksi
                face_count += 1

    # Menampilkan gambar asli
    axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[row, 0].set_title(f"{nama}\nAsli", fontsize=9)
    axes[row, 0].axis("off")

    # Menampilkan gambar dalam ruang warna YCrCb
    axes[row, 1].imshow(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB))
    axes[row, 1].set_title(f"YCrCb", fontsize=9)
    axes[row, 1].axis("off")

    # Menampilkan skin mask hasil segmentasi
    axes[row, 2].imshow(skin_mask, cmap='gray')
    axes[row, 2].set_title(f"Skin Mask", fontsize=9)
    axes[row, 2].axis("off")

    # Menampilkan hasil deteksi dengan bounding box
    axes[row, 3].imshow(cv2.cvtColor(img_skin_result, cv2.COLOR_BGR2RGB))
    axes[row, 3].set_title(f"Deteksi: {face_count} area", fontsize=9)
    axes[row, 3].axis("off")

    # Menampilkan informasi deteksi
    print(f"  {nama}: {face_count} area kulit terdeteksi")

# Menambahkan judul utama
plt.suptitle("Percobaan 2: Skin Color Detection (YCrCb)\n"
             "Alternatif sederhana untuk deteksi wajah",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi skin detection
output_path_2 = os.path.join(OUTPUT_DIR, "02_skin_detection.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 3. Perbandingan Haar Cascade pada Kondisi Berbeda
# ============================================================

print("\n[INFO] Perbandingan Haar Cascade pada Berbagai Kondisi")
print("-" * 50)

# Memuat cascade classifier untuk perbandingan
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Mendefinisikan variasi kondisi gambar untuk pengujian
kondisi_list = []

# Kondisi 1: gambar asli (normal)
kondisi_list.append(("Normal", img_single.copy()))

# Kondisi 2: gambar gelap (simulasi pencahayaan rendah)
img_gelap = cv2.convertScaleAbs(img_single, alpha=0.4, beta=0)
kondisi_list.append(("Gelap (40%)", img_gelap))

# Kondisi 3: gambar terang (simulasi overexposure)
img_terang = cv2.convertScaleAbs(img_single, alpha=1.8, beta=50)
kondisi_list.append(("Terang (180%)", img_terang))

# Kondisi 4: gambar dengan noise (simulasi kualitas rendah)
img_noise = img_single.copy().astype(np.float32)
noise = np.random.randn(*img_noise.shape) * 40
img_noise = np.clip(img_noise + noise, 0, 255).astype(np.uint8)
kondisi_list.append(("Noise (σ=40)", img_noise))

# Kondisi 5: gambar blur (simulasi out of focus)
img_blur = cv2.GaussianBlur(img_single, (15, 15), 5)
kondisi_list.append(("Blur (k=15)", img_blur))

# Kondisi 6: gambar dengan kontras tinggi
img_kontras = cv2.convertScaleAbs(img_single, alpha=2.5, beta=-100)
kondisi_list.append(("Kontras Tinggi", img_kontras))

# Membuat figure untuk perbandingan metode pada berbagai kondisi
fig, axes = plt.subplots(2, 6, figsize=(22, 8))

# Menyiapkan list untuk menyimpan hasil perbandingan
hasil_haar = []
hasil_skin = []

# Mendefinisikan range warna kulit di YCrCb
lower_skin = np.array([0, 133, 77], dtype=np.uint8)
upper_skin = np.array([255, 173, 127], dtype=np.uint8)

# Memproses setiap kondisi dan membandingkan metode deteksi
for col, (nama_kondisi, img_kondisi) in enumerate(kondisi_list):
    # --- Baris 1: Haar Cascade Detection ---
    # Mengkonversi ke grayscale untuk Haar
    gray_kondisi = cv2.cvtColor(img_kondisi, cv2.COLOR_BGR2GRAY)

    # Mencatat waktu mulai Haar
    t_haar_start = time.time()

    # Mendeteksi wajah dengan Haar Cascade
    faces_haar = face_cascade.detectMultiScale(gray_kondisi, 1.1, 5, minSize=(30, 30))

    # Menghitung waktu proses Haar
    t_haar = (time.time() - t_haar_start) * 1000

    # Membuat salinan gambar untuk menggambar hasil
    img_haar_res = img_kondisi.copy()

    # Menggambar bounding box pada wajah yang terdeteksi
    for (x, y, w, h) in faces_haar:
        cv2.rectangle(img_haar_res, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Menyimpan hasil
    hasil_haar.append(len(faces_haar))

    # Menampilkan hasil Haar pada baris atas
    axes[0, col].imshow(cv2.cvtColor(img_haar_res, cv2.COLOR_BGR2RGB))
    axes[0, col].set_title(f"Haar: {nama_kondisi}\n{len(faces_haar)} wajah ({t_haar:.1f}ms)", fontsize=8)
    axes[0, col].axis("off")

    # --- Baris 2: Skin Color Detection ---
    # Mencatat waktu mulai skin detection
    t_skin_start = time.time()

    # Mengkonversi ke YCrCb
    ycrcb_kondisi = cv2.cvtColor(img_kondisi, cv2.COLOR_BGR2YCrCb)

    # Membuat skin mask
    mask_kondisi = cv2.inRange(ycrcb_kondisi, lower_skin, upper_skin)

    # Membersihkan mask dengan operasi morfologi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_kondisi = cv2.morphologyEx(mask_kondisi, cv2.MORPH_OPEN, kernel)
    mask_kondisi = cv2.morphologyEx(mask_kondisi, cv2.MORPH_CLOSE, kernel)

    # Mencari kontur pada mask
    contours_kondisi, _ = cv2.findContours(mask_kondisi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Menghitung waktu proses skin detection
    t_skin = (time.time() - t_skin_start) * 1000

    # Membuat salinan gambar
    img_skin_res = img_kondisi.copy()

    # Menghitung area kulit terdeteksi
    skin_count = 0
    for cnt in contours_kondisi:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_skin_res, (x, y), (x + w, y + h), (0, 255, 0), 2)
            skin_count += 1

    # Menyimpan hasil
    hasil_skin.append(skin_count)

    # Menampilkan hasil skin detection pada baris bawah
    axes[1, col].imshow(cv2.cvtColor(img_skin_res, cv2.COLOR_BGR2RGB))
    axes[1, col].set_title(f"Skin: {nama_kondisi}\n{skin_count} area ({t_skin:.1f}ms)", fontsize=8)
    axes[1, col].axis("off")

    # Menampilkan perbandingan
    print(f"  {nama_kondisi:20s} → Haar: {len(faces_haar)} wajah | Skin: {skin_count} area")

# Menambahkan judul utama
plt.suptitle("Percobaan 2: Perbandingan Haar Cascade vs Skin Detection\n"
             "Robustness pada berbagai kondisi gambar",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil perbandingan
output_path_3 = os.path.join(OUTPUT_DIR, "02_haar_vs_dnn_comparison.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 2")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.dnn.blobFromImage()      → Preprocessing gambar untuk DNN")
print("     - scalefactor: penskalaan intensitas piksel")
print("     - size: ukuran input (biasanya 300x300 atau 224x224)")
print("     - mean: nilai mean yang dikurangi (normalisasi)")
print("     - swapRB: menukar channel R dan B")
print("     - Output: tensor 4D (batch, channels, height, width)")
print("  2. cv2.inRange()                → Segmentasi berdasarkan range warna")
print("     - Digunakan untuk deteksi warna kulit di YCrCb")
print("  3. cv2.findContours()           → Mencari kontur pada mask biner")
print("  4. cv2.morphologyEx()           → Operasi morfologi (open/close)")
print("\nPerbandingan Haar vs Skin Detection:")
for i, (nama, _) in enumerate(kondisi_list):
    print(f"  {nama:20s}: Haar={hasil_haar[i]} | Skin={hasil_skin[i]}")
print("\nKesimpulan:")
print("  - Haar Cascade: cepat, tapi sensitif terhadap pencahayaan/noise")
print("  - Skin Detection: sederhana, tapi tidak spesifik untuk wajah")
print("  - DNN: paling akurat (membutuhkan model pre-trained)")
print("=" * 60)
