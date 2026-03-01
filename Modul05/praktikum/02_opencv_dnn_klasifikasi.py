"""
==========================================================================
PERCOBAAN 2: KLASIFIKASI GAMBAR DENGAN OPENCV DNN
==========================================================================
Program ini mempelajari pipeline klasifikasi gambar menggunakan modul
cv2.dnn dari OpenCV. Karena file model besar mungkin tidak tersedia,
program ini mendemonstrasikan konsep klasifikasi menggunakan simulasi
neural network sederhana berbasis histogram warna.

Fungsi utama yang dipelajari:
- cv2.dnn.readNet()        : Memuat model neural network (konsep)
- cv2.dnn.blobFromImage()  : Membuat blob (input tensor) dari gambar
- net.setInput()           : Memasukkan data ke network (konsep)
- net.forward()            : Melakukan inferensi/forward pass (konsep)
- cv2.calcHist()           : Menghitung histogram warna sebagai fitur
- cv2.compareHist()        : Membandingkan histogram untuk klasifikasi

Konsep yang dipelajari:
- Pipeline klasifikasi gambar: preprocessing -> inferensi -> postprocessing
- Konsep confidence score dan top-K predictions
- Pengukuran waktu inferensi per gambar
- Simulasi klasifikasi berbasis fitur warna
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan modul DNN
import cv2

# Mengimpor NumPy untuk operasi array dan matematika
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi hasil klasifikasi
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu inferensi
import time

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
print("PERCOBAAN 2: KLASIFIKASI GAMBAR DENGAN OPENCV DNN")
print("=" * 60)

# ============================================================
# 1. Memahami konsep klasifikasi gambar dengan DNN
# ============================================================
print("\n--- 1. Konsep Pipeline Klasifikasi DNN ---")

# Menjelaskan pipeline klasifikasi DNN secara konseptual
print("""
  Pipeline Klasifikasi Gambar dengan DNN:
  1. Load Model    : cv2.dnn.readNet(model, config)
  2. Preprocessing : cv2.dnn.blobFromImage(img, scale, size, mean)
  3. Set Input     : net.setInput(blob)
  4. Forward Pass  : output = net.forward()
  5. Postprocessing: Interpretasi output (argmax, softmax)
""")

# Mendefinisikan daftar file gambar untuk klasifikasi
gambar_files = ["kucing.jpg", "anjing.jpg", "mobil.jpg", "bunga.jpg", "gedung.jpg"]

# Mendefinisikan label kelas untuk setiap gambar
gambar_labels = ["Kucing", "Anjing", "Mobil", "Bunga", "Gedung"]

# Mendefinisikan daftar semua kelas yang mungkin (termasuk kelas tambahan)
semua_kelas = ["Kucing", "Anjing", "Mobil", "Bunga", "Gedung",
               "Burung", "Ikan", "Sepeda", "Pohon", "Rumah"]

# Memuat semua gambar dari folder image
gambar_list = []
for fname in gambar_files:
    # Membaca gambar dari path yang ditentukan
    path = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(path)
    if img is not None:
        # Menambahkan gambar yang berhasil dimuat ke list
        gambar_list.append(img)
        print(f"  [OK] {fname} dimuat - ukuran: {img.shape}")
    else:
        # Menampilkan pesan error jika gambar tidak ditemukan
        print(f"  [ERROR] {fname} tidak ditemukan!")

# Memeriksa apakah ada gambar yang berhasil dimuat
if len(gambar_list) == 0:
    print("[ERROR] Tidak ada gambar! Jalankan download_image.py terlebih dahulu.")
    exit()

# ============================================================
# 2. Demonstrasi konsep cv2.dnn.readNet()
# ============================================================
print("\n--- 2. Konsep Pemuatan Model DNN ---")

# Menjelaskan format model yang didukung oleh OpenCV DNN
print("""
  Format model yang didukung cv2.dnn.readNet():
  - Caffe (.caffemodel + .prototxt)      : cv2.dnn.readNetFromCaffe()
  - TensorFlow (.pb + .pbtxt)            : cv2.dnn.readNetFromTensorflow()
  - Darknet/YOLO (.weights + .cfg)       : cv2.dnn.readNetFromDarknet()
  - ONNX (.onnx)                         : cv2.dnn.readNetFromONNX()
  - Torch (.t7 / .net)                   : cv2.dnn.readNetFromTorch()

  Contoh penggunaan (jika model tersedia):
    net = cv2.dnn.readNet("model.onnx")
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (224,224), swapRB=True)
    net.setInput(blob)
    output = net.forward()
""")

# Mendemonstrasikan pembuatan blob untuk berbagai gambar
print("  Demonstrasi pembuatan blob untuk setiap gambar:")
for i, (img, label) in enumerate(zip(gambar_list, gambar_labels)):
    # Membuat blob dari gambar dengan parameter standar ImageNet
    blob = cv2.dnn.blobFromImage(
        img,                    # Gambar input
        scalefactor=1.0/255.0,  # Normalisasi ke range [0, 1]
        size=(224, 224),        # Ukuran input model standar
        mean=(0, 0, 0),         # Mean subtraction
        swapRB=True,            # Konversi BGR ke RGB
        crop=False              # Tanpa cropping
    )
    # Menampilkan informasi blob yang dihasilkan
    print(f"  {label}: blob shape={blob.shape}, range=[{blob.min():.3f}, {blob.max():.3f}]")

# ============================================================
# 3. Simulasi klasifikasi dengan fitur histogram warna
# ============================================================
print("\n--- 3. Simulasi Klasifikasi Berbasis Histogram ---")

# Menjelaskan pendekatan simulasi
print("  Karena model besar tidak tersedia, kita simulasikan klasifikasi")
print("  menggunakan histogram warna sebagai 'fitur' neural network.\n")

def ekstrak_fitur_histogram(gambar, bins=32):
    """
    Mengekstrak fitur histogram warna dari gambar.
    Fungsi ini mensimulasikan 'feature extraction' yang dilakukan
    oleh layer-layer CNN pada model deep learning.
    """
    # Mengkonversi gambar dari BGR ke HSV untuk fitur warna yang lebih baik
    hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)

    # Menghitung histogram untuk channel Hue (warna)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])

    # Menghitung histogram untuk channel Saturation (kejenuhan warna)
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])

    # Menghitung histogram untuk channel Value (kecerahan)
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    # Menggabungkan ketiga histogram menjadi satu vektor fitur
    fitur = np.concatenate([hist_h, hist_s, hist_v]).flatten()

    # Menormalisasi vektor fitur agar total sum = 1
    fitur = fitur / (fitur.sum() + 1e-7)

    # Mengembalikan vektor fitur yang sudah dinormalisasi
    return fitur

# Mengekstrak fitur dari setiap gambar referensi (sebagai 'model' kita)
fitur_referensi = {}
for img, label in zip(gambar_list, gambar_labels):
    # Mengekstrak fitur histogram dari gambar
    fitur = ekstrak_fitur_histogram(img)
    # Menyimpan fitur ke dictionary dengan label sebagai key
    fitur_referensi[label] = fitur
    print(f"  Fitur {label}: dimensi={fitur.shape}, sum={fitur.sum():.4f}")

# ============================================================
# 4. Klasifikasi gambar dan prediksi top-5
# ============================================================
print("\n--- 4. Klasifikasi Gambar dengan Top-5 Predictions ---")

def klasifikasi_gambar(gambar, fitur_ref, daftar_kelas):
    """
    Mengklasifikasikan gambar berdasarkan kesamaan histogram.
    Mensimulasikan output softmax dari neural network.
    """
    # Mengekstrak fitur dari gambar yang akan diklasifikasikan
    fitur_input = ekstrak_fitur_histogram(gambar)

    # Menghitung skor kesamaan dengan setiap kelas referensi
    skor = {}
    for label, fitur_ref_val in fitur_ref.items():
        # Menggunakan korelasi histogram sebagai ukuran kesamaan
        similarity = cv2.compareHist(
            fitur_input.astype(np.float32),
            fitur_ref_val.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        # Menyimpan skor kesamaan (range -1 sampai 1)
        skor[label] = max(0, similarity)

    # Menambahkan skor kecil untuk kelas yang tidak ada di referensi
    for kelas in daftar_kelas:
        if kelas not in skor:
            # Memberikan skor random kecil untuk kelas lain
            skor[kelas] = np.random.uniform(0.01, 0.1)

    # Mengkonversi skor ke probabilitas menggunakan softmax
    skor_array = np.array(list(skor.values()))

    # Menerapkan operasi softmax: exp(x) / sum(exp(x))
    exp_skor = np.exp(skor_array * 5)  # Scaling untuk mempertajam distribusi
    probabilitas = exp_skor / exp_skor.sum()

    # Membuat dictionary label -> probabilitas
    hasil = dict(zip(skor.keys(), probabilitas))

    # Mengurutkan berdasarkan probabilitas tertinggi
    hasil_sorted = dict(sorted(hasil.items(), key=lambda x: x[1], reverse=True))

    # Mengembalikan hasil prediksi yang sudah diurutkan
    return hasil_sorted

# Menyimpan semua hasil klasifikasi untuk visualisasi
semua_hasil = []
waktu_inferensi = []

for i, (img, label_asli) in enumerate(zip(gambar_list, gambar_labels)):
    # Mengukur waktu mulai inferensi
    start_time = time.time()

    # Melakukan klasifikasi gambar
    hasil = klasifikasi_gambar(img, fitur_referensi, semua_kelas)

    # Mengukur waktu akhir inferensi
    elapsed = (time.time() - start_time) * 1000  # Konversi ke millisecond

    # Menyimpan waktu inferensi
    waktu_inferensi.append(elapsed)

    # Menyimpan hasil klasifikasi
    semua_hasil.append(hasil)

    # Menampilkan top-5 prediksi
    print(f"\n  Gambar: {gambar_files[i]} (Label asli: {label_asli})")
    print(f"  Waktu inferensi: {elapsed:.2f} ms")
    print(f"  Top-5 Predictions:")

    # Mengambil 5 prediksi teratas
    top5 = list(hasil.items())[:5]
    for rank, (kelas, prob) in enumerate(top5, 1):
        # Menampilkan ranking, kelas, dan probabilitas
        bar = "█" * int(prob * 40)
        print(f"    #{rank} {kelas:10s}: {prob:.4f} ({prob*100:.1f}%) {bar}")

# ============================================================
# 5. Visualisasi hasil klasifikasi
# ============================================================
print("\n--- 5. Visualisasi Hasil Klasifikasi ---")

# Membuat figure untuk menampilkan gambar dengan prediksi
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Meratakan array axes untuk iterasi mudah
axes_flat = axes.flatten()

for i, (img, label_asli) in enumerate(zip(gambar_list, gambar_labels)):
    # Mengkonversi gambar dari BGR ke RGB untuk matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Menampilkan gambar pada subplot
    axes_flat[i].imshow(img_rgb)

    # Mengambil prediksi teratas
    top1_kelas = list(semua_hasil[i].keys())[0]
    top1_prob = list(semua_hasil[i].values())[0]

    # Menentukan warna judul berdasarkan kebenaran prediksi
    warna_judul = 'green' if top1_kelas == label_asli else 'red'

    # Menambahkan judul dengan prediksi dan confidence
    axes_flat[i].set_title(
        f"Asli: {label_asli}\nPrediksi: {top1_kelas} ({top1_prob*100:.1f}%)",
        fontsize=10, color=warna_judul, fontweight='bold'
    )

    # Menghilangkan axis untuk tampilan yang bersih
    axes_flat[i].axis('off')

# Menyembunyikan subplot kosong jika ada
if len(gambar_list) < 6:
    axes_flat[5].axis('off')
    axes_flat[5].set_visible(False)

# Menambahkan judul utama
plt.suptitle("Percobaan 2: Hasil Klasifikasi Gambar (Simulasi DNN)",
             fontsize=14, fontweight='bold')

# Mengatur layout agar tidak tumpang tindih
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "02_klasifikasi_hasil.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure untuk membebaskan memori
plt.close()

# Menampilkan pesan bahwa file berhasil disimpan
print("  [SAVED] output/02_klasifikasi_hasil.png")

# ============================================================
# 6. Visualisasi confidence bar chart
# ============================================================
print("\n--- 6. Visualisasi Confidence Chart ---")

# Membuat figure dengan subplot untuk setiap gambar
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Meratakan array axes untuk iterasi mudah
axes_flat = axes.flatten()

# Mendefinisikan palet warna untuk bar chart
warna_bar = plt.cm.viridis(np.linspace(0.2, 0.8, 5))

for i, (label_asli, hasil) in enumerate(zip(gambar_labels, semua_hasil)):
    # Mengambil top-5 prediksi
    top5_kelas = list(hasil.keys())[:5]
    top5_prob = list(hasil.values())[:5]

    # Membuat bar chart horizontal
    y_pos = np.arange(len(top5_kelas))
    bars = axes_flat[i].barh(y_pos, top5_prob, color=warna_bar, edgecolor='gray')

    # Menambahkan label probabilitas pada setiap bar
    for bar, prob in zip(bars, top5_prob):
        # Menampilkan nilai probabilitas di ujung bar
        axes_flat[i].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                         f'{prob*100:.1f}%', va='center', fontsize=8)

    # Mengatur label sumbu Y dengan nama kelas
    axes_flat[i].set_yticks(y_pos)
    axes_flat[i].set_yticklabels(top5_kelas, fontsize=9)

    # Mengatur batas sumbu X
    axes_flat[i].set_xlim(0, max(top5_prob) * 1.3)

    # Menambahkan judul subplot
    axes_flat[i].set_title(f"Top-5: {label_asli} ({gambar_files[i]})", fontsize=10)

    # Menambahkan label sumbu X
    axes_flat[i].set_xlabel("Confidence", fontsize=8)

# Menyembunyikan subplot kosong jika ada
if len(gambar_list) < 6:
    axes_flat[5].axis('off')
    axes_flat[5].set_visible(False)

# Menambahkan judul utama
plt.suptitle("Percobaan 2: Confidence Score Top-5 Predictions per Gambar",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan chart ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "02_confidence_chart.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/02_confidence_chart.png")

# ============================================================
# 7. Pengukuran waktu inferensi
# ============================================================
print("\n--- 7. Analisis Waktu Inferensi ---")

# Menghitung statistik waktu inferensi
waktu_rata = np.mean(waktu_inferensi)
waktu_min = np.min(waktu_inferensi)
waktu_max = np.max(waktu_inferensi)
waktu_std = np.std(waktu_inferensi)

# Menampilkan statistik waktu inferensi
print(f"  Rata-rata waktu inferensi : {waktu_rata:.2f} ms")
print(f"  Waktu minimum             : {waktu_min:.2f} ms")
print(f"  Waktu maksimum            : {waktu_max:.2f} ms")
print(f"  Standar deviasi           : {waktu_std:.2f} ms")

# Menampilkan waktu per gambar
print("\n  Detail waktu per gambar:")
for fname, waktu in zip(gambar_files, waktu_inferensi):
    # Menampilkan nama file dan waktu inferensi
    print(f"    {fname:15s}: {waktu:.2f} ms")

# ============================================================
# 8. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 2")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. Pipeline klasifikasi DNN: load model -> blob -> set input -> forward
2. cv2.dnn.readNet() mendukung Caffe, TensorFlow, Darknet, ONNX, Torch
3. cv2.dnn.blobFromImage() mengubah gambar menjadi tensor 4D (N,C,H,W)
4. Simulasi klasifikasi menggunakan histogram warna sebagai fitur
5. Konsep top-K predictions dengan confidence score (softmax)
6. Pengukuran waktu inferensi untuk evaluasi performa

Output disimpan di folder: output/
- 02_klasifikasi_hasil.png  : Visualisasi gambar dengan prediksi
- 02_confidence_chart.png   : Bar chart confidence score top-5
""")
