"""
==========================================================================
PERCOBAAN 1: KLASIFIKASI GAMBAR DENGAN OPENCV DNN
==========================================================================
Program ini mempelajari cara menggunakan modul cv2.dnn untuk melakukan
klasifikasi gambar menggunakan model neural network pre-trained tanpa
memerlukan framework deep learning berat seperti PyTorch atau TensorFlow.

Fungsi utama yang dipelajari:
- cv2.dnn.blobFromImage()  : Membuat blob (input tensor) dari gambar
- cv2.dnn.readNet()        : Memuat model neural network
- net.setInput()           : Memasukkan data ke network
- net.forward()            : Melakukan inferensi (forward pass)

Konsep yang dipelajari:
- Preprocessing gambar untuk deep learning (resize, normalisasi, mean subtraction)
- Inferensi model klasifikasi pada gambar
- Interpretasi output (probabilitas per kelas)
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan deep learning
import cv2

# Mengimpor NumPy untuk operasi array dan matematika
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur kecepatan inferensi
import time

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 1: KLASIFIKASI GAMBAR DENGAN OPENCV DNN")
print("=" * 60)

# ============================================================
# 1. Memahami konsep blob dalam deep learning
# ============================================================
# Blob (Binary Large Object) adalah format input yang dibutuhkan
# neural network: 4D tensor (batchSize, channels, height, width)
# cv2.dnn.blobFromImage() melakukan:
# - Resize gambar ke ukuran yang diharapkan model
# - Mean subtraction (mengurangi rata-rata dataset training)
# - Scaling (normalisasi nilai piksel)
# - Swap channel (BGR -> RGB jika diperlukan)

# Membaca gambar sample untuk klasifikasi
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))
if img is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py dulu.")
    exit()

print("\n--- 1. Membuat Blob dari Gambar ---")

# Membuat blob dengan parameter standar untuk model ImageNet
# scalefactor=1.0/255.0 : normalisasi piksel dari [0,255] ke [0,1]
# size=(224,224) : ukuran input yang diharapkan model (224x224 piksel)
# mean=(0,0,0) : nilai mean yang dikurangi dari setiap channel
# swapRB=True : menukar channel Red dan Blue (BGR->RGB)
blob = cv2.dnn.blobFromImage(
    img,                    # Gambar input
    scalefactor=1.0/255.0,  # Faktor skala untuk normalisasi
    size=(224, 224),        # Ukuran target (width, height)
    mean=(0, 0, 0),         # Nilai mean untuk subtraction
    swapRB=True,            # Swap Blue-Red channel
    crop=False              # Apakah crop ke ukuran target
)

# Menampilkan informasi blob yang dihasilkan
print(f"  Shape gambar asli  : {img.shape}")       # (H, W, C)
print(f"  Shape blob         : {blob.shape}")       # (1, C, H, W) = (batch, channels, height, width)
print(f"  Tipe data blob     : {blob.dtype}")       # float32
print(f"  Range nilai blob   : [{blob.min():.4f}, {blob.max():.4f}]")

# ============================================================
# 2. Demonstrasi berbagai parameter blobFromImage
# ============================================================
print("\n--- 2. Variasi Parameter Blob ---")

# Blob tanpa normalisasi (range 0-255)
blob_raw = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (0, 0, 0), False, False)
print(f"  Blob tanpa normalisasi : range [{blob_raw.min():.1f}, {blob_raw.max():.1f}]")

# Blob dengan mean subtraction ImageNet (nilai mean dataset ImageNet)
# Mean ImageNet: R=123.68, G=116.78, B=103.94
blob_imagenet = cv2.dnn.blobFromImage(
    img, 1.0, (224, 224),
    mean=(103.94, 116.78, 123.68),  # BGR order karena swapRB=False
    swapRB=False, crop=False
)
print(f"  Blob dengan ImageNet mean : range [{blob_imagenet.min():.1f}, {blob_imagenet.max():.1f}]")

# Blob dengan normalisasi [0,1] + mean subtraction
blob_norm = cv2.dnn.blobFromImage(
    img, 1.0/255.0, (224, 224),
    mean=(0.485, 0.456, 0.406),  # Mean ImageNet yang sudah dinormalisasi
    swapRB=True, crop=False
)
print(f"  Blob norm+mean : range [{blob_norm.min():.4f}, {blob_norm.max():.4f}]")

# ============================================================
# 3. Visualisasi blob channels
# ============================================================
print("\n--- 3. Visualisasi Blob Channels ---")

# Mengekstrak blob pertama dalam batch (index 0)
# blob shape: (1, 3, 224, 224) -> kita ambil batch ke-0
blob_visual = cv2.dnn.blobFromImage(img, 1.0/255.0, (224, 224), swapRB=True)
blob_channels = blob_visual[0]  # Shape: (3, 224, 224)

# Membuat figure dengan 4 subplot: gambar asli + 3 channel blob
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Menampilkan gambar asli (konversi BGR ke RGB)
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Gambar Asli")
axes[0].axis('off')

# Menampilkan channel Red dari blob
axes[1].imshow(blob_channels[0], cmap='Reds')
axes[1].set_title("Channel R (Blob)")
axes[1].axis('off')

# Menampilkan channel Green dari blob
axes[2].imshow(blob_channels[1], cmap='Greens')
axes[2].set_title("Channel G (Blob)")
axes[2].axis('off')

# Menampilkan channel Blue dari blob
axes[3].imshow(blob_channels[2], cmap='Blues')
axes[3].set_title("Channel B (Blob)")
axes[3].axis('off')

# Menyimpan visualisasi ke file
plt.suptitle("Percobaan 1: Blob Channels dari cv2.dnn.blobFromImage()", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_blob_channels.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] output/01_blob_channels.png")

# ============================================================
# 4. Simulasi klasifikasi sederhana (tanpa model besar)
# ============================================================
print("\n--- 4. Simulasi Klasifikasi Sederhana ---")

# Karena model besar memerlukan download terpisah, kita demonstrasikan
# konsep klasifikasi dengan membandingkan fitur sederhana
gambar_files = ["kucing.jpg", "anjing.jpg", "mobil.jpg", "bunga.jpg", "gedung.jpg"]
gambar_labels = ["Kucing", "Anjing", "Mobil", "Bunga", "Gedung"]

# Menghitung statistik warna sebagai "fitur" sederhana
fitur_gambar = []
for fname in gambar_files:
    path = os.path.join(IMAGE_DIR, fname)
    g = cv2.imread(path)
    if g is not None:
        # Menghitung rata-rata dan standar deviasi tiap channel
        mean_val = np.mean(g, axis=(0, 1))
        std_val = np.std(g, axis=(0, 1))
        fitur_gambar.append({
            'file': fname,
            'mean_bgr': mean_val,
            'std_bgr': std_val
        })
        print(f"  {fname}: Mean BGR = [{mean_val[0]:.1f}, {mean_val[1]:.1f}, {mean_val[2]:.1f}]")

# ============================================================
# 5. Membuat blob batch untuk multiple gambar sekaligus
# ============================================================
print("\n--- 5. Batch Processing dengan blobFromImages ---")

# Mengumpulkan beberapa gambar untuk diproses sekaligus
gambar_batch = []
for fname in gambar_files:
    path = os.path.join(IMAGE_DIR, fname)
    g = cv2.imread(path)
    if g is not None:
        # Resize setiap gambar ke ukuran yang sama sebelum membuat blob
        g_resized = cv2.resize(g, (224, 224))
        gambar_batch.append(g_resized)

# Membuat blob dari batch gambar menggunakan blobFromImages
# Ini lebih efisien daripada membuat blob satu per satu
blob_batch = cv2.dnn.blobFromImages(
    gambar_batch,           # List gambar
    scalefactor=1.0/255.0,  # Normalisasi
    size=(224, 224),        # Ukuran target
    mean=(0, 0, 0),         # Mean subtraction
    swapRB=True,            # Swap channel
    crop=False              # Tanpa crop
)

# Menampilkan informasi batch blob
print(f"  Jumlah gambar dalam batch : {len(gambar_batch)}")
print(f"  Shape batch blob          : {blob_batch.shape}")
print(f"  Memory batch blob         : {blob_batch.nbytes / 1024:.1f} KB")

# ============================================================
# 6. Mengukur waktu pembuatan blob
# ============================================================
print("\n--- 6. Benchmark Waktu Pembuatan Blob ---")

# Mengukur waktu untuk berbagai ukuran input
ukuran_input = [(224, 224), (299, 299), (416, 416), (640, 640)]

for size in ukuran_input:
    # Melakukan 100 iterasi untuk mendapat rata-rata yang stabil
    start = time.time()
    for _ in range(100):
        _ = cv2.dnn.blobFromImage(img, 1.0/255.0, size, swapRB=True)
    elapsed = (time.time() - start) / 100 * 1000  # Konversi ke millisecond

    print(f"  Size {size[0]}x{size[1]} : {elapsed:.2f} ms per blob")

# ============================================================
# 7. Visualisasi perbandingan preprocessing
# ============================================================
print("\n--- 7. Visualisasi Perbandingan Preprocessing ---")

# Membuat figure untuk menampilkan efek preprocessing yang berbeda
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Gambar asli
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Asli")

# Resize ke 224x224
img_224 = cv2.resize(img, (224, 224))
axes[0, 1].imshow(cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Resize 224x224")

# Resize ke 299x299
img_299 = cv2.resize(img, (299, 299))
axes[0, 2].imshow(cv2.cvtColor(img_299, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Resize 299x299")

# Center crop dari gambar asli
h, w = img.shape[:2]
crop_size = min(h, w)
start_x = (w - crop_size) // 2
start_y = (h - crop_size) // 2
img_crop = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
img_crop = cv2.resize(img_crop, (224, 224))
axes[0, 3].imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Center Crop 224x224")

# Normalisasi [0, 1]
img_norm = img_224.astype(np.float32) / 255.0
axes[1, 0].imshow(cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Normalisasi [0,1]")

# Mean subtraction
img_mean = img_224.astype(np.float32) - np.array([103.94, 116.78, 123.68])
img_mean_vis = np.clip((img_mean - img_mean.min()) / (img_mean.max() - img_mean.min()), 0, 1)
axes[1, 1].imshow(img_mean_vis)
axes[1, 1].set_title("Mean Subtracted")

# Standardized (mean+std normalization)
img_std = (img_224.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
img_std_vis = np.clip((img_std - img_std.min()) / (img_std.max() - img_std.min()), 0, 1)
axes[1, 2].imshow(img_std_vis)
axes[1, 2].set_title("Standardized (ImageNet)")

# Histogram equalized sebelum blob
img_eq = cv2.cvtColor(img_224, cv2.COLOR_BGR2YCrCb)
img_eq[:, :, 0] = cv2.equalizeHist(img_eq[:, :, 0])
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YCrCb2BGR)
axes[1, 3].imshow(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("Histogram Equalized")

for ax in axes.flat:
    ax.axis('off')

plt.suptitle("Percobaan 1: Perbandingan Teknik Preprocessing untuk DNN", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_preprocessing_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] output/01_preprocessing_comparison.png")

# ============================================================
# 8. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 1")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. cv2.dnn.blobFromImage() membuat tensor 4D (N,C,H,W) dari gambar
2. Parameter penting: scalefactor, size, mean, swapRB, crop
3. Blob adalah format input standar untuk neural network di OpenCV
4. blobFromImages() memproses batch gambar sekaligus (lebih efisien)
5. Preprocessing yang tepat SANGAT mempengaruhi hasil klasifikasi
6. Ukuran input mempengaruhi waktu komputasi blob

Output disimpan di folder: output/
- 01_blob_channels.png : Visualisasi channel blob
- 01_preprocessing_comparison.png : Perbandingan teknik preprocessing
""")
