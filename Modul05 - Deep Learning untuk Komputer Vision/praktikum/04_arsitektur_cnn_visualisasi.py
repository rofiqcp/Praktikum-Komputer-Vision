"""
==========================================================================
PERCOBAAN 4: VISUALISASI ARSITEKTUR CNN
==========================================================================
Program ini mempelajari visualisasi arsitektur Convolutional Neural Network
(CNN) menggunakan fungsi gambar OpenCV. Program mendemonstrasikan operasi
konvolusi secara manual, visualisasi feature maps, dan operasi pooling.

Fungsi utama yang dipelajari:
- cv2.filter2D()       : Menerapkan kernel konvolusi pada gambar
- cv2.rectangle()      : Menggambar kotak untuk diagram arsitektur
- cv2.putText()        : Menambahkan teks pada diagram
- cv2.arrowedLine()    : Menggambar panah untuk aliran data
- np.convolve()        : Operasi konvolusi dengan numpy

Konsep yang dipelajari:
- Layer-layer CNN: Convolutional, Pooling, Fully Connected
- Operasi konvolusi 2D (filter/kernel)
- Feature maps pada berbagai level abstraksi
- Max pooling dan average pooling
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan menggambar
import cv2

# Mengimpor NumPy untuk operasi array dan konvolusi manual
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi feature maps
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
print("PERCOBAAN 4: VISUALISASI ARSITEKTUR CNN")
print("=" * 60)

# ============================================================
# 1. Menggambar diagram arsitektur CNN
# ============================================================
print("\n--- 1. Menggambar Diagram Arsitektur CNN ---")

# Membuat canvas putih untuk diagram arsitektur
canvas_width = 1400
canvas_height = 500
diagram = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Mendefinisikan layer-layer CNN dan propertinya
layers = [
    {"nama": "Input", "ukuran": "224x224x3", "warna": (200, 220, 255), "x": 50, "w": 80, "h": 150},
    {"nama": "Conv1", "ukuran": "112x112x64", "warna": (180, 255, 180), "x": 180, "w": 70, "h": 130},
    {"nama": "Pool1", "ukuran": "56x56x64", "warna": (255, 220, 180), "x": 300, "w": 60, "h": 110},
    {"nama": "Conv2", "ukuran": "56x56x128", "warna": (180, 255, 180), "x": 410, "w": 65, "h": 110},
    {"nama": "Pool2", "ukuran": "28x28x128", "warna": (255, 220, 180), "x": 525, "w": 55, "h": 90},
    {"nama": "Conv3", "ukuran": "28x28x256", "warna": (180, 255, 180), "x": 630, "w": 60, "h": 90},
    {"nama": "Pool3", "ukuran": "14x14x256", "warna": (255, 220, 180), "x": 740, "w": 50, "h": 70},
    {"nama": "Conv4", "ukuran": "14x14x512", "warna": (180, 255, 180), "x": 840, "w": 55, "h": 70},
    {"nama": "Pool4", "ukuran": "7x7x512", "warna": (255, 220, 180), "x": 945, "w": 45, "h": 55},
    {"nama": "Flatten", "ukuran": "25088", "warna": (255, 200, 255), "x": 1040, "w": 40, "h": 180},
    {"nama": "FC1", "ukuran": "4096", "warna": (200, 200, 255), "x": 1130, "w": 40, "h": 140},
    {"nama": "FC2", "ukuran": "1000", "warna": (255, 200, 200), "x": 1220, "w": 40, "h": 100},
    {"nama": "Output", "ukuran": "Softmax", "warna": (255, 255, 180), "x": 1310, "w": 60, "h": 80},
]

# Menghitung posisi Y tengah
y_center = canvas_height // 2

# Menggambar judul diagram
cv2.putText(diagram, "Arsitektur CNN (VGG-like)", (400, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

# Menggambar setiap layer pada diagram
for i, layer in enumerate(layers):
    # Menghitung koordinat kotak layer
    x1 = layer["x"]
    y1 = y_center - layer["h"] // 2
    x2 = x1 + layer["w"]
    y2 = y_center + layer["h"] // 2

    # Menggambar kotak layer dengan warna yang ditentukan
    cv2.rectangle(diagram, (x1, y1), (x2, y2), layer["warna"], -1)

    # Menggambar border kotak layer
    cv2.rectangle(diagram, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # Menambahkan nama layer di atas kotak
    text_size = cv2.getTextSize(layer["nama"], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    text_x = x1 + (layer["w"] - text_size[0]) // 2
    cv2.putText(diagram, layer["nama"], (text_x, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Menambahkan ukuran layer di bawah kotak
    size_text = layer["ukuran"]
    size_size = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
    size_x = x1 + (layer["w"] - size_size[0]) // 2
    cv2.putText(diagram, size_text, (size_x, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    # Menggambar panah ke layer berikutnya
    if i < len(layers) - 1:
        # Menghitung titik awal dan akhir panah
        arrow_start = (x2 + 2, y_center)
        arrow_end = (layers[i + 1]["x"] - 2, y_center)
        # Menggambar panah penghubung antar layer
        cv2.arrowedLine(diagram, arrow_start, arrow_end, (100, 100, 100), 2,
                       tipLength=0.3)

# Menambahkan legend di bagian bawah
legend_y = canvas_height - 50
legend_items = [
    ("Conv Layer", (180, 255, 180)),
    ("Pooling Layer", (255, 220, 180)),
    ("FC Layer", (200, 200, 255)),
    ("Output", (255, 255, 180))
]

# Menggambar kotak legend untuk setiap tipe layer
legend_x = 50
for label, warna in legend_items:
    # Menggambar kotak warna legend
    cv2.rectangle(diagram, (legend_x, legend_y), (legend_x + 20, legend_y + 15), warna, -1)
    cv2.rectangle(diagram, (legend_x, legend_y), (legend_x + 20, legend_y + 15), (0, 0, 0), 1)
    # Menambahkan teks label legend
    cv2.putText(diagram, label, (legend_x + 25, legend_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    legend_x += 150

# Menyimpan diagram arsitektur ke file
cv2.imwrite(os.path.join(OUTPUT_DIR, "04_arsitektur_cnn.png"), diagram)

# Menampilkan pesan penyimpanan
print("  [SAVED] output/04_arsitektur_cnn.png")
print(f"  Jumlah layer yang digambar: {len(layers)}")

# ============================================================
# 2. Operasi konvolusi manual dan dengan cv2.filter2D()
# ============================================================
print("\n--- 2. Operasi Konvolusi pada Gambar ---")

# Memuat gambar untuk operasi konvolusi
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Mengkonversi gambar ke grayscale untuk konvolusi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Meresize gambar untuk mempercepat proses
gray = cv2.resize(gray, (256, 256))

# Menampilkan informasi gambar
print(f"  Ukuran gambar: {gray.shape}")

# Mendefinisikan berbagai kernel konvolusi
kernels = {
    "Identity": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float32),

    "Edge Detection": np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32),

    "Sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32),

    "Gaussian Blur": np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0,

    "Emboss": np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ], dtype=np.float32),

    "Sobel X": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32),

    "Sobel Y": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32),

    "Laplacian": np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32),
}

# Menerapkan setiap kernel pada gambar menggunakan cv2.filter2D()
hasil_konvolusi = {}
for nama_kernel, kernel in kernels.items():
    # Menerapkan konvolusi dengan cv2.filter2D()
    hasil = cv2.filter2D(gray, cv2.CV_32F, kernel)
    # Menyimpan hasil konvolusi
    hasil_konvolusi[nama_kernel] = hasil
    # Menampilkan informasi hasil
    print(f"  Kernel {nama_kernel:18s}: range=[{hasil.min():.1f}, {hasil.max():.1f}]")

# ============================================================
# 3. Visualisasi feature maps (hasil konvolusi)
# ============================================================
print("\n--- 3. Visualisasi Feature Maps ---")

# Membuat figure dengan subplot untuk setiap kernel
fig, axes = plt.subplots(3, 3, figsize=(14, 14))

# Meratakan axes untuk iterasi
axes_flat = axes.flatten()

# Menampilkan gambar asli pada subplot pertama
axes_flat[0].imshow(gray, cmap='gray')
axes_flat[0].set_title("Gambar Asli", fontsize=10, fontweight='bold')
axes_flat[0].axis('off')

# Menampilkan hasil setiap kernel konvolusi
for idx, (nama_kernel, hasil) in enumerate(hasil_konvolusi.items()):
    if idx >= 8:
        break
    # Menormalisasi hasil untuk visualisasi
    hasil_norm = cv2.normalize(hasil, None, 0, 255, cv2.NORM_MINMAX)
    hasil_norm = hasil_norm.astype(np.uint8)

    # Menampilkan feature map
    axes_flat[idx + 1].imshow(hasil_norm, cmap='gray')
    axes_flat[idx + 1].set_title(f"Kernel: {nama_kernel}", fontsize=9)
    axes_flat[idx + 1].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 4: Feature Maps dari Berbagai Kernel Konvolusi",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "04_feature_maps.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/04_feature_maps.png")

# ============================================================
# 4. Visualisasi feature maps multi-level (sequential filtering)
# ============================================================
print("\n--- 4. Feature Maps Multi-Level (Simulasi Layer CNN) ---")

# Mensimulasikan feature maps pada beberapa layer berturut-turut
# Layer 1: Edge detection
layer1 = cv2.filter2D(gray, cv2.CV_32F, kernels["Edge Detection"])

# Menormalisasi output layer 1 untuk input layer 2
layer1_norm = cv2.normalize(layer1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Layer 2: Gaussian blur pada output layer 1 (simulasi smoothing)
layer2 = cv2.filter2D(layer1_norm, cv2.CV_32F, kernels["Gaussian Blur"])

# Menormalisasi output layer 2
layer2_norm = cv2.normalize(layer2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Layer 3: Sharpen pada output layer 2
layer3 = cv2.filter2D(layer2_norm, cv2.CV_32F, kernels["Sharpen"])

# Menormalisasi output layer 3
layer3_norm = cv2.normalize(layer3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Layer 4: Emboss pada output layer 3
layer4 = cv2.filter2D(layer3_norm, cv2.CV_32F, kernels["Emboss"])

# Menormalisasi output layer 4
layer4_norm = cv2.normalize(layer4, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Menampilkan informasi setiap layer
print(f"  Layer 1 (Edge Detection) : shape={layer1_norm.shape}")
print(f"  Layer 2 (Gaussian Blur)  : shape={layer2_norm.shape}")
print(f"  Layer 3 (Sharpen)        : shape={layer3_norm.shape}")
print(f"  Layer 4 (Emboss)         : shape={layer4_norm.shape}")

# ============================================================
# 5. Operasi pooling manual (Max Pooling dan Average Pooling)
# ============================================================
print("\n--- 5. Operasi Pooling Manual ---")

def max_pooling(gambar, pool_size=2, stride=2):
    """
    Implementasi max pooling secara manual.
    Max pooling mengambil nilai maksimum dari setiap region.
    """
    # Mendapatkan dimensi gambar
    h, w = gambar.shape[:2]

    # Menghitung dimensi output setelah pooling
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1

    # Membuat array kosong untuk hasil pooling
    output = np.zeros((h_out, w_out), dtype=gambar.dtype)

    # Melakukan max pooling dengan nested loop
    for i in range(h_out):
        for j in range(w_out):
            # Menentukan region yang akan di-pool
            region = gambar[
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size
            ]
            # Mengambil nilai maksimum dari region
            output[i, j] = np.max(region)

    # Mengembalikan hasil max pooling
    return output

def avg_pooling(gambar, pool_size=2, stride=2):
    """
    Implementasi average pooling secara manual.
    Average pooling mengambil nilai rata-rata dari setiap region.
    """
    # Mendapatkan dimensi gambar
    h, w = gambar.shape[:2]

    # Menghitung dimensi output setelah pooling
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1

    # Membuat array kosong untuk hasil pooling
    output = np.zeros((h_out, w_out), dtype=np.float32)

    # Melakukan average pooling dengan nested loop
    for i in range(h_out):
        for j in range(w_out):
            # Menentukan region yang akan di-pool
            region = gambar[
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size
            ]
            # Menghitung nilai rata-rata dari region
            output[i, j] = np.mean(region)

    # Mengembalikan hasil average pooling
    return output.astype(np.uint8)

# Menerapkan max pooling dengan berbagai ukuran pool
max_pool_2 = max_pooling(gray, pool_size=2, stride=2)
max_pool_4 = max_pooling(gray, pool_size=4, stride=4)
max_pool_8 = max_pooling(gray, pool_size=8, stride=8)

# Menerapkan average pooling dengan berbagai ukuran pool
avg_pool_2 = avg_pooling(gray, pool_size=2, stride=2)
avg_pool_4 = avg_pooling(gray, pool_size=4, stride=4)
avg_pool_8 = avg_pooling(gray, pool_size=8, stride=8)

# Menampilkan informasi dimensi setelah pooling
print(f"  Gambar asli           : {gray.shape}")
print(f"  Max Pool 2x2 stride 2 : {max_pool_2.shape}")
print(f"  Max Pool 4x4 stride 4 : {max_pool_4.shape}")
print(f"  Max Pool 8x8 stride 8 : {max_pool_8.shape}")
print(f"  Avg Pool 2x2 stride 2 : {avg_pool_2.shape}")
print(f"  Avg Pool 4x4 stride 4 : {avg_pool_4.shape}")
print(f"  Avg Pool 8x8 stride 8 : {avg_pool_8.shape}")

# ============================================================
# 6. Visualisasi operasi pooling
# ============================================================
print("\n--- 6. Visualisasi Operasi Pooling ---")

# Membuat figure untuk visualisasi pooling
fig, axes = plt.subplots(3, 3, figsize=(14, 14))

# Baris 1: Gambar asli dan multi-level feature maps
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Gambar Asli (256x256)", fontsize=10, fontweight='bold')
axes[0, 0].axis('off')

# Menampilkan layer 1 feature map
axes[0, 1].imshow(layer1_norm, cmap='gray')
axes[0, 1].set_title("Layer 1: Edge Detection", fontsize=10)
axes[0, 1].axis('off')

# Menampilkan layer 2 feature map
axes[0, 2].imshow(layer2_norm, cmap='gray')
axes[0, 2].set_title("Layer 2: Blur + Edge", fontsize=10)
axes[0, 2].axis('off')

# Baris 2: Max Pooling dengan berbagai ukuran
axes[1, 0].imshow(max_pool_2, cmap='gray')
axes[1, 0].set_title(f"Max Pool 2x2 ({max_pool_2.shape[0]}x{max_pool_2.shape[1]})", fontsize=10)
axes[1, 0].axis('off')

axes[1, 1].imshow(max_pool_4, cmap='gray')
axes[1, 1].set_title(f"Max Pool 4x4 ({max_pool_4.shape[0]}x{max_pool_4.shape[1]})", fontsize=10)
axes[1, 1].axis('off')

axes[1, 2].imshow(max_pool_8, cmap='gray')
axes[1, 2].set_title(f"Max Pool 8x8 ({max_pool_8.shape[0]}x{max_pool_8.shape[1]})", fontsize=10)
axes[1, 2].axis('off')

# Baris 3: Average Pooling dengan berbagai ukuran
axes[2, 0].imshow(avg_pool_2, cmap='gray')
axes[2, 0].set_title(f"Avg Pool 2x2 ({avg_pool_2.shape[0]}x{avg_pool_2.shape[1]})", fontsize=10)
axes[2, 0].axis('off')

axes[2, 1].imshow(avg_pool_4, cmap='gray')
axes[2, 1].set_title(f"Avg Pool 4x4 ({avg_pool_4.shape[0]}x{avg_pool_4.shape[1]})", fontsize=10)
axes[2, 1].axis('off')

axes[2, 2].imshow(avg_pool_8, cmap='gray')
axes[2, 2].set_title(f"Avg Pool 8x8 ({avg_pool_8.shape[0]}x{avg_pool_8.shape[1]})", fontsize=10)
axes[2, 2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 4: Operasi Pooling pada Gambar",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "04_pooling.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/04_pooling.png")

# ============================================================
# 7. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 4")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. Arsitektur CNN terdiri dari Conv, Pooling, dan Fully Connected layers
2. Konvolusi 2D menerapkan kernel/filter pada gambar untuk ekstraksi fitur
3. Berbagai kernel: edge detection, blur, sharpen, emboss, sobel
4. cv2.filter2D() melakukan konvolusi 2D secara efisien
5. Feature maps menunjukkan fitur yang diekstrak pada setiap level layer
6. Max pooling mengambil nilai maksimum dari region (mendeteksi fitur kuat)
7. Average pooling mengambil rata-rata dari region (smoothing)
8. Pooling mengurangi dimensi gambar (downsampling)

Output disimpan di folder: output/
- 04_arsitektur_cnn.png : Diagram arsitektur CNN
- 04_feature_maps.png   : Feature maps dari berbagai kernel
- 04_pooling.png        : Visualisasi operasi pooling
""")
