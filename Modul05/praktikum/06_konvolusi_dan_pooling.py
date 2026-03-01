"""
==========================================================================
PERCOBAAN 6: KONVOLUSI DAN POOLING
==========================================================================
Program ini mempelajari operasi konvolusi 2D dan pooling secara mendalam.
Konvolusi diimplementasikan dari nol (manual nested loop) dan dibandingkan
dengan hasil cv2.filter2D(). Pooling juga diimplementasikan dari nol.

Fungsi utama yang dipelajari:
- cv2.filter2D()       : Konvolusi 2D built-in OpenCV
- np.pad()             : Menambahkan padding pada array
- Manual konvolusi     : Implementasi nested loop dari nol
- Manual max pooling   : Implementasi dari nol
- Manual avg pooling   : Implementasi dari nol

Konsep yang dipelajari:
- Operasi konvolusi 2D: kernel × input = feature map
- Pengaruh berbagai kernel: identity, edge, sharpen, gaussian, sobel
- Padding: zero padding, same padding, valid (no padding)
- Stride: langkah pergeseran kernel
- Max pooling vs Average pooling
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

# Mengimpor time untuk benchmark perbandingan kecepatan
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
print("PERCOBAAN 6: KONVOLUSI DAN POOLING")
print("=" * 60)

# ============================================================
# 1. Implementasi konvolusi 2D dari nol
# ============================================================
print("\n--- 1. Implementasi Konvolusi 2D Manual ---")

def konvolusi_manual(gambar, kernel, padding='same'):
    """
    Implementasi konvolusi 2D dari nol menggunakan nested loop.
    Mendemonstrasikan bagaimana operasi konvolusi bekerja secara internal.

    Parameter:
    - gambar: array 2D (grayscale)
    - kernel: array 2D (filter)
    - padding: 'same' (output ukuran sama) atau 'valid' (tanpa padding)
    """
    # Mendapatkan dimensi gambar dan kernel
    h_img, w_img = gambar.shape
    h_ker, w_ker = kernel.shape

    # Menghitung ukuran padding yang dibutuhkan
    pad_h = h_ker // 2
    pad_w = w_ker // 2

    # Menambahkan zero padding pada gambar jika mode 'same'
    if padding == 'same':
        # Menambahkan padding nol di sekeliling gambar
        gambar_pad = np.pad(gambar, ((pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant', constant_values=0)
        # Menghitung dimensi output (sama dengan input)
        h_out = h_img
        w_out = w_img
    else:
        # Tanpa padding, output lebih kecil
        gambar_pad = gambar.copy()
        h_out = h_img - h_ker + 1
        w_out = w_img - w_ker + 1

    # Membuat array kosong untuk hasil konvolusi
    output = np.zeros((h_out, w_out), dtype=np.float32)

    # Melakukan konvolusi dengan nested loop
    for i in range(h_out):
        for j in range(w_out):
            # Mengekstrak region gambar yang sesuai ukuran kernel
            region = gambar_pad[i:i + h_ker, j:j + w_ker]

            # Mengalikan region dengan kernel element-wise lalu menjumlahkan
            output[i, j] = np.sum(region * kernel)

    # Mengembalikan hasil konvolusi
    return output

# Memuat gambar untuk percobaan konvolusi
img = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"), cv2.IMREAD_GRAYSCALE)

# Memeriksa apakah gambar berhasil dimuat
if img is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Meresize gambar ke ukuran kecil untuk konvolusi manual (agar tidak terlalu lama)
img_kecil = cv2.resize(img, (64, 64))

# Meresize gambar ke ukuran sedang untuk visualisasi filter2D
img_sedang = cv2.resize(img, (256, 256))

# Menampilkan informasi gambar
print(f"  Gambar kecil (manual)  : {img_kecil.shape}")
print(f"  Gambar sedang (filter2D): {img_sedang.shape}")

# ============================================================
# 2. Mendefinisikan berbagai kernel
# ============================================================
print("\n--- 2. Berbagai Kernel Konvolusi ---")

# Mendefinisikan dictionary kernel konvolusi
kernels = {
    "Identity": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float32),

    "Edge Detect": np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32),

    "Sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32),

    "Gaussian 3x3": np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0,

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

    "Box Blur": np.ones((3, 3), dtype=np.float32) / 9.0,

    "Laplacian": np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32),
}

# Menampilkan informasi setiap kernel
for nama, kernel in kernels.items():
    # Menampilkan nama kernel dan jumlah elemennya
    print(f"  {nama:15s}: ukuran={kernel.shape}, sum={kernel.sum():.2f}")

# ============================================================
# 3. Perbandingan konvolusi manual vs cv2.filter2D
# ============================================================
print("\n--- 3. Perbandingan Konvolusi Manual vs cv2.filter2D ---")

# Menggunakan gambar kecil untuk konvolusi manual
gambar_float = img_kecil.astype(np.float32)

# Memilih kernel edge detection untuk perbandingan
kernel_test = kernels["Edge Detect"]

# Mengukur waktu konvolusi manual
start_manual = time.time()
hasil_manual = konvolusi_manual(gambar_float, kernel_test, padding='same')
waktu_manual = (time.time() - start_manual) * 1000

# Mengukur waktu cv2.filter2D
start_cv2 = time.time()
hasil_cv2 = cv2.filter2D(gambar_float, cv2.CV_32F, kernel_test)
waktu_cv2 = (time.time() - start_cv2) * 1000

# Menghitung perbedaan antara kedua hasil
perbedaan = np.abs(hasil_manual - hasil_cv2)
max_diff = perbedaan.max()
mean_diff = perbedaan.mean()

# Menampilkan hasil perbandingan
print(f"  Konvolusi Manual : {waktu_manual:.2f} ms")
print(f"  cv2.filter2D     : {waktu_cv2:.2f} ms")
print(f"  Speedup          : {waktu_manual / max(waktu_cv2, 0.001):.1f}x lebih cepat")
print(f"  Max perbedaan    : {max_diff:.6f}")
print(f"  Mean perbedaan   : {mean_diff:.6f}")
print(f"  Hasil identik    : {'Ya' if max_diff < 0.01 else 'Hampir identik'}")

# ============================================================
# 4. Visualisasi perbandingan konvolusi manual vs filter2D
# ============================================================
print("\n--- 4. Visualisasi Perbandingan Konvolusi ---")

# Membuat figure untuk perbandingan
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Baris 1: Hasil konvolusi manual
axes[0, 0].imshow(img_kecil, cmap='gray')
axes[0, 0].set_title("Asli (64x64)", fontweight='bold')
axes[0, 0].axis('off')

# Menerapkan 3 kernel dengan konvolusi manual
kernels_demo = ["Edge Detect", "Sharpen", "Gaussian 3x3"]
for idx, nama in enumerate(kernels_demo):
    # Melakukan konvolusi manual
    hasil = konvolusi_manual(gambar_float, kernels[nama], padding='same')
    # Menormalisasi untuk visualisasi
    hasil_vis = cv2.normalize(hasil, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Menampilkan hasil pada subplot
    axes[0, idx + 1].imshow(hasil_vis, cmap='gray')
    axes[0, idx + 1].set_title(f"Manual: {nama}", fontsize=9)
    axes[0, idx + 1].axis('off')

# Baris 2: Hasil konvolusi cv2.filter2D pada gambar sedang
gambar_sedang_float = img_sedang.astype(np.float32)

# Menampilkan gambar asli
axes[1, 0].imshow(img_sedang, cmap='gray')
axes[1, 0].set_title("Asli (256x256)", fontweight='bold')
axes[1, 0].axis('off')

# Menerapkan 3 kernel dengan cv2.filter2D
for idx, nama in enumerate(kernels_demo):
    # Melakukan konvolusi dengan cv2.filter2D
    hasil = cv2.filter2D(gambar_sedang_float, cv2.CV_32F, kernels[nama])
    # Menormalisasi untuk visualisasi
    hasil_vis = cv2.normalize(hasil, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Menampilkan hasil pada subplot
    axes[1, idx + 1].imshow(hasil_vis, cmap='gray')
    axes[1, idx + 1].set_title(f"filter2D: {nama}", fontsize=9)
    axes[1, idx + 1].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 6: Konvolusi Manual vs cv2.filter2D()",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "06_konvolusi_manual.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/06_konvolusi_manual.png")

# ============================================================
# 5. Visualisasi efek semua kernel
# ============================================================
print("\n--- 5. Visualisasi Efek Berbagai Kernel ---")

# Membuat figure untuk semua kernel
fig, axes = plt.subplots(3, 3, figsize=(14, 14))

# Meratakan axes untuk iterasi
axes_flat = axes.flatten()

# Menampilkan gambar asli di subplot pertama
axes_flat[0].imshow(img_sedang, cmap='gray')
axes_flat[0].set_title("Gambar Asli", fontsize=10, fontweight='bold')
axes_flat[0].axis('off')

# Menerapkan setiap kernel dan menampilkan hasilnya
for idx, (nama, kernel) in enumerate(kernels.items()):
    if idx >= 8:
        break
    # Menerapkan konvolusi dengan cv2.filter2D
    hasil = cv2.filter2D(gambar_sedang_float, cv2.CV_32F, kernel)

    # Menormalisasi hasil untuk visualisasi
    hasil_vis = cv2.normalize(hasil, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Menampilkan hasil pada subplot
    axes_flat[idx + 1].imshow(hasil_vis, cmap='gray')
    axes_flat[idx + 1].set_title(f"{nama}", fontsize=9)
    axes_flat[idx + 1].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 6: Efek Berbagai Kernel Konvolusi pada Gambar",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "06_berbagai_kernel.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/06_berbagai_kernel.png")

# ============================================================
# 6. Implementasi pooling dari nol
# ============================================================
print("\n--- 6. Implementasi Pooling dari Nol ---")

def max_pooling_manual(gambar, pool_size=2, stride=2):
    """
    Implementasi max pooling secara manual.
    Mengambil nilai maksimum dari setiap region berukuran pool_size x pool_size.
    """
    # Mendapatkan dimensi gambar
    h, w = gambar.shape[:2]

    # Menghitung dimensi output
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1

    # Membuat array kosong untuk output
    output = np.zeros((h_out, w_out), dtype=gambar.dtype)

    # Melakukan max pooling dengan nested loop
    for i in range(h_out):
        for j in range(w_out):
            # Menentukan posisi awal region
            row_start = i * stride
            col_start = j * stride

            # Mengekstrak region dari gambar
            region = gambar[row_start:row_start + pool_size,
                           col_start:col_start + pool_size]

            # Mengambil nilai maksimum dari region
            output[i, j] = np.max(region)

    # Mengembalikan hasil max pooling
    return output

def avg_pooling_manual(gambar, pool_size=2, stride=2):
    """
    Implementasi average pooling secara manual.
    Mengambil nilai rata-rata dari setiap region berukuran pool_size x pool_size.
    """
    # Mendapatkan dimensi gambar
    h, w = gambar.shape[:2]

    # Menghitung dimensi output
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1

    # Membuat array kosong untuk output
    output = np.zeros((h_out, w_out), dtype=np.float32)

    # Melakukan average pooling dengan nested loop
    for i in range(h_out):
        for j in range(w_out):
            # Menentukan posisi awal region
            row_start = i * stride
            col_start = j * stride

            # Mengekstrak region dari gambar
            region = gambar[row_start:row_start + pool_size,
                           col_start:col_start + pool_size].astype(np.float32)

            # Menghitung nilai rata-rata dari region
            output[i, j] = np.mean(region)

    # Mengembalikan hasil average pooling
    return output.astype(np.uint8)

# Menerapkan pooling pada gambar sedang
print(f"  Gambar input: {img_sedang.shape}")

# Max pooling dengan berbagai konfigurasi
max_2x2_s2 = max_pooling_manual(img_sedang, pool_size=2, stride=2)
max_4x4_s4 = max_pooling_manual(img_sedang, pool_size=4, stride=4)
max_2x2_s1 = max_pooling_manual(img_sedang, pool_size=2, stride=1)

# Average pooling dengan berbagai konfigurasi
avg_2x2_s2 = avg_pooling_manual(img_sedang, pool_size=2, stride=2)
avg_4x4_s4 = avg_pooling_manual(img_sedang, pool_size=4, stride=4)
avg_2x2_s1 = avg_pooling_manual(img_sedang, pool_size=2, stride=1)

# Menampilkan dimensi output untuk setiap konfigurasi
print(f"\n  Max Pool 2x2 stride 2: {img_sedang.shape} -> {max_2x2_s2.shape}")
print(f"  Max Pool 4x4 stride 4: {img_sedang.shape} -> {max_4x4_s4.shape}")
print(f"  Max Pool 2x2 stride 1: {img_sedang.shape} -> {max_2x2_s1.shape}")
print(f"  Avg Pool 2x2 stride 2: {img_sedang.shape} -> {avg_2x2_s2.shape}")
print(f"  Avg Pool 4x4 stride 4: {img_sedang.shape} -> {avg_4x4_s4.shape}")
print(f"  Avg Pool 2x2 stride 1: {img_sedang.shape} -> {avg_2x2_s1.shape}")

# ============================================================
# 7. Visualisasi operasi pooling
# ============================================================
print("\n--- 7. Visualisasi Operasi Pooling ---")

# Membuat figure untuk visualisasi pooling
fig, axes = plt.subplots(3, 3, figsize=(14, 14))

# Baris 1: Gambar asli dan pengaruh stride
axes[0, 0].imshow(img_sedang, cmap='gray')
axes[0, 0].set_title(f"Asli ({img_sedang.shape[0]}x{img_sedang.shape[1]})",
                     fontsize=10, fontweight='bold')
axes[0, 0].axis('off')

# Menampilkan max pool dengan stride berbeda
axes[0, 1].imshow(max_2x2_s2, cmap='gray')
axes[0, 1].set_title(f"Max Pool 2x2, stride=2\n({max_2x2_s2.shape[0]}x{max_2x2_s2.shape[1]})",
                     fontsize=9)
axes[0, 1].axis('off')

axes[0, 2].imshow(max_2x2_s1, cmap='gray')
axes[0, 2].set_title(f"Max Pool 2x2, stride=1\n({max_2x2_s1.shape[0]}x{max_2x2_s1.shape[1]})",
                     fontsize=9)
axes[0, 2].axis('off')

# Baris 2: Max pooling dengan berbagai pool size
axes[1, 0].imshow(max_2x2_s2, cmap='gray')
axes[1, 0].set_title(f"Max Pool 2x2\n({max_2x2_s2.shape[0]}x{max_2x2_s2.shape[1]})", fontsize=9)
axes[1, 0].axis('off')

axes[1, 1].imshow(max_4x4_s4, cmap='gray')
axes[1, 1].set_title(f"Max Pool 4x4\n({max_4x4_s4.shape[0]}x{max_4x4_s4.shape[1]})", fontsize=9)
axes[1, 1].axis('off')

# Menampilkan perbedaan max vs avg
diff_max_avg = cv2.absdiff(max_2x2_s2, avg_2x2_s2)
axes[1, 2].imshow(diff_max_avg, cmap='hot')
axes[1, 2].set_title("Perbedaan Max vs Avg\n(Heatmap)", fontsize=9)
axes[1, 2].axis('off')

# Baris 3: Average pooling
axes[2, 0].imshow(avg_2x2_s2, cmap='gray')
axes[2, 0].set_title(f"Avg Pool 2x2\n({avg_2x2_s2.shape[0]}x{avg_2x2_s2.shape[1]})", fontsize=9)
axes[2, 0].axis('off')

axes[2, 1].imshow(avg_4x4_s4, cmap='gray')
axes[2, 1].set_title(f"Avg Pool 4x4\n({avg_4x4_s4.shape[0]}x{avg_4x4_s4.shape[1]})", fontsize=9)
axes[2, 1].axis('off')

axes[2, 2].imshow(avg_2x2_s1, cmap='gray')
axes[2, 2].set_title(f"Avg Pool 2x2, stride=1\n({avg_2x2_s1.shape[0]}x{avg_2x2_s1.shape[1]})",
                     fontsize=9)
axes[2, 2].axis('off')

# Menambahkan judul utama
plt.suptitle("Percobaan 6: Operasi Pooling Manual (Max Pool vs Average Pool)",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file output
plt.savefig(os.path.join(OUTPUT_DIR, "06_pooling_manual.png"),
            dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan penyimpanan
print("  [SAVED] output/06_pooling_manual.png")

# ============================================================
# 8. Demonstrasi efek padding pada konvolusi
# ============================================================
print("\n--- 8. Efek Padding pada Konvolusi ---")

# Konvolusi dengan same padding (output ukuran sama)
hasil_same = konvolusi_manual(gambar_float, kernels["Edge Detect"], padding='same')
print(f"  Same padding  : input={gambar_float.shape} -> output={hasil_same.shape}")

# Konvolusi dengan valid padding (tanpa padding, output lebih kecil)
hasil_valid = konvolusi_manual(gambar_float, kernels["Edge Detect"], padding='valid')
print(f"  Valid padding : input={gambar_float.shape} -> output={hasil_valid.shape}")

# Menghitung perbedaan dimensi
diff_h = gambar_float.shape[0] - hasil_valid.shape[0]
diff_w = gambar_float.shape[1] - hasil_valid.shape[1]
print(f"  Pengurangan dimensi (valid): {diff_h} piksel tinggi, {diff_w} piksel lebar")

# ============================================================
# 9. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 6")
print("=" * 60)
print("""
Konsep yang telah dipelajari:
1. Konvolusi 2D: kernel bergeser di atas gambar, hitung dot product tiap region
2. Implementasi manual menggunakan nested loop (lambat tapi edukatif)
3. cv2.filter2D() jauh lebih cepat karena dioptimasi internal
4. Berbagai kernel: identity, edge, sharpen, gaussian, sobel, laplacian
5. Padding 'same' mempertahankan ukuran output, 'valid' mengurangi
6. Max pooling menyimpan fitur terkuat (baik untuk deteksi)
7. Average pooling menyimpan rata-rata fitur (baik untuk smoothing)
8. Stride menentukan langkah pergeseran dan ukuran output

Output disimpan di folder: output/
- 06_konvolusi_manual.png : Perbandingan konvolusi manual vs filter2D
- 06_berbagai_kernel.png  : Efek berbagai kernel pada gambar
- 06_pooling_manual.png   : Visualisasi max pool dan average pool
""")
