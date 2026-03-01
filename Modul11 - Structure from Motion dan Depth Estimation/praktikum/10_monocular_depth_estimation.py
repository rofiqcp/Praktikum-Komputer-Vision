"""
==========================================================================
PERCOBAAN 10: MONOCULAR DEPTH ESTIMATION
==========================================================================
Program ini mempelajari cara mengestimasi depth (kedalaman) dari
SATU gambar saja menggunakan pendekatan berbasis depth cues visual
seperti gradien, blur, dan tekstur. Juga mencoba model deep learning
MiDaS jika tersedia, dengan fallback ke metode manual.

Konsep utama:
- Monocular depth estimation: estimasi kedalaman dari gambar tunggal
- Depth cues: gradien (edge), defocus blur, tekstur, posisi vertikal
- Perspective cue: objek lebih jauh cenderung ada di bagian atas gambar
- MiDaS: model deep learning dari Intel untuk monocular depth estimation
- Fusion dari beberapa depth cues menghasilkan peta kedalaman kasar

Fungsi utama yang dipelajari:
- cv2.Sobel()           : Menghitung gradien untuk edge-based depth cue
- cv2.Laplacian()       : Mendeteksi area fokus vs blur
- cv2.GaussianBlur()    : Menghaluskan gambar dan estimasi depth
- cv2.dnn.readNet()     : Memuat model MiDaS (jika tersedia)
- cv2.normalize()       : Normalisasi depth map untuk visualisasi
- cv2.applyColorMap()   : Menerapkan colormap pada depth map

Hasil: Depth map estimasi dari gambar tunggal menggunakan berbagai metode
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk membuat dan menyimpan visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan judul percobaan
print("=" * 60)
print("PERCOBAAN 10: MONOCULAR DEPTH ESTIMATION")
print("=" * 60)

# ============================================================
# 1. Memuat atau membuat gambar input
# ============================================================

# Menampilkan informasi tahap memuat gambar
print("\n--- Memuat Gambar Input ---")

# Mendefinisikan path gambar input
path_gambar = os.path.join(IMAGE_DIR, "gambar_depth.png")

# Membaca gambar input dalam format BGR
gambar = cv2.imread(path_gambar)

# Memeriksa apakah gambar berhasil dimuat, jika tidak download otomatis
if gambar is None:
    print("[WARN] gambar_depth.png tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
    gambar = cv2.imread(path_gambar)
if gambar is None:
    raise FileNotFoundError(
        "[ERROR] gambar_depth.png tidak tersedia setelah download.\n"
        "  Jalankan terlebih dahulu: python download_image.py"
    )
else:
    # Menampilkan informasi dimensi gambar
    print(f"[INFO] Ukuran gambar: {gambar.shape}")

# Mendapatkan dimensi gambar
h, w = gambar.shape[:2]

# Mengkonversi gambar ke grayscale
gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Depth Cue 1: Gradien (Edge-based)
# ============================================================

# Menampilkan informasi depth cue gradien
print("\n--- Depth Cue 1: Gradien (Sobel & Laplacian) ---")

# Menghitung gradien horizontal menggunakan Sobel
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Menghitung gradien vertikal menggunakan Sobel
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Menghitung magnitude gradien gabungan
gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# Menormalisasi magnitude gradien ke range 0-1
gradient_norm = gradient_mag / (gradient_mag.max() + 1e-8)

# Menghitung Laplacian untuk mendeteksi area fokus vs blur
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

# Menghitung magnitude absolut Laplacian
laplacian_abs = np.abs(laplacian)

# Menormalisasi Laplacian ke range 0-1
laplacian_norm = laplacian_abs / (laplacian_abs.max() + 1e-8)

# Area dengan gradien tinggi cenderung dekat (edge lebih tajam di dekat)
# Invert sehingga skor tinggi = dekat
depth_gradient = 1.0 - cv2.GaussianBlur(gradient_norm.astype(np.float32), (21, 21), 0)

# Menampilkan statistik
print(f"[INFO] Gradient magnitude range: [{gradient_mag.min():.2f}, {gradient_mag.max():.2f}]")
print(f"[INFO] Laplacian abs range: [{laplacian_abs.min():.2f}, {laplacian_abs.max():.2f}]")

# ============================================================
# 3. Depth Cue 2: Defocus Blur
# ============================================================

# Menampilkan informasi depth cue blur
print("\n--- Depth Cue 2: Defocus Blur ---")

# Menghitung variasi lokal menggunakan GaussianBlur pada gambar dan kuadratnya
# Tingkat blur menunjukkan kedalaman relatif

# Mengkonversi ke float untuk perhitungan
gray_float = gray.astype(np.float64)

# Menghitung rata-rata lokal
mean_local = cv2.GaussianBlur(gray_float, (31, 31), 0)

# Menghitung rata-rata kuadrat lokal
mean_sq_local = cv2.GaussianBlur(gray_float**2, (31, 31), 0)

# Menghitung varians lokal (ukuran ketajaman/blur)
variance_local = mean_sq_local - mean_local**2

# Mengganti nilai negatif kecil akibat numerik
variance_local = np.maximum(variance_local, 0)

# Menormalisasi varians lokal
variance_norm = variance_local / (variance_local.max() + 1e-8)

# Area dengan varians tinggi = tajam = dekat, blur = jauh
depth_blur = 1.0 - cv2.GaussianBlur(variance_norm.astype(np.float32), (21, 21), 0)

# Menampilkan statistik
print(f"[INFO] Varians lokal range: [{variance_local.min():.2f}, {variance_local.max():.2f}]")

# ============================================================
# 4. Depth Cue 3: Posisi Vertikal (Perspective)
# ============================================================

# Menampilkan informasi depth cue perspektif
print("\n--- Depth Cue 3: Posisi Vertikal (Perspective) ---")

# Membuat depth map berdasarkan posisi vertikal
# Asumsi: bagian bawah gambar lebih dekat, atas lebih jauh
depth_vertical = np.zeros((h, w), dtype=np.float32)

# Mengisi depth berdasarkan posisi y (baris)
for y in range(h):
    # Nilai depth berdasarkan posisi vertikal (0=atas/jauh, 1=bawah/dekat)
    depth_vertical[y, :] = y / (h - 1)

# Menerapkan Gaussian blur untuk menghaluskan transisi
depth_vertical = cv2.GaussianBlur(depth_vertical, (31, 31), 0)

# Menampilkan statistik
print(f"[INFO] Depth vertikal range: [{depth_vertical.min():.3f}, {depth_vertical.max():.3f}]")

# ============================================================
# 5. Depth Cue 4: Tekstur Density
# ============================================================

# Menampilkan informasi depth cue tekstur
print("\n--- Depth Cue 4: Tekstur Density ---")

# Menghitung kepadatan tekstur menggunakan perbedaan blur level
# Objek jauh memiliki tekstur lebih padat (detail kecil)

# Menghitung blur dengan kernel kecil
blur_small = cv2.GaussianBlur(gray_float, (5, 5), 0)

# Menghitung blur dengan kernel besar
blur_large = cv2.GaussianBlur(gray_float, (31, 31), 0)

# Menghitung perbedaan (area dengan banyak detail tinggi = dekat)
texture_diff = np.abs(blur_small - blur_large)

# Menormalisasi
texture_norm = texture_diff / (texture_diff.max() + 1e-8)

# Menghaluskan depth cue tekstur
depth_texture = 1.0 - cv2.GaussianBlur(texture_norm.astype(np.float32), (31, 31), 0)

# Menampilkan statistik
print(f"[INFO] Tekstur diff range: [{texture_diff.min():.2f}, {texture_diff.max():.2f}]")

# ============================================================
# 6. Fusi Depth Cues (Manual Depth Estimation)
# ============================================================

# Menampilkan informasi tahap fusi
print("\n--- Fusi Depth Cues Manual ---")

# Mendefinisikan bobot untuk setiap depth cue
w_gradient = 0.2
w_blur = 0.2
w_vertical = 0.4
w_texture = 0.2

# Menampilkan bobot yang digunakan
print(f"[INFO] Bobot: gradient={w_gradient}, blur={w_blur}, "
      f"vertical={w_vertical}, texture={w_texture}")

# Menggabungkan semua depth cue dengan bobot
depth_fused = (w_gradient * depth_gradient +
               w_blur * depth_blur +
               w_vertical * depth_vertical +
               w_texture * depth_texture)

# Menormalisasi depth fusi ke range 0-1
depth_fused = (depth_fused - depth_fused.min()) / (depth_fused.max() - depth_fused.min() + 1e-8)

# Mengkonversi ke uint8 untuk visualisasi
depth_fused_uint8 = (depth_fused * 255).astype(np.uint8)

# Menerapkan colormap PLASMA pada depth fusi
depth_fused_color = cv2.applyColorMap(depth_fused_uint8, cv2.COLORMAP_PLASMA)

# Menyimpan depth map fusi
path_fused = os.path.join(OUTPUT_DIR, "10_depth_fusi_manual.png")
cv2.imwrite(path_fused, depth_fused_color)
print(f"[SAVED] Depth fusi manual: {path_fused}")

# ============================================================
# 7. Mencoba Model MiDaS (Deep Learning)
# ============================================================

# Menampilkan informasi tahap MiDaS
print("\n--- Mencoba Model MiDaS (Deep Learning) ---")

# Mendefinisikan path model MiDaS
model_path = os.path.join(IMAGE_DIR, "model-small.onnx")

# Mencoba memuat model MiDaS
midas_available = False
try:
    # Memeriksa apakah file model ada
    if os.path.exists(model_path):
        # Memuat model MiDaS menggunakan cv2.dnn
        net = cv2.dnn.readNet(model_path)

        # Menyiapkan input blob dari gambar
        blob = cv2.dnn.blobFromImage(gambar, 1.0 / 255.0, (256, 256),
                                     (0.485, 0.456, 0.406), swapRB=True, crop=False)

        # Melakukan normalisasi standar deviasi
        blob[0, 0] = (blob[0, 0] - 0.485) / 0.229
        blob[0, 1] = (blob[0, 1] - 0.456) / 0.224
        blob[0, 2] = (blob[0, 2] - 0.406) / 0.225

        # Mengatur input untuk network
        net.setInput(blob)

        # Melakukan forward pass
        depth_midas = net.forward()

        # Mengubah ukuran output ke ukuran gambar asli
        depth_midas = cv2.resize(depth_midas[0, 0], (w, h))

        # Menormalisasi depth MiDaS
        depth_midas = cv2.normalize(depth_midas, None, 0, 1, cv2.NORM_MINMAX)

        # Menandai bahwa MiDaS tersedia
        midas_available = True
        print("[INFO] Model MiDaS berhasil dimuat dan dijalankan")
    else:
        print(f"[INFO] Model MiDaS tidak ditemukan di: {model_path}")
        print("[INFO] Menggunakan fallback metode manual saja")
except Exception as e:
    print(f"[INFO] Gagal memuat MiDaS: {e}")
    print("[INFO] Menggunakan fallback metode manual saja")

# Jika MiDaS tidak tersedia, buat depth estimasi alternatif
if not midas_available:
    # Membuat depth estimasi alternatif dari gabungan cue yang sudah ada
    # dengan penambahan Canny edge untuk simulasi
    print("[INFO] Membuat depth estimasi alternatif (pseudo-DL)...")

    # Mendeteksi edge menggunakan Canny
    edges = cv2.Canny(gray, 50, 150)

    # Mendilasi edge agar lebih tebal
    edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    # Menghitung distance transform dari invert edge
    dist_transform = cv2.distanceTransform(255 - edges_dilated, cv2.DIST_L2, 5)

    # Menormalisasi distance transform
    dist_norm = dist_transform / (dist_transform.max() + 1e-8)

    # Menggabungkan dengan depth vertikal dan fusi
    depth_midas = 0.5 * depth_fused + 0.3 * depth_vertical + 0.2 * dist_norm
    depth_midas = (depth_midas - depth_midas.min()) / (depth_midas.max() - depth_midas.min() + 1e-8)

# ============================================================
# 8. Overlay depth pada gambar original
# ============================================================

# Menampilkan informasi tahap overlay
print("\n--- Overlay Depth pada Gambar Original ---")

# Mengkonversi depth ke uint8
depth_overlay_uint8 = (depth_midas * 255).astype(np.uint8)

# Menerapkan colormap JET pada depth
depth_overlay_color = cv2.applyColorMap(depth_overlay_uint8, cv2.COLORMAP_JET)

# Membuat overlay dengan blending gambar original dan depth
alpha_blend = 0.5
overlay = cv2.addWeighted(gambar, alpha_blend, depth_overlay_color, 1 - alpha_blend, 0)

# Menyimpan overlay
path_overlay = os.path.join(OUTPUT_DIR, "10_depth_overlay.png")
cv2.imwrite(path_overlay, overlay)
print(f"[SAVED] Depth overlay: {path_overlay}")

# ============================================================
# 9. Visualisasi semua depth cues
# ============================================================

# Menampilkan informasi tahap visualisasi
print("\n--- Membuat Visualisasi Depth Cues ---")

# Membuat figure 2x3 untuk semua depth cues
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))

# Menampilkan gambar original
axes1[0, 0].imshow(cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB))
axes1[0, 0].set_title("Gambar Original", fontsize=12)
axes1[0, 0].axis("off")

# Menampilkan depth cue gradien
im_grad = axes1[0, 1].imshow(depth_gradient, cmap='plasma')
axes1[0, 1].set_title("Depth Cue: Gradien", fontsize=12)
axes1[0, 1].axis("off")
plt.colorbar(im_grad, ax=axes1[0, 1], fraction=0.046)

# Menampilkan depth cue blur
im_blur = axes1[0, 2].imshow(depth_blur, cmap='plasma')
axes1[0, 2].set_title("Depth Cue: Defocus Blur", fontsize=12)
axes1[0, 2].axis("off")
plt.colorbar(im_blur, ax=axes1[0, 2], fraction=0.046)

# Menampilkan depth cue posisi vertikal
im_vert = axes1[1, 0].imshow(depth_vertical, cmap='plasma')
axes1[1, 0].set_title("Depth Cue: Posisi Vertikal", fontsize=12)
axes1[1, 0].axis("off")
plt.colorbar(im_vert, ax=axes1[1, 0], fraction=0.046)

# Menampilkan depth cue tekstur
im_tex = axes1[1, 1].imshow(depth_texture, cmap='plasma')
axes1[1, 1].set_title("Depth Cue: Tekstur", fontsize=12)
axes1[1, 1].axis("off")
plt.colorbar(im_tex, ax=axes1[1, 1], fraction=0.046)

# Menampilkan depth fusi manual
im_fused = axes1[1, 2].imshow(depth_fused, cmap='plasma')
axes1[1, 2].set_title("Depth Fusi Manual", fontsize=12)
axes1[1, 2].axis("off")
plt.colorbar(im_fused, ax=axes1[1, 2], fraction=0.046)

# Mengatur judul utama
plt.suptitle("Percobaan 10: Depth Cues untuk Monocular Depth Estimation",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure depth cues
output_cues = os.path.join(OUTPUT_DIR, "10_depth_cues.png")
plt.savefig(output_cues, dpi=150, bbox_inches='tight')
print(f"[SAVED] Depth cues: {output_cues}")

# ============================================================
# 10. Visualisasi perbandingan metode
# ============================================================

# Menampilkan informasi tahap perbandingan
print("\n--- Membuat Visualisasi Perbandingan Metode ---")

# Membuat figure 2x2 untuk perbandingan
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Menampilkan gambar original
axes2[0, 0].imshow(cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title("Gambar Original", fontsize=12)
axes2[0, 0].axis("off")

# Menampilkan depth fusi manual
axes2[0, 1].imshow(depth_fused, cmap='plasma')
axes2[0, 1].set_title("Depth Fusi Manual", fontsize=12)
axes2[0, 1].axis("off")

# Menampilkan depth dari MiDaS atau alternatif
midas_label = "MiDaS Depth" if midas_available else "Pseudo-DL Depth (alternatif)"
axes2[1, 0].imshow(depth_midas, cmap='plasma')
axes2[1, 0].set_title(midas_label, fontsize=12)
axes2[1, 0].axis("off")

# Menampilkan overlay depth pada gambar
axes2[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes2[1, 1].set_title("Overlay Depth + Original", fontsize=12)
axes2[1, 1].axis("off")

# Mengatur judul utama
plt.suptitle("Percobaan 10: Monocular Depth Estimation - Perbandingan Metode",
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan figure perbandingan
output_compare = os.path.join(OUTPUT_DIR, "10_monocular_depth_estimation.png")
plt.savefig(output_compare, dpi=150, bbox_inches='tight')
print(f"[SAVED] Perbandingan metode: {output_compare}")

# Menampilkan semua figure
plt.show()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 10: MONOCULAR DEPTH ESTIMATION")
print("=" * 60)
print(f"1. Monocular depth estimation = estimasi kedalaman dari 1 gambar")
print(f"2. Depth cue gradien: area tajam (edge kuat) cenderung dekat")
print(f"3. Depth cue blur: area tidak fokus cenderung lebih jauh")
print(f"4. Depth cue vertikal: bagian bawah gambar cenderung dekat")
print(f"5. Depth cue tekstur: tekstur detail tinggi cenderung dekat")
print(f"6. Fusi bobot dari beberapa cue menghasilkan depth map kasar")
print(f"7. Model MiDaS (deep learning) memberikan hasil lebih akurat")
print(f"8. Overlay depth pada gambar membantu evaluasi visual")
print(f"9. Monocular depth bersifat relatif, bukan absolut")
print(f"10. Kombinasi cue manual berguna saat model DL tidak tersedia")
print("=" * 60)
