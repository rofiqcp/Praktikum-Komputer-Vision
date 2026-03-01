"""
==========================================================================
PERCOBAAN 1: HDR IMAGING PIPELINE
==========================================================================
Program ini mempelajari pipeline lengkap High Dynamic Range (HDR) imaging.
HDR menggabungkan beberapa foto dengan exposure berbeda menjadi satu gambar
yang memiliki rentang dinamis lebih luas dari kamera biasa.

Pipeline HDR:
1. Membaca gambar multi-exposure (exposure_1.png s/d exposure_5.png)
2. Menghitung Camera Response Function (CRF) dengan CalibrateDebevec
3. Menggabungkan menjadi HDR radiance map dengan MergeDebevec
4. Tone mapping agar bisa ditampilkan di layar biasa

Fungsi utama yang dipelajari:
- cv2.createCalibrateDebevec()  : Kalibrasi respon kamera (CRF)
- cv2.createMergeDebevec()      : Merge multi-exposure ke HDR
- cv2.createTonemap()           : Tone mapping global sederhana
- cv2.createTonemapReinhard()   : Tone mapping Reinhard

Hasil: Visualisasi setiap exposure dan hasil HDR setelah tone mapping
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan tipe data float
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi dan penyimpanan grafik
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output hasil percobaan
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 1: HDR IMAGING PIPELINE")
print("=" * 60)

# ============================================================
# 1. Membaca gambar multi-exposure
# ============================================================

# Mendefinisikan daftar nama file exposure (dari gelap ke terang)
exposure_files = [
    "exposure_1.png",  # Exposure paling gelap (1/30 detik)
    "exposure_2.png",  # Exposure gelap (1/15 detik)
    "exposure_3.png",  # Exposure normal (1/8 detik)
    "exposure_4.png",  # Exposure terang (1/4 detik)
    "exposure_5.png",  # Exposure paling terang (1/2 detik)
]

# Mendefinisikan waktu exposure dalam detik (sesuai metadata kamera)
# Semakin besar nilai = semakin lama sensor terbuka = semakin terang
exposure_times = np.array([1/30, 1/15, 1/8, 1/4, 1/2], dtype=np.float32)

# Membuat list kosong untuk menampung gambar-gambar yang dibaca
images = []

# Membaca setiap file gambar exposure satu per satu
for i, fname in enumerate(exposure_files):
    # Membentuk path lengkap ke file gambar
    img_path = os.path.join(IMAGE_DIR, fname)

    # Membaca gambar dalam format BGR (default OpenCV)
    img = cv2.imread(img_path)

    # Memeriksa apakah gambar berhasil dibaca
    if img is None:
        print(f"[ERROR] Gagal membaca: {img_path}")
        print("[INFO] Jalankan download_image.py terlebih dahulu!")
        exit()

    # Menambahkan gambar ke dalam list
    images.append(img)

    # Menampilkan info exposure time dan dimensi gambar
    print(f"[INFO] {fname} dimuat — Exposure: {exposure_times[i]:.4f}s, "
          f"Ukuran: {img.shape[1]}x{img.shape[0]}")

# Menampilkan jumlah total gambar yang berhasil dimuat
print(f"\n[INFO] Total {len(images)} gambar exposure berhasil dimuat")

# ============================================================
# 2. Menghitung Camera Response Function (CRF)
# cv2.createCalibrateDebevec() — metode Debevec & Malik (1997)
# CRF memetakan intensitas piksel → irradiance (energi cahaya)
# ============================================================

# Membuat objek kalibrasi Debevec dengan 256 sample
calibrate = cv2.createCalibrateDebevec(samples=70, random=False)

# Menghitung CRF dari kumpulan gambar dan waktu exposure
# Input: list gambar (uint8) dan array exposure times (float32)
# Output: curve 256x1x3 (untuk setiap channel B, G, R)
response_curve = calibrate.process(images, exposure_times)

print("[INFO] Camera Response Function (CRF) berhasil dihitung")
print(f"[INFO] Bentuk CRF: {response_curve.shape}")

# ============================================================
# 3. Menggabungkan gambar menjadi HDR radiance map
# cv2.createMergeDebevec() — merge menggunakan CRF
# Radiance map memiliki nilai float (bukan 0-255)
# ============================================================

# Membuat objek merge Debevec
merge_debevec = cv2.createMergeDebevec()

# Menggabungkan semua exposure menjadi satu HDR radiance map
# Input: list gambar, exposure times, dan CRF
hdr_image = merge_debevec.process(images, exposure_times, response_curve)

# Menampilkan statistik HDR map (rentang nilai sangat luas)
print(f"\n[INFO] HDR radiance map berhasil dibuat")
print(f"[INFO] Bentuk HDR: {hdr_image.shape}, Tipe: {hdr_image.dtype}")
print(f"[INFO] Rentang nilai: [{hdr_image.min():.4f}, {hdr_image.max():.4f}]")

# ============================================================
# 4. Tone Mapping — mengubah HDR ke LDR untuk tampilan
# Gambar HDR tidak bisa langsung ditampilkan di monitor biasa
# Tone mapping mengompresi rentang dinamis ke 0-255
# ============================================================

# --- 4a. Tone Mapping Global (linear sederhana) ---
# Membuat objek tonemap global dengan gamma 2.2
tonemap_global = cv2.createTonemap(gamma=2.2)

# Melakukan tone mapping (output: float 0-1)
ldr_global = tonemap_global.process(hdr_image)

# Clipping nilai agar berada di rentang 0-1
ldr_global = np.clip(ldr_global, 0, 1)

# Mengkonversi ke uint8 (0-255) untuk ditampilkan
ldr_global_8bit = (ldr_global * 255).astype(np.uint8)

print(f"\n[INFO] Tone mapping GLOBAL selesai (gamma=2.2)")

# --- 4b. Tone Mapping Reinhard ---
# Parameter: gamma, intensity, light_adapt, color_adapt
tonemap_reinhard = cv2.createTonemapReinhard(
    gamma=1.5,          # Koreksi gamma
    intensity=0.0,       # Intensitas global (-8 sampai 8)
    light_adapt=0.8,     # Adaptasi cahaya (0=global, 1=lokal)
    color_adapt=0.6      # Adaptasi warna (0=tidak ada, 1=penuh)
)

# Melakukan tone mapping Reinhard
ldr_reinhard = tonemap_reinhard.process(hdr_image)

# Clipping dan konversi ke uint8
ldr_reinhard = np.clip(ldr_reinhard, 0, 1)
ldr_reinhard_8bit = (ldr_reinhard * 255).astype(np.uint8)

print(f"[INFO] Tone mapping REINHARD selesai")

# ============================================================
# 5. Visualisasi semua hasil
# ============================================================

# Membuat figure besar dengan layout 2 baris x 4 kolom
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Label untuk setiap exposure (kelipatan dari exposure normal)
exposure_labels = ["0.3x (Gelap)", "0.6x", "1.0x (Normal)", "1.5x", "2.5x (Terang)"]

# Menampilkan 5 gambar exposure di baris atas
for i in range(5):
    # Konversi BGR (OpenCV) ke RGB (matplotlib)
    rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

    # Memilih posisi subplot (baris 0, kolom i)
    ax = axes[0, i] if i < 4 else axes[1, 0]

    # Menampilkan gambar exposure
    ax.imshow(rgb)

    # Memberikan judul dengan info exposure time
    ax.set_title(f"Exposure {i+1}\n{exposure_labels[i]}\n({exposure_times[i]:.4f}s)",
                 fontsize=9)

    # Menyembunyikan sumbu
    ax.axis("off")

# Menampilkan gambar exposure ke-5 di baris bawah kolom 0
if len(images) > 4:
    rgb5 = cv2.cvtColor(images[4], cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(rgb5)
    axes[1, 0].set_title(f"Exposure 5\n{exposure_labels[4]}\n({exposure_times[4]:.4f}s)",
                          fontsize=9)
    axes[1, 0].axis("off")

# Menampilkan CRF (Camera Response Function)
axes[1, 1].plot(response_curve[:, 0, 0], 'b-', label='Blue', linewidth=1.5)
axes[1, 1].plot(response_curve[:, 0, 1], 'g-', label='Green', linewidth=1.5)
axes[1, 1].plot(response_curve[:, 0, 2], 'r-', label='Red', linewidth=1.5)
axes[1, 1].set_title("Camera Response Function\n(CRF Debevec)", fontsize=9)
axes[1, 1].set_xlabel("Nilai Piksel (0-255)")
axes[1, 1].set_ylabel("Log Exposure")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Menampilkan hasil tone mapping global
ldr_global_rgb = cv2.cvtColor(ldr_global_8bit, cv2.COLOR_BGR2RGB)
axes[1, 2].imshow(ldr_global_rgb)
axes[1, 2].set_title("HDR → Tone Map\n(Global, gamma=2.2)", fontsize=9)
axes[1, 2].axis("off")

# Menampilkan hasil tone mapping Reinhard
ldr_reinhard_rgb = cv2.cvtColor(ldr_reinhard_8bit, cv2.COLOR_BGR2RGB)
axes[1, 3].imshow(ldr_reinhard_rgb)
axes[1, 3].set_title("HDR → Tone Map\n(Reinhard)", fontsize=9)
axes[1, 3].axis("off")

# Menyembunyikan subplot kosong di baris atas kolom 4-7 jika ada
# (baris atas hanya 5 exposure, kolom pertama diisi di baris bawah)

# Menambahkan judul utama figure
plt.suptitle("Percobaan 1: HDR Imaging Pipeline\n"
             "Multi-exposure → CRF → HDR Merge → Tone Mapping",
             fontsize=14, fontweight="bold")

# Mengatur layout agar tidak saling tumpang tindih
plt.tight_layout()

# Menyimpan hasil visualisasi ke file PNG
output_path = os.path.join(OUTPUT_DIR, "01_hdr_imaging_pipeline.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 1")
print("=" * 60)
print("Pipeline HDR Imaging:")
print("  1. Baca gambar multi-exposure (5 gambar, 1/30s - 1/2s)")
print("  2. Hitung CRF dengan cv2.createCalibrateDebevec()")
print("     - Memetakan intensitas piksel → energi cahaya")
print("  3. Merge ke HDR dengan cv2.createMergeDebevec()")
print("     - Gabungkan semua exposure berdasarkan CRF")
print(f"     - Rentang HDR: [{hdr_image.min():.2f}, {hdr_image.max():.2f}]")
print("  4. Tone mapping untuk tampilan layar:")
print("     - cv2.createTonemap(gamma)       → Global sederhana")
print("     - cv2.createTonemapReinhard(...)  → Reinhard operator")
print("  5. HDR menangkap detail di area gelap DAN terang")
print("=" * 60)
