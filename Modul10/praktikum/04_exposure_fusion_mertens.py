"""
==========================================================================
PERCOBAAN 4: EXPOSURE FUSION MERTENS
==========================================================================
Program ini mempelajari teknik Exposure Fusion menggunakan metode Mertens.
Berbeda dengan HDR pipeline (yang memerlukan CRF dan tone mapping),
Exposure Fusion langsung menggabungkan gambar LDR menjadi satu gambar
berkualitas tinggi tanpa membuat HDR radiance map.

Metode Mertens menggabungkan berdasarkan tiga metrik:
- Contrast   : Area dengan kontras tinggi diberi bobot lebih
- Saturation : Area dengan warna jenuh diberi bobot lebih
- Exposure   : Area dengan exposure baik (tidak terlalu terang/gelap)

Fungsi utama yang dipelajari:
- cv2.createMergeMertens(contrast_weight, saturation_weight, exposure_weight)
- Perbandingan variasi bobot (contrast, saturation, exposure)
- Perbandingan dengan simple averaging

Hasil: Visualisasi efek variasi bobot pada exposure fusion
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi numerik
import numpy as np

# Mengimpor os untuk manajemen path file
import os

# Mengimpor matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 4: EXPOSURE FUSION MERTENS")
print("=" * 60)

# ============================================================
# 1. Membaca gambar multi-exposure
# ============================================================

# Daftar file gambar dengan exposure berbeda
exposure_files = [
    "exposure_1.png", "exposure_2.png", "exposure_3.png",
    "exposure_4.png", "exposure_5.png"
]

# Label deskriptif untuk setiap exposure
exposure_labels = ["0.3x (Gelap)", "0.6x", "1.0x (Normal)", "1.5x", "2.5x (Terang)"]

# Membaca semua gambar
images = []
for fname in exposure_files:
    # Membentuk path lengkap
    img_path = os.path.join(IMAGE_DIR, fname)

    # Membaca gambar
    img = cv2.imread(img_path)

    # Validasi pembacaan
    if img is None:
        print(f"[ERROR] Gagal membaca: {img_path}")
        print("[INFO] Jalankan download_image.py terlebih dahulu!")
        exit()

    # Menambahkan gambar ke list
    images.append(img)

    # Menampilkan info dimensi
    print(f"[INFO] {fname} dimuat — {img.shape[1]}x{img.shape[0]}")

print(f"\n[INFO] Total {len(images)} gambar exposure dimuat")

# ============================================================
# 2. Simple Averaging sebagai baseline
# Rata-rata sederhana dari semua exposure
# ============================================================

# Mengkonversi list gambar ke array float32 untuk perhitungan
images_float = [img.astype(np.float32) for img in images]

# Menghitung rata-rata piksel dari semua exposure
average_image = np.mean(images_float, axis=0)

# Mengkonversi kembali ke uint8
average_8bit = np.clip(average_image, 0, 255).astype(np.uint8)

# Menghitung kecerahan rata-rata baseline
avg_brightness = np.mean(average_8bit)
print(f"\n[INFO] Simple Average — Kecerahan: {avg_brightness:.1f}")

# ============================================================
# 3. Exposure Fusion Mertens dengan variasi bobot
# cv2.createMergeMertens(contrast_weight, saturation_weight, exposure_weight)
# - contrast_weight  : Bobot untuk kontras lokal
# - saturation_weight: Bobot untuk saturasi warna
# - exposure_weight  : Bobot untuk kualitas exposure
# ============================================================

# Konfigurasi variasi bobot untuk percobaan
# Format: (contrast_w, saturation_w, exposure_w, label)
configs = [
    (1.0, 1.0, 1.0, "Default\nC=1.0, S=1.0, E=1.0"),
    (2.0, 1.0, 1.0, "High Contrast\nC=2.0, S=1.0, E=1.0"),
    (1.0, 2.0, 1.0, "High Saturation\nC=1.0, S=2.0, E=1.0"),
    (1.0, 1.0, 2.0, "High Exposure\nC=1.0, S=1.0, E=2.0"),
    (0.0, 1.0, 1.0, "No Contrast\nC=0.0, S=1.0, E=1.0"),
    (1.0, 0.0, 1.0, "No Saturation\nC=1.0, S=0.0, E=1.0"),
    (1.0, 1.0, 0.0, "No Exposure\nC=1.0, S=1.0, E=0.0"),
    (2.0, 2.0, 2.0, "All High\nC=2.0, S=2.0, E=2.0"),
]

# Menyimpan hasil setiap konfigurasi
results = []
brightness_values = []

print("\n--- Variasi Bobot Mertens ---")
for c_w, s_w, e_w, label in configs:
    # Membuat objek MergeMertens dengan bobot tertentu
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=c_w,
        saturation_weight=s_w,
        exposure_weight=e_w
    )

    # Menjalankan exposure fusion
    # Input: list gambar uint8, Output: gambar float (bisa > 1 atau < 0)
    fusion = merge_mertens.process(images)

    # Clipping nilai ke rentang 0-1
    fusion = np.clip(fusion, 0, 1)

    # Konversi ke uint8 (0-255)
    fusion_8bit = (fusion * 255).astype(np.uint8)

    # Menyimpan hasil
    results.append((fusion_8bit, label))

    # Menghitung kecerahan rata-rata
    brightness = np.mean(fusion_8bit)
    brightness_values.append(brightness)

    # Menampilkan info
    short_label = label.split(chr(10))[0]
    print(f"  {short_label:20s} — Kecerahan: {brightness:.1f}")

# ============================================================
# 4. Visualisasi hasil
# ============================================================

# Membuat figure besar 3 baris x 4 kolom
fig, axes = plt.subplots(3, 4, figsize=(22, 15))

# --- Baris 1, kolom 0-3: 4 konfigurasi pertama ---
for idx in range(4):
    rgb = cv2.cvtColor(results[idx][0], cv2.COLOR_BGR2RGB)
    axes[0, idx].imshow(rgb)
    axes[0, idx].set_title(f"Mertens #{idx+1}\n{results[idx][1]}", fontsize=8)
    axes[0, idx].axis("off")

# --- Baris 2, kolom 0-3: 4 konfigurasi berikutnya ---
for idx in range(4, 8):
    rgb = cv2.cvtColor(results[idx][0], cv2.COLOR_BGR2RGB)
    axes[1, idx-4].imshow(rgb)
    axes[1, idx-4].set_title(f"Mertens #{idx+1}\n{results[idx][1]}", fontsize=8)
    axes[1, idx-4].axis("off")

# --- Baris 3: Simple Average, Best Mertens, perbandingan ---
# Simple Average
avg_rgb = cv2.cvtColor(average_8bit, cv2.COLOR_BGR2RGB)
axes[2, 0].imshow(avg_rgb)
axes[2, 0].set_title("Simple Average\n(Baseline)", fontsize=9)
axes[2, 0].axis("off")

# Mertens Default (terbaik secara umum)
default_rgb = cv2.cvtColor(results[0][0], cv2.COLOR_BGR2RGB)
axes[2, 1].imshow(default_rgb)
axes[2, 1].set_title("Mertens Default\n(C=1, S=1, E=1)", fontsize=9)
axes[2, 1].axis("off")

# Exposure tengah untuk referensi
ref_rgb = cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB)
axes[2, 2].imshow(ref_rgb)
axes[2, 2].set_title("Referensi\nExposure Normal (1.0x)", fontsize=9)
axes[2, 2].axis("off")

# Bar chart kecerahan
short_labels = [c[1].split(chr(10))[0][:12] for c in configs]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(configs)))
axes[2, 3].barh(short_labels, brightness_values, color=colors)
axes[2, 3].axvline(x=avg_brightness, color='r', linestyle='--',
                    label=f'Simple Avg: {avg_brightness:.0f}')
axes[2, 3].set_title("Kecerahan Rata-rata\nSetiap Konfigurasi", fontsize=9)
axes[2, 3].set_xlabel("Mean Brightness")
axes[2, 3].legend(fontsize=7)
axes[2, 3].tick_params(labelsize=7)

# Menambahkan judul utama
plt.suptitle("Percobaan 4: Exposure Fusion Mertens — Variasi Bobot\n"
             "contrast_weight, saturation_weight, exposure_weight",
             fontsize=14, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi
output_path = os.path.join(OUTPUT_DIR, "04_exposure_fusion_mertens.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 4")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.createMergeMertens(contrast_w, saturation_w, exposure_w)")
print("     - contrast_weight  : Prioritas area kontras tinggi")
print("     - saturation_weight: Prioritas area warna jenuh")
print("     - exposure_weight  : Prioritas area exposure baik")
print("  2. Exposure Fusion vs HDR Pipeline:")
print("     - Tidak memerlukan exposure times atau CRF")
print("     - Lebih cepat dan sederhana")
print("     - Kualitas sudah cukup baik untuk banyak kasus")
print("  3. Simple Average menghasilkan gambar blur/pudar")
print("  4. Mertens mempertahankan detail terbaik dari setiap exposure")
print(f"  5. Kecerahan rata-rata: Simple Avg={avg_brightness:.0f}, "
      f"Mertens Default={brightness_values[0]:.0f}")
print("=" * 60)
