"""
==========================================================================
PERCOBAAN 18: TRANSFORMASI FOURIER (DFT)
==========================================================================
Discrete Fourier Transform (DFT) mengubah gambar dari domain spasial
ke domain frekuensi. Komponen frekuensi rendah = area smooth,
frekuensi tinggi = detail dan tepi.

Fungsi:
- cv2.dft(src, flags) → DFT 2D
- cv2.idft(src) → inverse DFT
- np.fft.fft2() / np.fft.ifft2() → alternatif NumPy
- cv2.magnitude() → hitung magnitude dari komponen real + imag
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(os.path.join(IMAGE_DIR, "spektrum.png"), cv2.IMREAD_GRAYSCALE)
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()
img = cv2.resize(img, (512, 512))

print("=" * 60)
print("PERCOBAAN 18: TRANSFORMASI FOURIER (DFT)")
print("=" * 60)

# ============================================================
# 1. DFT Menggunakan NumPy
# ============================================================
print("\n--- 1. DFT dengan NumPy ---")

# Hitung DFT 2D
f_transform = np.fft.fft2(img.astype(np.float32))
# Geser zero-frequency ke tengah
f_shift = np.fft.fftshift(f_transform)
# Magnitude spectrum (log scale untuk visualisasi)
magnitude = np.abs(f_shift)
mag_log = 20 * np.log(magnitude + 1)
# Phase spectrum
phase = np.angle(f_shift)

print(f"  Ukuran DFT: {f_transform.shape}")
print(f"  Magnitude max: {magnitude.max():.0f}")
print(f"  Phase range: [{phase.min():.2f}, {phase.max():.2f}]")

# ============================================================
# 2. DFT Menggunakan OpenCV
# ============================================================
print("\n--- 2. DFT dengan OpenCV ---")

# cv2.dft membutuhkan input float32
dft = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
# dft menghasilkan 2 channel: real, imaginary
dft_shift = np.fft.fftshift(dft)
# Hitung magnitude
mag_cv = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
mag_cv_log = 20 * np.log(mag_cv + 1)
print(f"  OpenCV DFT shape: {dft.shape}")

# ============================================================
# 3. Inverse DFT (Rekonstruksi)
# ============================================================
print("\n--- 3. Inverse DFT ---")

# Rekonstruksi dari DFT
f_ishift = np.fft.ifftshift(f_shift)
reconstructed = np.fft.ifft2(f_ishift)
reconstructed = np.abs(reconstructed).astype(np.uint8)

# Cek error rekonstruksi
error = np.mean(cv2.absdiff(img, reconstructed))
print(f"  Error rekonstruksi: {error:.4f}")

# ============================================================
# 4. Magnitude-Only dan Phase-Only Reconstruction
# ============================================================
print("\n--- 4. Magnitude vs Phase ---")

# Rekonstruksi hanya dari magnitude (phase = 0)
mag_only = np.abs(f_shift)
recon_mag = np.abs(np.fft.ifft2(np.fft.ifftshift(mag_only)))
recon_mag = cv2.normalize(recon_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Rekonstruksi hanya dari phase (magnitude = 1)
phase_only = np.exp(1j * phase)
recon_phase = np.abs(np.fft.ifft2(np.fft.ifftshift(phase_only)))
recon_phase = cv2.normalize(recon_phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

print("  Magnitude-only: kehilangan struktur spasial")
print("  Phase-only: mempertahankan struktur (edge, posisi)")

# ============================================================
# 5. DFT pada Gambar Lain
# ============================================================
print("\n--- 5. DFT Gambar Lain ---")

# Gambar kota
kota = cv2.imread(os.path.join(IMAGE_DIR, "kota.jpg"), cv2.IMREAD_GRAYSCALE)
if kota is not None:
    kota = cv2.resize(kota, (512, 512))
    f_kota = np.fft.fftshift(np.fft.fft2(kota.astype(np.float32)))
    mag_kota = 20 * np.log(np.abs(f_kota) + 1)
    print(f"  Kota DFT computed")

# Checkerboard - seharusnya menunjukkan titik-titik diskrit
checker = np.zeros((512, 512), dtype=np.uint8)
for i in range(512):
    for j in range(512):
        if ((i // 32) + (j // 32)) % 2 == 0:
            checker[i, j] = 255
f_checker = np.fft.fftshift(np.fft.fft2(checker.astype(np.float32)))
mag_checker = 20 * np.log(np.abs(f_checker) + 1)
print("  Checkerboard → titik frekuensi diskrit")

# ============================================================
# 6. Optimal DFT Size
# ============================================================
print("\n--- 6. Optimal DFT Size ---")

# OpenCV DFT lebih cepat untuk ukuran tertentu (2^n, dll)
rows, cols = img.shape
opt_rows = cv2.getOptimalDFTSize(rows)
opt_cols = cv2.getOptimalDFTSize(cols)
print(f"  Original: {rows}×{cols}")
print(f"  Optimal:  {opt_rows}×{opt_cols}")

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: DFT dasar
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Original (Spektrum)")
axes[0, 0].axis("off")

axes[0, 1].imshow(mag_log, cmap='gray')
axes[0, 1].set_title("Magnitude (log)")
axes[0, 1].axis("off")

axes[0, 2].imshow(phase, cmap='gray')
axes[0, 2].set_title("Phase")
axes[0, 2].axis("off")

axes[0, 3].imshow(reconstructed, cmap='gray')
axes[0, 3].set_title("Rekonstruksi")
axes[0, 3].axis("off")

# Baris 2: Mag vs Phase
axes[1, 0].imshow(recon_mag, cmap='gray')
axes[1, 0].set_title("Mag-Only Recon")
axes[1, 0].axis("off")

axes[1, 1].imshow(recon_phase, cmap='gray')
axes[1, 1].set_title("Phase-Only Recon")
axes[1, 1].axis("off")

if kota is not None:
    axes[1, 2].imshow(kota, cmap='gray')
    axes[1, 2].set_title("Kota")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(mag_kota, cmap='gray')
    axes[1, 3].set_title("DFT Kota")
    axes[1, 3].axis("off")

# Baris 3: Checkerboard
axes[2, 0].imshow(checker, cmap='gray')
axes[2, 0].set_title("Checkerboard")
axes[2, 0].axis("off")

axes[2, 1].imshow(mag_checker, cmap='gray')
axes[2, 1].set_title("DFT Checkerboard")
axes[2, 1].axis("off")

axes[2, 2].axis("off")
axes[2, 3].axis("off")

plt.suptitle("Percobaan 18: Transformasi Fourier (DFT)", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "18_dft_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 18")
print("=" * 60)
print("""
1. DFT mengubah gambar ke domain frekuensi (spatial → frequency)
2. np.fft.fft2() atau cv2.dft() untuk menghitung DFT
3. fftshift() menempatkan DC component (frek=0) di tengah
4. Magnitude spectrum menunjukkan kekuatan setiap frekuensi
5. Phase spectrum menyimpan posisi/struktur spasial
6. Phase lebih penting dari magnitude untuk persepsi visual
7. Pola periodik → titik diskrit di DFT
""")
