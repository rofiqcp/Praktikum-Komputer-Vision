"""
==========================================================================
PERCOBAAN 3: VISUALISASI OPTICAL FLOW DENGAN VEKTOR PANAH
==========================================================================
Program ini mempelajari berbagai cara memvisualisasikan optical flow:
1. HSV Color Coding (warna = arah, kecerahan = kecepatan)
2. Vektor panah (arrow/quiver) overlay pada gambar asli
3. Flow magnitude sebagai heatmap

Visualisasi yang baik sangat penting untuk memahami dan mendebug
hasil optical flow.

Fungsi utama yang dipelajari:
- cv2.arrowedLine()               : Menggambar panah berarah
- cv2.applyColorMap()             : Menerapkan colormap pada gambar
- cv2.cartToPolar()               : Konversi kartesian ke polar
- np.mgrid[]                      : Membuat grid koordinat

Hasil: Perbandingan 3 metode visualisasi optical flow
==========================================================================
"""

# Mengimpor library yang dibutuhkan
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 3: VISUALISASI OPTICAL FLOW DENGAN VEKTOR PANAH")
print("=" * 60)

# ============================================================
# 1. Membaca frame pair dan menghitung optical flow
# ============================================================

# Membaca dua frame statis yang sudah disiapkan
frame1_path = os.path.join(IMAGE_DIR, "frame_t0.png")
frame2_path = os.path.join(IMAGE_DIR, "frame_t1.png")

frame1 = cv2.imread(frame1_path)
frame2 = cv2.imread(frame2_path)

if frame1 is None or frame2 is None:
    print("[ERROR] Frame tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Mengkonversi ke grayscale karena optical flow butuh 1 channel
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

print(f"[INFO] Ukuran frame: {gray1.shape}")

# Menghitung dense optical flow menggunakan Farneback
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Mengekstrak komponen u (horizontal) dan v (vertikal)
flow_u = flow[..., 0]
flow_v = flow[..., 1]

# Menghitung magnitude dan sudut
magnitude, angle = cv2.cartToPolar(flow_u, flow_v)

print(f"[INFO] Flow dihitung: max magnitude = {magnitude.max():.2f} piksel")

# ============================================================
# 2. Visualisasi 1: HSV Color Coding
# Standar industri untuk visualisasi optical flow
# ============================================================

# Membuat gambar HSV
hsv = np.zeros((gray1.shape[0], gray1.shape[1], 3), dtype=np.uint8)

# Hue = arah gerakan (0-180 derajat, OpenCV menggunakan setengah derajat)
hsv[..., 0] = angle * 180 / np.pi / 2

# Saturation = penuh (255) agar warna terlihat jelas
hsv[..., 1] = 255

# Value = kecepatan gerakan (dinormalisasi ke 0-255)
hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

# Konversi HSV ke BGR
vis_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

print("[INFO] Visualisasi HSV selesai.")

# ============================================================
# 3. Visualisasi 2: Vektor Panah (Arrow/Quiver)
# Menggambar panah arah gerakan pada titik-titik grid
# ============================================================

# Membuat salinan frame pertama sebagai background
vis_arrow = frame1.copy()

# Mendefinisikan langkah grid (setiap berapa piksel ada panah)
step = 25

# Faktor skala untuk memperbesar panah agar terlihat jelas
scale = 3.0

# Iterasi melalui grid piksel
for y in range(step // 2, gray1.shape[0], step):
    for x in range(step // 2, gray1.shape[1], step):
        # Mengambil komponen flow di titik (x, y)
        fx = flow_u[y, x]
        fy = flow_v[y, x]

        # Menghitung titik ujung panah (posisi baru setelah flow)
        end_x = int(x + fx * scale)
        end_y = int(y + fy * scale)

        # Menghitung magnitude di titik ini
        mag = np.sqrt(fx**2 + fy**2)

        # Hanya gambar panah jika ada gerakan signifikan
        if mag > 0.5:
            # cv2.arrowedLine(img, start, end, color, thickness, tipLength)
            # Warna panah berdasarkan kecepatan: hijau=lambat, merah=cepat
            color_intensity = min(255, int(mag * 30))
            color = (0, 255 - color_intensity, color_intensity)

            cv2.arrowedLine(
                vis_arrow,
                (x, y),                # Titik awal panah
                (end_x, end_y),        # Titik ujung panah
                color,                 # Warna panah (BGR)
                1,                     # Ketebalan garis
                tipLength=0.3          # Panjang ujung panah (proporsi)
            )

print(f"[INFO] Visualisasi panah selesai (grid {step}px).")

# ============================================================
# 4. Visualisasi 3: Heatmap Magnitude
# Menggunakan colormap untuk menunjukkan area bergerak
# ============================================================

# Normalisasi magnitude ke range 0-255
mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Menerapkan colormap JET (biru=lambat, merah=cepat)
# cv2.applyColorMap(src, colormap) mengubah gambar grayscale jadi berwarna
vis_heatmap = cv2.applyColorMap(mag_normalized, cv2.COLORMAP_JET)

# Blend heatmap dengan frame asli untuk konteks
# cv2.addWeighted(src1, alpha1, src2, alpha2, gamma)
vis_heatmap_blend = cv2.addWeighted(frame1, 0.5, vis_heatmap, 0.5, 0)

print("[INFO] Visualisasi heatmap selesai.")

# ============================================================
# 5. Visualisasi 4: Color Wheel Legend
# Membuat roda warna sebagai legenda visualisasi HSV
# ============================================================

# Membuat roda warna 200x200 piksel
wheel_size = 200
wheel = np.zeros((wheel_size, wheel_size, 3), dtype=np.uint8)
center = wheel_size // 2

for y in range(wheel_size):
    for x in range(wheel_size):
        # Menghitung jarak dan sudut dari pusat
        dx = x - center
        dy = y - center
        dist = np.sqrt(dx**2 + dy**2)

        if dist <= center - 5:
            # Sudut dalam derajat (0-360)
            ang = np.arctan2(dy, dx)
            if ang < 0:
                ang += 2 * np.pi

            # HSV: Hue=sudut, Sat=jarak, Val=255
            wheel[y, x, 0] = int(ang * 180 / np.pi / 2)
            wheel[y, x, 1] = int(min(255, dist / center * 255))
            wheel[y, x, 2] = 255

# Konversi roda warna dari HSV ke BGR
wheel_bgr = cv2.cvtColor(wheel, cv2.COLOR_HSV2BGR)

print("[INFO] Roda warna (color wheel legend) selesai.")

# ============================================================
# 6. Menyimpan semua visualisasi
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Frame pertama (input)
axes[0, 0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Frame T=0 (Input)", fontsize=12)
axes[0, 0].axis("off")

# Frame kedua (input)
axes[0, 1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Frame T=1 (Input)", fontsize=12)
axes[0, 1].axis("off")

# Color wheel legend
axes[0, 2].imshow(cv2.cvtColor(wheel_bgr, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Color Wheel Legend\n(Warna=Arah, Sat=Kecepatan)", fontsize=11)
axes[0, 2].axis("off")

# HSV flow visualization
axes[1, 0].imshow(cv2.cvtColor(vis_hsv, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Vis 1: HSV Color Coding", fontsize=12)
axes[1, 0].axis("off")

# Arrow visualization
axes[1, 1].imshow(cv2.cvtColor(vis_arrow, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Vis 2: Vektor Panah", fontsize=12)
axes[1, 1].axis("off")

# Heatmap magnitude
axes[1, 2].imshow(cv2.cvtColor(vis_heatmap_blend, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Vis 3: Heatmap Magnitude", fontsize=12)
axes[1, 2].axis("off")

plt.suptitle("Percobaan 3: Tiga Metode Visualisasi Optical Flow",
             fontsize=14, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "03_visualisasi_optical_flow.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 3")
print("=" * 60)
print("Metode visualisasi optical flow:")
print("  1. HSV Color Coding:")
print("     - Hue = arah gerakan (0°-360°)")
print("     - Value = kecepatan (terang = cepat)")
print("     - Standar untuk paper dan evaluasi")
print("  2. Vektor Panah (Quiver):")
print("     - cv2.arrowedLine() untuk gambar panah")
print("     - Intuitif, menunjukkan arah & magnitude")
print("     - Perlu subsampling (grid) agar tidak terlalu padat")
print("  3. Heatmap Magnitude:")
print("     - cv2.applyColorMap() dengan COLORMAP_JET")
print("     - Menunjukkan area mana yang bergerak")
print("     - Blend dengan gambar asli untuk konteks")
print("=" * 60)
