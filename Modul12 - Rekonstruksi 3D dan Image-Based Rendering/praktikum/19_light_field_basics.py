"""
==========================================================
PERCOBAAN 19: DASAR-DASAR LIGHT FIELD
Mempelajari konsep light field: representasi 4D dari
cahaya (2D posisi + 2D arah) dan bagaimana menghasilkan
novel view dari light field array.

Fungsi utama:
- numpy 4D array operations
- cv2.resize(), cv2.warpAffine()
- matplotlib visualization
==========================================================
"""

# Mengimpor library numpy untuk operasi numerik
import numpy as np

# Mengimpor library os untuk operasi sistem file
import os

# Mengimpor library OpenCV untuk pemrosesan citra
import cv2

# Mengimpor library matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mencoba mengimpor Open3D untuk visualisasi tambahan
try:
    # Mengimpor library Open3D
    import open3d as o3d
    # Menandai ketersediaan Open3D
    HAS_OPEN3D = True
    print("[INFO] Open3D berhasil diimpor")
except ImportError:
    # Menandai bahwa Open3D tidak tersedia
    HAS_OPEN3D = False
    print("[INFO] Open3D tidak tersedia, menggunakan cv2/matplotlib")

# ========================================================
# KONFIGURASI DIREKTORI
# ========================================================

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Menentukan direktori untuk gambar/data input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Menentukan direktori untuk menyimpan hasil output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat direktori output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mencetak header utama percobaan
print("=" * 60)
print("PERCOBAAN 19: DASAR-DASAR LIGHT FIELD")
print("=" * 60)

# ========================================================
# BAGIAN 1: KONSEP LIGHT FIELD (PARAMETERISASI 4D)
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 1: Konsep Light Field (Parameterisasi 4D)")
print("=" * 60)

# Menjelaskan parameterisasi 4D light field
print("Light Field L(u, v, s, t):")
print("  (u, v) = posisi kamera pada bidang kamera")
print("  (s, t) = koordinat piksel pada bidang gambar")
print("  Light field menangkap semua sinar cahaya di ruang 3D")
print("  Array kamera NxN mensimulasikan pengambilan light field")

# Menentukan parameter light field array
jumlah_kamera_u = 5  # Jumlah kamera arah horizontal
jumlah_kamera_v = 5  # Jumlah kamera arah vertikal
resolusi_gambar = 200  # Resolusi setiap sub-aperture image

# Mencetak konfigurasi light field
print(f"\nKonfigurasi:")
print(f"  Grid kamera     : {jumlah_kamera_u}x{jumlah_kamera_v}")
print(f"  Resolusi gambar  : {resolusi_gambar}x{resolusi_gambar}")
print(f"  Total views      : {jumlah_kamera_u * jumlah_kamera_v}")

# ========================================================
# BAGIAN 2: MEMUAT SCENE NYATA DAN ESTIMASI DEPTH
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 2: Memuat Gambar Nyata dan Estimasi Depth")
print("=" * 60)

# Mendefinisikan path gambar scene
_scene_path = os.path.join(IMAGE_DIR, "multiview_00.png")

# Download otomatis jika gambar tidak tersedia
if not os.path.exists(_scene_path):
    print("  [WARN] multiview_00.png tidak ditemukan. Menjalankan download_image.py otomatis...")
    import subprocess as _subp, sys as _sys
    _dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_image.py")
    _subp.run([_sys.executable, _dl], check=False)
if not os.path.exists(_scene_path):
    raise FileNotFoundError(
        "[ERROR] multiview_00.png tidak tersedia.\n"
        "  Jalankan: python download_image.py"
    )

# Memuat dan resize gambar scene
_scene_raw = cv2.imread(_scene_path)
scene = cv2.resize(_scene_raw, (resolusi_gambar, resolusi_gambar))
print(f"  Gambar scene dimuat: {_scene_path}")

# Mengestimasi depth map dari gambar menggunakan edge strength
# (area dengan banyak edges/detail = dekat; area uniform = jauh)
_gray_sc  = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
_lap_sc   = np.abs(cv2.Laplacian(_gray_sc.astype(np.float32), cv2.CV_32F))
_lap_blur = cv2.GaussianBlur(_lap_sc, (21, 21), 0)
_lap_max  = _lap_blur.max() if _lap_blur.max() > 0 else 1
depth_map = (1.0 - (_lap_blur / _lap_max) * 0.8).astype(np.float32)  # 0=dekat, 1=jauh

# Mencetak informasi scene
print(f"Ukuran scene   : {scene.shape}")
print(f"Rentang depth  : [{depth_map.min():.2f}, {depth_map.max():.2f}]")

# ========================================================
# BAGIAN 3: SIMULASI LIGHT FIELD ARRAY (GRID OF VIEWS)
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 3: Simulasi Light Field Array (Grid of Views)")
print("=" * 60)

# Menentukan baseline antar kamera (dalam piksel)
baseline = 3.0

# Membuat array 4D untuk menyimpan light field
# Dimensi: (u, v, s, t, channels)
light_field = np.zeros(
    (jumlah_kamera_u, jumlah_kamera_v,
     resolusi_gambar, resolusi_gambar, 3),
    dtype=np.uint8
)

# Menghasilkan view dari setiap posisi kamera dalam grid
for u in range(jumlah_kamera_u):
    for v in range(jumlah_kamera_v):
        # Menghitung offset kamera relatif terhadap pusat grid
        offset_u = (u - jumlah_kamera_u // 2) * baseline
        offset_v = (v - jumlah_kamera_v // 2) * baseline

        # Menghitung disparitas berdasarkan depth dan offset
        # Disparitas = offset / depth (semakin dekat, semakin besar)
        disp_u = offset_u / np.maximum(depth_map, 0.1)
        disp_v = offset_v / np.maximum(depth_map, 0.1)

        # Membuat map untuk cv2.remap (inverse warping)
        map_x = np.zeros((resolusi_gambar, resolusi_gambar), dtype=np.float32)
        map_y = np.zeros((resolusi_gambar, resolusi_gambar), dtype=np.float32)

        # Menghitung koordinat sumber untuk setiap piksel target
        for s in range(resolusi_gambar):
            for t in range(resolusi_gambar):
                map_x[s, t] = t - disp_u[s, t]
                map_y[s, t] = s - disp_v[s, t]

        # Melakukan warping menggunakan cv2.remap
        view = cv2.remap(
            scene, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Menyimpan view ke light field array
        light_field[u, v] = view

# Mencetak informasi light field
print(f"Ukuran light field: {light_field.shape}")
print(f"Memory: {light_field.nbytes / 1024 / 1024:.2f} MB")

# ========================================================
# BAGIAN 4: EKSTRAKSI SUB-APERTURE IMAGES
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 4: Ekstraksi Sub-Aperture Images")
print("=" * 60)

# Membuat figure untuk menampilkan semua sub-aperture images
fig1, axes1 = plt.subplots(
    jumlah_kamera_u, jumlah_kamera_v,
    figsize=(15, 15)
)

# Mengatur judul figure
fig1.suptitle("Sub-Aperture Images (Light Field Array)",
              fontsize=14, fontweight='bold')

# Iterasi setiap posisi kamera
for u in range(jumlah_kamera_u):
    for v in range(jumlah_kamera_v):
        # Mengambil sub-aperture image
        sai = light_field[u, v]

        # Menampilkan sub-aperture image
        axes1[u, v].imshow(cv2.cvtColor(sai, cv2.COLOR_BGR2RGB))
        axes1[u, v].axis('off')

        # Menambahkan label posisi kamera
        if u == 0:
            axes1[u, v].set_title(f"v={v}", fontsize=8)
        if v == 0:
            axes1[u, v].set_ylabel(f"u={u}", fontsize=8)

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan visualisasi sub-aperture images
path_sai = os.path.join(OUTPUT_DIR, "19_light_field_sub_aperture.png")
plt.savefig(path_sai, dpi=150, bbox_inches='tight')
print(f"Sub-aperture images disimpan: {path_sai}")

# Menutup figure
plt.close(fig1)

# ========================================================
# BAGIAN 5: REFOCUS - SHIFT AND ADD
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 5: Refocus menggunakan Shift-and-Add")
print("=" * 60)

def refocus_light_field(light_field, alpha):
    """Refocus light field pada kedalaman tertentu menggunakan shift-and-add."""
    # Mendapatkan dimensi light field
    nu, nv, h, w, c = light_field.shape

    # Menghitung pusat grid kamera
    center_u = nu // 2
    center_v = nv // 2

    # Membuat akumulator untuk gambar refocused
    akumulator = np.zeros((h, w, c), dtype=np.float64)

    # Menghitung jumlah view yang berkontribusi
    jumlah_view = 0

    # Iterasi setiap view dalam light field
    for u in range(nu):
        for v in range(nv):
            # Menghitung shift berdasarkan alpha dan posisi kamera
            shift_x = (v - center_v) * alpha
            shift_y = (u - center_u) * alpha

            # Membuat matriks translasi untuk shifting
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

            # Melakukan shifting menggunakan warpAffine
            shifted = cv2.warpAffine(
                light_field[u, v], M, (w, h),
                borderMode=cv2.BORDER_REPLICATE
            )

            # Menambahkan view yang sudah di-shift ke akumulator
            akumulator += shifted.astype(np.float64)
            jumlah_view += 1

    # Menghitung rata-rata dari semua view
    hasil = (akumulator / jumlah_view).astype(np.uint8)

    return hasil

# Mendefinisikan berbagai nilai alpha untuk refocusing
# alpha > 0: fokus dekat, alpha < 0: fokus jauh, alpha = 0: fokus tengah
alpha_values = [-2.0, -1.0, 0.0, 1.0, 2.0]

# Menyimpan hasil refocusing
hasil_refocus = {}

# Melakukan refocusing untuk setiap alpha
for alpha in alpha_values:
    # Melakukan refocusing
    gambar_refocus = refocus_light_field(light_field, alpha)

    # Menyimpan hasil
    hasil_refocus[alpha] = gambar_refocus

    # Mencetak informasi
    print(f"Alpha={alpha:+.1f}: refocused (mean brightness={gambar_refocus.mean():.1f})")

# ========================================================
# BAGIAN 6: VISUALISASI GAMBAR REFOCUSED
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 6: Visualisasi Gambar Refocused")
print("=" * 60)

# Membuat figure untuk gambar refocused
fig2, axes2 = plt.subplots(1, len(alpha_values), figsize=(20, 5))

# Mengatur judul figure
fig2.suptitle("Refocusing Light Field pada Berbagai Kedalaman",
              fontsize=14, fontweight='bold')

# Menampilkan setiap hasil refocusing
for idx, alpha in enumerate(alpha_values):
    # Mengkonversi BGR ke RGB untuk matplotlib
    gambar_rgb = cv2.cvtColor(hasil_refocus[alpha], cv2.COLOR_BGR2RGB)

    # Menampilkan gambar refocused
    axes2[idx].imshow(gambar_rgb)

    # Menentukan label kedalaman fokus
    if alpha < 0:
        label_depth = "Jauh"
    elif alpha > 0:
        label_depth = "Dekat"
    else:
        label_depth = "Tengah"

    # Mengatur judul subplot
    axes2[idx].set_title(f"α={alpha:+.1f}\n({label_depth})")
    axes2[idx].axis('off')

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan visualisasi refocusing
path_refocus = os.path.join(OUTPUT_DIR, "19_light_field_refocus.png")
plt.savefig(path_refocus, dpi=150, bbox_inches='tight')
print(f"Refocusing disimpan: {path_refocus}")

# Menutup figure
plt.close(fig2)

# ========================================================
# BAGIAN 7: PERBANDINGAN FOKUS DEKAT VS JAUH
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 7: Perbandingan Fokus Dekat vs Jauh")
print("=" * 60)

# Membuat figure perbandingan fokus dekat vs jauh
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

# Mengatur judul figure
fig3.suptitle("Perbandingan Fokus: Dekat vs Tengah vs Jauh",
              fontsize=14, fontweight='bold')

# Menampilkan fokus dekat (alpha=2.0)
gambar_dekat = cv2.cvtColor(hasil_refocus[2.0], cv2.COLOR_BGR2RGB)
axes3[0].imshow(gambar_dekat)
axes3[0].set_title("Fokus Dekat (α=+2.0)\nObjek merah tajam")
axes3[0].axis('off')

# Menampilkan fokus tengah (alpha=0.0)
gambar_tengah = cv2.cvtColor(hasil_refocus[0.0], cv2.COLOR_BGR2RGB)
axes3[1].imshow(gambar_tengah)
axes3[1].set_title("Fokus Tengah (α=0.0)\nObjek hijau tajam")
axes3[1].axis('off')

# Menampilkan fokus jauh (alpha=-2.0)
gambar_jauh = cv2.cvtColor(hasil_refocus[-2.0], cv2.COLOR_BGR2RGB)
axes3[2].imshow(gambar_jauh)
axes3[2].set_title("Fokus Jauh (α=-2.0)\nBackground tajam")
axes3[2].axis('off')

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan perbandingan fokus
path_fokus = os.path.join(OUTPUT_DIR, "19_light_field_fokus_compare.png")
plt.savefig(path_fokus, dpi=150, bbox_inches='tight')
print(f"Perbandingan fokus disimpan: {path_fokus}")

# Menutup figure
plt.close(fig3)

# ========================================================
# BAGIAN 8: VISUALISASI STRUKTUR LIGHT FIELD
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 8: Visualisasi Struktur Light Field")
print("=" * 60)

# Membuat figure untuk struktur light field
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 14))

# Mengatur judul figure
fig4.suptitle("Struktur Light Field", fontsize=14, fontweight='bold')

# Menampilkan scene asli dan depth map
axes4[0, 0].imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
axes4[0, 0].set_title("Scene Asli (View Pusat)")
axes4[0, 0].axis('off')

# Menampilkan depth map
im_depth = axes4[0, 1].imshow(depth_map, cmap='viridis')
axes4[0, 1].set_title("Depth Map")
axes4[0, 1].axis('off')
plt.colorbar(im_depth, ax=axes4[0, 1], fraction=0.046)

# Membuat EPI (Epipolar Plane Image) - horizontal
# EPI adalah slice horizontal melalui light field pada y tertentu
y_epi = resolusi_gambar // 2  # Baris tengah
epi_horizontal = np.zeros((jumlah_kamera_u, resolusi_gambar, 3), dtype=np.uint8)
for u in range(jumlah_kamera_u):
    # Mengambil baris y_epi dari view (u, center_v)
    center_v = jumlah_kamera_v // 2
    epi_horizontal[u] = light_field[u, center_v, y_epi, :, :]

# Memperbesar EPI untuk visualisasi yang lebih jelas
epi_besar = cv2.resize(epi_horizontal, (resolusi_gambar, resolusi_gambar),
                        interpolation=cv2.INTER_NEAREST)

# Menampilkan EPI horizontal
axes4[1, 0].imshow(cv2.cvtColor(epi_besar, cv2.COLOR_BGR2RGB))
axes4[1, 0].set_title(f"EPI Horizontal (y={y_epi})")
axes4[1, 0].set_xlabel("Koordinat x (piksel)")
axes4[1, 0].set_ylabel("Posisi kamera u")

# Membuat EPI vertikal
x_epi = resolusi_gambar // 2  # Kolom tengah
epi_vertikal = np.zeros((jumlah_kamera_v, resolusi_gambar, 3), dtype=np.uint8)
for v in range(jumlah_kamera_v):
    # Mengambil kolom x_epi dari view (center_u, v)
    center_u = jumlah_kamera_u // 2
    epi_vertikal[v] = light_field[center_u, v, :, x_epi, :]

# Memperbesar EPI vertikal
epi_vert_besar = cv2.resize(epi_vertikal, (resolusi_gambar, resolusi_gambar),
                             interpolation=cv2.INTER_NEAREST)

# Menampilkan EPI vertikal
axes4[1, 1].imshow(cv2.cvtColor(epi_vert_besar, cv2.COLOR_BGR2RGB))
axes4[1, 1].set_title(f"EPI Vertikal (x={x_epi})")
axes4[1, 1].set_xlabel("Koordinat y (piksel)")
axes4[1, 1].set_ylabel("Posisi kamera v")

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan visualisasi struktur light field
path_struktur = os.path.join(OUTPUT_DIR, "19_light_field_struktur.png")
plt.savefig(path_struktur, dpi=150, bbox_inches='tight')
print(f"Struktur light field disimpan: {path_struktur}")

# Menutup figure
plt.close(fig4)

# ========================================================
# BAGIAN 9: MENYIMPAN GAMBAR REFOCUSED INDIVIDUAL
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 9: Menyimpan Gambar Refocused Individual")
print("=" * 60)

# Menyimpan setiap gambar refocused sebagai file terpisah
for alpha, gambar in hasil_refocus.items():
    # Membuat nama file berdasarkan alpha
    nama_file = f"19_light_field_alpha_{alpha:+.1f}.png"
    path_file = os.path.join(OUTPUT_DIR, nama_file.replace("+", "pos").replace("-", "neg"))

    # Menyimpan gambar menggunakan OpenCV
    cv2.imwrite(path_file, gambar)
    print(f"  Disimpan: {os.path.basename(path_file)}")

# Menyimpan scene asli dan depth map
path_scene = os.path.join(OUTPUT_DIR, "19_light_field_scene.png")
cv2.imwrite(path_scene, scene)
print(f"  Scene disimpan: {os.path.basename(path_scene)}")

# Menyimpan depth map sebagai visualisasi
depth_vis = (depth_map * 255).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
path_depth = os.path.join(OUTPUT_DIR, "19_light_field_depth.png")
cv2.imwrite(path_depth, depth_color)
print(f"  Depth map disimpan: {os.path.basename(path_depth)}")

# ========================================================
# RINGKASAN
# ========================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 19: DASAR-DASAR LIGHT FIELD")
print("=" * 60)
print(f"Grid kamera          : {jumlah_kamera_u}x{jumlah_kamera_v}")
print(f"Resolusi per view    : {resolusi_gambar}x{resolusi_gambar}")
print(f"Baseline antar kamera: {baseline} piksel")
print(f"Alpha values tested  : {alpha_values}")
print(f"Total views          : {jumlah_kamera_u * jumlah_kamera_v}")
print(f"\nSemua output disimpan di: {OUTPUT_DIR}")
print("=" * 60)
