"""
==========================================================================
PERCOBAAN 13: SCENE RECOGNITION / KLASIFIKASI SCENE
==========================================================================
Program ini mempelajari klasifikasi scene (pemandangan) menggunakan fitur
berbasis spatial. Scene recognition berbeda dari object recognition karena
fokus pada keseluruhan komposisi gambar, bukan objek individual.

Konsep yang dipelajari:
- GIST-like features: histogram gradient terorientasi pada spatial grid
- Spatial pyramid: membagi gambar menjadi grid dan menghitung fitur per cell
- Color histogram per region: distribusi warna di setiap bagian gambar
- Edge histogram per region: distribusi tepi di setiap bagian gambar
- Scene descriptor: gabungan fitur spatial (color + edge + gradient)
- Nearest-neighbor classification pada scene descriptor
- Perbandingan dengan klasifikasi berbasis color histogram global

Fungsi utama yang dipelajari:
- cv2.Sobel()                     : Menghitung gradient gambar
- cv2.calcHist()                  : Menghitung histogram warna
- cv2.Canny()                     : Deteksi tepi untuk edge histogram
- np.histogram()                  : Membuat histogram distribusi
- np.concatenate()                : Menggabungkan fitur menjadi descriptor

Hasil: Visualisasi fitur spatial, perbandingan scene, dan klasifikasi
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi array dan histogram
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 13: SCENE RECOGNITION / KLASIFIKASI SCENE")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Scene (3 Kategori)
# ============================================================

print("\n[INFO] Memuat gambar scene...")
print("-" * 50)

# Mendefinisikan kategori scene dan file gambar
scene_files = {
    "pantai": "scene_pantai.jpg",
    "kota": "scene_kota.jpg",
    "hutan": "scene_hutan.jpg"
}

# Membuat dictionary untuk menyimpan gambar scene
scene_images = {}

# Memuat setiap gambar scene
for scene_name, filename in scene_files.items():
    # Membaca gambar scene dari file
    img = cv2.imread(os.path.join(IMAGE_DIR, filename))

    # Memeriksa apakah gambar berhasil dimuat
    if img is not None:
        # Menyimpan gambar ke dictionary
        scene_images[scene_name] = img
        print(f"  [{scene_name}] Dimuat: {img.shape[1]}x{img.shape[0]} piksel")
    else:
        # Menampilkan pesan error
        print(f"  [{scene_name}] TIDAK DITEMUKAN: {filename}")

# Memeriksa jumlah scene yang berhasil dimuat
if len(scene_images) < 2:
    print("[ERROR] Minimal 2 gambar scene diperlukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()

# ============================================================
# 2. Ekstraksi GIST-like Features (Spatial Grid)
# ============================================================

print("\n[INFO] Mengekstrak GIST-like features...")
print("-" * 50)

# Mendefinisikan ukuran grid spatial (4x4)
GRID_ROWS = 4
GRID_COLS = 4

# Mendefinisikan jumlah bin untuk histogram orientasi gradient
N_ORIENT_BINS = 8

# Mendefinisikan jumlah bin untuk histogram warna per channel
N_COLOR_BINS = 16

# Mendefinisikan fungsi untuk menghitung fitur GIST-like per cell
def compute_gist_features(image, grid_rows=4, grid_cols=4, n_orient=8):
    """Menghitung fitur GIST-like: histogram gradient per cell spatial grid."""
    # Mengkonversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Menghitung gradient menggunakan Sobel
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Menghitung magnitude dan arah gradient
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi

    # Mengkonversi arah ke range [0, 360)
    orientation = orientation % 360

    # Mendapatkan dimensi gambar
    h, w = gray.shape

    # Menghitung ukuran setiap cell
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    # Menginisialisasi list untuk fitur GIST
    gist_features = []

    # Menghitung histogram gradient pada setiap cell
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Menghitung koordinat cell
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < grid_rows - 1 else h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < grid_cols - 1 else w

            # Mengambil magnitude dan orientasi pada cell ini
            cell_mag = magnitude[y1:y2, x1:x2].flatten()
            cell_orient = orientation[y1:y2, x1:x2].flatten()

            # Menghitung histogram orientasi tertimbang magnitude
            hist, _ = np.histogram(cell_orient, bins=n_orient,
                                   range=(0, 360), weights=cell_mag)

            # Menormalisasi histogram
            hist_norm = hist / (hist.sum() + 1e-7)

            # Menambahkan histogram ke fitur GIST
            gist_features.extend(hist_norm)

    # Mengembalikan fitur GIST sebagai numpy array
    return np.array(gist_features)

# Menghitung fitur GIST untuk setiap scene
scene_gist = {}

for scene_name, img in scene_images.items():
    # Menghitung fitur GIST
    gist = compute_gist_features(img, GRID_ROWS, GRID_COLS, N_ORIENT_BINS)

    # Menyimpan fitur GIST
    scene_gist[scene_name] = gist

    # Menampilkan info fitur
    print(f"  [{scene_name}] GIST features: {len(gist)} dimensi "
          f"({GRID_ROWS}x{GRID_COLS} cells x {N_ORIENT_BINS} bins)")

# ============================================================
# 3. Ekstraksi Fitur Spatial (Color + Edge per Cell)
# ============================================================

print("\n[INFO] Mengekstrak fitur spatial (color + edge per cell)...")
print("-" * 50)

# Mendefinisikan fungsi untuk menghitung scene descriptor lengkap
def compute_scene_descriptor(image, grid_rows=4, grid_cols=4,
                              n_color_bins=16, n_orient_bins=8):
    """Menghitung scene descriptor: color histogram + edge histogram + gradient per cell."""
    # Mengkonversi gambar ke HSV untuk color histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mengkonversi gambar ke grayscale untuk edge dan gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Menghitung tepi menggunakan Canny
    edges = cv2.Canny(gray, 50, 150)

    # Menghitung gradient
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi % 360

    # Mendapatkan dimensi gambar
    h, w = gray.shape

    # Menghitung ukuran setiap cell
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    # Menginisialisasi list untuk fitur gabungan
    all_features = []

    # Menghitung fitur per cell
    cell_features_list = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            # Menghitung koordinat cell
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < grid_rows - 1 else h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < grid_cols - 1 else w

            # --- Color histogram pada cell (H dan S channel) ---

            # Mengambil region HSV pada cell
            cell_hsv = hsv[y1:y2, x1:x2]

            # Menghitung histogram Hue
            hist_h, _ = np.histogram(cell_hsv[:, :, 0].flatten(),
                                     bins=n_color_bins, range=(0, 180))

            # Menormalisasi histogram Hue
            hist_h = hist_h / (hist_h.sum() + 1e-7)

            # Menghitung histogram Saturation
            hist_s, _ = np.histogram(cell_hsv[:, :, 1].flatten(),
                                     bins=n_color_bins, range=(0, 256))

            # Menormalisasi histogram Saturation
            hist_s = hist_s / (hist_s.sum() + 1e-7)

            # --- Edge density pada cell ---

            # Menghitung kepadatan tepi pada cell
            cell_edges = edges[y1:y2, x1:x2]
            edge_density = np.mean(cell_edges) / 255.0

            # --- Gradient histogram pada cell ---

            # Mengambil gradient pada cell
            cell_mag = magnitude[y1:y2, x1:x2].flatten()
            cell_orient = orientation[y1:y2, x1:x2].flatten()

            # Menghitung histogram orientasi
            hist_orient, _ = np.histogram(cell_orient, bins=n_orient_bins,
                                          range=(0, 360), weights=cell_mag)

            # Menormalisasi histogram orientasi
            hist_orient = hist_orient / (hist_orient.sum() + 1e-7)

            # Menggabungkan semua fitur cell
            cell_feat = np.concatenate([hist_h, hist_s, [edge_density], hist_orient])

            # Menambahkan fitur cell ke list
            all_features.extend(cell_feat)
            cell_features_list.append(cell_feat)

    # Mengembalikan descriptor dan fitur per cell
    return np.array(all_features), cell_features_list

# Menghitung scene descriptor untuk setiap scene
scene_descriptors = {}
scene_cell_features = {}

for scene_name, img in scene_images.items():
    # Menghitung scene descriptor
    desc, cell_feats = compute_scene_descriptor(
        img, GRID_ROWS, GRID_COLS, N_COLOR_BINS, N_ORIENT_BINS
    )

    # Menyimpan descriptor
    scene_descriptors[scene_name] = desc
    scene_cell_features[scene_name] = cell_feats

    # Menampilkan info descriptor
    feat_per_cell = len(cell_feats[0]) if cell_feats else 0
    print(f"  [{scene_name}] Descriptor: {len(desc)} dim "
          f"({GRID_ROWS*GRID_COLS} cells x {feat_per_cell} feat/cell)")

# ============================================================
# 4. Visualisasi Fitur Spatial
# ============================================================

print("\n[INFO] Menyimpan visualisasi fitur spatial...")

# Menentukan jumlah scene
n_scenes = len(scene_images)

# Membuat figure untuk visualisasi fitur spatial
fig, axes = plt.subplots(n_scenes, 4, figsize=(20, 5 * n_scenes))

# Memastikan axes 2D
if n_scenes == 1:
    axes = axes.reshape(1, -1)

# Menampilkan fitur per scene
for row, (scene_name, img) in enumerate(scene_images.items()):
    # Mengkonversi gambar ke RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mengkonversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Menghitung tepi Canny
    edges = cv2.Canny(gray, 50, 150)

    # --- Kolom 1: Gambar asli dengan grid overlay ---

    # Membuat salinan gambar untuk grid
    img_grid = img.copy()
    h, w = gray.shape
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS

    # Menggambar garis grid horizontal
    for r in range(1, GRID_ROWS):
        cv2.line(img_grid, (0, r * cell_h), (w, r * cell_h), (0, 255, 0), 2)

    # Menggambar garis grid vertikal
    for c in range(1, GRID_COLS):
        cv2.line(img_grid, (c * cell_w, 0), (c * cell_w, h), (0, 255, 0), 2)

    # Mengkonversi ke RGB
    img_grid_rgb = cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB)

    # Menampilkan gambar dengan grid
    axes[row, 0].imshow(img_grid_rgb)
    axes[row, 0].set_title(f"{scene_name}\n{GRID_ROWS}x{GRID_COLS} Grid", fontsize=10, fontweight="bold")
    axes[row, 0].axis("off")

    # --- Kolom 2: Edge density per cell ---

    # Menghitung edge density map
    edge_density_map = np.zeros((GRID_ROWS, GRID_COLS))

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # Menghitung koordinat cell
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < GRID_ROWS - 1 else h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < GRID_COLS - 1 else w

            # Menghitung edge density
            cell_edges = edges[y1:y2, x1:x2]
            edge_density_map[r, c] = np.mean(cell_edges) / 255.0

    # Menampilkan edge density map
    im = axes[row, 1].imshow(edge_density_map, cmap="hot", interpolation="nearest",
                              vmin=0, vmax=0.5)
    axes[row, 1].set_title("Edge Density per Cell", fontsize=10, fontweight="bold")

    # Menambahkan nilai pada setiap cell
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            axes[row, 1].text(c, r, f"{edge_density_map[r, c]:.2f}",
                             ha="center", va="center", fontsize=7, color="white")

    # --- Kolom 3: Dominant hue per cell ---

    # Menghitung dominant hue map
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_map = np.zeros((GRID_ROWS, GRID_COLS))

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # Menghitung koordinat cell
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < GRID_ROWS - 1 else h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < GRID_COLS - 1 else w

            # Menghitung rata-rata Hue pada cell
            cell_hue = hsv_img[y1:y2, x1:x2, 0]
            hue_map[r, c] = np.mean(cell_hue)

    # Menampilkan hue map
    axes[row, 2].imshow(hue_map, cmap="hsv", interpolation="nearest",
                        vmin=0, vmax=180)
    axes[row, 2].set_title("Dominant Hue per Cell", fontsize=10, fontweight="bold")

    # Menambahkan nilai pada setiap cell
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            axes[row, 2].text(c, r, f"{hue_map[r, c]:.0f}",
                             ha="center", va="center", fontsize=7,
                             color="black", fontweight="bold")

    # --- Kolom 4: Gradient magnitude per cell ---

    # Menghitung gradient
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)

    # Menghitung gradient magnitude map
    grad_map = np.zeros((GRID_ROWS, GRID_COLS))

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # Menghitung koordinat cell
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < GRID_ROWS - 1 else h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < GRID_COLS - 1 else w

            # Menghitung rata-rata gradient magnitude pada cell
            grad_map[r, c] = np.mean(magnitude[y1:y2, x1:x2])

    # Menampilkan gradient map
    axes[row, 3].imshow(grad_map, cmap="viridis", interpolation="nearest")
    axes[row, 3].set_title("Gradient Magnitude per Cell", fontsize=10, fontweight="bold")

    # Menambahkan nilai pada setiap cell
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            axes[row, 3].text(c, r, f"{grad_map[r, c]:.0f}",
                             ha="center", va="center", fontsize=7, color="white")

# Menambahkan judul utama figure
plt.suptitle("Percobaan 13: Fitur Spatial per Scene",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi fitur spatial
output_path_1 = os.path.join(OUTPUT_DIR, "13_scene_features.png")
plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_1}")

# Menutup figure
plt.close()

# ============================================================
# 5. Perbandingan Scene Descriptors
# ============================================================

print("\n[INFO] Membandingkan scene descriptors...")
print("-" * 50)

# Mendefinisikan nama scene yang tersedia
scene_names = list(scene_descriptors.keys())
n_total = len(scene_names)

# Menghitung matriks jarak antar scene descriptor
dist_spatial = np.zeros((n_total, n_total))

for i in range(n_total):
    for j in range(n_total):
        # Menghitung jarak Euclidean antar descriptor
        dist_spatial[i, j] = np.linalg.norm(
            scene_descriptors[scene_names[i]] - scene_descriptors[scene_names[j]]
        )

# --- Menghitung color histogram global sebagai pembanding ---

# Mendefinisikan fungsi untuk menghitung color histogram global
def compute_color_histogram(image, n_bins=64):
    """Menghitung histogram warna global dalam HSV."""
    # Mengkonversi ke HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Menghitung histogram H dan S
    hist_h = cv2.calcHist([hsv], [0], None, [n_bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [n_bins], [0, 256]).flatten()

    # Menormalisasi histogram
    hist_h = hist_h / (hist_h.sum() + 1e-7)
    hist_s = hist_s / (hist_s.sum() + 1e-7)

    # Menggabungkan histogram
    return np.concatenate([hist_h, hist_s])

# Menghitung color histogram global untuk setiap scene
scene_color_hist = {}

for scene_name, img in scene_images.items():
    # Menghitung histogram warna global
    color_hist = compute_color_histogram(img)

    # Menyimpan histogram
    scene_color_hist[scene_name] = color_hist

    # Menampilkan info histogram
    print(f"  [{scene_name}] Color histogram: {len(color_hist)} dim")

# Menghitung matriks jarak color histogram
dist_color = np.zeros((n_total, n_total))

for i in range(n_total):
    for j in range(n_total):
        # Menghitung jarak Euclidean antar color histogram
        dist_color[i, j] = np.linalg.norm(
            scene_color_hist[scene_names[i]] - scene_color_hist[scene_names[j]]
        )

# Menampilkan matriks jarak
print("\n  Matriks Jarak - Spatial Descriptor:")
print(f"  {'':>8s}", end="")
for name in scene_names:
    print(f"  {name:>8s}", end="")
print()
for i, name in enumerate(scene_names):
    print(f"  {name:>8s}", end="")
    for j in range(n_total):
        print(f"  {dist_spatial[i, j]:8.4f}", end="")
    print()

print("\n  Matriks Jarak - Color Histogram:")
print(f"  {'':>8s}", end="")
for name in scene_names:
    print(f"  {name:>8s}", end="")
print()
for i, name in enumerate(scene_names):
    print(f"  {name:>8s}", end="")
    for j in range(n_total):
        print(f"  {dist_color[i, j]:8.4f}", end="")
    print()

# ============================================================
# 6. Visualisasi Spatial Pyramid dan Descriptor
# ============================================================

print("\n[INFO] Menyimpan visualisasi spatial descriptors...")

# Membuat figure untuk perbandingan descriptor
fig, axes = plt.subplots(2, n_total, figsize=(6 * n_total, 8))

# Memastikan axes 2D
if n_total == 1:
    axes = axes.reshape(-1, 1)

# Menampilkan descriptor per scene
for col, scene_name in enumerate(scene_names):
    # --- Baris 1: Gambar dan spatial descriptor ---

    # Mengambil gambar scene
    img_rgb = cv2.cvtColor(scene_images[scene_name], cv2.COLOR_BGR2RGB)

    # Menampilkan gambar
    axes[0, col].imshow(img_rgb)
    axes[0, col].set_title(f"Scene: {scene_name}", fontsize=11, fontweight="bold")
    axes[0, col].axis("off")

    # --- Baris 2: Spatial descriptor sebagai bar chart ---

    # Mengambil scene descriptor
    desc = scene_descriptors[scene_name]

    # Menampilkan descriptor sebagai bar chart
    axes[1, col].bar(range(len(desc)), desc, color="steelblue", alpha=0.7, width=1.0)
    axes[1, col].set_title(f"Scene Descriptor ({len(desc)} dim)", fontsize=10)
    axes[1, col].set_xlabel("Feature Index", fontsize=9)
    axes[1, col].set_ylabel("Value", fontsize=9)
    axes[1, col].set_xlim(0, len(desc))

# Menambahkan judul utama
plt.suptitle("Percobaan 13: Scene Descriptor per Kategori",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi spatial descriptors
output_path_2 = os.path.join(OUTPUT_DIR, "13_scene_spatial.png")
plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_2}")

# Menutup figure
plt.close()

# ============================================================
# 7. Klasifikasi Scene dan Perbandingan Metode
# ============================================================

print("\n[INFO] Melakukan klasifikasi scene...")
print("-" * 50)

# Melakukan klasifikasi leave-one-out untuk kedua metode
print("\n  Klasifikasi (Spatial Descriptor):")
spatial_results = []

for i, query in enumerate(scene_names):
    # Mencari scene terdekat (selain dirinya sendiri)
    min_dist = float("inf")
    predicted = ""

    for j, ref in enumerate(scene_names):
        if i == j:
            continue
        if dist_spatial[i, j] < min_dist:
            min_dist = dist_spatial[i, j]
            predicted = ref

    # Menyimpan hasil
    spatial_results.append((query, predicted, min_dist))
    print(f"    {query} → nearest: {predicted} (dist: {min_dist:.4f})")

print("\n  Klasifikasi (Color Histogram):")
color_results = []

for i, query in enumerate(scene_names):
    # Mencari scene terdekat
    min_dist = float("inf")
    predicted = ""

    for j, ref in enumerate(scene_names):
        if i == j:
            continue
        if dist_color[i, j] < min_dist:
            min_dist = dist_color[i, j]
            predicted = ref

    # Menyimpan hasil
    color_results.append((query, predicted, min_dist))
    print(f"    {query} → nearest: {predicted} (dist: {min_dist:.4f})")

# ============================================================
# 8. Visualisasi Klasifikasi dan Perbandingan
# ============================================================

print("\n[INFO] Menyimpan visualisasi klasifikasi...")

# Membuat figure untuk hasil klasifikasi
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Subplot 1: Heatmap jarak spatial ---
im1 = axes[0, 0].imshow(dist_spatial, cmap="YlOrRd", interpolation="nearest")
axes[0, 0].set_xticks(range(n_total))
axes[0, 0].set_xticklabels(scene_names)
axes[0, 0].set_yticks(range(n_total))
axes[0, 0].set_yticklabels(scene_names)
axes[0, 0].set_title("Jarak Spatial Descriptor", fontsize=11, fontweight="bold")

# Menambahkan nilai pada heatmap
for i in range(n_total):
    for j in range(n_total):
        axes[0, 0].text(j, i, f"{dist_spatial[i, j]:.3f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if dist_spatial[i, j] > dist_spatial.max() * 0.5 else "black")

# Menambahkan colorbar
plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

# --- Subplot 2: Heatmap jarak color histogram ---
im2 = axes[0, 1].imshow(dist_color, cmap="YlOrRd", interpolation="nearest")
axes[0, 1].set_xticks(range(n_total))
axes[0, 1].set_xticklabels(scene_names)
axes[0, 1].set_yticks(range(n_total))
axes[0, 1].set_yticklabels(scene_names)
axes[0, 1].set_title("Jarak Color Histogram", fontsize=11, fontweight="bold")

# Menambahkan nilai pada heatmap
for i in range(n_total):
    for j in range(n_total):
        axes[0, 1].text(j, i, f"{dist_color[i, j]:.3f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if dist_color[i, j] > dist_color.max() * 0.5 else "black")

# Menambahkan colorbar
plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

# --- Subplot 3: Gambar scene side by side ---
for col_idx, scene_name in enumerate(scene_names):
    # Menghitung wilayah untuk menampilkan gambar mini
    x_start = col_idx / n_total
    x_end = (col_idx + 1) / n_total

    # Menampilkan gambar scene
    img_rgb = cv2.cvtColor(scene_images[scene_name], cv2.COLOR_BGR2RGB)

    # Membuat inset axes
    ax_inset = fig.add_axes([0.05 + col_idx * 0.14, 0.1, 0.12, 0.15])
    ax_inset.imshow(img_rgb)
    ax_inset.set_title(scene_name, fontsize=8)
    ax_inset.axis("off")

# Menampilkan color histogram global
for i, scene_name in enumerate(scene_names):
    hist = scene_color_hist[scene_name]
    half = len(hist) // 2
    axes[1, 0].plot(range(half), hist[:half], label=f"{scene_name} (Hue)", linewidth=1.5)

axes[1, 0].set_title("Color Histogram (Hue)", fontsize=11, fontweight="bold")
axes[1, 0].set_xlabel("Bin", fontsize=9)
axes[1, 0].set_ylabel("Frekuensi", fontsize=9)
axes[1, 0].legend(fontsize=8)

# Menampilkan GIST features sebagai perbandingan
for scene_name in scene_names:
    gist = scene_gist[scene_name]
    axes[1, 1].plot(range(len(gist)), gist, label=scene_name, linewidth=1.0, alpha=0.8)

axes[1, 1].set_title("GIST-like Features", fontsize=11, fontweight="bold")
axes[1, 1].set_xlabel("Feature Index", fontsize=9)
axes[1, 1].set_ylabel("Value", fontsize=9)
axes[1, 1].legend(fontsize=8)

# Menambahkan judul utama
plt.suptitle("Percobaan 13: Klasifikasi Scene - Perbandingan Metode",
             fontsize=13, fontweight="bold")

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi klasifikasi
output_path_3 = os.path.join(OUTPUT_DIR, "13_scene_classification.png")
plt.savefig(output_path_3, dpi=150, bbox_inches="tight")
print(f"[OUTPUT] Disimpan: {output_path_3}")

# Menutup figure
plt.close()

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 13")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. GIST-like Features:")
print("     - Membagi gambar menjadi 4x4 grid spatial")
print("     - Menghitung histogram gradient terorientasi per cell")
print(f"     - Total fitur: {GRID_ROWS*GRID_COLS}x{N_ORIENT_BINS} = "
      f"{GRID_ROWS*GRID_COLS*N_ORIENT_BINS} dimensi")
print("  2. Scene Descriptor (Spatial Features):")
print("     - Color histogram (Hue + Saturation) per cell")
print("     - Edge density (kepadatan tepi) per cell")
print("     - Gradient orientation histogram per cell")
print("     - Menangkap layout spasial scene")
print("  3. Color Histogram Global:")
print("     - cv2.calcHist(): histogram HSV global")
print("     - Tidak menangkap informasi spasial")
print("  4. Nearest-Neighbor Classification:")
print("     - Euclidean distance pada descriptor")
print("     - Spatial features menangkap layout scene lebih baik")
print(f"\nHasil klasifikasi:")
print("  Spatial Descriptor:")
for q, p, d in spatial_results:
    print(f"    {q} → {p} (jarak: {d:.4f})")
print("  Color Histogram:")
for q, p, d in color_results:
    print(f"    {q} → {p} (jarak: {d:.4f})")
print("=" * 60)
