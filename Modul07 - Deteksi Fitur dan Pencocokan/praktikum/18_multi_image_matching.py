"""
==========================================================================
PERCOBAAN 18: MULTI-IMAGE FEATURE MATCHING
==========================================================================
Program ini melakukan feature matching antar 3+ gambar panorama
(pano_left, pano_center, pano_right). Program membangun match graph
yang menunjukkan gambar mana yang saling cocok, menghitung confidence
setiap pasangan, memvisualisasikan matriks matching sebagai heatmap,
menemukan connected components, dan menentukan urutan matching optimal.

Konsep yang dipelajari:
- Multi-image matching: mencocokkan fitur antar lebih dari 2 gambar
- Match graph: representasi hubungan antar gambar
- Matriks matching: jumlah matches antar setiap pasangan gambar
- Connected components: kelompok gambar yang saling terhubung
- Confidence score: tingkat kepercayaan kecocokan antar pasangan
- Optimal matching order: urutan terbaik untuk menggabungkan gambar

Fungsi utama yang dipelajari:
- cv2.SIFT_create()      : Detektor dan deskriptor SIFT
- cv2.FlannBasedMatcher() : FLANN-based matcher untuk matching cepat
- knnMatch()              : K-Nearest Neighbor matching
- Ratio test              : Penyaringan matches menggunakan rasio Lowe

Hasil: Matriks matching, visualisasi pasangan terbaik, dan match graph
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan fitur
import cv2

# Mengimpor NumPy untuk operasi array dan matriks numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu pemrosesan
import time

# Mengimpor itertools untuk kombinasi pasangan
import itertools

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
print("PERCOBAAN 18: MULTI-IMAGE FEATURE MATCHING")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar-Gambar Panorama dan Tambahan
# ============================================================

# Menampilkan header bagian
print("\n--- 1. Memuat Gambar ---")

# Mendefinisikan daftar gambar yang akan digunakan
daftar_gambar = [
    'pano_left.jpg',
    'pano_center.jpg',
    'pano_right.jpg',
    'scene_left.jpg',
    'scene_right.jpg',
]

# Menyiapkan dictionary untuk menyimpan data gambar
gambar_data = {}

# Membuat detektor SIFT
sift = cv2.SIFT_create()

# Melakukan iterasi untuk memuat setiap gambar
for nama_file in daftar_gambar:
    # Membaca gambar dari file
    img = cv2.imread(os.path.join(IMAGE_DIR, nama_file))

    # Memeriksa apakah gambar berhasil dimuat
    if img is None:
        print(f"[WARNING] {nama_file} tidak ditemukan, dilewati")
        continue

    # Mengkonversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mendeteksi keypoint dan menghitung deskriptor SIFT
    kp, des = sift.detectAndCompute(gray, None)

    # Menyimpan data gambar
    gambar_data[nama_file] = {
        'gambar': img,
        'gray': gray,
        'keypoints': kp,
        'deskriptor': des,
        'jumlah_kp': len(kp)
    }

    # Menampilkan informasi gambar
    print(f"  [OK] {nama_file}: {img.shape}, {len(kp)} keypoints")

# Mendapatkan daftar nama gambar yang berhasil dimuat
nama_gambar = list(gambar_data.keys())

# Menampilkan total gambar yang dimuat
print(f"\n[INFO] Total gambar dimuat: {len(nama_gambar)}")

# Memeriksa apakah ada minimal 3 gambar
if len(nama_gambar) < 3:
    print("[ERROR] Minimal diperlukan 3 gambar! Jalankan download_image.py.")
    exit()

# ============================================================
# 2. Matching Semua Pasangan Gambar
# ============================================================

# Menampilkan header bagian
print("\n--- 2. Matching Semua Pasangan ---")

# Mendefinisikan parameter FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Membuat FLANN-based matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Mendapatkan semua kombinasi pasangan gambar
pasangan = list(itertools.combinations(range(len(nama_gambar)), 2))

# Menampilkan jumlah pasangan
print(f"[INFO] Jumlah pasangan: {len(pasangan)}")

# Menyiapkan matriks matching (jumlah good matches per pasangan)
n = len(nama_gambar)
matriks_match = np.zeros((n, n), dtype=np.int32)

# Menyiapkan matriks confidence
matriks_confidence = np.zeros((n, n), dtype=np.float64)

# Menyiapkan dictionary untuk detail matching setiap pasangan
detail_pasangan = {}

# Menghitung total keypoint antar semua pasangan
total_matched_kp = 0

# Melakukan iterasi untuk setiap pasangan
for idx_a, idx_b in pasangan:
    # Mendapatkan nama gambar A dan B
    nama_a = nama_gambar[idx_a]
    nama_b = nama_gambar[idx_b]

    # Mendapatkan deskriptor kedua gambar
    des_a = gambar_data[nama_a]['deskriptor']
    des_b = gambar_data[nama_b]['deskriptor']

    # Memeriksa apakah kedua deskriptor valid
    if des_a is None or des_b is None:
        print(f"  {nama_a} vs {nama_b}: SKIP (tidak ada deskriptor)")
        continue

    # Mengukur waktu matching
    waktu_mulai = time.time()

    # Melakukan KNN matching dengan k=2
    matches = flann.knnMatch(des_a, des_b, k=2)

    # Menerapkan ratio test Lowe
    good_matches = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n_m = m_pair
            if m.distance < 0.75 * n_m.distance:
                good_matches.append(m)

    # Menghitung waktu matching
    waktu_match = time.time() - waktu_mulai

    # Menghitung jumlah good matches
    jumlah_good = len(good_matches)

    # Menghitung confidence score (normalized)
    min_kp = min(gambar_data[nama_a]['jumlah_kp'], gambar_data[nama_b]['jumlah_kp'])
    confidence = jumlah_good / max(min_kp, 1)

    # Menyimpan hasil ke matriks (simetris)
    matriks_match[idx_a, idx_b] = jumlah_good
    matriks_match[idx_b, idx_a] = jumlah_good
    matriks_confidence[idx_a, idx_b] = confidence
    matriks_confidence[idx_b, idx_a] = confidence

    # Menambahkan ke total matched keypoints
    total_matched_kp += jumlah_good

    # Menyimpan detail pasangan
    detail_pasangan[(idx_a, idx_b)] = {
        'nama_a': nama_a,
        'nama_b': nama_b,
        'good_matches': good_matches,
        'jumlah': jumlah_good,
        'confidence': confidence,
        'waktu': waktu_match
    }

    # Menampilkan hasil matching
    print(f"  {nama_a} vs {nama_b}: {jumlah_good} matches, "
          f"confidence={confidence:.3f}, waktu={waktu_match:.3f}s")

# Menampilkan total matched keypoints
print(f"\n[INFO] Total matched keypoints antar semua pasangan: {total_matched_kp}")

# ============================================================
# 3. Visualisasi Matriks Matching sebagai Heatmap
# ============================================================

# Menampilkan header bagian
print("\n--- 3. Visualisasi Matriks Matching ---")

# Membuat figure dengan 2 subplot (jumlah matches dan confidence)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Heatmap Jumlah Matches ---
# Menampilkan heatmap matriks jumlah matches
im1 = axes[0].imshow(matriks_match, cmap='YlOrRd', aspect='auto')

# Mengatur label sumbu
label_singkat = [n.replace('.jpg', '') for n in nama_gambar]
axes[0].set_xticks(range(n))
axes[0].set_xticklabels(label_singkat, rotation=45, ha='right', fontsize=9)
axes[0].set_yticks(range(n))
axes[0].set_yticklabels(label_singkat, fontsize=9)

# Menambahkan nilai di setiap sel
for i in range(n):
    for j in range(n):
        val = matriks_match[i, j]
        warna_t = 'white' if val > matriks_match.max() * 0.6 else 'black'
        axes[0].text(j, i, str(val), ha='center', va='center',
                     fontsize=9, fontweight='bold', color=warna_t)

# Menambahkan colorbar
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Mengatur judul
axes[0].set_title('Jumlah Good Matches\nper Pasangan', fontsize=12, fontweight='bold')

# --- Heatmap Confidence ---
# Menampilkan heatmap matriks confidence
im2 = axes[1].imshow(matriks_confidence, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Mengatur label sumbu
axes[1].set_xticks(range(n))
axes[1].set_xticklabels(label_singkat, rotation=45, ha='right', fontsize=9)
axes[1].set_yticks(range(n))
axes[1].set_yticklabels(label_singkat, fontsize=9)

# Menambahkan nilai di setiap sel
for i in range(n):
    for j in range(n):
        val = matriks_confidence[i, j]
        warna_t = 'white' if val < 0.3 else 'black'
        axes[1].text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=9, fontweight='bold', color=warna_t)

# Menambahkan colorbar
plt.colorbar(im2, ax=axes[1], shrink=0.8)

# Mengatur judul
axes[1].set_title('Confidence Score\nper Pasangan', fontsize=12, fontweight='bold')

# Menambahkan judul utama
fig.suptitle('Percobaan 18: Matriks Multi-Image Matching', fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan matriks matching
output_path_matrix = os.path.join(OUTPUT_DIR, "18_multi_match_matrix.png")
plt.savefig(output_path_matrix, dpi=150, bbox_inches='tight')
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_matrix}")

# ============================================================
# 4. Visualisasi Best Matches untuk Setiap Pasangan
# ============================================================

# Menampilkan header bagian
print("\n--- 4. Visualisasi Best Matches ---")

# Mengurutkan pasangan berdasarkan jumlah good matches (terbanyak dulu)
pasangan_sorted = sorted(detail_pasangan.items(), key=lambda x: x[1]['jumlah'], reverse=True)

# Mengambil top-3 pasangan terbaik (atau semua jika kurang dari 3)
top_pasangan = pasangan_sorted[:min(3, len(pasangan_sorted))]

# Membuat figure untuk best matches
fig, axes = plt.subplots(len(top_pasangan), 1, figsize=(18, 6 * len(top_pasangan)))

# Menghandle kasus jika hanya 1 pasangan
if len(top_pasangan) == 1:
    axes = [axes]

# Melakukan iterasi untuk setiap top pasangan
for idx, ((idx_a, idx_b), detail) in enumerate(top_pasangan):
    # Mendapatkan data gambar A dan B
    data_a = gambar_data[detail['nama_a']]
    data_b = gambar_data[detail['nama_b']]

    # Menggambar matches menggunakan drawMatches
    img_matches = cv2.drawMatches(
        data_a['gambar'], data_a['keypoints'],
        data_b['gambar'], data_b['keypoints'],
        detail['good_matches'][:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Menampilkan gambar matches
    axes[idx].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(
        f'#{idx+1}: {detail["nama_a"]} ↔ {detail["nama_b"]} | '
        f'Matches: {detail["jumlah"]}, Confidence: {detail["confidence"]:.3f}',
        fontsize=12, fontweight='bold'
    )
    axes[idx].axis('off')

    # Menampilkan informasi pasangan
    print(f"  Top-{idx+1}: {detail['nama_a']} ↔ {detail['nama_b']}: "
          f"{detail['jumlah']} matches, confidence={detail['confidence']:.3f}")

# Menambahkan judul utama
fig.suptitle('Percobaan 18: Top Pasangan Matching', fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi pasangan terbaik
output_path_pairs = os.path.join(OUTPUT_DIR, "18_multi_match_pairs.png")
plt.savefig(output_path_pairs, dpi=150, bbox_inches='tight')
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_pairs}")

# ============================================================
# 5. Match Graph dan Connected Components
# ============================================================

# Menampilkan header bagian
print("\n--- 5. Match Graph dan Connected Components ---")

# Mendefinisikan threshold untuk menganggap dua gambar terhubung
THRESHOLD_MATCHES = 15

# Membangun adjacency list dari match graph
adjacency = {i: set() for i in range(n)}

# Menambahkan edge jika jumlah matches melebihi threshold
for (idx_a, idx_b), detail in detail_pasangan.items():
    if detail['jumlah'] >= THRESHOLD_MATCHES:
        adjacency[idx_a].add(idx_b)
        adjacency[idx_b].add(idx_a)

# Menampilkan adjacency list
print(f"[INFO] Threshold koneksi: {THRESHOLD_MATCHES} matches")
for i in range(n):
    tetangga = [nama_gambar[j].replace('.jpg', '') for j in adjacency[i]]
    print(f"  {nama_gambar[i].replace('.jpg', '')}: terhubung dengan {tetangga}")

# Mencari connected components menggunakan BFS
visited = set()
components = []

# Melakukan iterasi untuk setiap node
for start in range(n):
    # Melewatkan node yang sudah dikunjungi
    if start in visited:
        continue

    # Melakukan BFS dari node ini
    komponen = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        komponen.append(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                queue.append(neighbor)

    # Menyimpan komponen yang ditemukan
    components.append(komponen)

# Menampilkan connected components
print(f"\n[INFO] Jumlah connected components: {len(components)}")
for i, comp in enumerate(components):
    nama_comp = [nama_gambar[j].replace('.jpg', '') for j in comp]
    print(f"  Komponen {i+1}: {nama_comp}")

# Menentukan urutan matching optimal (berdasarkan confidence tertinggi)
print(f"\n[INFO] Urutan matching optimal (Greedy berdasarkan confidence):")

# Menyiapkan list pasangan yang sudah diurutkan
pasangan_urut = sorted(detail_pasangan.items(), key=lambda x: x[1]['confidence'], reverse=True)

# Menampilkan urutan optimal
for rank, ((ia, ib), det) in enumerate(pasangan_urut):
    print(f"  {rank+1}. {det['nama_a']} ↔ {det['nama_b']}: confidence={det['confidence']:.3f}")

# ============================================================
# 6. Visualisasi Match Graph
# ============================================================

# Menampilkan header bagian
print("\n--- 6. Visualisasi Match Graph ---")

# Membuat figure untuk match graph
fig, ax = plt.subplots(figsize=(12, 10))

# Menghitung posisi node dalam lingkaran
sudut_lingkaran = np.linspace(0, 2 * np.pi, n, endpoint=False)
radius = 3.0
posisi_node = {}
for i in range(n):
    posisi_node[i] = (radius * np.cos(sudut_lingkaran[i]),
                       radius * np.sin(sudut_lingkaran[i]))

# Menggambar edge (garis antar node yang terhubung)
for (idx_a, idx_b), detail in detail_pasangan.items():
    # Mendapatkan posisi kedua node
    xa, ya = posisi_node[idx_a]
    xb, yb = posisi_node[idx_b]

    # Menentukan ketebalan garis berdasarkan jumlah matches
    lebar_garis = max(1, min(detail['jumlah'] / 10, 8))

    # Menentukan warna garis berdasarkan confidence
    if detail['confidence'] >= 0.3:
        warna_garis = '#2ecc71'  # hijau
    elif detail['confidence'] >= 0.1:
        warna_garis = '#f39c12'  # kuning
    else:
        warna_garis = '#e74c3c'  # merah

    # Menentukan alpha berdasarkan jumlah matches
    alpha_garis = min(detail['jumlah'] / 50, 1.0) if detail['jumlah'] > 0 else 0.1

    # Menggambar garis
    ax.plot([xa, xb], [ya, yb], color=warna_garis, linewidth=lebar_garis,
            alpha=alpha_garis, zorder=1)

    # Menambahkan label jumlah matches di tengah garis
    mx = (xa + xb) / 2
    my = (ya + yb) / 2
    ax.text(mx, my, str(detail['jumlah']), fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Menentukan warna node berdasarkan connected component
warna_komponen = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

# Menggambar node
for i in range(n):
    # Menentukan warna berdasarkan komponen
    warna_node = '#3498db'
    for c_idx, comp in enumerate(components):
        if i in comp:
            warna_node = warna_komponen[c_idx % len(warna_komponen)]
            break

    # Mendapatkan posisi node
    x, y = posisi_node[i]

    # Menggambar lingkaran node
    circle = plt.Circle((x, y), 0.5, color=warna_node, zorder=2, alpha=0.9)
    ax.add_patch(circle)

    # Menambahkan label nama gambar
    nama_singkat = nama_gambar[i].replace('.jpg', '')
    ax.text(x, y, nama_singkat, ha='center', va='center', fontsize=8,
            fontweight='bold', color='white', zorder=3)

# Menambahkan legenda untuk warna garis
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2ecc71', linewidth=3, label='Confidence ≥ 0.3 (Kuat)'),
    Line2D([0], [0], color='#f39c12', linewidth=3, label='Confidence 0.1-0.3 (Sedang)'),
    Line2D([0], [0], color='#e74c3c', linewidth=3, label='Confidence < 0.1 (Lemah)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Mengatur batas axes
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# Mengatur judul
ax.set_title(f'Match Graph ({len(nama_gambar)} gambar, {len(components)} komponen)\n'
             f'Threshold: {THRESHOLD_MATCHES} matches, Total matched: {total_matched_kp}',
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan match graph
output_path_graph = os.path.join(OUTPUT_DIR, "18_match_graph.png")
plt.savefig(output_path_graph, dpi=150, bbox_inches='tight')
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_graph}")

# ============================================================
# 7. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 18: MULTI-IMAGE FEATURE MATCHING")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan multi-image matching
print("1. Multi-image matching mencocokkan fitur antar semua pasangan")
print("   gambar untuk membangun representasi hubungan antar gambar.")

# Menampilkan tentang matriks matching
print("2. Matriks matching menunjukkan jumlah good matches dan")
print("   confidence score untuk setiap pasangan gambar.")

# Menampilkan tentang connected components
print("3. Connected components mengidentifikasi kelompok gambar")
print("   yang saling terhubung (overlap scene yang sama).")

# Menampilkan tentang match graph
print("4. Match graph memvisualisasikan hubungan antar gambar")
print("   dengan ketebalan garis menunjukkan kekuatan koneksi.")

# Menampilkan tentang urutan optimal
print("5. Urutan matching optimal ditentukan berdasarkan")
print("   confidence tertinggi untuk stitching/rekonstruksi.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 18_multi_match_matrix.png")
print("  - 18_multi_match_pairs.png")
print("  - 18_match_graph.png")

# Menampilkan garis penutup
print("=" * 60)
