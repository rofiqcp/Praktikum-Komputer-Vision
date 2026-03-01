"""
==========================================================
PERCOBAAN 6: MESH PROCESSING DAN SIMPLIFICATION
Mempelajari teknik pengolahan mesh 3D: simplifikasi
(decimation), smoothing, analisis properti mesh.

Fungsi utama:
- open3d simplify_quadric_decimation()
- open3d filter_smooth_laplacian()
- open3d compute_vertex_normals()
==========================================================
"""

# Mengimpor library numpy untuk operasi numerik
import numpy as np

# Mengimpor library os untuk operasi sistem file
import os

# Mengimpor library matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# Mengimpor Axes3D untuk plot 3D
from mpl_toolkits.mplot3d import Axes3D

# Mengimpor art3d untuk polygon 3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Mengimpor Delaunay dari scipy
from scipy.spatial import Delaunay, KDTree

# Mencoba mengimpor Open3D, jika gagal gunakan fallback
try:
    # Mengimpor library Open3D
    import open3d as o3d
    # Menandai ketersediaan Open3D
    HAS_OPEN3D = True
    print("[INFO] Open3D berhasil diimpor")
except ImportError:
    # Menandai bahwa Open3D tidak tersedia
    HAS_OPEN3D = False
    print("[INFO] Open3D tidak tersedia, menggunakan fallback NumPy+SciPy")

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
print("PERCOBAAN 6: MESH PROCESSING DAN SIMPLIFICATION")
print("=" * 60)
print()


def load_ply_manual(filepath):
    """
    Memuat file PLY secara manual tanpa Open3D.
    """
    points = []
    colors = []
    normals = []
    has_colors = False
    has_normals = False
    num_vertices = 0

    with open(filepath, 'r') as f:
        in_header = True
        for line in f:
            line = line.strip()
            if in_header:
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                elif line.startswith("property"):
                    if "red" in line:
                        has_colors = True
                    if "nx" in line:
                        has_normals = True
                elif line == "end_header":
                    in_header = False
                continue
            if num_vertices > 0:
                vals = line.split()
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
                idx = 3
                if has_colors:
                    colors.append([int(vals[idx]), int(vals[idx+1]), int(vals[idx+2])])
                    idx += 3
                if has_normals:
                    normals.append([float(vals[idx]), float(vals[idx+1]), float(vals[idx+2])])
                num_vertices -= 1

    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8) if colors else None
    normals = np.array(normals, dtype=np.float64) if normals else None
    return points, colors, normals


def create_mesh_from_point_cloud(points):
    """
    Membuat mesh dari point cloud menggunakan Delaunay triangulation.
    """
    # Menghitung PCA untuk proyeksi 2D
    centroid = np.mean(points, axis=0)
    cov = (points - centroid).T @ (points - centroid)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Memproyeksikan ke 2D
    proj_2d = (points - centroid) @ eigvecs[:, 1:]

    # Melakukan Delaunay triangulation
    tri = Delaunay(proj_2d)
    triangles = tri.simplices

    # Memfilter segitiga dengan sisi terlalu panjang
    tree = KDTree(points)
    dists, _ = tree.query(points, k=2)
    avg_dist = np.mean(dists[:, 1])
    max_edge = avg_dist * 3.0

    filtered = []
    for t in triangles:
        e1 = np.linalg.norm(points[t[0]] - points[t[1]])
        e2 = np.linalg.norm(points[t[1]] - points[t[2]])
        e3 = np.linalg.norm(points[t[2]] - points[t[0]])
        if e1 < max_edge and e2 < max_edge and e3 < max_edge:
            filtered.append(t)

    return np.array(filtered) if filtered else triangles


def simplify_mesh_manual(vertices, triangles, target_ratio):
    """
    Simplifikasi mesh sederhana dengan menghapus segitiga secara acak.
    Ini adalah pendekatan sederhana (vertex clustering) sebagai fallback.
    """
    # Menentukan jumlah target segitiga
    target_count = max(int(len(triangles) * target_ratio), 10)

    # Menghitung resolusi grid untuk vertex clustering
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_range = bbox_max - bbox_min

    # Menghitung ukuran voxel berdasarkan rasio target
    n_divisions = max(int((target_count / len(triangles)) ** (1/3) * 50), 5)
    voxel_size = bbox_range / n_divisions

    # Membuat grid key untuk setiap vertex
    grid_keys = np.floor((vertices - bbox_min) / (voxel_size + 1e-10)).astype(int)

    # Membuat mapping dari grid key ke cluster
    cluster_map = {}
    cluster_vertices = []
    vertex_to_cluster = np.zeros(len(vertices), dtype=int)

    # Mengiterasi setiap vertex untuk clustering
    for i in range(len(vertices)):
        key = tuple(grid_keys[i])
        if key not in cluster_map:
            cluster_map[key] = len(cluster_vertices)
            cluster_vertices.append([])
        cluster_vertices[cluster_map[key]].append(i)
        vertex_to_cluster[i] = cluster_map[key]

    # Menghitung centroid setiap cluster
    new_vertices = np.zeros((len(cluster_vertices), 3))
    for idx, cluster in enumerate(cluster_vertices):
        new_vertices[idx] = np.mean(vertices[cluster], axis=0)

    # Memetakan segitiga lama ke vertex baru
    new_triangles = []
    seen = set()
    for tri in triangles:
        new_tri = (vertex_to_cluster[tri[0]], vertex_to_cluster[tri[1]], vertex_to_cluster[tri[2]])
        if new_tri[0] != new_tri[1] and new_tri[1] != new_tri[2] and new_tri[0] != new_tri[2]:
            sorted_tri = tuple(sorted(new_tri))
            if sorted_tri not in seen:
                seen.add(sorted_tri)
                new_triangles.append(new_tri)

    # Mengkonversi ke numpy array
    new_triangles = np.array(new_triangles) if new_triangles else np.zeros((0, 3), dtype=int)

    # Mengembalikan mesh yang disimplifikasi
    return new_vertices, new_triangles


def laplacian_smooth_manual(vertices, triangles, iterations=5, lamb=0.5):
    """
    Laplacian smoothing sederhana pada mesh.
    """
    # Menyalin vertices agar tidak mengubah data asli
    smoothed = vertices.copy()

    # Membangun daftar adjacency dari segitiga
    adjacency = {i: set() for i in range(len(vertices))}
    for tri in triangles:
        adjacency[tri[0]].update([tri[1], tri[2]])
        adjacency[tri[1]].update([tri[0], tri[2]])
        adjacency[tri[2]].update([tri[0], tri[1]])

    # Melakukan iterasi smoothing
    for it in range(iterations):
        new_pos = smoothed.copy()
        for i in range(len(smoothed)):
            neighbors = list(adjacency[i])
            if len(neighbors) > 0:
                # Menghitung rata-rata posisi tetangga
                avg = np.mean(smoothed[neighbors], axis=0)
                # Menggerakkan vertex ke arah rata-rata
                new_pos[i] = smoothed[i] + lamb * (avg - smoothed[i])
        smoothed = new_pos

    # Mengembalikan vertices yang sudah di-smooth
    return smoothed


def taubin_smooth_manual(vertices, triangles, iterations=5, lamb=0.5, mu=-0.53):
    """
    Taubin smoothing: Laplacian bolak-balik dengan lambda dan mu.
    Mengurangi shrinkage dibanding Laplacian biasa.
    """
    # Menyalin vertices
    smoothed = vertices.copy()

    # Membangun daftar adjacency
    adjacency = {i: set() for i in range(len(vertices))}
    for tri in triangles:
        adjacency[tri[0]].update([tri[1], tri[2]])
        adjacency[tri[1]].update([tri[0], tri[2]])
        adjacency[tri[2]].update([tri[0], tri[1]])

    # Melakukan iterasi Taubin smoothing
    for it in range(iterations):
        # Langkah 1: Laplacian dengan lambda (shrink)
        new_pos = smoothed.copy()
        for i in range(len(smoothed)):
            neighbors = list(adjacency[i])
            if len(neighbors) > 0:
                avg = np.mean(smoothed[neighbors], axis=0)
                new_pos[i] = smoothed[i] + lamb * (avg - smoothed[i])
        smoothed = new_pos

        # Langkah 2: Laplacian dengan mu (inflate)
        new_pos = smoothed.copy()
        for i in range(len(smoothed)):
            neighbors = list(adjacency[i])
            if len(neighbors) > 0:
                avg = np.mean(smoothed[neighbors], axis=0)
                new_pos[i] = smoothed[i] + mu * (avg - smoothed[i])
        smoothed = new_pos

    # Mengembalikan vertices yang sudah di-smooth
    return smoothed


def compute_normals_manual(vertices, triangles):
    """
    Menghitung vertex normals dari mesh.
    """
    # Menginisialisasi array normal
    vertex_normals = np.zeros_like(vertices)

    # Mengiterasi setiap segitiga
    for tri in triangles:
        # Mengambil tiga titik segitiga
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        # Menghitung normal segitiga dengan cross product
        normal = np.cross(v1 - v0, v2 - v0)
        # Mengakumulasi normal ke setiap vertex
        vertex_normals[tri[0]] += normal
        vertex_normals[tri[1]] += normal
        vertex_normals[tri[2]] += normal

    # Menormalisasi setiap vertex normal
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / (norms + 1e-10)

    # Mengembalikan normal
    return vertex_normals


def visualize_mesh_matplotlib(vertices, triangles, title="Mesh", ax=None, color='steelblue'):
    """
    Memvisualisasikan mesh 3D dengan matplotlib.
    """
    # Membuat axes jika belum tersedia
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    # Membatasi jumlah segitiga yang ditampilkan
    max_tris = 3000
    display_tris = triangles[:max_tris] if len(triangles) > max_tris else triangles

    # Membuat koleksi polygon
    verts = vertices[display_tris]
    mesh_col = Poly3DCollection(verts, alpha=0.6)
    mesh_col.set_facecolor(color)
    mesh_col.set_edgecolor('gray')
    mesh_col.set_linewidth(0.1)

    # Menambahkan ke axes
    ax.add_collection3d(mesh_col)

    # Mengatur batas axes
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z', fontsize=7)

    # Mengembalikan axes
    return ax


# ========================================================
# 1. MEMUAT/MEMBUAT MESH
# ========================================================
print("=" * 60)
print("1. MEMUAT/MEMBUAT MESH")
print("=" * 60)

# Menentukan path file point cloud
ply_path = os.path.join(IMAGE_DIR, "bunny_point_cloud.ply")

# Variabel untuk mesh
mesh_vertices = None
mesh_triangles = None

# Mengecek ketersediaan Open3D
if HAS_OPEN3D:
    # Memuat point cloud dengan Open3D
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"  Dimuat: {len(pcd.points)} titik dari PLY")

    # Mengestimasi normal jika belum ada
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Membuat mesh dengan Poisson reconstruction
    print("  Membuat mesh dengan Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)

    # Mengekstrak data mesh
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_triangles = np.asarray(mesh.triangles)

    # Menyimpan referensi mesh Open3D
    mesh_o3d = mesh
else:
    # Memuat point cloud secara manual
    points, colors, normals = load_ply_manual(ply_path)
    print(f"  Dimuat: {len(points)} titik dari PLY")

    # Membuat mesh dengan Delaunay
    print("  Membuat mesh dengan Delaunay triangulation...")
    mesh_triangles = create_mesh_from_point_cloud(points)
    mesh_vertices = points.copy()

# Mencetak properti mesh awal
print(f"  Vertices: {len(mesh_vertices)}")
print(f"  Triangles: {len(mesh_triangles)}")
print()

# ========================================================
# 2. PROPERTI MESH
# ========================================================
print("=" * 60)
print("2. PROPERTI MESH")
print("=" * 60)

# Menghitung bounding box
bbox_min = mesh_vertices.min(axis=0)
bbox_max = mesh_vertices.max(axis=0)
bbox_size = bbox_max - bbox_min
print(f"  Bounding Box Min:  [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]")
print(f"  Bounding Box Max:  [{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]")
print(f"  Bounding Box Size: [{bbox_size[0]:.4f}, {bbox_size[1]:.4f}, {bbox_size[2]:.4f}]")

# Menghitung area permukaan total
total_area = 0.0
areas = []
for tri in mesh_triangles:
    v0, v1, v2 = mesh_vertices[tri[0]], mesh_vertices[tri[1]], mesh_vertices[tri[2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    areas.append(area)
    total_area += area

# Mencetak statistik area
print(f"  Total area permukaan: {total_area:.4f}")
print(f"  Area rata-rata segitiga: {np.mean(areas):.6f}")
print(f"  Area min/max: {min(areas):.8f} / {max(areas):.6f}")

# Menghitung vertex normals
normals = compute_normals_manual(mesh_vertices, mesh_triangles)
print(f"  Vertex normals dihitung: {len(normals)}")
print()

# ========================================================
# 3. MESH SIMPLIFICATION
# ========================================================
print("=" * 60)
print("3. MESH SIMPLIFICATION (DECIMATION)")
print("=" * 60)

# Menentukan level simplifikasi
simplification_levels = [0.5, 0.25, 0.1]
simplified_meshes = {}

# Menghitung jumlah segitiga asli
original_tri_count = len(mesh_triangles)
print(f"  Original: {original_tri_count} triangles")

# Mengiterasi setiap level simplifikasi
for ratio in simplification_levels:
    # Menghitung target jumlah segitiga
    target = int(original_tri_count * ratio)
    label = f"{int((1-ratio)*100)}%"

    # Mengecek ketersediaan Open3D
    if HAS_OPEN3D:
        # Menggunakan quadric decimation
        simplified = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target)
        s_verts = np.asarray(simplified.vertices)
        s_tris = np.asarray(simplified.triangles)
    else:
        # Menggunakan vertex clustering manual
        s_verts, s_tris = simplify_mesh_manual(mesh_vertices, mesh_triangles, ratio)

    # Menyimpan hasil simplifikasi
    simplified_meshes[label] = (s_verts, s_tris)

    # Mencetak informasi simplifikasi
    reduction = (1 - len(s_tris) / original_tri_count) * 100
    print(f"  Reduksi {label}: {len(s_tris)} triangles ({reduction:.1f}% dikurangi)")

print()

# ========================================================
# 4. LAPLACIAN SMOOTHING
# ========================================================
print("=" * 60)
print("4. LAPLACIAN SMOOTHING")
print("=" * 60)

# Menentukan variasi iterasi Laplacian
laplacian_iterations = [1, 5, 20]
smoothed_meshes = {}

# Mengiterasi setiap jumlah iterasi
for n_iter in laplacian_iterations:
    # Mengecek ketersediaan Open3D
    if HAS_OPEN3D:
        # Menggunakan Laplacian smoothing Open3D
        smoothed = mesh_o3d.__copy__()
        smoothed = smoothed.filter_smooth_laplacian(number_of_iterations=n_iter)
        sm_verts = np.asarray(smoothed.vertices)
        sm_tris = np.asarray(smoothed.triangles)
    else:
        # Menggunakan Laplacian smoothing manual
        sm_verts = laplacian_smooth_manual(mesh_vertices, mesh_triangles, iterations=n_iter)
        sm_tris = mesh_triangles.copy()

    # Menyimpan hasil smoothing
    smoothed_meshes[f"Laplacian {n_iter}x"] = (sm_verts, sm_tris)

    # Menghitung perubahan rata-rata posisi vertex
    displacement = np.mean(np.linalg.norm(sm_verts[:len(mesh_vertices)] - mesh_vertices[:len(sm_verts)], axis=1))
    print(f"  Laplacian {n_iter:2d} iterasi: avg displacement = {displacement:.6f}")

print()

# ========================================================
# 5. TAUBIN SMOOTHING
# ========================================================
print("=" * 60)
print("5. TAUBIN SMOOTHING")
print("=" * 60)

# Menentukan variasi iterasi Taubin
taubin_iterations = [5, 20]
taubin_meshes = {}

# Mengiterasi setiap jumlah iterasi
for n_iter in taubin_iterations:
    # Mengecek ketersediaan Open3D
    if HAS_OPEN3D:
        # Menggunakan Taubin smoothing Open3D
        taubin = mesh_o3d.__copy__()
        taubin = taubin.filter_smooth_taubin(number_of_iterations=n_iter)
        tb_verts = np.asarray(taubin.vertices)
        tb_tris = np.asarray(taubin.triangles)
    else:
        # Menggunakan Taubin smoothing manual
        tb_verts = taubin_smooth_manual(mesh_vertices, mesh_triangles, iterations=n_iter)
        tb_tris = mesh_triangles.copy()

    # Menyimpan hasil Taubin smoothing
    taubin_meshes[f"Taubin {n_iter}x"] = (tb_verts, tb_tris)

    # Menghitung perubahan posisi vertex
    displacement = np.mean(np.linalg.norm(tb_verts[:len(mesh_vertices)] - mesh_vertices[:len(tb_verts)], axis=1))
    print(f"  Taubin {n_iter:2d} iterasi: avg displacement = {displacement:.6f}")

print()

# ========================================================
# 6. VISUALISASI MESH NORMALS
# ========================================================
print("=" * 60)
print("6. VISUALISASI MESH NORMALS")
print("=" * 60)

# Membuat figure untuk visualisasi normals
fig_normals = plt.figure(figsize=(10, 8))
ax_norm = fig_normals.add_subplot(111, projection='3d')

# Mensubsampling untuk visualisasi yang bersih
sample_step = max(len(mesh_vertices) // 200, 1)
sampled_pts = mesh_vertices[::sample_step]
sampled_norms = normals[::sample_step]

# Menampilkan titik-titik mesh
ax_norm.scatter(sampled_pts[:, 0], sampled_pts[:, 1], sampled_pts[:, 2],
                c='steelblue', s=5, alpha=0.6)

# Menentukan panjang panah normal
arrow_length = bbox_size.max() * 0.05

# Menampilkan normal sebagai panah (quiver)
ax_norm.quiver(sampled_pts[:, 0], sampled_pts[:, 1], sampled_pts[:, 2],
               sampled_norms[:, 0] * arrow_length,
               sampled_norms[:, 1] * arrow_length,
               sampled_norms[:, 2] * arrow_length,
               color='red', alpha=0.6, arrow_length_ratio=0.3, linewidth=0.8)

# Mengatur judul dan label
ax_norm.set_title('Mesh dengan Vertex Normals', fontsize=12)
ax_norm.set_xlabel('X')
ax_norm.set_ylabel('Y')
ax_norm.set_zlabel('Z')

# Menyimpan visualisasi normals
normals_path = os.path.join(OUTPUT_DIR, "06_mesh_normals.png")
plt.savefig(normals_path, dpi=150, bbox_inches='tight')
print(f"  Disimpan: {normals_path}")
plt.close()
print()

# ========================================================
# 7. PERBANDINGAN VISUAL
# ========================================================
print("=" * 60)
print("7. PERBANDINGAN ORIGINAL VS SIMPLIFIED VS SMOOTHED")
print("=" * 60)

# Membuat figure perbandingan simplifikasi
fig_simp = plt.figure(figsize=(16, 5))

# Subplot 1: Original mesh
ax_orig = fig_simp.add_subplot(141, projection='3d')
visualize_mesh_matplotlib(mesh_vertices, mesh_triangles,
                          title=f'Original\n({len(mesh_triangles)} tris)', ax=ax_orig)

# Subplot 2-4: Simplified meshes
for idx, (label, (s_v, s_t)) in enumerate(simplified_meshes.items()):
    ax_s = fig_simp.add_subplot(1, 4, idx + 2, projection='3d')
    # Menentukan warna berdasarkan level
    colors_list = ['coral', 'orange', 'gold']
    visualize_mesh_matplotlib(s_v, s_t,
                              title=f'Reduksi {label}\n({len(s_t)} tris)',
                              ax=ax_s, color=colors_list[idx % 3])

# Mengatur layout
plt.suptitle('Mesh Simplification (Decimation)', fontsize=13)
plt.tight_layout()

# Menyimpan perbandingan simplifikasi
simp_path = os.path.join(OUTPUT_DIR, "06_mesh_simplification.png")
plt.savefig(simp_path, dpi=150, bbox_inches='tight')
print(f"  Disimpan: {simp_path}")
plt.close()

# Membuat figure perbandingan smoothing
fig_smooth = plt.figure(figsize=(16, 5))

# Subplot 1: Original
ax_sm0 = fig_smooth.add_subplot(141, projection='3d')
visualize_mesh_matplotlib(mesh_vertices, mesh_triangles,
                          title='Original', ax=ax_sm0)

# Subplot 2-4: Smoothed meshes (mix Laplacian dan Taubin)
all_smooth = {**smoothed_meshes, **taubin_meshes}
# Mengambil 3 variasi smoothing untuk ditampilkan
smooth_keys = list(all_smooth.keys())[:3]
smooth_colors = ['lightgreen', 'mediumseagreen', 'mediumpurple']

for idx, key in enumerate(smooth_keys):
    ax_sm = fig_smooth.add_subplot(1, 4, idx + 2, projection='3d')
    sm_v, sm_t = all_smooth[key]
    visualize_mesh_matplotlib(sm_v, sm_t, title=key, ax=ax_sm,
                              color=smooth_colors[idx % 3])

# Mengatur layout
plt.suptitle('Mesh Smoothing Comparison', fontsize=13)
plt.tight_layout()

# Menyimpan perbandingan smoothing
smooth_path = os.path.join(OUTPUT_DIR, "06_mesh_smoothing.png")
plt.savefig(smooth_path, dpi=150, bbox_inches='tight')
print(f"  Disimpan: {smooth_path}")
plt.close()
print()

# ========================================================
# 8. TABEL PERBANDINGAN
# ========================================================
print("=" * 60)
print("8. TABEL PERBANDINGAN MESH")
print("=" * 60)

# Mencetak header tabel
print(f"\n  {'Nama Mesh':<25} {'Vertices':<12} {'Triangles':<12} {'Reduksi %':<12}")
print(f"  {'-'*60}")

# Mencetak data original
print(f"  {'Original':<25} {len(mesh_vertices):<12} {len(mesh_triangles):<12} {'0.0%':<12}")

# Mencetak data simplifikasi
for label, (s_v, s_t) in simplified_meshes.items():
    reduction = (1 - len(s_t) / len(mesh_triangles)) * 100
    print(f"  {f'Simplified {label}':<25} {len(s_v):<12} {len(s_t):<12} {f'{reduction:.1f}%':<12}")

# Mencetak data smoothing
for label, (sm_v, sm_t) in smoothed_meshes.items():
    print(f"  {label:<25} {len(sm_v):<12} {len(sm_t):<12} {'N/A':<12}")

# Mencetak data Taubin
for label, (tb_v, tb_t) in taubin_meshes.items():
    print(f"  {label:<25} {len(tb_v):<12} {len(tb_t):<12} {'N/A':<12}")

print()

# ========================================================
# 9. MENYIMPAN SEMUA VISUALISASI
# ========================================================
print("=" * 60)
print("9. RINGKASAN OUTPUT")
print("=" * 60)

# Mencetak ringkasan akhir
print(f"\n  Output tersimpan di: {OUTPUT_DIR}")
print(f"  File yang dihasilkan:")
print(f"    - 06_mesh_normals.png")
print(f"    - 06_mesh_simplification.png")
print(f"    - 06_mesh_smoothing.png")
print()

# Mencetak penutup
print("=" * 60)
print("PERCOBAAN 6 SELESAI")
print("=" * 60)
