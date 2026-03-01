"""
==========================================================
PERCOBAAN 16: MARCHING CUBES
Mempelajari algoritma Marching Cubes untuk mengekstrak
mesh isosurface dari volumetric data (implicit function
atau voxel grid).

Fungsi utama:
- skimage.measure.marching_cubes (atau manual implementation)
- numpy 3D array operations
- matplotlib plot_trisurf()
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

# Mengimpor art3d untuk menampilkan polygon 3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Mencoba mengimpor marching_cubes dari skimage
try:
    # Mengimpor fungsi marching_cubes dari scikit-image
    from skimage.measure import marching_cubes
    # Menandai ketersediaan skimage
    HAS_SKIMAGE = True
    print("[INFO] scikit-image marching_cubes berhasil diimpor")
except ImportError:
    # Menandai bahwa skimage tidak tersedia
    HAS_SKIMAGE = False
    print("[INFO] scikit-image tidak tersedia, menggunakan implementasi manual")

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
    print("[INFO] Open3D tidak tersedia, menggunakan matplotlib saja")

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
print("PERCOBAAN 16: MARCHING CUBES")
print("=" * 60)

# ========================================================
# BAGIAN 1: MEMBUAT 3D SCALAR FIELD (SPHERE)
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 1: Membuat 3D Scalar Field (Sphere)")
print("=" * 60)

# Menentukan resolusi grid 3D
resolusi = 50

# Membuat rentang koordinat dari -1.5 sampai 1.5
lin = np.linspace(-1.5, 1.5, resolusi)

# Membuat meshgrid 3D untuk koordinat x, y, z
X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')

# Mendefinisikan fungsi implisit untuk bola: f = x^2 + y^2 + z^2 - r^2
radius_bola = 1.0

# Menghitung scalar field untuk bola
field_bola = X**2 + Y**2 + Z**2 - radius_bola**2

# Mencetak informasi scalar field
print(f"Resolusi grid: {resolusi}x{resolusi}x{resolusi}")
print(f"Rentang scalar field: [{field_bola.min():.3f}, {field_bola.max():.3f}]")
print(f"Radius bola: {radius_bola}")

# ========================================================
# BAGIAN 2: EKSTRAKSI ISOSURFACE MENGGUNAKAN MARCHING CUBES
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 2: Ekstraksi Isosurface (Marching Cubes)")
print("=" * 60)

def marching_cubes_manual(volume, threshold=0.0, spacing=(1, 1, 1)):
    """Implementasi sederhana marching cubes menggunakan interpolasi tepi."""
    # Mendapatkan dimensi volume
    nx, ny, nz = volume.shape

    # Menyimpan daftar vertex dan face
    vertices = []
    faces = []

    # Membuat array biner: True jika di bawah threshold
    binary = volume <= threshold

    # Iterasi setiap sel dalam grid (kecuali batas)
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # Mengambil 8 sudut kubus
                cube_vals = [
                    binary[i, j, k], binary[i+1, j, k],
                    binary[i+1, j+1, k], binary[i, j+1, k],
                    binary[i, j, k+1], binary[i+1, j, k+1],
                    binary[i+1, j+1, k+1], binary[i, j+1, k+1]
                ]

                # Menghitung jumlah vertex di dalam isosurface
                n_inside = sum(cube_vals)

                # Melewati sel yang sepenuhnya di dalam atau di luar
                if n_inside == 0 or n_inside == 8:
                    continue

                # Menghitung posisi rata-rata vertex yang di dalam
                positions_inside = []
                positions_outside = []

                # Mendefinisikan posisi 8 sudut kubus
                corners = [
                    (i, j, k), (i+1, j, k), (i+1, j+1, k), (i, j+1, k),
                    (i, j, k+1), (i+1, j, k+1), (i+1, j+1, k+1), (i, j+1, k+1)
                ]

                # Memisahkan sudut berdasarkan posisi relatif terhadap threshold
                for idx_c, (ci, cj, ck) in enumerate(corners):
                    if cube_vals[idx_c]:
                        positions_inside.append((ci * spacing[0], cj * spacing[1], ck * spacing[2]))
                    else:
                        positions_outside.append((ci * spacing[0], cj * spacing[1], ck * spacing[2]))

                # Menghitung titik tengah sebagai perkiraan posisi vertex mesh
                center = np.mean(positions_inside + positions_outside, axis=0)

                # Menambahkan vertex ke daftar
                vertices.append(center)

    # Mengkonversi ke numpy array
    vertices = np.array(vertices) if len(vertices) > 0 else np.zeros((0, 3))

    # Membuat faces sederhana menggunakan Delaunay jika cukup vertex
    if len(vertices) > 3:
        # Mengimpor ConvexHull sebagai alternatif
        from scipy.spatial import ConvexHull
        try:
            # Menghitung ConvexHull dari vertices untuk membuat faces
            hull = ConvexHull(vertices)
            faces = hull.simplices
        except Exception:
            faces = np.zeros((0, 3), dtype=int)
    else:
        faces = np.zeros((0, 3), dtype=int)

    return vertices, faces

# Menentukan threshold isosurface (level set = 0)
threshold = 0.0

# Mengekstrak isosurface menggunakan metode yang tersedia
if HAS_SKIMAGE:
    # Menggunakan skimage marching_cubes
    verts_bola, faces_bola, normals_bola, values_bola = marching_cubes(
        field_bola, level=threshold
    )
    # Menskalakan vertices ke koordinat asli
    skala = 3.0 / resolusi
    verts_bola = verts_bola * skala - 1.5
    print(f"[skimage] Jumlah vertices: {len(verts_bola)}")
    print(f"[skimage] Jumlah faces: {len(faces_bola)}")
else:
    # Menggunakan implementasi manual
    spacing = (3.0/resolusi, 3.0/resolusi, 3.0/resolusi)
    verts_bola, faces_bola = marching_cubes_manual(field_bola, threshold, spacing)
    # Menggeser vertices ke koordinat asli
    verts_bola = verts_bola - 1.5
    print(f"[manual] Jumlah vertices: {len(verts_bola)}")
    print(f"[manual] Jumlah faces: {len(faces_bola)}")

# ========================================================
# BAGIAN 3: BERBAGAI BENTUK IMPLICIT FUNCTION
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 3: Berbagai Bentuk Implicit Function")
print("=" * 60)

# Mendefinisikan fungsi implisit untuk torus
R_torus = 0.8  # Radius mayor torus
r_torus = 0.3  # Radius minor torus

# Menghitung field torus: (sqrt(x^2+y^2) - R)^2 + z^2 - r^2
field_torus = (np.sqrt(X**2 + Y**2) - R_torus)**2 + Z**2 - r_torus**2
print(f"Field torus: R={R_torus}, r={r_torus}")

# Mendefinisikan fungsi implisit untuk double sphere
offset = 0.6  # Jarak antar pusat bola

# Menghitung field double sphere: minimum dari dua bola
field_sphere1 = (X - offset)**2 + Y**2 + Z**2 - 0.5**2
field_sphere2 = (X + offset)**2 + Y**2 + Z**2 - 0.5**2

# Menggabungkan dua bola menggunakan operasi minimum (union)
field_double = np.minimum(field_sphere1, field_sphere2)
print(f"Field double sphere: offset={offset}, radius=0.5")

# Membuat dictionary untuk menyimpan hasil semua bentuk
bentuk_dict = {
    'Sphere': field_bola,
    'Torus': field_torus,
    'Double Sphere': field_double
}

# Menyimpan hasil marching cubes untuk setiap bentuk
hasil_mc = {}

# Mengekstrak isosurface untuk setiap bentuk
for nama, field in bentuk_dict.items():
    # Mencetak nama bentuk yang sedang diproses
    print(f"\nMemproses: {nama}")

    if HAS_SKIMAGE:
        # Mengekstrak mesh menggunakan skimage
        v, f, n, val = marching_cubes(field, level=threshold)
        # Menskalakan ke koordinat asli
        v = v * (3.0 / resolusi) - 1.5
    else:
        # Mengekstrak mesh menggunakan metode manual
        spacing = (3.0/resolusi, 3.0/resolusi, 3.0/resolusi)
        v, f = marching_cubes_manual(field, threshold, spacing)
        v = v - 1.5

    # Menyimpan hasil ke dictionary
    hasil_mc[nama] = {'vertices': v, 'faces': f}
    print(f"  Vertices: {len(v)}, Faces: {len(f)}")

# ========================================================
# BAGIAN 4: VARIASI RESOLUSI (COARSE VS FINE)
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 4: Pengaruh Resolusi Grid")
print("=" * 60)

# Mendefinisikan berbagai resolusi untuk perbandingan
resolusi_list = [15, 30, 60]

# Menyimpan hasil per resolusi
hasil_resolusi = {}

# Mengekstrak isosurface bola pada berbagai resolusi
for res in resolusi_list:
    # Membuat grid dengan resolusi tertentu
    lin_r = np.linspace(-1.5, 1.5, res)
    Xr, Yr, Zr = np.meshgrid(lin_r, lin_r, lin_r, indexing='ij')

    # Menghitung field bola pada resolusi ini
    field_r = Xr**2 + Yr**2 + Zr**2 - 1.0

    if HAS_SKIMAGE:
        # Mengekstrak mesh menggunakan skimage
        v_r, f_r, _, _ = marching_cubes(field_r, level=0.0)
        # Menskalakan ke koordinat asli
        v_r = v_r * (3.0 / res) - 1.5
    else:
        # Mengekstrak mesh menggunakan metode manual
        sp = (3.0/res, 3.0/res, 3.0/res)
        v_r, f_r = marching_cubes_manual(field_r, 0.0, sp)
        v_r = v_r - 1.5

    # Menyimpan hasil resolusi
    hasil_resolusi[res] = {'vertices': v_r, 'faces': f_r}
    print(f"Resolusi {res}: Vertices={len(v_r)}, Faces={len(f_r)}")

# ========================================================
# BAGIAN 5: VISUALISASI MESH BERBAGAI BENTUK
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 5: Visualisasi Mesh Berbagai Bentuk")
print("=" * 60)

# Membuat figure untuk menampilkan 3 bentuk
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

# Mengatur judul figure
fig1.suptitle("Marching Cubes - Berbagai Bentuk Isosurface", fontsize=14, fontweight='bold')

# Iterasi setiap bentuk untuk visualisasi
for idx, (nama, data) in enumerate(hasil_mc.items()):
    # Mengambil axes saat ini
    ax = axes1[idx]

    # Mendapatkan vertices dan faces
    v = data['vertices']
    f = data['faces']

    # Menampilkan mesh jika ada faces
    if len(f) > 0 and len(v) > 0:
        # Menampilkan menggunakan plot_trisurf
        ax.plot_trisurf(
            v[:, 0], v[:, 1], v[:, 2],
            triangles=f, cmap='viridis', alpha=0.8, edgecolor='none'
        )
    elif len(v) > 0:
        # Menampilkan sebagai scatter plot jika tidak ada faces
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, alpha=0.5)

    # Mengatur judul subplot
    ax.set_title(f"{nama}\nV={len(v)}, F={len(f)}")

    # Mengatur label sumbu
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan visualisasi berbagai bentuk
path_bentuk = os.path.join(OUTPUT_DIR, "16_marching_cubes_bentuk.png")
plt.savefig(path_bentuk, dpi=150, bbox_inches='tight')
print(f"Visualisasi bentuk disimpan: {path_bentuk}")

# Menutup figure
plt.close(fig1)

# ========================================================
# BAGIAN 6: VISUALISASI PERBANDINGAN RESOLUSI
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 6: Visualisasi Perbandingan Resolusi")
print("=" * 60)

# Membuat figure untuk perbandingan resolusi
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

# Mengatur judul figure
fig2.suptitle("Pengaruh Resolusi Grid pada Marching Cubes", fontsize=14, fontweight='bold')

# Iterasi setiap resolusi untuk visualisasi
for idx, res in enumerate(resolusi_list):
    # Mengambil axes saat ini
    ax = axes2[idx]

    # Mendapatkan data untuk resolusi ini
    data = hasil_resolusi[res]
    v = data['vertices']
    f = data['faces']

    # Menampilkan mesh
    if len(f) > 0 and len(v) > 0:
        # Menggunakan plot_trisurf untuk menampilkan mesh
        ax.plot_trisurf(
            v[:, 0], v[:, 1], v[:, 2],
            triangles=f, cmap='plasma', alpha=0.8, edgecolor='none'
        )
    elif len(v) > 0:
        # Menampilkan sebagai scatter plot
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, alpha=0.5)

    # Mengatur judul subplot
    ax.set_title(f"Resolusi {res}³\nV={len(v)}, F={len(f)}")

    # Mengatur label sumbu
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan visualisasi perbandingan resolusi
path_resolusi = os.path.join(OUTPUT_DIR, "16_marching_cubes_resolusi.png")
plt.savefig(path_resolusi, dpi=150, bbox_inches='tight')
print(f"Visualisasi resolusi disimpan: {path_resolusi}")

# Menutup figure
plt.close(fig2)

# ========================================================
# BAGIAN 7: STATISTIK MESH PER BENTUK
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 7: Statistik Mesh Per Bentuk")
print("=" * 60)

# Iterasi setiap bentuk untuk mencetak statistik
for nama, data in hasil_mc.items():
    # Mendapatkan vertices dan faces
    v = data['vertices']
    f = data['faces']

    # Mencetak header statistik
    print(f"\n--- {nama} ---")
    print(f"  Jumlah vertices   : {len(v)}")
    print(f"  Jumlah faces      : {len(f)}")

    if len(v) > 0:
        # Menghitung bounding box
        bbox_min = v.min(axis=0)
        bbox_max = v.max(axis=0)
        bbox_size = bbox_max - bbox_min
        print(f"  Bounding box min  : ({bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f})")
        print(f"  Bounding box max  : ({bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f})")
        print(f"  Ukuran bbox       : ({bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f})")

    if len(f) > 0 and len(v) > 3:
        # Menghitung luas rata-rata face (perkiraan)
        luas_total = 0.0
        for face in f[:min(len(f), 1000)]:
            # Menghitung luas segitiga menggunakan cross product
            v0, v1, v2 = v[face[0]], v[face[1]], v[face[2]]
            luas_face = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            luas_total += luas_face

        # Menghitung rata-rata luas face
        luas_rata = luas_total / min(len(f), 1000)
        print(f"  Luas rata-rata face: {luas_rata:.6f}")

# Iterasi statistik resolusi
print("\n--- Statistik Resolusi ---")
for res, data in hasil_resolusi.items():
    print(f"  Resolusi {res:3d}: V={len(data['vertices']):6d}, F={len(data['faces']):6d}")

# ========================================================
# BAGIAN 8: VISUALISASI WIREFRAME MESH
# ========================================================
print("\n" + "=" * 60)
print("BAGIAN 8: Visualisasi Wireframe Mesh")
print("=" * 60)

# Membuat figure untuk wireframe
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})

# Mengatur judul figure
fig3.suptitle("Solid vs Wireframe Rendering", fontsize=14, fontweight='bold')

# Mendapatkan data bola
v_bola = hasil_mc['Sphere']['vertices']
f_bola = hasil_mc['Sphere']['faces']

# Menampilkan solid rendering di subplot pertama
ax_solid = axes3[0]
if len(f_bola) > 0 and len(v_bola) > 0:
    # Menampilkan mesh solid
    ax_solid.plot_trisurf(
        v_bola[:, 0], v_bola[:, 1], v_bola[:, 2],
        triangles=f_bola, cmap='coolwarm', alpha=0.9, edgecolor='none'
    )
ax_solid.set_title("Solid Rendering")
ax_solid.set_xlabel("X")
ax_solid.set_ylabel("Y")
ax_solid.set_zlabel("Z")

# Menampilkan wireframe di subplot kedua
ax_wire = axes3[1]
if len(f_bola) > 0 and len(v_bola) > 0:
    # Menampilkan mesh dengan wireframe
    ax_wire.plot_trisurf(
        v_bola[:, 0], v_bola[:, 1], v_bola[:, 2],
        triangles=f_bola, color='cyan', alpha=0.3,
        edgecolor='darkblue', linewidth=0.3
    )
ax_wire.set_title("Wireframe Rendering")
ax_wire.set_xlabel("X")
ax_wire.set_ylabel("Y")
ax_wire.set_zlabel("Z")

# Menyesuaikan layout
plt.tight_layout()

# Menyimpan visualisasi wireframe
path_wireframe = os.path.join(OUTPUT_DIR, "16_marching_cubes_wireframe.png")
plt.savefig(path_wireframe, dpi=150, bbox_inches='tight')
print(f"Visualisasi wireframe disimpan: {path_wireframe}")

# Menutup figure
plt.close(fig3)

# ========================================================
# RINGKASAN
# ========================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 16: MARCHING CUBES")
print("=" * 60)
print(f"Metode yang digunakan: {'skimage' if HAS_SKIMAGE else 'manual'}")
print(f"Bentuk yang diekstrak: {list(hasil_mc.keys())}")
print(f"Resolusi yang diuji: {resolusi_list}")
print(f"\nSemua output disimpan di: {OUTPUT_DIR}")
print("=" * 60)
