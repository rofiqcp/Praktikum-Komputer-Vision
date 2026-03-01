# MATERI MODUL 12: REKONSTRUKSI 3D DAN IMAGE-BASED RENDERING

---

## 1. Pendahuluan
Modul ini membahas tahap akhir pipeline komputer vision: merekonstruksi objek 3D dari data citra dan merender tampilan baru dari model yang dihasilkan. Rekonstruksi 3D mengubah informasi 2D (gambar, depth map, point cloud) menjadi representasi geometri 3D (mesh, voxel, implicit surface). Image-Based Rendering (IBR) menghasilkan pandangan (view) baru dari sebuah scene tanpa perlu model geometri eksplisit yang sempurna, cukup dari kumpulan foto yang ada.

Referensi utama: **Szeliski, "Computer Vision: Algorithms and Applications", 2nd Ed., Ch. 13 (3D Reconstruction) & Ch. 14 (Image-Based Rendering)**.

---

## 2. Point Cloud Processing

### 2.1 Representasi Point Cloud
Point cloud adalah kumpulan titik 3D $(x, y, z)$ — bisa dilengkapi warna $(r, g, b)$ dan normal $(n_x, n_y, n_z)$. Format umum: PLY, PCD, XYZ, LAS.

### 2.2 Filtering dan Preprocessing
- **Statistical Outlier Removal**: Hapus titik yang jaraknya ke tetangga terdekat terlalu jauh.
  $$d_i = \frac{1}{k} \sum_{j=1}^{k} \| p_i - p_j \|, \quad \text{hapus jika } d_i > \mu_d + \alpha \cdot \sigma_d$$

- **Voxel Downsampling**: Bagi ruang menjadi voxel grid, ambil satu titik per voxel (centroid).
- **Radius Outlier Removal**: Hapus titik dengan tetangga < threshold dalam radius tertentu.

### 2.3 Normal Estimation
Normal di setiap titik dihitung dari Principal Component Analysis (PCA) lokal:
1. Ambil $k$-nearest neighbors.
2. Hitung matriks kovarians.
3. Eigenvector dengan eigenvalue terkecil = surface normal.

$$\mathbf{C} = \frac{1}{k}\sum_{i=1}^{k}(\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{p}_i - \bar{\mathbf{p}})^T$$

### 2.4 Point Cloud Registration
**Iterative Closest Point (ICP)** menyelaraskan dua point cloud:

1. **Find correspondences**: Untuk setiap titik di *source*, cari titik terdekat di *target*.
2. **Estimate transformation**: Hitung rotasi $\mathbf{R}$ dan translasi $\mathbf{t}$ yang meminimalkan:
   $$E = \sum_{i} \| \mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \|^2$$
3. **Apply dan iterasi** hingga konvergen.

Variasi: **Point-to-Plane ICP** — meminimalkan jarak titik ke bidang tangent target:
$$E = \sum_{i} \left[ (\mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i) \cdot \mathbf{n}_i \right]^2$$

```python
import open3d as o3d

source = o3d.io.read_point_cloud("source.ply")
target = o3d.io.read_point_cloud("target.ply")

# ICP registration
threshold = 0.02
reg = o3d.pipelines.registration.registration_icp(
    source, target, threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
)
source.transform(reg.transformation)
```

---

## 3. Surface Reconstruction

### 3.1 Poisson Surface Reconstruction
Metode ini merekonstruksi implicit surface dari oriented point cloud (titik + normal). Ide: cari fungsi indikator $\chi$ yang gradiennya sesuai dengan field normal:
$$\nabla \chi = \mathbf{V} \quad \Rightarrow \quad \nabla \cdot \nabla \chi = \nabla \cdot \mathbf{V}$$

Diselesaikan sebagai persamaan Poisson $\Delta \chi = \nabla \cdot \mathbf{V}$ menggunakan octree. Surface = isosurface $\chi = \text{threshold}$.

```python
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)
# Remove low-density vertices
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)
```

### 3.2 Ball Pivoting Algorithm (BPA)
Bayangkan bola dengan radius $r$ menggelinding di atas point cloud. Setiap kali menyentuh 3 titik, membentuk segitiga. Sederhana tetapi sensitif terhadap pemilihan radius.

### 3.3 Alpha Shapes
Generalisasi convex hull. Parameter $\alpha$ mengontrol detail:
- $\alpha = 0$: convex hull.
- $\alpha$ besar: semakin banyak detail dan lubang.

### 3.4 Marching Cubes
Mengekstrak mesh dari volumetric representation (voxel grid atau implicit function). Membagi ruang menjadi kubus, lalu menentukan konfigurasi vertex in/out untuk menghasilkan segitiga.

### 3.5 Mesh Post-Processing
- **Decimation**: Kurangi jumlah segitiga (simplification).
- **Smoothing**: Laplacian smoothing, Taubin smoothing.
- **Hole filling**: Tutup lubang pada mesh.
- **Texturing**: Proyeksikan warna dari gambar ke mesh.

```python
# Mesh simplification
mesh_simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)

# Smoothing
mesh_smooth = mesh.filter_smooth_laplacian(number_of_iterations=5)
```

---

## 4. Volumetric Reconstruction

### 4.1 TSDF (Truncated Signed Distance Function)
Representasi volumetrik yang menyimpan signed distance ke surface terdekat di setiap voxel. TSDF di-*integrate* dari multiple depth maps:

$$TSDF(\mathbf{x}) = \frac{\sum_i w_i \cdot d_i(\mathbf{x})}{\sum_i w_i}$$

di mana $d_i$ = signed distance dari voxel $\mathbf{x}$ ke surface pada depth frame $i$.

**KinectFusion** pipeline:
1. Capture depth frame.
2. Track camera pose (ICP terhadap model).
3. Integrate depth ke TSDF volume.
4. Raycast TSDF untuk render.

```python
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=4.0/512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)
for i, rgbd in enumerate(rgbd_images):
    volume.integrate(rgbd, intrinsic, extrinsics[i])
mesh = volume.extract_triangle_mesh()
```

### 4.2 Voxel Hashing
Representasi sparse voxel untuk scene besar tanpa mengalokasikan grid penuh. Hanya voxel di dekat surface yang disimpan (hash table).

---

## 5. Multi-View Stereo (MVS)

### 5.1 Plane-Sweeping Stereo
Evaluasi depth hypothesis pada serangkaian bidang fronto-parallel:

Untuk setiap depth $d$:
1. Homography-warp reference ke neighbor views.
2. Hitung photo-consistency cost.
3. Pilih $d$ dengan cost minimum.

### 5.2 Patch-Based MVS (PMVS/CMVS)
1. Detect features → match → triangulate (sparse).
2. Expand patches ke area featureless.
3. Filter inconsistent patches.

Tool: **COLMAP** — software open-source untuk SfM + MVS pipeline lengkap.

---

## 6. Image-Based Rendering (IBR)

### 6.1 Prinsip Dasar
IBR menghasilkan novel views dari kumpulan foto tanpa model 3D eksplisit yang sempurna. Spektrum representasi:

| Representasi | Geometri | Contoh |
|-------------|----------|--------|
| Tanpa geometri | Tidak ada | Light field, Lumigraph |
| Implicit | Depth maps | View interpolation |
| Explicit | Mesh + texture | Traditional rendering |
| Neural | Network weights | NeRF |

### 6.2 View Interpolation
Diberikan dua view dengan depth, hasilkan view antara:

1. Warp kedua view ke viewpoint baru menggunakan depth-based reprojection.
2. Blend hasil warp.
3. Handle disocclusion (inpainting area yang tidak terlihat).

$$\mathbf{p}' = \mathbf{K}' (\mathbf{R}' \mathbf{R}^{-1}) (\mathbf{K}^{-1} \mathbf{p} \cdot d - \mathbf{t}) + \mathbf{K}' \mathbf{t}'$$

### 6.3 Image Warping
Transformasi piksel dari satu view ke view lain berdasarkan geometri (depth dan camera pose):

- **Forward warping**: Splat pixels dari source ke target. Masalah: holes dan aliasing.
- **Inverse warping**: Untuk setiap pixel di target, cari asalnya di source. Butuh depth di target view.

### 6.4 Texture Mapping
Proyeksikan warna dari foto ke mesh 3D:
1. UV parameterization.
2. Multi-view blending: pilih foto terbaik berdasarkan angle dan resolusi.
3. Seam optimization untuk transisi halus.

### 6.5 Light Fields
Representasi 4D dari semua sinar di scene. Parameterisasi dua-bidang $(u, v, s, t)$:
- $(u, v)$: posisi di camera plane.
- $(s, t)$: posisi di focal plane.

Novel view = slice dari light field.

---

## 7. Neural Scene Representations

### 7.1 Neural Radiance Fields (NeRF)
NeRF merepresentasikan scene sebagai fungsi kontinu:

$$F_\theta : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

- Input: posisi 3D $\mathbf{x} = (x, y, z)$ dan viewing direction $\mathbf{d} = (\theta, \phi)$.
- Output: warna $\mathbf{c} = (r, g, b)$ dan density $\sigma$.

Volume rendering integral:
$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

di mana $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right)$.

**Positional encoding**:
$$\gamma(p) = [\sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)]$$

### 7.2 3D Gaussian Splatting
Representasi scene sebagai kumpulan 3D Gaussian:
- Setiap Gaussian: posisi $\mu$, kovarians $\Sigma$, opacity $\alpha$, spherical harmonics untuk warna view-dependent.
- Rendering: differentiable rasterization (jauh lebih cepat daripada NeRF).
- Training: optimize posisi, shape, warna via gradient descent.

### 7.3 Instant NGP
Percepatan NeRF menggunakan multi-resolution hash encoding:
- Hash table menyimpan feature vectors pada berbagai resolusi.
- Training <5 menit (vs berjam-jam pada NeRF original).

---

## 8. Aplikasi Rekonstruksi 3D

| Aplikasi | Teknik Utama |
|----------|-------------|
| Digital twin | TSDF + texturing |
| Cultural heritage | SfM + MVS + mesh |
| Robotics mapping | Visual SLAM + TSDF |
| AR content creation | 3D Gaussian splatting |
| VR/Gaming | NeRF → mesh export |
| Medical imaging | Volumetric reconstruction |
| E-commerce | Object 3D scanning |
| Architecture | LiDAR + photo reconstruction |

---

## 9. Pipeline Lengkap

```
Gambar Multi-View
      │
      ▼
  SfM (Modul 11)
      │
  ┌───┴───┐
  │       │
  ▼       ▼
Sparse 3D   Camera Poses
Points       & Intrinsics
  │       │
  └───┬───┘
      │
      ▼
  Dense MVS
      │
  ┌───┴───┐
  │       │
  ▼       ▼
Dense         Depth Maps
Point Cloud
  │       │
  ▼       ▼
Surface       TSDF
Recon         Integration
  │       │
  └───┬───┘
      │
      ▼
  Textured Mesh
      │
  ┌───┴───┐
  │       │
  ▼       ▼
Export 3D   Novel View
(OBJ/PLY)   Synthesis
            (NeRF/3DGS)
```

---

## 10. Tools dan Library

| Library | Fungsi |
|---------|--------|
| **Open3D** | Point cloud, mesh, TSDF, ICP, visualization |
| **COLMAP** | SfM + MVS end-to-end |
| **PyMeshLab** | Mesh processing (Poisson, BPA, simplification) |
| **trimesh** | Mesh I/O, operations, boolean |
| **nerfstudio** | NeRF training & rendering |
| **gsplat** | 3D Gaussian splatting |
| **OpenCV** | Stereo, depth, warping |
| **PCL (via pclpy)** | Point cloud processing (C++ with Python bindings) |
| **Meshroom** | Open-source photogrammetry (GUI) |

---

## Referensi
1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed. Springer. Ch. 13–14.
2. Kazhdan, M., Hoppe, H. (2013). Screened Poisson Surface Reconstruction. *ACM ToG*.
3. Curless, B., Levoy, M. (1996). A Volumetric Method for Building Complex Models from Range Images. *SIGGRAPH*.
4. Newcombe, R. et al. (2011). KinectFusion: Real-Time Dense Surface Mapping and Tracking. *ISMAR*.
5. Mildenhall, B. et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV*.
6. Kerbl, B. et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM ToG*.
7. Müller, T. et al. (2022). Instant Neural Graphics Primitives with a Multiresolution Hash Encoding. *ACM ToG*.
8. Schönberger, J.L., Frahm, J.M. (2016). Structure-from-Motion Revisited. *CVPR*.
9. Open3D Documentation. http://www.open3d.org/docs/
10. COLMAP Documentation. https://colmap.github.io/
