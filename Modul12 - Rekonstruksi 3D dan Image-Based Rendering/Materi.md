# MATERI MODUL 12: REKONSTRUKSI 3D DAN IMAGE-BASED RENDERING

## 12.1 Pendahuluan Rekonstruksi 3D

Rekonstruksi 3D adalah proses membangun model tiga dimensi dari data sensor (gambar, depth, point cloud). Modul ini mencakup seluruh pipeline dari representasi data 3D, pemrosesan point cloud, rekonstruksi permukaan, hingga teknik rendering modern.

**Representasi Data 3D:**
| Representasi | Deskripsi | Contoh Format |
|---|---|---|
| Point Cloud | Kumpulan titik (x,y,z) | PLY, PCD, XYZ |
| Mesh | Vertices + Faces (triangle) | OBJ, STL, PLY |
| Volume | Voxel grid (3D array) | TSDF, occupancy |
| Implicit | Fungsi f(x,y,z) = 0 | SDF, NeRF |

## 12.2 Point Cloud Basics

Point cloud merepresentasikan permukaan objek sebagai kumpulan titik 3D:

$$P = \{(x_i, y_i, z_i) \mid i = 1, \ldots, N\}$$

**Format PLY (Polygon File Format):**
```
ply
format ascii 1.0
element vertex N
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
x1 y1 z1 r1 g1 b1
...
```

Sumber point cloud: LiDAR, structured light, stereo matching, depth camera (Kinect), SfM.

## 12.3 Point Cloud Filtering

### Voxel Downsampling
Membagi ruang 3D menjadi voxel grid berukuran $v$, setiap voxel yang berisi titik diwakilkan oleh centroid:

$$\mathbf{p}_{voxel} = \frac{1}{|S|} \sum_{i \in S} \mathbf{p}_i$$

### Statistical Outlier Removal (SOR)
Hitung rata-rata jarak ke $k$ tetangga terdekat $\bar{d}_i$. Hapus titik jika:

$$\bar{d}_i > \mu_d + \alpha \cdot \sigma_d$$

dimana $\mu_d$ dan $\sigma_d$ adalah mean dan standar deviasi dari seluruh $\bar{d}$.

### Radius Outlier Removal
Hapus titik yang memiliki $< n_{min}$ tetangga dalam radius $r$.

## 12.4 Normal Estimation

Normal permukaan diestimasi menggunakan **PCA lokal**:

1. Ambil $k$ tetangga terdekat dari titik $\mathbf{p}_i$
2. Hitung covariance matrix:
$$C = \frac{1}{k} \sum_{j=1}^{k} (\mathbf{p}_j - \bar{\mathbf{p}})(\mathbf{p}_j - \bar{\mathbf{p}})^T$$
3. Eigenvector dari eigenvalue terkecil = normal $\mathbf{n}_i$

**Orientasi konsisten:** arahkan semua normal ke sisi yang sama (misal ke kamera).

## 12.5 ICP Registration

**Iterative Closest Point (ICP)** menyelaraskan source $P$ ke target $Q$:

1. **Closest Point:** untuk setiap $\mathbf{p}_i \in P$, cari $\mathbf{q}_j \in Q$ terdekat
2. **Transformation:** minimize:
$$E(R, \mathbf{t}) = \sum_i \|R\mathbf{p}_i + \mathbf{t} - \mathbf{q}_{c(i)}\|^2$$
3. **SVD Solution:**
$$H = \sum_i (\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{q}_{c(i)} - \bar{\mathbf{q}})^T$$
$$H = U \Sigma V^T \implies R = V U^T, \quad \mathbf{t} = \bar{\mathbf{q}} - R\bar{\mathbf{p}}$$
4. **Iterasi** sampai konvergen: $\Delta E < \epsilon$

**Variasi:** Point-to-Plane ICP minimize jarak ke bidang tangent:
$$E = \sum_i \left[(\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_{c(i)}) \cdot \mathbf{n}_{c(i)}\right]^2$$

## 12.6 Surface Reconstruction

### Poisson Surface Reconstruction
Filosofi: cari indicator function $\chi$ dimana $\nabla \chi = \mathbf{V}$ (oriented normals).

Solve Poisson equation:
$$\Delta \chi = \nabla \cdot \mathbf{V}$$

Isosurface $\chi = \tau$ diextract dengan Marching Cubes.

**Kelebihan:** menghasilkan mesh watertight (tertutup), robust terhadap noise.
**Kekurangan:** memerlukan normal yang baik, bisa menghasilkan artefak di area sparse.

### Ball Pivoting Algorithm (BPA)
Bola berradius $\rho$ menggelinding di permukaan point cloud:
1. Bola menyentuh 3 titik → buat triangle
2. Pivot bola ke edge baru → cari titik ketiga baru
3. Ulangi sampai seluruh permukaan tercover

**Kelebihan:** preservasi detail, cocok untuk scan data.
**Kekurangan:** sensitif terhadap pilihan radius, tidak menutup gaps.

### Alpha Shapes
Generalisasi convex hull: hapus simpleks dari Delaunay triangulation dimana circumradius > $1/\alpha$:

$$\alpha\text{-shape} = \{T \in \text{Delaunay} \mid R_{circumscribed}(T) < \frac{1}{\alpha}\}$$

- $\alpha \to 0$: hanya titik (empty shape)
- $\alpha \to \infty$: convex hull

## 12.7 Marching Cubes

Algoritma untuk mengekstrak isosurface dari scalar field $f(x,y,z) = c$:

1. Bagi volume menjadi kubus-kubus (voxel)
2. Untuk setiap kubus (8 corner), klasifikasi inside/outside
3. $2^8 = 256$ konfigurasi → lookup table → triangle vertices
4. Interpolasi posisi vertex pada edge yang di-cross

**Marching Cubes 33:** mengatasi ambiguitas konfigurasi tertentu.

## 12.8 TSDF Integration

**Truncated Signed Distance Function** menyimpan jarak bertanda ke permukaan terdekat dalam voxel grid:

$$\text{TSDF}(\mathbf{x}) = \text{clamp}\left(\frac{d_{meas} - d_{proj}}{\delta}, -1, 1\right)$$

dimana $\delta$ adalah truncation distance.

**Fusion (KinectFusion):**
$$\text{TSDF}_{k+1} = \frac{W_k \cdot \text{TSDF}_k + w_{k+1} \cdot \text{tsdf}_{k+1}}{W_k + w_{k+1}}$$

Permukaan diekstrak dimana TSDF = 0 (zero-crossing) menggunakan Marching Cubes.

## 12.9 Image Warping dan View Synthesis

### Forward Warping
Untuk setiap piksel $(u, v)$ di source:
$$\mathbf{X} = D(u,v) \cdot K^{-1}[u, v, 1]^T$$
$$[u', v', 1]^T \propto K'(R\mathbf{X} + \mathbf{t})$$

**Masalah:** holes (piksel target tidak ter-cover) dan conflicts (beberapa source ke satu target).

### Inverse Warping
Untuk setiap piksel $(u', v')$ di target:
$$\mathbf{X}' = D'(u',v') \cdot K'^{-1}[u', v', 1]^T$$
$$[u, v, 1]^T \propto K(R^T\mathbf{X}' - R^T\mathbf{t})$$

**Keuntungan:** tidak ada holes, tapi memerlukan depth di view target.

## 12.10 View Interpolation

Mensintesis gambar dari viewpoint intermediate:

**Linear blend:**
$$I_\alpha = (1-\alpha) \cdot I_1 + \alpha \cdot I_2$$

**Flow-based interpolation:**
1. Hitung optical flow $\mathbf{F}_{1 \to 2}$
2. Warp $I_1$ dengan $\alpha \cdot \mathbf{F}$ dan $I_2$ dengan $(1-\alpha) \cdot \mathbf{F}$
3. Blend warped images

## 12.11 Neural Radiance Fields (NeRF)

NeRF (Mildenhall et al., 2020) merepresentasikan scene sebagai fungsi kontinu:

$$F_\theta: (\mathbf{x}, \mathbf{d}) \to (\mathbf{c}, \sigma)$$

dimana $\mathbf{x} = (x,y,z)$ posisi, $\mathbf{d} = (\theta, \phi)$ arah pandang, $\mathbf{c}$ warna RGB, $\sigma$ density.

**Volume Rendering:**
$$C(\mathbf{r}) = \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot \mathbf{c}_i$$
$$T_i = \prod_{j=1}^{i-1}(1 - \alpha_j), \quad \alpha_i = 1 - e^{-\sigma_i \delta_i}$$

**Positional Encoding:**
$$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$$

**Training:** minimize MSE antara rendered pixel dan ground truth:
$$\mathcal{L} = \sum_{\mathbf{r}} \|C(\mathbf{r}) - C_{gt}(\mathbf{r})\|^2$$

## 12.12 3D Gaussian Splatting

3DGS (Kerbl et al., 2023) merepresentasikan scene sebagai kumpulan 3D Gaussian:

Setiap Gaussian: $G_i = (\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \alpha_i, \mathbf{c}_i)$
- $\boldsymbol{\mu}_i$: posisi 3D (mean)
- $\boldsymbol{\Sigma}_i$: 3D covariance matrix
- $\alpha_i$: opacity
- $\mathbf{c}_i$: spherical harmonics coefficients (view-dependent color)

**Rendering (Differentiable Splatting):**
1. Project 3D Gaussians ke 2D
2. Sort berdasarkan depth
3. Alpha compositing:
$$C = \sum_{i=1}^{N} \mathbf{c}_i \cdot \alpha_i \cdot \prod_{j=1}^{i-1}(1 - \alpha_j)$$

**Keunggulan:**
- Training: 10-100× lebih cepat dari NeRF
- Rendering: real-time (~100 FPS) vs NeRF (~0.1 FPS)
- Kualitas: setara atau lebih baik dari NeRF

## 12.13 Light Field

Light field merepresentasikan semua sinar cahaya dalam scene:

$$L(u, v, s, t)$$

dimana $(u,v)$ posisi kamera dan $(s,t)$ arah ray pada sensor plane.

**Epipolar Plane Image (EPI):** irisan 2D dari light field (fixkan satu dimensi).

**Aplikasi:**
- Post-capture refocusing
- Depth estimation dari slope EPI
- View synthesis tanpa geometry

## 12.14 RGBD Pipeline

Pipeline dari RGBD camera (Kinect, RealSense):

1. **Akuisisi:** RGB + Depth per frame
2. **Deprojection:** pixel (u,v,d) → 3D point (X,Y,Z)
3. **Registration:** ICP antar frame
4. **Fusion:** TSDF integration
5. **Extraction:** Marching Cubes → mesh
6. **Texturing:** proyeksi warna dari RGB frames

## 12.15 Perbandingan Metode Rekonstruksi

| Metode | Input | Output | Kecepatan | Kualitas |
|---|---|---|---|---|
| Poisson | Points + Normals | Watertight mesh | Sedang | Baik |
| BPA | Points + Normals | Open mesh | Cepat | Tergantung density |
| Alpha Shapes | Points | Boundary | Cepat | Sedang |
| Marching Cubes | Volume/SDF | Mesh | Cepat | Tergantung resolusi |
| TSDF + MC | Depth maps | Mesh | Lambat (fusi) | Baik |
| NeRF | Images + Poses | Novel views | Lambat training | Sangat baik |
| 3DGS | Images + Poses | Novel views | Cepat training | Sangat baik |

## Referensi
1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed. Chapter 12-14.
2. Mildenhall, B., et al. (2020). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis."
3. Kerbl, B., et al. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering."
4. Curless, B. & Levoy, M. (1996). "A Volumetric Method for Building Complex Models from Range Images."
5. Lorensen, W. & Cline, H. (1987). "Marching Cubes: A High Resolution 3D Surface Construction Algorithm."
6. Bernardini, F., et al. (1999). "The Ball-Pivoting Algorithm for Surface Reconstruction."
7. Newcombe, R., et al. (2011). "KinectFusion: Real-Time Dense Surface Mapping and Tracking."
8. Open3D Documentation: http://www.open3d.org/docs/
