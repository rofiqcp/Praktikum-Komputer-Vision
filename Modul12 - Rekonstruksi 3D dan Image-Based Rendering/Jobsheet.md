# JOBSHEET MODUL 12: REKONSTRUKSI 3D DAN IMAGE-BASED RENDERING

## Tujuan Praktikum
Setelah menyelesaikan praktikum ini, mahasiswa diharapkan mampu:
1. Memahami representasi data 3D (point cloud, mesh, volume)
2. Menerapkan algoritma point cloud processing (filtering, normal estimation, ICP)
3. Menerapkan teknik surface reconstruction (Poisson, BPA, Alpha Shapes, Marching Cubes)
4. Memahami konsep TSDF integration dan volumetric fusion
5. Menerapkan image warping dengan depth untuk view synthesis
6. Memahami konsep Neural Radiance Fields (NeRF) dan 3D Gaussian Splatting
7. Mengimplementasikan pipeline rekonstruksi 3D end-to-end

## Alat dan Bahan
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- SciPy (opsional, untuk kNN dan Delaunay)
- Open3D (opsional, untuk operasi point cloud lanjut)
- Dataset: bunny PLY, RGBD frames, multiview images

---

## Percobaan 1: Point Cloud Basics
**File:** `01_point_cloud_basics.py`

### Dasar Teori
Point cloud adalah kumpulan titik 3D {(x_i, y_i, z_i)} yang merepresentasikan permukaan objek. Format umum: PLY, PCD, XYZ.

### Langkah Kerja
1. Buka terminal dan jalankan `python 01_point_cloud_basics.py`
2. Program membuat point cloud sintetis (sphere dan cube)
3. Menyimpan ke format PLY (ASCII) dengan header standar
4. Membaca PLY dari folder image/ (bunny_point_cloud.ply)
5. Visualisasi scatter 3D dengan colormap berdasarkan koordinat Z
6. Output tersimpan di `output/01_point_cloud.png`

### Analisis
- Perhatikan distribusi titik pada sphere vs cube
- Bagaimana format PLY menyimpan data vertex?
- Apa perbedaan point cloud dengan mesh?

---

## Percobaan 2: Point Cloud Filtering
**File:** `02_point_cloud_filtering.py`

### Dasar Teori
Filtering menghilangkan noise dan outlier:
- **Voxel Downsampling**: bagi ruang 3D menjadi grid, ambil centroid per voxel
- **Statistical Outlier Removal (SOR)**: hapus titik dengan jarak rata-rata > μ + kσ
- **Radius Outlier Removal**: hapus titik dengan < n tetangga dalam radius r

### Langkah Kerja
1. Jalankan `python 02_point_cloud_filtering.py`
2. Program membuat point cloud dengan 50 outlier
3. Menerapkan voxel downsample (voxel_size=0.15)
4. Menerapkan SOR (k=15, std_ratio=1.5)
5. Visualisasi perbandingan noisy vs filtered
6. Output di `output/02_filtering.png`

### Analisis
- Berapa titik tersisa setelah voxel downsample?
- Apakah SOR berhasil menghapus semua outlier?
- Bagaimana trade-off antara voxel_size kecil vs besar?

---

## Percobaan 3: Normal Estimation
**File:** `03_normal_estimation.py`

### Dasar Teori
Vektor normal diestimasi menggunakan PCA lokal: hitung covariance matrix dari k tetangga terdekat, eigenvector terkecil = arah normal permukaan.

### Langkah Kerja
1. Jalankan `python 03_normal_estimation.py`
2. Program membuat point cloud sphere sintetis
3. Mengestimasi normal menggunakan PCA lokal (k=15)
4. Membandingkan dengan ground truth normal (normal sphere = posisi ternormalisasi)
5. Visualisasi quiver plot 3D
6. Output di `output/03_normals.png`

### Analisis
- Berapa akurasi estimasi normal (|dot product| dengan GT)?
- Bagaimana pengaruh k (jumlah tetangga) terhadap kualitas normal?
- Mengapa orientasi normal perlu dikonsistenkan?

---

## Percobaan 4: ICP Registration
**File:** `04_icp_registration.py`

### Dasar Teori
Iterative Closest Point (ICP) menyelaraskan dua point cloud:
1. Cari pasangan terdekat (nearest neighbor)
2. Hitung R, t optimal (SVD pada cross-covariance)
3. Terapkan transformasi
4. Iterasi sampai konvergen

### Langkah Kerja
1. Jalankan `python 04_icp_registration.py`
2. Program membuat target dan source (source = rotasi + translasi + noise)
3. Menjalankan ICP point-to-point (50 iterasi)
4. Visualisasi before/after alignment dan kurva konvergensi
5. Output di `output/04_icp.png`

### Analisis
- Berapa iterasi sampai konvergen?
- Apa pengaruh noise terhadap akurasi ICP?
- Jelaskan perbedaan point-to-point vs point-to-plane ICP

---

## Percobaan 5: Surface Reconstruction (Poisson)
**File:** `05_surface_reconstruction_poisson.py`

### Dasar Teori
Poisson Surface Reconstruction memecahkan persamaan Poisson ∆χ = ∇·V untuk mendapatkan indicator function χ, lalu mengekstrak isosurface. Input: point cloud + normals.

### Langkah Kerja
1. Jalankan `python 05_surface_reconstruction_poisson.py`
2. Program mendemonstrasikan pipeline: Point Cloud → Normals → Poisson → Mesh
3. Visualisasi 3 tahap: input points, estimated normals, reconstructed surface
4. Output di `output/05_poisson.png`

### Analisis
- Apa peran normal dalam Poisson reconstruction?
- Bagaimana kualitas normal mempengaruhi hasil rekonstruksi?
- Kapan Poisson lebih baik dari BPA?

---

## Percobaan 6: Mesh Processing
**File:** `06_mesh_processing.py`

### Dasar Teori
Operasi dasar mesh:
- **Decimation**: mengurangi jumlah vertex/face
- **Smoothing**: Laplacian smoothing (averaging tetangga)
- **Analisis topologi**: Euler characteristic V-E+F=2

### Langkah Kerja
1. Jalankan `python 06_mesh_processing.py`
2. Program membuat mesh terrain sinusoidal
3. Menerapkan decimation (subsample 30%)
4. Menerapkan Laplacian smoothing (5 iterasi)
5. Visualisasi original vs decimated vs smoothed
6. Output di `output/06_mesh.png`

### Analisis
- Berapa vertex sebelum dan sesudah decimation?
- Apakah Laplacian smoothing menghilangkan detail?
- Bagaimana memilih jumlah iterasi smoothing?

---

## Percobaan 7: TSDF Integration
**File:** `07_tsdf_integration.py`

### Dasar Teori
Truncated Signed Distance Function (TSDF) menyimpan jarak bertanda ke permukaan terdekat dalam voxel grid. Depth map berturutan diintegrasikan ke volume menggunakan running weighted average. Digunakan di KinectFusion.

### Langkah Kerja
1. Jalankan `python 07_tsdf_integration.py`
2. Program membuat TSDF volume (30×30×30 voxel)
3. Mengintegrasi satu depth map sintetis ke volume
4. Visualisasi depth map, TSDF slice, dan weight slice
5. Output di `output/07_tsdf.png`

### Analisis
- Berapa voxel yang terupdate setelah satu frame?
- Apa arti nilai positif/negatif dalam TSDF?
- Bagaimana truncation distance mempengaruhi hasil?

---

## Percobaan 8: Image Warping dengan Depth
**File:** `08_image_warping_depth.py`

### Dasar Teori
Image warping menggunakan depth map untuk mensintesis gambar dari viewpoint baru:
- **Forward warping**: untuk setiap piksel source, hitung posisi di target (masalah: holes)
- **Inverse warping**: untuk setiap piksel target, cari di source (masalah: perlu depth target)

### Langkah Kerja
1. Jalankan `python 08_image_warping_depth.py`
2. Program memuat/membuat gambar dan depth map
3. Melakukan forward warping dengan translasi kecil
4. Visualisasi source, depth map, dan hasil warp
5. Output di `output/08_warping.png`

### Analisis
- Apa yang terjadi pada area yang terhalang (occlusion)?
- Mengapa forward warping menghasilkan holes?
- Bagaimana inverse warping mengatasi masalah holes?

---

## Percobaan 9: View Interpolation
**File:** `09_view_interpolation.py`

### Dasar Teori
View interpolation mensintesis gambar di antara dua viewpoint:
- **Linear blend**: interpolasi pixel intensitas langsung
- **Flow-based**: gunakan optical flow untuk warp → blend

### Langkah Kerja
1. Jalankan `python 09_view_interpolation.py`
2. Program memuat dua gambar multiview
3. Menghasilkan interpolasi dengan 5 nilai alpha (0, 0.25, 0.5, 0.75, 1.0)
4. Perbandingan linear blend vs flow-based interpolation
5. Output di `output/09_view_interp.png`

### Analisis
- Bandingkan kualitas linear blend vs flow-based
- Pada alpha berapa artefak paling terlihat?
- Apa kelemahan interpolasi berbasis optical flow?

---

## Percobaan 10: Neural Rendering (Konsep NeRF)
**File:** `10_neural_rendering_konsep.py`

### Dasar Teori
Neural Radiance Fields (NeRF) merepresentasikan scene sebagai fungsi kontinu F(x,d) → (c,σ) yang dipelajari MLP. Rendering menggunakan volume rendering: C(r) = Σ T_i · α_i · c_i. Input: multi-view images + camera poses.

### Langkah Kerja
1. Jalankan `python 10_neural_rendering_konsep.py`
2. Program memvisualisasikan pipeline NeRF sebagai diagram
3. Mendemonstrasikan volume rendering dengan density field sintetis
4. Output di `output/10_nerf_pipeline.png` dan `output/10_volume_rendering.png`

### Analisis
- Apa input dan output dari MLP dalam NeRF?
- Bagaimana volume rendering mengakumulasi warna sepanjang ray?
- Apa kelemahan NeRF dibanding metode klasik?

---

## Percobaan 11: Ball Pivoting Algorithm
**File:** `11_ball_pivoting_algorithm.py`

### Dasar Teori
Ball Pivoting Algorithm (BPA): sebuah bola (radius ρ) menggelinding di permukaan point cloud. Ketika bola menyentuh 3 titik, segitiga terbentuk. Cocok untuk point cloud dengan density merata.

### Langkah Kerja
1. Jalankan `python 11_ball_pivoting_algorithm.py`
2. Program mendemonstrasikan konsep BPA dalam 2D
3. Visualisasi 3 tahap: input points, ball rolling, result mesh
4. Output di `output/11_bpa.png`

### Analisis
- Bagaimana radius bola mempengaruhi hasil mesh?
- Apa yang terjadi jika density point cloud tidak merata?
- Bandingkan BPA dengan Poisson reconstruction

---

## Percobaan 12: Alpha Shapes
**File:** `12_alpha_shapes.py`

### Dasar Teori
Alpha Shape: subset dari Delaunay triangulation dimana simplex dengan circumradius > 1/α dihapus. α kecil = boundary ketat, α besar = cembung.

### Langkah Kerja
1. Jalankan `python 12_alpha_shapes.py`
2. Program membuat point cloud 2D (ring shapes)
3. Menghitung alpha shapes dengan 4 nilai α (0.5, 1.0, 2.0, 5.0)
4. Visualisasi perbandingan jumlah triangle per α
5. Output di `output/12_alpha_shapes.png`

### Analisis
- Bagaimana α mempengaruhi jumlah triangle?
- Pada α berapa bentuk annular (cincin) terlihat?
- Apa hubungan alpha shape dengan convex hull (α → ∞)?

---

## Percobaan 13: Mesh Texturing
**File:** `13_mesh_texturing.py`

### Dasar Teori
Mesh texturing memetakan warna dari gambar 2D ke mesh 3D melalui proyeksi: untuk setiap vertex 3D, proyeksikan ke piksel gambar dan ambil warnanya.

### Langkah Kerja
1. Jalankan `python 13_mesh_texturing.py`
2. Program memuat gambar sebagai sumber tekstur
3. Membuat mesh grid planar 3D
4. Memproyeksikan setiap vertex ke gambar dan mengambil warna
5. Visualisasi mesh: untextured vs textured
6. Output di `output/13_texturing.png`

### Analisis
- Apakah proyeksi sederhana cukup untuk texturing?
- Bagaimana menangani occlusion saat texturing dari multiple views?
- Apa peran UV mapping dalam texturing?

---

## Percobaan 14: RGBD Point Cloud
**File:** `14_rgbd_point_cloud.py`

### Dasar Teori
RGBD camera (mis. Kinect) menghasilkan gambar RGB + Depth Map. Setiap piksel dengan depth valid dikonversi ke titik 3D berwarna: X = (u-cx)·Z/fx, Y = (v-cy)·Z/fy.

### Langkah Kerja
1. Jalankan `python 14_rgbd_point_cloud.py`
2. Program memuat RGBD data (rgbd_color_00.png + rgbd_depth_00.png)
3. Mengkonversi ke colored point cloud menggunakan intrinsik kamera
4. Visualisasi color, depth, dan 3D point cloud
5. Output di `output/14_rgbd_cloud.png`

### Analisis
- Berapa titik 3D yang dihasilkan?
- Bagaimana kualitas depth mempengaruhi point cloud?
- Apa keuntungan RGBD dibanding stereo vision?

---

## Percobaan 15: Point Cloud Segmentation
**File:** `15_point_cloud_segmentation.py`

### Dasar Teori
Segmentasi point cloud:
- **RANSAC plane**: fitting bidang menggunakan RANSAC untuk menghapus lantai/dinding
- **Euclidean clustering**: grouping titik berdasarkan jarak antar tetangga

### Langkah Kerja
1. Jalankan `python 15_point_cloud_segmentation.py`
2. Program membuat scene dengan plane (lantai) dan 2 objek
3. Melakukan RANSAC plane segmentation
4. Melakukan Euclidean clustering pada non-plane points
5. Visualisasi input, plane segmentation, dan clustering
6. Output di `output/15_segmentation.png`

### Analisis
- Berapa titik termasuk plane vs objek?
- Berapa cluster terdeteksi?
- Apa pengaruh threshold pada RANSAC dan clustering?

---

## Percobaan 16: Marching Cubes
**File:** `16_marching_cubes.py`

### Dasar Teori
Marching Cubes mengekstrak isosurface dari volume 3D: periksa setiap kubus (8 corner), jika ada perubahan tanda antara inside/outside, buat triangle. 256 konfigurasi dari lookup table.

### Langkah Kerja
1. Jalankan `python 16_marching_cubes.py`
2. Program membuat SDF volume (sphere r=1)
3. Mengekstrak isosurface menggunakan marching cubes sederhana
4. Visualisasi SDF slice, isosurface, dan ground truth sphere
5. Output di `output/16_marching_cubes.png`

### Analisis
- Berapa voxel yang di-cross oleh isosurface?
- Bagaimana resolusi volume mempengaruhi kualitas?
- Apa keuntungan marching cubes dibanding point-based rendering?

---

## Percobaan 17: Volumetric Rendering
**File:** `17_volumetric_rendering.py`

### Dasar Teori
Volumetric rendering: tembak ray dari kamera melalui density field, akumulasi warna dan opacity sepanjang ray menggunakan integrasi numerik: C = Σ T_i · α_i · c_i.

### Langkah Kerja
1. Jalankan `python 17_volumetric_rendering.py`
2. Program membuat volume 3D sintetis (dua bola Gaussian)
3. Melakukan ray casting orthographic
4. Visualisasi volume slices dan rendered image
5. Output di `output/17_volume_render.png`

### Analisis
- Bagaimana density field menentukan warna pixel?
- Apa efek transmittance pada objek yang terhalang?
- Bandingkan volumetric rendering vs surface rendering

---

## Percobaan 18: Light Field Basics
**File:** `18_light_field_basics.py`

### Dasar Teori
Light field: representasi 4D (u,v,s,t) dari semua sinar cahaya dalam scene. Diparameterkan oleh dua bidang: posisi kamera (u,v) dan arah (s,t). Epipolar Plane Image (EPI): irisan 2D dari light field.

### Langkah Kerja
1. Jalankan `python 18_light_field_basics.py`
2. Program membuat light field sintetis (5×5 views)
3. Menampilkan seluruh grid view
4. Mengekstrak Epipolar Plane Image (EPI)
5. Output di `output/18_light_field.png` dan `output/18_epi.png`

### Analisis
- Bagaimana objek bergeser antar view yang berbeda?
- Apa informasi yang terkandung dalam EPI?
- Bagaimana light field digunakan untuk refocusing?

---

## Percobaan 19: 3D Gaussian Splatting (Konsep)
**File:** `19_3d_gaussian_splatting_konsep.py`

### Dasar Teori
3D Gaussian Splatting (Kerbl et al., 2023) merepresentasikan scene sebagai kumpulan 3D Gaussian, masing-masing dengan posisi (μ), covariance (Σ), opacity (α), dan spherical harmonics warna (c). Rendering menggunakan differentiable splatting → real-time ~100 FPS.

### Langkah Kerja
1. Jalankan `python 19_3d_gaussian_splatting_konsep.py`
2. Program memvisualisasikan 3D Gaussians sebagai lingkaran 2D
3. Mendemonstrasikan splatting process
4. Menampilkan pipeline diagram 3DGS
5. Output di `output/19_3dgs_konsep.png`

### Analisis
- Apa parameter setiap Gaussian dalam 3DGS?
- Bagaimana 3DGS dibandingkan NeRF dalam kecepatan training/rendering?
- Mengapa 3DGS bisa real-time sedangkan NeRF tidak?

---

## Percobaan 20: Pipeline Rekonstruksi 3D Lengkap
**File:** `20_3d_reconstruction_pipeline.py`

### Dasar Teori
Pipeline end-to-end: Input (RGB + Depth) → Point Cloud → Filtering (Voxel Downsample) → Normal Estimation → Surface Reconstruction → Visualisasi 3D.

### Langkah Kerja
1. Jalankan `python 20_3d_reconstruction_pipeline.py`
2. Program membuat input RGB + Depth sintetis
3. Generate colored point cloud dari RGBD
4. Menerapkan voxel downsample
5. Visualisasi 6 tahap pipeline
6. Output di `output/20_pipeline.png`

### Analisis
- Berapa titik sebelum dan sesudah downsampling?
- Bagaimana distribusi depth pada histogram?
- Bagaimana pipeline ini bisa diperluas dengan mesh reconstruction?

---

## Pertanyaan Akhir
1. Jelaskan perbedaan representasi point cloud, mesh, dan volume!
2. Kapan menggunakan Poisson, BPA, atau Marching Cubes untuk surface reconstruction?
3. Apa kelebihan dan kekurangan NeRF dibandingkan 3D Gaussian Splatting?
4. Bagaimana TSDF integration mengkombinasikan multiple depth maps?
5. Jelaskan pipeline lengkap dari RGBD input hingga 3D model!

## Kesimpulan
Tuliskan kesimpulan dari seluruh percobaan yang telah dilakukan, meliputi pemahaman tentang representasi 3D, algoritma rekonstruksi permukaan, teknik rendering berbasis volume dan image, serta metode modern (NeRF, 3DGS).
