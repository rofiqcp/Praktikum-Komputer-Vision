# JOBSHEET MODUL 12: REKONSTRUKSI 3D DAN IMAGE-BASED RENDERING

---

## Tujuan Praktikum
Setelah menyelesaikan modul ini, mahasiswa mampu:
1. Memproses point cloud: loading, filtering, normal estimation, downsampling.
2. Melakukan registrasi point cloud menggunakan ICP.
3. Merekonstruksi surface dari point cloud (Poisson, BPA).
4. Melakukan integrasi volumetrik TSDF dari depth maps.
5. Memproses dan menyederhanakan mesh 3D.
6. Melakukan texture mapping dari gambar ke mesh.
7. Melakukan image warping berbasis depth.
8. Mengimplementasikan view interpolation sederhana.
9. Memahami konsep light field dan multiplane images.
10. Menjalankan neural scene representation (NeRF/3DGS) dasar.

---

## Alat dan Bahan

### Software
- Python 3.8+
- OpenCV 4.x (`opencv-python`, `opencv-contrib-python`)
- Open3D (`open3d`)
- NumPy, Matplotlib
- trimesh (`trimesh`)
- PyMeshLab (`pymeshlab`) — opsional
- nerfstudio atau nerfacc — opsional untuk Percobaan 10
- Jupyter Notebook / VS Code

### Hardware
- Komputer dengan GPU (direkomendasikan untuk Percobaan 10)
- Webcam atau kamera smartphone

### Dataset
- Bunny/Dragon/Armadillo dari Stanford 3D Scanning Repository
- COLMAP sample datasets
- DTU MVS dataset (subset)
- Indoor scene RGB-D data (TUM RGB-D / ICL-NUIM)
- Gambar multi-view sendiri (minimal 20 foto satu objek)

---

## Percobaan 1: Point Cloud Basics

### Tujuan
Memuat, memvisualisasikan, dan memahami struktur data point cloud.

### Langkah Kerja
1. **Install Open3D**: `pip install open3d`.
2. **Download sample data**: Unduh Bunny point cloud dari Stanford repository (PLY format).
3. **Load dan visualisasi**:
   ```python
   import open3d as o3d
   import numpy as np

   pcd = o3d.io.read_point_cloud("bunny.ply")
   print(f"Points: {len(pcd.points)}")
   print(f"Has normals: {pcd.has_normals()}")
   print(f"Has colors: {pcd.has_colors()}")
   o3d.visualization.draw_geometries([pcd])
   ```
4. **Konversi ke NumPy** dan analisis statistik:
   ```python
   points = np.asarray(pcd.points)
   print(f"Bounding box min: {points.min(axis=0)}")
   print(f"Bounding box max: {points.max(axis=0)}")
   print(f"Centroid: {points.mean(axis=0)}")
   ```
5. **Buat point cloud dari scratch**: Generate sphere point cloud dan visualisasikan.
6. **Paint point cloud**: Beri warna berdasarkan koordinat z (height colormap).
7. **Crop point cloud**: Gunakan `pcd.crop()` untuk memilih region tertentu.
8. **Export**: Simpan hasil dalam format PLY dan PCD.

### Analisis
- Berapa jumlah titik pada Bunny? Apa bounding box-nya?
- Apa perbedaan format PLY, PCD, dan XYZ?
- Bagaimana height colormap membantu visualisasi?

---

## Percobaan 2: Point Cloud Filtering dan Downsampling

### Tujuan
Menerapkan teknik preprocessing untuk membersihkan dan menyederhanakan point cloud.

### Langkah Kerja
1. **Tambahkan noise** ke point cloud bersih:
   ```python
   noise = np.random.normal(0, 0.005, size=points.shape)
   pcd_noisy = o3d.geometry.PointCloud()
   pcd_noisy.points = o3d.utility.Vector3dVector(points + noise)
   ```
2. **Voxel downsampling**:
   ```python
   pcd_down = pcd_noisy.voxel_down_sample(voxel_size=0.005)
   print(f"Before: {len(pcd_noisy.points)}, After: {len(pcd_down.points)}")
   ```
3. **Statistical Outlier Removal**:
   ```python
   cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
   pcd_clean = pcd_down.select_by_index(ind)
   pcd_outliers = pcd_down.select_by_index(ind, invert=True)
   ```
4. **Radius Outlier Removal**:
   ```python
   cl, ind = pcd_down.remove_radius_outlier(nb_points=16, radius=0.01)
   ```
5. **Variasi parameter**: Coba berbagai `voxel_size` (0.001, 0.005, 0.01, 0.02). Ukur jumlah titik dan kualitas visual.
6. **Variasi noise level**: Tambahkan noise σ = 0.001, 0.005, 0.01. Bandingkan efektivitas filtering.
7. **Visualisasi side-by-side**: Tampilkan noisy, downsampled, dan cleaned berdampingan.
8. **Ukur waktu** untuk setiap operasi.

### Analisis
- Bagaimana voxel_size mempengaruhi trade-off antara detail dan jumlah titik?
- Kapan statistical vs radius outlier removal lebih cocok?
- Pada noise level berapa filtering mulai kehilangan detail penting?

---

## Percobaan 3: Normal Estimation dan Orientasi

### Tujuan
Menghitung surface normals dan memahami pengaruhnya pada rekonstruksi.

### Langkah Kerja
1. **Estimate normals**:
   ```python
   pcd.estimate_normals(
       search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
   )
   ```
2. **Visualisasi normals**:
   ```python
   o3d.visualization.draw_geometries([pcd], point_show_normal=True)
   ```
3. **Orient normals** secara konsisten:
   ```python
   pcd.orient_normals_consistent_tangent_plane(k=15)
   # Atau orient terhadap viewpoint
   pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 5]))
   ```
4. **Variasi parameter radius**: Gunakan radius 0.005, 0.01, 0.02, 0.05. Perhatikan smoothness normal.
5. **Variasi max_nn**: Gunakan 10, 30, 50, 100 neighbors.
6. **Efek pada rekonstruksi**: Lakukan Poisson reconstruction dua kali — sekali dengan normal baik, sekali dengan random normal. Bandingkan.
7. **Normal map visualization**: Encode normal direction sebagai warna RGB (nx→R, ny→G, nz→B).
8. **Curvature estimation**: Hitung curvature lokal dari eigenvalue PCA. Visualisasikan high-curvature regions.

### Analisis
- Mengapa orientasi konsisten penting untuk Poisson reconstruction?
- Bagaimana radius dan max_nn mempengaruhi kualitas normal?
- Di area mana curvature tinggi pada Bunny model?

---

## Percobaan 4: Point Cloud Registration (ICP)

### Tujuan
Menyelaraskan dua point cloud menggunakan Iterative Closest Point.

### Langkah Kerja
1. **Buat dua "scan" berbeda** dengan menerapkan transformasi pada satu point cloud:
   ```python
   source = o3d.io.read_point_cloud("bunny.ply")
   target = o3d.geometry.PointCloud(source)

   # Rotasi + translasi
   T = np.eye(4)
   T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0.1, 0.2, 0.05])
   T[:3, 3] = [0.02, 0.01, -0.01]
   source.transform(T)
   ```
2. **Point-to-Point ICP**:
   ```python
   threshold = 0.02
   reg_p2p = o3d.pipelines.registration.registration_icp(
       source, target, threshold,
       np.eye(4),
       o3d.pipelines.registration.TransformationEstimationPointToPoint()
   )
   print(f"Fitness: {reg_p2p.fitness}")
   print(f"RMSE: {reg_p2p.inlier_rmse}")
   source.transform(reg_p2p.transformation)
   ```
3. **Point-to-Plane ICP** (memerlukan normals):
   ```python
   target.estimate_normals(...)
   reg_p2pl = o3d.pipelines.registration.registration_icp(
       source, target, threshold,
       np.eye(4),
       o3d.pipelines.registration.TransformationEstimationPointToPlane()
   )
   ```
4. **Bandingkan Point-to-Point vs Point-to-Plane**: Jumlah iterasi, fitness, RMSE, waktu.
5. **Colored ICP**: Jika point cloud berwarna:
   ```python
   reg_color = o3d.pipelines.registration.registration_colored_icp(
       source, target, threshold
   )
   ```
6. **Global registration** (RANSAC + FPFH features) sebagai initial alignment:
   ```python
   # Compute FPFH features
   source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
       source, o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=100))
   target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
       target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=100))

   result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
       source, target, source_fpfh, target_fpfh, True, threshold,
       o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
       3, [], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
   ```
7. **Pipeline lengkap**: Global registration → refine dengan ICP.
8. **Registrasi multi-scan**: Align 3+ scans secara berurutan. Visualisasi akumulasi.

### Analisis
- Mengapa Point-to-Plane ICP biasanya konvergen lebih cepat?
- Kapan global registration diperlukan vs langsung ICP?
- Apa efek accumulated drift pada multi-scan registration?

---

## Percobaan 5: Surface Reconstruction (Poisson dan BPA)

### Tujuan
Merekonstruksi mesh permukaan dari point cloud menggunakan berbagai metode.

### Langkah Kerja
1. **Persiapan**: Load point cloud, estimate normals, orient normals.
2. **Poisson Reconstruction**:
   ```python
   mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
       pcd, depth=9, width=0, scale=1.1, linear_fit=False
   )
   # Visualize densities
   densities = np.asarray(densities)
   density_colors = plt.get_cmap('plasma')(
       (densities - densities.min()) / (densities.max() - densities.min()))
   mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors[:, :3])
   ```
3. **Trim low-density regions**:
   ```python
   vertices_to_remove = densities < np.quantile(densities, 0.01)
   mesh.remove_vertices_by_mask(vertices_to_remove)
   ```
4. **Ball Pivoting Algorithm**:
   ```python
   radii = [0.005, 0.01, 0.02, 0.04]
   mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
       pcd, o3d.utility.DoubleVector(radii)
   )
   ```
5. **Alpha Shapes**:
   ```python
   mesh_alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
       pcd, alpha=0.03
   )
   ```
6. **Variasi depth Poisson** (6, 8, 9, 10, 12): Ukur triangle count, waktu, visual quality.
7. **Variasi radii BPA**: Coba single radius vs multi-radius. Perhatikan holes.
8. **Bandingkan tiga metode**: Tabel perbandingan (triangle count, time, quality, holes, watertight).
9. **Simpan mesh** dalam format OBJ dan PLY.

### Analisis
- Bagaimana depth parameter mempengaruhi detail pada Poisson reconstruction?
- Mengapa BPA sering menghasilkan mesh dengan holes?
- Kapan Alpha Shapes lebih cocok dibandingkan Poisson?

---

## Percobaan 6: Mesh Processing dan Simplification

### Tujuan
Memproses, menyederhanakan, dan memperbaiki mesh 3D.

### Langkah Kerja
1. **Load mesh** hasil rekonstruksi atau download mesh dari repository.
2. **Analisis mesh**:
   ```python
   mesh = o3d.io.read_triangle_mesh("reconstructed.ply")
   mesh.compute_vertex_normals()
   print(f"Vertices: {len(mesh.vertices)}")
   print(f"Triangles: {len(mesh.triangles)}")
   print(f"Is watertight: {mesh.is_watertight()}")
   print(f"Is self-intersecting: {mesh.is_self_intersecting()}")
   ```
3. **Mesh simplification** (Quadric Decimation):
   ```python
   targets = [50000, 10000, 5000, 1000]
   for target in targets:
       mesh_simple = mesh.simplify_quadric_decimation(
           target_number_of_triangles=target)
       print(f"Target {target}: actual {len(mesh_simple.triangles)} triangles")
   ```
4. **Laplacian Smoothing**:
   ```python
   mesh_smooth = mesh.filter_smooth_laplacian(number_of_iterations=10)
   mesh_smooth.compute_vertex_normals()
   ```
5. **Taubin Smoothing** (mengurangi shrinkage):
   ```python
   mesh_taubin = mesh.filter_smooth_taubin(number_of_iterations=10)
   ```
6. **Subdivision**:
   ```python
   mesh_subdiv = mesh.subdivide_midpoint(number_of_iterations=1)
   mesh_subdiv_loop = mesh.subdivide_loop(number_of_iterations=1)
   ```
7. **Menggunakan trimesh** untuk operasi tambahan:
   ```python
   import trimesh
   mesh_t = trimesh.load("result.ply")
   # Boolean operations
   mesh_a = trimesh.creation.box()
   mesh_b = trimesh.creation.icosphere()
   result = trimesh.boolean.intersection([mesh_a, mesh_b])
   # Convex hull
   hull = mesh_t.convex_hull
   # Volume
   print(f"Volume: {mesh_t.volume}")
   ```
8. **Pipeline lengkap**: Load → simplify → smooth → compute normals → export.
9. **Visualisasi wireframe**: Tampilkan mesh sebagai wireframe untuk melihat topology.

### Analisis
- Berapa level simplification yang masih mempertahankan bentuk objek?
- Apa perbedaan Laplacian vs Taubin smoothing?
- Bagaimana subdivision meningkatkan kualitas visual mesh?

---

## Percobaan 7: TSDF Volume Integration

### Tujuan
Melakukan rekonstruksi volumetrik dari sequence RGB-D menggunakan TSDF.

### Langkah Kerja
1. **Download RGB-D dataset**: Gunakan TUM RGB-D atau buat sendiri dari Kinect/RealSense.
2. **Load RGB-D data**:
   ```python
   color_raw = o3d.io.read_image("color_000.png")
   depth_raw = o3d.io.read_image("depth_000.png")
   rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
       color_raw, depth_raw, depth_trunc=3.0, convert_rgb_to_intensity=False
   )
   ```
3. **Setup TSDF volume**:
   ```python
   volume = o3d.pipelines.integration.ScalableTSDFVolume(
       voxel_length=4.0/512.0,
       sdf_trunc=0.04,
       color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
   )
   ```
4. **Integrate multiple frames**:
   ```python
   intrinsic = o3d.camera.PinholeCameraIntrinsic(
       o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
   for i in range(num_frames):
       rgbd = load_rgbd(i)
       extrinsic = load_pose(i)  # 4x4 matrix
       volume.integrate(rgbd, intrinsic, extrinsic)
   ```
5. **Extract mesh**:
   ```python
   mesh = volume.extract_triangle_mesh()
   mesh.compute_vertex_normals()
   o3d.visualization.draw_geometries([mesh])
   ```
6. **Extract point cloud**: `pcd = volume.extract_point_cloud()`.
7. **Variasi voxel_length**: 0.5/512, 2/512, 4/512, 8/512. Bandingkan detail vs memory.
8. **Variasi sdf_trunc**: 0.02, 0.04, 0.08. Perhatikan efek pada surface.
9. **Tambahkan camera tracking** (frame-to-model ICP) untuk pose estimation otomatis.
10. **Simpan hasil** dalam PLY dan OBJ.

### Analisis
- Bagaimana voxel_length mempengaruhi resolusi dan memory?
- Apa efek jumlah frame terhadap kualitas rekonstruksi?
- Apa keuntungan TSDF dibandingkan point cloud biasa?

---

## Percobaan 8: Image Warping dan View Synthesis

### Tujuan
Melakukan depth-based image warping untuk menghasilkan novel views.

### Langkah Kerja
1. **Load stereo pair + depth** (gunakan hasil dari Modul 11 atau dataset):
   ```python
   import cv2
   img_left = cv2.imread("left.png")
   depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
   ```
2. **Definisikan camera intrinsics** (K matrix).
3. **Forward warping**:
   ```python
   def forward_warp(img, depth, K, R, t):
       h, w = img.shape[:2]
       output = np.zeros_like(img)
       zbuffer = np.full((h, w), np.inf)
       fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
       for v in range(h):
           for u in range(w):
               d = depth[v, u]
               if d <= 0: continue
               X = (u - cx) * d / fx
               Y = (v - cy) * d / fy
               P3d = R @ np.array([X, Y, d]) + t
               if P3d[2] <= 0: continue
               u2 = int(fx * P3d[0] / P3d[2] + cx)
               v2 = int(fy * P3d[1] / P3d[2] + cy)
               if 0 <= u2 < w and 0 <= v2 < h:
                   if P3d[2] < zbuffer[v2, u2]:
                       zbuffer[v2, u2] = P3d[2]
                       output[v2, u2] = img[v, u]
       return output
   ```
4. **Identifikasi disoccluded pixels** (holes) pada forward warp output.
5. **Inpainting holes**:
   ```python
   mask = (output.sum(axis=2) == 0).astype(np.uint8) * 255
   output_inpainted = cv2.inpaint(output, mask, 3, cv2.INPAINT_TELEA)
   ```
6. **Vectorized forward warp** (faster):
   ```python
   def forward_warp_fast(img, depth, K, R, t):
       h, w = depth.shape
       u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
       fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
       valid = depth > 0
       X = (u_coords[valid] - cx) * depth[valid] / fx
       Y = (v_coords[valid] - cy) * depth[valid] / fy
       Z = depth[valid]
       pts = np.stack([X, Y, Z], axis=1)
       pts_new = (R @ pts.T).T + t
       u_new = (fx * pts_new[:, 0] / pts_new[:, 2] + cx).astype(int)
       v_new = (fy * pts_new[:, 1] / pts_new[:, 2] + cy).astype(int)
       # ... z-buffer dan assign
       return output
   ```
7. **Buat animasi fly-through**: Warp gambar ke serangkaian viewpoint, simpan sebagai video.
8. **Visualisasi depth** sebagai color-mapped image.

### Analisis
- Di area mana saja disocclusion paling banyak terjadi? Mengapa?
- Bandingkan kecepatan loop vs vectorized warping.
- Bagaimana kualitas inpainting mempengaruhi hasil akhir?

---

## Percobaan 9: View Interpolation dan Blending

### Tujuan
Menghasilkan view antara dari dua gambar dengan depth menggunakan interpolasi.

### Langkah Kerja
1. **Load dua view dengan depth**:
   ```python
   img1 = cv2.imread("view1.png")
   img2 = cv2.imread("view2.png")
   depth1 = load_depth("depth1.png")
   depth2 = load_depth("depth2.png")
   ```
2. **Definisikan viewpoints**: Pose kamera 1 dan kamera 2.
3. **Interpolasi pose** pada parameter $t \in [0, 1]$:
   ```python
   from scipy.spatial.transform import Rotation, Slerp
   
   R1 = Rotation.from_matrix(R_cam1)
   R2 = Rotation.from_matrix(R_cam2)
   slerp = Slerp([0, 1], Rotation.concatenate([R1, R2]))
   
   for t in np.linspace(0, 1, 30):
       R_interp = slerp(t).as_matrix()
       t_interp = (1-t) * t_cam1 + t * t_cam2
   ```
4. **Warp kedua view** ke intermediate viewpoint:
   ```python
   warped1 = forward_warp(img1, depth1, K, R_interp @ R1.T, t_interp - R_interp @ R1.T @ t_cam1)
   warped2 = forward_warp(img2, depth2, K, R_interp @ R2.T, t_interp - R_interp @ R2.T @ t_cam2)
   ```
5. **Blend kedua warped views**:
   ```python
   # Simple linear blend
   mask1 = (warped1.sum(axis=2) > 0).astype(float)
   mask2 = (warped2.sum(axis=2) > 0).astype(float)
   weight1 = mask1 * (1 - t)
   weight2 = mask2 * t
   total = weight1 + weight2 + 1e-8
   blended = (warped1 * weight1[..., None] + warped2 * weight2[..., None]) / total[..., None]
   ```
6. **Handle disocclusion**: Area yang tidak terlihat di satu view, gunakan view lainnya.
7. **Buat video interpolasi** (30 frame dari view1 ke view2):
   ```python
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter('interpolation.mp4', fourcc, 15, (w, h))
   for frame in frames:
       out.write(frame.astype(np.uint8))
   out.release()
   ```
8. **Bandingkan** simple blend vs Laplacian pyramid blend.
9. **Interpolasi tanpa depth** (optical flow based): Gunakan Farnebäck flow dari Modul 9 sebagai perbandingan.

### Analisis
- Di mana artefak paling terlihat? Mengapa?
- Apakah depth-based interpolation lebih baik dari flow-based?
- Bagaimana kualitas depth mempengaruhi hasil interpolasi?

---

## Percobaan 10: Neural Scene Representation (NeRF / 3D Gaussian Splatting)

### Tujuan
Menjalankan pipeline neural rendering untuk menghasilkan novel view synthesis berkualitas tinggi.

### Langkah Kerja

> **Catatan**: Percobaan ini membutuhkan GPU. Jika tidak tersedia, gunakan Google Colab atau jalankan versi simplified.

1. **Persiapkan dataset multi-view** (pilih salah satu):
   - Download sample dataset (NeRF synthetic: lego, chair, etc.).
   - Buat sendiri: foto objek dari 20-50 sudut berbeda.

2. **Jalankan COLMAP** untuk mendapatkan camera poses:
   ```bash
   colmap feature_extractor --database_path db.db --image_path ./images
   colmap exhaustive_matcher --database_path db.db
   colmap mapper --database_path db.db --image_path ./images --output_path ./sparse
   ```

3. **Opsi A: NeRF dengan nerfstudio**:
   ```bash
   pip install nerfstudio
   ns-process-data images --data ./images --output-dir ./processed
   ns-train nerfacto --data ./processed
   ns-viewer --load-config outputs/processed/nerfacto/config.yml
   ```

4. **Opsi B: Simplified NeRF (Tiny NeRF)**:
   ```python
   import torch
   import torch.nn as nn

   class TinyNeRF(nn.Module):
       def __init__(self, pos_encoding_dim=6):
           super().__init__()
           input_dim = 3 + 3 * 2 * pos_encoding_dim
           self.net = nn.Sequential(
               nn.Linear(input_dim, 256), nn.ReLU(),
               nn.Linear(256, 256), nn.ReLU(),
               nn.Linear(256, 256), nn.ReLU(),
               nn.Linear(256, 4)  # RGB + sigma
           )

       def forward(self, x):
           return self.net(x)

   def positional_encoding(x, L=6):
       encodings = [x]
       for i in range(L):
           encodings.append(torch.sin(2**i * np.pi * x))
           encodings.append(torch.cos(2**i * np.pi * x))
       return torch.cat(encodings, dim=-1)
   ```

5. **Volume rendering**:
   ```python
   def render_rays(model, rays_o, rays_d, near, far, num_samples=64):
       t = torch.linspace(near, far, num_samples)
       pts = rays_o[..., None, :] + rays_d[..., None, :] * t[..., :, None]
       flat_pts = pts.reshape(-1, 3)
       encoded = positional_encoding(flat_pts)
       raw = model(encoded).reshape(*pts.shape[:-1], 4)
       rgb = torch.sigmoid(raw[..., :3])
       sigma = torch.relu(raw[..., 3])
       # Alpha compositing
       dists = t[..., 1:] - t[..., :-1]
       alpha = 1 - torch.exp(-sigma[..., :-1] * dists)
       T = torch.cumprod(1 - alpha + 1e-10, dim=-1)
       T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)
       weights = alpha * T
       rgb_map = (weights[..., None] * rgb[..., :-1, :]).sum(dim=-2)
       return rgb_map
   ```

6. **Opsi C: 3D Gaussian Splatting** (jika GPU memadai):
   ```bash
   git clone https://github.com/graphdeco-inria/gaussian-splatting
   python train.py -s ./data/my_scene
   python render.py -m ./output
   ```

7. **Evaluasi kualitas** (PSNR, SSIM):
   ```python
   from skimage.metrics import structural_similarity as ssim
   psnr = 10 * np.log10(255**2 / np.mean((rendered - ground_truth)**2))
   ssim_val = ssim(rendered, ground_truth, multichannel=True)
   ```

8. **Render novel views**: Generate video orbit di sekitar objek.

9. **Bandingkan** pendekatan tradisional (mesh + texture) vs neural (NeRF/3DGS) untuk kualitas dan kecepatan.

10. **Dokumentasikan** setup, training time, dan hasil visual.

### Analisis
- Berapa lama training NeRF/3DGS pada dataset Anda?
- Bagaimana kualitas novel view dibandingkan input views?
- Apa trade-off antara NeRF (implicit) vs 3DGS (explicit)?
- Kapan pendekatan tradisional lebih praktis dari neural rendering?

---

## Kesimpulan
Setelah menyelesaikan 10 percobaan, mahasiswa memahami:
1. Pipeline lengkap dari point cloud mentah hingga mesh 3D bertekstur.
2. Teknik filtering, registration, dan surface reconstruction.
3. Volumetric integration (TSDF) untuk dense reconstruction.
4. Depth-based image warping dan view interpolation.
5. Konsep dan implementasi dasar neural scene representation.
6. Trade-off antara metode tradisional dan neural rendering.

---

## Referensi
1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed. Ch. 13–14.
2. Open3D Documentation. http://www.open3d.org/docs/
3. COLMAP Documentation. https://colmap.github.io/
4. Mildenhall, B. et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields. *ECCV*.
5. Kerbl, B. et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM ToG*.
