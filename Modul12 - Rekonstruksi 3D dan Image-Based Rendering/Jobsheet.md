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

## Percobaan 11: Ball Pivoting Algorithm (BPA) Detail

### Tujuan
Memahami dan mengimplementasikan Ball Pivoting Algorithm secara mendalam, menganalisis pengaruh pemilihan radius bola terhadap kualitas mesh yang dihasilkan, serta membandingkan BPA dengan fallback Delaunay triangulation.

### Dasar Teori
Ball Pivoting Algorithm (BPA) merekonstruksi surface mesh dari point cloud dengan cara "menggelindingkan" bola virtual di atas titik-titik. Bola dengan radius $r$ menyentuh tiga titik sekaligus dan membentuk segitiga. Pemilihan radius menentukan detail dan coverage mesh: radius terlalu kecil menghasilkan banyak holes, radius terlalu besar menghasilkan mesh yang over-smoothed. Penggunaan multiple radii (multi-pass) memungkinkan BPA menangkap detail halus sekaligus mengisi area sparse. Ketika Open3D tidak tersedia, SciPy `Delaunay()` digunakan sebagai fallback untuk triangulasi.

### Langkah Kerja
1. **Load point cloud** dari file PLY atau generate point cloud sintetis (bunny/torus) menggunakan `load_ply_manual()`.
2. **Estimasi normal** menggunakan PCA pada k-nearest neighbors dengan fungsi `estimate_normals_simple(points, k=10)`.
3. **BPA dengan single radius**: Jalankan `create_from_point_cloud_ball_pivoting()` dengan satu radius (misal `radii=[0.005]`). Hitung jumlah triangles dan coverage.
4. **BPA multi-radius**: Gunakan beberapa radius `radii=[0.005, 0.01, 0.02, 0.04]` secara berurutan. Bandingkan coverage terhadap single radius.
5. **Variasi radius**: Uji radius kecil (0.003), sedang (0.01), dan besar (0.05). Catat triangle count, holes, dan waktu eksekusi.
6. **Fallback Delaunay**: Jika Open3D tidak tersedia, gunakan `scipy.spatial.Delaunay()` dan filter simplex berdasarkan circumradius (mirip alpha shape).
7. **Visualisasi perbandingan**: Tampilkan mesh dari berbagai radius secara side-by-side menggunakan `Poly3DCollection` pada subplot matplotlib.
8. **Analisis coverage**: Hitung persentase titik yang "tertutup" mesh dan identifikasi area dengan holes menggunakan operasi numpy.

### Analisis
- Bagaimana pemilihan radius bola mempengaruhi jumlah triangle dan coverage mesh?
- Mengapa multi-radius BPA menghasilkan mesh yang lebih lengkap dibandingkan single radius?
- Di area mana holes paling sering muncul, dan apa hubungannya dengan densitas point cloud?
- Bagaimana performa Delaunay fallback dibandingkan BPA Open3D dalam hal kualitas mesh?

---

## Percobaan 12: Alpha Shapes

### Tujuan
Memahami konsep Alpha Shapes sebagai generalisasi convex hull yang memungkinkan rekonstruksi bentuk concave, serta menganalisis pengaruh parameter alpha terhadap hasil rekonstruksi.

### Dasar Teori
Alpha Shape adalah generalisasi dari convex hull. Algoritma ini dimulai dari Delaunay triangulation, lalu menghapus simplex yang circumradius-nya lebih besar dari $1/\alpha$. Ketika $\alpha \to 0$, hasilnya mendekati convex hull; ketika $\alpha$ besar, hasilnya menangkap detail concave. Formula circumradius segitiga dengan sisi $a, b, c$ dan luas $A$ adalah $R = \frac{abc}{4A}$. `scipy.spatial.ConvexHull()` digunakan untuk perbandingan, dan `scipy.spatial.Delaunay()` untuk triangulasi dasar.

### Langkah Kerja
1. **Buat point cloud sintetis** berbentuk torus menggunakan `generate_sample_point_cloud(num_points=2000)` untuk bentuk concave yang jelas.
2. **Hitung Delaunay triangulation** dengan `scipy.spatial.Delaunay(points)` sebagai base triangulasi.
3. **Implementasi alpha filtering**: Gunakan fungsi `alpha_shape_delaunay(points, alpha)` yang menghitung `compute_circumradius(p1, p2, p3)` dan menghapus simplex dengan circumradius $> 1/\alpha$.
4. **Variasi parameter alpha**: Uji $\alpha = 0.5, 1.0, 2.0, 5.0, 10.0$. Catat jumlah simplex yang lolos filter.
5. **Bandingkan dengan Convex Hull**: Hitung `ConvexHull(points)` dan bandingkan secara visual — convex hull gagal menangkap lubang torus.
6. **Alpha Shape dengan Open3D**: Gunakan `create_from_point_cloud_alpha_shape(pcd, alpha=0.03)` dan bandingkan hasilnya.
7. **Visualisasi perbandingan**: Tampilkan convex hull, alpha shape kecil, dan alpha shape besar secara berdampingan dengan `Poly3DCollection`.
8. **Analisis boundary edges**: Identifikasi tepi-tepi yang membentuk batas alpha shape dan visualisasikan sebagai wireframe.

### Analisis
- Bagaimana parameter alpha mengontrol trade-off antara convexity dan detail concave?
- Pada nilai alpha berapa bentuk torus mulai terlihat jelas (lubang tengah muncul)?
- Mengapa convex hull tidak cocok untuk merekonstruksi objek dengan concavity?
- Apa hubungan antara circumradius threshold dan densitas lokal point cloud?

---

## Percobaan 13: Texture Mapping pada Mesh

### Tujuan
Mempelajari cara memproyeksikan warna dan tekstur dari gambar 2D ke permukaan mesh 3D menggunakan camera projection model dan interpolasi warna.

### Dasar Teori
Texture mapping memetakan gambar 2D ke permukaan mesh 3D melalui proyeksi kamera. Dengan matriks intrinsik $K$ (focal length, principal point), vektor rotasi $\mathbf{r}$, dan vektor translasi $\mathbf{t}$, setiap vertex 3D diproyeksikan ke pixel 2D menggunakan `cv2.projectPoints()`. Warna pada posisi tersebut disampling dari gambar tekstur. Teknik ini menggunakan `matplotlib.tri.Triangulation` dan `plot_trisurf()` untuk visualisasi mesh bertekstur.

### Langkah Kerja
1. **Buat mesh terrain** menggunakan `create_terrain_mesh(grid_size=30)` yang menghasilkan permukaan berbukit dengan fungsi sinusoidal $Z = 0.5\sin(1.5X)\cos(1.5Y) + 0.3\sin(3X)$.
2. **Buat gambar tekstur** sintetis (checkerboard berwarna + gradien) menggunakan `create_texture_image(width=512, height=512)`.
3. **Definisikan parameter kamera**: Buat matriks intrinsik $K$, `rvec`, `tvec`, dan `dist_coeffs` dengan `create_camera_parameters()`.
4. **Proyeksikan vertices** ke bidang gambar menggunakan `cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)`.
5. **Sampling warna**: Gunakan `sample_colors_from_image(image, image_points)` untuk mengambil warna RGB dari tekstur pada posisi proyeksi.
6. **Visualisasi mesh bertekstur**: Tampilkan mesh 3D dengan `plot_trisurf()` menggunakan `facecolors` dari vertex colors.
7. **Variasi viewpoint kamera**: Ubah `rvec` dan `tvec` untuk berbagai sudut pandang dan amati perubahan pemetaan tekstur.
8. **Bandingkan metode**: Bandingkan kualitas visual antara vertex coloring, nearest-neighbor sampling, dan interpolasi bilinear.

### Analisis
- Bagaimana perubahan posisi kamera mempengaruhi distorsi pemetaan tekstur?
- Di area mana terjadi stretching atau kompresi tekstur, dan mengapa?
- Apa pengaruh resolusi tekstur terhadap kualitas visual mesh bertekstur?
- Mengapa beberapa vertex memiliki warna default (area di luar gambar)?

---

## Percobaan 14: Colorization Point Cloud

### Tujuan
Mempelajari berbagai teknik pewarnaan point cloud: berdasarkan ketinggian (Z-height), jarak dari centroid, arah normal, dan curvature, menggunakan colormap untuk visualisasi informasi geometris.

### Dasar Teori
Colorization point cloud memetakan atribut geometris ke warna untuk memudahkan analisis visual. Height colormap menggunakan koordinat $z$ yang dinormalisasi ke colormap (plasma, viridis). Distance colormap menggunakan jarak Euclidean $d = \|\mathbf{p} - \mathbf{c}\|$ dari centroid. Normal direction colormap meng-encode vektor normal $(n_x, n_y, n_z)$ ke RGB. Curvature colormap mengukur variasi normal pada tetangga: $\kappa_i = \frac{1}{k}\sum_{j \in \mathcal{N}(i)} (1 - |n_i \cdot n_j|)$.

### Langkah Kerja
1. **Muat point cloud** dari file PLY atau generate sintetis (bola bergelombang) menggunakan `generate_sample_point_cloud(num_points=3000)`.
2. **Height colormap**: Normalisasi koordinat z ke [0, 1] dan petakan ke colormap `plt.cm.plasma()`. Visualisasikan dengan scatter 3D.
3. **Distance colormap**: Hitung jarak setiap titik dari centroid menggunakan `np.linalg.norm()`. Petakan ke colormap `plt.cm.viridis()`.
4. **Estimasi normal**: Gunakan `estimate_normals_pca(points, k=15)` dengan dekomposisi eigen pada matriks kovarians tetangga.
5. **Normal direction colormap**: Encode komponen normal sebagai warna: $R = |n_x|$, $G = |n_y|$, $B = |n_z|$.
6. **Estimasi curvature**: Gunakan `estimate_curvature(points, normals, k=15)` yang menghitung rata-rata perbedaan normal dengan tetangga.
7. **Curvature colormap**: Petakan curvature ke colormap `plt.cm.hot()` untuk menyoroti area tajam (high curvature).
8. **Perbandingan subplot**: Tampilkan keempat metode colorization secara berdampingan dalam satu figure dan simpan ke output.

### Analisis
- Informasi geometris apa yang paling mudah diidentifikasi dengan height colormap vs curvature colormap?
- Di area mana curvature tinggi pada objek sintetis, dan mengapa?
- Bagaimana jumlah tetangga (k) pada estimasi normal mempengaruhi result curvature?
- Apa kelebihan normal direction colormap dibandingkan height colormap untuk analisis geometri surface?

---

## Percobaan 15: Segmentasi Point Cloud

### Tujuan
Mempelajari teknik segmentasi point cloud menggunakan RANSAC plane fitting untuk mendeteksi bidang datar, dan clustering berbasis Euclidean distance (mirip DBSCAN) untuk memisahkan objek-objek individual.

### Dasar Teori
Segmentasi point cloud bertujuan memisahkan region yang berbeda. RANSAC plane fitting memilih 3 titik acak, menghitung persamaan plane $ax + by + cz + d = 0$ dengan normal $\mathbf{n} = (\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1)$, dan menghitung inlier (titik dengan jarak $< \epsilon$). Proses diulang dan model terbaik dipilih. Euclidean clustering mirip DBSCAN menggunakan `KDTree` untuk mencari tetangga dalam radius $\epsilon$ dan melakukan BFS/floodfill untuk membentuk cluster.

### Langkah Kerja
1. **Buat scene sintetis**: Gunakan `generate_scene_point_cloud()` yang menghasilkan lantai datar (plane) + bola + kubus + silinder, total ~3300 titik dengan label ground truth.
2. **RANSAC plane fitting**: Jalankan `ransac_plane_fitting(points, num_iterations=1000, distance_threshold=0.05)` untuk mendeteksi bidang lantai.
3. **Pisahkan inlier/outlier**: Pisahkan titik lantai (inlier) dan objek (outlier) berdasarkan hasil RANSAC.
4. **Euclidean clustering**: Jalankan `euclidean_clustering(points_outlier, eps=0.3, min_points=10)` pada titik non-lantai untuk memisahkan bola, kubus, dan silinder.
5. **Variasi parameter RANSAC**: Ubah `distance_threshold` (0.02, 0.05, 0.1) dan `num_iterations` (100, 500, 1000). Amati efek pada akurasi plane detection.
6. **Variasi parameter clustering**: Ubah `eps` (0.1, 0.3, 0.5) dan `min_points` (5, 10, 20). Hitung jumlah cluster yang terbentuk.
7. **Evaluasi**: Bandingkan label cluster dengan label ground truth. Hitung akurasi segmentasi untuk setiap objek.
8. **Visualisasi**: Tampilkan scene dengan warna per-cluster dan per-label ground truth secara side-by-side.

### Analisis
- Berapa iterasi RANSAC minimal yang dibutuhkan untuk mendeteksi plane secara konsisten?
- Bagaimana parameter `eps` pada clustering mempengaruhi jumlah cluster dan noise assignment?
- Apakah segmentasi berhasil memisahkan semua objek? Objek mana yang paling sulit?
- Apa kelemahan pendekatan RANSAC + clustering dibandingkan metode learning-based?

---

## Percobaan 16: Marching Cubes

### Tujuan
Memahami algoritma Marching Cubes untuk mengekstrak mesh isosurface dari volumetric data (scalar field / implicit function), serta menganalisis pengaruh resolusi grid dan bentuk implicit function.

### Dasar Teori
Marching Cubes mengekstrak isosurface (permukaan dengan nilai konstan) dari 3D scalar field. Setiap sel (voxel) diperiksa: jika beberapa sudut di atas dan beberapa di bawah threshold, maka surface memotong sel tersebut. Posisi vertex mesh di-interpolasi di tepi sel. Untuk bola, implicit function adalah $f(x,y,z) = x^2 + y^2 + z^2 - r^2$. Untuk torus: $f = (\sqrt{x^2 + y^2} - R)^2 + z^2 - r^2$. `skimage.measure.marching_cubes()` digunakan jika tersedia, dengan fallback manual.

### Langkah Kerja
1. **Buat scalar field bola**: Definisikan grid 3D resolusi $50^3$ dan hitung $f = x^2 + y^2 + z^2 - r^2$ dengan $r = 1.0$.
2. **Ekstraksi isosurface bola**: Jalankan `marching_cubes(field_bola, level=0.0)` dari scikit-image atau gunakan `marching_cubes_manual()` sebagai fallback.
3. **Buat scalar field torus**: Hitung $f = (\sqrt{x^2 + y^2} - R)^2 + z^2 - r^2$ dengan $R = 1.0$, $r = 0.4$.
4. **Kombinasi implicit functions**: Gunakan operasi `np.minimum()` (union) dan `np.maximum()` (intersection) untuk membuat bentuk CSG (Constructive Solid Geometry).
5. **Variasi resolusi**: Uji resolusi grid $20^3$, $50^3$, $80^3$, $100^3$. Catat jumlah vertex, faces, dan waktu komputasi.
6. **Variasi threshold**: Ubah level set (0.0, 0.1, -0.1) untuk melihat efek pada ukuran isosurface yang dihasilkan.
7. **Visualisasi mesh**: Tampilkan mesh menggunakan `Poly3DCollection` dan `plot_trisurf()` dengan lighting/shading.
8. **Export mesh**: Simpan hasil dalam format PLY menggunakan Open3D atau manual write.

### Analisis
- Bagaimana resolusi grid mempengaruhi trade-off antara detail mesh dan waktu komputasi?
- Apa efek perubahan threshold/level pada bentuk isosurface yang diekstrak?
- Mengapa implementasi manual marching cubes menghasilkan kualitas lebih rendah dari scikit-image?
- Bagaimana operasi CSG (union, intersection) pada implicit function memungkinkan pembuatan bentuk kompleks?

---

## Percobaan 17: Volumetric Rendering Sederhana

### Tujuan
Memahami konsep dasar volumetric rendering: ray casting melalui data volume 3D untuk menghasilkan gambar 2D, termasuk teknik Maximum Intensity Projection (MIP), average projection, dan first-surface rendering.

### Dasar Teori
Volumetric rendering menampilkan data 3D tanpa perlu ekstraksi surface mesh. Sinar (ray) ditembakkan dari kamera melalui setiap piksel dan disample sepanjang volume. Teknik MIP mencatat nilai density maksimum $I = \max_t \rho(\mathbf{o} + t\mathbf{d})$; average projection merata-ratakan semua sampel $I = \frac{1}{N}\sum_t \rho_t$; first-surface rendering berhenti pada sampel pertama di atas threshold. Arah ray dihitung dari posisi kamera, field of view, dan vektor up/right.

### Langkah Kerja
1. **Buat volume 3D**: Definisikan grid voxel $64^3$ dengan dua bola (besar dan kecil) dan satu kubus menggunakan implicit function dengan smooth falloff.
2. **Definisikan kamera virtual**: Tentukan posisi kamera, arah pandang, vektor up, dan FOV. Generate ray directions menggunakan fungsi `buat_rays()`.
3. **MIP rendering**: Jalankan `ray_cast_volume(volume, pos_kamera, ray_dirs, mode='mip', n_samples=100)`. Untuk setiap ray, simpan nilai density maksimum.
4. **Average projection**: Jalankan dengan `mode='average'`. Akumulasikan density semua sampel dan bagi dengan jumlah sampel.
5. **First-surface rendering**: Jalankan dengan `mode='first_surface', threshold=0.3`. Berhenti pada sampel pertama di atas threshold untuk efek opaque surface.
6. **Multi-view rendering**: Render volume dari 4 sudut pandang berbeda (depan, samping, atas, diagonal).
7. **Variasi jumlah sampel**: Uji `n_samples` = 30, 100, 200. Bandingkan kualitas dan waktu rendering.
8. **Simpan hasil**: Simpan gambar rendering dari berbagai mode dan viewpoint ke direktori output.

### Analisis
- Apa perbedaan visual antara MIP, average projection, dan first-surface rendering?
- Bagaimana jumlah sampel per ray mempengaruhi kualitas dan kecepatan rendering?
- Mengapa MIP cocok untuk visualisasi data medis (CT/MRI) tetapi kurang realistis untuk objek padat?
- Bagaimana resolusi volume ($64^3$ vs $128^3$) mempengaruhi detail dan waktu render?

---

## Percobaan 18: Forward vs Inverse Warping Detail

### Tujuan
Memahami perbedaan mendasar antara forward warping dan inverse warping secara mendalam, termasuk masalah holes, splatting, z-buffer, dan interpolasi bilinear, serta teknik inpainting untuk memperbaiki artefak.

### Dasar Teori
Forward warping memetakan setiap piksel sumber ke posisi target berdasarkan disparitas, mengakibatkan holes (piksel target tanpa sumber) dan memerlukan z-buffer untuk menangani oklusi. Inverse warping (menggunakan `cv2.remap()`) menghitung untuk setiap piksel target dari mana sampel diambil di gambar sumber, menghindari holes tetapi memerlukan inverse mapping. Splatting mengatasi holes pada forward warp dengan menyebarkan kontribusi ke neighboring pixels. `cv2.inpaint()` dengan metode Telea atau Navier-Stokes dapat menutup holes.

### Langkah Kerja
1. **Buat gambar test**: Buat checkerboard berwarna dengan lingkaran, persegi, dan teks "WARP", atau muat dari file. Buat depth map sintetis dengan gradien dan foreground objects.
2. **Hitung disparitas**: Gunakan fungsi `hitung_disparitas(depth, baseline, focal)` dengan rumus disparitas $= \frac{baseline \times focal}{depth \times width}$.
3. **Forward warping**: Implementasikan `forward_warp(gambar, disparitas)` dengan z-buffer per-piksel: iterasi setiap piksel sumber, hitung posisi target, update hanya jika z-valuenya lebih kecil.
4. **Deteksi dan hitung holes**: Identifikasi piksel kosong (mask_filled == 0) dan hitung persentase holes.
5. **Splatting 3×3**: Implementasikan forward warp dengan splatting kernel — setiap piksel sumber berkontribusi ke area 3×3 di target.
6. **Inverse warping**: Gunakan `cv2.remap()` dengan map_x dan map_y yang dihitung dari disparitas. Bandingkan hasilnya — tidak ada holes.
7. **Inpainting holes**: Terapkan `cv2.inpaint(forward_result, mask, 3, cv2.INPAINT_TELEA)` untuk mengisi holes pada forward warping.
8. **Perbandingan artefak**: Tampilkan side-by-side: forward (holes), forward+splatting, forward+inpaint, dan inverse warp. Analisis kelebihan/kekurangan tiap metode.

### Analisis
- Di area mana holes paling banyak terjadi pada forward warping, dan apa hubungannya dengan depth discontinuity?
- Bagaimana splatting 3×3 mengurangi holes tetapi menyebabkan blurring?
- Mengapa inverse warping tidak memiliki holes? Apa kelemahannya?
- Seberapa efektif inpainting Telea dalam memperbaiki area disoccluded — kapan gagal?

---

## Percobaan 19: Dasar-Dasar Light Field

### Tujuan
Memahami konsep light field sebagai representasi 4D cahaya $L(u, v, s, t)$, mensimulasikan light field array dari scene sintetis, dan mengimplementasikan synthetic aperture refocusing dengan teknik shift-and-add.

### Dasar Teori
Light field $L(u, v, s, t)$ merepresentasikan semua sinar cahaya di ruang 3D, dimana $(u, v)$ adalah posisi kamera pada bidang kamera dan $(s, t)$ adalah koordinat piksel pada bidang gambar. Array kamera $N \times N$ mensimulasikan pengambilan light field. Sub-aperture image adalah irisan pada $(u_0, v_0)$ tertentu. Synthetic refocusing dilakukan dengan shift-and-add: setiap sub-aperture view digeser berdasarkan disparitas pada depth target $d_{focus}$, lalu dirata-ratakan. Objek pada $d_{focus}$ tampak tajam, lainnya blur.

### Langkah Kerja
1. **Buat scene sintetis**: Buat gambar 200×200 dengan tiga objek pada depth berbeda — persegi merah (dekat, $d=0.2$), lingkaran hijau (tengah, $d=0.5$), segitiga biru (jauh, $d=0.9$).
2. **Konfigurasi light field**: Tentukan grid kamera 5×5 dengan baseline 3.0 piksel. Alokasikan array 4D `light_field[u, v, s, t, channels]`.
3. **Simulasi 25 views**: Untuk setiap posisi kamera $(u, v)$, hitung disparitas $= offset / depth$ dan gunakan `cv2.remap()` untuk warping inverse dari scene dasar.
4. **Visualisasi sub-aperture images**: Tampilkan grid 5×5 dari semua views untuk melihat pergeseran paralaks antar objek.
5. **Shift-and-add refocusing**: Implementasikan refocusing dengan depth focus $d_{focus} = 0.2$ (fokus dekat), $0.5$ (tengah), $0.9$ (jauh). Geser setiap view sebesar $-offset / d_{focus}$ dan rata-ratakan.
6. **Visualisasi refocus results**: Tampilkan tiga gambar refocus side-by-side — amati perbedaan ketajaman objek.
7. **EPI (Epipolar Plane Image)**: Ekstrak slice horizontal dari light field pada baris tertentu untuk melihat slope yang merepresentasikan depth.
8. **Analisis paralaks**: Ukur pergeseran piksel dari objek dekat vs jauh pada view yang berbeda.

### Analisis
- Bagaimana baseline antar kamera mempengaruhi depth-of-field pada synthetic refocusing?
- Mengapa objek pada depth fokus tampak tajam sementara objek lain blur?
- Informasi depth apa yang dapat diekstrak dari slope pada EPI (Epipolar Plane Image)?
- Apa trade-off antara jumlah view dalam grid kamera dan kualitas refocusing?

---

## Percobaan 20: Visualisasi 3D dan Export

### Tujuan
Mempelajari berbagai teknik visualisasi data 3D dari berbagai sudut pandang, serta format export standar (PLY, OBJ, XYZ, CSV) dan pembuatan turntable animation video.

### Dasar Teori
Visualisasi 3D memerlukan rendering dari berbagai viewpoint untuk pemahaman geometri lengkap. Turntable animation memutar kamera 360° sekeliling objek dan menyimpan setiap frame ke video menggunakan `cv2.VideoWriter()`. Format export umum: PLY (header + vertex/face data, mendukung warna dan normal), OBJ (teks berbasis Wavefront, `v`/`vn`/`f`), XYZ (koordinat saja), dan CSV (tabular). Matplotlib 3D dengan `view_init(elev, azim)` mengontrol sudut pandang rendering.

### Langkah Kerja
1. **Buat point cloud dan mesh**: Generate point cloud 3D berbentuk bola terdeformasi dengan "telinga" (~3400 titik), beri warna berdasarkan height colormap. Buat mesh convex hull dari subset titik menggunakan `scipy.spatial.ConvexHull`.
2. **Render 36 sudut pandang**: Untuk setiap azimuth $= 0°, 10°, 20°, ..., 350°$, render point cloud menggunakan `ax.view_init(elev=25, azim=azim)` dan simpan frame.
3. **Buat turntable video**: Gunakan `cv2.VideoWriter('turntable.avi', fourcc, fps=15, frameSize)` untuk menyusun 36 frame menjadi video animasi rotasi.
4. **Export PLY**: Tulis file PLY dengan header dan data vertex ($x, y, z, r, g, b$) secara manual.
5. **Export OBJ**: Tulis file OBJ dengan format `v x y z`, `vn nx ny nz`, dan `f v1//n1 v2//n2 v3//n3`.
6. **Export XYZ dan CSV**: Export point cloud dalam format XYZ (spasi-separated) dan CSV (comma-separated dengan header kolom).
7. **Multi-angle static render**: Render 6 gambar dari sudut pandang standar (depan, belakang, kiri, kanan, atas, bawah) dan simpan ke output.
8. **Visualisasi mesh wireframe**: Tampilkan mesh sebagai wireframe dan solid secara berdampingan menggunakan `Poly3DCollection`.

### Analisis
- Format mana (PLY/OBJ/XYZ/CSV) yang paling compact, dan mana yang paling interoperable?
- Berapa frame rate minimum pada turntable video agar rotasi terlihat halus?
- Apa kelebihan rendering multi-angle statis dibandingkan video interaktif untuk presentasi?
- Bagaimana kualitas visual rendering matplotlib 3D dibandingkan dengan Open3D viewer?

---

## Kesimpulan
Setelah menyelesaikan 20 percobaan, mahasiswa memahami:
1. Pipeline lengkap dari point cloud mentah hingga mesh 3D bertekstur.
2. Teknik filtering, registration, dan surface reconstruction.
3. Volumetric integration (TSDF) untuk dense reconstruction.
4. Depth-based image warping dan view interpolation.
5. Konsep dan implementasi dasar neural scene representation.
6. Trade-off antara metode tradisional dan neural rendering.
7. Ball Pivoting Algorithm dan Alpha Shapes untuk rekonstruksi surface dengan variasi parameter.
8. Texture mapping dari gambar 2D ke mesh 3D menggunakan camera projection.
9. Colorization point cloud berdasarkan atribut geometris (height, distance, normal, curvature).
10. Segmentasi point cloud menggunakan RANSAC plane fitting dan Euclidean clustering.
11. Marching Cubes untuk ekstraksi isosurface dari volumetric data dan implicit functions.
12. Volumetric rendering (MIP, average, first-surface) melalui ray casting pada volume 3D.
13. Perbedaan forward vs inverse warping, termasuk holes, splatting, dan inpainting.
14. Konsep light field 4D, sub-aperture images, dan synthetic aperture refocusing.
15. Teknik visualisasi 3D multi-angle, turntable video, dan export ke berbagai format standar.

---

## Referensi
1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed. Ch. 13–14.
2. Open3D Documentation. http://www.open3d.org/docs/
3. COLMAP Documentation. https://colmap.github.io/
4. Mildenhall, B. et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields. *ECCV*.
5. Kerbl, B. et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM ToG*.
