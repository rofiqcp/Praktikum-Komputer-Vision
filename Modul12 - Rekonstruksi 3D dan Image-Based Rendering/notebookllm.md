# NotebookLM Prompts — Modul 12: Rekonstruksi 3D dan Image-Based Rendering

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 12: Rekonstruksi 3D dan Image-Based Rendering. Referensi Szeliski (2022) Ch.13–14. Tiap slide ~500 kata, sertakan diagram, formula, dan kode Open3D/Python.

**Slide 1** — Judul "Modul 12: Rekonstruksi 3D dan Image-Based Rendering", subtitle "Dari Point Cloud ke Novel View Synthesis", ilustrasi Poisson mesh + NeRF render, referensi Szeliski Ch.13–14.

**Slide 2** — Pendahuluan: Rekonstruksi 3D → mengubah data 2D (gambar, depth map) ke representasi 3D (mesh, voxel, implicit surface). Image-Based Rendering (IBR) → render novel view dari kumpulan foto, tanpa model geometri sempurna. Pipeline: SfM → point cloud → surface reconstruction → texturing → IBR. Diagram end-to-end.

**Slide 3** — Point Cloud: kumpulan titik 3D (x,y,z) + warna (r,g,b) + normal (nx,ny,nz). Format: PLY, PCD, XYZ, LAS. Open3D: o3d.io.read/write_point_cloud(). Visualisasi: o3d.visualization.draw_geometries([pcd]). Statistical Outlier Removal: hapus titik d_i>μ_d+α·σ_d. Voxel Downsampling: satu titik per voxel (centroid).

**Slide 4** — Normal Estimation: PCA lokal dari k-nearest neighbors. C=(1/k)Σ(p_i−p̄)(p_i−p̄)ᵀ. Eigenvector eigenvalue terkecil = surface normal. pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)). Orientasi normal: orient_normals_towards_camera_location().

**Slide 5** — ICP Registration (Iterative Closest Point): (1) Find correspondences (nearest neighbor), (2) Estimate R,t yang minimize E=Σ‖Rp_i+t−q_i‖², (3) Apply dan iterasi. Point-to-Plane ICP: E=Σ[(Rp_i+t−q_i)·n_i]². o3d.pipelines.registration.registration_icp(). Fitness, RMSE convergence.

**Slide 6** — Surface Reconstruction: Poisson (implicit → Laplacian Poisson: Δχ=∇·V, isosurface χ=threshold, octree depth 8–10). BPA (Ball Pivoting Algorithm: bola radius r menggelinding, sentuh 3 titik → segitiga). Alpha Shapes (generalisasi convex hull, parameter α kontrol detail). Marching Cubes (volumetric → mesh dari isosurface).

**Slide 7** — Mesh Post-processing: Decimation (simplify_quadric_decimation, target triangle count). Smoothing (filter_smooth_laplacian, Taubin smoothing). Hole filling. Texturing (proyeksikan warna gambar ke mesh). Evaluasi mesh: watertight?, self-intersections?, vertex normals consistency?

**Slide 8** — TSDF (Truncated Signed Distance Function): representasi volumetrik, TSDF(x)=Σ_i w_i·d_i(x)/Σ_i w_i. d_i = signed distance ke surface per depth frame. KinectFusion pipeline: (1) capture depth, (2) pose tracking (ICP vs model), (3) integrate ke TSDF volume, (4) raycast untuk render. o3d.pipelines.integration.ScalableTSDFVolume.

**Slide 9** — Image Warping & Novel View: Forward warp (src→dst, holes/splatting). Inverse warp (dst←src, bilinear). Disocclusion: area yang terblok di source view → visible di target → perlu inpainting. Pose interpolation: lerp translasi, SLERP rotasi quaternion → kamera intermediate.

**Slide 10** — Neural Rendering & NeRF: MLP f_θ(x,y,z,θ,ϕ) → (RGB, density σ). Volume rendering: C=Σ_i T_i·α_i·c_i, T_i=exp(−Σ_{j<i}σ_j·δ_j). Train: minimize L=‖C_render−C_gt‖². 3D Gaussian Splatting: representasi eksplisit lebih cepat. Konsep dan aplikasi novel view synthesis.

**Slide 11** — Percobaan 1–4: Point Cloud Basics (load PLY, visualisasi Open3D, properties — jumlah titik, bounding box, centroid). Filtering & Downsampling (statistical outlier removal, voxel grid downsampling, perbandingan przed/po). Normal Estimation (visualisasi normal arrow, PCA, oriented consistently). ICP Registration (align 2 partial scan, P2P vs P2Plane, fitness+RMSE).

**Slide 12** — Percobaan 5–8: Surface Reconstruction (Poisson depth=8/9/10 vs BPA multi-radius — mesh visual comparison, triangle count). Mesh Processing (simplification 50%/25%/10%, Laplacian smoothing iterations, Taubin, hole count). TSDF Integration (inkremental build dari depth sequence, raycast rendering per frame). Image Warping (forward warp + holes visualization, inverse warp tanpa holes).

**Slide 13** — Percobaan 9–12: View Interpolation (smooth transition video antar 2 viewpoints, pose lerp+SLERP, blending). Neural Rendering (MLP regression gambar sederhana, konsep NeRF architecture demo). Ball Pivoting Detail (multi-radius analysis, coverage vs triangle quality). Alpha Shapes (parameter α sweep, concave shape modeling).

**Slide 14** — Percobaan 13–16: Mesh Texturing (project color dari gambar ke mesh — UV mapping, trisurf texture). Point Cloud Colorization (6 metode: depth, normal, height, curvature, cluster, RGB photo). Point Cloud Segmentation (RANSAC plane fitting, DBSCAN clustering, semantic coloring). Marching Cubes (isosurface dari sphere/torus implicit function, threshold sweep).

**Slide 15** — Percobaan 17–20: Volumetric Rendering (MIP, average, first-surface projection dari voxel grid). Forward vs Inverse Warp comparison (holes pattern, splatting artifact, bilinear quality). Light Field Basics (4D representation, sub-aperture views, synthetic refocus — aperture integration). 3D Visualization & Export (PLY/OBJ/GLTF, turntable rotation video, resolution comparison). Setup: open3d, numpy, matplotlib, scipy, trimesh.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 12, Slide 16–30. Slide 16–25: analisis, rekap, koneksi, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Rekap percobaan 1–10: point cloud basics, filtering+downsampling, normal estimation, ICP registration, surface reconstruction (Poisson+BPA), mesh processing, TSDF integration, image warping, view interpolation, neural rendering. Grid thumbnail. Tabel metode, library, kualitas output.

**Slide 17** — Rekap percobaan 11–20: BPA detail, alpha shapes, mesh texturing, point cloud colorization (6 metode), point cloud segmentation, Marching Cubes, volumetric rendering, forward vs inverse warp, light field basics, 3D export + turntable video. Tabel metrik: triangle count, fitness, RMSE, export format.

**Slide 18** — Analisis mendalam: Mengapa Poisson lebih smooth dari BPA namun BPA lebih faithful terhadap noise? ICP convergence — kapan Point-to-Plane lebih cepat? Hubungan TSDF threshold dengan mesh detail. Forward warp holes — mengapa splatting tidak sempurna? NeRF mengapa butuh ratusan gambar training?

**Slide 19** — Koneksi antar modul: Point cloud dari SfM/stereo (Modul 11) → surface reconstruction (Modul 12). ICP registration analog dengan bundle adjustment (Modul 8). Image warping menggunakan homography/depth (Modul 2+11). TSDF integrate depth map dari stereo (Modul 11). NeRF = neural extension dari IBR pipeline.

**Slide 20** — Best practices: Normal estimation radius harus sesuai point cloud density (tidak terlalu kecil/besar). Poisson depth terlalu dalam → detail noise, terlalu dangkal → loss fine detail. Remove low-density vertices setelah Poisson (densities quantile 0.01). ICP butuh good initial alignment — gunakan global registration (FPFH+RANSAC) terlebih dahulu.

**Slide 21** — Aplikasi nyata: Cultural heritage digitization (museum artefak → PLY/OBJ). Construction monitoring (point cloud sebelum/sesudah proyek → diff). Dental/medical scanning (TSDF integrate structured light). Autonomous vehicle HD mapping (LiDAR point cloud → driveable surface). Video game asset creation dari scan nyata.

**Slide 22** — Perbandingan surface reconstruction: Poisson vs BPA vs Alpha Shapes vs Marching Cubes — tabel watertight, concave support, noise sensitivity, computation time, output quality. Point cloud registration: Point-to-Point vs Point-to-Plane ICP — convergence speed, accuracy, computation per iteration.

**Slide 23** — Checklist kompetensi: point cloud load/visualize/filter, normal estimation, ICP registration, Poisson+BPA reconstruction, mesh processing (simplify+smooth), TSDF pipeline, image warping (forward+inverse), view interpolation, NeRF concept, 3D export. Self-assessment tabel per percobaan.

**Slide 24** — Kuis: (1) Open3D fungsi untuk voxel downsampling? (2) Perbedaan Point-to-Point vs Point-to-Plane ICP? (3) Parameter Poisson depth: lebih besar → mesh bagaimana? (4) Disocclusion dalam view interpolation artinya? (5) NeRF memprediksi apa per titik 3D?

**Slide 25** — Diskusi: TSDF vs Poisson untuk rekonstruksi real-time (KinectFusion). Kapan BPA lebih tepat dari Poisson (tipis, open surface)? Tradeoff resolusi voxel grid TSDF: detail vs memori. NeRF vs 3D Gaussian Splatting untuk novel view synthesis: kualitas, kecepatan rendering, kebutuhan data.

**Slide 26** — Project "3D Reconstruction & Rendering System". Improvisasi tersedia per percobaan (13 total): 3D Object Scanner (video turntable + COLMAP + Open3D interaktif), Point Cloud Comparator (Chamfer+Hausdorff distance heatmap), Adaptive LOD System (octree spatial partitioning), Surface Curvature Analyzer (Gaussian+mean curvature colormap), Multi-Scan Assembler (4–6 partial scans + pose graph), Reconstruction Quality Benchmark (3 metode × 5 dataset), 3D Model Optimizer (glTF/GLB export, LOD 3 levels), Mini KinectFusion (TUM/ICL-NUIM dataset), Parallax Photo Effect (MiDaS depth + 2.5D video), Virtual Camera Dolly (slider UI view interpolation), NeRF Turntable (30–50 foto + PSNR/SSIM), Scene Relighting NeRF, End-to-End Photogrammetry Pipeline (COLMAP → OBJ otomatis). Deliverable: .py, output/, laporan.

**Slide 27** — Soal proyek pilihan: tambahan 2 soal untuk diversifikasi. Proyek A (Dokumentasi Budaya): scan benda antik 15+ foto → COLMAP SfM → Dense MVS → Poisson mesh → texture → export OBJ → viewer web sederhana. Proyek B (Virtual Studio): depth dari stereo simulasikan → TSDF integrate → raycast render → bokeh depth-based → style transfer → composite dengan virtual background.

**Slide 28** — Rubrik Project: Fungsionalitas 35%, Integrasi Percobaan 20%, Kualitas Kode 15%, Dokumentasi 15%, Kreativitas 15%. Bonus +5 scan fisik objek nyata (Open3D turntable), +3 NeRF/3DGS rendering turntable video. Format ZIP NIM_Nama_Project12.zip. Deadline 1 minggu.

**Slide 29** — Tugas Video: Tunjukkan visualisasi 3D dari berbagai angle (rotate di Open3D / matplotlib). Bandingkan metode secara visual side-by-side. Demo 20 percobaan LIVE (40–60 mnt). Materi (10–15 mnt): wajib diagram pipeline SfM→dense→mesh→render. Demo Project (10–15 mnt) — tunjukkan 3D output dan rendered views. Analisis & Penutup (5 mnt).

**Slide 30** — Rubrik Video: Pembukaan (5), Materi (10), 20 Percobaan (40 — 2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 scan objek nyata + 3D visualisasi rotasi, +3 novel view synthesis (NeRF/3DGS). Penalti −2/percobaan tidak tampil. "Rekonstruksi Dunia Nyata, Render Pandangan Baru — Selesaikan Pipeline 3D Vision!"
