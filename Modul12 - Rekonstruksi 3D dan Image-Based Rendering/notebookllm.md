# PROMPT NOTEBOOKLLM MODUL 12: REKONSTRUKSI 3D DAN IMAGE-BASED RENDERING

## Prompt 1 (Slide 1-15): Materi Dasar dan Point Cloud Processing

```
Buatkan presentasi 15 slide untuk mata kuliah Komputer Vision, Modul 12: Rekonstruksi 3D dan Image-Based Rendering, bagian 1 (dasar dan point cloud processing).

Slide 1: Judul - "Modul 12: Rekonstruksi 3D dan Image-Based Rendering (Bagian 1: Dasar & Point Cloud)"
Slide 2: Tujuan Pembelajaran - memahami representasi 3D, point cloud processing, surface reconstruction, dan konsep rendering modern (NeRF, 3DGS)
Slide 3: Representasi Data 3D - point cloud (titik x,y,z), mesh (vertices+faces), volume (voxel grid), implicit function (SDF, NeRF). Tabel perbandingan format: PLY, PCD, OBJ, STL
Slide 4: Point Cloud Basics - definisi P = {(xi,yi,zi)}, sumber data (LiDAR, Kinect, stereo, SfM), format PLY header. Percobaan 01: buat dan visualisasi point cloud sintetis
Slide 5: Point Cloud Filtering - voxel downsampling (centroid per voxel), SOR (hapus outlier d > μ+ασ), radius filter. Percobaan 02: filtering dengan parameter berbeda
Slide 6: Normal Estimation - PCA lokal: covariance C dari k-tetangga, eigenvalue terkecil = normal. Orientasi konsisten. Percobaan 03: estimasi normal sphere
Slide 7: ICP Registration - Iterative Closest Point: (1) closest point, (2) SVD solve R,t, (3) apply, (4) iterate. Point-to-point vs point-to-plane. Percobaan 04
Slide 8: Poisson Surface Reconstruction - solve Δχ = ∇·V, extract isosurface. Input: points+normals, output: watertight mesh. Percobaan 05
Slide 9: Ball Pivoting Algorithm - bola radius ρ menggelinding, menyentuh 3 titik = triangle. Cocok untuk scan data density merata. Percobaan 11
Slide 10: Alpha Shapes - subset Delaunay triangulation, hapus simplex dengan circumradius > 1/α. α→0: empty, α→∞: convex hull. Percobaan 12
Slide 11: Marching Cubes - extract isosurface dari SDF volume. 8 corners per cube, 256 konfigurasi lookup table. Interpolasi vertex pada edge. Percobaan 16
Slide 12: Mesh Processing - decimation (reduce vertices), Laplacian smoothing (average neighbors), topologi Euler V-E+F=2. Percobaan 06
Slide 13: TSDF Integration - Truncated SDF dalam voxel grid, running weighted average dari multiple depth maps. KinectFusion pipeline. Percobaan 07
Slide 14: Perbandingan Metode - tabel: Poisson vs BPA vs Alpha vs MC vs TSDF (input, output, kecepatan, kualitas). Kapan menggunakan metode mana?
Slide 15: Ringkasan Bagian 1 - dari point cloud ke mesh, pipeline lengkap, pentingnya preprocessing (filtering, normals). Preview: image-based rendering dan neural methods
```

## Prompt 2 (Slide 16-30): Image-Based Rendering dan Neural Methods

```
Buatkan presentasi 15 slide untuk Modul 12, bagian 2: Image-Based Rendering, Neural Rendering, dan teknik rendering modern.

Slide 16: Judul Bagian 2 - "Image-Based Rendering & Neural Rendering Methods"
Slide 17: Image Warping dengan Depth - forward warping (per-pixel projection, holes problem), inverse warping (no holes, need target depth). Percobaan 08
Slide 18: View Interpolation - sintesis view intermediate: linear blend vs optical flow-based. Alpha parameter 0-1. Percobaan 09: perbandingan 5 alpha values
Slide 19: Light Field - representasi 4D L(u,v,s,t), EPI (Epipolar Plane Image), aplikasi: refocusing, depth estimation. Percobaan 18
Slide 20: Volumetric Rendering - ray casting: C(r) = Σ Ti·αi·ci, transmittance T, absorption, emission. Percobaan 17
Slide 21: NeRF Pendahuluan - Neural Radiance Fields (Mildenhall 2020), representasi implisit: MLP F(x,d) → (c,σ). Input: multi-view images + poses
Slide 22: NeRF Pipeline Detail - ray sampling r=o+td → positional encoding γ(p) → MLP → (RGB,σ) → volume rendering → pixel color. Loss: MSE vs GT. Percobaan 10
Slide 23: NeRF Training & Inference - stratified sampling, hierarchical volume rendering (coarse+fine network), training ~hours, rendering ~seconds per image
Slide 24: 3D Gaussian Splatting - Kerbl 2023, representasi: kumpulan 3D Gaussian G=(μ,Σ,α,c). Differentiable rasterization. Percobaan 19
Slide 25: 3DGS Pipeline - SfM initialization → Gaussian optimization → adaptive densification/pruning → tile-based rasterizer → real-time rendering ~100 FPS
Slide 26: NeRF vs 3DGS - tabel perbandingan: training time (hours vs minutes), rendering speed (slow vs real-time), quality (comparable), storage, editability
Slide 27: RGBD Pipeline - Kinect/RealSense: RGB+Depth → deprojection → colored point cloud → registration → fusion. Percobaan 14
Slide 28: Mesh Texturing - proyeksi warna dari gambar ke mesh 3D via camera projection. Multi-view texturing dengan z-buffer visibility. Percobaan 13
Slide 29: Point Cloud Segmentation - RANSAC plane fitting, Euclidean clustering, aplikasi: robot navigation, scene understanding. Percobaan 15
Slide 30: Ringkasan Bagian 2 - dari image-based rendering klasik ke neural rendering, evolusi: IBR → NeRF → 3DGS, trade-off kecepatan vs kualitas
```

## Prompt 3 (Slide 31-45): Project, Pipeline, dan Evaluasi

```
Buatkan presentasi 15 slide untuk Modul 12, bagian 3: Pipeline lengkap, project hands-on, dan evaluasi.

Slide 31: Judul Bagian 3 - "Pipeline Rekonstruksi 3D & Project Hands-on"
Slide 32: Pipeline End-to-End - diagram: Input(RGB+Depth) → Point Cloud → Filtering → Normals → Surface Recon → Texturing → Export. Percobaan 20: demo pipeline lengkap
Slide 33: TSDF Pipeline Detail - multi-frame: depth alignment → TSDF update → weight accumulation → marching cubes extraction → mesh texturing. Percobaan 07 expanded
Slide 34: Project 1: Multi-Resolution Downsampling - bandingkan 5 voxel size, analisis trade-off jumlah titik vs representasi. Kode dan visualisasi perbandingan
Slide 35: Project 2: Robust ICP Comparison - point-to-point vs point-to-plane, variasi noise σ, kurva konvergensi. Tips: threshold stopping, max iterations
Slide 36: Project 3: Surface Reconstruction Comparison - Poisson vs BPA vs Alpha vs MC pada dataset sama. Metrik: triangle count, coverage, execution time
Slide 37: Project 4: View Synthesis Pipeline - depth-based warping + inpainting, optical flow interpolation. Metrik: PSNR, SSIM, visual quality
Slide 38: Project 5: RGBD Scene Reconstruction - Kinect-style pipeline: deproject → filter → segment → cluster → reconstruct per-object. Aplikasi: robot, AR
Slide 39: Soal Cerita 1-3 - survei candi (filtering pipeline), robot navigasi (plane+cluster), QC pabrik (ICP alignment). Pendekatan dan solusi implementasi
Slide 40: Soal Cerita 4-6 - virtual tour (NeRF pipeline), dental scan (adaptive smoothing), drone mapping (view interpolation). Contoh output
Slide 41: Soal Cerita 7-10 - gaming asset (mesh simplification), AR depth (disparity→depth→3D), medical imaging (MC isosurface), digital twin (TSDF fusion). Demo pipeline
Slide 42: Tugas Video Modul 12 - struktur: pembukaan, penjelasan materi (representasi 3D, pipeline), demo 20 percobaan, demo project (min 3 improvisasi), penutup
Slide 43: Tips Produksi Video - durasi 15-25 menit, screen record kode+output, narasi bahasa Indonesia, tunjukkan perbandingan metode, highlight insight
Slide 44: Rubrik Penilaian - implementasi kode 30%, output visual 25%, analisis 20%, improvisasi 15%, soal cerita 10%. Format: NIM_Nama_Project12.zip
Slide 45: Penutup & Refleksi - evolusi 3D reconstruction: classical (ICP, Poisson, MC) → deep learning (NeRF, 3DGS) → real-time (Gaussian Splatting). Masa depan: mobile 3D, metaverse, digital twins
```
