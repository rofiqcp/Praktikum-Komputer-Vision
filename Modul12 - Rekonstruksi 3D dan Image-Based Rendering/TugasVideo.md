# TUGAS VIDEO MODUL 12: REKONSTRUKSI 3D DAN IMAGE-BASED RENDERING

## Ketentuan Video
- **Format:** MP4 (H.264)
- **Nama file:** `Video_Modul12_NIM_Nama.mp4`
- **Durasi:** 15-25 menit
- **Resolusi:** Minimal 720p
- **Audio:** Narasi jelas dalam Bahasa Indonesia
- **Screen recording:** Tampilkan kode + output

---

## Struktur Video

### 1. Pembukaan (1-2 menit)
- Perkenalan: Nama, NIM, Mata Kuliah
- Judul: "Modul 12 - Rekonstruksi 3D dan Image-Based Rendering"
- Overview singkat topik yang akan dibahas

### 2. Penjelasan Materi (5-7 menit)
Jelaskan konsep utama:
- Representasi data 3D (point cloud, mesh, volume, implicit)
- Pipeline rekonstruksi 3D: akuisisi → preprocessing → reconstruction → rendering
- Point cloud processing: filtering, normal estimation, registration (ICP)
- Surface reconstruction: Poisson, BPA, Alpha Shapes, Marching Cubes
- TSDF integration dan KinectFusion
- Image-based rendering: warping, view interpolation
- NeRF dan 3D Gaussian Splatting (konsep dan perbandingan)

### 3. Demo Percobaan (5-10 menit)
Jalankan dan jelaskan setiap percobaan:

| No | File | Topik |
|---|---|---|
| 01 | `01_point_cloud_basics.py` | Membuat, load, dan visualisasi point cloud |
| 02 | `02_point_cloud_filtering.py` | Voxel downsample dan SOR |
| 03 | `03_normal_estimation.py` | Estimasi normal via PCA lokal |
| 04 | `04_icp_registration.py` | ICP point-to-point alignment |
| 05 | `05_surface_reconstruction_poisson.py` | Pipeline Poisson reconstruction |
| 06 | `06_mesh_processing.py` | Decimation dan Laplacian smoothing |
| 07 | `07_tsdf_integration.py` | TSDF volume integration |
| 08 | `08_image_warping_depth.py` | Forward warping dengan depth |
| 09 | `09_view_interpolation.py` | Linear blend vs flow-based |
| 10 | `10_neural_rendering_konsep.py` | Pipeline NeRF dan volume rendering |
| 11 | `11_ball_pivoting_algorithm.py` | Konsep BPA 2D |
| 12 | `12_alpha_shapes.py` | Alpha shapes variasi parameter |
| 13 | `13_mesh_texturing.py` | Proyeksi tekstur ke mesh |
| 14 | `14_rgbd_point_cloud.py` | RGBD ke colored point cloud |
| 15 | `15_point_cloud_segmentation.py` | Plane segmentation + clustering |
| 16 | `16_marching_cubes.py` | Isosurface extraction |
| 17 | `17_volumetric_rendering.py` | Ray casting volume 3D |
| 18 | `18_light_field_basics.py` | Light field grid dan EPI |
| 19 | `19_3d_gaussian_splatting_konsep.py` | Konsep 3DGS pipeline |
| 20 | `20_3d_reconstruction_pipeline.py` | Pipeline end-to-end |

### 4. Demo Project (3-5 menit)
- Tunjukkan minimal 3 improvisasi project yang dikerjakan
- Jelaskan pendekatan dan hasil yang diperoleh
- Bandingkan metode dan analisis kelebihan/kekurangan

### 5. Penutup (1-2 menit)
- Rangkuman pembelajaran
- Perbandingan metode rekonstruksi (klasik vs modern)
- Tantangan dan potensi pengembangan (real-time, mobile, AR/VR)

---

## Rubrik Penilaian Video

| Komponen | Bobot | Kriteria |
|---|---|---|
| Pembukaan & Struktur | 10% | Perkenalan jelas, struktur teratur |
| Penjelasan Materi | 25% | Konsep benar, analogi mudah dipahami |
| Demo Percobaan | 30% | Semua 20 percobaan dijalankan, output ditampilkan |
| Demo Project | 20% | Minimal 3 improvisasi, analisis mendalam |
| Kualitas Presentasi | 15% | Audio jelas, visual bagus, durasi tepat |
