# PROJECT MODUL 12: REKONSTRUKSI 3D DAN IMAGE-BASED RENDERING

## Ketentuan Umum
- Format pengumpulan: **NIM_Nama_Project12.zip**
- Berisi: source code (.py), output images, dan laporan singkat (PDF)
- Gunakan dataset dari folder `image/` atau dataset sendiri
- Setiap project harus menghasilkan output visual di folder `output/`

---

## Improvisasi Project (Pilih minimal 3)

### 1. Multi-Resolution Voxel Downsampling
Bandingkan 5 voxel size (0.01, 0.05, 0.1, 0.5, 1.0) pada point cloud. Visualisasi dan analisis trade-off antara jumlah titik vs kecepatan vs kualitas representasi.

### 2. Robust ICP dengan Variasi Noise
Implementasikan ICP point-to-point dan point-to-plane. Bandingkan konvergensi pada berbagai level noise (σ = 0.01, 0.05, 0.1, 0.5). Visualisasi kurva error per iterasi untuk setiap varian.

### 3. Surface Reconstruction Comparison
Bandingkan 4 metode rekonstruksi (Poisson, BPA, Alpha Shapes, Marching Cubes) pada point cloud yang sama. Analisis kualitas mesh, jumlah triangle, dan waktu eksekusi.

### 4. TSDF Multi-Frame Fusion
Integrasi 5+ depth map dari viewpoint berbeda ke TSDF volume. Visualisasi volume sebelum/sesudah fusion dan extract mesh. Analisis pengaruh truncation distance.

### 5. Depth-Based View Synthesis
Implementasikan forward + inverse warping untuk mensintesis 5 novel view dari satu gambar + depth map. Analisis artefak (holes, ghosting) dan implementasikan inpainting sederhana.

### 6. Optical Flow View Interpolation
Buat video interpolasi halus (20 frame) antara dua gambar multiview. Bandingkan linear blend vs flow-based. Hitung PSNR/SSIM jika ground truth tersedia.

### 7. Light Field Processing
Buat light field dari 5×5 grid gambar sintetis. Implementasikan refocusing dengan shift-and-add. Visualisasi EPI dan analisis slope untuk estimasi depth.

### 8. NeRF Volume Rendering Variants
Implementasikan volume rendering dengan berbagai stratified sampling (uniform, stratified, hierarchical). Visualisasi efek jumlah sample terhadap kualitas rendered image.

### 9. 3D Gaussian Splatting Analysis
Buat visualisasi 3DGS konseptual: variasikan jumlah Gaussian (10, 50, 200, 1000), ukuran, dan opacity. Bandingkan kualitas splatted image. Visualisasi proses densification/pruning.

### 10. Point Cloud Classification
Implementasikan klasifikasi sederhana point cloud berdasarkan fitur geometrik (normal, curvature, planarity). Segmentasi menjadi floor, wall, dan object.

### 11. Mesh Simplification Pipeline
Implementasikan pipeline: load mesh → decimation (25%, 50%, 75%) → smoothing → normal recomputation. Analisis perubahan visual dan metrik (Hausdorff distance estimasi).

### 12. RGBD Scene Reconstruction
Gunakan RGBD data untuk rekonstruksi scene: deproject → colored point cloud → voxel filter → plane segmentation → per-object clustering.

### 13. Texture Mapping Multi-View
Implementasikan texturing mesh dari 3+ gambar berbeda viewpoint. Gunakan z-buffer untuk menentukan view terbaik per face. Visualisasi mesh textured dari sudut pandang baru.

### 14. Volumetric Rendering Artistic
Buat volume 3D dengan multiple objek (sphere, torus, cube) dan render dengan berbagai transfer function (warna berdasarkan density). Implementasikan absorption + emission model.

### 15. Rekonstruksi 3D End-to-End
Pipeline lengkap: RGBD → Point Cloud → Filtering → Normal Estimation → Surface Reconstruction → Texturing → Export PLY. Visualisasi setiap tahap dan hitung metrik kualitas.

---

## Soal Cerita (Kerjakan semua)

### Soal 1: Survei Bangunan Bersejarah
Sebuah museum ingin merekonstruksi 3D model candi bersejarah dari data LiDAR. Point cloud yang diperoleh memiliki 5 juta titik dengan banyak noise dan outlier dari vegetasi. Jelaskan pipeline filtering yang tepat (voxel downsample → SOR → radius filter) dan tentukan parameter optimal untuk setiap tahap. Implementasikan demonstrasi dengan point cloud sintetis yang merepresentasikan permukaan candi.

### Soal 2: Robot Navigasi Indoor
Robot menggunakan Kinect untuk navigasi. Diperlukan deteksi lantai (plane) dan segmentasi objek penghalang. Implementasikan pipeline RGBD → Point Cloud → RANSAC Plane → Euclidean Clustering. Robot harus tahu jarak ke setiap cluster. Demonstrasikan dengan data RGBD sintetis.

### Soal 3: Pabrik Quality Control
Pabrik automobile menggunakan structured light scanner untuk inspeksi bodi mobil. Dua scan dari posisi berbeda harus digabungkan menggunakan ICP. Implementasikan ICP dengan threshold konvergensi yang tepat dan visualisasi error alignment.

### Soal 4: Arsitek Virtual Tour
Arsitek ingin membuat virtual tour rumah dari 20 foto smartphone. Jelaskan pipeline SfM → NeRF/3DGS untuk menghasilkan novel views. Implementasikan demo volume rendering sederhana dan visualisasi pipeline diagram.

### Soal 5: Dental 3D Scanning
Dental scanner menghasilkan mesh gigi pasien. Mesh perlu di-smooth untuk menghilangkan artefak scan tanpa kehilangan detail penting (cusp, groove). Implementasikan Laplacian smoothing dengan parameter adaptif dan bandingkan 3 level smoothing.

### Soal 6: Drone Mapping
Drone mengambil foto aerial dari 5 posisi untuk pemetaan lahan. Implementasikan view interpolation antara foto-foto tersebut untuk menghasilkan video flythrough. Gunakan optical flow-based interpolation dan analisis kualitas.

### Soal 7: Gaming 3D Asset Creation
Game developer ingin mengkonversi scan 3D menjadi low-poly mesh. Implementasikan pipeline: high-poly point cloud → surface reconstruction → mesh simplification (10% vertices). Bandingkan visual quality sebelum dan sesudah simplification.

### Soal 8: Augmented Reality Depth
Aplikasi AR memerlukan depth map untuk menempatkan objek virtual. Dari stereo camera diperoleh disparity map. Konversikan ke depth map → colored point cloud → visualisasi adegan 3D dengan objek virtual tambahan.

### Soal 9: Medical Imaging
CT scan menghasilkan volume 3D dari organ tubuh. Implementasikan Marching Cubes untuk mengekstrak permukaan organ dari volume sintetis (SDF ellipsoid). Variasikan iso-value dan analisis pengaruhnya terhadap mesh yang dihasilkan.

### Soal 10: Smart City Digital Twin
Kota ingin membuat digital twin dari kawasan komersial. Data: RGBD dari 10 frame berurutan. Implementasikan TSDF fusion dari multiple depth frames, extract mesh, dan texturing. Analisis kualitas fusion vs jumlah frame.

---

## Rubrik Penilaian

| Komponen | Bobot | Kriteria |
|---|---|---|
| Implementasi Kode | 30% | Kode berjalan, style def/fungsi, komentar bahasa Indonesia |
| Output Visual | 25% | Gambar tersimpan di output/, visualisasi informatif |
| Analisis & Laporan | 20% | Penjelasan hasil, perbandingan metode, parameter tuning |
| Improvisasi (3+) | 15% | Kreativitas, kedalaman eksplorasi, variasi metode |
| Soal Cerita | 10% | Kelengkapan jawaban, relevansi implementasi |
