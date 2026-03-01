# PROJECT MODUL 12: REKONSTRUKSI 3D DAN IMAGE-BASED RENDERING

---

## Deskripsi
Project ini mengembangkan seluruh percobaan menjadi aplikasi rekonstruksi 3D dan rendering novel view yang lengkap dan aplikatif.

---

## Improvisasi Percobaan

### Dari Percobaan 1 (Point Cloud Basics)
**Improvisasi 1: 3D Object Scanner Sederhana**
Buat aplikasi yang menerima video seputar objek (rotate object on turntable), jalankan SfM + MVS (COLMAP), lalu load dan visualisasi hasilnya sebagai point cloud interaktif menggunakan Open3D. Tambahkan fitur: crop, color edit, measurement tool (jarak antar 2 titik klik).

**Improvisasi 2: Point Cloud Comparator**
Buat tool untuk membandingkan dua point cloud: hitung Chamfer distance dan Hausdorff distance. Visualisasikan perbedaan sebagai heatmap warna pada point cloud. Aplikasikan untuk membandingkan hasil scan berbeda dari objek yang sama.

### Dari Percobaan 2 (Filtering & Downsampling)
**Improvisasi 3: Adaptive Point Cloud LOD System**
Implementasikan sistem Level-of-Detail (LOD) untuk point cloud besar. Berdasarkan jarak kamera, tampilkan resolusi berbeda (full di dekat, downsampled di jauh). Gunakan octree untuk spatial partitioning dan voxel downsampling pada level yang tepat.

### Dari Percobaan 3 (Normal Estimation)
**Improvisasi 4: Surface Curvature Analyzer**
Buat tool analisis curvature: hitung Gaussian curvature dan mean curvature dari point cloud. Visualisasikan dengan colormap. Gunakan untuk deteksi fitur geometris (edge, corner, planar region). Terapkan pada objek berbeda (sphere, cube, complex object).

### Dari Percobaan 4 (ICP Registration)
**Improvisasi 5: Multi-Scan Assembler**
Buat pipeline yang mengambil 4-6 partial scans dari objek dan melakukan registrasi otomatis (global registration + ICP refinement) secara berurutan. Implementasikan pose graph optimization untuk mengurangi drift. Output: unified point cloud + merged mesh.

### Dari Percobaan 5 (Surface Reconstruction)
**Improvisasi 6: Reconstruction Quality Benchmark**
Buat benchmark yang membandingkan 3 metode (Poisson, BPA, Alpha Shapes) pada 5 dataset berbeda. Ukur: waktu, triangle count, mesh quality metrics (watertight, self-intersection), visual quality (render dari sudut tetap). Generate laporan otomatis berupa tabel dan gambar perbandingan.

### Dari Percobaan 6 (Mesh Processing)
**Improvisasi 7: 3D Model Optimizer untuk Web/Mobile**
Buat pipeline yang menerima high-poly mesh dan menghasilkan optimized version untuk web/mobile deployment: simplification dengan target triangle budget, texture atlas generation, LOD generation (3 levels), export ke glTF/GLB format. Bandingkan visual quality tiap level.

### Dari Percobaan 7 (TSDF Integration)
**Improvisasi 8: Mini KinectFusion**
Implementasikan simplified KinectFusion pipeline: frame-to-model tracking (ICP), TSDF integration, raycasting untuk rendering. Gunakan RGB-D dataset dari TUM atau ICL-NUIM. Visualisasikan rekonstruksi progresif (frame demi frame).

### Dari Percobaan 8 (Image Warping)
**Improvisasi 9: Parallax Photo Effect Generator**
Buat aplikasi yang mengubah foto 2D + depth map (dari MiDaS Modul 11) menjadi efek parallax 3D (seperti Facebook 3D Photo). Generate video looping di mana kamera bergerak sedikit (translate dan rotate). Handle disocclusion dengan inpainting.

### Dari Percobaan 9 (View Interpolation)
**Improvisasi 10: Virtual Camera Dolly System**
Implementasikan virtual camera yang bergerak mulus antara dua foto scene yang sama. Gunakan depth-based warping + blending. Buat UI sederhana untuk mengontrol posisi kamera virtual (slider). Output: smooth video transition antara viewpoints.

### Dari Percobaan 10 (Neural Rendering)
**Improvisasi 11: NeRF Object Turntable Renderer**
Train NeRF atau 3D Gaussian Splatting pada objek kecil (dari 30-50 foto). Render video turntable 360° di sekitar objek. Bandingkan kualitas dengan mesh tradisional yang di-render dengan texture. Hitung PSNR dan SSIM.

**Improvisasi 12: Scene Relighting dengan NeRF**
Menggunakan trained NeRF model, eksplorasi modifikasi rendering: ubah background, adjust density threshold, render dari viewpoint yang tidak ada di training set. Dokumentasikan limitasi dan artefak pada extrapolated views.

### Proyek Gabungan
**Improvisasi 13: End-to-End Photogrammetry Pipeline**
Buat pipeline lengkap: ambil 30-50 foto objek → COLMAP SfM → dense reconstruction → Poisson mesh → texture mapping → export OBJ/PLY. Semua langkah otomatis dari command line. Implementasikan quality checks di setiap tahap.

**Improvisasi 14: AR Object Placement**
Gunakan mesh 3D yang direkonstruksi, lalu "tempatkan" ke scene baru menggunakan homography atau pose estimation. Render objek virtual di atas real image dengan lighting estimation sederhana (ambient + directional). Output: composited image.

**Improvisasi 15: Cultural Heritage Documentation**
Rekonstruksi 3D objek heritage (patung, relief, bangunan bagian). Pipeline: capture → SfM → dense → mesh → annotate (add labels pada bagian penting). Generate web-based viewer menggunakan three.js atau potree. Buat video presentasi.

---

## Soal Cerita

### Soal 1: Arkeologi Digital
Sebuah tim arkeologi menemukan pecahan keramik kuno. Mereka memfoto setiap pecahan dari 20 sudut berbeda.

**Tugas**:
- Rekonstruksi 3D setiap pecahan (SfM + Poisson/MVS).
- Register pecahan-pecahan menggunakan ICP (simulasikan dengan partial point clouds).
- Visualisasi gabungan.
- **Buat laporan**: metode, parameter, screenshot dari 4 sudut pandang, waktu processing per tahap.

**Rubrik**:
| Komponen | Bobot |
|----------|-------|
| Rekonstruksi tiap pecahan | 25% |
| Registrasi antar pecahan | 25% |
| Kualitas mesh akhir | 20% |
| Visualisasi dan dokumentasi | 15% |
| Analisis dan kesimpulan | 15% |

---

### Soal 2: Quality Control Manufaktur
Pabrik otomotif ingin membandingkan komponen produksi dengan model CAD referensi.

**Tugas**:
- Rekonstruksi 3D komponen dari foto.
- Register hasil scan ke model referensi (gunakan ICP).
- Hitung deviasi (Hausdorff / Chamfer distance).
- Visualisasikan deviasi sebagai heatmap pada mesh.
- Tentukan PASS/FAIL berdasarkan threshold deviasi.

**Rubrik**:
| Komponen | Bobot |
|----------|-------|
| Rekonstruksi 3D | 20% |
| Registrasi ke referensi | 20% |
| Perhitungan deviasi | 20% |
| Heatmap visualisasi | 20% |
| Keputusan QC + laporan | 20% |

---

### Soal 3: Virtual Tour Generator
Agen properti ingin membuat virtual tour rumah dari foto-foto interior.

**Tugas**:
- Rekonstruksi 3D beberapa ruangan (TSDF integration dari RGB-D, atau SfM+mesh dari foto).
- Texture mapping dari foto asli ke mesh.
- Render novel views (virtual walkthrough path).
- Generate video fly-through.
- Overlay info (nama ruangan, ukuran) pada output.

**Rubrik**:
| Komponen | Bobot |
|----------|-------|
| Rekonstruksi per ruangan | 25% |
| Texture quality | 20% |
| Novel view rendering | 25% |
| Video fly-through | 15% |
| Informasi overlay + dokumentasi | 15% |

---

### Soal 4: Scan-to-BIM untuk Renovasi
Tim arsitek ingin membuat model 3D ruangan kantor yang akan direnovasi.

**Tugas**:
- Scan ruangan (3-4 posisi, simulasi dengan dataset RGB-D).
- TSDF integration → mesh.
- Identifikasi bidang datar besar (dinding, lantai, ceiling) menggunakan RANSAC plane fitting.
- Hitung dimensi ruangan (panjang × lebar × tinggi).
- Visualisasi dengan planes yang diberi warna berbeda.

**Rubrik**:
| Komponen | Bobot |
|----------|-------|
| TSDF reconstruction | 25% |
| Plane detection | 25% |
| Pengukuran dimensi | 20% |
| Visualisasi | 15% |
| Akurasi dan analisis | 15% |

---

### Soal 5: E-Commerce 3D Product Viewer
Toko online ingin menampilkan produk dalam 3D yang bisa diputar pelanggan.

**Tugas**:
- Foto produk kecil (sepatu/tas/mainan) dari 30+ sudut di turntable.
- Rekonstruksi 3D (SfM → dense → mesh → texture).
- Optimasi mesh untuk web (simplify ke <50k triangles).
- Render 36 views (setiap 10°) untuk web viewer.
- Hitung PSNR antara rendered view dan foto asli terdekat.

**Rubrik**:
| Komponen | Bobot |
|----------|-------|
| Kualitas rekonstruksi 3D | 25% |
| Optimasi mesh | 20% |
| Multi-view rendering | 25% |
| PSNR evaluation | 15% |
| Dokumentasi pipeline | 15% |

---

### Soal 6: Robot Navigation Map Building
Robot bergerak di koridor dan membangun peta 3D real-time dari depth sensor.

**Tugas**:
- Simulasikan dengan RGB-D dataset sequence.
- Implementasikan frame-to-frame ICP tracking.
- Build global TSDF volume secara inkremental.
- Setiap 10 frame, extract mesh dan visualisasikan progress.
- Plot trajectory kamera (path robot).

**Rubrik**:
| Komponen | Bobot |
|----------|-------|
| ICP tracking | 25% |
| TSDF integration inkremental | 25% |
| Trajectory plot | 15% |
| Visualisasi progress | 20% |
| Analisis drift + dokumentasi | 15% |

---

### Soal 7: 3D Face Reconstruction
Sistem keamanan ingin merekonstruksi 3D wajah dari beberapa foto wajah.

**Tugas**:
- Ambil 10-15 foto wajah dari sudut berbeda (frontal, 45°, profile).
- Jalankan SfM → dense reconstruction.
- Surface reconstruction (Poisson).
- Smooth mesh dan improve quality.
- Render wajah dari sudut baru yang tidak ada di input.
- Bandingkan face recognition accuracy pada rendered vs real photo.

**Rubrik**:
| Komponen | Bobot |
|----------|-------|
| SfM + Dense reconstruction | 25% |
| Mesh quality (Poisson + smoothing) | 25% |
| Novel view rendering | 20% |
| Face recognition comparison | 15% |
| Dokumentasi + analisis | 15% |

---

### Soal 8: Disaster Damage Assessment dari Drone
Pasca gempa, drone mengambil foto bangunan rusak dari udara.

**Tugas**:
- Rekonstruksi 3D bangunan dari foto aerial (gunakan dataset structure-from-motion, atau simulasi).
- Identifikasi area kerusakan: compare dengan model "sebelum" (generate atau download).
- Hitung volume perubahan (using difference of point clouds).
- Visualisasi perubahan dengan heatmap.
- Generate laporan: area damage, severity estimation.

**Rubrik**:
| Komponen | Bobot |
|----------|-------|
| Rekonstruksi 3D from aerial | 25% |
| Change detection | 25% |
| Volume estimation | 20% |
| Heatmap + laporan | 15% |
| Analisis dan kesimpulan | 15% |

---

### Soal 9: Parallax Video Creator untuk Media Sosial
Content creator ingin mengubah foto landscape menjadi video 3D parallax yang menarik.

**Tugas**:
- Ambil foto landscape berkualitas tinggi.
- Estimate depth menggunakan MiDaS (dari Modul 11).
- Manual refine depth di area yang salah (edge artifacts).
- Forward warp dengan camera motion path (slow horizontal pan + subtle zoom).
- Inpaint disoccluded areas.
- Output: video MP4 3-5 detik, loopable.

**Rubrik**:
| Komponen | Bobot |
|----------|-------|
| Depth estimation quality | 20% |
| Depth refinement | 15% |
| Warping implementation | 25% |
| Inpainting quality | 20% |
| Video output quality | 20% |

---

### Soal 10: Museum Virtual Exhibition
Museum ingin membuat pameran virtual di mana pengunjung bisa melihat artefak 3D secara interaktif.

**Tugas**:
- Rekonstruksi 3D minimal 3 objek berbeda (masing-masing dari 20+ foto).
- Optimasi setiap mesh (simplify, texture bake).
- Buat "virtual room" (simple 3D environment) dan tempatkan objek.
- Implementasi NeRF atau 3DGS pada salah satu objek sebagai perbandingan.
- Render video walkthrough museum virtual.
- Generate quality report: PSNR, mesh stats, rendering speed.

**Rubrik**:
| Komponen | Bobot |
|----------|-------|
| Rekonstruksi 3 objek | 25% |
| Mesh optimization | 15% |
| Neural rendering (1 objek) | 20% |
| Virtual room + placement | 15% |
| Video walkthrough + report | 15% |
| Kreativitas dan presentation | 10% |

---

## Rubrik Penilaian Project Keseluruhan

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Fungsionalitas | 35% | Program berjalan, output benar |
| Integrasi Multi-Teknik | 20% | Menggabungkan registration, reconstruction, rendering |
| Kualitas Kode | 15% | Modular, terdokumentasi, efisien |
| Dokumentasi & Laporan | 15% | README, screenshot 3D, analisis quantitative |
| Kreativitas & Inovasi | 15% | Solusi unik, visualization, extra features |

---

## Ketentuan Pengumpulan
- **Deadline**: 2 minggu setelah modul selesai.
- **Format**: ZIP berisi folder project.
- **Struktur folder**:
  ```
  Modul12_Project_[NIM]/
  ├── README.md
  ├── src/
  │   ├── reconstruction.py
  │   ├── registration.py
  │   ├── rendering.py
  │   └── utils.py
  ├── data/
  │   └── (sample images, point clouds)
  ├── output/
  │   ├── meshes/
  │   ├── renders/
  │   └── videos/
  └── docs/
      ├── screenshots/
      └── report.md
  ```
- **Naming**: `Modul12_Project_[NIM]_[Nama].zip`
