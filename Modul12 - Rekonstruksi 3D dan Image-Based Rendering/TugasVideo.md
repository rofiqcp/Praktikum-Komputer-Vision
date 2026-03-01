# TUGAS VIDEO MODUL 12: REKONSTRUKSI 3D DAN IMAGE-BASED RENDERING

---

## Deskripsi Tugas
Buat video laporan yang mendemonstrasikan seluruh materi, 20 percobaan, dan project Rekonstruksi 3D & Image-Based Rendering. Fokus pada visualisasi 3D dan novel view synthesis.

---

## Struktur Video

### 1. Pembukaan (Maks 2 menit)
- Perkenalan: Nama, NIM, kelas, modul.
- Overview topik Rekonstruksi 3D dan Image-Based Rendering.

### 2. Penjelasan Materi (10–15 menit)
- Point cloud processing: filtering, normals, registration.
- Surface reconstruction: Poisson vs BPA vs Alpha Shapes.
- TSDF integration: prinsip dan pipeline.
- Image warping: forward/inverse, disocclusion.
- View interpolation: pose interpolation, blending.
- Neural rendering: NeRF dan 3D Gaussian Splatting (konsep).
- **Wajib**: Diagram pipeline lengkap (SfM → dense → mesh → render).

### 3. Demo 20 Percobaan (40–60 menit)

| No | Percobaan | Poin Penting |
|----|-----------|--------------|
| 1 | Point Cloud Basics | Load, visualisasi, properties |
| 2 | Filtering & Downsampling | Noise removal, voxel grid |
| 3 | Normal Estimation | Normal visualization, PCA |
| 4 | ICP Registration | P2P vs P2Plane, fitness |
| 5 | Surface Reconstruction | Poisson vs BPA perbandingan |
| 6 | Mesh Processing | Simplification, smoothing |
| 7 | TSDF Integration | Inkremental build dari depth |
| 8 | Image Warping | Forward warp, disocclusion |
| 9 | View Interpolation | Smooth transition video |
| 10 | Neural Rendering | MLP image regression, konsep |
| 11 | Ball Pivoting Detail | Multi-radius, coverage analysis |
| 12 | Alpha Shapes | Alpha parameter, concave shape |
| 13 | Mesh Texturing | Projecting color, trisurf |
| 14 | Point Cloud Colorization | 6 metode colorization |
| 15 | Point Cloud Segmentation | RANSAC plane, clustering |
| 16 | Marching Cubes | Isosurface, sphere/torus |
| 17 | Volumetric Rendering | MIP, average, first-surface |
| 18 | Forward vs Inverse Warp | Holes, splatting, bilinear |
| 19 | Light Field Basics | 4D, sub-aperture, refocus |
| 20 | 3D Visualization & Export | PLY/OBJ, turntable video |

- Tunjukkan visualisasi 3D dari berbagai angle (rotate di Open3D/matplotlib).
- Bandingkan metode secara visual side-by-side.

### 4. Demo Project (10–15 menit)
- Demo project soal cerita.
- Tunjukkan 3D output dan rendered views.

### 5. Analisis dan Penutup (5 menit)
- Perbandingan metode tradisional vs neural.
- Limitasi dan potensi ke depan.
- Kesimpulan seluruh modul.

---

## Ketentuan Teknis

| Aspek | Ketentuan |
|-------|-----------|
| Durasi | 75–100 menit |
| Resolusi | Minimal 1080p |
| Recording | Screen recording + webcam |
| Webcam | Tunjukkan objek fisik saat capture foto untuk rekonstruksi |
| Audio | Narasi jelas |
| Platform | YouTube (Unlisted) atau Google Drive |
| Format | MP4 |

---

## Rubrik Penilaian Video

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Pembukaan | 5% | Profesional |
| Penjelasan Materi | 15% | Akurat, diagram pipeline |
| Demo 20 Percobaan | 40% | Semua berjalan, 3D visualization |
| Demo Project | 20% | Fitur lengkap, 3D output |
| Analisis & Kesimpulan | 10% | Kritis, perbandingan metode |
| Kualitas Video | 10% | Resolusi, audio |

### Bonus & Penalti
| Item | Nilai |
|------|-------|
| Visualisasi 3D interaktif (rotate di Open3D/web viewer) | +5 |
| Demo NeRF/3DGS training + rendering | +5 |
| Rekonstruksi objek sendiri (bukan dataset) | +5 |
| Video < 45 menit | -10 |
| Tidak ada webcam | -5 |
| Audio tidak jelas | -5 |
| Percobaan error tanpa penjelasan | -5 per percobaan |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: Link YouTube (Unlisted) atau Google Drive.
- **Naming**: `[NIM]_[Nama]_Video_Modul12`
