# PROJECT MODUL 11: STRUCTURE FROM MOTION DAN DEPTH ESTIMATION

---

## Deskripsi Umum
Project mengintegrasikan konsep multi-view geometry: epipolar geometry, triangulasi, SfM, stereo matching, dan monocular depth estimation. Pilih minimal 1 soal cerita.

---

## Daftar Improvisasi Percobaan (15 Pengembangan)

1. **Full SfM Pipeline** — Rekonstruksi 3D objek dari 10+ gambar (incremental SfM lengkap).
2. **Dense Reconstruction** — Gunakan stereo matching + SfM → dense point cloud.
3. **Stereo Video Depth** — Real-time depth dari dual webcam.
4. **Depth-based Obstacle Detection** — Alert jika objek terlalu dekat (stereo atau monocular).
5. **3D Object Scanner** — Capture objek dari turntable → reconstruct.
6. **Visual Odometry with Scale** — VO dengan scale recovery menggunakan known object.
7. **Disparity Refinement** — Guided filter, bilateral filter, atau confidence-based refinement.
8. **Multi-baseline Stereo** — 3+ kamera untuk mengisi holes pada disparity.
9. **Depth Colorized Video** — Overlay depth as heatmap pada video real-time.
10. **Egomotion from Dashcam** — Estimasi trajectory kendaraan dari video dashcam.
11. **Plane Detection from Depth** — Deteksi bidang datar (lantai, dinding) dari depth map.
12. **Depth-aware Image Editing** — Selection berdasarkan depth (select foreground/background).
13. **Stereo Anaglyph Generator** — Buat gambar 3D anaglyph (red-cyan) dari stereo pair.
14. **Loop Closure Detection** — Deteksi saat kamera kembali ke lokasi yang sama (SLAM basic).
15. **Depth Map Super Resolution** — Upscale sparse/low-res depth → dense high-res.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: 3D Object Digitizer
Museum ingin mendigitalisasi artefak menjadi model 3D. Buatlah: (a) capture 15+ foto objek dari berbagai sudut, (b) feature matching multi-view, (c) SfM: recover poses + triangulasi, (d) dense point cloud, (e) visualisasi 3D interaktif (Open3D/Matplotlib), (f) export PLY/OBJ, (g) color point cloud.

### Soal 2: Sistem Navigasi Robot (Visual Odometry)
Robot warehouse memerlukan navigasi visual. Buatlah: (a) rekam video dari "robot" (gerak maju, belok), (b) feature tracking frame-by-frame, (c) essential matrix → recover pose, (d) plot trajectory bird's-eye view, (e) deteksi obstacle dari depth (monocular), (f) alert jarak dekat, (g) visualisasi trajectory + orientation.

### Soal 3: Depth Map dari Dual Kamera Smartphone
Developer aplikasi ingin menggunakan dual kamera HP untuk depth. Buatlah: (a) simulasikan stereo (2 foto geser horizontal), (b) stereo calibration, (c) rectification, (d) SGBM disparity, (e) depth map, (f) synthetic bokeh dari depth, (g) depth-based foreground segmentation.

### Soal 4: 3D Scene Reconstruction dari CCTV
Arsitek ingin merekonstruksi ruangan dari 2 CCTV. Buatlah: (a) calibrate 2 kamera, (b) stereo rectification, (c) disparity + depth, (d) triangulasi titik 3D, (e) visualisasi point cloud ruangan, (f) estimasi dimensi ruangan (panjang, lebar), (g) floor plan sederhana dari top-view projection.

### Soal 5: Augmented Reality Depth Demo
Developer AR ingin tool yang menggunakan depth untuk placement objek. Buatlah: (a) monocular depth estimation (MiDaS), (b) deteksi bidang datar (floor) dari depth, (c) "place" virtual objek pada floor, (d) occlusion handling (objek di depan menutupi virtual object), (e) real-time dari webcam, (f) ukuran objek sesuai depth, (g) multiple virtual objects.

### Soal 6: Sistem Pengukuran Jarak dari Stereo
Surveyor memerlukan tool pengukuran jarak dari foto. Buatlah: (a) stereo calibration, (b) rectification + SGBM, (c) depth map metrik, (d) klik 2 titik → hitung jarak 3D, (e) klik objek → hitung depth, (f) accuracy validation (ukur objek diketahui), (g) export measurements.

### Soal 7: Monocular Depth untuk Video Editing
Editor video ingin tool depth-based effects. Buatlah: (a) monocular depth per frame, (b) depth-based fog effect (semakin jauh semakin kabur), (c) depth-based color grading (warna berbeda per distance), (d) synthetic tilt-shift, (e) parallax 2.5D effect, (f) real-time preview, (g) export video.

### Soal 8: Building Facade Reconstruction
Developer perlu merekonstruksi facade bangunan dari foto jalan. Buatlah: (a) 5+ foto facade dari sudut berbeda, (b) SfM: recover cameras + 3D points, (c) dense matching untuk facade, (d) plane fitting pada facade, (e) texture mapping sederhana, (f) dimensi estimasi, (g) 3D visualization.

### Soal 9: Stereo Anaglyph Photo Tool
Fotografer ingin membuat foto 3D anaglyph. Buatlah: (a) capture stereo pair (shift kamera), (b) auto-alignment, (c) generate red-cyan anaglyph, (d) adjustable parallax, (e) crop + optimize, (f) support batch mode, (g) export anaglyph JPEG.

### Soal 10: Parking Distance Estimator
Sistem parkir memerlukan estimasi jarak kendaraan. Buatlah: (a) stereo kamera simulasi, (b) calibration, (c) depth estimation (SGBM), (d) deteksi objek (YOLO) + depth per objek, (e) display jarak di layar, (f) alert jika < 1 meter, (g) bird's-eye view parking slot.

---

## Rubrik Penilaian Project

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Fungsionalitas | 35% | Semua fitur berjalan |
| Integrasi Percobaan | 20% | Menggunakan konsep ≥5 percobaan |
| Kualitas Kode | 15% | Clean, modular |
| Dokumentasi | 15% | README, screenshot, 3D visualisasi |
| Kreativitas | 15% | Fitur tambahan, 3D visualization, accuracy analysis |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project11.zip`
- **Isi**: Source code, dataset gambar, calibration data, README.md, 3D output screenshots.
