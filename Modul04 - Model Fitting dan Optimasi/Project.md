# PROJECT MODUL 4: MODEL FITTING DAN OPTIMASI

---

## Deskripsi Umum

Project ini mengintegrasikan seluruh konsep model fitting dan optimasi dari 20 percobaan: least squares (OLS/WLS/TLS), RANSAC, Hough Transform, homography, regularisasi, fitting kontur, template matching, segmentasi, optical flow, cross-validation, denoising, dan pipeline gabungan. Mahasiswa memilih minimal 1 soal cerita dan mengimplementasikan solusi yang menggunakan minimal **10 dari 20 konsep** percobaan.

---

## Daftar Improvisasi Percobaan (20 Pengembangan)

1. **Automatic Lane Detector** — Hough lines + RANSAC untuk deteksi jalur jalan dari dashcam.
2. **Coin Counter** — Hough circles + fitting untuk menghitung dan mengklasifikasi koin.
3. **Robust Image Aligner** — Feature matching + RANSAC homography untuk mengoreksi gambar.
4. **Vanishing Point Detector** — Hough lines + intersection analysis untuk titik hilang.
5. **Panorama Stitcher Sederhana** — Feature matching + homography untuk menjahit 2 gambar.
6. **Motion Tracker** — Optical flow + trail visualization untuk tracking objek bergerak.
7. **Robust Line Fitter App** — Interactive RANSAC demo: user klik titik, lihat fitting real-time.
8. **Circle/Ellipse Detector** — Hough + RANSAC + fitting ellips untuk deteksi bentuk.
9. **Denoising Benchmark** — Bandingkan 5+ metode denoising (NLM, TV, bilateral, dll.) dengan PSNR.
10. **Homography-based AR** — Overlay gambar virtual pada planar surface.
11. **Multi-model RANSAC** — Deteksi multiple lines/circles secara bersamaan.
12. **Grid Pattern Detector** — Hough + intersections untuk mendeteksi checkerboard/grid.
13. **Video Stabilizer** — Optical flow + homografi untuk stabilisasi video goyang.
14. **GrabCut Interactive Segmenter** — Segmentasi objek interaktif dengan GUI.
15. **Template-based Object Finder** — Multi-scale template matching + NMS.
16. **Regularization Hyperparameter Tuner** — Auto-tune Ridge/Lasso alpha via CV.
17. **Optical Flow Visualizer** — HSV + quiver + warping untuk analisis gerakan.
18. **Cross-Validation Model Selector** — Benchmark OLS vs WLS vs TLS vs RANSAC via K-Fold.
19. **Document Scanner Pro** — Edge → Hough → homography → warp → enhance pipeline.
20. **Fitting Method Comparator** — Dashboard perbandingan semua metode fitting pada dataset.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Penghitung Kendaraan di Jalan Tol
Operator jalan tol memerlukan sistem penghitung kendaraan dari rekaman CCTV. Buatlah sistem yang: (a) deteksi garis batas counting (Hough lines), (b) optical flow untuk mendeteksi arah gerakan, (c) RANSAC untuk mengestimasi kecepatan rata-rata kendaraan, (d) Hough circles untuk mendeteksi roda, (e) tracking via optical flow, (f) statistik per jam, (g) visualisasi hasil.

### Soal 2: Document Scanner Pro
Startup digitasi dokumen memerlukan scanner otomatis dari foto smartphone. Buatlah: (a) deteksi tepi dokumen, (b) Hough lines untuk mendeteksi garis tepi, (c) intersection 4 garis → 4 sudut, (d) homografi untuk koreksi perspektif, (e) enhancement (contrast, sharpen), (f) batch processing, (g) export standar.

### Soal 3: Sistem QC Bearing Industri
Pabrik bearing memerlukan tool QC untuk menghitung bearing di nampan. Buatlah: (a) preprocessing, (b) Hough circle detection, (c) RANSAC filtering false positive, (d) fitting ellips untuk klasifikasi ukuran, (e) penandaan bearing cacat, (f) statistik dan report, (g) visualisasi anotasi.

### Soal 4: Lane Keeping Assistant Simulator
Tim riset self-driving car memerlukan prototipe lane detection. Buatlah: (a) edge detection, (b) Hough line detection, (c) RANSAC lane selection, (d) vanishing point estimation, (e) overlay lane markers, (f) estimasi deviasi dari center, (g) alarm jika keluar jalur.

### Soal 5: Object Tracking dalam Video Olahraga
Klub bola ingin menganalisis pergerakan bola. Buatlah: (a) deteksi bola (Hough circles), (b) optical flow tracking, (c) RANSAC trajectory estimation, (d) smooth trajectory, (e) estimasi kecepatan bola, (f) visualisasi trajectory, (g) statistik per babak.

### Soal 6: Panorama Creator dari Webcam
Aplikasi foto ingin fitur sweep panorama. Buatlah: (a) capture multiple frames, (b) feature detection + matching, (c) homography estimation (RANSAC), (d) warp + blend, (e) handle drift, (f) crop hasil, (g) simpan panorama.

### Soal 7: Sistem Deteksi Kerusakan Jalan
Dinas PU memerlukan tool analisis kerusakan jalan. Buatlah: (a) edge detection retakan, (b) Hough lines pola retakan, (c) RANSAC filtering, (d) morfologi untuk menghubungkan retakan, (e) klasifikasi severity, (f) hitung area rusak, (g) laporan.

### Soal 8: Sistem Registrasi Gambar Satelit
Tim pemetaan ingin mengoreksi gambar satelit temporal. Buatlah: (a) feature matching, (b) RANSAC homography, (c) warp ke referensi, (d) difference image, (e) overlay transparency, (f) annotasi area berubah, (g) report.

### Soal 9: Motion Detection Security System
Sistem keamanan memerlukan deteksi gerakan dari webcam. Buatlah: (a) optical flow antar frame, (b) threshold magnitude → motion mask, (c) RANSAC memisahkan camera shake vs object motion, (d) bounding box, (e) logging timestamp, (f) rekam clip saat motion, (g) alert level.

### Soal 10: Tool Analisis Gerak Atletik
Pelatih lari ingin menganalisis teknik atlet dari video. Buatlah: (a) dense optical flow pada area tubuh, (b) tracking joint points, (c) smooth trajectory, (d) hitung kecepatan per joint, (e) identifikasi fase lari, (f) perbandingan antar atlet, (g) visualisasi skeleton + flow overlay.

---

## Rubrik Penilaian Project

| Komponen | Bobot | Deskripsi |
|----------|-------|-----------|
| **Fungsionalitas** | 35% | Semua fitur berjalan dengan benar dan robust |
| **Integrasi Percobaan** | 20% | Minimal 10 dari 20 konsep percobaan terintegrasi |
| **Kualitas Kode** | 15% | Modular, terdokumentasi, penamaan deskriptif |
| **Dokumentasi** | 15% | README, screenshot, penjelasan alur program |
| **Kreativitas** | 15% | Fitur tambahan, UI, solusi inovatif |

### Detail Penilaian Integrasi
| Jumlah Konsep | Skor Integrasi |
|---------------|----------------|
| 15–20 konsep | 90–100% |
| 10–14 konsep | 70–89% |
| 7–9 konsep | 50–69% |
| 4–6 konsep | 30–49% |
| < 4 konsep | 0–29% |

### Skala Nilai
| Range | Grade | Predikat |
|-------|-------|----------|
| 90–100 | A | Sangat Memuaskan |
| 80–89 | AB | Memuaskan |
| 70–79 | B | Baik |
| 60–69 | BC | Cukup Baik |
| 50–59 | C | Cukup |
| 40–49 | D | Kurang |
| 0–39 | E | Gagal |

### Penalti
| Pelanggaran | Penalti |
|-------------|---------|
| Terlambat 1 hari | −10% |
| Terlambat 2–3 hari | −20% |
| Terlambat > 3 hari | −40% |
| Plagiarisme | Nilai 0 + sanksi akademik |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project04.zip`
- **Isi**: Source code, README.md, screenshot (min. 5), data sample.
