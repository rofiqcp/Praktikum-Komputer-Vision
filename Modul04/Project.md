# PROJECT MODUL 4: MODEL FITTING DAN OPTIMASI

---

## Deskripsi Umum
Project mengintegrasikan konsep model fitting (RANSAC, Hough, homografi) dan optimasi (regularisasi, interpolasi, optical flow). Pilih minimal 1 soal cerita.

---

## Daftar Improvisasi Percobaan (15 Pengembangan)

1. **Automatic Lane Detector** — Hough lines + RANSAC untuk deteksi jalur jalan dari video dashcam.
2. **Coin Counter** — Hough circles + feature matching untuk menghitung dan mengklasifikasi koin.
3. **Robust Image Aligner** — Feature matching + RANSAC homography untuk mengoreksi pasangan gambar.
4. **Vanishing Point Detector** — Hough lines + intersection analysis untuk menemukan titik hilang.
5. **Panorama Stitcher Sederhana** — Feature matching + homography untuk menjahit 2 gambar.
6. **Motion Tracker** — Optical flow + trail visualization untuk tracking objek bergerak.
7. **Sparse-to-Dense Depth** — RBF interpolation untuk menghasilkan depth map dense dari sparse points.
8. **Robust Line Fitter App** — Interactive RANSAC demo: user klik titik, lihat fitting real-time.
9. **Circle/Ellipse Detector** — Hough + RANSAC untuk deteksi bentuk di gambar industri.
10. **Denoising Benchmark** — Bandingkan 5+ metode denoising (TV, bilateral, NLM, BM3D, dll.).
11. **Homography-based AR** — Overlay gambar virtual pada planar surface menggunakan homografi.
12. **Multi-model RANSAC** — Deteksi multiple lines/circles secara bersamaan.
13. **Grid Pattern Detector** — Hough + intersections untuk mendeteksi grid (checkerboard, spreadsheet).
14. **Video Stabilizer Sederhana** — Optical flow + homografi untuk stabilisasi video.
15. **Interpolated Animation** — RBF untuk morphing control points antar dua konfigurasi.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Penghitung Kendaraan di Jalan Tol
Operator jalan tol memerlukan sistem penghitung kendaraan dari rekaman CCTV. Buatlah sistem yang: (a) deteksi garis batas counting (Hough lines), (b) optical flow untuk mendeteksi arah gerakan, (c) RANSAC untuk mengestimasi kecepatan rata-rata kendaraan, (d) Hough circles untuk mendeteksi roda (estimasi jenis kendaraan), (e) tracking via optical flow, (f) statistik per jam, (g) visualisasi hasil.

### Soal 2: Document Scanner Pro
Startup digitasi dokumen memerlukan scanner otomatis dari foto smartphone. Buatlah: (a) deteksi tepi dokumen, (b) Hough lines untuk mendeteksi garis tepi, (c) intersection 4 garis → 4 sudut dokumen, (d) homografi untuk koreksi perspektif, (e) enhancement (contrast, sharpen), (f) batch processing, (g) export PDF-ready.

### Soal 3: Sistem Pencacah Objek Lingkaran dalam Industri
Pabrik bearing memerlukan tool QC untuk menghitung jumlah bearing di nampan. Buatlah: (a) preprocessing (blur, contrast), (b) Hough circle detection, (c) RANSAC filtering untuk false positive, (d) klasifikasi ukuran bearing, (e) penandaan bearing yang cacat (tidak bulat sempurna), (f) statistik dan report, (g) visualisasi anotasi.

### Soal 4: Lane Keeping Assistant Simulator
Tim riset self-driving car memerlukan prototipe lane detection. Buatlah: (a) edge detection pada gambar jalan, (b) Hough line detection untuk kandidat lane, (c) RANSAC untuk memilih lane yang valid, (d) vanishing point estimation, (e) overlay lane markers pada gambar, (f) estimasi deviasi dari center lane, (g) alarm jika keluar jalur.

### Soal 5: Object Tracking dalam Video Olahraga
Klub bola lokal ingin menganalisis pergerakan bola dari rekaman. Buatlah: (a) deteksi bola (Hough circles), (b) optical flow tracking, (c) RANSAC untuk mengestimasi trajectory, (d) RBF interpolation untuk smooth trajectory, (e) estimasi kecepatan bola, (f) visualisasi trajectory, (g) statistik per babak.

### Soal 6: Panorama Creator dari Webcam
Aplikasi foto ingin fitur sweep panorama. Buatlah: (a) capture multiple frames dari webcam, (b) feature detection dan matching, (c) homography estimation (RANSAC-based), (d) warp dan blend images, (e) handle drift (bundle adjustment sederhana), (f) crop hasil, (g) simpan panorama.

### Soal 7: Sistem Deteksi Kerusakan Jalan
Dinas PU memerlukan tool analisis kerusakan jalan dari foto drone. Buatlah: (a) edge detection untuk mendeteksi retakan, (b) Hough lines untuk mendeteksi pola retakan, (c) RANSAC untuk memisahkan retakan nyata dari noise, (d) morfologi untuk menghubungkan retakan terputus, (e) klasifikasi severity, (f) hitung persentase area rusak, (g) laporan GPS-tagged (simulasi).

### Soal 8: Sistem Registrasi Gambar Satelit
Tim pemetaan ingin mengoreksi gambar satelit dari waktu berbeda. Buatlah: (a) feature matching antar dua gambar temporal, (b) RANSAC-based homography, (c) warp salah satu gambar ke referensi, (d) difference image untuk mendeteksi perubahan, (e) overlay with transparency, (f) annotasi area berubah, (g) report.

### Soal 9: Motion Detection Security System
Sistem keamanan rumah memerlukan deteksi gerakan dari webcam. Buatlah: (a) optical flow antar frame, (b) threshold magnitude flow → motion mask, (c) RANSAC untuk memisahkan camera shake dari object motion, (d) bounding box pada area bergerak, (e) logging timestamp event, (f) rekam clip saat motion terdeteksi, (g) alert level (low/medium/high).

### Soal 10: Tool Analisis Gerak Atletik
Pelatih lari ingin menganalisis teknik atlet dari video. Buatlah: (a) optical flow dense pada area tubuh atlet, (b) tracking joint points, (c) RBF interpolation untuk smooth trajectory, (d) hitung kecepatan per joint, (e) identifikasi fase lari (stance, swing), (f) perbandingan antar atlet, (g) visualisasi skeleton + flow overlay.

---

## Rubrik Penilaian Project

| Komponen | Bobot |
|----------|-------|
| Fungsionalitas | 35% |
| Integrasi Percobaan | 20% |
| Kualitas Kode | 15% |
| Dokumentasi | 15% |
| Kreativitas | 15% |

*(Detail dan konversi nilai mengikuti format standar)*

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project04.zip`
