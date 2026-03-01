# TUGAS VIDEO MODUL 11: STRUCTURE FROM MOTION DAN DEPTH ESTIMATION

---

## Deskripsi Tugas
Buat video laporan yang mendemonstrasikan seluruh materi, 10 percobaan, dan project SfM & Depth Estimation. Fokus pada visualisasi 3D dan depth maps.

---

## Struktur Video

### 1. Pembukaan (Maks 2 menit)
- Perkenalan: Nama, NIM, kelas, modul.
- Overview topik SfM dan Depth Estimation.

### 2. Penjelasan Materi (10–15 menit)
- Epipolar geometry: Fundamental vs Essential matrix.
- Triangulasi: prinsip ray intersection.
- SfM pipeline: features → matches → F/E → pose → triangulate → BA.
- Stereo vision: calibration, rectification, matching.
- BM vs SGBM: perbedaan cost function.
- Monocular depth: MiDaS, limitasi.
- **Wajib**: Diagram epipolar geometry + SfM pipeline + stereo pipeline.

### 3. Demo 10 Percobaan (30–40 menit)

| No | Percobaan | Poin Penting |
|----|-----------|--------------|
| 1 | Feature Matching Multi-View | Matches antar 5 view |
| 2 | Fundamental Matrix | Epipolar lines visualization |
| 3 | Essential Matrix + Pose | R, t recovery |
| 4 | Triangulasi 3D | 3D point cloud plot |
| 5 | Visual Odometry | Trajectory plot |
| 6 | Stereo Calibration | Checkerboard, parameters |
| 7 | Stereo Rectification | Horizontal alignment verification |
| 8 | Block Matching | Disparity map, parameter tuning |
| 9 | SGBM | BM vs SGBM comparison |
| 10 | Monocular Depth | MiDaS on multiple images |

- Tunjukkan checkerboard fisik dan proses kalibrasi.
- Tunjukkan gambar 3D dari berbagai angle.

### 4. Demo Project (10–15 menit)
- Demo project soal cerita.
- Tunjukkan 3D output (point cloud, depth map).

### 5. Analisis dan Penutup (5 menit)
- Rangkuman perbandingan metode.
- Limitasi dan potensi.
- Kesimpulan.

---

## Ketentuan Teknis

| Aspek | Ketentuan |
|-------|-----------|
| Durasi | 60–75 menit |
| Resolusi | Minimal 1080p |
| Recording | Screen recording + webcam |
| Webcam | Tunjukkan checkerboard + objek saat capture |
| Audio | Narasi jelas |
| Platform | YouTube (Unlisted) atau Google Drive |
| Format | MP4 |

---

## Rubrik Penilaian Video

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Pembukaan | 5% | Profesional |
| Penjelasan Materi | 15% | Akurat, diagram epipolar |
| Demo 10 Percobaan | 40% | Semua berjalan, 3D visualization |
| Demo Project | 20% | Fitur lengkap |
| Analisis & Kesimpulan | 10% | Kritis |
| Kualitas Video | 10% | Resolusi, audio |

### Bonus & Penalti
| Item | Nilai |
|------|-------|
| Visualisasi 3D interaktif (Open3D/rotating plot) | +5 |
| Kalibrasi dengan checkerboard sendiri | +5 |
| Video < 45 menit | -10 |
| Tidak ada webcam | -5 |
| Audio tidak jelas | -5 |
| Percobaan error tanpa penjelasan | -5 per percobaan |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: Link YouTube (Unlisted) atau Google Drive.
- **Naming**: `[NIM]_[Nama]_Video_Modul11`
