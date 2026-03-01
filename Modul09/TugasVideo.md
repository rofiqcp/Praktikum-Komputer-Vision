# TUGAS VIDEO MODUL 9: ESTIMASI GERAK (MOTION ESTIMATION)

---

## Deskripsi Tugas
Buat video laporan yang mendemonstrasikan seluruh materi, 10 percobaan, dan project motion estimation. Fokus pada demo real-time dan visualisasi yang jelas.

---

## Struktur Video

### 1. Pembukaan (Maks 2 menit)
- Perkenalan: Nama, NIM, kelas, modul.
- Overview topik Motion Estimation.

### 2. Penjelasan Materi (10–15 menit)
- Optical flow: brightness constancy, aperture problem.
- Lucas-Kanade: formulasi, pyramidal, sparse.
- Farnebäck: dense flow, visualisasi HSV.
- Background subtraction: MOG2, KNN.
- Object tracking: KCF, CSRT, MOSSE.
- MHI, video stabilization, frame interpolation.
- **Wajib**: Diagram optical flow equation + pipeline stabilization.

### 3. Demo 10 Percobaan (30–40 menit)

| No | Percobaan | Poin Penting |
|----|-----------|--------------|
| 1 | Sparse Optical Flow (LK) | Trail tracking, re-detection |
| 2 | Dense Optical Flow (Farnebäck) | HSV visualization, flow arrows |
| 3 | Background Subtraction | MOG2 vs KNN, morphology cleanup |
| 4 | Single Object Tracking | Tracker comparison table |
| 5 | Multi-Object Tracking | Trajectory visualization |
| 6 | Motion History Image | MHI + motion segments |
| 7 | Video Stabilization | Before/after + trajectory plot |
| 8 | Phase Correlation | Translational alignment demo |
| 9 | Frame Interpolation | Linear vs flow-based comparison |
| 10 | Activity Recognition | Confusion matrix, accuracy |

- Demo REAL-TIME dari webcam untuk percobaan 1–6.
- Tunjukkan gerakan fisik (lambaikan tangan, bergerak) di depan kamera.

### 4. Demo Project (10–15 menit)
- Demo project soal cerita.
- Tunjukkan input video → processing → output.

### 5. Analisis dan Penutup (5 menit)
- Rangkuman perbandingan metode.
- Tantangan real-time processing.
- Kesimpulan.

---

## Ketentuan Teknis

| Aspek | Ketentuan |
|-------|-----------|
| Durasi | 60–75 menit |
| Resolusi | Minimal 1080p |
| Recording | Screen recording + webcam |
| Webcam | Tunjukkan gerakan fisik saat demo optical flow/tracking |
| Audio | Narasi jelas |
| Platform | YouTube (Unlisted) atau Google Drive |
| Format | MP4 |

---

## Rubrik Penilaian Video

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Pembukaan | 5% | Profesional |
| Penjelasan Materi | 15% | Akurat, diagram |
| Demo 10 Percobaan | 40% | Semua berjalan, real-time demo |
| Demo Project | 20% | Fitur lengkap |
| Analisis & Kesimpulan | 10% | Kritis |
| Kualitas Video | 10% | Resolusi, audio |

### Bonus & Penalti
| Item | Nilai |
|------|-------|
| Demo real-time webcam untuk ≥5 percobaan | +5 |
| Tabel perbandingan tracker + flow lengkap | +5 |
| Video < 45 menit | -10 |
| Tidak ada webcam | -5 |
| Audio tidak jelas | -5 |
| Percobaan error tanpa penjelasan | -5 per percobaan |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: Link YouTube (Unlisted) atau Google Drive.
- **Naming**: `[NIM]_[Nama]_Video_Modul09`
