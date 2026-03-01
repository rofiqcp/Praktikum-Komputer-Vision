# TUGAS VIDEO MODUL 9: ESTIMASI GERAK (MOTION ESTIMATION)

---

## Deskripsi Tugas
Buat video laporan yang mendemonstrasikan seluruh materi, 20 percobaan, dan project motion estimation. Fokus pada demo real-time dan visualisasi yang jelas.

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

### 3. Demo 20 Percobaan (40–60 menit)

| No | Percobaan | Poin Penting |
|----|-----------|--------------|
| 1 | Sparse Optical Flow (LK) | Trail tracking, re-detection |
| 2 | Dense Optical Flow (Farnebäck) | HSV visualization, flow arrows |
| 3 | Visualisasi Optical Flow | Arrow plot, HSV, magnitude |
| 4 | Background Subtraction MOG2 | Learning rate, shadow detection |
| 5 | Background Subtraction KNN | Perbandingan dengan MOG2 |
| 6 | Frame Differencing | Gerakan antar frame, threshold |
| 7 | Running Average Background | Alpha parameter, adaptasi |
| 8 | Object Tracking CSRT | ROI selection, accuracy |
| 9 | Object Tracking KCF | Speed vs accuracy trade-off |
| 10 | Multi-Object Tracking | Multiple ROI, trajectory |
| 11 | Motion History Image | MHI temporal, motion segments |
| 12 | Video Stabilization | Before/after, trajectory smooth |
| 13 | Frame Interpolation Linear | Blending antar frame |
| 14 | Frame Interpolation Flow | Optical flow-based warping |
| 15 | Magnitude & Arah Flow | Analisis kuantitatif flow |
| 16 | Feature Trajectory | Long-term tracking, trail |
| 17 | BGS Comparison | MOG2 vs KNN vs frame diff |
| 18 | Deteksi Gerakan Contour | Contour area, bounding box |
| 19 | Optical Flow Real-Time | Simulasi interaktif |
| 20 | Estimasi Kecepatan Objek | Piksel/frame → km/h |

- Demo REAL-TIME dari webcam untuk percobaan 1–10.
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
| Durasi | 75–100 menit |
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
| Demo 20 Percobaan | 40% | Semua berjalan, real-time demo |
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
