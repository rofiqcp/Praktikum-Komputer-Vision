# TUGAS VIDEO MODUL 7: DETEKSI FITUR DAN PENCOCOKAN

---

## Deskripsi Tugas
Buat video laporan praktikum yang mendemonstrasikan seluruh materi, 20 percobaan, dan project feature detection & matching.

---

## Struktur Video

### 1. Pembukaan (Maks 2 menit)
- Perkenalan: Nama, NIM, kelas, modul.
- Overview singkat topik Feature Detection & Matching.

### 2. Penjelasan Materi (15–20 menit)
- Corner detection: Harris, Shi-Tomasi — formula dan interpretasi.
- Scale-space dan blob detection: LoG, DoG.
- Detektor modern: SIFT, ORB, AKAZE, FAST — arsitektur dan perbedaan.
- Matching: BF, FLANN, Lowe's ratio test.
- Geometric verification: Homography + RANSAC.
- **Wajib**: Diagram/slide perbandingan detektor + pipeline matching.

### 3. Demo 20 Percobaan (50–65 menit)

| No | Percobaan | Poin Penting |
|----|-----------|--------------|
| 1 | Harris Corner Detection | Heatmap response, parameter tuning |
| 2 | Shi-Tomasi Corner Detection | Comparison Harris vs Shi-Tomasi |
| 3 | SIFT Detection | Rich keypoints, scale + orientasi |
| 4 | ORB Detection | Speed comparison vs SIFT |
| 5 | AKAZE + FAST | Multi-detector comparison table |
| 6 | Brute-Force Matching | SIFT vs ORB matching visual |
| 7 | FLANN + Ratio Test | Ratio threshold sweep |
| 8 | Homography + RANSAC | Inlier/outlier visual, warped overlay |
| 9 | Object Detection via Features | Template detection di scene |
| 10 | Feature Invariance Rotasi | Keypoint consistency di berbagai sudut rotasi |
| 11 | Feature Invariance Skala | Keypoint consistency di berbagai ukuran skala |
| 12 | Feature Invariance Iluminasi | Robustness terhadap brightness/contrast/gamma |
| 13 | Perbandingan Deskriptor | SIFT vs ORB vs AKAZE: dimensi, speed, akurasi |
| 14 | Geometric Verification Detail | RANSAC tuning, inlier analysis, reprojection error |
| 15 | Image Retrieval | Database indexing + query retrieval |
| 16 | AR Marker Detection | Marker detection + homography overlay |
| 17 | Keypoint Repeatability | Repeatability rate di berbagai transformasi |
| 18 | Multi-Image Matching | Feature tracks across 3+ gambar |
| 19 | Feature Matching Pipeline | Class-based modular pipeline |
| 20 | Proyek Feature Matching App | Aplikasi lengkap + evaluasi |

- Jalankan kode LIVE.
- Tunjukkan objek fisik (buku, kartu) saat demo matching/detection.

### 4. Demo Project (10–15 menit)
- Tunjukkan project yang dipilih dari Soal Cerita.
- Demo fitur lengkap.
- Tunjukkan metrik evaluasi.

### 5. Analisis dan Penutup (5 menit)
- Rangkuman temuan dan perbandingan metode.
- Tantangan dan solusi.
- Kesimpulan dan rekomendasi.

---

## Ketentuan Teknis

| Aspek | Ketentuan |
|-------|-----------|
| Durasi | 90–120 menit |
| Resolusi | Minimal 1080p |
| Recording | Screen recording + webcam (picture-in-picture) |
| Webcam | Wajah terlihat jelas, tunjukkan objek fisik saat demo |
| Audio | Narasi jelas |
| Platform | YouTube (Unlisted) atau Google Drive |
| Format | MP4 |

---

## Rubrik Penilaian Video

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Pembukaan | 5% | Profesional, lengkap |
| Penjelasan Materi | 15% | Akurat, diagram jelas |
| Demo 20 Percobaan | 40% | Semua berjalan, parameter tuning |
| Demo Project | 20% | Fitur lengkap, berjalan baik |
| Analisis & Kesimpulan | 10% | Kritis, perbandingan kuantitatif |
| Kualitas Video | 10% | Resolusi, audio, editing |

### Bonus & Penalti
| Item | Nilai |
|------|-------|
| Demo real-time dengan objek fisik | +5 |
| Tabel perbandingan komprehensif (semua detektor+matcher) | +5 |
| Video < 60 menit | -10 |
| Tidak ada webcam | -5 |
| Audio tidak jelas | -5 |
| Percobaan error tanpa penjelasan | -5 per percobaan |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: Link YouTube (Unlisted) atau Google Drive.
- **Naming**: `[NIM]_[Nama]_Video_Modul07`
