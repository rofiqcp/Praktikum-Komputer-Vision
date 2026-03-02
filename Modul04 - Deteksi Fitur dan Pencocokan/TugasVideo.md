# TUGAS VIDEO - Modul 4: Deteksi Fitur dan Pencocokan

---

## Deskripsi Tugas

Buatlah video edukatif berdurasi **10-15 menit** yang mendemonstrasikan implementasi dan hasil dari topik-topik dalam Modul 4: Deteksi Fitur dan Pencocokan. Video ini dimaksudkan sebagai dokumentasi pembelajaran sekaligus sarana untuk memaparkan pemahaman Anda kepada audiens teknis.

---

## Ketentuan Umum

| Aspek | Ketentuan |
|-------|-----------|
| Durasi | 10-15 menit (tidak boleh lebih dari 15 menit) |
| Format | MP4 (H.264), resolusi minimal 1280x720 (HD) |
| Suara | Audio narasi wajib ada, jelas, dan sinkron dengan visual |
| Bahasa | Indonesia (campuran Inggris untuk istilah teknis diperbolehkan) |
| Mode | Individu atau kelompok (maks. 2 orang) |
| Presentator | Wajah dan/atau layar komputer terlihat |

---

## Struktur Video yang Direkomendasikan

### Bagian 1: Pembukaan (1-2 menit)
- Perkenalan: nama, NIM, topik yang dibahas
- Overview singkat Modul 4 dan mengapa deteksi fitur penting
- Daftar topik yang akan didemonstrasikan
- Tampilkan contoh aplikasi nyata (screenshot SLAM, AR, panorama) untuk motivasi

### Bagian 2: Deteksi Keypoint (2-3 menit)
Demonstrasikan minimal **4 dari 6 topik** berikut secara visual dan berikan narasi penjelasan:
- Percobaan 1: Harris corner - tampilkan R response map + corner yang terdeteksi
- Percobaan 2: Shi-Tomasi - tampilkan goodFeaturesToTrack + hasil cornerSubPix
- Percobaan 3: SIFT keypoints - tampilkan DRAW_RICH_KEYPOINTS (orientasi + skala)
- Percobaan 4: ORB keypoints - bandingkan distribusi dengan SIFT secara visual
- Percobaan 5: AKAZE vs FAST - tampilkan perbedaan NMS aktif vs nonaktif
- Sertakan narasi: "Perbedaan utama antara Harris dan SIFT adalah..."

### Bagian 3: Matching dan Verifikasi (3-4 menit)
Demonstrasikan minimal **5 dari 8 topik** berikut:
- Percobaan 6: BFMatcher - tampilkan semua match, kemudian setelah ratio test 0.75
- Percobaan 7: FLANN matching - bandingkan kecepatan FLANN vs BF pada terminal
- Percobaan 8: Homography + RANSAC - tampilkan inlier (hijau) vs outlier (merah)
- Percobaan 9: Object detection - tampilkan bounding polygon pada scene
- Percobaan 14: Geometric verification detail - tampilkan statistik inlier ratio
- Percobaan 16: AR marker - tracking marker dalam video (atau frame diam)
- Percobaan 18: Multi-image matching - tampilkan similarity matrix heatmap
- Percobaan 19: Pipeline lengkap - jalankan dan tampilkan setiap tahap

### Bagian 4: Analisis dan Perbandingan (2-3 menit)
Tampilkan hasil analisis dari minimal **4 dari 6 topik** berikut:
- Percobaan 10: Grafik match rate vs sudut rotasi (SIFT vs ORB vs AKAZE)
- Percobaan 11: Grafik match rate vs scale factor
- Percobaan 12: Bar chart ketahanan iluminasi berbagai detektor
- Percobaan 13: Tabel benchmark deskriptor (waktu, dimensi, precision)
- Percobaan 15: Image retrieval - tampilkan query + top-3 hasil
- Percobaan 17: Keypoint repeatability score bar chart
- Percobaan 20: Aplikasi final berjalan end-to-end

### Bagian 5: Penutup (1 menit)
- Ringkasan temuan: detektor mana terbaik untuk apa
- Jawaban singkat: apa yang paling menarik/sulit dari modul ini?
- Kesimpulan pribadi tentang aplikasi fitur lokal dalam computer vision

---

## Persyaratan Teknologi Demo

### Requirements Wajib
1. Semua kode yang dijalankan harus **benar-benar berjalan** pada demo (bukan mock/screenshot statis)
2. Tampilkan **output terminal** (print statements) untuk nilai numerik (waktu, jumlah match, dll.)
3. Semua visualisasi gambar harus **jelas dan tidak blur** dalam video
4. Minimal **2 gambar berbeda** digunakan untuk menunjukkan generalitas

### Requirements Visual
- Ukuran font terminal/IDE harus cukup besar untuk dibaca dalam video (minimal 14pt)
- Zoom in pada bagian penting jika menggunakan resolusi tinggi
- Gunakan warna yang kontras untuk overlay keypoint dan match lines

---

## Kriteria Penilaian Video

| Komponen | Bobot | Indikator |
|----------|-------|-----------|
| Kelengkapan Demo | 30% | Minimal 15 dari 20 topik didemonstrasikan dalam batas waktu |
| Kualitas Penjelasan | 25% | Narasi akurat, jelas, tidak membaca kode saja |
| Kualitas Teknis Video | 15% | Resolusi HD, audio jernih, tidak ada lag/freeze |
| Analisis dan Insight | 20% | Memberikan analisis bermakna, bukan hanya menjalankan kode |
| Kreativitas Presentasi | 10% | Penggunaan overlay teks, animasi transisi, atau efek visual |
| **Total** | **100%** | |

### Kriteria Tidak Lulus (video ditolak)
- Durasi kurang dari 8 menit atau lebih dari 17 menit
- Tidak ada narasi audio
- Kurang dari 10 topik yang didemonstrasikan
- Kode yang ditampilkan tidak berjalan (hanya screenshot/mock)
- Resolusi di bawah 720p

---

## Topik yang Harus Tercakup dalam Video

### Wajib (minimal 15 dari 20)
- [ ] 01: Harris Corner Detection - R response + threshold visualization
- [ ] 02: Shi-Tomasi + cornerSubPix - before/after comparison
- [ ] 03: SIFT - rich keypoints visualization
- [ ] 04: ORB - keypoints + descriptor info
- [ ] 05: AKAZE vs FAST - side by side
- [ ] 06: BF Matching - match count before/after ratio test
- [ ] 07: FLANN Matching - speed comparison
- [ ] 08: Homography RANSAC - inlier/outlier visualization
- [ ] 09: Object Detection - bounding polygon on scene
- [ ] 10: Rotation invariance - graph output
- [ ] 11: Scale invariance - graph output
- [ ] 12: Illumination invariance - bar chart
- [ ] 13: Descriptor comparison - benchmark table
- [ ] 14: Geometric verification - statistics
- [ ] 15: Image Retrieval - top-3 results display
- [ ] 16: AR Marker - detection on at least one frame
- [ ] 17: Repeatability - benchmark chart
- [ ] 18: Multi-image matching - similarity heatmap
- [ ] 19: Complete pipeline - all stages shown
- [ ] 20: Feature matching app - interactive demo

---

## Format Pengumpulan

### Berkas yang Dikumpulkan
- **File video** (.mp4, H.264) - ukuran maksimum 500 MB
- **Link YouTube/Google Drive** (alternatif jika file terlalu besar)
- Jika via link: pastikan akses publik atau dibagikan ke dosen

### Penamaan File
```
Video_KV_Modul4_[NIM]_[Nama].mp4
```

### Cara Pengumpulan
Upload ke platform LMS yang ditentukan dosen atau kirimkan link melalui form yang disediakan.

### Batas Waktu
Dikumpulkan maksimal 5 hari setelah pertemuan Modul 4.

---

## Tips Membuat Video yang Baik

1. **Rekam layar dengan OBS Studio** (gratis) untuk kualitas terbaik
2. **Jalankan kode sebelum merekam** untuk memastikan tidak ada error pada saat demo
3. **Buat skrip narasi** terlebih dahulu agar penjelasan lebih terstruktur
4. **Edit video** untuk memotong bagian menunggu/loading yang terlalu panjang
5. **Tambahkan teks overlay** untuk label nama percobaan di setiap transisi
6. **Gunakan kamera terpisah** (atau Picture-in-Picture) untuk menunjukkan wajah Anda
7. **Test audio** sebelum perekaman utama - pastikan tidak ada noise yang mengganggu
