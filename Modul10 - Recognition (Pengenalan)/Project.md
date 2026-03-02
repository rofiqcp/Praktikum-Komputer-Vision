# PROJECT MODUL 10: RECOGNITION (PENGENALAN)

---

## Deskripsi Umum
Project mengintegrasikan konsep recognition dari 20 percobaan: face detection (Haar, DNN), face recognition (LBPH, Eigenfaces, embedding), face landmarks, OCR (preprocessing, Tesseract, scene text), pedestrian detection (HOG), vehicle detection, gesture recognition, object classification (BoVW), scene recognition, metrik evaluasi (classification, detection, ROC), face tracking, dan pipeline lengkap. Pilih minimal 1 soal cerita.

---

## Daftar Improvisasi Percobaan (15 Pengembangan)

1. **Sistem Absensi Wajah** — Face recognition + database + laporan kehadiran otomatis.
2. **Real-time Emotion Detector** — Deteksi ekspresi wajah (senang, sedih, marah, netral).
3. **Age & Gender Estimator** — Estimasi umur dan gender via DNN model.
4. **ANPR (Automatic Number Plate Recognition)** — OCR pada plat nomor kendaraan.
5. **Multi-face Video Tracker** — Tracking + recognition pada video panjang.
6. **QR Code & Barcode Scanner** — `cv2.QRCodeDetector()` + decode data.
7. **Sign Language Recognizer** — Pengenalan bahasa isyarat dari gesture tangan.
8. **Document Scanner** — Deteksi dokumen + perspective correction + OCR.
9. **Face Verification System** — 1:1 verification (apakah ini orang yang sama?).
10. **Scene Classifier Real-time** — Klasifikasi scene dari webcam (indoor/outdoor/nature/urban).
11. **Pedestrian Counter** — Hitung jumlah pejalan kaki dari video CCTV.
12. **Face Similarity Search** — Input wajah → cari yang paling mirip dari database.
13. **Handwriting Recognition** — OCR untuk tulisan tangan.
14. **Multi-object Detector** — Gabungkan deteksi wajah + pedestrian + kendaraan.
15. **Recognition Dashboard** — Visualisasi real-time: detection count, recognition accuracy, confidence histogram.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Keamanan Gedung Pintar
Perusahaan properti mengembangkan gedung pintar yang memerlukan sistem pengenalan wajah untuk akses masuk. Buatlah: (a) database wajah karyawan (min 5 orang), (b) face detection + recognition pipeline, (c) anti-spoofing sederhana (blink detection), (d) logging akses (waktu, nama, confidence), (e) alert untuk wajah unknown, (f) dashboard statistik akses harian, (g) export laporan.

### Soal 2: Sistem Parkir Otomatis
Perusahaan parkir memerlukan sistem ANPR otomatis untuk mencatat kendaraan masuk/keluar. Buatlah: (a) deteksi area plat nomor pada gambar kendaraan, (b) preprocessing plat (binarisasi, denoise), (c) OCR karakter plat, (d) database kendaraan terdaftar, (e) verifikasi plat vs database, (f) pencatatan waktu masuk/keluar, (g) statistik kendaraan per hari.

### Soal 3: Asisten Toko Retail
Toko retail ingin menganalisis traffic pelanggan. Buatlah: (a) pedestrian detection pada video CCTV, (b) counting pengunjung masuk/keluar, (c) face detection untuk analisis demografis (estimasi umur/gender), (d) tracking path pengunjung, (e) heatmap area populer, (f) statistik per jam, (g) export laporan harian.

### Soal 4: Sistem Ujian Anti-Cheating
Kampus memerlukan proctoring untuk ujian online. Buatlah: (a) face detection + verification (apakah ini peserta yang terdaftar), (b) head pose estimation (apakah melihat layar), (c) deteksi multiple faces (ada orang lain?), (d) deteksi objek terlarang (handphone via HOG/cascade), (e) logging pelanggaran, (f) screenshot saat pelanggaran, (g) report per peserta.

### Soal 5: Sistem Pengelola Perpustakaan Digital
Perpustakaan ingin mendigitalisasi buku lama. Buatlah: (a) deteksi halaman buku dari foto, (b) perspective correction, (c) preprocessing OCR, (d) OCR teks dari halaman, (e) deteksi dan skip gambar/tabel, (f) compile teks per halaman, (g) export ke format teks.

### Soal 6: Sistem Monitoring Lalu Lintas
Dinas perhubungan memerlukan monitoring jalan raya. Buatlah: (a) vehicle detection dari video CCTV, (b) pedestrian detection, (c) counting kendaraan per jalur, (d) estimasi kecepatan (frame-based), (e) deteksi pelanggaran (pejalan kaki di jalan), (f) statistik traffic per jam, (g) alert kecelakaan.

### Soal 7: Sistem Pengenalan Produk
Supermarket ingin self-checkout dengan pengenalan produk dari kamera. Buatlah: (a) klasifikasi produk (min 5 kategori) via BoVW/CNN, (b) deteksi barcode/QR, (c) database produk + harga, (d) matching produk → harga, (e) keranjang belanja otomatis, (f) total harga, (g) struk digital.

### Soal 8: Sistem Monitoring Kesehatan Wajah
Klinik kecantikan ingin analisis wajah otomatis. Buatlah: (a) face detection + landmark 68 titik, (b) analisis simetri wajah, (c) deteksi area bermasalah (simulasi), (d) face embedding untuk tracking perubahan before/after, (e) perbandingan jarak embedding, (f) laporan analisis, (g) visualisasi overlay.

### Soal 9: Sistem Presensi Rapat
Perusahaan ingin absensi rapat otomatis dari foto grup. Buatlah: (a) multi-face detection pada foto grup, (b) face recognition per wajah vs database karyawan, (c) attendance list otomatis, (d) handling unknown faces, (e) statistik kehadiran per karyawan, (f) export laporan rapat, (g) notifikasi yang tidak hadir.

### Soal 10: Gesture-Controlled Presentation
Dosen ingin mengontrol presentasi dengan gesture tangan. Buatlah: (a) hand gesture recognition real-time, (b) mapping gesture → aksi (swipe=next, fist=prev, open=pointer), (c) visual feedback gesture yang terdeteksi, (d) counting gesture, (e) gesture history log, (f) accuracy tracker, (g) demo dengan slide presentasi.

---

## Rubrik Penilaian Project

| Komponen | Bobot |
|----------|-------|
| Fungsionalitas | 35% |
| Integrasi Percobaan | 20% |
| Kualitas Kode | 15% |
| Dokumentasi | 15% |
| Kreativitas | 15% |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project10.zip`
- **Catatan**: Sertakan model/database yang diperlukan.
