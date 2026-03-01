# PROJECT MODUL 3: PEMROSESAN CITRA

---

## Deskripsi Umum

Project ini mengintegrasikan seluruh teknik pemrosesan citra dari 10 percobaan. Mahasiswa memilih minimal 1 soal cerita dan membangun solusi yang menggunakan minimal 5 konsep percobaan (filtering, thresholding, edge detection, morfologi, Fourier, dll.).

---

## Daftar Improvisasi Percobaan (15 Pengembangan)

1. **Auto Image Enhancer** — Pipeline enhancement otomatis yang menganalisis gambar dan memilih parameter optimal.
2. **Noise Removal Benchmark** — Bandingkan 5+ metode denoising pada berbagai jenis noise (Gaussian, salt-pepper, speckle).
3. **Document Binarizer** — Thresholding adaptif multi-tahap untuk binarisasi dokumen dengan pencahayaan buruk.
4. **Edge-based Object Counter** — Hitung jumlah objek dalam gambar menggunakan edge detection + morfologi + contour.
5. **Frequency Domain Editor** — Remove pola periodik (moiré) dari gambar scan menggunakan notch filter di domain Fourier.
6. **HDR Tone Mapper Sederhana** — Gabungkan beberapa exposure menggunakan Laplacian blending dan gamma mapping.
7. **Skin Detection System** — Deteksi area kulit menggunakan thresholding di HSV space + morfologi.
8. **Adaptive Enhancement per Region** — Bagi gambar ke region, analisis histogram per region, enhance masing-masing.
9. **Pencil Sketch Effect** — Buat efek sketsa pensil dari foto menggunakan edge detection + blending.
10. **Multi-scale Edge Detector** — Deteksi edge pada berbagai skala menggunakan Gaussian pyramid + Canny.
11. **Image Segmentation Sederhana** — Segmentasi objek menggunakan kombinasi thresholding + morfologi + watershed.
12. **Cartoon Effect Generator** — Bilateral filter (smoothing) + edge detection → gabungkan untuk efek kartun.
13. **Text Extraction Preprocessor** — Pipeline preprocessing gambar dokumen untuk meningkatkan akurasi OCR.
14. **Fourier Watermarking** — Sisipkan watermark tak terlihat di domain frekuensi.
15. **Real-time Filter App** — Aplikasi webcam real-time dengan pilihan filter yang bisa di-switch.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Preprocessing Dokumen Arsip
Museum Nasional memiliki koleksi 500+ scan dokumen bersejarah yang pudar dan bernoda. Buatlah pipeline preprocessing yang: (a) analisis histogram untuk menentukan tingkat kerusakan, (b) auto-adjust brightness dan contrast, (c) terapkan CLAHE untuk meningkatkan keterbacaan, (d) binarisasi adaptif untuk memperjelas teks, (e) morfologi untuk menghilangkan noise dan memperjelas huruf, (f) batch processing seluruh folder, (g) simpan dengan format lossless dan buat laporan kualitas before-after.

### Soal 2: Quality Control Pengemasan Produk
Pabrik minuman ingin mendeteksi cacat pada label botol. Buatlah sistem yang: (a) membaca foto botol, (b) crop ROI area label, (c) edge detection untuk mendeteksi batas label, (d) thresholding untuk mendeteksi area yang tidak tercetak, (e) morfologi untuk menghilangkan false positive, (f) hitung persentase area cacat, (g) klasifikasi PASS/FAIL berdasarkan threshold, (h) buat laporan QC dengan gambar beranotasi.

### Soal 3: Aplikasi Filter Foto untuk Media Sosial
Startup lokal ingin membuat alternatif Instagram sederhana. Buatlah aplikasi yang menyediakan: (a) filter brightness/contrast manual, (b) filter sepia/vintage menggunakan LUT, (c) filter sketch menggunakan edge detection + blending, (d) filter cartoon menggunakan bilateral + edge, (e) filter HDR menggunakan CLAHE, (f) filter blur artistik (bokeh simulasi) dengan Gaussian, (g) preview semua filter, (h) export dengan watermark.

### Soal 4: Sistem Penghitung Sel Darah Otomatis
Lab biologi kampus memerlukan tool untuk menghitung jumlah sel darah merah dari foto mikroskop. Buatlah sistem yang: (a) enhance gambar mikroskop (contrast + CLAHE), (b) konversi ke grayscale dan thresholding, (c) operasi morfologi untuk memisahkan sel yang berdekatan, (d) deteksi kontur dan hitung jumlah sel, (e) gambar outline tiap sel yang terdeteksi, (f) hitung statistik (jumlah, rata-rata ukuran, distribusi), (g) buat laporan dengan visualisasi.

### Soal 5: Sistem Deteksi Plat Nomor (Preprocessing)
Sistem parkir memerlukan preprocessing gambar CCTV sebelum OCR. Buatlah pipeline yang: (a) crop ROI area plat, (b) konversi ke grayscale, (c) CLAHE untuk menormalisasi pencahayaan, (d) bilateral filter untuk denoising tanpa menghilangkan edge, (e) edge detection + morfologi untuk menemukan batas karakter, (f) thresholding adaptif untuk binarisasi, (g) resize ke ukuran standar, (h) simpan hasil dengan confidence score per step.

### Soal 6: Tool Restorasi Foto Lama
Studio foto menerima banyak pesanan restorasi foto lama. Buatlah tool yang: (a) analisis kerusakan (noise, fading, spots), (b) denoising adaptif, (c) contrast enhancement dengan CLAHE, (d) removal spot/scratch menggunakan morfologi + inpainting mask, (e) sharpening adaptif, (f) colorization sederhana (apply sepia/tint), (g) before-after comparison.

### Soal 7: Sistem Monitoring Traffic dari CCTV
Dinas Perhubungan ingin mendeteksi kepadatan lalu lintas dari CCTV. Buatlah sistem yang: (a) preprocessing frame video (denoising), (b) edge detection untuk mendeteksi kendaraan, (c) thresholding dan morfologi untuk segmentasi area jalan, (d) hitung piksel area kendaraan vs area kosong, (e) klasifikasi kepadatan (sepi/normal/padat/macet), (f) buat grafik kepadatan sepanjang waktu, (g) tampilkan heatmap kepadatan.

### Soal 8: Sistem Analisis Kualitas Buah
Petani buah ingin grading kualitas buah berdasarkan tampilan visual. Buatlah sistem yang: (a) segmentasi buah dari background (HSV thresholding + morfologi), (b) analisis warna (ekstrak warna dominan), (c) deteksi bintik/cacat (edge detection pada ROI), (d) hitung persentase area cacat, (e) grading (Grade A/B/C), (f) batch processing, (g) buat laporan grading dengan visualisasi.

### Soal 9: Pendeteksi Kebocoran Pipa (Thermal Image Processing)
Perusahaan utilitas menggunakan kamera thermal untuk deteksi kebocoran. Buatlah tool yang: (a) load thermal image (pseudocolor), (b) konversi ke grayscale, (c) histogram analysis untuk identifikasi anomali suhu, (d) thresholding adaptif untuk area panas, (e) Fourier analysis untuk filter noise sensor, (f) morfologi untuk clean-up, (g) highlight dan annotasi area potensial kebocoran.

### Soal 10: Alat Peraga Pembelajaran Image Processing
Dosen mata kuliah PCD memerlukan alat peraga interaktif. Buatlah aplikasi yang: (a) load gambar, (b) panel kontrol untuk setiap operasi (brightness, threshold, filter, edge, morfologi), (c) tampilkan hasil per operasi secara real-time, (d) tampilkan histogram yang update otomatis, (e) kombinasikan operasi secara berurutan (pipeline builder), (f) simpan hasil dan konfigurasi, (g) mode demo otomatis.

---

## Rubrik Penilaian Project

| Komponen | Bobot |
|----------|-------|
| **Fungsionalitas** | 35% |
| **Integrasi Percobaan** | 20% |
| **Kualitas Kode** | 15% |
| **Dokumentasi** | 15% |
| **Kreativitas** | 15% |

*(Detail rubrik sama dengan format standar — lihat Modul 01)*

### Konversi Nilai
| Range | Grade |
|-------|-------|
| 90–100 | A |
| 80–89 | AB |
| 70–79 | B |
| 60–69 | BC |
| 50–59 | C |
| 0–49 | D/E |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project03.zip`
