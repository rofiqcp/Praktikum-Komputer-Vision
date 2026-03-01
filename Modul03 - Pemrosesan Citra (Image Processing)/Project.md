# PROJECT MODUL 3: PEMROSESAN CITRA (IMAGE PROCESSING)

---

## Deskripsi Umum

Project ini mengintegrasikan seluruh teknik pemrosesan citra dari 20 percobaan. Mahasiswa memilih minimal 1 soal cerita dan membangun solusi yang menggunakan minimal **10 dari 20 konsep** percobaan (brightness/contrast, histogram, CLAHE, gamma, thresholding, konvolusi, blur, sharpening, edge detection, morfologi, Fourier, filtering frekuensi, compositing, dll.).

---

## Daftar Improvisasi Percobaan (20 Pengembangan)

1. **Auto Image Enhancer** — Pipeline enhancement otomatis: analisis gambar → pilih parameter optimal.
2. **Noise Removal Benchmark** — Bandingkan 5+ metode denoising pada berbagai jenis noise.
3. **Document Binarizer** — Thresholding adaptif multi-tahap untuk dokumen dengan pencahayaan buruk.
4. **Edge-based Object Counter** — Hitung objek menggunakan edge detection + morfologi + contour.
5. **Frequency Domain Editor** — Remove pola periodik (moiré) menggunakan notch filter Fourier.
6. **HDR Tone Mapper** — Gabungkan beberapa exposure menggunakan Laplacian blending + gamma mapping.
7. **Skin Detection System** — Deteksi area kulit menggunakan HSV thresholding + morfologi.
8. **Adaptive Enhancement per Region** — Analisis histogram per region, enhance masing-masing.
9. **Pencil Sketch Effect** — Efek sketsa pensil: edge detection + inversion + blending.
10. **Multi-scale Edge Detector** — Deteksi edge pada berbagai skala: Gaussian pyramid + Canny.
11. **Image Segmentation Pipeline** — Thresholding + morfologi + watershed segmentasi.
12. **Cartoon Effect Generator** — Bilateral filter + edge detection → gabungkan untuk efek kartun.
13. **Text Extraction Preprocessor** — Pipeline preprocessing dokumen untuk meningkatkan akurasi OCR.
14. **Fourier Watermarking** — Sisipkan watermark tak terlihat di domain frekuensi.
15. **Real-time Filter App** — Webcam real-time dengan pilihan filter yang bisa di-switch.
16. **CLAHE Parameter Optimizer** — Auto-tune clipLimit dan tileGridSize berdasarkan gambar.
17. **Morphological Feature Extractor** — Ekstraksi fitur geometri (area, perimeter, convexity, solidity).
18. **Unsharp Mask HDR** — Pipeline sharpening + CLAHE + gamma untuk hasil HDR.
19. **Frequency Band Analyzer** — Analisis dan visualisasi kontribusi setiap band frekuensi.
20. **Alpha Matte Generator** — Trimap-based alpha matting untuk compositing objek.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Preprocessing Dokumen Arsip
Museum Nasional memiliki koleksi 500+ scan dokumen bersejarah yang pudar dan bernoda. Buatlah pipeline preprocessing yang: (a) analisis histogram untuk menentukan tingkat kerusakan, (b) auto-adjust brightness dan contrast, (c) terapkan CLAHE untuk meningkatkan keterbacaan, (d) binarisasi adaptif untuk memperjelas teks, (e) morfologi untuk menghilangkan noise dan memperjelas huruf, (f) batch processing seluruh folder, (g) simpan dengan format lossless dan buat laporan kualitas before-after.

### Soal 2: Quality Control Pengemasan Produk
Pabrik minuman ingin mendeteksi cacat pada label botol. Buatlah sistem yang: (a) membaca foto botol, (b) crop ROI area label, (c) edge detection untuk mendeteksi batas label, (d) thresholding untuk mendeteksi area yang tidak tercetak, (e) morfologi untuk menghilangkan false positive, (f) hitung persentase area cacat, (g) klasifikasi PASS/FAIL berdasarkan threshold, (h) buat laporan QC dengan gambar beranotasi.

### Soal 3: Aplikasi Filter Foto untuk Media Sosial
Startup lokal ingin membuat alternatif Instagram sederhana. Buatlah aplikasi yang menyediakan: (a) filter brightness/contrast manual, (b) filter sepia/vintage menggunakan LUT, (c) filter sketch menggunakan edge detection + blending, (d) filter cartoon menggunakan bilateral + edge, (e) filter HDR menggunakan CLAHE, (f) filter blur artistik (bokeh simulasi) dengan Gaussian, (g) preview semua filter, (h) export dengan watermark.

### Soal 4: Sistem Penghitung Sel Darah Otomatis
Lab biologi kampus memerlukan tool untuk menghitung sel darah merah dari foto mikroskop. Buatlah sistem yang: (a) enhance gambar mikroskop (contrast + CLAHE), (b) konversi ke grayscale dan thresholding, (c) operasi morfologi untuk memisahkan sel berdekatan, (d) deteksi kontur dan hitung jumlah sel, (e) gambar outline tiap sel, (f) hitung statistik (jumlah, rata-rata ukuran, distribusi), (g) buat laporan dengan visualisasi.

### Soal 5: Sistem Deteksi Plat Nomor (Preprocessing)
Sistem parkir memerlukan preprocessing gambar CCTV sebelum OCR. Buatlah pipeline yang: (a) crop ROI area plat, (b) konversi ke grayscale, (c) CLAHE untuk menormalisasi pencahayaan, (d) bilateral filter untuk denoising tanpa menghilangkan edge, (e) edge detection + morfologi untuk menemukan batas karakter, (f) thresholding adaptif untuk binarisasi, (g) resize ke ukuran standar, (h) simpan hasil per step.

### Soal 6: Tool Restorasi Foto Lama
Studio foto menerima banyak pesanan restorasi foto lama. Buatlah tool yang: (a) analisis kerusakan (noise, fading, spots), (b) denoising adaptif, (c) contrast enhancement dengan CLAHE, (d) removal spot/scratch menggunakan morfologi + inpainting mask, (e) sharpening adaptif, (f) colorization sederhana (apply sepia/tint), (g) before-after comparison.

### Soal 7: Sistem Monitoring Traffic dari CCTV
Dinas Perhubungan ingin mendeteksi kepadatan lalu lintas dari CCTV. Buatlah sistem yang: (a) preprocessing frame video (denoising), (b) edge detection untuk mendeteksi kendaraan, (c) thresholding dan morfologi untuk segmentasi area jalan, (d) hitung piksel area kendaraan vs area kosong, (e) klasifikasi kepadatan (sepi/normal/padat/macet), (f) buat grafik kepadatan, (g) tampilkan heatmap kepadatan.

### Soal 8: Sistem Analisis Kualitas Buah
Petani buah ingin grading kualitas buah berdasarkan tampilan visual. Buatlah sistem yang: (a) segmentasi buah dari background (HSV thresholding + morfologi), (b) analisis warna dominan, (c) deteksi bintik/cacat (edge detection pada ROI), (d) hitung persentase area cacat, (e) grading (Grade A/B/C), (f) batch processing, (g) buat laporan grading.

### Soal 9: Pendeteksi Kebocoran Pipa (Thermal Image Processing)
Perusahaan utilitas menggunakan kamera thermal untuk deteksi kebocoran. Buatlah tool yang: (a) load thermal image (pseudocolor), (b) konversi ke grayscale, (c) histogram analysis untuk identifikasi anomali suhu, (d) thresholding adaptif untuk area panas, (e) Fourier analysis untuk filter noise sensor, (f) morfologi untuk clean-up, (g) highlight dan annotasi area potensial kebocoran.

### Soal 10: Alat Peraga Pembelajaran Image Processing
Dosen mata kuliah PCD memerlukan alat peraga interaktif. Buatlah aplikasi yang: (a) load gambar, (b) panel kontrol untuk setiap operasi (brightness, threshold, filter, edge, morfologi), (c) tampilkan hasil per operasi secara real-time, (d) tampilkan histogram yang update otomatis, (e) kombinasikan operasi secara berurutan (pipeline builder), (f) simpan hasil dan konfigurasi, (g) mode demo otomatis.

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
- **Format**: ZIP — `NIM_Nama_Project03.zip`
- **Isi**: Source code, README.md, screenshot (min. 5), data sample.
