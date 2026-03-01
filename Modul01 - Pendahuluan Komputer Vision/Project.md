# PROJECT MODUL 1: PENDAHULUAN KOMPUTER VISION

---

## Deskripsi Umum

Project ini merupakan pengembangan dan improvisasi dari 20 percobaan praktikum Modul 1. Mahasiswa diminta memilih **minimal 1 soal cerita** dari 10 pilihan di bawah, kemudian mengimplementasikan solusi menggunakan seluruh konsep yang telah dipelajari. Setiap project harus mengintegrasikan minimal 10 dari 20 percobaan praktikum.

---

## Daftar Improvisasi Percobaan (20 Pengembangan)

Berikut adalah pengembangan lanjutan dari setiap percobaan dasar:

1. **Batch Image Loader** — Buat sistem yang membaca semua gambar dari sebuah folder secara otomatis, mendeteksi format, dan menampilkan thumbnail grid.
2. **Image Metadata Analyzer** — Buat tool yang mengekstrak dan membandingkan properti (resolusi, channel, warna dominan) dari kumpulan gambar kemudian menghasilkan report CSV.
3. **Real-time Color Space Explorer** — Buat aplikasi dengan trackbar yang menampilkan konversi ruang warna secara real-time, termasuk channel-wise visualization.
4. **Pixel Art Generator** — Buat generator pixel art dari foto: downscale drastis kemudian upscale dengan nearest neighbor, terapkan palette warna terbatas.
5. **Image Calculator** — Buat kalkulator gambar yang mendemonstrasikan semua operasi aritmatika dengan GUI trackbar untuk mengatur parameter secara interaktif.
6. **Logo Watermark Tool** — Buat tool untuk menambahkan logo watermark menggunakan operasi bitwise dengan transparansi yang bisa diatur.
7. **Interactive Drawing Tool** — Buat tool menggambar interaktif menggunakan mouse callback OpenCV untuk menggambar bentuk geometris secara real-time.
8. **Annotation Tool Sederhana** — Buat tool anotasi yang memungkinkan user menambahkan teks, label, dan bounding box via mouse callback dan keyboard input.
9. **Smart ROI Extractor** — Buat tool yang mendeteksi area menarik dari gambar (berdasarkan variance piksel) dan otomatis crop region tersebut.
10. **Batch Image Resizer & Converter** — Buat tool command-line untuk resize batch gambar ke berbagai ukuran dan konversi antar format sekaligus.
11. **Auto Crop Tool** — Buat tool yang secara otomatis memotong area putih/kosong di sekitar objek dalam gambar.
12. **Panorama Rotator** — Buat viewer gambar yang menampilkan rotasi bertahap 360° dengan transisi smooth.
13. **Augmentasi Data Sederhana** — Buat pipeline augmentasi data yang menerapkan flip, rotasi, dan modifikasi acak untuk training dataset.
14. **Photo Frame Generator** — Buat generator frame foto otomatis dengan berbagai tipe border dan dekorasi.
15. **Channel Mixer** — Buat tool interaktif untuk mencampur dan menukar channel warna dengan slider untuk setiap channel.
16. **Image Blending Transition** — Buat video transisi smooth antar dua gambar menggunakan alpha blending bertahap (fade-in/fade-out), simpan sebagai video.
17. **Auto Brightness/Contrast** — Buat tool yang secara otomatis mengoreksi gambar under/overexposed berdasarkan analisis histogram.
18. **Histogram Matcher** — Ambil histogram dari satu gambar dan terapkan ke gambar lain (histogram specification/matching).
19. **Color Filter Simulator** — Simulasikan efek filter warna (sepia, cool, warm, vintage) menggunakan operasi masking dan manipulasi channel.
20. **Image Quality Comparator** — Buat tool yang menyimpan satu gambar di berbagai kualitas JPEG (10–100), menghitung PSNR dan SSIM, dan membuat grafik perbandingan otomatis.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Katalog Produk UMKM
Ibu Sari memiliki toko kue tradisional dan ingin membuat katalog digital produknya untuk dijual online. Dia memiliki 30 foto kue yang diambil dari berbagai sudut dengan pencahayaan tidak konsisten. Buatlah sistem yang dapat: (a) membaca semua foto dari folder, (b) menstandarkan ukuran ke 800×600 piksel, (c) menambahkan watermark nama toko "Kue Tradisional Bu Sari", (d) menambahkan border dekoratif, (e) mengekstrak warna dominan setiap kue, (f) menyimpan dalam format JPEG kualitas 85% dengan nama terorganisir, dan (g) membuat halaman katalog collage 3×3.

### Soal 2: Sistem Presensi Kartu Mahasiswa
Kampus X ingin membuat prototype sistem presensi berbasis foto kartu mahasiswa. Setiap mahasiswa menunjukkan kartu ke kamera. Buatlah sistem yang dapat: (a) menangkap gambar dari webcam atau membaca dari file, (b) mendeteksi dan crop area kartu (ROI), (c) resize ke ukuran standar 400×250, (d) mengkonversi ke grayscale untuk penyimpanan efisien, (e) menambahkan timestamp dan nomor urut, (f) menyimpan dengan format penamaan `YYYYMMDD_HHMMSS_kartu.jpg`, dan (g) membuat collage harian 4×4 sebagai ringkasan presensi.

### Soal 3: Alat Bantu Analisis Citra Medis Sederhana
Sebuah klinik kecil membutuhkan tool sederhana untuk mempersiapkan citra X-ray sebelum dikirim ke dokter spesialis. Buatlah tool yang dapat: (a) membaca citra X-ray (grayscale), (b) menampilkan histogram distribusi intensitas, (c) melakukan adjustment brightness dan contrast melalui slider interaktif, (d) menerapkan color mapping (pseudocolor) untuk highlight area tertentu, (e) menambahkan anotasi (panah, teks) untuk menandai area mencurigakan, (f) mengekspor dalam format PNG lossless, dan (g) membuat laporan komparasi gambar before-after.

### Soal 4: Photo Booth Festival Kampus
Panitia festival kampus ingin membuat photo booth digital sederhana. Buatlah aplikasi yang dapat: (a) menangkap foto dari webcam, (b) menerapkan filter warna (sepia, cool, warm, grayscale), (c) menambahkan frame dekoratif bertema festival, (d) menambahkan teks "Festival Kampus 2026" dengan font menarik, (e) membuat strip photo booth (4 foto vertikal), (f) menyimpan dengan nama unik berdasarkan timestamp, dan (g) membuat montage semua pengunjung sebagai kenangan.

### Soal 5: Quality Control Produk Tekstil
Sebuah pabrik tekstil ingin melakukan inspeksi visual awal pada kain sebelum dikirim ke pelanggan. Buatlah sistem yang dapat: (a) membaca foto kain dari kamera, (b) mengekstrak ROI area yang dicurigai cacat, (c) menganalisis distribusi warna (histogram) untuk mendeteksi inkonsistensi, (d) membandingkan warna kain aktual dengan warna referensi menggunakan ruang warna Lab, (e) menandai area dengan perbedaan warna signifikan, (f) menghitung persentase area cacat, dan (g) menghasilkan laporan quality control dengan gambar beranotasi.

### Soal 6: Sistem Arsip Dokumen Digital
Sebuah kantor kelurahan ingin mendigitalkan arsip surat-surat lama. Buatlah sistem yang dapat: (a) membaca scan dokumen, (b) melakukan auto-crop untuk membuang area putih berlebih, (c) mengkonversi ke grayscale, (d) melakukan penyesuaian contrast agar teks lebih terbaca, (e) meresize ke ukuran standar A4 (2480×3508 piksel pada 300 DPI), (f) menambahkan header "ARSIP KELURAHAN" dan nomor halaman, dan (g) menyimpan dalam format PDF-ready (PNG lossless).

### Soal 7: Aplikasi Instagram Filter Sederhana
Seorang mahasiswa ingin membuat aplikasi filter foto sederhana ala Instagram. Buatlah aplikasi dengan: (a) load foto dari file, (b) terapkan minimal 6 filter (grayscale, sepia, negative, brightness+, vignette, blur artistik), (c) tampilkan preview semua filter dalam grid 2×3, (d) user memilih filter via input, (e) atur intensitas filter via trackbar, (f) tambahkan border dan watermark, dan (g) simpan hasil final dalam kualitas optimal.

### Soal 8: Sistem Monitoring Tanaman Sederhana
Petani Pak Budi ingin memantau pertumbuhan tanamannya menggunakan foto harian. Buatlah sistem yang dapat: (a) membaca foto tanaman harian, (b) mengekstrak area tanaman menggunakan ROI, (c) menganalisis persentase warna hijau menggunakan HSV masking, (d) membuat grafik perubahan warna hijau dari hari ke hari, (e) membuat perbandingan side-by-side foto antar hari, (f) menandai area tanaman yang menguning (potensi penyakit), dan (g) menghasilkan laporan mingguan berupa collage progress.

### Soal 9: Tool Perbandingan Resolusi untuk E-Commerce
Sebuah marketplace ingin menentukan standar kualitas foto produk. Buatlah tool analisis yang dapat: (a) membaca foto produk asli, (b) membuat versi dengan berbagai resolusi (thumbnail 150×150, medium 600×600, HD 1200×1200), (c) membandingkan kualitas visual tiap resolusi secara side-by-side, (d) menghitung PSNR dan estimasi SSIM antara setiap versi dan asli, (e) membuat grafik ukuran file vs kualitas, (f) membuat perbandingan metode interpolasi, dan (g) merekomendasikan standar resolusi optimal.

### Soal 10: Sistem Pembuatan ID Card Otomatis
Organisasi mahasiswa perlu membuat ID card untuk 100+ anggota. Buatlah sistem yang dapat: (a) membaca foto anggota dari folder, (b) auto-crop area wajah menggunakan ROI (manual atau Haar cascade sederhana), (c) resize foto ke ukuran standar pas foto, (d) menerapkan background putih/biru, (e) membuat template ID card dengan logo, nama, NIM, dan jabatan (dari file CSV), (f) menyimpan setiap ID card sebagai file terpisah, dan (g) membuat lembar cetak berisi 8 ID card per halaman A4.

---

## Rubrik Penilaian Project

### Komponen Penilaian

| Komponen | Bobot | Deskripsi |
|----------|-------|-----------|
| **Fungsionalitas** | 35% | Semua fitur yang diminta berjalan dengan benar |
| **Integrasi Percobaan** | 20% | Minimal 10 dari 20 konsep percobaan terintegrasi dalam project |
| **Kualitas Kode** | 15% | Terstruktur, modular, ada komentar, penamaan variabel deskriptif |
| **Dokumentasi** | 15% | README lengkap, cara instalasi, screenshot, penjelasan alur |
| **Kreativitas & Fitur Tambahan** | 15% | Improvisasi di luar requirement minimum |

### Detail Rubrik

#### Fungsionalitas (35%)
| Skor | Kriteria |
|------|----------|
| 90–100 | Semua fitur berjalan sempurna, ada error handling, edge case tertangani |
| 75–89 | Semua fitur utama berjalan, minor bug tidak kritis |
| 60–74 | Sebagian besar fitur berjalan, 1–2 fitur bermasalah |
| 40–59 | Hanya setengah fitur yang berjalan |
| 0–39 | Mayoritas fitur tidak berjalan |

#### Integrasi Percobaan (20%)
| Skor | Kriteria |
|------|----------|
| 90–100 | 16–20 konsep percobaan terintegrasi secara koheren |
| 75–89 | 12–15 konsep terintegrasi |
| 60–74 | 10 konsep terintegrasi (minimum requirement) |
| 40–59 | 6–9 konsep terintegrasi |
| 0–39 | Kurang dari 6 konsep |

#### Kualitas Kode (15%)
| Skor | Kriteria |
|------|----------|
| 90–100 | Modular (fungsi/class), PEP8, docstring, type hints, no hardcode |
| 75–89 | Terstruktur baik, ada komentar, penamaan jelas |
| 60–74 | Kode berjalan, ada komentar minimal |
| 40–59 | Kode berantakan tapi berjalan |
| 0–39 | Kode tidak berjalan atau plagiat |

#### Dokumentasi (15%)
| Skor | Kriteria |
|------|----------|
| 90–100 | README lengkap, screenshot, diagram alur, video demo |
| 75–89 | README baik, ada screenshot |
| 60–74 | README ada tapi kurang detail |
| 40–59 | Dokumentasi minimal |
| 0–39 | Tidak ada dokumentasi |

#### Kreativitas (15%)
| Skor | Kriteria |
|------|----------|
| 90–100 | Fitur tambahan inovatif, UI/UX baik, solusi elegan |
| 75–89 | Ada 2–3 fitur tambahan yang berguna |
| 60–74 | Ada 1 fitur tambahan |
| 40–59 | Hanya mengerjakan minimum requirement |
| 0–39 | Tidak lengkap |

### Konversi Nilai
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
| Tidak dapat dijalankan | Maksimal 50% |

---

## Format Pengumpulan

- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project01.zip`
- **Isi**: Source code, README.md, screenshot (min. 5), data sample.
