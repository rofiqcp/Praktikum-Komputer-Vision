# PROJECT MODUL 6: RECOGNITION (PENGENALAN)

---

## Deskripsi Umum
Project mengintegrasikan seluruh konsep recognition yang dipelajari dalam 20 percobaan: face detection, face recognition, face landmarks, OCR, scene text detection, pedestrian/vehicle detection, hand gesture recognition, object classification (BoVW), scene recognition, face embedding, tracking, evaluation metrics, dan pipeline integration. Pilih minimal 1 soal cerita dan kembangkan menjadi aplikasi lengkap.

---

## Daftar Improvisasi Percobaan (15 Pengembangan)

1. **Multi-face Attendance System** — Deteksi + recognition banyak wajah sekaligus dalam satu frame.
2. **Emotion-aware Recognition** — Kombinasikan face recognition + analisis emosi real-time.
3. **Age & Gender Profiling** — Buat profiler demografis dari feed kamera.
4. **Anti-spoofing Face Detection** — Deteksi apakah wajah dari layar HP/foto vs wajah asli.
5. **Multi-language OCR Pipeline** — OCR yang mendukung 3+ bahasa dengan auto-detection bahasa.
6. **Receipt/Invoice Parser** — OCR khusus untuk membaca struk belanja terstruktur.
7. **License Plate Recognition** — Deteksi plat + OCR nomor kendaraan.
8. **Sign Language Translator** — Pengenalan huruf/kata bahasa isyarat dari 21 landmarks.
9. **Gesture-controlled Presentation** — Kontrol slide PowerPoint dengan gestur tangan.
10. **Product Recognition** — Klasifikasi produk retail dari foto (BoVW + DL hybrid).
11. **Document Scanner + OCR** — Perspective correction → binarize → OCR → export PDF.
12. **Face Clustering** — Otomatis kelompokkan foto berdasarkan wajah (tanpa label).
13. **Pedestrian Counter** — Hitung jumlah pejalan kaki yang melewati area tertentu.
14. **Custom Object Classifier** — Train classifier untuk domain spesifik dengan Flask web UI.
15. **Interactive Recognition Dashboard** — OpenCV GUI untuk switch antara face/text/pedestrian detection.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Absensi Wajah untuk Ruang Kelas
Sebuah kampus ingin mengganti absensi manual dengan sistem pengenalan wajah otomatis. CCTV kelas menangkap gambar setiap 5 menit. Buatlah: (a) face detection pada gambar kelas (DNN), (b) face recognition untuk identifikasi mahasiswa terdaftar (min 5 orang), (c) database wajah mahasiswa, (d) logging kehadiran per waktu, (e) deteksi wajah tidak dikenali (unknown), (f) laporan kehadiran harian, (g) real-time demo dari webcam.

### Soal 2: Sistem Pembaca KTP Otomatis
Kantor kelurahan memerlukan sistem untuk mengekstrak informasi dari foto KTP secara otomatis. Buatlah: (a) deteksi area KTP dalam foto (perspective correction jika miring), (b) deteksi area teks (EAST atau CRAFT), (c) OCR nama, NIK, tempat/tanggal lahir, alamat, (d) parsing terstruktur ke format JSON, (e) validasi format NIK (16 digit), (f) confidence score per field, (g) batch processing untuk banyak KTP.

### Soal 3: Sistem Keamanan Smart Home
Pemilik rumah ingin CCTV yang bisa mengenali penghuni vs orang asing. Buatlah: (a) face detection real-time, (b) face recognition penghuni rumah (min 3 orang), (c) alert jika wajah tidak dikenali, (d) pedestrian detection di halaman, (e) logging event dengan screenshot, (f) analisis emosi pengunjung, (g) dashboard log kejadian.

### Soal 4: Aplikasi Penerjemah Teks Real-time
Turis asing memerlukan aplikasi untuk menerjemahkan teks pada papan nama/menu restoran. Buatlah: (a) text detection pada gambar (EAST), (b) OCR teks terdeteksi, (c) deteksi bahasa, (d) terjemahan otomatis (bisa pakai library translate), (e) overlay terjemahan pada gambar asli, (f) support gambar statis + webcam, (g) history terjemahan.

### Soal 5: Sistem Counting Pengunjung Mall
Pengelola mall ingin menghitung pengunjung secara otomatis. Buatlah: (a) pedestrian detection dari CCTV (HOG + YOLO), (b) counting masuk/keluar berdasarkan crossing line, (c) tracking sederhana antar frame, (d) statistik per jam, (e) heatmap area ramai, (f) grafik real-time, (g) export laporan harian.

### Soal 6: Sistem Pengenalan Isyarat untuk Difabel
Organisasi disabilitas memerlukan alat bantu komunikasi. Buatlah: (a) hand detection (MediaPipe), (b) finger landmark extraction, (c) klasifikasi 10+ isyarat (angka 0-9 atau huruf A-J), (d) display isyarat yang dikenali, (e) mode "kalimat" (gabungkan huruf), (f) real-time dari webcam, (g) training mode untuk menambah gesture baru.

### Soal 7: Smart Parking Gate
Pengelola parkir ingin gerbang otomatis berdasarkan plat nomor. Buatlah: (a) deteksi area plat nomor, (b) OCR plat nomor, (c) database kendaraan terdaftar, (d) gate open/close logic, (e) logging waktu masuk/keluar, (f) hitung durasi parkir + biaya, (g) handle plat tidak terbaca (alert manual).

### Soal 8: Museum Interactive Guide
Museum ingin aplikasi yang mengenali artefak dari foto pengunjung. Buatlah: (a) dataset gambar 10 artefak museum, (b) klasifikasi artefak (transfer learning), (c) BoVW sebagai baseline, (d) tampilkan informasi artefak setelah dikenali, (e) similarity search (artefak serupa), (f) logging artefak populer, (g) interface upload foto.

### Soal 9: Sistem Monitoring Gudang
Perusahaan logistik ingin memantau aktivitas di gudang. Buatlah: (a) pedestrian detection pekerja, (b) counting orang di area tertentu, (c) deteksi helm safety (YOLO custom atau classification), (d) alert jika tidak pakai helm, (e) face recognition pekerja terdaftar, (f) logging aktivitas per shift, (g) dashboard ringkasan.

### Soal 10: Gesture-based Game Controller
Developer game memerlukan kontrol berbasis gestur. Buatlah: (a) hand tracking real-time (MediaPipe), (b) klasifikasi 5+ gestur (fist, open, point, peace, thumb up), (c) mapping gestur ke aksi game (move, jump, shoot, pause, restart), (d) implementasi game sederhana (atau kontrol game existing), (e) calibrasi per pengguna, (f) FPS counter + latency display, (g) multiplayer 2 tangan.

---

## Rubrik Penilaian Project

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Fungsionalitas | 35% | Semua fitur berjalan sesuai deskripsi |
| Integrasi Percobaan | 20% | Menggunakan konsep dari ≥10 percobaan |
| Kualitas Kode | 15% | Clean code, modular, well-commented |
| Dokumentasi | 15% | README, instruksi penggunaan, screenshot |
| Kreativitas | 15% | Fitur tambahan, UI/UX, solusi inovatif |

### Penilaian Detail

| Nilai | Kriteria |
|-------|----------|
| A (85-100) | Semua fitur + real-time + fitur bonus |
| B (70-84) | Semua fitur dasar + dokumentasi lengkap |
| C (55-69) | Fitur utama berjalan, beberapa fitur minor belum |
| D (40-54) | Hanya sebagian fitur, banyak bug |
| E (<40) | Tidak dapat dijalankan |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project06.zip`
- **Isi**: Source code, dataset (atau instruksi download), model weights, README.md.
