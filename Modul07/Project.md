# PROJECT MODUL 7: DETEKSI FITUR DAN PENCOCOKAN

---

## Deskripsi Umum
Project mengintegrasikan seluruh konsep feature detection & matching: berbagai detektor, deskriptor, matcher, geometric verification, dan aplikasi dunia nyata. Pilih minimal 1 soal cerita.

---

## Daftar Improvisasi Percobaan (15 Pengembangan)

1. **Multi-object Detector** — Deteksi beberapa objek berbeda dalam satu scene menggunakan database template.
2. **Rotation-invariance Benchmark** — Benchmark semua detektor pada gambar yang dirotasi 0–360° (setiap 15°).
3. **Scale-invariance Benchmark** — Benchmark pada gambar yang di-scale 25%–300%.
4. **Illumination Robustness Test** — Uji matching pada gambar dengan brightness/contrast bervariasi.
5. **Real-time Object Tracker** — Track objek template di video/webcam frame-by-frame.
6. **Image Retrieval Engine** — Cari gambar paling mirip dari database 50+ gambar.
7. **Panoramic Image Pair Finder** — Otomatis temukan pasangan gambar yang overlap dari kumpulan foto.
8. **Affine-invariant Matching** — Gunakan affine model (bukan homography) untuk matching permukaan non-planar.
9. **Feature Heatmap Generator** — Visualisasikan densitas fitur sebagai heatmap untuk analisis gambar.
10. **Copy-Move Forgery Detection** — Deteksi area gambar yang di-copy-paste menggunakan self-matching.
11. **AR Card Demo** — Deteksi kartu dan overlay gambar 3D/video di atasnya.
12. **Logo Spotter** — Deteksi logo brand tertentu dari foto produk.
13. **Visual Dictionary Builder** — Buat visual vocabulary dari dataset dan analisis cluster distribution.
14. **Keypoint Stability Analysis** — Analisis repeatability keypoint di berbagai kondisi (noise, blur, compression).
15. **Deep Feature Comparison** — Bandingkan SIFT/ORB dengan deep features (SuperPoint) jika tersedia.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Pengenalan Sampul Buku untuk Perpustakaan
Perpustakaan digital ingin mengidentifikasi buku dari foto sampul. Buatlah: (a) database fitur dari 10 buku, (b) deteksi + match menggunakan SIFT/ORB, (c) verifikasi RANSAC, (d) tampilkan info buku jika terdeteksi, (e) ranking kemiripan jika ada beberapa kandidat, (f) handle buku yang tidak ada di database, (g) real-time dari webcam.

### Soal 2: Pendeteksi Uang Kertas
Bank memerlukan sistem verifikasi uang kertas dari foto. Buatlah: (a) database fitur 5 denominasi uang, (b) deteksi denominasi dari foto, (c) handle uang dari sudut/rotasi berbeda, (d) scoring kemiripan, (e) deteksi uang palsu (fitur kurang match), (f) batch processing banyak foto, (g) statistik akurasi per denominasi.

### Soal 3: Sistem Augmented Reality Marker
Startup AR ingin tool untuk mendeteksi marker dan overlay konten. Buatlah: (a) desain 5 marker berbeda (gambar/pola), (b) deteksi marker real-time dari webcam, (c) estimasi homography, (d) warp gambar/video overlay ke area marker, (e) support multiple marker simultaneous, (f) smooth overlay (filter jitter), (g) FPS display.

### Soal 4: Sistem Verifikasi Tanda Tangan
Bank memerlukan verifikasi tanda tangan otomatis. Buatlah: (a) kumpulkan 5 tanda tangan per 3 orang, (b) preprocessing (binarize, crop), (c) feature extraction + matching, (d) scoring kemiripan, (e) threshold accept/reject, (f) handle tanda tangan palsu, (g) confusion matrix per orang, (h) ROC curve.

### Soal 5: Image Forgery Detector
Media memerlukan tool untuk mendeteksi manipulasi gambar. Buatlah: (a) copy-move detection (self-matching SIFT), (b) visualisasi area yang di-clone, (c) splicing detection (analisis konsistensi fitur), (d) scoring forgery level (0–100), (e) uji pada 5 gambar asli + 5 gambar manipulasi, (f) hitung precision/recall, (g) report per gambar.

### Soal 6: Pemandu Wisata Visual
Aplikasi wisata ingin mengenali landmark dari foto. Buatlah: (a) database 10 landmark/tempat wisata, (b) feature matching dari foto user, (c) ranking kemiripan, (d) informasi tempat wisata, (e) handle foto dari sudut berbeda, (f) peta lokasi, (g) works offline (fitur pre-computed).

### Soal 7: Quality Control — Part Matching di Pabrik
Pabrik ingin memverifikasi bahwa part yang diproduksi sesuai dengan template. Buatlah: (a) database template 5 part, (b) foto part dari conveyor, (c) matching + homography, (d) overlay alignment untuk visual comparison, (e) scoring similarity, (f) keputusan pass/fail, (g) logging dan statistik.

### Soal 8: Pendeteksi Duplikat Foto
Platform foto stok ingin mendeteksi duplikat atau near-duplikat. Buatlah: (a) database 50 gambar, (b) extract + index fitur semua gambar, (c) query: cari gambar mirip, (d) handle transformasi (crop, resize, filter), (e) ranking kemiripan, (f) threshold duplikat, (g) batch comparison matrix.

### Soal 9: Sistem Navigasi Visual untuk Robot
Robot perlu mengenali lokasi dari gambar lingkungan. Buatlah: (a) database gambar 10 lokasi, (b) capture gambar dari "robot" (webcam), (c) match ke database, (d) tentukan lokasi terdekat, (e) track perpindahan antar lokasi, (f) visualisasi peta + posisi, (g) akurasi lokalisasi.

### Soal 10: Card Game Scanner
Kolektor kartu (Pokemon, MTG) ingin mengidentifikasi kartu dari foto. Buatlah: (a) database 10 kartu, (b) deteksi + identifikasi dari foto, (c) handle kartu miring/rotasi, (d) multiple kartu dalam satu foto, (e) informasi nama + harga, (f) real-time dari webcam, (g) statistik koleksi.

---

## Rubrik Penilaian Project

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Fungsionalitas | 35% | Semua fitur berjalan sesuai deskripsi |
| Integrasi Percobaan | 20% | Menggunakan konsep dari ≥5 percobaan |
| Kualitas Kode | 15% | Clean code, modular, well-commented |
| Dokumentasi | 15% | README, instruksi, screenshot |
| Kreativitas | 15% | Fitur tambahan, optimasi, UI |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project07.zip`
- **Isi**: Source code, database gambar, README.md.
