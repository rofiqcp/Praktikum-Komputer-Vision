# PROJECT MODUL 2: PEMBENTUKAN CITRA (IMAGE FORMATION)

---

## Deskripsi Umum

Project ini mengintegrasikan seluruh konsep pembentukan citra dari 20 percobaan: transformasi geometri, kalibrasi kamera, proyeksi, koreksi distorsi, fotometri, interpolasi, image pyramid, remapping, dan pembuatan citra sintetis. Mahasiswa memilih minimal 1 soal cerita dan mengimplementasikan solusi yang menggunakan minimal **10 dari 20 konsep** percobaan.

---

## Daftar Improvisasi Percobaan (20 Pengembangan)

1. **Panoramic Image Warper** — Warp beberapa gambar menggunakan homografi lalu gabungkan.
2. **Automated Document Scanner** — Deteksi kontur dokumen, koreksi perspektif otomatis, enhance contrast.
3. **Real-time Camera Calibrator** — Kalibrasi kamera secara interaktif dengan feedback real-time.
4. **Virtual Billboard Replacement** — Ganti isi billboard di foto dengan gambar lain menggunakan homografi.
5. **Augmented Reality Cube** — Overlay kubus 3D virtual di atas marker checkerboard.
6. **Lens Distortion Simulator** — Simulasi berbagai jenis distorsi lensa dan koreksinya.
7. **Bird's Eye View Generator** — Konversi gambar jalan menjadi tampilan atas menggunakan inverse perspective.
8. **Image Morphing Tool** — Morph antara dua gambar menggunakan transformasi affine per triangulasi.
9. **Camera Pose Estimator** — Estimasi posisi dan orientasi kamera dari marker yang diketahui.
10. **Anti-aliasing Quality Analyzer** — Tool untuk menganalisis dan membandingkan metode anti-aliasing.
11. **Gamma Correction Auto-Tuner** — Auto-adjust gamma berdasarkan histogram untuk optimasi visual.
12. **Photo Rectifier** — Koreksi foto bangunan yang miring (keystone correction).
13. **Multi-scale Image Analyzer** — Image pyramid untuk deteksi fitur di berbagai skala.
14. **Polar Coordinate Image Unwrapper** — Unwrap objek lingkaran (iris, jam, radar) ke koordinat Cartesian.
15. **Custom Remap Effect Gallery** — Koleksi efek distorsi artistik menggunakan cv2.remap.
16. **Image Shearing Animation** — Animasi shearing interaktif dengan slider real-time.
17. **Synthetic Test Pattern Generator** — Generator pola uji kalibrasi kustom (grid, Siemens star, dll.).
18. **Log/Power Transform HDR Viewer** — Visualisasi HDR image dengan transformasi log/power adaptif.
19. **Multi-view Image Aligner** — Align gambar dari banyak sudut pandang ke satu referensi.
20. **Scale-Invariant Template Matcher** — Match template pada berbagai skala menggunakan piramida.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Document Scanner Kantor
Sebuah kantor notaris perlu mendigitalkan ratusan dokumen legal. Dokumen difoto menggunakan smartphone dari sudut yang bervariasi. Buatlah sistem yang: (a) mendeteksi kontur dokumen otomatis, (b) menerapkan koreksi perspektif agar dokumen tegak, (c) mengoreksi distorsi lensa (jika kamera sudah dikalibrasi), (d) memperbaiki pencahayaan dengan gamma correction, (e) menyimpan hasil sebagai gambar standar ukuran A4, (f) melakukan batch processing untuk seluruh folder, (g) membuat thumbnail preview.

### Soal 2: Augmented Reality Sederhana
Tim riset kampus ingin membuat demo AR untuk pameran teknologi. Buatlah sistem AR sederhana yang: (a) mengkalibrasi kamera, (b) mendeteksi marker checkerboard secara real-time, (c) memproyeksikan kubus 3D wireframe di atas marker, (d) menambahkan sumbu koordinat XYZ berwarna, (e) merender objek-objek sederhana (piramida, bola), (f) mendukung rotasi dan translasi sesuai pose marker, (g) menambahkan teks label pada objek virtual.

### Soal 3: Sistem Koreksi Foto Produk E-Commerce
Marketplace online memerlukan tool standarisasi foto produk. Buatlah tool yang: (a) mendeteksi dan memperbaiki kemiringan foto produk, (b) mengoreksi distorsi lensa wide-angle, (c) menormalkan pencahayaan dengan gamma auto-adjust, (d) menerapkan white balance otomatis, (e) resize ke beberapa format standar marketplace, (f) crop area produk dengan transformasi perspektif, (g) generate thumbnail anti-aliased.

### Soal 4: Virtual Tour Viewer Sederhana
Agen properti ingin membuat virtual tour sederhana dari foto-foto rumah. Buatlah viewer yang: (a) membaca gambar panoramik dari berbagai ruangan, (b) menerapkan cylindrical/spherical projection, (c) memungkinkan navigasi kiri-kanan dengan keyboard, (d) mengoreksi distorsi lensa fisheye, (e) menambahkan hotspot area yang bisa diklik, (f) melakukan transisi antar ruangan dengan warping, (g) membuat minimap dari bird's eye view.

### Soal 5: Sistem Pengukuran Berbasis Kamera
Bengkel otomotif ingin mengukur dimensi komponen menggunakan kamera. Buatlah sistem yang: (a) melakukan kalibrasi kamera dengan checkerboard, (b) mengoreksi distorsi lensa, (c) mendeteksi objek referensi skala yang diketahui, (d) menghitung homografi ke bidang datar, (e) mengukur panjang dan lebar objek dalam satuan cm/mm, (f) menampilkan anotasi pengukuran pada gambar, (g) menyimpan laporan pengukuran.

### Soal 6: Alat Bantu Mapping Drone Sederhana
Tim surveyor menggunakan drone untuk pemetaan area pertanian. Buatlah tool yang: (a) membaca gambar aerial dari drone, (b) mengoreksi distorsi lensa drone, (c) menerapkan bird's eye view transform, (d) menstitch 2–4 gambar yang overlap, (e) menambahkan grid koordinat, (f) menghitung estimasi luas area, (g) menghasilkan peta mosaik dengan anotasi.

### Soal 7: Sistem Parkir Cerdas Sederhana
Pengelola parkir mal ingin mendeteksi plat nomor dari CCTV yang dipasang miring. Buatlah sistem prototype yang: (a) membaca gambar dari kamera CCTV (simulasi), (b) mendeteksi dan crop area plat nomor, (c) mengoreksi perspektif plat agar tegak, (d) meningkatkan kontras dengan gamma correction, (e) menyimpan hasil crop plat dengan timestamp, (f) menampilkan log kendaraan masuk, (g) estimasi waktu parkir.

### Soal 8: Photo Correction Suite
Studio foto memerlukan tool batch correction untuk foto klien. Buatlah suite yang: (a) auto-rotate foto berdasarkan orientasi, (b) koreksi perspektif (foto miring/tilted), (c) koreksi chromatic aberration dan distorsi lensa, (d) gamma correction otomatis, (e) white balance adjustment, (f) anti-aliased resize ke berbagai ukuran cetak (4R, 5R, A4), (g) simpan dengan metadata EXIF preserved.

### Soal 9: Sistem Kalibrasi Multi-Kamera
Lab robotik ingin mengkalibrasi 3 webcam yang dipasang dari sudut berbeda. Buatlah sistem yang: (a) mengkalibrasi setiap kamera secara individual, (b) menampilkan reprojection error per kamera, (c) mengoreksi distorsi masing-masing kamera, (d) memproyeksikan sumbu 3D pada marker di ketiga view, (e) menyimpan hasil kalibrasi, (f) membandingkan parameter antar kamera, (g) memvisualisasikan posisi ketiga kamera.

### Soal 10: Simulator Efek Lensa Kamera
Mahasiswa fotografi ingin memahami efek berbagai jenis lensa. Buatlah simulator yang: (a) simulasi efek focal length berbeda (wide, normal, tele) pada scene 3D sederhana, (b) simulasi depth of field (bokeh), (c) simulasi distorsi barrel (wide-angle) dan pincushion (tele), (d) simulasi vignette effect, (e) simulasi chromatic aberration, (f) perbandingan side-by-side antar "lensa", (g) interaktif dengan slider untuk parameter.

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
- **Format**: ZIP — `NIM_Nama_Project02.zip`
- **Isi**: Source code, README.md, screenshot (min. 5), data sample, hasil kalibrasi.
