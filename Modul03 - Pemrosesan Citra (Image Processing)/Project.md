# PROJECT MODUL 3: PEMROSESAN CITRA (IMAGE PROCESSING)

---

## Deskripsi

Project ini mengintegrasikan teknik-teknik pemrosesan citra dari Modul 3 ke dalam sebuah pipeline
analisis citra yang lengkap dan kohesif. Mahasiswa membangun sistem pemrosesan citra end-to-end
yang menggabungkan preprocessing, segmentasi, deteksi fitur, analisis morfologi, dan analisis
domain frekuensi.

---

## Proyek Utama: Pipeline Analisis Citra Lengkap

### Latar Belakang

Dalam aplikasi nyata seperti inspeksi kualitas produk, analisis citra medis, atau pertanian presisi,
sebuah sistem pemrosesan citra tidak bekerja dengan satu teknik saja, melainkan mengombinasikan
berbagai metode dalam urutan yang logis. Project ini mensimulasikan scenario tersebut.

### Deskripsi Pipeline

Bangun sebuah program Python yang mengimplementasikan pipeline analisis citra berikut:

**Tahap 1 - Pemuatan dan Pra-Pemrosesan:**
- Muat gambar input (bisa dari file atau kamera)
- Terapkan CLAHE pada channel L dari ruang warna LAB untuk normalisasi kontras
- Terapkan bilateral filter untuk mengurangi noise sambil mempertahankan tepi

**Tahap 2 - Segmentasi Warna:**
- Konversi gambar ke ruang warna HSV
- Segmentasikan objek berdasarkan warna target menggunakan `cv2.inRange()`
- Hasilkan mask biner untuk setiap warna

**Tahap 3 - Deteksi Tepi:**
- Terapkan Canny edge detection pada gambar yang sudah dipreproses
- Optionally gabungkan dengan hasil Sobel untuk analisis arah tepi

**Tahap 4 - Penyempurnaan Morfologi:**
- Terapkan opening untuk menghilangkan noise kecil pada mask
- Terapkan closing untuk mengisi lubang dalam objek
- Gunakan Top-Hat atau Black-Hat jika pencahayaan tidak merata

**Tahap 5 - Analisis Connected Components:**
- Label setiap objek yang teridentifikasi
- Hitung statistik: jumlah objek, area, bounding box, centroid
- Filter objek berdasarkan ukuran dan bentuk

**Tahap 6 - Analisis Domain Frekuensi (Opsional Lanjutan):**
- Hitung DFT dari gambar grayscale
- Tampilkan magnitude spectrum
- Identifikasi komponen frekuensi dominan

**Tahap 7 - Visualisasi Hasil:**
- Tampilkan setiap tahap pipeline berdampingan
- Buat laporan teks dengan statistik hasil analisis
- Simpan semua gambar hasil ke folder `output/`

---

## Spesifikasi Teknis

Program harus memenuhi persyaratan teknis berikut:

1. **CLAHE atau Histogram Equalization:** Gunakan `cv2.createCLAHE()` dengan parameter yang dapat dikonfigurasi (clipLimit dan tileGridSize).

2. **Filter Edge-Preserving:** Implementasikan bilateral filter (`cv2.bilateralFilter`) dengan parameter d, sigmaColor, sigmaSpace yang dapat disesuaikan.

3. **Segmentasi HSV:** Gunakan `cv2.inRange()` dengan setidaknya dua warna berbeda. Warna merah harus menggunakan dua range terpisah.

4. **Minimal Dua Tipe Thresholding:** Implementasikan thresholding global (Otsu) dan adaptive thresholding, lalu bandingkan hasilnya.

5. **Canny Edge Detection:** Gunakan `cv2.Canny()` dengan hysteresis thresholding yang dapat dikonfigurasi.

6. **Operasi Morfologi Lengkap:** Implementasikan setidaknya erosi, dilasi, opening, dan closing menggunakan `cv2.morphologyEx()`.

7. **Connected Components:** Gunakan `cv2.connectedComponentsWithStats()` dan filter komponen berdasarkan area minimal dan maksimal.

8. **Transformasi Fourier:** Hitung DFT menggunakan `np.fft.fft2()` dan visualisasikan magnitude spectrum.

9. **Filter Frekuensi:** Implementasikan minimal satu filter di domain frekuensi (low-pass atau high-pass) dan bandingkan dengan hasil domain spasial.

10. **Visualisasi Komprehensif:** Tampilkan seluruh tahap pipeline dalam satu figure Matplotlib dengan subplot yang berlabel.

11. **Laporan Statistik:** Cetak atau simpan laporan berisi: jumlah objek terdeteksi, area rata-rata, objek terbesar, distribusi ukuran.

12. **Kontrol Parameter:** Semua parameter penting (threshold, kernel size, clip limit, dll.) harus didefinisikan sebagai variabel di bagian atas program agar mudah disesuaikan.

---

## Improvisasi yang Disarankan

Berikut adalah ide-ide pengembangan tambahan untuk nilai kreativitas:

1. **Versi Real-Time dengan Webcam:** Implementasikan pipeline yang berjalan secara real-time menggunakan `cv2.VideoCapture(0)`. Optimalkan untuk minimal 10 FPS dengan mengurangi resolusi dan menggunakan teknik yang lebih cepat.

2. **Batch Processing:** Buat program yang memproses seluruh gambar dalam sebuah folder secara otomatis, menghasilkan laporan CSV dengan statistik setiap gambar dan menampilkan gambar yang "gagal" memenuhi kriteria.

3. **Pipeline Domain Frekuensi:** Buat pipeline yang memfilter gambar menggunakan band-pass filter di domain frekuensi untuk mengisolasi tekstur pada skala tertentu, lalu analisis hasilnya dengan morfologi.

4. **Visualisasi Pohon Kontur:** Implementasikan visualisasi hirarki kontur (RETR_TREE) sebagai tree diagram, menampilkan relasi parent-child antar kontur.

5. **Penghitung Objek Adaptif:** Sistem yang secara otomatis menentukan nilai threshold terbaik (Otsu/Canny) berdasarkan karakteristik histogram gambar input.

6. **Analisis Granulometri:** Gunakan serangkaian opening dengan structuring element yang makin besar untuk menganalisis distribusi ukuran objek (morphological granulometry).

7. **Segmentasi Multi-Warna Interaktif:** GUI sederhana menggunakan OpenCV trackbar untuk mengatur parameter HSV secara real-time dan langsung melihat hasil segmentasi.

8. **Perbandingan Filter Otomatis:** Program yang secara sistematis membandingkan 5 kombinasi filter berbeda (Gaussian, median, bilateral, dll.) menggunakan metrik PSNR dan SSIM, lalu merekomendasikan filter terbaik untuk gambar tertentu.

9. **Export ke PDF:** Gunakan matplotlib untuk menyimpan seluruh pipeline (semua tahap) ke dalam satu file PDF multi-halaman sebagai laporan otomatis.

10. **Integrasi dengan Contour Detection:** Setelah pipeline selesai, tampilkan kontur setiap objek yang terdeteksi di connected components beserta informasi area, perimeter, dan roundness.

---

## Format Pengumpulan

### File yang Dikumpulkan:
- **File Python (.py):** Satu file utama pipeline (misalnya `project_m03_pipeline.py`) beserta file pendukung jika ada.
- **Laporan PDF:** Maksimal 5 halaman, berisi: deskripsi pipeline, screenshot hasil setiap tahap, analisis dan kesimpulan, referensi.
- **Video Demo (MP4):** Durasi 2-5 menit, mendemonstrasikan program berjalan dan menjelaskan setiap tahap.

### Struktur Folder Pengumpulan:
```
NIM_Nama_ProjectM03/
  project_m03_pipeline.py
  laporan_project_m03.pdf
  demo_project_m03.mp4
  output/
    (gambar-gambar hasil tiap tahap)
  README.txt
    (instruksi menjalankan program)
```

---

## Kriteria Penilaian

| Aspek | Bobot | Kriteria |
|-------|-------|----------|
| **Implementasi Teknis** | 40% | Ketepatan implementasi minimal 10 teknik yang diwajibkan, kualitas kode (dokumentasi, modularitas, parameter yang dapat dikonfigurasi), tidak ada error saat dijalankan |
| **Analisis Hasil** | 25% | Kualitas analisis pada setiap tahap pipeline, interpretasi hasil yang akurat, identifikasi kelebihan dan kekurangan setiap teknik |
| **Kreativitas** | 15% | Penambahan fitur di luar spesifikasi minimum, ide orisinal dalam menggabungkan teknik-teknik, pemilihan kasus penggunaan yang menarik |
| **Laporan** | 10% | Kelengkapan dan kejelasan laporan PDF, kualitas screenshot dan visualisasi, ketepatan bahasa teknis |
| **Demo Video** | 10% | Kejelasan demonstrasi dan penjelasan verbal, mencakup semua tahap pipeline utama, kualitas rekaman |

### Rubrik Detail Implementasi Teknis (40%)

| Sub-Aspek | Nilai |
|-----------|-------|
| Semua 12 spesifikasi teknis terpenuhi dengan benar | 36-40 |
| 10-11 spesifikasi terpenuhi | 30-35 |
| 7-9 spesifikasi terpenuhi | 22-29 |
| 4-6 spesifikasi terpenuhi | 14-21 |
| Kurang dari 4 spesifikasi | 0-13 |

---

## Timeline Pengerjaan (Saran)

| Minggu | Kegiatan |
|--------|----------|
| Minggu 1 | Merancang arsitektur pipeline, menyiapkan dataset, implementasi tahap 1-2 |
| Minggu 2 | Implementasi tahap 3-5, uji coba dan debugging |
| Minggu 3 | Implementasi tahap 6-7, penambahan fitur improvisasi |
| Minggu 4 | Penulisan laporan, perekaman video demo, finalisasi dan pengumpulan |

---

*Project ini merupakan bagian dari penilaian Praktikum Komputer Vision - Modul 3: Pemrosesan Citra*
