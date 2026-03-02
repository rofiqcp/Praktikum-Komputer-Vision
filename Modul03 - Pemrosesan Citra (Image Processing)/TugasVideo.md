# TUGAS VIDEO - MODUL 3: PEMROSESAN CITRA (IMAGE PROCESSING)

---

## Deskripsi Tugas

Buatlah sebuah video edukasi berdurasi 10 hingga 15 menit yang mendemonstrasikan teknik-teknik
pemrosesan citra dari Modul 3. Video harus menggabungkan penjelasan konseptual singkat dengan
demonstrasi kode Python yang berjalan secara langsung, sehingga penonton dapat memahami baik
teori maupun implementasinya.

---

## Topik yang Harus Didemonstrasikan

Minimal **15 dari 20 topik** berikut harus ditampilkan dalam video. Setiap topik membutuhkan
waktu 30 hingga 60 detik demonstrasi.

### Daftar 20 Topik Praktikum

**01 - Deteksi Kontur (`01_contour_detection.py`)**
Tunjukkan: proses findContours pada gambar biner, perbedaan RETR_TREE vs RETR_EXTERNAL,
dan cara menggambar boundingRect di sekitar setiap kontur.

**02 - Histogram Equalization (`02_histogram_equalization.py`)**
Tunjukkan: gambar asli dengan kontras rendah, tampilan histogram sebelum dan sesudah equalization,
dan perbandingan visual antara equalizeHist dan histogram stretching.

**03 - CLAHE (`03_clahe.py`)**
Tunjukkan: perbandingan equalizeHist global vs. CLAHE pada gambar wajah atau gambar medis,
efek clipLimit berbeda, dan mengapa CLAHE menghasilkan hasil lebih natural.

**04 - Segmentasi Warna HSV (`04_color_segmentation_hsv.py`)**
Tunjukkan: konversi gambar ke HSV, pembuatan mask menggunakan inRange untuk warna tertentu,
dan aplikasi mask pada gambar asli sehingga hanya objek berwarna tertentu yang tampak.

**05 - Thresholding Global (`05_thresholding_global.py`)**
Tunjukkan: hasil lima tipe thresholding (BINARY, BINARY_INV, TRUNC, TOZERO, TOZERO_INV)
pada satu gambar yang sama dengan nilai threshold 127.

**06 - Thresholding Otsu dan Triangle (`06_thresholding_otsu_triangle.py`)**
Tunjukkan: nilai threshold yang dipilih otomatis oleh Otsu, histogram gambar bimodal yang
mengindikasikan kondisi ideal untuk Otsu, dan perbandingan Otsu vs. Triangle.

**07 - Adaptive Thresholding (`07_adaptive_thresholding.py`)**
Tunjukkan: gambar dokumen dengan pencahayaan tidak merata, kegagalan global thresholding,
dan keberhasilan adaptive thresholding (MEAN dan GAUSSIAN) dalam menangani kasus ini.

**08 - Konvolusi dan Filter2D (`08_konvolusi_dan_filter2d.py`)**
Tunjukkan: tampilan visual kernel yang sedang diaplikasikan, hasil kernel identitas (tidak berubah),
kernel blur, kernel sharpening, dan kernel emboss pada gambar yang sama.

**09 - Gaussian Blur (`09_gaussian_blur.py`)**
Tunjukkan: efek kenaikan ukuran kernel (3x3, 7x7, 15x15) terhadap tingkat kehalusan gambar,
dan perbedaan Gaussian blur vs. averaging blur dalam mempertahankan transisi halus.

**10 - Median dan Bilateral Filter (`10_median_dan_bilateral_filter.py`)**
Tunjukkan: gambar dengan salt-and-pepper noise, perbandingan Gaussian blur vs. median filter
dalam menghilangkan noise, dan keunggulan bilateral filter dalam mempertahankan tepi.

**11 - Sharpening (`11_sharpening.py`)**
Tunjukkan: unsharp masking dengan alpha berbeda, perbandingan gambar asli vs. gambar yang
disharp, dan efek oversharpening (ringing artifacts) saat alpha terlalu besar.

**12 - Deteksi Tepi Sobel (`12_deteksi_tepi_sobel.py`)**
Tunjukkan: Sobel_X yang mendeteksi tepi vertikal, Sobel_Y yang mendeteksi tepi horizontal,
magnitude gradien gabungan, dan peta arah gradien berwarna.

**13 - Deteksi Tepi Canny (`13_deteksi_tepi_canny.py`)**
Tunjukkan: proses Canny secara keseluruhan, pentingnya memilih threshold bawah dan atas,
perbandingan hasil Canny (garis tipis) vs. Sobel (tebal), dan efek pre-blur Gaussian.

**14 - Deteksi Tepi Laplacian (`14_deteksi_tepi_laplacian.py`)**
Tunjukkan: sensitivitas Laplacian terhadap noise (tanpa pre-blur), perbaikan dengan LoG,
dan konsep zero-crossing sebagai penanda lokasi tepi sebenarnya.

**15 - Morfologi Erosi dan Dilasi (`15_morfologi_erosi_dilasi.py`)**
Tunjukkan: efek erosi yang mengecilkan objek dan menghilangkan noise kecil, efek dilasi
yang memperbesar objek dan mengisi lubang, perbedaan structuring element.

**16 - Morfologi Lanjut (`16_morfologi_lanjut.py`)**
Tunjukkan: opening yang membersihkan spot noise kecil, closing yang mengisi lubang dalam
objek, dan morphological gradient sebagai alternatif edge detection.

**17 - Top-Hat dan Black-Hat (`17_tophat_blackhat.py`)**
Tunjukkan: Top-Hat yang menonjolkan objek terang pada latar tidak merata, Black-Hat yang
menonjolkan objek gelap, dan aplikasi nyata seperti deteksi teks atau granula.

**18 - Transformasi Fourier (`18_transformasi_fourier.py`)**
Tunjukkan: magnitude spectrum gambar (pola simetris), efek fftshift yang memindahkan DC ke
tengah, dan perbedaan spectrum gambar dengan tekstur berbeda (horizontal vs. vertikal vs. acak).

**19 - Filter Frekuensi (`19_filter_frekuensi.py`)**
Tunjukkan: low-pass filter di domain frekuensi yang memblur gambar, high-pass filter yang
mempertahankan tepi, ringing artifacts pada ideal filter, dan perbandingan dengan Butterworth.

**20 - Connected Components (`20_connected_components.py`)**
Tunjukkan: gambar biner dengan banyak objek, hasil labeling dengan warna berbeda per komponen,
statistik setiap blob (area, bounding box, centroid), dan filter berdasarkan ukuran.

---

## Struktur Video (Total 15 Menit)

### Bagian Intro - 1 Menit
- Perkenalkan diri dan tujuan video
- Sebutkan modul yang dibahas: Modul 3 Pemrosesan Citra
- Tunjukkan daftar topik yang akan didemonstrasikan
- Sebutkan library yang digunakan: OpenCV, NumPy, Matplotlib

### Bagian 1: Histogram, Thresholding, dan Segmentasi (3 Menit)

**Topik 02 + 03 (60 detik) - Histogram dan CLAHE**
- Mulai dengan gambar kontras rendah
- Tunjukkan histogram sebelum equalization
- Demonstrasikan equalizeHist dan CLAHE berdampingan
- Jelaskan mengapa CLAHE lebih unggul untuk gambar dengan pencahayaan tidak merata

**Topik 05 + 06 + 07 (90 detik) - Thresholding**
- Thresholding global: tampilkan 5 tipe dalam satu slide
- Otsu: tunjukkan histogram bimodal dan nilai threshold otomatis
- Adaptive: tunjukkan kegagalan global vs. keberhasilan adaptive pada dokumen

**Topik 04 (30 detik) - Segmentasi Warna HSV**
- Tunjukkan gambar buah atau tanaman
- Segmentasi satu warna (misalnya hijau) secara langsung
- Tampilkan mask dan hasil masking

### Bagian 2: Filter Spasial, Sharpening, dan Deteksi Tepi (4 Menit)

**Topik 08 + 09 (60 detik) - Konvolusi dan Gaussian Blur**
- Tunjukkan kernel sebagai matriks angka
- Demonstrasikan beberapa kernel berbeda: identity, avg, sharp, emboss
- Gaussian blur dengan ukuran kernel berbeda

**Topik 10 (60 detik) - Median dan Bilateral Filter**
- Tambahkan salt-and-pepper noise secara langsung di depan kamera
- Tunjukkan kegagalan Gaussian, keberhasilan median
- Demonstrasikan bilateral: smooth area tapi tepi tetap tajam

**Topik 11 (30 detik) - Sharpening**
- Unsharp masking dengan alpha rendah vs. tinggi
- Tunjukkan ringing artifacts pada alpha terlalu besar

**Topik 12 + 13 + 14 (90 detik) - Deteksi Tepi**
- Sobel X dan Y secara visual
- Canny: tepi tipis, jumlah tepi berubah sesuai threshold
- Laplacian: sensitif noise, perbaikan dengan LoG

### Bagian 3: Morfologi, Fourier, dan Connected Components (4 Menit)

**Topik 01 (45 detik) - Deteksi Kontur**
- Temukan kontur pada gambar biner
- Gambar bounding rect setiap kontur
- Tunjukkan perbedaan RETR_TREE vs. RETR_EXTERNAL

**Topik 15 + 16 + 17 (90 detik) - Morfologi**
- Erosi: objek mengecil, noise hilang
- Dilasi: objek membesar, lubang terisi
- Opening dan closing: aplikasi pembersihan
- Top-Hat atau Black-Hat: pilih salah satu yang paling visual menarik

**Topik 18 + 19 (60 detik) - Fourier dan Filter Frekuensi**
- Magnitude spectrum: tampilkan pola simetris
- LPF di domain frekuensi: gambar menjadi blur
- HPF: hanya tepi yang tersisa
- Sebutkan hubungan dengan spatial filtering

**Topik 20 (45 detik) - Connected Components**
- Gambar biner dengan banyak objek
- Labeling berwarna-warni
- Tampilkan statistik setiap blob

### Bagian 4: Contoh Aplikasi Nyata (2 Menit)

Pilih salah satu skenario aplikasi nyata dan demonstrasikan pipeline sederhana:

**Opsi A - Inspeksi Kualitas Buah:**
Preprocessing (CLAHE + bilateral) -> Segmentasi warna (HSV) -> Morfologi cleanup ->
Connected components -> Hitung dan klasifikasikan buah berdasarkan ukuran

**Opsi B - Analisis Dokumen Bercetak:**
Grayscale -> Adaptive thresholding -> Morfologi (closing) -> Connected components ->
Identifikasi kata atau karakter berdasarkan bounding box

**Opsi C - Deteksi Cacat Produk:**
CLAHE -> Gaussian blur -> Canny edge detection -> Morfologi -> Analisis kontur
untuk menemukan cacat berukuran tertentu

Jelaskan: (1) mengapa setiap tahap diperlukan, (2) parameter kunci yang digunakan.

### Bagian 5: Kesimpulan (1 Menit)
- Rekap teknik-teknik yang paling penting
- Sebutkan 2-3 insight menarik yang ditemukan selama praktikum
- Saran untuk eksplorasi lebih lanjut (misalnya deep learning untuk segmentasi)
- Ucapan terima kasih dan penutup

---

## Aspek Teknis yang Dinilai

| Aspek | Bobot | Indikator |
|-------|-------|-----------|
| **Kejelasan Penjelasan** | 30% | Penjelasan konsep akurat dan mudah dipahami, penggunaan bahasa yang tepat, kemampuan menjelaskan "mengapa" bukan hanya "apa" |
| **Demonstrasi Visual** | 30% | Kode terlihat jelas, output gambar visible dan informatif, terdapat perbandingan before/after, penggunaan subplot yang baik |
| **Kelengkapan Topik** | 20% | Jumlah topik yang didemonstrasikan (minimal 15 dari 20), kedalaman pembahasan setiap topik |
| **Kualitas Video dan Audio** | 20% | Resolusi minimal 720p, audio jernih dan volume konsisten, tidak ada lag saat screen recording, latar belakang bersih |

---

## Panduan Teknis Pembuatan Video

### Software yang Direkomendasikan
- **Screen Recording:** OBS Studio (gratis), Bandicam, atau ShareX
- **Editing:** DaVinci Resolve (gratis), Kdenlive, atau CapCut
- **Presentasi:** Bisa menggunakan Jupyter Notebook, VS Code, atau kombinasi

### Tips Kualitas Video
- Resolusi minimal 1280x720 (HD); disarankan 1920x1080 (Full HD)
- Frame rate minimal 30 FPS
- Pastikan font terminal dan editor cukup besar untuk dibaca (minimal 14pt)
- Gunakan tema terang (light theme) agar teks terlihat lebih jelas di video
- Rekam audio terpisah menggunakan headset atau microphone eksternal jika memungkinkan

### Tips Alur Demonstrasi
- Tunjukkan kode SEBELUM menjalankan hasil, bukan langsung hasilnya
- Highlight baris kode yang penting saat menjelaskan
- Beri jeda sejenak setelah setiap hasil muncul agar penonton bisa melihat
- Tambahkan caption atau overlay teks untuk nama fungsi yang digunakan

---

## Format Pengumpulan

### Format File
- Format video: **MP4** (codec H.264 direkomendasikan)
- Resolusi minimal: 720p (1280x720 piksel)
- Resolusi yang disarankan: 1080p (1920x1080 piksel)
- Frame rate: minimal 30 FPS
- Audio: stereo atau mono, bitrate minimal 128 kbps

### Platform Upload
- Upload ke **Google Drive** dan set agar "Anyone with the link can view"
- ATAU upload ke **YouTube** dengan visibility Unlisted atau Public
- Pastikan link dapat diakses tanpa login

### Link Pengumpulan
- Submit link video melalui platform LMS (Moodle/Google Classroom/e-learning kampus)
- Format judul file/video: `[NIM]_[Nama]_TugasVideo_Modul03`
- Contoh: `20230001_BudiSantoso_TugasVideo_Modul03`

### Deadline
- Deadline pengumpulan: **sesuai jadwal yang ditetapkan dosen**
- Keterlambatan: pengurangan nilai 10 poin per hari keterlambatan

---

## Contoh Struktur Script / Outline

```
INTRO (1 menit)
  - "Halo, nama saya [Nama], NIM [NIM]"
  - "Video ini mendemonstrasikan 20 teknik pemrosesan citra dari Modul 3"
  - "Library yang digunakan: OpenCV 4.x, NumPy, Matplotlib"
  - [tampilkan daftar topik]

BAGIAN 1 - HISTOGRAM DAN THRESHOLDING (3 menit)
  - [buka file 02_histogram_equalization.py]
  - [jalankan, jelaskan output]
  - [buka file 03_clahe.py]
  - [jalankan, bandingkan dengan equalizeHist]
  - [lanjut ke 05, 06, 07 thresholding]
  - ...

BAGIAN 2 - FILTER DAN DETEKSI TEPI (4 menit)
  - [percobaan 08 hingga 14]
  - ...

BAGIAN 3 - MORFOLOGI DAN ANALISIS LANJUT (4 menit)
  - [percobaan 01, 15 hingga 20]
  - ...

APLIKASI NYATA (2 menit)
  - [demo pipeline pilihan]

KESIMPULAN (1 menit)
  - [insight dan penutup]
```

---

## Checklist Sebelum Mengumpulkan

Pastikan semua poin berikut sudah terpenuhi:

- [ ] Video berdurasi 10-15 menit
- [ ] Minimal 15 dari 20 topik didemonstrasikan
- [ ] Setiap topik mendapatkan penjelasan konseptual singkat, bukan hanya run kode saja
- [ ] Kode terlihat jelas dan output gambar terlihat informatif
- [ ] Audio terdengar jelas dan volume konsisten
- [ ] Resolusi minimal 720p
- [ ] Link dapat diakses tanpa login
- [ ] Judul file sesuai format: `[NIM]_[Nama]_TugasVideo_Modul03`
- [ ] Link sudah di-submit sebelum deadline

---

## Penilaian Bonus

Poin bonus sebesar 5-10 poin dapat diberikan untuk:

- Video yang sangat komunikatif dan mudah dipahami orang awam sekalipun
- Demonstrasi semua 20 topik secara lengkap (bukan hanya 15 minimum)
- Studi kasus aplikasi nyata yang orisinal dan menarik
- Kualitas editing yang profesional (transisi, caption, music latar)
- Video yang berhasil viral atau mendapat engagement positif jika di-upload di YouTube

---

*Tugas Video ini merupakan bagian dari penilaian Praktikum Komputer Vision - Modul 3: Pemrosesan Citra*
