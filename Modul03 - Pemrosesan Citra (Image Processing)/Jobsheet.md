# JOBSHEET MODUL 3: PEMROSESAN CITRA (IMAGE PROCESSING)

**Mata Kuliah:** Praktikum Komputer Vision
**Modul:** 3
**Topik:** Pemrosesan Citra (Image Processing)
**Durasi:** 4 x 50 menit

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa diharapkan mampu:

1. Mendeteksi dan menganalisis kontur (contour) pada gambar menggunakan OpenCV, termasuk hirarki dan pendekatan kontur.
2. Menerapkan histogram equalization untuk peningkatan kontras gambar secara global maupun adaptif.
3. Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization) untuk equalization adaptif dengan pembatasan clip limit.
4. Melakukan segmentasi warna menggunakan ruang warna HSV dengan teknik in-range masking.
5. Menerapkan berbagai metode thresholding: global (binary, trunc, tozero), Otsu, Triangle, dan adaptif.
6. Memahami prinsip konvolusi 2D dan mengimplementasikan filter spasial dengan kernel kustom menggunakan filter2D.
7. Menerapkan berbagai blur filter: Gaussian blur, median filter, dan bilateral filter untuk reduksi noise.
8. Melakukan sharpening gambar menggunakan unsharp mask, Laplacian sharpening, dan kernel kustom.
9. Mendeteksi tepi pada gambar menggunakan operator Sobel, Canny, dan Laplacian.
10. Melakukan operasi morfologi dasar: erosi, dilasi, opening, closing, dan gradient morfologi.
11. Menerapkan transformasi Top-Hat dan Black-Hat untuk analisis tekstur dan pencahayaan lokal.
12. Menganalisis gambar di domain frekuensi menggunakan Discrete Fourier Transform (DFT/FFT).
13. Menerapkan connected components labeling untuk identifikasi dan analisis blob pada gambar biner.

---

## 2. ALAT DAN BAHAN

### Perangkat Keras
- Komputer/laptop dengan RAM minimal 4 GB
- Kamera webcam (opsional, untuk percobaan real-time)

### Perangkat Lunak
- Python 3.8 atau lebih baru
- OpenCV (`cv2`) versi 4.5+
- NumPy versi 1.21+
- Matplotlib versi 3.4+
- SciPy (opsional, untuk filter tambahan)
- IDE: VS Code, PyCharm, atau Jupyter Notebook

### Dataset / Gambar
- Gambar grayscale untuk thresholding dan morfologi (foto koin, teks, sel darah)
- Gambar berwarna untuk segmentasi HSV (foto buah, tanaman, atau objek berwarna)
- Gambar dengan noise untuk percobaan filter
- Gambar dengan tekstur untuk analisis Fourier
- Gambar dengan banyak objek untuk connected components

### Instalasi Dependensi
```
pip install opencv-python numpy matplotlib scipy
```

---

## 3. PERSIAPAN

1. Pastikan Python dan semua library telah terinstal dengan menjalankan:
   ```
   python -c "import cv2; import numpy; import matplotlib; print('OK')"
   ```
2. Unduh atau siapkan folder `praktikum/` di dalam Modul03.
3. Siapkan gambar-gambar sampel dan simpan ke folder yang dapat diakses setiap skrip.
4. Buka setiap file Python percobaan menggunakan editor pilihan Anda.
5. Baca seluruh komentar dan penjelasan di dalam setiap file sebelum menjalankannya.
6. Catat setiap hasil observasi ke dalam laporan praktikum.
7. Pastikan direktori kerja sudah benar sebelum menjalankan skrip.

---

## 4. LANGKAH KERJA

---

### Percobaan 1: Deteksi Kontur
**File:** `01_contour_detection.py`
**Tujuan:** Memahami cara mendeteksi dan menggambar kontur pada gambar menggunakan `cv2.findContours()` dan `cv2.drawContours()`, serta memahami hirarki dan metode aproksimasi kontur.

**Langkah Kerja:**
1. Buka file `01_contour_detection.py` menggunakan editor teks atau IDE.
2. Jalankan skrip dengan perintah `python 01_contour_detection.py`.
3. Amati perbedaan hasil kontur dengan mode `RETR_EXTERNAL`, `RETR_LIST`, dan `RETR_TREE`.
4. Amati perbedaan jumlah titik kontur antara `CHAIN_APPROX_NONE` dan `CHAIN_APPROX_SIMPLE`.
5. Bandingkan hasil `cv2.boundingRect()`, `cv2.minAreaRect()`, dan `cv2.minEnclosingCircle()` pada gambar yang sama.
6. Periksa folder `output/` dan pastikan file hasil gambar tersimpan dengan benar.

---

### Percobaan 2: Histogram Equalization
**File:** `02_histogram_equalization.py`
**Tujuan:** Menerapkan `cv2.equalizeHist()` untuk meratakan distribusi histogram gambar grayscale, serta membandingkannya dengan histogram stretching manual menggunakan normalisasi piksel min-max.

**Langkah Kerja:**
1. Buka file `02_histogram_equalization.py` di editor.
2. Jalankan skrip dan amati gambar input yang memiliki kontras rendah.
3. Amati histogram sebelum dan sesudah equalizeHist - perhatikan distribusi piksel menjadi lebih merata.
4. Amati perbedaan visual antara equalizeHist dan histogram stretching manual (perubahan range min-max piksel).
5. Bandingkan hasil pada gambar yang terlalu gelap vs. gambar yang terlalu terang.
6. Periksa folder `output/` untuk memastikan semua file perbandingan tersimpan.

---

### Percobaan 3: CLAHE
**File:** `03_clahe.py`
**Tujuan:** Menerapkan Contrast Limited Adaptive Histogram Equalization (CLAHE) yang membagi gambar menjadi tile kecil dan membatasi penguatan kontras dengan `clipLimit`, menghasilkan peningkatan kontras lebih seragam daripada equalizeHist global.

**Langkah Kerja:**
1. Buka file `03_clahe.py` di editor.
2. Jalankan skrip dengan berbagai nilai `clipLimit` (1.0, 2.0, 5.0, 10.0) dan amati hasilnya.
3. Amati perbedaan hasil dengan `tileGridSize` kecil (4x4) vs. besar (16x16).
4. Bandingkan hasil CLAHE dengan equalizeHist global pada gambar medis atau wajah dengan pencahayaan tidak merata.
5. Amati artifact berlebih pada equalizeHist vs. tampilan lebih natural pada CLAHE.
6. Simpan dan periksa gambar perbandingan di folder `output/`.

---

### Percobaan 4: Segmentasi Warna HSV
**File:** `04_color_segmentation_hsv.py`
**Tujuan:** Melakukan segmentasi objek berdasarkan warna menggunakan ruang warna HSV dan fungsi `cv2.inRange()` untuk membuat mask biner yang memisahkan piksel sesuai rentang warna target.

**Langkah Kerja:**
1. Buka file `04_color_segmentation_hsv.py` di editor.
2. Jalankan skrip pada gambar berwarna (buah, tanaman, atau objek berwarna).
3. Amati hasil konversi BGR ke HSV dan distribusi nilai H, S, V untuk warna target.
4. Amati pengaruh mengubah batas bawah dan atas HSV terhadap hasil segmentasi.
5. Bandingkan hasil segmentasi warna merah (yang memerlukan dua range) vs. warna hijau/biru.
6. Periksa folder `output/` untuk gambar mask dan gambar hasil masking.

---

### Percobaan 5: Thresholding Global
**File:** `05_thresholding_global.py`
**Tujuan:** Memahami berbagai tipe thresholding global menggunakan `cv2.threshold()`: BINARY, BINARY_INV, TRUNC, TOZERO, dan TOZERO_INV dengan nilai threshold yang ditentukan secara manual.

**Langkah Kerja:**
1. Buka file `05_thresholding_global.py` di editor.
2. Jalankan skrip dan amati perbedaan setiap tipe thresholding pada gambar yang sama.
3. Amati pengaruh nilai threshold (50, 100, 127, 180) terhadap jumlah piksel putih dan hitam.
4. Amati perbedaan BINARY vs. BINARY_INV dan TOZERO vs. TOZERO_INV.
5. Bandingkan hasil thresholding pada gambar dengan distribusi piksel bimodal vs. multimodal.
6. Periksa gambar hasil di folder `output/`.

---

### Percobaan 6: Thresholding Otsu dan Triangle
**File:** `06_thresholding_otsu_triangle.py`
**Tujuan:** Menerapkan metode Otsu yang secara otomatis menentukan nilai threshold optimal dengan memaksimalkan variance antar-kelas (between-class variance), serta metode Triangle untuk distribusi histogram miring.

**Langkah Kerja:**
1. Buka file `06_thresholding_otsu_triangle.py` di editor.
2. Jalankan skrip dan catat nilai threshold yang dipilih otomatis oleh metode Otsu.
3. Amati bahwa Otsu bekerja optimal pada gambar dengan histogram bimodal (dua puncak jelas).
4. Amati perbedaan nilai threshold dan hasil antara metode Otsu dan Triangle pada beberapa gambar.
5. Bandingkan threshold otomatis Otsu dengan threshold manual yang ditentukan secara visual.
6. Periksa gambar hasil thresholding di folder `output/`.

---

### Percobaan 7: Adaptive Thresholding
**File:** `07_adaptive_thresholding.py`
**Tujuan:** Menerapkan adaptive thresholding yang menghitung nilai threshold secara lokal per blok piksel menggunakan metode mean atau Gaussian, berguna untuk gambar dengan pencahayaan tidak merata.

**Langkah Kerja:**
1. Buka file `07_adaptive_thresholding.py` di editor.
2. Jalankan skrip pada gambar dokumen dengan pencahayaan tidak merata (bayangan, gradasi).
3. Amati perbedaan hasil `ADAPTIVE_THRESH_MEAN_C` vs. `ADAPTIVE_THRESH_GAUSSIAN_C`.
4. Amati pengaruh ukuran blok (blockSize: 11, 21, 51) terhadap ketajaman dan noise hasil thresholding.
5. Bandingkan hasil adaptive thresholding vs. global thresholding pada gambar dengan pencahayaan tidak merata.
6. Periksa folder `output/` untuk semua file hasil.

---

### Percobaan 8: Konvolusi dan Filter2D
**File:** `08_konvolusi_dan_filter2d.py`
**Tujuan:** Memahami prinsip konvolusi 2D dan mengimplementasikan filter spasial menggunakan `cv2.filter2D()` dengan berbagai kernel kustom (identitas, blur rata-rata, sharpening, emboss).

**Langkah Kerja:**
1. Buka file `08_konvolusi_dan_filter2d.py` di editor.
2. Jalankan skrip dan amati hasil kernel identitas - gambar tidak berubah.
3. Amati kernel rata-rata (averaging kernel 3x3, 5x5) dan efek blurring yang dihasilkan.
4. Amati kernel sharpening dan emboss serta efeknya terhadap tepi gambar.
5. Bandingkan hasil `cv2.filter2D()` dengan implementasi konvolusi manual menggunakan NumPy.
6. Periksa folder `output/` untuk semua gambar hasil berbagai kernel.

---

### Percobaan 9: Gaussian Blur
**File:** `09_gaussian_blur.py`
**Tujuan:** Menerapkan Gaussian blur menggunakan `cv2.GaussianBlur()`, `cv2.blur()`, dan `cv2.boxFilter()`, serta memahami pengaruh ukuran kernel dan nilai sigma terhadap tingkat blurring.

**Langkah Kerja:**
1. Buka file `09_gaussian_blur.py` di editor.
2. Jalankan skrip dan amati perbedaan visual antara Gaussian blur, averaging blur, dan box filter.
3. Amati pengaruh ukuran kernel (3, 5, 9, 15, 21) terhadap tingkat kehalusan gambar.
4. Amati pengaruh nilai sigma (0.5, 1.0, 2.0, 5.0) terhadap distribusi bobot Gaussian.
5. Bandingkan efektivitas reduksi noise antara averaging blur vs. Gaussian blur.
6. Periksa folder `output/` untuk gambar perbandingan semua metode blur.

---

### Percobaan 10: Median dan Bilateral Filter
**File:** `10_median_dan_bilateral_filter.py`
**Tujuan:** Menerapkan median filter untuk menghilangkan salt-and-pepper noise dan bilateral filter untuk menghaluskan gambar sambil mempertahankan tepi (edge-preserving smoothing).

**Langkah Kerja:**
1. Buka file `10_median_dan_bilateral_filter.py` di editor.
2. Tambahkan salt-and-pepper noise pada gambar dan amati hasilnya.
3. Amati perbedaan antara Gaussian blur dan median filter dalam menangani salt-and-pepper noise.
4. Amati hasil bilateral filter - perhatikan tepi tetap tajam meskipun area rata menjadi halus.
5. Bandingkan waktu komputasi antara median filter dan bilateral filter dengan ukuran kernel yang sama.
6. Periksa folder `output/` untuk semua gambar perbandingan.

---

### Percobaan 11: Sharpening
**File:** `11_sharpening.py`
**Tujuan:** Menerapkan sharpening gambar menggunakan unsharp masking dengan rumus `sharpened = orig + alpha * (orig - blur)`, Laplacian sharpening, dan kernel sharpening kustom.

**Langkah Kerja:**
1. Buka file `11_sharpening.py` di editor.
2. Jalankan skrip dan amati hasil unsharp mask dengan berbagai nilai alpha (0.5, 1.0, 2.0, 5.0).
3. Amati Laplacian sharpening dan bagaimana detail tepi gambar dipertegas.
4. Amati perbedaan kernel sharpening standar 3x3 vs. high-boost filtering.
5. Bandingkan oversharpening (ringing artifacts) pada nilai alpha terlalu tinggi vs. sharpening optimal.
6. Periksa folder `output/` untuk semua hasil gambar.

---

### Percobaan 12: Deteksi Tepi Sobel
**File:** `12_deteksi_tepi_sobel.py`
**Tujuan:** Mendeteksi tepi gambar menggunakan operator Sobel yang menghitung gradien horizontal (Gx) dan vertikal (Gy), serta magnitude gradien G = sqrt(Gx^2 + Gy^2) dan arah tepi theta = arctan(Gy/Gx).

**Langkah Kerja:**
1. Buka file `12_deteksi_tepi_sobel.py` di editor.
2. Jalankan skrip dan amati Sobel_X (tepi vertikal) dan Sobel_Y (tepi horizontal) secara terpisah.
3. Amati magnitude gradien dan visualisasinya sebagai peta intensitas.
4. Amati peta arah gradien dan visualisasi dengan peta warna.
5. Bandingkan ukuran kernel Sobel ksize=1 (Scharr), ksize=3, dan ksize=5 terhadap ketebalan tepi.
6. Periksa folder `output/` untuk semua hasil deteksi tepi.

---

### Percobaan 13: Deteksi Tepi Canny
**File:** `13_deteksi_tepi_canny.py`
**Tujuan:** Menerapkan algoritma Canny edge detection multi-tahap: Gaussian smoothing, komputasi gradien Sobel, Non-Maximum Suppression (NMS), dan hysteresis thresholding dengan threshold bawah dan atas.

**Langkah Kerja:**
1. Buka file `13_deteksi_tepi_canny.py` di editor.
2. Jalankan skrip dan amati pengaruh threshold bawah dan atas terhadap jumlah tepi yang terdeteksi.
3. Amati bahwa tepi Canny berupa garis tipis satu piksel (berkat NMS) dibandingkan Sobel yang tebal.
4. Amati pengaruh Gaussian blur pra-proses (ksize=3, 5, 7) terhadap noise pada tepi Canny.
5. Bandingkan hasil Canny dengan Sobel dan Laplacian pada gambar yang sama.
6. Periksa folder `output/` untuk semua gambar hasil deteksi tepi Canny.

---

### Percobaan 14: Deteksi Tepi Laplacian
**File:** `14_deteksi_tepi_laplacian.py`
**Tujuan:** Mendeteksi tepi menggunakan operator Laplacian (turunan kedua dari intensitas gambar), Laplacian of Gaussian (LoG), zero-crossing, dan Difference of Gaussians (DoG) sebagai aproksimasi LoG.

**Langkah Kerja:**
1. Buka file `14_deteksi_tepi_laplacian.py` di editor.
2. Jalankan skrip dan amati perbedaan Laplacian pada gambar dengan dan tanpa Gaussian pre-blur.
3. Amati konsep zero-crossing pada gambar Laplacian yang menandai lokasi tepi sebenarnya.
4. Amati DoG sebagai aproksimasi LoG dan bandingkan hasilnya dengan LoG langsung.
5. Bandingkan sensitivitas noise antara Laplacian vs. Sobel vs. Canny.
6. Periksa folder `output/` untuk semua gambar hasil.

---

### Percobaan 15: Morfologi Erosi dan Dilasi
**File:** `15_morfologi_erosi_dilasi.py`
**Tujuan:** Menerapkan operasi morfologi dasar erosi dan dilasi dengan berbagai structuring element (persegi, elips, silang) dan jumlah iterasi untuk memahami efeknya pada objek biner.

**Langkah Kerja:**
1. Buka file `15_morfologi_erosi_dilasi.py` di editor.
2. Jalankan skrip pada gambar biner dan amati efek erosi (objek mengecil, noise kecil hilang).
3. Amati efek dilasi (objek membesar, lubang kecil terisi).
4. Amati perbedaan structuring element berbentuk persegi vs. elips vs. silang pada hasil erosi/dilasi.
5. Bandingkan hasil iterasi 1x, 3x, dan 5x erosi/dilasi terhadap ukuran objek.
6. Periksa folder `output/` untuk semua gambar hasil morfologi.

---

### Percobaan 16: Morfologi Lanjut
**File:** `16_morfologi_lanjut.py`
**Tujuan:** Menerapkan operasi morfologi lanjut: opening (erosi kemudian dilasi), closing (dilasi kemudian erosi), dan morphological gradient sebagai perbedaan dilasi dan erosi untuk aplikasi pembersihan dan pengisian.

**Langkah Kerja:**
1. Buka file `16_morfologi_lanjut.py` di editor.
2. Jalankan skrip dan amati opening yang menghilangkan noise spot kecil tanpa mengubah bentuk objek besar.
3. Amati closing yang mengisi lubang kecil dalam objek tanpa mengubah bentuk luar.
4. Amati morphological gradient sebagai pendeteksi tepi alternatif berbasis morfologi.
5. Bandingkan opening vs. closing pada gambar yang sama untuk memahami dualitas keduanya.
6. Periksa folder `output/` untuk semua gambar hasil.

---

### Percobaan 17: Top-Hat dan Black-Hat Transform
**File:** `17_tophat_blackhat.py`
**Tujuan:** Menerapkan transformasi Top-Hat (gambar dikurangi opening-nya) untuk mendeteksi objek lebih terang dari latar, dan Black-Hat (closing dikurangi gambar) untuk mendeteksi objek lebih gelap dari latar.

**Langkah Kerja:**
1. Buka file `17_tophat_blackhat.py` di editor.
2. Jalankan skrip dan amati Top-Hat yang menonjolkan detail terang pada latar gelap heterogen.
3. Amati Black-Hat yang menonjolkan detail gelap pada latar terang yang tidak merata.
4. Amati pengaruh ukuran structuring element terhadap ukuran objek yang terdeteksi.
5. Bandingkan Top-Hat dan Black-Hat pada gambar teks, sel darah, atau tekstur kain.
6. Periksa folder `output/` untuk semua gambar hasil.

---

### Percobaan 18: Transformasi Fourier
**File:** `18_transformasi_fourier.py`
**Tujuan:** Menganalisis gambar di domain frekuensi menggunakan Discrete Fourier Transform (DFT) dengan `np.fft.fft2()`, serta memvisualisasikan magnitude spectrum dan phase spectrum.

**Langkah Kerja:**
1. Buka file `18_transformasi_fourier.py` di editor.
2. Jalankan skrip dan amati magnitude spectrum - energi terkonsentrasi di tengah (frekuensi rendah).
3. Amati pengaruh `np.fft.fftshift()` yang memindahkan frekuensi nol ke tengah gambar.
4. Amati perbedaan magnitude spectrum gambar dengan tekstur horizontal, vertikal, dan acak.
5. Bandingkan hasil DFT menggunakan `cv2.dft()` dan `np.fft.fft2()`.
6. Periksa folder `output/` untuk file magnitude spectrum dan phase spectrum.

---

### Percobaan 19: Filter Frekuensi
**File:** `19_filter_frekuensi.py`
**Tujuan:** Menerapkan filter ideal low-pass, high-pass, dan band-pass di domain frekuensi, serta membandingkan dengan filter Butterworth dan Gaussian di domain frekuensi untuk meminimalkan ringing artifacts.

**Langkah Kerja:**
1. Buka file `19_filter_frekuensi.py` di editor.
2. Jalankan skrip dan amati hasil low-pass filter di domain frekuensi - gambar menjadi blur.
3. Amati hasil high-pass filter - hanya tepi dan detail halus yang tersisa.
4. Amati ringing artifacts pada ideal low-pass filter vs. transisi halus pada Gaussian/Butterworth.
5. Bandingkan band-pass filter di domain frekuensi dengan spatial domain filtering setara.
6. Periksa folder `output/` untuk semua gambar hasil filtering.

---

### Percobaan 20: Connected Components
**File:** `20_connected_components.py`
**Tujuan:** Menerapkan connected components labeling menggunakan `cv2.connectedComponentsWithStats()` untuk mengidentifikasi, menghitung, dan menganalisis blob/objek individual pada gambar biner berdasarkan konektivitas piksel.

**Langkah Kerja:**
1. Buka file `20_connected_components.py` di editor.
2. Jalankan skrip pada gambar biner dengan beberapa objek terpisah.
3. Amati output: jumlah komponen, label per piksel, statistik area dan bounding box, serta centroid setiap blob.
4. Amati visualisasi setiap komponen dengan warna berbeda (labeled image berwarna-warni).
5. Bandingkan 4-connectivity vs. 8-connectivity terhadap jumlah komponen yang terdeteksi.
6. Periksa folder `output/` untuk gambar labeled components dan file statistik yang tersimpan.

---

## 5. TUGAS AKHIR

1. **Tugas Integrasi Pipeline:** Buat sebuah program Python yang menggabungkan minimal 5 teknik dari modul ini menjadi sebuah pipeline pemrosesan citra otomatis. Input: gambar warna; output: laporan analisis dengan jumlah objek, ukuran rata-rata, dan visualisasi hasil setiap tahap.

2. **Tugas Komparasi Thresholding:** Lakukan eksperimen sistematis membandingkan thresholding global (manual), Otsu, Triangle, dan adaptive pada 3 gambar berbeda (dokumen teks, foto objek, gambar medis). Buat tabel evaluasi kuantitatif dan analisis kapan setiap metode lebih unggul.

3. **Tugas Analisis Domain Frekuensi:** Buat program yang menampilkan hubungan antara filter spasial (Gaussian blur, sharpening) dengan representasinya di domain frekuensi (LPF, HPF). Tunjukkan bahwa operasi konvolusi di domain spasial setara dengan perkalian di domain frekuensi.

4. **Tugas Morfologi Lanjut:** Implementasikan sistem penghitung objek otomatis pada gambar biner yang menggunakan operasi morfologi untuk membersihkan noise, memisahkan objek yang menempel, dan menganalisis setiap objek menggunakan connected components.

5. **Tugas Refleksi Konseptual:** Tulis esai singkat (500-700 kata) yang menjelaskan trade-off antara filter yang mempertahankan tepi (bilateral, median) vs. filter standar (Gaussian). Sertakan contoh kasus penggunaan nyata di bidang medis, industri, atau otomotif.

---

## 6. KRITERIA PENILAIAN

| Aspek | Bobot | Indikator |
|-------|-------|-----------|
| **Persiapan** | 20 poin | Pemahaman teori sebelum praktikum, persiapan bahan dan lingkungan, kemampuan menjawab pertanyaan pre-test |
| **Pelaksanaan** | 50 poin | Kelengkapan menjalankan 20 percobaan, ketepatan observasi, kemampuan troubleshooting, aktif bertanya dan berdiskusi |
| **Laporan** | 30 poin | Kelengkapan dokumentasi hasil setiap percobaan, kualitas analisis dan diskusi, penyelesaian tugas akhir, kerapian penulisan |

### Detail Penilaian Persiapan (20 poin)
- Instalasi dan konfigurasi lingkungan: 5 poin
- Pemahaman konsep dasar (pre-test singkat): 10 poin
- Kesiapan dataset dan folder struktur: 5 poin

### Detail Penilaian Pelaksanaan (50 poin)
- Percobaan 1-5 (kontur, histogram, CLAHE, HSV, threshold global): 15 poin
- Percobaan 6-11 (Otsu, adaptive, konvolusi, blur, median/bilateral, sharpening): 15 poin
- Percobaan 12-14 (deteksi tepi Sobel, Canny, Laplacian): 10 poin
- Percobaan 15-20 (morfologi, top-hat, Fourier, filter frekuensi, connected components): 10 poin

### Detail Penilaian Laporan (30 poin)
- Dokumentasi hasil 20 percobaan (screenshot dan analisis): 15 poin
- Penyelesaian 5 tugas akhir: 10 poin
- Kerapian, format, dan referensi: 5 poin

---

*Jobsheet ini disusun untuk Praktikum Komputer Vision - Modul 3: Pemrosesan Citra*
