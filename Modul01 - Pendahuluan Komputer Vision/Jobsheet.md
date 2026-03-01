# JOBSHEET PRAKTIKUM
# MODUL 1: PENDAHULUAN KOMPUTER VISION

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa mampu:
1. Memahami konsep dasar Computer Vision dan representasi citra digital.
2. Menginstal dan mengonfigurasi environment Python dengan OpenCV.
3. Melakukan operasi loading, display, dan penyimpanan gambar.
4. Mengakses dan memanipulasi properti serta piksel gambar.
5. Mengkonversi gambar antar ruang warna (BGR, RGB, Grayscale, HSV).
6. Melakukan operasi aritmatika dan blending pada gambar.
7. Menggambar bentuk geometris dan anotasi pada gambar.
8. Memahami Region of Interest (ROI) dan cropping.
9. Melakukan resizing dengan berbagai metode interpolasi.
10. Menyimpan gambar dalam berbagai format dan menganalisis perbedaannya.

---

## 2. ALAT DAN BAHAN

### A. Perangkat Keras
- Laptop/PC (min. Intel Core i3, RAM 4 GB, storage 10 GB free)
- Webcam (opsional)

### B. Perangkat Lunak
- Python 3.8+
- Visual Studio Code atau PyCharm
- Terminal / Command Prompt

### C. Library Python
```bash
pip install opencv-python numpy matplotlib pillow
```

### D. Dataset
- Gambar sample disediakan di folder `data/images/` atau diunduh melalui script `setup_images.py`.

---

## 3. LANGKAH KERJA

### Persiapan Awal
1. Buat folder kerja `Modul01/praktikum/`.
2. Buat virtual environment dan install library.
3. Jalankan `setup_images.py` untuk menyiapkan gambar sample.
4. Verifikasi instalasi dengan mencetak versi OpenCV.

---

### Percobaan 1: Loading dan Membaca Gambar

**Tujuan**: Memahami cara membaca gambar dari file menggunakan `cv2.imread()` dengan berbagai mode.

**Langkah Kerja**:
1. Buat file `01_loading_gambar.py`.
2. Gunakan `cv2.imread()` untuk membaca gambar dalam mode:
   - `cv2.IMREAD_COLOR` — gambar berwarna (BGR, 3 channel).
   - `cv2.IMREAD_GRAYSCALE` — gambar grayscale (1 channel).
   - `cv2.IMREAD_UNCHANGED` — termasuk alpha channel (jika ada).
3. Cetak shape, dtype, dan tipe objek dari masing-masing mode.
4. Tambahkan validasi apakah gambar berhasil dibaca (`img is not None`).
5. Coba baca gambar dengan format berbeda: `.jpg`, `.png`, `.bmp`.
6. Amati dan catat perbedaan shape untuk setiap mode loading.

**Output yang Diharapkan**:
- Console menampilkan shape, dtype, dan status loading untuk setiap mode.

---

### Percobaan 2: Menampilkan Gambar (OpenCV & Matplotlib)

**Tujuan**: Menampilkan gambar menggunakan dua metode berbeda dan memahami perbedaan format warna.

**Langkah Kerja**:
1. Buat file `02_menampilkan_gambar.py`.
2. Baca sebuah gambar berwarna dengan `cv2.imread()`.
3. Tampilkan dengan `cv2.imshow()` — amati warna yang benar.
4. Tampilkan dengan `matplotlib.pyplot.imshow()` — amati warna yang terbalik (BGR vs RGB).
5. Konversi BGR → RGB menggunakan `cv2.cvtColor()`, lalu tampilkan kembali di Matplotlib.
6. Tampilkan gambar grayscale di kedua library.
7. Gunakan `cv2.waitKey(0)` dan `cv2.destroyAllWindows()` untuk window OpenCV.

**Output yang Diharapkan**:
- Window OpenCV menampilkan gambar dengan warna benar.
- Plot Matplotlib menampilkan perbandingan BGR vs RGB.

---

### Percobaan 3: Properti dan Metadata Gambar

**Tujuan**: Mengekstrak dan memahami properti-properti penting dari gambar digital.

**Langkah Kerja**:
1. Buat file `03_properti_gambar.py`.
2. Baca gambar berwarna dan gambar grayscale.
3. Ekstrak dan cetak properti berikut:
   - `img.shape` — dimensi gambar (height, width, channels).
   - `img.dtype` — tipe data (uint8, float32, dll.).
   - `img.size` — total jumlah elemen.
   - `img.ndim` — jumlah dimensi array.
   - Hitung total piksel: `height × width`.
   - Hitung ukuran memori: `img.nbytes`.
4. Hitung nilai statistik: min, max, mean, std dari intensitas piksel.
5. Bandingkan properti gambar berwarna vs grayscale.
6. Coba dengan 3 gambar berbeda ukuran dan format.
7. Buat tabel ringkasan properti.

**Output yang Diharapkan**:
- Console menampilkan tabel properti lengkap untuk setiap gambar.

---

### Percobaan 4: Konversi Ruang Warna

**Tujuan**: Mengkonversi gambar antar ruang warna dan memahami karakteristik masing-masing.

**Langkah Kerja**:
1. Buat file `04_konversi_warna.py`.
2. Baca gambar berwarna (BGR).
3. Konversi ke ruang warna berikut menggunakan `cv2.cvtColor()`:
   - BGR → RGB
   - BGR → Grayscale
   - BGR → HSV
   - BGR → Lab
   - BGR → YCrCb
4. Tampilkan setiap channel secara terpisah (misal: H, S, V dari HSV).
5. Buat visualisasi grid yang menampilkan semua ruang warna side-by-side.
6. Buat mask warna menggunakan range HSV dengan `cv2.inRange()`.
7. Analisis histogram untuk setiap ruang warna menggunakan `cv2.calcHist()`.
8. Simpan visualisasi ke folder output.

**Output yang Diharapkan**:
- Gambar grid perbandingan ruang warna.
- Histogram distribusi intensitas per channel untuk setiap ruang warna.

---

### Percobaan 5: Akses dan Manipulasi Piksel

**Tujuan**: Memahami cara mengakses, membaca, dan memodifikasi nilai piksel secara langsung.

**Langkah Kerja**:
1. Buat file `05_manipulasi_piksel.py`.
2. Baca gambar berwarna.
3. Akses piksel individual: `img[100, 200]` → cetak nilai BGR.
4. Modifikasi piksel tunggal: `img[100, 200] = [0, 0, 255]` (set ke merah).
5. Modifikasi region piksel (blok 50×50) dengan warna solid.
6. Salin region piksel ke lokasi lain (manual copy).
7. Buat efek checkerboard dengan manipulasi piksel.
8. Bandingkan kecepatan akses piksel individual vs slicing NumPy:
   - Loop Python pixel-by-pixel.
   - Operasi NumPy vectorized.
9. Buat gambar sintetis (gradient horizontal, gradient vertikal).
10. Simpan semua hasil ke folder output.

**Output yang Diharapkan**:
- Gambar dengan region yang dimodifikasi.
- Pola checkerboard dan gradient buatan.
- Perbandingan waktu eksekusi akses piksel.

---

### Percobaan 6: Operasi Aritmatika Gambar

**Tujuan**: Melakukan operasi aritmatika pada gambar (penjumlahan, pengurangan, blending).

**Langkah Kerja**:
1. Buat file `06_operasi_aritmatika.py`.
2. Baca dua gambar dengan ukuran sama (resize jika perlu).
3. Lakukan penjumlahan: `cv2.add(img1, img2)` — bandingkan dengan `img1 + img2` (NumPy).
4. Jelaskan perbedaan saturated addition (OpenCV) vs modulo addition (NumPy).
5. Lakukan pengurangan: `cv2.subtract(img1, img2)`.
6. Lakukan blending: `cv2.addWeighted(img1, alpha, img2, beta, gamma)`.
7. Variasikan nilai alpha dari 0.0 hingga 1.0 (buat animasi frame).
8. Buat efek brightness adjustment: tambahkan/kurangi nilai konstan ke semua piksel.
9. Buat efek kontras: kalikan gambar dengan faktor konstan.
10. Tampilkan hasil semua operasi dalam satu figure grid.

**Output yang Diharapkan**:
- Grid perbandingan operasi aritmatika.
- Sequence blending dengan berbagai alpha.

---

### Percobaan 7: Menggambar Bentuk dan Anotasi

**Tujuan**: Menggambar bentuk geometris dan menambahkan teks anotasi pada gambar.

**Langkah Kerja**:
1. Buat file `07_menggambar_shapes.py`.
2. Buat canvas kosong (gambar hitam) menggunakan `np.zeros()`.
3. Gambar bentuk-bentuk berikut:
   - Garis: `cv2.line()` — berbagai warna dan ketebalan.
   - Persegi panjang: `cv2.rectangle()` — filled dan outline.
   - Lingkaran: `cv2.circle()` — filled dan outline.
   - Elips: `cv2.ellipse()` — dengan berbagai sudut.
   - Poligon: `cv2.polylines()` dan `cv2.fillPoly()`.
   - Panah: `cv2.arrowedLine()`.
4. Tambahkan teks dengan `cv2.putText()`:
   - Coba berbagai font (`FONT_HERSHEY_SIMPLEX`, `FONT_HERSHEY_COMPLEX`, dll.).
   - Variasikan ukuran, warna, dan ketebalan.
5. Gambar di atas gambar foto (bukan hanya canvas kosong).
6. Buat anotasi bounding box pada objek tertentu di foto.
7. Simpan hasil ke folder output.

**Output yang Diharapkan**:
- Canvas dengan berbagai bentuk geometris.
- Foto dengan anotasi bounding box dan teks.

---

### Percobaan 8: Region of Interest (ROI)

**Tujuan**: Memahami konsep ROI untuk memproses area tertentu dari gambar.

**Langkah Kerja**:
1. Buat file `08_region_of_interest.py`.
2. Baca gambar berwarna.
3. Definisikan ROI menggunakan slicing: `roi = img[y1:y2, x1:x2]`.
4. Tampilkan ROI sebagai gambar terpisah.
5. Salin ROI ke lokasi lain dalam gambar yang sama.
6. Terapkan operasi hanya pada ROI:
   - Konversi ROI ke grayscale dan kembalikan ke gambar asli.
   - Terapkan blur hanya pada ROI.
   - Ubah brightness hanya pada ROI.
7. Buat efek "sensor" (blur area wajah atau teks sensitif).
8. Buat efek "spotlight" (hanya ROI terlihat, sisanya gelap).
9. Implementasikan simple masking menggunakan ROI.
10. Simpan semua hasil ke folder output.

**Output yang Diharapkan**:
- Gambar dengan berbagai efek ROI (blur, spotlight, copy).
- Demonstrasi masking sederhana.

---

### Percobaan 9: Resizing dan Transformasi Dasar

**Tujuan**: Melakukan resize gambar dengan berbagai metode interpolasi dan transformasi geometri dasar.

**Langkah Kerja**:
1. Buat file `09_resizing_transformasi.py`.
2. Baca gambar dan resize dengan:
   - Ukuran absolut: `cv2.resize(img, (width, height))`.
   - Faktor skala: `cv2.resize(img, None, fx=0.5, fy=0.5)`.
3. Bandingkan metode interpolasi:
   - `cv2.INTER_NEAREST` — nearest neighbor.
   - `cv2.INTER_LINEAR` — bilinear (default).
   - `cv2.INTER_CUBIC` — bicubic.
   - `cv2.INTER_AREA` — untuk downscaling.
   - `cv2.INTER_LANCZOS4` — Lanczos.
4. Perbesar gambar 4× dan bandingkan kualitas visual tiap interpolasi.
5. Lakukan flipping:
   - `cv2.flip(img, 0)` — vertikal.
   - `cv2.flip(img, 1)` — horizontal.
   - `cv2.flip(img, -1)` — both.
6. Lakukan rotasi 90° menggunakan `cv2.rotate()`.
7. Pertahankan aspect ratio saat resize.
8. Buat thumbnail dari gambar besar.
9. Tampilkan perbandingan semua metode dalam grid.
10. Simpan hasil ke folder output.

**Output yang Diharapkan**:
- Grid perbandingan kualitas interpolasi.
- Gambar yang di-flip dan dirotasi.

---

### Percobaan 10: Menyimpan dan Mengekspor Gambar

**Tujuan**: Menyimpan gambar dalam berbagai format dan menganalisis trade-off ukuran vs kualitas.

**Langkah Kerja**:
1. Buat file `10_menyimpan_gambar.py`.
2. Baca gambar berwarna.
3. Simpan dalam berbagai format:
   - JPEG dengan kualitas berbeda: `cv2.imwrite('out.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])` (q = 10, 30, 50, 70, 90, 100).
   - PNG dengan kompresi berbeda: `cv2.IMWRITE_PNG_COMPRESSION` (0–9).
   - BMP (tanpa kompresi).
   - TIFF.
4. Catat ukuran file untuk setiap konfigurasi.
5. Hitung rasio kompresi: `ukuran_original / ukuran_compressed`.
6. Hitung PSNR (Peak Signal-to-Noise Ratio) antara gambar asli dan hasil kompresi JPEG.
7. Buat grafik: kualitas JPEG vs ukuran file vs PSNR.
8. Simpan gambar grayscale dan bandingkan ukuran dengan versi berwarna.
9. Buat batch export: simpan 1 gambar ke semua format sekaligus.
10. Buat laporan ringkasan perbandingan format.

**Output yang Diharapkan**:
- File gambar dalam berbagai format dan kualitas.
- Grafik perbandingan ukuran vs kualitas.
- Tabel ringkasan kompresi.

---

## 4. ANALISIS

### Analisis Percobaan 1 — Loading Gambar
- Bandingkan shape gambar yang dibaca dalam mode COLOR vs GRAYSCALE vs UNCHANGED.
- Jelaskan mengapa mode UNCHANGED penting untuk gambar PNG dengan transparansi.
- Diskusikan validasi input dan error handling saat file tidak ditemukan.

### Analisis Percobaan 2 — Menampilkan Gambar
- Jelaskan mengapa warna tertukar saat gambar BGR ditampilkan di Matplotlib.
- Diskusikan kelebihan dan kekurangan OpenCV window vs Matplotlib plot.
- Kapan sebaiknya menggunakan masing-masing metode display?

### Analisis Percobaan 3 — Properti Gambar
- Hitung memori yang dibutuhkan untuk menyimpan gambar di RAM.
- Jelaskan hubungan antara resolusi, channel, dan ukuran memori.
- Mengapa dtype uint8 (0–255) paling umum digunakan?

### Analisis Percobaan 4 — Konversi Warna
- Bandingkan histogram RGB vs HSV vs Lab untuk gambar yang sama.
- Jelaskan mengapa HSV lebih cocok untuk deteksi warna dibanding RGB.
- Dalam kasus apa Lab color space lebih unggul?

### Analisis Percobaan 5 — Manipulasi Piksel
- Hitung speedup yang diperoleh dari operasi vectorized NumPy vs loop Python.
- Diskusikan konsep cache-friendly access pattern pada array multidimensi.
- Mengapa slicing lebih efisien daripada nested loop?

### Analisis Percobaan 6 — Operasi Aritmatika
- Jelaskan perbedaan saturated vs modulo arithmetic dengan contoh numerik.
- Analisis efek alpha blending terhadap visibilitas gambar.
- Bagaimana brightness dan contrast mempengaruhi histogram gambar?

### Analisis Percobaan 7 — Menggambar Shapes
- Diskusikan anti-aliasing pada garis dan bentuk (`cv2.LINE_AA`).
- Jelaskan sistem koordinat yang digunakan (origin di kiri atas).
- Bagaimana anotasi visual berguna dalam aplikasi computer vision?

### Analisis Percobaan 8 — Region of Interest
- Diskusikan efisiensi memori saat bekerja dengan ROI (view vs copy).
- Jelaskan aplikasi ROI dalam object tracking dan censoring.
- Bandingkan pendekatan ROI rectangular vs masking non-rectangular.

### Analisis Percobaan 9 — Resizing
- Bandingkan kualitas visual dari setiap metode interpolasi.
- Jelaskan kapan nearest neighbor cocok dan kapan tidak.
- Mengapa INTER_AREA direkomendasikan untuk downscaling?

### Analisis Percobaan 10 — Menyimpan Gambar
- Analisis grafik JPEG quality vs file size vs PSNR.
- Tentukan sweet spot kualitas JPEG untuk web publishing.
- Jelaskan perbedaan lossy (JPEG) vs lossless (PNG) compression.

---

## 5. KESIMPULAN

Setelah menyelesaikan seluruh percobaan, buatlah kesimpulan yang mencakup:

1. **Konsep dasar** representasi gambar digital sebagai array NumPy dan implikasinya terhadap pemrosesan.
2. **Ruang warna** yang dipelajari (BGR, RGB, Grayscale, HSV, Lab) beserta kegunaan masing-masing.
3. **Operasi dasar** yang dikuasai: loading, display, akses piksel, aritmatika, drawing, ROI, resize, save.
4. **Trade-off** antara kualitas dan efisiensi (kompresi gambar, metode interpolasi, akses piksel).
5. **Relevansi** operasi dasar ini sebagai fondasi untuk modul-modul selanjutnya.

Format penulisan:
- Berdasarkan data dan pengamatan dari percobaan.
- Menggunakan bahasa ilmiah dan terstruktur.
- Mencantumkan insight atau temuan yang tidak terduga.
- Menghubungkan dengan aplikasi dunia nyata.
