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
6. Melakukan operasi aritmatika dan bitwise pada gambar.
7. Menggambar bentuk geometris dan menulis teks anotasi pada gambar.
8. Memahami Region of Interest (ROI), cropping, dan masking.
9. Melakukan resize, rotasi, flip, padding, dan transformasi geometri dasar.
10. Memahami splitting/merging channel, blending, brightness/contrast, dan histogram.
11. Menyimpan gambar dalam berbagai format dan menganalisis perbedaannya.

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
- Gambar sample disediakan di folder `praktikum/image/` atau diunduh melalui script `download_image.py`.

---

## 3. LANGKAH KERJA

### Persiapan Awal
1. Buka folder `Modul01 - Pendahuluan Komputer Vision/praktikum/`.
2. Buat virtual environment dan install library (`pip install -r requirements.txt`).
3. Jalankan `python download_image.py` untuk mengunduh gambar sample ke folder `image/`.
4. Verifikasi instalasi dengan mencetak versi OpenCV.
5. Pastikan folder `output/` tersedia untuk menyimpan hasil percobaan.

---

### Percobaan 1: Loading dan Menampilkan Gambar
**File**: `01_loading_dan_menampilkan_gambar.py`

**Tujuan**: Memahami cara membaca dan menampilkan gambar menggunakan OpenCV dan Matplotlib.

**Langkah Kerja**:
1. Gunakan `cv2.imread()` untuk membaca gambar dalam mode `IMREAD_COLOR`, `IMREAD_GRAYSCALE`, dan `IMREAD_UNCHANGED`.
2. Cetak `shape`, `dtype`, dan status loading untuk setiap mode.
3. Tampilkan gambar menggunakan `matplotlib.pyplot.imshow()`.
4. Konversi BGR → RGB untuk tampilan Matplotlib yang benar.
5. Tampilkan gambar grayscale dan bandingkan dengan versi berwarna.
6. Simpan visualisasi perbandingan ke folder `output/`.

**Output yang Diharapkan**: Gambar perbandingan mode loading dan konversi BGR→RGB.

---

### Percobaan 2: Properti Gambar
**File**: `02_properti_gambar.py`

**Tujuan**: Mengekstrak dan memahami properti-properti penting dari gambar digital.

**Langkah Kerja**:
1. Baca gambar berwarna dan grayscale.
2. Ekstrak properti: `shape`, `dtype`, `size`, `ndim`, `nbytes`.
3. Hitung statistik piksel: min, max, mean, std.
4. Bandingkan properti gambar berwarna vs grayscale.
5. Analisis 3 gambar berbeda ukuran dan format.
6. Buat tabel ringkasan properti.

**Output yang Diharapkan**: Tabel properti lengkap dan perbandingan statistik.

---

### Percobaan 3: Konversi Ruang Warna
**File**: `03_konversi_ruang_warna.py`

**Tujuan**: Mengkonversi gambar antar ruang warna dan memahami karakteristiknya.

**Langkah Kerja**:
1. Konversi gambar BGR ke: RGB, Grayscale, HSV, Lab, YCrCb.
2. Tampilkan setiap channel secara terpisah (H, S, V dari HSV, dll.).
3. Buat visualisasi grid semua ruang warna.
4. Buat mask warna menggunakan `cv2.inRange()` pada HSV.
5. Analisis histogram untuk setiap ruang warna dengan `cv2.calcHist()`.

**Output yang Diharapkan**: Grid perbandingan ruang warna dan histogram per channel.

---

### Percobaan 4: Akses dan Manipulasi Piksel
**File**: `04_akses_manipulasi_piksel.py`

**Tujuan**: Memahami cara mengakses, membaca, dan memodifikasi nilai piksel.

**Langkah Kerja**:
1. Akses piksel individual: `img[y, x]` → cetak nilai BGR.
2. Modifikasi piksel tunggal dan region piksel.
3. Salin region piksel ke lokasi lain.
4. Buat efek checkerboard dan gradient.
5. Bandingkan kecepatan loop Python vs slicing NumPy (vectorized).

**Output yang Diharapkan**: Gambar yang dimodifikasi, pola sintetis, perbandingan waktu eksekusi.

---

### Percobaan 5: Operasi Aritmatika Gambar
**File**: `05_operasi_aritmatika_gambar.py`

**Tujuan**: Melakukan operasi aritmatika (penjumlahan, pengurangan, blending) pada gambar.

**Langkah Kerja**:
1. Lakukan penjumlahan `cv2.add()` vs `+` NumPy (saturated vs modulo).
2. Lakukan pengurangan `cv2.subtract()`.
3. Lakukan blending `cv2.addWeighted()` dengan variasi alpha.
4. Buat efek brightness dan contrast adjustment.
5. Tampilkan semua operasi dalam grid perbandingan.

**Output yang Diharapkan**: Grid operasi aritmatika dan sequence blending.

---

### Percobaan 6: Operasi Bitwise
**File**: `06_operasi_bitwise.py`

**Tujuan**: Memahami operasi bitwise (AND, OR, XOR, NOT) pada gambar.

**Langkah Kerja**:
1. Buat dua gambar biner sederhana (kotak dan lingkaran).
2. Terapkan `cv2.bitwise_and()`, `cv2.bitwise_or()`, `cv2.bitwise_xor()`, `cv2.bitwise_not()`.
3. Terapkan operasi bitwise dengan mask pada gambar nyata.
4. Buat efek logo overlay menggunakan bitwise operations.
5. Visualisasikan hasil semua operasi.

**Output yang Diharapkan**: Grid perbandingan operasi bitwise dan efek logo overlay.

---

### Percobaan 7: Menggambar Bentuk Geometris
**File**: `07_menggambar_bentuk_geometris.py`

**Tujuan**: Menggambar berbagai bentuk geometris pada gambar.

**Langkah Kerja**:
1. Buat canvas kosong dengan `np.zeros()`.
2. Gambar: garis (`cv2.line()`), persegi (`cv2.rectangle()`), lingkaran (`cv2.circle()`), elips (`cv2.ellipse()`), poligon (`cv2.polylines()`, `cv2.fillPoly()`), panah (`cv2.arrowedLine()`).
3. Variasikan warna, ketebalan, dan opsi filled/outline.
4. Gambar bentuk di atas gambar foto.
5. Buat anotasi bounding box pada objek.

**Output yang Diharapkan**: Canvas dengan bentuk geometris dan foto beranotasi.

---

### Percobaan 8: Menulis Teks pada Gambar
**File**: `08_menulis_teks_pada_gambar.py`

**Tujuan**: Menambahkan teks anotasi pada gambar dengan berbagai konfigurasi.

**Langkah Kerja**:
1. Gunakan `cv2.putText()` dengan berbagai parameter.
2. Coba semua font OpenCV (`FONT_HERSHEY_SIMPLEX`, `FONT_HERSHEY_COMPLEX`, dll.).
3. Variasikan ukuran (`fontScale`), warna, dan ketebalan.
4. Gunakan `cv2.getTextSize()` untuk menghitung ukuran teks.
5. Buat teks dengan background (bounding box teks).

**Output yang Diharapkan**: Galeri font dan demonstrasi text rendering.

---

### Percobaan 9: Region of Interest (ROI)
**File**: `09_region_of_interest.py`

**Tujuan**: Memahami konsep ROI untuk memproses area tertentu dari gambar.

**Langkah Kerja**:
1. Definisikan ROI menggunakan slicing: `roi = img[y1:y2, x1:x2]`.
2. Tampilkan dan salin ROI ke lokasi lain.
3. Terapkan operasi hanya pada ROI (grayscale, blur, brightness).
4. Buat efek "sensor" (blur area sensitif) dan "spotlight".
5. Implementasikan simple masking menggunakan ROI.

**Output yang Diharapkan**: Gambar dengan efek ROI (blur, spotlight, copy, masking).

---

### Percobaan 10: Resize dan Scaling
**File**: `10_resize_dan_scaling.py`

**Tujuan**: Melakukan resize gambar dengan berbagai metode interpolasi.

**Langkah Kerja**:
1. Resize dengan ukuran absolut dan faktor skala (`fx`, `fy`).
2. Bandingkan metode interpolasi: `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC`, `INTER_AREA`, `INTER_LANCZOS4`.
3. Perbesar gambar 4× dan bandingkan kualitas visual.
4. Pertahankan aspect ratio saat resize.
5. Buat thumbnail dari gambar besar.

**Output yang Diharapkan**: Grid perbandingan kualitas interpolasi dan thumbnail.

---

### Percobaan 11: Cropping Gambar
**File**: `11_cropping_gambar.py`

**Tujuan**: Memahami teknik cropping gambar menggunakan array slicing.

**Langkah Kerja**:
1. Crop gambar dengan slicing NumPy: `cropped = img[y:y+h, x:x+w]`.
2. Crop area tengah, kiri-atas, kanan-bawah.
3. Buat center crop dengan berbagai rasio.
4. Buat grid of crops dari satu gambar besar.
5. Crop dan resize untuk standarisasi ukuran.

**Output yang Diharapkan**: Berbagai variasi crop dan grid hasil cropping.

---

### Percobaan 12: Rotasi Gambar
**File**: `12_rotasi_gambar.py`

**Tujuan**: Melakukan rotasi gambar dengan berbagai sudut dan metode.

**Langkah Kerja**:
1. Rotasi 90°/180°/270° menggunakan `cv2.rotate()`.
2. Rotasi sudut bebas menggunakan `cv2.getRotationMatrix2D()` dan `cv2.warpAffine()`.
3. Rotasi dengan penyesuaian ukuran canvas agar gambar tidak terpotong.
4. Rotasi di sekitar titik pusat kustom.
5. Visualisasi rotasi bertahap (animasi frame).

**Output yang Diharapkan**: Gambar dengan berbagai sudut rotasi dan perbandingan metode.

---

### Percobaan 13: Flip Gambar
**File**: `13_flip_gambar.py`

**Tujuan**: Memahami efek flipping (pencerminan) pada gambar.

**Langkah Kerja**:
1. Flip horizontal (`flipCode=1`), vertikal (`flipCode=0`), dan both (`flipCode=-1`).
2. Implementasi flip manual menggunakan NumPy slicing.
3. Buat efek mirror (separuh gambar + flip).
4. Kombinasikan flip dengan operasi lain.
5. Visualisasi semua variasi flip.

**Output yang Diharapkan**: Grid perbandingan semua jenis flip dan efek mirror.

---

### Percobaan 14: Padding dan Border Gambar
**File**: `14_padding_border_gambar.py`

**Tujuan**: Menambahkan padding/border pada gambar dengan berbagai metode.

**Langkah Kerja**:
1. Gunakan `cv2.copyMakeBorder()` dengan tipe: `BORDER_CONSTANT`, `BORDER_REPLICATE`, `BORDER_REFLECT`, `BORDER_WRAP`.
2. Buat border dekoratif dengan warna kustom.
3. Buat frame foto dengan border bertingkat.
4. Tambahkan padding untuk square padding.
5. Bandingkan efek visual setiap metode border.

**Output yang Diharapkan**: Perbandingan semua tipe border dan efek frame.

---

### Percobaan 15: Splitting dan Merging Channel
**File**: `15_splitting_merging_channel.py`

**Tujuan**: Memisahkan dan menggabungkan kembali channel warna gambar.

**Langkah Kerja**:
1. Split channel BGR menggunakan `cv2.split()`.
2. Tampilkan setiap channel sebagai gambar grayscale.
3. Merge kembali channel menggunakan `cv2.merge()`.
4. Tukar urutan channel (BGR→RGB, swap R-B).
5. Buat efek warna dan false-color dari channel tertentu.

**Output yang Diharapkan**: Visualisasi channel individual dan efek modifikasi channel.

---

### Percobaan 16: Blending Dua Gambar
**File**: `16_blending_dua_gambar.py`

**Tujuan**: Menggabungkan dua gambar dengan alpha blending dan teknik lain.

**Langkah Kerja**:
1. Gunakan `cv2.addWeighted()` dengan variasi alpha (0.0 s.d. 1.0).
2. Buat efek transisi (fade) antar dua gambar.
3. Buat blending berdasarkan mask (region-specific blending).
4. Implementasi seamless blending sederhana.
5. Visualisasi sequence alpha blending.

**Output yang Diharapkan**: Sequence blending dan overlay berbasis mask.

---

### Percobaan 17: Brightness dan Contrast
**File**: `17_brightness_dan_contrast.py`

**Tujuan**: Mengatur brightness dan contrast gambar secara manual dan otomatis.

**Langkah Kerja**:
1. Atur brightness dengan menambah/kurangi nilai konstan.
2. Atur contrast dengan mengalikan faktor (`new = alpha * img + beta`).
3. Gunakan `cv2.convertScaleAbs()`.
4. Implementasi auto-contrast menggunakan normalisasi.
5. Bandingkan histogram sebelum dan setelah adjustment.

**Output yang Diharapkan**: Grid perbandingan brightness/contrast dan histogram.

---

### Percobaan 18: Histogram Gambar
**File**: `18_histogram_gambar.py`

**Tujuan**: Menghitung dan memvisualisasikan histogram distribusi intensitas.

**Langkah Kerja**:
1. Hitung histogram menggunakan `cv2.calcHist()`.
2. Tampilkan histogram grayscale dan per-channel (B, G, R).
3. Hitung histogram pada ROI tertentu menggunakan mask.
4. Bandingkan histogram gambar terang, gelap, dan normal.
5. Implementasi histogram equalization (`cv2.equalizeHist()`).

**Output yang Diharapkan**: Plot histogram dan perbandingan distribusi intensitas.

---

### Percobaan 19: Masking Gambar
**File**: `19_masking_gambar.py`

**Tujuan**: Memahami penggunaan mask untuk operasi selektif pada gambar.

**Langkah Kerja**:
1. Buat mask biner (hitam-putih) untuk area tertentu.
2. Terapkan mask dengan `cv2.bitwise_and()`.
3. Buat mask berbentuk lingkaran, elips, dan poligon.
4. Buat mask dari thresholding dan deteksi warna (HSV inRange).
5. Kombinasikan beberapa mask dengan operasi bitwise.

**Output yang Diharapkan**: Hasil masking dengan berbagai bentuk dan metode.

---

### Percobaan 20: Menyimpan Berbagai Format
**File**: `20_menyimpan_berbagai_format.py`

**Tujuan**: Menyimpan gambar dalam berbagai format dan menganalisis trade-off ukuran vs kualitas.

**Langkah Kerja**:
1. Simpan dalam format JPEG (kualitas 10–100), PNG (kompresi 0–9), BMP, TIFF.
2. Catat ukuran file dan hitung rasio kompresi.
3. Hitung PSNR antara gambar asli dan hasil kompresi JPEG.
4. Buat grafik kualitas JPEG vs ukuran file vs PSNR.
5. Bandingkan ukuran file berwarna vs grayscale.
6. Buat laporan ringkasan perbandingan format.

**Output yang Diharapkan**: Grafik perbandingan format dan tabel ringkasan kompresi.

---

## 4. ANALISIS

### Analisis Percobaan 1–3: Loading, Properti, dan Ruang Warna
- Bandingkan shape gambar dalam mode COLOR vs GRAYSCALE vs UNCHANGED.
- Jelaskan hubungan antara resolusi, channel, dan ukuran memori.
- Mengapa HSV lebih cocok untuk deteksi warna dibanding RGB?

### Analisis Percobaan 4–5: Manipulasi Piksel dan Aritmatika
- Hitung speedup operasi vectorized NumPy vs loop Python.
- Jelaskan perbedaan saturated addition (OpenCV) vs modulo addition (NumPy).
- Bagaimana brightness dan contrast mempengaruhi histogram?

### Analisis Percobaan 6–8: Bitwise, Drawing, dan Teks
- Diskusikan penggunaan operasi bitwise untuk masking dan overlay.
- Jelaskan anti-aliasing pada garis dan bentuk (`cv2.LINE_AA`).
- Bagaimana anotasi visual digunakan dalam aplikasi computer vision?

### Analisis Percobaan 9–11: ROI, Resize, dan Cropping
- Diskusikan efisiensi memori ROI (view vs copy).
- Bandingkan kualitas visual setiap metode interpolasi saat enlarge.
- Mengapa `INTER_AREA` direkomendasikan untuk downscaling?

### Analisis Percobaan 12–14: Rotasi, Flip, dan Padding
- Jelaskan perbedaan rotasi 90° (`cv2.rotate`) vs sudut bebas (`warpAffine`).
- Kapan flip horizontal digunakan dalam augmentasi data?
- Bandingkan efek visual setiap tipe border pada `copyMakeBorder()`.

### Analisis Percobaan 15–16: Channel dan Blending
- Jelaskan informasi yang terkandung dalam setiap channel warna.
- Analisis efek alpha blending terhadap visibilitas gambar.
- Bagaimana mask-based blending berbeda dari global blending?

### Analisis Percobaan 17–18: Brightness/Contrast dan Histogram
- Analisis perubahan histogram setelah brightness/contrast adjustment.
- Jelaskan manfaat histogram equalization pada gambar low-contrast.

### Analisis Percobaan 19–20: Masking dan Format Penyimpanan
- Diskusikan kegunaan color-based masking menggunakan HSV.
- Analisis grafik JPEG quality vs file size vs PSNR.
- Tentukan sweet spot kualitas JPEG untuk web publishing.

---

## 5. KESIMPULAN

Setelah menyelesaikan seluruh 20 percobaan, buatlah kesimpulan yang mencakup:

1. **Konsep dasar** representasi gambar digital sebagai array NumPy dan implikasinya terhadap pemrosesan.
2. **Ruang warna** yang dipelajari (BGR, RGB, Grayscale, HSV, Lab, YCrCb) beserta kegunaan masing-masing.
3. **Operasi dasar** yang dikuasai: loading, display, akses piksel, aritmatika, bitwise, drawing, teks, ROI, resize, crop, rotasi, flip, padding, channel operations, blending, brightness/contrast, histogram, masking, saving.
4. **Trade-off** antara kualitas dan efisiensi (kompresi gambar, metode interpolasi, akses piksel).
5. **Relevansi** operasi dasar ini sebagai fondasi untuk modul-modul selanjutnya.

Format penulisan:
- Berdasarkan data dan pengamatan dari percobaan.
- Menggunakan bahasa ilmiah dan terstruktur.
- Mencantumkan insight atau temuan yang tidak terduga.
- Menghubungkan dengan aplikasi dunia nyata.
