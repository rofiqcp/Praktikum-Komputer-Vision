# JOBSHEET PRAKTIKUM
# MODUL 3: PEMROSESAN CITRA (IMAGE PROCESSING)

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa mampu:
1. Melakukan penyesuaian brightness dan contrast gambar.
2. Menerapkan histogram equalization dan CLAHE untuk perbaikan kontras.
3. Menerapkan gamma correction untuk koreksi pencahayaan.
4. Melakukan thresholding global, Otsu/Triangle, dan adaptif.
5. Memahami prinsip konvolusi dan filter2D.
6. Menerapkan Gaussian blur, median filter, dan bilateral filter.
7. Melakukan sharpening gambar menggunakan berbagai kernel.
8. Mendeteksi tepi dengan Sobel, Canny, dan Laplacian.
9. Melakukan operasi morfologi dasar dan lanjut.
10. Menerapkan Top Hat dan Black Hat transform.
11. Menganalisis gambar di domain frekuensi (Fourier).
12. Menerapkan filter frekuensi (low-pass, high-pass, band-pass).
13. Melakukan alpha blending dan compositing gambar.

---

## 2. ALAT DAN BAHAN

### A. Perangkat Keras
- Laptop/PC (min. Intel Core i3, RAM 4 GB)

### B. Perangkat Lunak
- Python 3.8+, VS Code
- Library: `opencv-python`, `numpy`, `matplotlib`, `scipy`

### C. Dataset
- Jalankan `download_image.py` di folder `praktikum/` untuk menyiapkan gambar sample.
- Gambar akan tersimpan di folder `praktikum/image/`.

---

## 3. PERSIAPAN

1. Masuk ke folder `praktikum/`.
2. Jalankan: `python download_image.py`
3. Pastikan folder `image/` berisi gambar sample yang dibutuhkan.
4. Folder `output/` akan dibuat otomatis oleh setiap program.

---

## 4. LANGKAH KERJA

### Percobaan 1: Brightness dan Contrast
**File**: `01_brightness_dan_contrast.py`

**Tujuan**: Mengubah brightness dan kontras gambar menggunakan operasi titik: $g(x,y) = \alpha \cdot f(x,y) + \beta$.

**Langkah Kerja**:
1. Buka dan pelajari file `01_brightness_dan_contrast.py`.
2. Jalankan program: `python 01_brightness_dan_contrast.py`.
3. Amati efek variasi α (contrast) dan β (brightness).
4. Perhatikan efek saturasi (clipping di 0 dan 255).
5. Amati perubahan histogram sebelum dan sesudah adjustment.
6. Periksa output di folder `output/`.

---

### Percobaan 2: Histogram Equalization
**File**: `02_histogram_equalization.py`

**Tujuan**: Meningkatkan kontras gambar melalui pemerataan histogram menggunakan `cv2.equalizeHist`.

**Langkah Kerja**:
1. Buka dan pelajari file `02_histogram_equalization.py`.
2. Jalankan program: `python 02_histogram_equalization.py`.
3. Amati gambar low-contrast sebelum dan sesudah equalization.
4. Perhatikan histogram yang menjadi lebih merata.
5. Amati equalization pada gambar berwarna (via YCrCb/HSV).
6. Periksa output di folder `output/`.

---

### Percobaan 3: CLAHE
**File**: `03_clahe.py`

**Tujuan**: Menerapkan Contrast Limited Adaptive Histogram Equalization (CLAHE) yang lebih unggul dari global equalization.

**Langkah Kerja**:
1. Buka dan pelajari file `03_clahe.py`.
2. Jalankan program: `python 03_clahe.py`.
3. Amati efek variasi clipLimit (1.0, 2.0, 4.0, 8.0).
4. Perhatikan efek variasi tileGridSize.
5. Bandingkan CLAHE vs equalizeHist secara visual.
6. Periksa output di folder `output/`.

---

### Percobaan 4: Gamma Correction
**File**: `04_gamma_correction.py`

**Tujuan**: Menerapkan koreksi gamma: $I_{out} = (I_{in}/255)^{\gamma} \times 255$ menggunakan LUT.

**Langkah Kerja**:
1. Buka dan pelajari file `04_gamma_correction.py`.
2. Jalankan program: `python 04_gamma_correction.py`.
3. Amati efek berbagai nilai gamma (0.3, 0.5, 1.0, 1.5, 2.0, 3.0).
4. Perhatikan γ < 1 mencerahkan dan γ > 1 menggelapkan.
5. Amati perbandingan histogram sebelum dan sesudah gamma.
6. Periksa output di folder `output/`.

---

### Percobaan 5: Thresholding Global
**File**: `05_thresholding_global.py`

**Tujuan**: Mengkonversi gambar ke biner menggunakan threshold manual (binary, binary inverse, truncate, tozero).

**Langkah Kerja**:
1. Buka dan pelajari file `05_thresholding_global.py`.
2. Jalankan program: `python 05_thresholding_global.py`.
3. Amati hasil thresholding dengan nilai T yang berbeda (100, 127, 150, 200).
4. Perhatikan perbedaan tipe threshold (BINARY, BINARY_INV, TRUNC, TOZERO).
5. Amati histogram dengan garis threshold.
6. Periksa output di folder `output/`.

---

### Percobaan 6: Thresholding Otsu dan Triangle
**File**: `06_thresholding_otsu_triangle.py`

**Tujuan**: Menerapkan thresholding otomatis menggunakan metode Otsu dan Triangle untuk menentukan nilai threshold optimal.

**Langkah Kerja**:
1. Buka dan pelajari file `06_thresholding_otsu_triangle.py`.
2. Jalankan program: `python 06_thresholding_otsu_triangle.py`.
3. Amati threshold optimal yang ditemukan oleh Otsu dan Triangle.
4. Bandingkan hasil Otsu vs Triangle pada gambar yang sama.
5. Perhatikan pada histogram bimodal vs unimodal.
6. Periksa output di folder `output/`.

---

### Percobaan 7: Adaptive Thresholding
**File**: `07_adaptive_thresholding.py`

**Tujuan**: Menerapkan thresholding adaptif (Mean dan Gaussian) untuk gambar dengan pencahayaan tidak merata.

**Langkah Kerja**:
1. Buka dan pelajari file `07_adaptive_thresholding.py`.
2. Jalankan program: `python 07_adaptive_thresholding.py`.
3. Amati perbedaan ADAPTIVE_THRESH_MEAN_C vs ADAPTIVE_THRESH_GAUSSIAN_C.
4. Perhatikan efek variasi block size (11, 21, 51) dan C value.
5. Bandingkan adaptive vs global threshold pada gambar dokumen.
6. Periksa output di folder `output/`.

---

### Percobaan 8: Konvolusi dan Filter2D
**File**: `08_konvolusi_dan_filter2d.py`

**Tujuan**: Memahami prinsip konvolusi dan menerapkan custom kernel menggunakan `cv2.filter2D`.

**Langkah Kerja**:
1. Buka dan pelajari file `08_konvolusi_dan_filter2d.py`.
2. Jalankan program: `python 08_konvolusi_dan_filter2d.py`.
3. Amati efek berbagai custom kernel (averaging, edge detect, emboss, dll.).
4. Perhatikan prinsip konvolusi: kernel × region gambar.
5. Amati perbedaan padding (BORDER_CONSTANT, BORDER_REPLICATE, dll.).
6. Periksa output di folder `output/`.

---

### Percobaan 9: Gaussian Blur
**File**: `09_gaussian_blur.py`

**Tujuan**: Menerapkan Gaussian blur dengan berbagai kernel size dan sigma untuk smoothing gambar.

**Langkah Kerja**:
1. Buka dan pelajari file `09_gaussian_blur.py`.
2. Jalankan program: `python 09_gaussian_blur.py`.
3. Amati efek variasi kernel size (3, 5, 7, 11, 21).
4. Perhatikan efek variasi sigma.
5. Amati box blur vs Gaussian blur.
6. Periksa output di folder `output/`.

---

### Percobaan 10: Median dan Bilateral Filter
**File**: `10_median_dan_bilateral_filter.py`

**Tujuan**: Membandingkan median filter (untuk salt-and-pepper noise) dan bilateral filter (edge-preserving smoothing).

**Langkah Kerja**:
1. Buka dan pelajari file `10_median_dan_bilateral_filter.py`.
2. Jalankan program: `python 10_median_dan_bilateral_filter.py`.
3. Amati kemampuan median filter menghilangkan salt-and-pepper noise.
4. Perhatikan bilateral filter yang mempertahankan edge.
5. Bandingkan Gaussian vs median vs bilateral pada gambar noisy.
6. Periksa output di folder `output/`.

---

### Percobaan 11: Sharpening
**File**: `11_sharpening.py`

**Tujuan**: Mempertajam gambar menggunakan sharpening kernel dan unsharp masking.

**Langkah Kerja**:
1. Buka dan pelajari file `11_sharpening.py`.
2. Jalankan program: `python 11_sharpening.py`.
3. Amati efek berbagai sharpening kernel (3×3, 5×5).
4. Perhatikan unsharp masking: sharpened = original + α × (original − blurred).
5. Amati over-sharpening (halo artifact) pada parameter terlalu tinggi.
6. Periksa output di folder `output/`.

---

### Percobaan 12: Deteksi Tepi Sobel
**File**: `12_deteksi_tepi_sobel.py`

**Tujuan**: Mendeteksi tepi menggunakan operator Sobel (gradien horizontal dan vertikal).

**Langkah Kerja**:
1. Buka dan pelajari file `12_deteksi_tepi_sobel.py`.
2. Jalankan program: `python 12_deteksi_tepi_sobel.py`.
3. Amati Sobel X (tepi vertikal) dan Sobel Y (tepi horizontal).
4. Perhatikan magnitude: $\sqrt{G_x^2 + G_y^2}$.
5. Bandingkan Sobel dengan operator Scharr.
6. Periksa output di folder `output/`.

---

### Percobaan 13: Deteksi Tepi Canny
**File**: `13_deteksi_tepi_canny.py`

**Tujuan**: Mendeteksi tepi menggunakan algoritma Canny (multi-stage: gradient, NMS, hysteresis).

**Langkah Kerja**:
1. Buka dan pelajari file `13_deteksi_tepi_canny.py`.
2. Jalankan program: `python 13_deteksi_tepi_canny.py`.
3. Amati efek variasi low dan high threshold.
4. Perhatikan pentingnya Gaussian blur sebelum Canny.
5. Bandingkan Canny dengan Sobel secara visual.
6. Periksa output di folder `output/`.

---

### Percobaan 14: Deteksi Tepi Laplacian
**File**: `14_deteksi_tepi_laplacian.py`

**Tujuan**: Mendeteksi tepi menggunakan operator Laplacian (derivatif orde kedua).

**Langkah Kerja**:
1. Buka dan pelajari file `14_deteksi_tepi_laplacian.py`.
2. Jalankan program: `python 14_deteksi_tepi_laplacian.py`.
3. Amati Laplacian yang mendeteksi tepi di semua arah.
4. Perhatikan sensitivitas Laplacian terhadap noise.
5. Amati LoG (Laplacian of Gaussian) dan zero-crossing.
6. Periksa output di folder `output/`.

---

### Percobaan 15: Morfologi — Erosi dan Dilasi
**File**: `15_morfologi_erosi_dilasi.py`

**Tujuan**: Menerapkan operasi morfologi dasar: erosi (mengecilkan) dan dilasi (memperbesar) objek.

**Langkah Kerja**:
1. Buka dan pelajari file `15_morfologi_erosi_dilasi.py`.
2. Jalankan program: `python 15_morfologi_erosi_dilasi.py`.
3. Amati efek erosi: objek mengecil, noise hilang.
4. Amati efek dilasi: objek membesar, gap tertutup.
5. Perhatikan efek variasi kernel size dan bentuk (RECT, ELLIPSE, CROSS).
6. Periksa output di folder `output/`.

---

### Percobaan 16: Morfologi Lanjut
**File**: `16_morfologi_lanjut.py`

**Tujuan**: Menerapkan operasi morfologi lanjut: opening, closing, gradient, dan kombinasi.

**Langkah Kerja**:
1. Buka dan pelajari file `16_morfologi_lanjut.py`.
2. Jalankan program: `python 16_morfologi_lanjut.py`.
3. Amati opening (erosi → dilasi): menghilangkan noise kecil.
4. Amati closing (dilasi → erosi): menutup lubang kecil.
5. Amati morphological gradient: mengekstrak outline objek.
6. Periksa output di folder `output/`.

---

### Percobaan 17: Top Hat dan Black Hat
**File**: `17_tophat_blackhat.py`

**Tujuan**: Menerapkan Top Hat (objek terang pada latar gelap) dan Black Hat (objek gelap pada latar terang).

**Langkah Kerja**:
1. Buka dan pelajari file `17_tophat_blackhat.py`.
2. Jalankan program: `python 17_tophat_blackhat.py`.
3. Amati Top Hat = original − opening (mengekstrak fitur terang kecil).
4. Amati Black Hat = closing − original (mengekstrak fitur gelap kecil).
5. Perhatikan aplikasi untuk menghilangkan pencahayaan tidak merata.
6. Periksa output di folder `output/`.

---

### Percobaan 18: Transformasi Fourier
**File**: `18_transformasi_fourier.py`

**Tujuan**: Menganalisis gambar di domain frekuensi menggunakan DFT (Discrete Fourier Transform).

**Langkah Kerja**:
1. Buka dan pelajari file `18_transformasi_fourier.py`.
2. Jalankan program: `python 18_transformasi_fourier.py`.
3. Amati magnitude spectrum dan phase spectrum.
4. Perhatikan komponen frekuensi rendah (tengah) vs tinggi (pinggir).
5. Amati hubungan antara konten gambar dan spectrum-nya.
6. Periksa output di folder `output/`.

---

### Percobaan 19: Filter Frekuensi
**File**: `19_filter_frekuensi.py`

**Tujuan**: Menerapkan filter di domain frekuensi: low-pass, high-pass, band-pass, dan notch filter.

**Langkah Kerja**:
1. Buka dan pelajari file `19_filter_frekuensi.py`.
2. Jalankan program: `python 19_filter_frekuensi.py`.
3. Amati efek low-pass filter (blur/smoothing di frekuensi).
4. Amati efek high-pass filter (edge detection di frekuensi).
5. Bandingkan ideal vs Gaussian vs Butterworth filter.
6. Periksa output di folder `output/`.

---

### Percobaan 20: Alpha Blending dan Compositing
**File**: `20_alpha_blending_compositing.py`

**Tujuan**: Menggabungkan gambar menggunakan alpha blending, mask, dan Laplacian pyramid blending.

**Langkah Kerja**:
1. Buka dan pelajari file `20_alpha_blending_compositing.py`.
2. Jalankan program: `python 20_alpha_blending_compositing.py`.
3. Amati alpha blending sederhana: $out = \alpha \cdot img_1 + (1-\alpha) \cdot img_2$.
4. Perhatikan Laplacian pyramid blending untuk transisi mulus.
5. Amati chroma keying (green screen removal) sederhana.
6. Periksa output di folder `output/`.

---

## 5. ANALISIS

### Analisis Percobaan 1–2: Brightness/Contrast dan Histogram Equalization
- Jelaskan efek saturasi saat α tinggi (clipping di 0 dan 255).
- Mengapa equalizeHist bisa membuat gambar terlihat over-processed?
- Bagaimana histogram berubah setelah equalization?

### Analisis Percobaan 3–4: CLAHE dan Gamma Correction
- Apa keunggulan CLAHE dibanding global equalization?
- Bagaimana clipLimit mempengaruhi hasil CLAHE?
- Mengapa γ < 1 mencerahkan dan γ > 1 menggelapkan?

### Analisis Percobaan 5–7: Thresholding (Global, Otsu/Triangle, Adaptive)
- Bandingkan akurasi Otsu vs Triangle pada histogram bimodal dan unimodal.
- Kapan adaptive thresholding lebih unggul dari global?
- Bagaimana block size dan C mempengaruhi hasil adaptive threshold?

### Analisis Percobaan 8–10: Konvolusi, Gaussian Blur, Median & Bilateral
- Jelaskan prinsip konvolusi: kernel × region gambar.
- Bandingkan kemampuan denoising: Gaussian vs median vs bilateral.
- Mengapa bilateral filter mempertahankan edge?

### Analisis Percobaan 11: Sharpening
- Jelaskan prinsip unsharp masking.
- Apa yang terjadi jika parameter sharpening terlalu tinggi (halo artifact)?
- Bandingkan sharpening kernel 3×3 vs 5×5.

### Analisis Percobaan 12–14: Deteksi Tepi (Sobel, Canny, Laplacian)
- Bandingkan ketiga metode: mana yang paling robust untuk berbagai gambar?
- Apa efek threshold pada hasil Canny?
- Mengapa Laplacian sensitif terhadap noise?

### Analisis Percobaan 15–17: Morfologi (Erosi/Dilasi, Lanjut, TopHat/BlackHat)
- Jelaskan perbedaan opening vs closing dan kapan masing-masing digunakan.
- Bagaimana ukuran structuring element mempengaruhi hasil?
- Jelaskan aplikasi Top Hat untuk mengoreksi pencahayaan tidak merata.

### Analisis Percobaan 18–19: Fourier dan Filter Frekuensi
- Jelaskan apa yang direpresentasikan frekuensi rendah vs tinggi.
- Bandingkan filtering di domain spasial vs frekuensi.
- Diskusikan artefak ringing pada ideal filter vs Gaussian filter.

### Analisis Percobaan 20: Alpha Blending dan Compositing
- Mengapa Laplacian blending menghasilkan seam yang lebih halus?
- Jelaskan peran mask dalam compositing.
- Diskusikan limitasi chroma keying sederhana.

---

## 6. KESIMPULAN

Buatlah kesimpulan yang mencakup:
1. Pemahaman tentang operasi titik (brightness, contrast, gamma, histogram).
2. Efektivitas berbagai metode thresholding untuk segmentasi sederhana.
3. Prinsip konvolusi dan perannya dalam filtering spasial.
4. Perbandingan metode denoising dan trade-off smoothing vs detail.
5. Perbandingan metode edge detection dan aplikasinya.
6. Kegunaan operasi morfologi dalam pemrosesan gambar biner.
7. Pemahaman domain frekuensi dan hubungannya dengan domain spasial.
8. Teknik compositing dan alpha blending untuk penggabungan gambar.
