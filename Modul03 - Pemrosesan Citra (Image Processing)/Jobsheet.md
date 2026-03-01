# JOBSHEET PRAKTIKUM
# MODUL 3: PEMROSESAN CITRA (IMAGE PROCESSING)

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa mampu:
1. Melakukan penyesuaian brightness dan contrast gambar.
2. Menerapkan gamma correction untuk koreksi pencahayaan.
3. Melakukan thresholding global, Otsu, dan adaptif.
4. Memahami dan menerapkan histogram equalization dan CLAHE.
5. Melakukan spatial filtering (blur, sharpen) menggunakan konvolusi.
6. Mendeteksi tepi dengan Sobel, Canny, dan Laplacian.
7. Melakukan operasi morfologi (erosi, dilasi, opening, closing).
8. Membangun pipeline image enhancement end-to-end.
9. Melakukan alpha compositing dan matting.
10. Menerapkan transformasi Fourier dan filtering di domain frekuensi.

---

## 2. ALAT DAN BAHAN

### A. Perangkat Keras
- Laptop/PC (min. Intel Core i3, RAM 4 GB)

### B. Perangkat Lunak
- Python 3.8+, VS Code
- Library: `opencv-python`, `numpy`, `matplotlib`, `scipy`

### C. Dataset
- Gambar sample di folder `data/images/` (termasuk gambar low-contrast, noisy, dokumen).
- Jalankan `setup_images.py`.

---

## 3. LANGKAH KERJA

### Percobaan 1: Brightness dan Contrast Adjustment

**Tujuan**: Mengubah brightness dan kontras gambar menggunakan operasi titik.

**Langkah Kerja**:
1. Buat file `01_brightness_contrast.py`.
2. Baca gambar berwarna.
3. Implementasikan rumus: `g(x,y) = α·f(x,y) + β`.
4. Variasikan α (0.5, 1.0, 1.5, 2.0) dan β (−50, 0, +50, +100).
5. Gunakan `cv2.convertScaleAbs(img, alpha=α, beta=β)`.
6. Tampilkan grid 4×4 untuk kombinasi α dan β.
7. Buat trackbar interaktif untuk adjustment real-time.
8. Hitung dan tampilkan histogram sebelum dan sesudah adjustment.
9. Implementasikan auto-contrast (linear stretch ke range 0–255).
10. Simpan hasil ke folder output.

---

### Percobaan 2: Gamma Correction

**Tujuan**: Menerapkan koreksi gamma untuk memperbaiki gambar gelap/terang.

**Langkah Kerja**:
1. Buat file `02_gamma_correction.py`.
2. Baca gambar yang under-exposed dan over-exposed.
3. Implementasikan gamma correction: `output = ((input/255)^γ) × 255`.
4. Buat LUT (Look-Up Table) untuk gamma correction cepat.
5. Terapkan dengan `cv2.LUT()`.
6. Variasikan γ: 0.3, 0.5, 1.0, 1.5, 2.0, 3.0.
7. Tampilkan perbandingan side-by-side.
8. Implementasikan auto-gamma: estimasi γ optimal berdasarkan mean intensity.
9. Bandingkan histogram sebelum dan sesudah.
10. Simpan hasil ke folder output.

---

### Percobaan 3: Thresholding

**Tujuan**: Mengkonversi gambar ke biner menggunakan berbagai metode thresholding.

**Langkah Kerja**:
1. Buat file `03_thresholding.py`.
2. Baca gambar dan konversi ke grayscale.
3. Terapkan binary thresholding dengan nilai T manual (100, 127, 150, 200).
4. Terapkan Otsu's thresholding: `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)`.
5. Terapkan adaptive thresholding:
   - Mean: `cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_MEAN_C, ...)`.
   - Gaussian: `cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_GAUSSIAN_C, ...)`.
6. Variasikan block size (11, 21, 51) dan C value.
7. Terapkan pada gambar dokumen teks dan bandingkan hasilnya.
8. Terapkan pada gambar dengan pencahayaan tidak merata.
9. Plot histogram dengan garis threshold.
10. Simpan perbandingan semua metode.

---

### Percobaan 4: Histogram Equalization

**Tujuan**: Meningkatkan kontras gambar melalui pemerataan histogram.

**Langkah Kerja**:
1. Buat file `04_histogram_equalization.py`.
2. Baca gambar low-contrast (gelap atau flat).
3. Terapkan histogram equalization: `cv2.equalizeHist(gray)`.
4. Tampilkan histogram sebelum dan sesudah equalization.
5. Terapkan pada gambar berwarna:
   - Konversi RGB → YCrCb, equalize channel Y, konversi balik.
   - Atau konversi RGB → HSV, equalize channel V.
6. Terapkan CLAHE: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`.
7. Variasikan clipLimit (1.0, 2.0, 4.0, 8.0) dan tileGridSize.
8. Bandingkan equalizeHist vs CLAHE secara visual.
9. Terapkan pada gambar medis (X-ray) untuk meningkatkan visibilitas.
10. Simpan perbandingan dan histogram.

---

### Percobaan 5: Spatial Filtering (Blur dan Sharpen)

**Tujuan**: Menerapkan filter spasial menggunakan konvolusi untuk smoothing dan sharpening.

**Langkah Kerja**:
1. Buat file `05_spatial_filtering.py`.
2. Baca gambar berwarna dan gambar yang noisy.
3. Terapkan berbagai blur:
   - Box filter: `cv2.blur(img, (k, k))`.
   - Gaussian blur: `cv2.GaussianBlur(img, (k, k), sigma)`.
   - Median filter: `cv2.medianBlur(img, k)`.
   - Bilateral filter: `cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)`.
4. Variasikan kernel size: 3, 5, 7, 11, 21.
5. Implementasikan sharpening kernel manual:
   ```
   kernel = [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]]
   ```
6. Terapkan unsharp masking: `sharpened = original + α(original − blurred)`.
7. Bandingkan efektivitas median vs Gaussian untuk menghilangkan salt-and-pepper noise.
8. Buat custom kernel emboss dan edge-enhancement.
9. Ukur waktu eksekusi setiap filter.
10. Simpan perbandingan visual.

---

### Percobaan 6: Deteksi Tepi (Edge Detection)

**Tujuan**: Mendeteksi tepi pada gambar menggunakan berbagai operator.

**Langkah Kerja**:
1. Buat file `06_edge_detection.py`.
2. Baca gambar dan konversi ke grayscale.
3. Terapkan Sobel edge detection:
   - `cv2.Sobel(gray, cv2.CV_64F, 1, 0)` — gradien horizontal.
   - `cv2.Sobel(gray, cv2.CV_64F, 0, 1)` — gradien vertikal.
   - Hitung magnitude: `np.sqrt(sobel_x² + sobel_y²)`.
4. Terapkan Canny edge detection dengan berbagai threshold:
   - Low threshold = 50, 100, 150.
   - High threshold = 100, 200, 300.
5. Terapkan Laplacian: `cv2.Laplacian(gray, cv2.CV_64F)`.
6. Terapkan Scharr operator: `cv2.Scharr()`.
7. Bandingkan semua metode secara visual.
8. Implementasikan Canny interaktif dengan trackbar untuk threshold.
9. Terapkan pada gambar dengan kompleksitas berbeda (simple object vs scene).
10. Simpan semua hasil perbandingan.

---

### Percobaan 7: Operasi Morfologi

**Tujuan**: Menerapkan operasi morfologi untuk membersihkan gambar biner dan mengekstrak fitur.

**Langkah Kerja**:
1. Buat file `07_operasi_morfologi.py`.
2. Baca gambar, konversi ke grayscale, lalu threshold.
3. Buat structuring element: `kernel = cv2.getStructuringElement(shape, (size, size))`.
   - Coba `MORPH_RECT`, `MORPH_ELLIPSE`, `MORPH_CROSS`.
4. Terapkan erosi: `cv2.erode()`.
5. Terapkan dilasi: `cv2.dilate()`.
6. Terapkan opening: `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)`.
7. Terapkan closing: `cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)`.
8. Terapkan gradient: `cv2.MORPH_GRADIENT`.
9. Terapkan top hat dan black hat.
10. Gunakan morfologi untuk:
    - Menghilangkan noise kecil dari gambar biner (opening).
    - Menghubungkan komponen yang terputus (closing).
    - Mengekstrak outline objek (gradient).

---

### Percobaan 8: Pipeline Image Enhancement

**Tujuan**: Membangun pipeline pemrosesan gambar end-to-end yang mengombinasikan beberapa teknik.

**Langkah Kerja**:
1. Buat file `08_enhancement_pipeline.py`.
2. Baca gambar berkualitas rendah (gelap, noise, low contrast).
3. Bangun pipeline:
   - Step 1: Denoising (Gaussian blur atau bilateral).
   - Step 2: Contrast enhancement (CLAHE).
   - Step 3: Sharpening (unsharp mask).
   - Step 4: Color correction (white balance).
4. Tampilkan hasil per step secara berurutan.
5. Implementasikan pipeline sebagai fungsi yang bisa di-reuse.
6. Terapkan pada 3 gambar berbeda dan bandingkan hasilnya.
7. Buat versi pipeline dengan parameter yang bisa di-tune via argumen.
8. Tambahkan metrik kualitas: mean, std, entropy.
9. Bandingkan pipeline Anda dengan hanya menggunakan equalizeHist saja.
10. Simpan before-after comparison.

---

### Percobaan 9: Compositing dan Alpha Blending

**Tujuan**: Menggabungkan gambar menggunakan alpha compositing dan Laplacian blending.

**Langkah Kerja**:
1. Buat file `09_compositing.py`.
2. Baca dua gambar dan buat alpha mask.
3. Implementasikan alpha blending sederhana: `out = α·img1 + (1−α)·img2`.
4. Buat mask gradient (soft transition) menggunakan linear gradient.
5. Implementasikan Laplacian pyramid blending:
   - Build Laplacian pyramid untuk kedua gambar.
   - Build Gaussian pyramid untuk mask.
   - Blend setiap level.
   - Reconstruct dari blended Laplacian pyramid.
6. Bandingkan hasil hard-cut vs soft blend vs Laplacian blend.
7. Terapkan pada kasus: menggabungkan setengah apel/jeruk.
8. Buat efek double exposure (overlay dua foto).
9. Terapkan green screen removal (chroma keying) sederhana.
10. Simpan semua perbandingan.

---

### Percobaan 10: Transformasi Fourier

**Tujuan**: Menganalisis gambar di domain frekuensi menggunakan DFT.

**Langkah Kerja**:
1. Buat file `10_fourier_transform.py`.
2. Baca gambar grayscale.
3. Hitung DFT: `dft = np.fft.fft2(img)`, shift: `np.fft.fftshift(dft)`.
4. Tampilkan magnitude spectrum: `20 * np.log(np.abs(dft_shift))`.
5. Buat ideal low-pass filter di domain frekuensi dan terapkan.
6. Buat ideal high-pass filter dan terapkan.
7. Buat Gaussian low-pass dan high-pass filter.
8. Buat band-pass filter.
9. Inverse DFT: rekonstruksi gambar setelah filtering.
10. Bandingkan spatial domain filtering vs frequency domain filtering hasilnya.

---

## 4. ANALISIS

### Analisis Percobaan 1 — Brightness & Contrast
- Jelaskan efek saturasi saat α tinggi (clipping di 0 dan 255).
- Bagaimana histogram berubah saat α dan β divariasikan?
- Kapan auto-contrast berguna?

### Analisis Percobaan 2 — Gamma Correction
- Mengapa γ < 1 mencerahkan dan γ > 1 menggelapkan?
- Diskusikan hubungan gamma correction dengan persepsi visual manusia.
- Apa keunggulan LUT vs komputasi langsung?

### Analisis Percobaan 3 — Thresholding
- Bandingkan akurasi Otsu vs manual threshold.
- Kapan adaptive thresholding lebih unggul dari global?
- Bagaimana block size mempengaruhi hasil adaptive threshold?

### Analisis Percobaan 4 — Histogram Equalization
- Mengapa equalizeHist bisa membuat gambar terlihat over-processed?
- Apa keunggulan CLAHE dibanding global equalization?
- Bagaimana clipLimit mempengaruhi hasil CLAHE?

### Analisis Percobaan 5 — Spatial Filtering
- Bandingkan kemampuan denoising: Gaussian vs median vs bilateral.
- Mengapa bilateral filter mempertahankan edge?
- Jelaskan trade-off antara smoothing dan detail preservation.

### Analisis Percobaan 6 — Edge Detection
- Bandingkan Sobel vs Canny vs Laplacian: Mana yang paling robust?
- Apa efek threshold pada hasil Canny?
- Mengapa pre-processing (blur) penting sebelum edge detection?

### Analisis Percobaan 7 — Morfologi
- Jelaskan perbedaan efek opening vs closing.
- Bagaimana ukuran structuring element mempengaruhi hasil?
- Kapan menggunakan erosi vs dilasi?

### Analisis Percobaan 8 — Enhancement Pipeline
- Apakah urutan step dalam pipeline berpengaruh? Buktikan.
- Parameter mana yang paling kritis dalam pipeline?
- Bandingkan metrik kualitas: pipeline vs single-step enhancement.

### Analisis Percobaan 9 — Compositing
- Mengapa Laplacian blending menghasilkan seam yang lebih halus?
- Jelaskan peran mask dalam compositing.
- Diskusikan limitasi chroma keying sederhana.

### Analisis Percobaan 10 — Fourier
- Jelaskan apa yang direpresentasikan oleh komponen frekuensi rendah vs tinggi.
- Bandingkan filtering di domain spasial vs frekuensi.
- Diskusikan artefak ringing pada ideal filter vs Gaussian filter.

---

## 5. KESIMPULAN

Buatlah kesimpulan yang mencakup:
1. Pemahaman tentang operasi titik (brightness, contrast, gamma, histogram).
2. Efektivitas berbagai metode thresholding untuk segmentasi sederhana.
3. Peran spatial filtering dalam denoising dan sharpening serta trade-off-nya.
4. Perbandingan metode edge detection dan aplikasinya.
5. Kegunaan operasi morfologi dalam pemrosesan gambar biner.
6. Pemahaman domain frekuensi dan hubungannya dengan domain spasial.
