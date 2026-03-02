# JOBSHEET PRAKTIKUM
# MODUL 5: MODEL FITTING DAN OPTIMASI

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa mampu:
1. Mengimplementasikan Ordinary Least Squares (OLS) untuk fitting model linear.
2. Menerapkan Weighted Least Squares (WLS) dan Total Least Squares (TLS).
3. Mengimplementasikan RANSAC untuk estimasi model yang robust terhadap outlier.
4. Mendeteksi garis dan lingkaran menggunakan Hough Transform.
5. Mengestimasi homografi dari pasangan titik korespondensi.
6. Melakukan koreksi perspektif menggunakan homografi.
7. Menerapkan IRLS (Iteratively Reweighted Least Squares) untuk robust fitting.
8. Memahami regularisasi Ridge dan Lasso dalam model fitting.
9. Melakukan fitting ellips dan analisis kontur.
10. Menerapkan template matching dan graph cut segmentation.
11. Mengestimasi optical flow (Lucas-Kanade dan dense Farneback).
12. Melakukan feature matching dengan RANSAC-based homography.
13. Menerapkan cross-validation untuk model selection.
14. Menggunakan optimasi untuk denoising gambar.
15. Membangun pipeline gabungan model fitting end-to-end.

---

## 2. ALAT DAN BAHAN

### A. Perangkat Keras
- Laptop/PC (min. Intel Core i3, RAM 4 GB)

### B. Perangkat Lunak
- Python 3.8+, VS Code
- Library: `opencv-python`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`

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

### Percobaan 1: Ordinary Least Squares (OLS)
**File**: `01_ordinary_least_squares.py`

**Tujuan**: Mengimplementasikan fitting model linear menggunakan OLS — meminimalkan jumlah kuadrat residual vertikal.

**Langkah Kerja**:
1. Buka dan pelajari file `01_ordinary_least_squares.py`.
2. Jalankan program: `python 01_ordinary_least_squares.py`.
3. Amati fitting garis pada data sintetis (tanpa dan dengan noise).
4. Perhatikan formula normal equation: $\hat{\beta} = (X^TX)^{-1}X^Ty$.
5. Amati residual plot dan error metrics (MSE, R²).
6. Periksa output di folder `output/`.

---

### Percobaan 2: Weighted Least Squares (WLS)
**File**: `02_weighted_least_squares.py`

**Tujuan**: Menerapkan WLS yang memberikan bobot berbeda pada observasi berdasarkan reliability.

**Langkah Kerja**:
1. Buka dan pelajari file `02_weighted_least_squares.py`.
2. Jalankan program: `python 02_weighted_least_squares.py`.
3. Amati perbedaan fitting OLS vs WLS pada data heteroskedastik.
4. Perhatikan efek bobot terhadap fitting line.
5. Amati bagaimana WLS mengurangi pengaruh observasi tidak reliable.
6. Periksa output di folder `output/`.

---

### Percobaan 3: Total Least Squares (TLS)
**File**: `03_total_least_squares.py`

**Tujuan**: Mengimplementasikan TLS yang meminimalkan jarak orthogonal (tegak lurus) ke model.

**Langkah Kerja**:
1. Buka dan pelajari file `03_total_least_squares.py`.
2. Jalankan program: `python 03_total_least_squares.py`.
3. Amati perbedaan OLS (jarak vertikal) vs TLS (jarak tegak lurus).
4. Perhatikan penggunaan SVD untuk solusi TLS.
5. Amati kasus di mana TLS lebih tepat daripada OLS.
6. Periksa output di folder `output/`.

---

### Percobaan 4: RANSAC Fitting Garis
**File**: `04_ransac_fitting_garis.py`

**Tujuan**: Mengimplementasikan RANSAC untuk fitting garis yang robust terhadap outlier.

**Langkah Kerja**:
1. Buka dan pelajari file `04_ransac_fitting_garis.py`.
2. Jalankan program: `python 04_ransac_fitting_garis.py`.
3. Amati data sintetis dengan inlier dan outlier yang jelas.
4. Perhatikan proses RANSAC: random sampling → fit → count inliers → update best.
5. Bandingkan hasil RANSAC vs OLS pada data yang sama.
6. Periksa output di folder `output/`.

---

### Percobaan 5: RANSAC Fitting Lingkaran
**File**: `05_ransac_fitting_lingkaran.py`

**Tujuan**: Menerapkan RANSAC untuk fitting lingkaran (estimasi center dan radius).

**Langkah Kerja**:
1. Buka dan pelajari file `05_ransac_fitting_lingkaran.py`.
2. Jalankan program: `python 05_ransac_fitting_lingkaran.py`.
3. Amati fitting lingkaran pada data titik dengan outlier.
4. Perhatikan minimum 3 titik untuk menentukan lingkaran.
5. Bandingkan RANSAC circle vs least squares circle fit.
6. Periksa output di folder `output/`.

---

### Percobaan 6: Hough Transform Garis
**File**: `06_hough_transform_garis.py`

**Tujuan**: Mendeteksi garis pada gambar menggunakan Hough Transform dan probabilistic Hough.

**Langkah Kerja**:
1. Buka dan pelajari file `06_hough_transform_garis.py`.
2. Jalankan program: `python 06_hough_transform_garis.py`.
3. Amati akumulator Hough space (ρ, θ).
4. Perhatikan perbedaan `cv2.HoughLines` vs `cv2.HoughLinesP` (probabilistic).
5. Amati efek threshold dan parameter minLineLength/maxLineGap.
6. Periksa output di folder `output/`.

---

### Percobaan 7: Hough Transform Lingkaran
**File**: `07_hough_transform_lingkaran.py`

**Tujuan**: Mendeteksi lingkaran pada gambar menggunakan Hough Circle Transform.

**Langkah Kerja**:
1. Buka dan pelajari file `07_hough_transform_lingkaran.py`.
2. Jalankan program: `python 07_hough_transform_lingkaran.py`.
3. Amati deteksi lingkaran pada gambar koin atau objek bulat.
4. Perhatikan parameter: dp, minDist, param1, param2, minRadius, maxRadius.
5. Amati efek tuning parameter terhadap jumlah deteksi.
6. Periksa output di folder `output/`.

---

### Percobaan 8: Homography Estimation
**File**: `08_homography_estimation.py`

**Tujuan**: Mengestimasi matriks homografi 3×3 dari pasangan titik korespondensi.

**Langkah Kerja**:
1. Buka dan pelajari file `08_homography_estimation.py`.
2. Jalankan program: `python 08_homography_estimation.py`.
3. Amati estimasi homografi dari 4+ pasang titik.
4. Perhatikan perbedaan `cv2.findHomography` dengan metode RANSAC, LMEDS, RHO.
5. Amati warpPerspective menggunakan homografi yang diestimasi.
6. Periksa output di folder `output/`.

---

### Percobaan 9: Koreksi Perspektif Dokumen
**File**: `09_koreksi_perspektif_dokumen.py`

**Tujuan**: Menerapkan homografi untuk koreksi perspektif dokumen (document scanner).

**Langkah Kerja**:
1. Buka dan pelajari file `09_koreksi_perspektif_dokumen.py`.
2. Jalankan program: `python 09_koreksi_perspektif_dokumen.py`.
3. Amati deteksi kontur dokumen dan 4 titik sudut.
4. Perhatikan penghitungan homografi dan warp ke tampilan tegak lurus.
5. Amati pipeline lengkap: preprocessing → contour → homography → warp.
6. Periksa output di folder `output/`.

---

### Percobaan 10: IRLS (Iteratively Reweighted Least Squares)
**File**: `10_irls_robust_fitting.py`

**Tujuan**: Mengimplementasikan IRLS yang melakukan fitting iteratif dengan bobot yang di-update berdasarkan residual.

**Langkah Kerja**:
1. Buka dan pelajari file `10_irls_robust_fitting.py`.
2. Jalankan program: `python 10_irls_robust_fitting.py`.
3. Amati proses iteratif: bobot besar untuk inlier, kecil untuk outlier.
4. Perhatikan konvergensi IRLS dari iterasi ke iterasi.
5. Bandingkan IRLS vs OLS vs RANSAC pada data yang sama.
6. Periksa output di folder `output/`.

---

### Percobaan 11: Regularisasi Ridge dan Lasso
**File**: `11_regularisasi_ridge_lasso.py`

**Tujuan**: Memahami regularisasi L2 (Ridge) dan L1 (Lasso) untuk mencegah overfitting.

**Langkah Kerja**:
1. Buka dan pelajari file `11_regularisasi_ridge_lasso.py`.
2. Jalankan program: `python 11_regularisasi_ridge_lasso.py`.
3. Amati efek parameter regularisasi λ (alpha) pada fitting.
4. Perhatikan Ridge: menyusutkan koefisien, Lasso: membuat koefisien = 0 (sparsity).
5. Amati trade-off bias-variance dengan variasi λ.
6. Periksa output di folder `output/`.

---

### Percobaan 12: Fitting Ellips dan Kontur
**File**: `12_fitting_ellips_kontur.py`

**Tujuan**: Melakukan fitting ellips, minimum area rectangle, convex hull, dan polygon approximation pada kontur.

**Langkah Kerja**:
1. Buka dan pelajari file `12_fitting_ellips_kontur.py`.
2. Jalankan program: `python 12_fitting_ellips_kontur.py`.
3. Amati fitEllipse, fitEllipseAMS, dan fitEllipseDirect.
4. Perhatikan minAreaRect, convexHull, dan approxPolyDP.
5. Bandingkan berbagai metode fitting pada bentuk yang sama.
6. Periksa output di folder `output/`.

---

### Percobaan 13: Template Matching
**File**: `13_template_matching.py`

**Tujuan**: Mencocokkan template pada gambar menggunakan berbagai metode korelasi dan NMS.

**Langkah Kerja**:
1. Buka dan pelajari file `13_template_matching.py`.
2. Jalankan program: `python 13_template_matching.py`.
3. Amati 6 metode matchTemplate (TM_CCOEFF, TM_CCORR, TM_SQDIFF + versi NORMED).
4. Perhatikan multi-object detection dan Non-Maximum Suppression (NMS).
5. Amati multi-scale template matching.
6. Periksa output di folder `output/`.

---

### Percobaan 14: Graph Cut Segmentation
**File**: `14_graph_cut_segmentation.py`

**Tujuan**: Menerapkan segmentasi gambar menggunakan GrabCut, Watershed, dan MRF energy minimization.

**Langkah Kerja**:
1. Buka dan pelajari file `14_graph_cut_segmentation.py`.
2. Jalankan program: `python 14_graph_cut_segmentation.py`.
3. Amati GrabCut segmentation (rectangle-based dan mask-based).
4. Perhatikan Watershed segmentation.
5. Amati MRF ICM (Iterated Conditional Modes) energy minimization.
6. Periksa output di folder `output/`.

---

### Percobaan 15: K-Means Clustering untuk Segmentasi Citra
**File**: `15_kmeans_clustering.py`

**Tujuan**: Menerapkan K-Means clustering untuk segmentasi warna dan kuantisasi gambar.

**Langkah Kerja**:
1. Buka dan pelajari file `15_kmeans_clustering.py`.
2. Jalankan program: `python 15_kmeans_clustering.py`.
3. Amati K-Means pada data 2D sintetis (3 cluster).
4. Perhatikan kuantisasi warna dengan berbagai nilai K (2, 4, 8, 16, 32).
5. Amati segmentasi gambar berbasis fitur posisi+warna.
6. Pelajari metode Elbow untuk memilih K optimal.
7. Periksa output di folder `output/`.

---

### Percobaan 16: Kalman Filter Dasar
**File**: `16_kalman_filter_dasar.py`

**Tujuan**: Mempelajari state estimation menggunakan Kalman Filter untuk tracking objek.

**Langkah Kerja**:
1. Buka dan pelajari file `16_kalman_filter_dasar.py`.
2. Jalankan program: `python 16_kalman_filter_dasar.py`.
3. Amati Kalman Filter 1D (estimasi posisi dari pengukuran noisy).
4. Perhatikan Kalman Filter 2D untuk tracking titik bergerak.
5. Amati perbandingan prediksi vs pengukuran vs estimasi Kalman.
6. Periksa output di folder `output/`.

---

### Percobaan 17: Convex Hull dan Shape Fitting
**File**: `17_convex_hull_shape_fitting.py`

**Tujuan**: Mempelajari fitting bentuk geometri menggunakan convex hull, minAreaRect, dan minEnclosingCircle.

**Langkah Kerja**:
1. Buka dan pelajari file `17_convex_hull_shape_fitting.py`.
2. Jalankan program: `python 17_convex_hull_shape_fitting.py`.
3. Amati cv2.convexHull() pada berbagai bentuk kontur.
4. Perhatikan cv2.minAreaRect() dan cv2.minEnclosingCircle().
5. Amati perbandingan area kontur vs convex hull (convexity defects).
6. Periksa output di folder `output/`.

---

### Percobaan 18: Cross-Validation dan Model Selection
**File**: `18_cross_validation_model_selection.py`

**Tujuan**: Menerapkan K-Fold dan LOO cross-validation untuk memilih model dan parameter optimal.

**Langkah Kerja**:
1. Buka dan pelajari file `18_cross_validation_model_selection.py`.
2. Jalankan program: `python 18_cross_validation_model_selection.py`.
3. Amati K-Fold CV dan Leave-One-Out CV.
4. Perhatikan train error vs CV error (overfitting detection).
5. Amati RANSAC threshold selection via cross-validation dan learning curves.
6. Periksa output di folder `output/`.

---

### Percobaan 19: Denoising via Optimasi
**File**: `19_denoising_optimasi.py`

**Tujuan**: Menerapkan denoising berbasis optimasi: NLM, Total Variation, dan bilateral filter.

**Langkah Kerja**:
1. Buka dan pelajari file `19_denoising_optimasi.py`.
2. Jalankan program: `python 19_denoising_optimasi.py`.
3. Amati Non-Local Means denoising (fastNlMeansDenoising).
4. Perhatikan Total Variation denoising via gradient descent.
5. Bandingkan metode berdasarkan PSNR.
6. Periksa output di folder `output/`.

---

### Percobaan 20: Pipeline Gabungan
**File**: `20_pipeline_gabungan.py`

**Tujuan**: Membangun pipeline end-to-end yang mengintegrasikan berbagai teknik model fitting.

**Langkah Kerja**:
1. Buka dan pelajari file `20_pipeline_gabungan.py`.
2. Jalankan program: `python 20_pipeline_gabungan.py`.
3. Amati Pipeline 1: Lane detection (Canny → Hough → RANSAC).
4. Amati Pipeline 2: Transform estimation (ORB → match → homography → warp).
5. Amati Pipeline 3: Benchmark fitting methods (OLS, WLS, TLS, RANSAC, IRLS).
6. Periksa output di folder `output/`.

---

## 5. ANALISIS

### Analisis Percobaan 1–3: OLS, WLS, dan TLS
- Jelaskan perbedaan residual yang diminimalkan oleh OLS, WLS, dan TLS.
- Kapan WLS lebih tepat digunakan daripada OLS?
- Mengapa TLS lebih cocok ketika kedua variabel memiliki noise?

### Analisis Percobaan 4–5: RANSAC
- Jelaskan mengapa RANSAC robust terhadap outlier.
- Bagaimana jumlah iterasi mempengaruhi probabilitas menemukan model terbaik?
- Bandingkan RANSAC garis (2 titik) vs lingkaran (3 titik).

### Analisis Percobaan 6–7: Hough Transform
- Jelaskan prinsip voting di Hough space.
- Apa trade-off antara resolution parameter dan akurasi deteksi?
- Bandingkan standard Hough vs probabilistic Hough.

### Analisis Percobaan 8–9: Homography dan Koreksi Perspektif
- Jelaskan mengapa homografi memerlukan minimal 4 pasang titik.
- Apa perbedaan metode RANSAC, LMEDS, dan RHO pada findHomography?
- Bagaimana pipeline document scanner bekerja secara end-to-end?

### Analisis Percobaan 10–11: IRLS dan Regularisasi
- Jelaskan proses iteratif IRLS dan kapan ia konvergen.
- Bandingkan efek Ridge (L2) vs Lasso (L1) pada koefisien model.
- Bagaimana trade-off bias-variance terkait dengan parameter regularisasi?

### Analisis Percobaan 12–13: Fitting Kontur dan Template Matching
- Bandingkan fitEllipse, AMS, dan Direct pada berbagai bentuk.
- Jelaskan kekuatan dan kelemahan template matching.
- Mengapa multi-scale matching diperlukan?

### Analisis Percobaan 14: Graph Cut Segmentation
- Jelaskan prinsip energy minimization dalam GrabCut.
- Bandingkan GrabCut vs Watershed: kapan masing-masing lebih unggul?
- Apa limitasi MRF-based segmentation?

### Analisis Percobaan 15–17: K-Means, Kalman Filter, dan Shape Fitting
- Jelaskan bagaimana K-Means melakukan clustering berbasis jarak Euclidean.
- Bagaimana metode Elbow membantu memilih jumlah cluster optimal?
- Jelaskan proses prediksi dan koreksi pada Kalman Filter.
- Bandingkan convex hull, minAreaRect, dan minEnclosingCircle untuk fitting bentuk.

### Analisis Percobaan 18–20: Cross-Validation, Denoising, dan Pipeline
- Bandingkan NLM, Total Variation, dan bilateral filter dari segi PSNR dan visual.
- Jelaskan bagaimana pipeline lane detection bekerja.
- Apa critical step dalam pipeline transform estimation?

---

## 6. KESIMPULAN

Buatlah kesimpulan yang mencakup:
1. Pemahaman tentang berbagai metode least squares (OLS, WLS, TLS).
2. Peran RANSAC dalam estimasi model yang robust terhadap outlier.
3. Hough Transform sebagai metode voting untuk deteksi bentuk geometri.
4. Homografi dan aplikasinya dalam koreksi perspektif.
5. Regularisasi sebagai teknik pencegahan overfitting.
6. Optical flow untuk estimasi gerakan antar frame.
7. Cross-validation untuk model selection yang objektif.
8. Integrasi berbagai teknik fitting dalam pipeline end-to-end.
