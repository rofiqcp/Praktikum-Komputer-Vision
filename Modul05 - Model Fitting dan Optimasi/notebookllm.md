# NotebookLM Prompts  Modul 5: Model Fitting dan Optimasi

---

## PROMPT 1  Slide 115 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 5: Model Fitting dan Optimasi. Referensi Szeliski (2022) Ch.6. Tiap slide ~300 kata, sertakan formula dan kode OpenCV.

**Slide 1**  Judul "Modul 5: Model Fitting dan Optimasi", subtitle "Fondasi Estimasi Parameter dalam Computer Vision", ilustrasi pipeline fitting model ke data visual.

**Slide 2**  Ordinary Least Squares: minimasi $\sum(y_i - \hat{y}_i)^2$. Solusi analitik $\beta = (X^TX)^{-1}X^Ty$. 
p.linalg.lstsq(). Aplikasi: fitting garis, polinomial. Limitasi: sensitif outlier.

**Slide 3**  Weighted Least Squares dan Total Least Squares: WLS  bobot per observasi $\sum w_i(y_i - \hat{y}_i)^2$. TLS  minimasi jarak ortogonal, noise di x dan y. Perbandingan visual OLS vs WLS vs TLS.

**Slide 4**  RANSAC: Random Sample Consensus  (1) sample minimal, (2) fit model, (3) hitung inlier, (4) ulangi N kali, (5) ambil model terbaik. Jumlah iterasi: $N = \log(1-p)/\log(1-w^n)$. cv2.findHomography(..., cv2.RANSAC).

**Slide 5**  RANSAC Fitting Lingkaran: 3 titik minimal  persamaan lingkaran. Threshold jarak ke lingkaran. Aplikasi: deteksi objek bulat, iris mata, penghitung koin.

**Slide 6**  Hough Transform Garis: ruang parameter $(\rho, \theta)$, voting accumulator. cv2.HoughLines() dan cv2.HoughLinesP(). Deteksi garis pada edge image (CannyHough).

**Slide 7**  Hough Transform Lingkaran: 3D accumulator $(x_c, y_c, r)$. cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50). Deteksi lingkaran: koin, mata, roda.

**Slide 8**  Homography Estimation: transformasi projective 33, 8 DOF. cv2.findHomography(). Koreksi perspektif dokumen: cv2.getPerspectiveTransform() + cv2.warpPerspective().

**Slide 9**  IRLS (Iteratively Reweighted Least Squares): robust fitting iteratif. Update bobot berdasarkan residual: Huber, Tukey bisquare. Percobaan 10: perbandingan L2, Huber, Tukey pada data dengan outlier.

**Slide 10**  Regularisasi Ridge dan Lasso: Ridge (L2) $+ \lambda\|\beta\|_2^2$ menyusutkan koefisien. Lasso (L1) $+ \lambda\|\beta\|_1$ menghasilkan sparsity. Bias-variance tradeoff. Percobaan 11: perbandingan overfitting.

**Slide 11**  Fitting Ellips dan Template Matching: cv2.fitEllipse(), cv2.minAreaRect(). Template matching: cv2.matchTemplate() dengan TM_CCOEFF_NORMED. Multi-scale template matching.

**Slide 12**  Graph Cut Segmentation: GrabCut (cv2.grabCut())  foreground/background via graph min-cut. Watershed  topographic segmentation cv2.watershed(). Percobaan 14.

**Slide 13**  K-Means Clustering dan Kalman Filter: cv2.kmeans() untuk color quantization dan segmentasi. Kalman filter: predict-correct untuk tracking. cv2.KalmanFilter(4,2).

**Slide 14**  Convex Hull dan Shape Fitting: cv2.convexHull(), cv2.convexityDefects(). cv2.fitLine() dengan norm L1, L2, Huber. cv2.minEnclosingCircle().

**Slide 15**  Cross-Validation, Denoising, Pipeline: K-Fold CV untuk model selection. NLM denoising, TV denoising, bilateral filter. Pipeline gabungan: integrasi teknik dalam satu workflow.

---

## PROMPT 2  Slide 1630 (Materi Lanjutan + Analisis)

Lanjutkan slide Modul 5, Slide 1630. Slide 1625: analisis mendalam dan rekap percobaan. Slide 2630: kuis dan koneksi antar modul. Tiap slide ~300 kata.

**Slide 16**  Rekap Percobaan 15: OLS (fitting garis/polinomial), WLS (data heteroscedastic), TLS (noise dua arah), RANSAC garis (robust fitting), RANSAC lingkaran (3-titik minimal). Tabel perbandingan metode fitting.

**Slide 17**  Rekap Percobaan 610: Hough garis (lane detection), Hough lingkaran (coin detection), Homography (perspective warp), Koreksi perspektif (document scanner), IRLS (robust estimation). Grid visual output.

**Slide 18**  Rekap Percobaan 1115: Regularisasi (overfitting prevention), Fitting ellips (shape analysis), Template matching (object localization), Graph Cut (interactive segmentation), K-Means (color quantization).

**Slide 19**  Rekap Percobaan 1620: Kalman filter (noisy tracking), Convex hull (shape analysis), Cross-validation (model selection), Denoising optimasi (NLM/TV/bilateral), Pipeline gabungan (integrated workflow).

**Slide 20**  Analisis: Kapan OLS cukup vs perlu RANSAC? Threshold inlier mempengaruhi hasil RANSAC. Ridge vs Lasso: kapan masing-masing lebih tepat? K-Means: pemilihan K (elbow method, silhouette).

**Slide 21**  Perbandingan metode fitting: OLS (cepat, sensitif outlier), WLS (bobot manual), TLS (noise 2D), RANSAC (robust, sampling), IRLS (iteratif robust). Tabel: akurasi, robustness, kecepatan, use case.

**Slide 22**  Hough Transform mendalam: resolusi accumulator mempengaruhi akurasi. HoughLinesP (probabilistic) vs HoughLines (standard). Minimum votes, gap, minimum length. Diagram accumulator space.

**Slide 23**  Homography dan aplikasi: 4 pasang titik  8 persamaan. DLT (Direct Linear Transform). Aplikasi: AR overlay, document scanner, image rectification. Reprojection error sebagai metrik kualitas.

**Slide 24**  Optimasi dan Regularisasi mendalam: Gradient descent (batch, SGD). Learning rate scheduling. L1 menghasilkan sparse solution. Cross-validation mencegah overfitting. Bias-variance decomposition.

**Slide 25**  Kalman Filter mendalam: state-space model, transition matrix, measurement matrix. Process noise vs measurement noise. Tuning: Q besar  percaya measurement. R besar  percaya model. Aplikasi: tracking, sensor fusion.

**Slide 26**  Koneksi antar modul: Homography fitting (M5)  stitching (M6). RANSAC (M5)  feature matching verification (M4). Kalman filter (M5)  object tracking (M7). Denoising (M5)  image enhancement (M8).

**Slide 27**  Best practices: Selalu cek distribusi residual sebelum pilih metode. Gunakan RANSAC untuk data dengan outlier. Regularisasi untuk fitur banyak/data sedikit. Validasi model dengan cross-validation, bukan training error.

**Slide 28**  Aplikasi nyata: Document scanner (Hough + homography). Lane detection (Canny + Hough + RANSAC). QC industri (Hough circles + ellipse fitting). Image segmentation (GrabCut + K-Means). Visual tracking (Kalman filter).

**Slide 29**  Checklist kompetensi: OLS/WLS/TLS, RANSAC, Hough (garis+lingkaran), Homography, IRLS, Regularisasi, Template matching, GrabCut, K-Means, Kalman filter, Convex hull, Cross-validation, Denoising.

**Slide 30**  Kuis: (1) Berapa DOF homography? (2) Formula iterasi RANSAC? (3) Perbedaan Ridge dan Lasso? (4) Hough transform menggunakan ruang parameter apa? (5) Kalman filter terdiri dari tahap apa saja?

---

## PROMPT 3  Slide 3145 (Analisis + Project + Tugas Video)

Lanjutkan slide Modul 5, Slide 3145. Slide 3135: analisis lanjutan. Slide 3641: Project. Slide 4245: Tugas Video. Tiap slide ~300 kata.

**Slide 31**  Diskusi: Mengapa RANSAC lebih baik dari OLS untuk data dunia nyata? Bagaimana memilih threshold inlier? Trade-off iterasi vs akurasi. Kapan Hough lebih baik dari fitting langsung?

**Slide 32**  Perbandingan denoising: NLM (patch similarity), TV (gradient sparsity), bilateral (edge-preserving). PSNR dan SSIM sebagai metrik. Tabel: metode, kecepatan, kualitas, parameter.

**Slide 33**  Pipeline design: urutan operasi mempengaruhi hasil. Denoising  fitting  validasi. Modular design: fungsi terpisah per tahap. Error propagation antar tahap.

**Slide 34**  Advanced topics: Mean Shift clustering, DBSCAN. Connected components analysis. Multi-scale template matching. Elastic net (L1+L2). Fondasi untuk deep learning fitting.

**Slide 35**  Ringkasan materi: 13 konsep utama, 20 percobaan. Dari fitting sederhana (OLS) ke pipeline kompleks (percobaan 20). Penguasaan: estimasi parameter  robust estimation  optimasi  validasi.

**Slide 36**  Project "Aplikasi Model Fitting": 10 soal cerita. Integrasikan min 10 dari 20 konsep. Contoh: Penghitung Kendaraan (Hough+RANSAC+tracking), Document Scanner Pro (Houghhomographyperspektif), QC Bearing (circles+ellipse).

**Slide 37**  Soal cerita 47: Lane Keeping Assistant (edge+Hough+RANSAC+vanishing point), Tracking Video Olahraga (circle detection+optical flow), Peta Topografi (fitting kontur+interpolasi), Sistem Penghitung Sel (circle/ellipse fitting+counting).

**Slide 38**  Soal cerita 810: Deteksi Cacat Permukaan (template matching+anomaly), Sistem Registrasi Gambar Medis (RANSAC+homography), Dashboard Analisis Fitting Interaktif (GUI+semua metode).

**Slide 39**  20 improvisasi project: Multi-model Selector, Noise Robustness Benchmark, Real-time Lane Detector, Interactive RANSAC Visualizer, Panorama Homography, Template Matching Game, dll.

**Slide 40**  Rubrik Project: Fungsionalitas 35%, Integrasi 20%, Kode 15%, Dokumentasi 15%, Kreativitas 15%. Bonus +5 demo real-time, +3 integrasi >15 konsep. Format: NIM_Nama_Project05.zip.

**Slide 41**  Tips project: Gunakan class-based design. Setiap percobaan jadi modul terpisah. Unit test per fungsi. README dengan screenshot. Output visual disimpan ke folder output/.

**Slide 42**  Tugas Video: Demo 20 percobaan (4055 menit). Highlights: RANSAC dengan outlier, Hough detection, homography warp, Kalman tracking, pipeline gabungan.

**Slide 43**  Tugas Video Materi: Penjelasan 1015 menit. Wajib: diagram perbandingan metode fitting (OLS vs RANSAC vs Hough), pipeline model fitting, bias-variance tradeoff.

**Slide 44**  Tugas Video Project: Demo 1015 menit. Tunjukkan semua fitur, visual output, before/after. Analisis: metode mana paling efektif untuk problem apa.

**Slide 45**  Rubrik Video: Pembukaan 5%, Materi 15%, 20 Percobaan 40%, Project 20%, Analisis 10%, Kualitas 10%. Bonus +5 demo objek nyata. Penalti per percobaan error tanpa penjelasan. "Fit the Model, Solve the Problem!"
