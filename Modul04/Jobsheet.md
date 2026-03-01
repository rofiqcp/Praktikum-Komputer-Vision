# JOBSHEET PRAKTIKUM
# MODUL 4: MODEL FITTING DAN OPTIMASI

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa mampu:
1. Mendeteksi fitur (corner, keypoint) pada gambar.
2. Melakukan feature matching antar dua gambar.
3. Mengimplementasikan RANSAC untuk estimasi model yang robust.
4. Mendeteksi garis menggunakan Hough Transform.
5. Mendeteksi lingkaran menggunakan Hough Circle Transform.
6. Mengestimasi homografi dari pasangan titik korespondensi.
7. Melakukan koreksi perspektif menggunakan homografi.
8. Mengestimasi optical flow (sparse dan dense).
9. Mengimplementasikan interpolasi scattered data (RBF).
10. Menerapkan denoising berbasis regularisasi variasional.

---

## 2. ALAT DAN BAHAN

### Perangkat Lunak
- Python 3.8+, OpenCV, NumPy, Matplotlib, SciPy

### Dataset
- Gambar sample (bangunan, jalan, objek geometris, pasangan stereo).
- Video pendek untuk optical flow.

---

## 3. LANGKAH KERJA

### Percobaan 1: Deteksi Fitur (Corner & Keypoint)

**Tujuan**: Mendeteksi titik-titik fitur penting pada gambar.

**Langkah Kerja**:
1. Buat file `01_feature_detection.py`.
2. Baca gambar dan konversi ke grayscale.
3. Terapkan Harris Corner Detection: `cv2.cornerHarris()`.
4. Terapkan Shi-Tomasi (Good Features to Track): `cv2.goodFeaturesToTrack()`.
5. Terapkan SIFT: `cv2.SIFT_create()` → `detect()`.
6. Terapkan ORB: `cv2.ORB_create()` → `detect()`.
7. Visualisasikan keypoint menggunakan `cv2.drawKeypoints()`.
8. Bandingkan jumlah dan distribusi keypoint tiap metode.
9. Variasikan parameter (threshold, nFeatures, dll.).
10. Simpan visualisasi perbandingan.

---

### Percobaan 2: Feature Matching

**Tujuan**: Mencocokkan fitur antar dua gambar.

**Langkah Kerja**:
1. Buat file `02_feature_matching.py`.
2. Baca dua gambar yang memiliki area overlap.
3. Deteksi keypoint dan deskriptor (SIFT atau ORB).
4. Terapkan Brute-Force Matching: `cv2.BFMatcher()`.
5. Terapkan FLANN-based Matching: `cv2.FlannBasedMatcher()`.
6. Terapkan ratio test (Lowe's ratio): filter match dengan ratio < 0.75.
7. Visualisasikan match: `cv2.drawMatches()` dan `cv2.drawMatchesKnn()`.
8. Bandingkan jumlah dan kualitas match: BF vs FLANN, SIFT vs ORB.
9. Terapkan cross-check matching.
10. Simpan hasil visualisasi.

---

### Percobaan 3: RANSAC

**Tujuan**: Implementasi RANSAC untuk estimasi model garis yang robust terhadap outlier.

**Langkah Kerja**:
1. Buat file `03_ransac.py`.
2. Generate data sintetis: titik-titik pada garis + outlier acak.
3. Implementasikan RANSAC dari scratch:
   - Random sampling 2 titik → fit garis.
   - Hitung inlier (jarak < threshold).
   - Iterasi N kali, simpan model terbaik.
4. Visualisasikan: titik data, garis RANSAC, inlier vs outlier.
5. Bandingkan hasil RANSAC vs OLS (least squares tanpa RANSAC).
6. Variasikan rasio outlier (10%, 30%, 50%, 70%) dan amati robustness.
7. Variasikan threshold ε dan jumlah iterasi N.
8. Terapkan RANSAC pada data 2D (fit garis dan lingkaran).
9. Gunakan `cv2.findHomography(..., cv2.RANSAC)` untuk contoh built-in.
10. Plot: jumlah iterasi vs akurasi.

---

### Percobaan 4: Hough Transform — Deteksi Garis

**Tujuan**: Mendeteksi garis lurus pada gambar menggunakan Hough Transform.

**Langkah Kerja**:
1. Buat file `04_hough_lines.py`.
2. Baca gambar (bangunan, jalan, papan tulis).
3. Preprocessing: grayscale → blur → Canny edge.
4. Terapkan Standard Hough Transform: `cv2.HoughLines()`.
5. Terapkan Probabilistic Hough Transform: `cv2.HoughLinesP()`.
6. Gambar garis yang terdeteksi di atas gambar asli.
7. Variasikan parameter: `threshold`, `minLineLength`, `maxLineGap`.
8. Filter garis berdasarkan sudut (horizontal, vertikal, diagonal).
9. Hitung intersection antar garis.
10. Terapkan pada gambar jalanan untuk lane detection sederhana.

---

### Percobaan 5: Hough Transform — Deteksi Lingkaran

**Tujuan**: Mendeteksi lingkaran pada gambar.

**Langkah Kerja**:
1. Buat file `05_hough_circles.py`.
2. Baca gambar yang mengandung objek bulat (koin, bola, iris mata).
3. Preprocessing: grayscale → median blur.
4. Terapkan `cv2.HoughCircles()` dengan metode `cv2.HOUGH_GRADIENT`.
5. Gambar lingkaran dan center yang terdeteksi.
6. Variasikan parameter: `dp`, `minDist`, `param1`, `param2`, `minRadius`, `maxRadius`.
7. Terapkan pada gambar koin → hitung jumlah koin.
8. Terapkan pada gambar roda/ban.
9. Filter false positive berdasarkan ukuran dan posisi.
10. Simpan hasil deteksi dengan anotasi.

---

### Percobaan 6: Estimasi Homografi

**Tujuan**: Mengestimasi matriks homografi dari titik korespondensi.

**Langkah Kerja**:
1. Buat file `06_homography.py`.
2. Baca dua gambar dengan area overlap (misal: buku dari dua sudut).
3. Deteksi dan match fitur (SIFT + FLANN + ratio test).
4. Estimasi homografi: `H, mask = cv2.findHomography(src, dst, cv2.RANSAC)`.
5. Visualisasikan inlier matches.
6. Warp gambar pertama ke perspektif gambar kedua: `cv2.warpPerspective()`.
7. Overlay hasil warp dengan gambar kedua.
8. Variasikan metode: `cv2.RANSAC` vs `cv2.LMEDS` vs `0` (least squares).
9. Bandingkan akurasi dan robustness tiap metode.
10. Coba dengan jumlah match yang berbeda (4, 10, 50, 100+).

---

### Percobaan 7: Koreksi Perspektif

**Tujuan**: Menerapkan homografi untuk mengoreksi perspektif objek planar.

**Langkah Kerja**:
1. Buat file `07_perspective_correction.py`.
2. Baca gambar dokumen/buku/papan yang diambil dari sudut miring.
3. Deteksi sudut objek secara manual (klik mouse) atau otomatis (contour).
4. Definisikan titik tujuan (persegi panjang).
5. Hitung homografi dan warp.
6. Implementasikan auto-detection sudut dokumen:
   - Canny edge → contour → approxPolyDP → 4 sudut.
7. Terapkan pada beberapa gambar dokumen.
8. Tambahkan validasi: pastikan 4 titik membentuk convex quadrilateral.
9. Enhancement setelah warp: sharpen, contrast adjust.
10. Simpan hasil before-after.

---

### Percobaan 8: Optical Flow

**Tujuan**: Mengestimasi gerakan piksel antar dua frame berurutan.

**Langkah Kerja**:
1. Buat file `08_optical_flow.py`.
2. Baca video atau buat dua frame sintetis dengan displacement.
3. Sparse optical flow (Lucas-Kanade):
   - Deteksi fitur di frame 1: `cv2.goodFeaturesToTrack()`.
   - Track ke frame 2: `cv2.calcOpticalFlowPyrLK()`.
   - Gambar trail gerakan.
4. Dense optical flow (Farnebäck):
   - `cv2.calcOpticalFlowFarneback()`.
   - Visualisaikan sebagai HSV (hue = arah, value = magnitude).
5. Bandingkan sparse vs dense flow secara visual.
6. Simpan visualisasi optical flow.
7. Hitung magnitude rata-rata flow sebagai indikator jumlah gerakan.
8. Terapkan pada video real (webcam atau file).
9. Implementasikan motion detection sederhana berdasarkan flow magnitude.
10. Simpan visualisasi frame-by-frame.

---

### Percobaan 9: Interpolasi Scattered Data (RBF)

**Tujuan**: Menggunakan Radial Basis Function untuk interpolasi data tersebar.

**Langkah Kerja**:
1. Buat file `09_scattered_interpolation_rbf.py`.
2. Generate titik-titik sampel acak dengan nilai (misal: elevasi, suhu).
3. Gunakan `scipy.interpolate.Rbf` untuk interpolasi.
4. Buat grid reguler dan evaluasi interpolasi pada grid.
5. Visualisasikan surface interpolasi (colormap 2D atau 3D).
6. Bandingkan kernel RBF: `multiquadric`, `gaussian`, `linear`, `thin_plate`.
7. Terapkan pada kasus: rekonstruksi depth map dari sparse points.
8. Terapkan pada kasus: smooth displacement field dari sparse matches.
9. Variasikan jumlah titik sampel dan amati kualitas interpolasi.
10. Bandingkan dengan `scipy.interpolate.griddata`.

---

### Percobaan 10: Denoising dengan Regularisasi Variasional

**Tujuan**: Menerapkan Total Variation (TV) regularization untuk image denoising.

**Langkah Kerja**:
1. Buat file `10_variational_denoising.py`.
2. Baca gambar bersih dan tambahkan Gaussian noise.
3. Implementasikan TV denoising sederhana menggunakan iterasi:
   - Minimize: $E(u) = \|u - f\|^2 + \lambda \|\nabla u\|_1$.
4. Gunakan `skimage.restoration.denoise_tv_chambolle` atau implementasi manual.
5. Variasikan λ (weight regulaisasi) dan amati trade-off smoothing vs detail.
6. Bandingkan dengan Gaussian blur dan bilateral filter.
7. Hitung PSNR untuk setiap metode.
8. Visualisasikan per-pixel error map.
9. Terapkan pada gambar medis noisy.
10. Buat grafik: λ vs PSNR.

---

## 4. ANALISIS

### Analisis Percobaan 1 — Deteksi Fitur
- Bandingkan distribusi dan repeatability fitur: Harris vs Shi-Tomasi vs SIFT vs ORB.
- Metode mana yang paling banyak mendeteksi fitur? Apakah lebih banyak = lebih baik?

### Analisis Percobaan 2 — Feature Matching
- Berapa persen match yang benar (setelah ratio test)?
- Bandingkan kecepatan BF vs FLANN.

### Analisis Percobaan 3 — RANSAC
- Pada rasio outlier berapa RANSAC mulai gagal?
- Bagaimana threshold ε mempengaruhi jumlah inlier?

### Analisis Percobaan 4 — Hough Lines
- Bandingkan Standard vs Probabilistic Hough: Pro dan kontra.
- Bagaimana kualitas edge detection mempengaruhi deteksi garis?

### Analisis Percobaan 5 — Hough Circles
- False positive apa yang sering muncul? Bagaimana meminimalkannya?
- Pengaruh preprocessing terhadap akurasi deteksi.

### Analisis Percobaan 6 — Homografi
- Bandingkan akurasi RANSAC vs LMEDS vs plain least squares.
- Minimal berapa pasang match untuk homografi yang baik?

### Analisis Percobaan 7 — Koreksi Perspektif
- Seberapa akurat auto-detection sudut dibanding manual?
- Apa kelemahan method contour-based detection?

### Analisis Percobaan 8 — Optical Flow
- Kapan Lucas-Kanade gagal (apa asumsinya)?
- Bandingkan sparse vs dense flow: kapan masing-masing cocok?

### Analisis Percobaan 9 — RBF Interpolation
- Kernel RBF mana yang menghasilkan interpolasi paling smooth? Paling akurat?
- Bagaimana densitas titik sampel mempengaruhi kualitas?

### Analisis Percobaan 10 — TV Denoising
- Pada λ berapa terjadi balance optimal antara smoothing dan detail?
- Bandingkan TV vs Gaussian vs bilateral filter: metrik dan visual.

---

## 5. KESIMPULAN

Buatlah kesimpulan yang mencakup:
1. Perbandingan metode deteksi fitur dan matching.
2. Keunggulan RANSAC untuk data dengan outlier.
3. Aplikasi Hough Transform dan keterbatasannya.
4. Peran homografi dalam computer vision.
5. Trade-off dalam regularisasi dan denoising.
