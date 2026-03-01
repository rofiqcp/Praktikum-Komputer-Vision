# JOBSHEET MODUL 7: DETEKSI FITUR DAN PENCOCOKAN

---

## Tujuan Praktikum
1. Memahami konsep dan implementasi berbagai detektor fitur (Harris, Shi-Tomasi, SIFT, ORB, AKAZE, FAST).
2. Memahami dan membandingkan deskriptor fitur.
3. Mengimplementasikan feature matching (Brute-Force, FLANN, Ratio Test).
4. Mengimplementasikan geometric verification (Homography + RANSAC).
5. Membangun aplikasi berbasis feature matching.

---

## Alat dan Bahan
- **Hardware**: PC/Laptop, webcam, objek fisik untuk foto (buku, kartu, poster).
- **Software**: Python 3.8+, Jupyter Notebook / VS Code.
- **Library**: OpenCV (`opencv-contrib-python`), NumPy, Matplotlib.
- **Dataset**: Pasangan gambar dengan overlapping area, gambar objek dari sudut berbeda.

---

## Percobaan 1: Harris Corner Detection

### Tujuan
Mendeteksi corners menggunakan Harris Corner Detector dan memahami parameter-nya.

### Dasar Teori
Harris Corner menghitung auto-correlation matrix dari gradien intensitas. Corner terdeteksi di titik dengan response function $R > \text{threshold}$.

### Langkah Kerja
1. Load gambar (pilih gambar dengan banyak sudut: checkerboard, bangunan).
2. Konversi ke grayscale dan float32.
3. Terapkan `cv2.cornerHarris()` dengan `blockSize=2, ksize=3, k=0.04`.
4. Dilasi hasil untuk memperbesar titik corner.
5. Threshold: tandai piksel dengan $R > 0.01 \times R_{max}$ sebagai corner.
6. Tampilkan corner berwarna merah pada gambar asli.
7. Variasikan `blockSize` (2, 3, 5, 7) → amati perbedaan.
8. Variasikan `k` (0.01, 0.04, 0.06, 0.1) → amati perbedaan.
9. Variasikan threshold (0.001, 0.01, 0.05, 0.1) → amati jumlah corner.
10. Tampilkan heatmap response function + corner overlay dalam grid 2×3.

### Analisis Percobaan 1
- Bagaimana `blockSize` mempengaruhi lokasi dan jumlah corner?
- Nilai `k` berapa yang menghasilkan corner paling stabil?
- Pada threshold berapa terjadi trade-off antara false positive dan false negative?
- Apakah Harris mendeteksi corner pada tepi objek melengkung?

---

## Percobaan 2: Shi-Tomasi Corner Detection

### Tujuan
Mendeteksi corners menggunakan Shi-Tomasi (Good Features to Track) dan membandingkan dengan Harris.

### Dasar Teori
Shi-Tomasi menggunakan $\min(\lambda_1, \lambda_2)$ sebagai response function. Corner terdeteksi jika minimum eigenvalue di atas threshold.

### Langkah Kerja
1. Load gambar yang sama dengan Percobaan 1.
2. Terapkan `cv2.goodFeaturesToTrack(maxCorners=100, qualityLevel=0.01, minDistance=10)`.
3. Visualisasikan corners sebagai lingkaran hijau pada gambar.
4. Variasikan `maxCorners` (25, 50, 100, 500).
5. Variasikan `qualityLevel` (0.001, 0.01, 0.05, 0.1).
6. Variasikan `minDistance` (5, 10, 20, 50).
7. Overlay Shi-Tomasi corners (hijau) dan Harris corners (merah) pada gambar yang sama.
8. Hitung overlap: berapa corner yang terdeteksi oleh kedua metode.
9. Uji pada gambar dengan tekstur kompleks (misal: daun, kain).
10. Tampilkan perbandingan Harris vs Shi-Tomasi dalam format tabel + visual.

### Analisis Percobaan 2
- Apakah Shi-Tomasi menghasilkan distribusi corner yang lebih merata?
- Pada setting apa Shi-Tomasi dan Harris menghasilkan corner yang sama?
- Bagaimana `minDistance` mempengaruhi distribusi spasial corner?
- Metode mana yang lebih cocok untuk tracking? Mengapa?

---

## Percobaan 3: SIFT Feature Detection dan Description

### Tujuan
Mendeteksi keypoints dan mengekstrak deskriptor menggunakan SIFT.

### Dasar Teori
SIFT mendeteksi blob pada scale-space (DoG), merefine lokasi sub-pixel, menentukan orientasi, dan menghitung deskriptor 128-D yang invariant terhadap skala dan rotasi.

### Langkah Kerja
1. Load gambar dan konversi ke grayscale.
2. Buat SIFT detector: `cv2.SIFT_create()`.
3. Deteksi keypoints dan compute descriptors.
4. Visualisasikan keypoints dengan `drawKeypoints` (flag `DRAW_RICH_KEYPOINTS` untuk ukuran + orientasi).
5. Cetak informasi keypoint: jumlah, ukuran rata-rata, orientasi.
6. Variasikan `nfeatures` (100, 500, 1000, 2000).
7. Variasikan `contrastThreshold` (0.01, 0.04, 0.08) dan `edgeThreshold` (5, 10, 20).
8. Analisis distribusi ukuran keypoint (histogram).
9. Analisis distribusi orientasi keypoint (polar plot).
10. Uji pada gambar yang di-scale (50%, 100%, 200%) → deteksi di ketiga gambar → bandingkan keypoints.

### Analisis Percobaan 3
- Berapa keypoints terdeteksi pada setting default?
- Bagaimana `contrastThreshold` mempengaruhi jumlah keypoints?
- Apakah keypoints SIFT konsisten pada gambar yang di-scale? Tunjukkan bukti.
- Apa distribusi orientasi yang dominan pada gambar Anda?

---

## Percobaan 4: ORB Feature Detection dan Description

### Tujuan
Mendeteksi keypoints dan mengekstrak deskriptor binary menggunakan ORB.

### Dasar Teori
ORB menggabungkan FAST detector (cepat) dengan rotated BRIEF descriptor (binary, efisien). ORB gratis dan cocok untuk aplikasi real-time.

### Langkah Kerja
1. Buat ORB detector: `cv2.ORB_create(nfeatures=500)`.
2. Deteksi keypoints dan compute descriptors.
3. Visualisasikan keypoints (rich keypoints).
4. Cetak info: jumlah keypoints, descriptor shape, dtype.
5. Variasikan `nfeatures` (100, 500, 1000, 5000).
6. Variasikan `scaleFactor` (1.1, 1.2, 1.5) dan `nlevels` (4, 8, 12).
7. Bandingkan distribusi spasial ORB vs SIFT pada gambar yang sama.
8. Ukur waktu deteksi ORB vs SIFT (10x percobaan, rata-rata).
9. Uji pada gambar yang dirotasi (0°, 45°, 90°, 180°) → deteksi → bandingkan.
10. Tampilkan side-by-side: ORB vs SIFT keypoints + timing table.

### Analisis Percobaan 4
- Berapa kali ORB lebih cepat dari SIFT?
- Apakah distribusi keypoints ORB vs SIFT berbeda? Bagaimana?
- Apakah ORB keypoints konsisten pada rotasi gambar?
- Dalam kondisi apa ORB lebih cocok daripada SIFT?

---

## Percobaan 5: AKAZE dan FAST Detection

### Tujuan
Mengeksplorasi detektor AKAZE (nonlinear scale space) dan FAST (corner cepat).

### Dasar Teori
AKAZE menggunakan nonlinear diffusion (Perona-Malik) untuk scale space yang mempertahankan boundary lebih baik. FAST mendeteksi corner dengan membandingkan intensitas pada circle radius 3.

### Langkah Kerja
1. Buat AKAZE detector: `cv2.AKAZE_create()`.
2. Deteksi keypoints + descriptors AKAZE.
3. Buat FAST detector: `cv2.FastFeatureDetector_create(threshold=25)`.
4. Deteksi keypoints FAST (tanpa descriptor).
5. Visualisasikan keduanya side-by-side.
6. Variasikan FAST threshold (10, 25, 50, 100).
7. Variasikan AKAZE `descriptor_type` dan `threshold`.
8. Ukur timing: SIFT vs ORB vs AKAZE vs FAST.
9. Hitung dan plot jumlah keypoints per detektor (bar chart).
10. Buat tabel perbandingan komprehensif: jumlah kp, waktu, tipe descriptor, dimensi.

### Analisis Percobaan 5
- Detektor mana yang paling banyak menghasilkan keypoints?
- Detektor mana yang paling cepat?
- Bagaimana AKAZE dibandingkan SIFT dalam lokasi keypoint?
- Kapan FAST lebih cocok digunakan (meskipun tanpa deskriptor)?

---

## Percobaan 6: Brute-Force Feature Matching

### Tujuan
Mencocokkan fitur antara dua gambar menggunakan Brute-Force Matcher.

### Dasar Teori
BF Matcher menghitung jarak antara setiap deskriptor query dan semua deskriptor training. Crosscheck memastikan kecocokan bilateral.

### Langkah Kerja
1. Load dua gambar dari scene yang sama (sudut sedikit berbeda atau overlap).
2. Deteksi keypoints + descriptors (SIFT) pada kedua gambar.
3. Buat BFMatcher dengan `cv2.NORM_L2` dan `crossCheck=True`.
4. Match dan sort berdasarkan distance.
5. Visualisasikan top 20, 50, dan 100 matches.
6. Plot histogram jarak match.
7. Ulangi dengan ORB: BFMatcher `cv2.NORM_HAMMING`.
8. Bandingkan match SIFT vs ORB (jumlah, jarak rata-rata, visual).
9. Uji pada pasangan gambar dengan sudut pandang sangat berbeda.
10. Uji pada pasangan gambar dengan perubahan iluminasi.

### Analisis Percobaan 6
- Berapa good matches untuk SIFT vs ORB?
- Bagaimana distribusi jarak match (histogram)?
- Pada sudut pandang berapa matching mulai gagal?
- Apakah crosscheck efektif mengurangi false matches?

---

## Percobaan 7: FLANN Matching + Lowe's Ratio Test

### Tujuan
Mengimplementasikan FLANN matcher yang lebih cepat dan Lowe's ratio test untuk filtering.

### Dasar Teori
FLANN menggunakan approximate nearest neighbor search. Lowe's ratio test membuang match yang ambiguous: jika match terbaik tidak secara signifikan lebih baik dari match kedua, discard.

### Langkah Kerja
1. Load pasangan gambar (gunakan yang sama dengan Percobaan 6).
2. Deteksi SIFT keypoints + descriptors.
3. Setup FLANN matcher (index KDTREE, 5 trees, 50 checks).
4. KNN match dengan k=2.
5. Terapkan Lowe's ratio test dengan `ratio=0.75`.
6. Visualisasikan good matches setelah ratio test.
7. Variasikan ratio threshold (0.5, 0.6, 0.7, 0.8, 0.9).
8. Plot jumlah good matches vs ratio threshold.
9. Ukur waktu: BF matcher vs FLANN matcher.
10. Bandingkan kualitas match: BF+crosscheck vs FLANN+ratio test.

### Analisis Percobaan 7
- Pada ratio berapa trade-off antara jumlah match dan akurasi optimal?
- Berapa kali FLANN lebih cepat dari BF matcher?
- Apakah FLANN + ratio test menghasilkan match yang lebih bersih dari BF + crosscheck?
- Mengapa ratio test efektif menghilangkan false matches?

---

## Percobaan 8: Homography Estimation dengan RANSAC

### Tujuan
Mengestimasi transformasi homography menggunakan matched features + RANSAC.

### Dasar Teori
Homography memetakan titik di satu plane ke plane lain. RANSAC menemukan model terbaik meskipun ada outlier di antara matches.

### Langkah Kerja
1. Load dua gambar yang berisi bidang datar yang sama (buku, poster, papan).
2. Deteksi SIFT + match menggunakan FLANN + ratio test.
3. Ambil lokasi keypoints dari good matches.
4. Hitung homography: `cv2.findHomography(src, dst, cv2.RANSAC, 5.0)`.
5. Visualisasikan inliers (hijau) dan outliers (merah) pada gambar match.
6. Warp gambar pertama menggunakan homography → overlay pada gambar kedua.
7. Variasikan RANSAC threshold (1, 3, 5, 10, 20).
8. Plot jumlah inlier vs threshold.
9. Bandingkan RANSAC vs LMEDS (Least Median of Squares).
10. Gambar bounding box objek dari gambar 1 ke gambar 2 (perspektif).

### Analisis Percobaan 8
- Berapa persen inlier pada RANSAC threshold=5?
- Bagaimana threshold mempengaruhi akurasi homography?
- Kapan LMEDS lebih baik dari RANSAC?
- Apakah bounding box transformasi akurat pada objek target?

---

## Percobaan 9: Object Detection via Feature Matching

### Tujuan
Mendeteksi objek spesifik dalam scene menggunakan feature matching + homography.

### Dasar Teori
Dengan mencocokkan fitur antara gambar template objek dan gambar scene, lalu mengestimasi homography, kita bisa mendeteksi dan melokalisasi objek spesifik.

### Langkah Kerja
1. Ambil foto objek (buku/kartu/poster) sebagai template — latar polos.
2. Ambil foto scene yang mengandung objek tersebut dari sudut berbeda.
3. Deteksi fitur (SIFT/ORB) + match (FLANN/BF).
4. Estimasi homography (RANSAC).
5. Jika cukup inlier (>10), gambar bounding box objek di scene.
6. Uji pada 5 scene berbeda (sudut, jarak, pencahayaan bervariasi).
7. Implementasikan scoring: jumlah inlier, persentase, confidence.
8. Tentukan threshold minimum inlier untuk "objek terdeteksi".
9. Uji pada scene yang TIDAK mengandung objek → pastikan tidak false positive.
10. Buat tabel deteksi: scene | inlier | status (detected/not detected).

### Analisis Percobaan 9
- Berapa minimum inlier agar deteksi reliable?
- Pada sudut pandang (derajat) berapa deteksi mulai gagal?
- Apakah ada false positive pada scene tanpa objek?
- Fitur detektor mana (SIFT vs ORB) yang menghasilkan deteksi lebih robust?

---

## Percobaan 10: Real-World Application — Feature Matching Pipeline

### Tujuan
Membangun pipeline feature matching end-to-end untuk satu aplikasi nyata.

### Dasar Teori
Menggabungkan semua konsep: memilih detektor + deskriptor + matcher + verifikasi yang optimal untuk use case tertentu.

### Langkah Kerja
1. Pilih satu aplikasi: (a) Image retrieval, (b) AR marker detection, (c) Document matching, atau (d) Logo recognition.
2. Buat database: 5 referensi gambar.
3. Buat query: 10 gambar test (5 match, 5 non-match).
4. Implementasikan pipeline: detect → describe → match → verify → decide.
5. Uji 3 kombinasi detektor-matcher: (SIFT+FLANN, ORB+BF, AKAZE+BF).
6. Hitung akurasi per kombinasi: True Positive, False Positive, True Negative, False Negative.
7. Hitung precision, recall, F1-score per kombinasi.
8. Ukur waktu per kombinasi.
9. Pilih kombinasi terbaik berdasarkan F1 dan waktu.
10. Dokumentasikan pipeline final + hasil evaluasi dalam tabel ringkasan.

### Analisis Percobaan 10
- Kombinasi mana yang memiliki F1-score tertinggi?
- Kombinasi mana yang paling cepat?
- Apa trade-off antara akurasi dan kecepatan?
- Bagaimana Anda akan memilih metode untuk deployment real-world?

---

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Perbandingan detektor fitur (Harris, Shi-Tomasi, SIFT, ORB, AKAZE, FAST).
2. Perbandingan matcher (BF, FLANN) dan filtering (crosscheck, ratio test).
3. Efektivitas RANSAC untuk geometric verification.
4. Rekomendasi kombinasi optimal untuk berbagai use case.
5. Keterbatasan feature-based methods dan potensi alternatif (deep learning features).

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`).
- **Naming**: `NIM_Nama_Modul07.ipynb`
