# JOBSHEET MODUL 7: DETEKSI FITUR DAN PENCOCOKAN

---

## Tujuan Praktikum
1. Memahami konsep dan implementasi berbagai detektor fitur (Harris, Shi-Tomasi, SIFT, ORB, AKAZE, FAST).
2. Memahami dan membandingkan deskriptor fitur.
3. Mengimplementasikan feature matching (Brute-Force, FLANN, Ratio Test).
4. Mengimplementasikan geometric verification (Homography + RANSAC).
5. Membangun aplikasi berbasis feature matching.
6. Menganalisis invariansi fitur terhadap rotasi, skala, dan iluminasi.
7. Membandingkan properti deskriptor (SIFT, ORB, AKAZE) secara kuantitatif.
8. Mengimplementasikan Content-Based Image Retrieval (CBIR) dan AR marker detection.
9. Mengukur repeatability keypoint dan melakukan multi-image matching.
10. Membangun pipeline feature matching lengkap (OOP) dan proyek aplikasi akhir.

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

## Percobaan 11: Feature Invariance terhadap Skala

### Tujuan
Menguji invariansi detektor fitur terhadap perubahan skala gambar.

### Dasar Teori
Detektor berbasis scale-space (SIFT, AKAZE) dirancang untuk mendeteksi keypoints pada berbagai skala. Pengujian pada gambar yang di-resize membuktikan apakah keypoints konsisten terdeteksi di lokasi yang berkorespondensi.

### Langkah Kerja
1. Load gambar asli (resolusi tinggi, banyak fitur).
2. Buat 4 versi skala: 50%, 100%, 200%, 300%.
3. Deteksi keypoints menggunakan SIFT pada semua versi.
4. Deteksi keypoints menggunakan ORB pada semua versi.
5. Deteksi keypoints menggunakan AKAZE pada semua versi.
6. Hitung jumlah keypoints per skala per detektor.
7. Normalisasi posisi keypoints ke koordinat gambar asli.
8. Hitung keypoint correspondence: berapa keypoints pada versi skala cocok dengan keypoints gambar asli.
9. Plot jumlah keypoints vs skala untuk setiap detektor.
10. Buat tabel ringkasan: detektor × skala → jumlah kp, match rate, waktu.

### Analisis Percobaan 11
- Detektor mana yang paling konsisten menghasilkan keypoints di semua skala?
- Berapa persen keypoints yang berkorespondensi antara skala 50% dan 100%?
- Apakah skala 300% memperkenalkan keypoints palsu (noise)?
- Mengapa SIFT lebih invariant terhadap skala dibandingkan ORB?

---

## Percobaan 12: Feature Invariance terhadap Iluminasi

### Tujuan
Menguji ketahanan detektor dan deskriptor fitur terhadap perubahan kondisi pencahayaan.

### Dasar Teori
Perubahan iluminasi (brightness, contrast, gamma) mengubah nilai intensitas piksel tetapi seharusnya tidak mengubah struktur geometris fitur. Deskriptor yang baik harus robust terhadap variasi pencahayaan.

### Langkah Kerja
1. Load gambar asli.
2. Buat variasi brightness: +50, +100, −50, −100 (clamp 0–255).
3. Buat variasi contrast: ×0.5, ×1.0, ×1.5, ×2.0.
4. Buat variasi gamma: γ = 0.5, 1.0, 1.5, 2.0.
5. Deteksi + deskripsi (SIFT, ORB, AKAZE) pada semua variasi.
6. Match deskriptor setiap variasi terhadap gambar asli (FLANN + ratio test).
7. Hitung jumlah good matches per variasi per detektor.
8. Hitung match rate (good matches / total keypoints).
9. Visualisasikan matches pada kondisi terburuk.
10. Buat tabel dan plot: variasi iluminasi × detektor → match rate.

### Analisis Percobaan 12
- Pada perubahan brightness berapa matching mulai menurun drastis?
- Detektor mana yang paling robust terhadap perubahan contrast?
- Apakah gamma correction mempengaruhi semua detektor secara merata?
- Strategi preprocessing apa yang bisa meningkatkan robustness iluminasi?

---

## Percobaan 13: Perbandingan Deskriptor Fitur

### Tujuan
Membandingkan properti dan performa deskriptor SIFT, ORB, dan AKAZE secara komprehensif.

### Dasar Teori
Deskriptor fitur memiliki properti berbeda: SIFT menghasilkan vektor float 128-D, ORB menghasilkan binary 256-bit, AKAZE menghasilkan binary descriptor. Perbandingan meliputi dimensionalitas, kecepatan komputasi, dan akurasi matching.

### Langkah Kerja
1. Load pasangan gambar (scene yang sama, sudut sedikit berbeda).
2. Deteksi + deskripsi menggunakan SIFT, ORB, dan AKAZE.
3. Cetak properti deskriptor: shape, dtype, ukuran memori.
4. Ukur waktu deteksi + deskripsi (rata-rata 10 kali).
5. Match masing-masing deskriptor (BF matcher dengan norm yang sesuai).
6. Terapkan ratio test dan hitung jumlah good matches.
7. Hitung akurasi matching menggunakan ground truth (jika tersedia) atau manual inspection.
8. Visualisasikan distribusi jarak match (histogram per detektor).
9. Bandingkan ukuran memori total deskriptor per gambar.
10. Buat tabel perbandingan komprehensif: dimensi, tipe, waktu, jumlah match, akurasi, memori.

### Analisis Percobaan 13
- Deskriptor mana yang memiliki dimensi paling rendah namun akurasi match tinggi?
- Berapa kali lebih cepat ORB dibandingkan SIFT dalam deskripsi?
- Apakah binary descriptor (ORB, AKAZE) menghasilkan match yang sama baiknya dengan float (SIFT)?
- Dalam skenario apa masing-masing deskriptor paling unggul?

---

## Percobaan 14: Geometric Verification Detail

### Tujuan
Mengeksplorasi detail geometric verification: tuning RANSAC, analisis inlier, dan kualitas model fitting.

### Dasar Teori
RANSAC memiliki parameter yang mempengaruhi kualitas estimasi: threshold reproyeksi, jumlah iterasi, dan confidence level. Analisis inlier/outlier ratio serta residual error memberikan gambaran kualitas model homography.

### Langkah Kerja
1. Load pasangan gambar dengan bidang datar yang jelas.
2. Deteksi SIFT + match (FLANN + ratio test).
3. Estimasi homography dengan RANSAC threshold bervariasi: 1.0, 3.0, 5.0, 10.0, 20.0.
4. Untuk setiap threshold, catat: jumlah inlier, inlier ratio, reprojection error rata-rata.
5. Variasikan `maxIters` RANSAC: 100, 500, 1000, 5000.
6. Variasikan `confidence`: 0.9, 0.95, 0.99, 0.999.
7. Visualisasikan inlier (hijau) dan outlier (merah) untuk setiap konfigurasi.
8. Hitung reprojection error per match point.
9. Plot histogram reprojection error.
10. Bandingkan RANSAC vs LMEDS vs RHO: inlier count, error, waktu.

### Analisis Percobaan 14
- Pada threshold berapa inlier ratio optimal (banyak inlier, error rendah)?
- Apakah menambah iterasi RANSAC signifikan meningkatkan kualitas?
- Bagaimana perbandingan RANSAC vs LMEDS vs RHO untuk data Anda?
- Berapa reprojection error rata-rata pada konfigurasi terbaik?

---

## Percobaan 15: Image Retrieval

### Tujuan
Membangun sistem image retrieval sederhana menggunakan fitur lokal.

### Dasar Teori
Image retrieval menggunakan deskriptor fitur untuk mengindeks gambar dan menemukan gambar yang paling mirip dengan query. Pendekatan Bag of Visual Words (BoVW) atau direct feature matching dapat digunakan.

### Langkah Kerja
1. Kumpulkan database 10+ gambar dari berbagai kategori (bangunan, objek, pemandangan).
2. Ekstrak deskriptor SIFT dari semua gambar database.
3. Simpan deskriptor beserta metadata gambar.
4. Pilih 5 gambar query (3 ada di database dengan sudut berbeda, 2 tidak ada).
5. Untuk setiap query, match terhadap semua gambar database.
6. Hitung similarity score: jumlah good matches per gambar database.
7. Ranking gambar database berdasarkan similarity score.
8. Visualisasikan top-3 retrieved images per query.
9. Evaluasi: apakah gambar yang benar ada di top-1 / top-3?
10. Ukur waktu retrieval dan diskusikan skalabilitas.

### Analisis Percobaan 15
- Berapa akurasi top-1 dan top-3 retrieval?
- Apakah query dari sudut berbeda berhasil menemukan gambar yang benar?
- Bagaimana waktu retrieval meningkat seiring bertambahnya database?
- Apa kelemahan pendekatan direct matching untuk retrieval skala besar?

---

## Percobaan 16: AR Marker Detection

### Tujuan
Mendeteksi marker planar dan melakukan overlay konten virtual menggunakan homography.

### Dasar Teori
AR marker detection mendeteksi gambar referensi (marker) pada scene, mengestimasi homography antara marker dan deteksi pada scene, lalu menggunakan homography untuk mewarpkan konten overlay ke posisi marker.

### Langkah Kerja
1. Siapkan gambar marker (gambar dengan banyak fitur, misal: poster, cover buku).
2. Siapkan gambar overlay (logo, gambar AR yang ingin ditampilkan).
3. Foto marker dalam scene nyata dari berbagai sudut.
4. Deteksi fitur (SIFT/ORB) pada marker dan scene.
5. Match fitur dan estimasi homography.
6. Jika cukup inlier, warp gambar overlay ke area marker pada scene.
7. Blend overlay dengan scene (alpha blending atau seamless clone).
8. Uji pada 5 scene dengan sudut, jarak, dan pencahayaan berbeda.
9. Ukur keberhasilan overlay (visual inspection + inlier count).
10. Tampilkan grid hasil: scene asli → deteksi → overlay.

### Analisis Percobaan 16
- Pada sudut berapa overlay mulai terdistorsi atau gagal?
- Berapa minimum inlier agar overlay stabil?
- Apakah pencahayaan mempengaruhi akurasi overlay?
- Apa perbedaan menggunakan SIFT vs ORB untuk AR marker detection?

---

## Percobaan 17: Keypoint Repeatability

### Tujuan
Mengukur secara kuantitatif berapa banyak keypoints yang terdeteksi ulang pada gambar yang ditransformasi.

### Dasar Teori
Repeatability adalah metrik fundamental untuk mengevaluasi detektor fitur. Repeatability rate mengukur proporsi keypoints yang terdeteksi di kedua gambar (asli dan transformasi) pada lokasi yang berkorespondensi (dalam toleransi tertentu).

### Langkah Kerja
1. Load gambar asli.
2. Buat transformasi: rotasi (30°, 60°, 90°), skala (0.5, 1.5, 2.0), blur (σ=1, 2, 3), noise (σ=10, 25, 50).
3. Deteksi keypoints (SIFT, ORB, AKAZE) pada gambar asli dan setiap transformasi.
4. Transformasi balik posisi keypoints dari gambar transformasi ke koordinat gambar asli.
5. Hitung repeatability: keypoints yang jarak-nya < ε piksel (misal ε=5) dari keypoint asli.
6. Hitung repeatability rate = matched keypoints / min(kp_asli, kp_transformasi).
7. Plot repeatability rate vs tipe transformasi per detektor.
8. Plot repeatability rate vs intensitas transformasi (misal vs sudut rotasi).
9. Identifikasi transformasi mana yang paling menurunkan repeatability.
10. Buat tabel rangkuman: detektor × transformasi → repeatability rate.

### Analisis Percobaan 17
- Detektor mana yang memiliki repeatability tertinggi secara keseluruhan?
- Transformasi apa yang paling menurunkan repeatability?
- Apakah ada korelasi antara jumlah keypoints dan repeatability?
- Bagaimana noise mempengaruhi repeatability dibandingkan transformasi geometris?

---

## Percobaan 18: Multi-Image Matching

### Tujuan
Mencocokkan fitur secara simultan pada 3 atau lebih gambar.

### Dasar Teori
Multi-image matching dibutuhkan untuk aplikasi seperti panorama stitching dan 3D reconstruction. Matching dilakukan secara pairwise, lalu track fitur yang konsisten melintasi banyak gambar (feature tracks).

### Langkah Kerja
1. Ambil 4–5 gambar dari scene yang sama dengan overlap berturutan.
2. Deteksi fitur (SIFT) pada semua gambar.
3. Lakukan pairwise matching: gambar 1↔2, 2↔3, 3↔4, dst.
4. Juga lakukan matching non-berturutan: 1↔3, 2↔4 (jika overlap ada).
5. Bangun feature tracks: fitur yang muncul di ≥3 gambar.
6. Visualisasikan graph konektivitas antar gambar (jumlah match per pasangan).
7. Visualisasikan feature tracks pada gambar-gambar yang terhubung.
8. Hitung statistik: jumlah tracks, panjang rata-rata track, coverage.
9. Identifikasi pasangan gambar dengan match terkuat dan terlemah.
10. Buat tabel: pasangan gambar → jumlah matches, inlier ratio, homography quality.

### Analisis Percobaan 18
- Berapa banyak feature tracks yang muncul di ≥3 gambar?
- Apakah matching non-berturutan menghasilkan matches bermakna?
- Pasangan gambar mana yang memiliki konektivitas terkuat?
- Apa tantangan utama dalam multi-image matching dibandingkan dua gambar?

---

## Percobaan 19: Feature Matching Pipeline Lengkap

### Tujuan
Mengimplementasikan pipeline feature matching end-to-end menggunakan pendekatan berbasis class (OOP).

### Dasar Teori
Pipeline yang terstruktur memudahkan eksperimen dan deployment. Class-based design memungkinkan penggantian komponen (detektor, deskriptor, matcher, verifier) secara modular.

### Langkah Kerja
1. Desain class `FeatureMatchingPipeline` dengan method: `detect()`, `describe()`, `match()`, `verify()`.
2. Implementasikan constructor yang menerima parameter: detector_type, matcher_type, ratio_threshold, ransac_threshold.
3. Implementasikan method `detect()` yang mendukung SIFT, ORB, dan AKAZE.
4. Implementasikan method `match()` yang mendukung BF dan FLANN + ratio test.
5. Implementasikan method `verify()` untuk homography estimation + RANSAC.
6. Tambahkan method `evaluate()` yang menghitung precision, recall, F1.
7. Tambahkan method `visualize()` untuk menampilkan hasil matching.
8. Uji pipeline pada 3 pasangan gambar dengan konfigurasi berbeda.
9. Bandingkan minimal 3 konfigurasi pipeline (kombinasi detektor + matcher).
10. Dokumentasikan API class dan buat contoh penggunaan lengkap.

### Analisis Percobaan 19
- Apakah desain modular memudahkan perbandingan konfigurasi?
- Konfigurasi pipeline mana yang menghasilkan performa terbaik?
- Apa keuntungan class-based approach dibandingkan script prosedural?
- Komponen mana yang paling mempengaruhi performa keseluruhan pipeline?

---

## Percobaan 20: Proyek Feature Matching App

### Tujuan
Membangun aplikasi feature matching lengkap yang mengintegrasikan seluruh konsep dari Percobaan 1–19.

### Dasar Teori
Proyek akhir menggabungkan semua teknik: deteksi fitur, deskripsi, matching, geometric verification, dan evaluasi menjadi satu aplikasi yang fungsional dan user-friendly.

### Langkah Kerja
1. Pilih domain aplikasi: image retrieval, AR, object detection, atau document matching.
2. Bangun database referensi minimal 10 gambar.
3. Implementasikan pipeline lengkap menggunakan class dari Percobaan 19.
4. Tambahkan UI sederhana (command-line menu atau GUI dasar).
5. Implementasikan mode batch processing untuk banyak query.
6. Tambahkan logging dan reporting (hasil per query, statistik keseluruhan).
7. Evaluasi pada 20 query gambar (10 match, 10 non-match).
8. Hitung metrik: precision, recall, F1-score, waktu rata-rata per query.
9. Optimasi: pilih konfigurasi terbaik berdasarkan evaluasi.
10. Dokumentasikan aplikasi: README, screenshot, instruksi penggunaan.

### Analisis Percobaan 20
- Apakah aplikasi berhasil menangani semua test case?
- Berapa F1-score keseluruhan aplikasi?
- Apa bottleneck utama dalam hal kecepatan?
- Fitur apa yang bisa ditambahkan untuk meningkatkan aplikasi ke level produksi?

---

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Perbandingan detektor fitur (Harris, Shi-Tomasi, SIFT, ORB, AKAZE, FAST).
2. Perbandingan matcher (BF, FLANN) dan filtering (crosscheck, ratio test).
3. Efektivitas RANSAC untuk geometric verification.
4. Invariansi fitur terhadap rotasi, skala, dan perubahan iluminasi.
5. Perbandingan kuantitatif deskriptor (dimensi, kecepatan, akurasi).
6. Efektivitas geometric verification dan tuning RANSAC.
7. Penerapan image retrieval dan AR marker detection.
8. Repeatability keypoint di berbagai kondisi transformasi.
9. Strategi multi-image matching dan pipeline end-to-end.
10. Rekomendasi kombinasi optimal untuk berbagai use case.
11. Keterbatasan feature-based methods dan potensi alternatif (deep learning features).

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`).
- **Naming**: `NIM_Nama_Modul07.ipynb`
