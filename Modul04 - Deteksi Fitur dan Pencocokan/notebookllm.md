# NotebookLM Prompts — Modul 4: Deteksi Fitur dan Pencocokan

---

## PROMPT 1 — Slide 1-15 (Materi Dasar + Percobaan 1-8)

Buat 15 slide presentasi akademik Modul 4: Deteksi Fitur dan Pencocokan. Referensi Szeliski (2022) Ch.7. Tiap slide informatif, sertakan formula dan kode OpenCV Python.

**Slide 1** — Judul Modul 4: Fitur lokal = titik yang dapat diidentifikasi ulang secara konsisten di berbagai kondisi. Lebih baik dari pixel matching: sparsity (100-1000 poin vs jutaan piksel), semantik tinggi, robust terhadap oklusi parsial. Aplikasi: image stitching, AR, SLAM, object recognition, image retrieval.

**Slide 2** — Mengapa Fitur Lokal: Fitur baik bersifat repeatable (ditemukan ulang), distinctive (bisa dibedakan), dan compact (representasi kecil). Target invariansi: rotasi (orientasi assignment), skala (scale-space), iluminasi (normalisasi), viewpoint (affine invariance). Sparsity memungkinkan pencarian efisien di database besar.

**Slide 3** — Harris Corner Detection: Matriks $M = \sum w \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}$ dari gradien lokal. Respons $R = \det(M) - k\cdot\text{trace}(M)^2$ (k 0.04-0.06). Eigenvalue kecil = flat; satu besar = edge; dua besar = corner. cv2.cornerHarris(gray,blockSize,ksize,k).

**Slide 4** — Shi-Tomasi Corner: Kriteria $\min(\lambda_1,\lambda_2) > \text{threshold}$ lebih stabil dari Harris. cv2.goodFeaturesToTrack(gray,maxCorners,qualityLevel,minDistance) kembalikan array pojok terbaik. cornerSubPix merefine lokasi ke sub-piksel via iterasi least-square untuk presisi tinggi.

**Slide 5** — SIFT Overview: Scale-Invariant Feature Transform deteksi keypoint via extrema piramid DoG (5 oktaf x 4 skala). Eliminasi keypoint low-contrast dan edge via uji Hessian. Orientasi: histogram gradient dominan. Deskriptor 128-D HOG. Invarian rotasi dan skala penuh. David Lowe 2004.

**Slide 6** — SIFT Descriptor: Patch 16x16 dibagi 4x4 sub-blok; tiap blok histogram 8 orientasi = 4x4x8 = 128 float32. Normalisasi L2, clamp nilai > 0.2, normalisasi ulang untuk robustness iluminasi non-linear. sift.detectAndCompute(gray,None) kembalikan keypoints dan descriptors sekaligus.

**Slide 7** — ORB: Oriented FAST and Rotated BRIEF. FAST circle test 16 piksel deteksi corner. rBRIEF descriptor 256-bit binary; orientasi via moment centroid untuk invariansi rotasi. Jarak Hamming (XOR popcount) sangat cepat. Bebas paten, 40x lebih cepat dari SIFT, cocok real-time.

**Slide 8** — AKAZE dan FAST: AKAZE: nonlinear diffusion scale space lebih tajam dari Gaussian pyramid, descriptor M-LDB 61 bytes binary. FAST: uji 12 piksel dari 16 lingkaran; corner jika semua lebih terang/gelap dari pusat+threshold. Non-maximum suppression hapus duplikat terdekat.

**Slide 9** — Brute Force Matcher: BFMatcher(normType,crossCheck) bandingkan setiap descriptor query ke semua train. NORM_L2 untuk SIFT float, NORM_HAMMING untuk ORB binary. match() satu match terbaik. knnMatch(k=2) kembalikan 2 match terdekat untuk Lowe ratio test saring false positive.

**Slide 10** — Lowe Ratio Test: Jika $d_1/d_2 < \text{ratio}$ (biasanya 0.75) match dianggap reliabel karena jauh lebih dekat dari runner-up. Ratio lebih kecil = lebih ketat = presisi tinggi. Ratio lebih besar = recall tinggi. Trade-off precision-recall disesuaikan kebutuhan aplikasi konkrit.

**Slide 11** — FLANN Matching: Fast Library for Approximate Nearest Neighbors. SIFT float: FLANN_INDEX_KDTREE, trees=5, checks=50. ORB binary: FLANN_INDEX_LSH, table_number=6, key_size=12. 10-100x lebih cepat dari BF untuk dataset besar karena approximate search dengan struktur index efisien.

**Slide 12** — Homography Matrix: Transformasi proyektif planar $H$ 3x3 petakan titik sumber ke tujuan (koordinat homogenus). Minimal 4 pasang titik. Direct Linear Transform + SVD. cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5.0) kembalikan H dan mask inlier. cv2.warpPerspective terapkan transformasi.

**Slide 13** — RANSAC: Random Sample Consensus iteratif: pilih 4 pasang acak, hitung H (DLT+SVD), hitung reprojection error $||x'_i - Hx_i||$, count inlier (error < threshold). $N = \log(1-p)/\log(1-(1-\epsilon)^s)$; p=0.99. Output: H terbaik + mask inlier. Robust hingga 50% outlier.

**Slide 14** — Pipeline Lengkap Feature Matching: SIFT.detectAndCompute pada dua gambar. FLANN knnMatch (k=2) hasilkan 500 candidate. Ratio test (0.75) saring jadi 120 good match. findHomography RANSAC (5px) hasilkan 55-80 inlier. warpPerspective terapkan H untuk alignment atau stitching.

**Slide 15** — Ringkasan Bagian 1: Tabel 5 detektor: Harris (non-scale, L2, lambat), Shi-Tomasi (non-scale, L2, sedang), SIFT (128-float, L2, lambat, invarian penuh), ORB (256-bit, Hamming, cepat), AKAZE (488-bit, Hamming, sedang). Tips: ORB real-time, SIFT akurasi, AKAZE kompromi.

---

## PROMPT 2 — Slide 16-30 (Analisis + Perbandingan + Tips)

Lanjutkan presentasi Modul 4: Deteksi Fitur dan Pencocokan, slide 16-30. Fokus analisis mendalam, perbandingan, tips praktis, dan kuis.

**Slide 16** — Rekap Percobaan 1-10: Tabel: 01 Harris, 02 Shi-Tomasi+cornerSubPix, 03 SIFT, 04 ORB, 05 AKAZE+FAST, 06 BFMatcher, 07 FLANN, 08 findHomography+warpPerspective, 09 object detection via matching, 10 uji invariansi rotasi dan skala. Fungsi utama dan parameter penting tiap percobaan.

**Slide 17** — Analisis Invariansi Rotasi: SIFT hitung orientasi dari histogram gradient dominan, rotasi descriptor sesuai = invarian penuh. ORB gunakan moment centroid untuk orientasi, rotasi pola BRIEF sesuai. Grafik: jumlah match vs sudut rotasi 0-180 derajat untuk SIFT dan ORB berdampingan.

**Slide 18** — Analisis Invariansi Skala: SIFT scale-space 5 oktaf x 4 level sigma (faktor sqrt(2)), keypoint di extrema 3D piramid DoG. Grafik: match rate vs scale factor (0.3x-3.0x). ORB pyramid (nlevels=8, scaleFactor=1.2) efektif untuk rentang skala moderat (0.5x-2.0x).

**Slide 19** — Analisis Invariansi Iluminasi: SIFT normalisasi L2, clamp > 0.2, normalisasi ulang; robust terhadap brightness bias dan kontras. Grafik: match rate vs beta (-100 hingga +100) dan noise sigma (0-50). ORB lebih sensitif iluminasi karena descriptor binary tanpa normalisasi bertahap.

**Slide 20** — Perbandingan Deskriptor Lengkap: SIFT 128 float32, L2, akurat, lambat. ORB 32 uint8/256-bit, Hamming, cepat, gratis. AKAZE 61 uint8/488-bit, Hamming, akurasi antara. BRISK 64 uint8/512-bit, Hamming, kecepatan antara. Pilih berdasarkan hardware, lisensi, dan akurasi vs kecepatan.

**Slide 21** — Image Retrieval Berbasis Fitur: Query: detect descriptor, cocokkan ke N database via FLANN. Hitung inlier RANSAC per pasang. Rank berdasarkan inlier count. Precision@k = jumlah relevan di top-k dibagi k. Mean Average Precision (mAP) evaluasi sistem retrieval secara komprehensif.

**Slide 22** — Geometric Verification: RANSAC: pilih 4 titik, hitung H (DLT+SVD), reprojection error $e_i = ||x'_i - Hx_i||$, count inlier (e < 5px). Homography (8 DOF) untuk scene planar. Fundamental Matrix (7 DOF) untuk scene 3D umum. USAC_MAGSAC lebih robust untuk data sangat noisy.

**Slide 23** — AR Marker Detection: Deteksi SIFT atau ORB pada marker template. Cocokkan ke setiap frame video. Jika inlier RANSAC > 10, estimasi H. Overlay grafis via cv2.warpPerspective untuk bingkai virtual. Aplikasi: instruksi layar, tur virtual, game AR berbasis marker gambar fisik.

**Slide 24** — Keypoint Repeatability: $R = |\{k \in K_1 : \exists k' \in K_2, d(k,k') < \epsilon\}| / \min(|K_1|,|K_2|)$. Benchmark Oxford VGG: 6 kondisi (blur, JPEG, iluminasi, zoom+rotasi, affine, viewpoint). SIFT dan AKAZE umumnya unggul dibanding ORB pada benchmark ini.

**Slide 25** — Multi-Image Matching: N gambar: hitung N(N-1)/2 pasang. Matriks similaritas N x N (nilai = inlier count). Visualisasi heatmap. Graf: nodes = gambar, edges = strength match. Clustering via scipy.cluster.hierarchy atau SpectralClustering pada matriks adjacency gambar serupa.

**Slide 26** — Analisis BF vs FLANN: BFMatcher: O(N) per query, akurat 100% (exact NN), baik untuk database < 1000 keypoints. FLANN: approximate NN, 10-100x lebih cepat, recall sekitar 95%. Untuk database 10.000+ gambar, FLANN wajib agar latensi di bawah satu detik per query.

**Slide 27** — Analisis SIFT vs ORB Praktis: SIFT: akurasi lebih tinggi (skala 0.3x-3.0x), iluminasi robust, descriptor 128-float butuh memori lebih. ORB: 10-40x lebih cepat, binary 256-bit hemat memori, cocok mobile. Sejak paten SIFT expire 2020 keduanya tersedia. Pilih ORB real-time, SIFT akurasi.

**Slide 28** — Tips dan Troubleshooting: SIFT butuh opencv-contrib-python-headless. ORB descriptor harus uint8 bukan float32. Reshape keypoints ke (-1,1,2) float32 sebelum findHomography. Ratio threshold 0.7-0.8 sweet spot; 0.5 terlalu ketat, 0.9 terlalu longgar. Minimal 4 good match sebelum findHomography.

**Slide 29** — Kuis 5 Soal: (1) Dimensi descriptor SIFT berapa float? (2) Metrik jarak untuk ORB binary? (3) Berapa pasang titik minimum untuk Homography? (4) FLANN index type untuk SIFT float: KDTree atau LSH? (5) Nilai Lowe ratio test dalam paper original adalah berapa?

**Slide 30** — Ringkasan Semua Teknik: Pipeline: (1) corner detection (Harris, Shi-Tomasi), (2) keypoint+descriptor (SIFT, ORB, AKAZE, FAST), (3) matching (BF, FLANN), (4) ratio test, (5) RANSAC Homography, (6) warpPerspective. Diagram akurasi vs kecepatan sebagai panduan pemilihan metode.

---

## PROMPT 3 — Slide 31-45 (Materi Lanjut + Project + TugasVideo)

Lanjutkan presentasi Modul 4: Deteksi Fitur dan Pencocokan, slide 31-45. Slide 31-35 materi lanjutan, slide 36-41 project, slide 42-45 tugas video.

**Slide 31** — Bag of Visual Words: Kumpulkan N descriptor, cluster menjadi K visual words via k-means. Encode gambar sebagai histogram frekuensi K words. TF-IDF weighting kurangi bobot words sangat umum. Inverted index per word untuk retrieval O(1) tanpa scan seluruh database. Cocok sistem retrieval skala besar.

**Slide 32** — Deep Feature Matching: SuperPoint: CNN end-to-end prediksi keypoint heatmap + descriptor 256-D, dilatih self-supervised. SuperGlue: graph neural network matcher dengan attention mechanism. Lebih akurat dari SIFT+BF untuk viewpoint ekstrem, pencahayaan dramatis, dan scene textureless.

**Slide 33** — Epipolar Geometry: Fundamental matrix F (3x3, rank 2): $x'^T F x = 0$. Epipolar line $l' = Fx$. Essential matrix E = K'^T F K menggunakan kalibrasi K. cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC) dan cv2.computeCorrespondEpilines untuk visualisasi.

**Slide 34** — Image Stitching Overview: Panorama: (1) SIFT tiap gambar, (2) FLANN match, (3) RANSAC Homography, (4) warpPerspective + blending. Cylindrical projection untuk FOV lebar kurangi distorsi. Multi-band blending untuk seamless seam. Shortcut: cv2.Stitcher_create() otomatis.

**Slide 35** — SLAM Basics: Track keypoints ORB antar frame estimasi motion kamera (visual odometry). Keyframe selection dari parallax besar. Loop closure via Bag of Visual Words koreksi drift. ORB-SLAM3 open-source mendukung monocular, stereo, dan RGB-D camera secara simultan.

**Slide 36** — Project Overview: "Aplikasi Pencocokan dan Pengenalan Gambar Berbasis Fitur" - 4 modul: (1) Benchmark detektor, (2) Robust matcher + evaluasi, (3) Object recognizer template, (4) Image retrieval mAP. Kelompok 2-3 mahasiswa. Luaran: kode modular, output visual, laporan PDF 5+ halaman.

**Slide 37** — Project Modul 1 - DetectorBenchmark: Harris, Shi-Tomasi, SIFT, ORB, AKAZE pada 5 kondisi: original, rotasi 45 derajat, scale 0.5x, blur sigma 3, noise sigma 20. Hitung jumlah keypoints dan repeatability vs original. Bar chart 5 detektor x 5 kondisi dan heatmap repeatability 5x5.

**Slide 38** — Project Modul 2 - RobustMatcher: SIFT detectAndCompute. FLANN knnMatch (k=2). Sweep ratio 0.5-0.9: precision-recall via groundtruth homography. Sweep RANSAC threshold 1-10px. Plot precision-recall curve dan F1 vs threshold. Visualisasi inlier garis hijau, outlier garis merah.

**Slide 39** — Project Modul 3 - ObjectRecognizer: Database 5 template objek, extract SIFT offline. Query: detect SIFT, match ke semua template via FLANN + ratio test + RANSAC. Rank top-3 berdasarkan inlier count. Tampilkan query di kiri, top-3 di kanan dengan inlier count dan bounding box overlay.

**Slide 40** — Project Modul 4 - ImageRetrieval: Database 20 gambar dari 5 kategori. Query 5 gambar (1 per kategori). Rank berdasarkan RANSAC inlier count. Hitung precision@1, precision@4, dan Average Precision per query. mAP sebagai metrik tunggal keseluruhan. Visualisasi grid ranking.

**Slide 41** — Project Kriteria Penilaian: Implementasi Teknis 40% (4 modul berjalan, kode PEP8). Analisis Hasil 25% (repeatability, precision-recall, mAP; bukan hanya visual). Kreativitas 15% (GUI, real-time webcam, domain unik). Laporan 20% (PDF 5+ halaman metodologi-hasil-diskusi-referensi).

**Slide 42** — TugasVideo Overview: Video 10-15 menit demo minimal 15 dari 20 file praktikum Modul 4. Struktur: intro 1 menit, corner detection 2 menit, SIFT+ORB 3 menit, matching+RANSAC 4 menit, aplikasi 3 menit, penutup 1 menit. Screen recorder (OBS), narasi Indonesia jelas.

**Slide 43** — TugasVideo Bagian 1: Demo corner dan descriptor: Harris corner heatmap respons R (colormap), cornerSubPix sebelum/sesudah, SIFT keypoints (ukuran lingkaran = sigma, orientasi = panah), ORB 256-bit binary descriptor pattern visual sebagai grid test piksel pasangan yang membedakan descriptor.

**Slide 44** — TugasVideo Bagian 2: Demo matching dan aplikasi: BFMatcher vs FLANN bar chart waktu, jumlah match sebelum/sesudah ratio test, RANSAC inlier (hijau) vs outlier (merah) di drawMatches, hasil warpPerspective alignment, object detection bounding box overlay pada objek terdeteksi.

**Slide 45** — Penutup dan Referensi: Szeliski (2022) Computer Vision Ch.7, Lowe SIFT (IJCV 2004), Rublee et al. ORB (ICCV 2011), Fischler dan Bolles RANSAC (CACM 1981), Alcantarilla AKAZE (BMVC 2013). Selanjutnya: Modul 5. Seluruh 13 tujuan Modul 4 tercakup dari corner detection hingga deep feature matching.

---
