# NotebookLM Prompts — Modul 7: Deteksi Fitur dan Pencocokan

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 7: Deteksi Fitur dan Pencocokan. Referensi Szeliski (2022) Ch.7. Tiap slide ~500 kata, sertakan diagram, formula, dan kode OpenCV.

**Slide 1** — Judul "Modul 7: Deteksi Fitur dan Pencocokan", subtitle "Fondasi Image Stitching, 3D Reconstruction, dan AR", ilustrasi pipeline feature matching, referensi Szeliski Ch.7.

**Slide 2** — Motivasi dan pipeline: (1) Feature Detection → (2) Feature Description → (3) Feature Matching → (4) Geometric Verification. Properti fitur yang baik: Repeatability, Distinctiveness, Locality, Efficiency. Aplikasi: panorama, visual SLAM, AR, 3D reconstruction.

**Slide 3** — Harris Corner Detector: Auto-correlation matrix M=Σ w(x,y)[I_x² I_xI_y; I_xI_y I_y²]. Corner Response R=det(M)−k·trace(M)². Interpretasi: R>0 = corner, R<0 = edge, |R|≈0 = flat. cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04). Parameter tuning dan visualisasi.

**Slide 4** — Shi-Tomasi (Good Features to Track): R=min(λ₁,λ₂). Lebih stabil dari Harris. cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10). Perbandingan Harris vs Shi-Tomasi: jumlah, kualitas, stabilitas di berbagai kondisi.

**Slide 5** — Blob Detection: LoG (Laplacian of Gaussian) — ∇²_norm G = σ²(∂²G/∂x²+∂²G/∂y²). Blob di maxima/minima scale-space 3D (x,y,σ). DoG (Difference of Gaussians) — aproksimasi efisien LoG, digunakan SIFT. Scale-space octave pyramid diagram.

**Slide 6** — SIFT (Scale-Invariant Feature Transform): 4 tahap: scale-space extrema detection (DoG), keypoint localization (sub-pixel + filter low contrast/edge), orientation assignment (histogram gradien 36 bins), descriptor generation (4×4 grid × 8 orientasi = 128-D). Invariant: skala, rotasi, sebagian iluminasi. cv2.SIFT_create(nfeatures=500).

**Slide 7** — ORB (Oriented FAST and Rotated BRIEF): FAST keypoint + orientasi via centroid patch. rBRIEF descriptor binary 256-bit. Free, fast, alternatif SIFT. cv2.ORB_create(nfeatures=500). AKAZE: non-linear diffusion scale-space, descriptor M-LDB binary. FAST: corner threshold-based, cv2.FastFeatureDetector_create().

**Slide 8** — Feature Matching: Brute-Force (BF) — bandingkan semua pasangan, Hamming untuk binary (ORB), L2 untuk float (SIFT). cv2.BFMatcher(cv2.NORM_HAMMING). FLANN (Fast Library for Approximate Nearest Neighbors) — KD-tree/LSH, lebih cepat untuk database besar. cv2.FlannBasedMatcher().

**Slide 9** — Lowe's Ratio Test: Filter false matches — hanya terima jika dₙ₁/dₙ₂ < 0.75. Geometric Verification: Homography H (8 DOF, 4 pasang titik): x'=Hx. cv2.findHomography(src, dst, cv2.RANSAC, 5.0). Inlier/outlier visualisasi, reprojection error. warpPerspective() untuk overlay.

**Slide 10** — Percobaan 1–4: Harris corner detection (heatmap response, parameter k/blockSize sweep). Shi-Tomasi (perbandingan dengan Harris, maxCorners/qualityLevel). SIFT detection (rich keypoints, skala+orientasi). ORB detection (speed vs SIFT, binary descriptor).

**Slide 11** — Percobaan 5–7: AKAZE + FAST multi-detector comparison table (jumlah keypoints, waktu deteksi, memori). Brute-Force matching SIFT vs ORB (visual match lines). FLANN + ratio test (threshold sweep 0.65–0.85, jumlah good matches).

**Slide 12** — Percobaan 8–10: Homography + RANSAC (inlier/outlier visual, dekomposisi H, reprojection error). Object detection via features (template di scene — warp overlay). Feature invariance rotasi (keypoint consistency 0°–360° tiap 15°, repeatability rate).

**Slide 13** — Percobaan 11–14: Feature invariance skala (25%–300% benchmark). Feature invariance iluminasi (brightness/contrast/gamma variasi). Perbandingan deskriptor SIFT vs ORB vs AKAZE (dimensi, waktu, akurasi matching). Geometric verification detail (RANSAC parameter tuning, inlier analysis).

**Slide 14** — Percobaan 15–17: Image retrieval (database indexing + query retrieval, ranking kemiripan). AR marker detection (deteksi marker + homography overlay real-time). Keypoint repeatability analysis (repeatability rate di berbagai transformasi, noise, blur, JPEG compression).

**Slide 15** — Percobaan 18–20: Multi-image matching (feature tracks across 3+ gambar, track linking). Feature matching pipeline (class-based modular: Detector+Descriptor+Matcher+Verifier). Proyek feature matching app (lengkap + evaluasi). Setup: opencv-contrib-python, numpy, matplotlib.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 7, Slide 16–30. Slide 16–25: analisis, rekap, koneksi, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Rekap percobaan 1–10: Harris, Shi-Tomasi, SIFT, ORB, AKAZE+FAST, BF matching, FLANN+ratio test, homography+RANSAC, object detection via features, invariance rotasi. Grid thumbnail. Tabel detektor, deskriptor, matcher, jumlah keypoints, waktu.

**Slide 17** — Rekap percobaan 11–20: invariance skala, invariance iluminasi, perbandingan deskriptor, geometric verification, image retrieval, AR marker, repeatability, multi-image, modular pipeline, proyek app. Tabel aplikasi nyata tiap percobaan dan metrik evaluasi.

**Slide 18** — Analisis mendalam: Mengapa SIFT memerlukan DoG multi-scale? Pengaruh k pada Harris (0.04–0.06). Kapan FLANN lebih baik dari BF? Tradeoff SIFT (128-D float, lambat) vs ORB (256-bit binary, cepat) untuk embedded system.

**Slide 19** — Koneksi antar modul: Feature detection (Modul 7) → image stitching (Modul 8). Feature tracks → visual odometry (Modul 9). Keypoints multi-view → SfM triangulasi (Modul 11). BoVW recognition (Modul 6) menggunakan SIFT dari Modul 7. Fondasi pipeline CV modern.

**Slide 20** — Best practices: detect pada grayscale. Non-Maximum Suppression untuk mencegah keypoints menumpuk. CachingDescriptors untuk database retrieval besar. Threshold ratio test 0.7–0.75. Cek jumlah inlier > 10 sebelum accept homography.

**Slide 21** — Aplikasi nyata: AR marker dengan warp overlay real-time (kamera webcam, stabil). Document scanner (4 corner → perspektif correction via homography). Visual search engine (SIFT→FLANN→ranked results). Logo/brand detection dari foto produk.

**Slide 22** — Perbandingan detektor: tabel Harris, Shi-Tomasi, SIFT, ORB, AKAZE, FAST — repeatability, distinctiveness, invariance (skala/rotasi/iluminasi), kecepatan, lisensi. Rekomendasi: SIFT untuk akurasi, ORB untuk real-time/edge.

**Slide 23** — Checklist kompetensi: Harris/Shi-Tomasi corner detection, SIFT/ORB/AKAZE detection+description, BF dan FLANN matching, ratio test filtering, homography+RANSAC estimation, geometric verification, image retrieval, AR overlay. Self-assessment tabel.

**Slide 24** — Kuis: (1) Dimensi deskriptor SIFT? (2) Rumus Harris corner response? (3) Ratio test Lowe: threshold dan tujuan? (4) Minimum titik untuk homography? (5) Perbedaan FAST dan Harris dalam deteksi corner?

**Slide 25** — Diskusi: Mengapa RANSAC penting setelah matching? Kapan Shi-Tomasi lebih baik dari Harris? Bagaimana menangani dataset gambar sangat besar dengan FLANN? Apakah deep features (SuperPoint, D2-Net) akan menggantikan SIFT sepenuhnya?

**Slide 26** — Project "Aplikasi Feature Matching". 10 soal cerita: Pengenalan Sampul Buku (SIFT+database), Pendeteksi Uang Kertas (denominasi+scoring), AR Marker System (5 marker+overlay real-time), Verifikasi Tanda Tangan, Image Forgery Detector (copy-move self-matching), Pemandu Wisata Visual (10 landmark database), Quality Control Part Matching, Pendeteksi Duplikat Foto (50+ gambar), Navigasi Visual Robot (10 lokasi), Card Game Scanner (10 kartu database). Deliverable: .py, output/, laporan.

**Slide 27** — 15 improvisasi: Multi-object Detector, Rotation Invariance Benchmark, Scale Invariance Benchmark, Illumination Robustness Test, Real-time Object Tracker, Image Retrieval Engine (50+), Panoramic Image Pair Finder, Affine-invariant Matching, Feature Heatmap Generator, Copy-Move Forgery Detection, AR Card Demo, Logo Spotter, Visual Dictionary Builder, Keypoint Stability Analysis, Deep Feature Comparison (SuperPoint).

**Slide 28** — Rubrik Project: Fungsionalitas 35%, Integrasi Percobaan 20%, Kualitas Kode 15%, Dokumentasi 15%, Kreativitas 15%. Format ZIP NIM_Nama_Project07.zip. Bonus +5 demo objek fisik live (buku/kartu), +3 database >50 items. Penalti jika verifikasi RANSAC tidak diimplementasikan.

**Slide 29** — Tugas Video: Demo 20 percobaan LIVE (50–65 mnt), 2 poin/percobaan. Penjelasan Materi (15–20 mnt) — wajib: diagram perbandingan detektor + pipeline matching. Tunjukkan objek fisik (buku, kartu) saat demo matching/detection. Demo Project (10–15 mnt) + Analisis (5 mnt).

**Slide 30** — Rubrik Video: Pembukaan (5), Materi (10), 20 Percobaan (40 — 2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 demo matching objek fisik nyata, +3 real-time tracking. Penalti −2/percobaan tidak tampil. "Temukan Korespondensi, Sambungkan Dunia Visual!"
