# NotebookLM Prompts — Modul 4: Model Fitting dan Optimasi

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 4: Model Fitting dan Optimasi. Referensi Szeliski (2022) Ch.4. Tiap slide ~500 kata, sertakan diagram, formula, dan kode OpenCV/NumPy.

**Slide 1** — Judul "Modul 4: Model Fitting dan Optimasi", subtitle "Dari Data Noisy ke Model Robust", ilustrasi titik data dengan outlier dan garis RANSAC. Nama MK, semester, referensi Szeliski Ch.4.

**Slide 2** — Tantangan fitting: noise (error kecil), outlier (data menyimpang), overfitting, underfitting. Aplikasi: garis pada edge, estimasi homografi, optical flow, kalibrasi lensa.

**Slide 3** — OLS: minimasi ||y−Xβ||², normal equation β̂=(XᵀX)⁻¹Xᵀy, np.linalg.lstsq(). Percobaan 1: fit polynomial degree 1–3, visualisasi residual dan RMSE per degree.

**Slide 4** — WLS: β̂=(XᵀWX)⁻¹XᵀWy, W=diag(wᵢ), kapan variance tidak homogen. TLS: jarak orthogonal via SVD, noise di x dan y. Percobaan 2–3: perbandingan OLS vs WLS vs TLS.

**Slide 5** — RANSAC: sample s titik→fit model→hitung inlier (residual<ε)→ulangi N kali→re-fit. N=log(1−p)/log(1−wˢ). Percobaan 4: garis robust dengan 30% outlier. Parameter: ε, N, s.

**Slide 6** — IRLS: OLS→residual→update bobot ψ(rᵢ)→WLS→konvergen. Huber: wᵢ=min(1,c/|rᵢ|). Tukey: (1−(r/c)²)² jika |r|<c. Percobaan 5: Huber vs Tukey, grafik konvergensi iterasi.

**Slide 7** — Ridge (L2): +λ||β||², susutkan koefisien. Lasso (L1): +λ||β||₁, sparsity (koefisien=0). Bias-variance: λ kecil→overfit, λ besar→underfit. Percobaan 6–7: kurva koefisien vs λ.

**Slide 8** — Hough Lines: tiap titik vote pada ρ=x·cosθ+y·sinθ. cv2.HoughLines() dan HoughLinesP() (probabilistic, output segmen). Percobaan 8: deteksi garis jalan dan gedung.

**Slide 9** — Hough Circles: voting di (cx,cy,r). HoughCircles(HOUGH_GRADIENT): dp, minDist, param1/2, minRadius/maxRadius. Percobaan 9: deteksi koin/iris. Tips: GaussianBlur sebelum HoughCircles.

**Slide 10** — Homography: transformasi projective 3×3, 8 DOF, 4 pasang titik, DLT→SVD. findHomography(srcPts,dstPts,RANSAC,5.0). Percobaan 10: estimasi H, warpPerspective. Aplikasi: panorama, AR planar.

**Slide 11** — Kontur: findContours()+CHAIN_APPROX_SIMPLE. fitEllipse(), fitEllipseAMS(), fitEllipseDirect(). minAreaRect()→rotated bounding box. convexHull(). Percobaan 11: fit ellips pada bentuk organik.

**Slide 12** — Template Matching: R(x,y)=similarity(I[x:x+w], T). Metode: TM_SQDIFF/CCORR/CCOEFF dan _NORMED. matchTemplate()+minMaxLoc(). NMS untuk multi-instance. Percobaan 12.

**Slide 13** — GrabCut: graph cut interaktif, definisikan rect, grabCut(img,mask,rect,bgdModel,fgdModel,5,GC_INIT_WITH_RECT). Watershed: topographic surface, watershed(). Percobaan 13–14: perbandingan segmentasi.

**Slide 14** — Lucas-Kanade: Brightness Constancy→OFC Ixu+Iyv+It=0→WLS dalam window. calcOpticalFlowPyrLK()→sparse flow. Percobaan 15: tracking titik antar frame, visualisasi trail.

**Slide 15** — Setup: env, pip install opencv-python numpy matplotlib scikit-learn, download_image.py, folder image/output/. Ringkasan percobaan 1–8: OLS/WLS/TLS/RANSAC/IRLS/Ridge/Lasso/Hough. Tabel nama, file, algoritma.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 4, Slide 16–30. Slide 16–25: Farneback, feature matching, CV, denoising, pipeline, rekap, diskusi. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Farneback dense flow: polynomial expansion, setiap piksel. calcOpticalFlowFarneback(prev,next,None,pyr_scale,levels,winsize,iterations,poly_n,poly_sigma). Visualisasi HSV: hue=arah, saturation=magnitude. Percobaan 16: heatmap traffic.

**Slide 17** — Feature Matching: ORB (binary/cepat) vs SIFT (float/robust). BFMatcher+crossCheck, FLANN. Ratio test Lowe d1/d2<0.75. findHomography(RANSAC). Percobaan 17: match dua sudut pandang objek sama.

**Slide 18** — Cross-Validation: K-Fold (K bagian, rotasi test), LOO (K=n). Pilih degree polynomial, λ optimal Ridge/Lasso. sklearn KFold+cross_val_score(). Percobaan 18: plot train vs CV error, degree optimal.

**Slide 19** — NLM: rata-rata piksel serupa di seluruh gambar, fastNlMeansDenoising(h). TV: minimasi ||u−f||²+λ||∇u||₁, preserve edge. Percobaan 19: perbandingan NLM/TV/Gaussian/Median, PSNR+SSIM.

**Slide 20** — Pipeline Gabungan: load frame→Canny edges→Hough lines→LK optical flow→visualisasi annotated. Desain modular: tiap step fungsi terpisah. Percobaan 20 (pipeline_gabungan).

**Slide 21** — Rekap percobaan 1–10: OLS/WLS/TLS/RANSAC/IRLS/Ridge/Lasso/Hough Lines/Hough Circles/Homography. Grid thumbnail output. Pola: semua berbasis minimasi fungsi cost.

**Slide 22** — Rekap percobaan 11–20: kontur/ellips, template+NMS, GrabCut, watershed, LK flow, Farneback, feature matching, K-fold CV, NLM/TV denoising, pipeline. Grid thumbnail.

**Slide 23** — Perbandingan robust: RANSAC (probabilistik, outlier >40%), IRLS (deterministik, konvergen smooth), Hough (voting, bentuk parametrik). Tabel kapan memilih tiap metode.

**Slide 24** — Kuis: (1) Normal equation OLS? (2) Formula N iterasi RANSAC? (3) DOF homography? (4) Perbedaan sparse vs dense optical flow? (5) L1 vs L2 regularisasi—efeknya?

**Slide 25** — Diskusi: RANSAC lebih baik dari OLS saat outlier >20%? Lasso vs Ridge kapan? LOO vs K-Fold bias-variance? NLM lebih efektif dari median untuk Gaussian noise? Template vs feature matching kelemahan?

**Slide 26** — Project: "Aplikasi Model Fitting Terpadu". Min. 10 dari 20 konsep. Tema: Panorama Builder (feature+Hough+RANSAC), Motion Analyzer (LK+Farneback), Robust Regression Tool, Segmentation Suite (GrabCut+Watershed), Document Scanner (Hough+perspektif). Deliverable: .py, output/, laporan PDF.

**Slide 27** — 20 opsi improvisasi (pilih min. 10): OLS multi-degree, WLS bobot jarak, TLS vs OLS, RANSAC threshold sweep, IRLS Huber vs Tukey, Ridge λ curve, Lasso sparsity plot, HoughLinesP sweep, HoughCircles objek, homography manual, fitEllipse 3 metode, template multi-scale, GrabCut iterasi, watershed manual, LK pyramid sweep, Farneback sweep, ratio test sweep, K-fold degree plot, NLM h-sweep, pipeline video.

**Slide 28** — Rubrik: Integrasi 0–40 (≥15=40, 12–14=35, 10–11=30). Fungsionalitas 0–30. Kreativitas 0–20. Dokumentasi 0–10. Total 100. Bonus +5 demo video real-time, +5 RANSAC dari nol.

**Slide 29** — Video: 15–20 menit, screen+face-cam. Struktur: pembukaan (2 mnt), teori 5 konsep (3–4 mnt), demo 20 percobaan live (2 poin/percobaan=40 poin), project (5 mnt), penutup (1 mnt). Submit link LMS.

**Slide 30** — Rubrik video: Pembukaan (5), Teori (10), 20 percobaan (40—2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 animasi RANSAC, +3 Hough accumulator. Penalti −2/percobaan tidak tampil. "Fit Model Robust pada Data Nyata!"
