# NotebookLM Prompts  Modul 7: Estimasi Gerak (Motion Estimation)

---

## PROMPT 1  Slide 115 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 7: Estimasi Gerak (Motion Estimation). Referensi Szeliski (2022) Ch.9. Tiap slide ~300 kata, sertakan formula dan kode OpenCV.

**Slide 1**  Judul "Modul 7: Estimasi Gerak", subtitle "Optical Flow, Background Subtraction, dan Object Tracking", ilustrasi pipeline motion estimation dari video.

**Slide 2**  Optical Flow: Brightness Constancy $I(x,y,t) = I(x+u,y+v,t+1)$. Constraint equation: $I_xu + I_yv + I_t = 0$. Aperture problem: satu persamaan, dua unknowns. Perlu asumsi tambahan.

**Slide 3**  Lucas-Kanade (Sparse): asumsi flow konstan dalam window. Overdetermined system  least squares. cv2.calcOpticalFlowPyrLK(). Pyramidal LK untuk large motion. Good features to track sebagai input.

**Slide 4**  Dense Optical Flow (Farneback): estimasi flow setiap piksel via polynomial expansion. cv2.calcOpticalFlowFarneback(). Visualisasi HSV: Hue=arah, Saturation=magnitude. Lebih lengkap tapi lebih lambat.

**Slide 5**  Background Subtraction: MOG2 (cv2.createBackgroundSubtractorMOG2())  Gaussian mixture per piksel, adaptif, shadow detection. KNN (cv2.createBackgroundSubtractorKNN())  non-parametric. Learning rate parameter.

**Slide 6**  Frame Differencing dan Running Average: Frame diff $|f_t - f_{t-1}| > T$  sederhana, noise. Running average $B_t = \alpha \cdot f_t + (1-\alpha) \cdot B_{t-1}$  adaptif background model.

**Slide 7**  Object Tracking: CSRT (akurat, lambat), KCF (cepat, less accurate), MOSSE (tercepat). cv2.TrackerCSRT_create(). ROI selection  track across frames. Multi-object: cv2.MultiTracker_create().

**Slide 8**  Percobaan 13: Sparse optical flow Lucas-Kanade (trail tracking, re-detection), Dense optical flow Farneback (HSV visualization, flow arrows), Visualisasi optical flow (arrow plot, magnitude heatmap).

**Slide 9**  Percobaan 47: Background subtraction MOG2 (learning rate, shadow), BGS KNN (perbandingan MOG2), Frame differencing (threshold tuning), Running average background (alpha parameter).

**Slide 10**  Percobaan 810: Object tracking CSRT (ROI selection, accuracy), Object tracking KCF (speed vs accuracy), Multi-object tracking (multiple ROI, trajectory visualization).

**Slide 11**  Motion History Image: temporal motion template. cv2.motempl.updateMotionHistory(). Visualisasi intensitas menunjukkan recency gerakan. Segmentasi area gerak. Percobaan 11.

**Slide 12**  Video Stabilization: (1) Feature detection antar frame, (2) Estimasi transformasi, (3) Smooth trajectory (moving average/Kalman), (4) Apply smoothed transform. Before/after comparison. Percobaan 12.

**Slide 13**  Percobaan 1315: Frame interpolation linear (blending antar frame), Frame interpolation flow-based (optical flow warping), Optical flow magnitude & direction (analisis kuantitatif).

**Slide 14**  Percobaan 1618: Feature trajectory tracking (long-term tracking, trail akumulasi), Background subtraction comparison (MOG2 vs KNN vs frame diff tabel), Deteksi gerakan contour (bounding box, counting).

**Slide 15**  Percobaan 1920: Optical flow real-time simulasi (interaktif), Estimasi kecepatan objek (pixel/frame  km/h, kalibrasi). Setup: opencv-contrib-python, video input (webcam/file).

---

## PROMPT 2  Slide 1630 (Materi Lanjutan + Analisis)

Lanjutkan slide Modul 7, Slide 1630. Analisis mendalam, rekap, koneksi antar modul. Tiap slide ~300 kata.

**Slide 16**  Kalman Filter untuk tracking: predict-correct loop. State $[x,y,v_x,v_y]$. cv2.KalmanFilter(4,2). Transition matrix (constant velocity model). Process noise vs measurement noise tuning.

**Slide 17**  Mean Shift dan CAMShift: Mean Shift  iterative mode seeking pada histogram backprojection. CAMShift  adaptive window size. cv2.meanShift(), cv2.CamShift(). Color histogram-based tracking.

**Slide 18**  SORT dan DeepSORT: SORT = Kalman + Hungarian algorithm. DeepSORT += appearance descriptor (Re-ID). Multi-object tracking pipeline: detect  predict  associate  update.

**Slide 19**  Rekap Percobaan 15: Sparse LK (feature trails), Dense Farneback (HSV flow), Visualisasi flow (arrow+magnitude), MOG2 (adaptive background), KNN (non-parametric BGS).

**Slide 20**  Rekap Percobaan 610: Frame differencing (simple motion), Running average (alpha tuning), CSRT tracking (high accuracy), KCF tracking (high speed), Multi-object tracking (multiple targets).

**Slide 21**  Rekap Percobaan 1115: MHI (temporal motion template), Video stabilization (trajectory smoothing), Frame interpolation linear, Frame interpolation flow, Magnitude dan direction analysis.

**Slide 22**  Rekap Percobaan 1620: Feature trajectory (long-term trails), BGS comparison table, Contour-based detection (counting), Real-time flow simulation, Speed estimation (pixelkm/h).

**Slide 23**  Analisis: Sparse vs dense flow trade-off. MOG2 vs KNN: kapan masing-masing lebih baik? CSRT vs KCF: akurasi vs kecepatan. Kalman filter: tuning Q dan R parameters.

**Slide 24**  Perbandingan tracker tabel lengkap: CSRT (akurat, lambat), KCF (cepat), MOSSE (tercepat), Mean Shift (histogram), CAMShift (adaptive), SORT (multi-object). Metrik: accuracy, FPS, robustness.

**Slide 25**  Video stabilization mendalam: rolling shutter compensation. Trajectory smoothing window. Cropping vs inpainting border. Real-time vs offline stabilization trade-off.

**Slide 26**  Koneksi antar modul: Feature detection (M4)  feature tracking. Homography (M5)  frame alignment. Panorama (M6)  video panorama. Computational photography (M8)  frame averaging denoising.

**Slide 27**  Best practices: Re-detect features setiap N frame (LK). Learning rate BGS: 0.01 untuk scene stabil, 0.1 untuk berubah cepat. ROI tracking: reinitialize saat confidence drop. Kalman: tune Q/R per scenario.

**Slide 28**  Aplikasi nyata: Traffic monitoring (counting+speed), Security surveillance (motion detection+recording), Sports analytics (player tracking+trajectory), Drone stabilization, AR tracking.

**Slide 29**  Checklist kompetensi: LK dan Farneback flow, MOG2 dan KNN BGS, Frame differencing, CSRT/KCF tracking, Multi-object tracking, MHI, Video stabilization, Frame interpolation, Speed estimation.

**Slide 30**  Kuis: (1) Brightness constancy equation? (2) Aperture problem? (3) Perbedaan MOG2 dan KNN? (4) CSRT vs KCF: mana lebih akurat? (5) Kalman filter: apa fungsi predict dan correct?

---

## PROMPT 3  Slide 3145 (Analisis + Project + Tugas Video)

Lanjutkan Modul 7, Slide 3145. Slide 3135: analisis lanjutan. Slide 3641: Project. Slide 4245: Tugas Video. Tiap slide ~300 kata.

**Slide 31**  Flow-based applications: action recognition dari flow pattern. Anomaly detection (unusual motion). Crowd analysis (density+flow direction). Gesture recognition.

**Slide 32**  Advanced BGS: handling illumination changes (gradual vs sudden). Handling camera jitter. Static object detection. Shadow removal pipeline.

**Slide 33**  Multi-object tracking challenges: ID assignment, occlusion handling, re-identification. Evaluation metrics: MOTA, MOTP, ID switches, track fragmentation.

**Slide 34**  Detection + Tracking integration: YOLO  SORT pipeline. Detection every frame  track between detections. Online vs offline tracking. Batch vs stream processing.

**Slide 35**  Ringkasan: dari optical flow dasar ke pipeline tracking lengkap. 20 percobaan: flow  BGS  tracking  analysis. Fondasi untuk autonomous driving dan surveillance.

**Slide 36**  Project "Estimasi Gerak": 10 soal cerita. Integrasikan min 10 konsep. Contoh: Sistem Monitoring Lalu Lintas (BGS+tracking+counting+speed), Sistem Keamanan Motion Detection (MOG2+recording+alarm).

**Slide 37**  Soal cerita 35: Video Stabilizer Pro (trajectory smoothing, Kalman), Penghitung Pengunjung (overhead, line crossing, heatmap), Analisis Gerakan Olahraga (dense flow, MHI, body region).

**Slide 38**  Soal cerita 610: Object Following Drone Simulator, Gesture Recognition System, People Counter Bidirectional, Activity Classification, Motion-based Video Summarizer.

**Slide 39**  15 improvisasi: Real-time Flow Visualizer, Tracker Comparison Benchmark, Speed Radar App, Crowd Density Estimator, Fall Detection, Virtual Tripod, dll.

**Slide 40**  Rubrik: Fungsionalitas 35%, Integrasi 20%, Kode 15%, Dokumentasi 15%, Kreativitas 15%. Bonus +5 demo webcam real-time, +3 multi-object tracking. Format: NIM_Nama_Project07.zip.

**Slide 41**  Tips project: Gunakan video sendiri (rekam di kampus). Class-based design. Visualisasi trajectory dan heatmap. Save video output ke folder output/.

**Slide 42**  Tugas Video: Demo 20 percobaan (4060 menit). Highlights: optical flow visualisasi, BGS comparison, tracking demo, video stabilization before/after, speed estimation.

**Slide 43**  Tugas Video Materi: 1015 menit. Wajib: diagram optical flow constraint, perbandingan BGS methods, pipeline tracking (detectpredictassociateupdate).

**Slide 44**  Tugas Video Project: Demo 1015 menit. Gunakan video sendiri. Tunjukkan real-time processing. Analisis performa: FPS, accuracy, failure cases.

**Slide 45**  Rubrik Video: Pembukaan 5%, Materi 15%, 20 Percobaan 40%, Project 20%, Analisis 10%, Kualitas 10%. Bonus +5 demo webcam live. Penalti per percobaan error. "Track Every Motion, Understand Every Frame!"
