# NotebookLM Prompts — Modul 9: Estimasi Gerak (Motion Estimation)

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 9: Estimasi Gerak (Motion Estimation). Referensi Szeliski (2022) Ch.9. Tiap slide ~500 kata, sertakan diagram, formula, dan kode OpenCV.

**Slide 1** — Judul "Modul 9: Estimasi Gerak (Motion Estimation)", subtitle "Dari Frame ke Frame — Memahami Gerak dalam Video", ilustrasi optical flow arrows, referensi Szeliski Ch.9.

**Slide 2** — Motivasi: motion estimation = fondasi video analysis, object tracking, stabilization, action recognition, autonomous navigation. Definisi Optical Flow: bidang vektor 2D (u,v) menunjukkan perpindahan apparent setiap piksel antara dua frame. Aplikasi visualisasi: arrows, HSV heatmap.

**Slide 3** — Optical Flow Constraint Equation: Brightness Constancy Assumption — I(x,y,t)=I(x+u,y+v,t+1). Taylor expansion: I_x·u + I_y·v + I_t = 0 (satu persamaan, dua unknown). Aperture Problem: hanya komponen normal yang dapat ditentukan → butuh constraint tambahan.

**Slide 4** — Lucas-Kanade (Sparse): asumsikan semua piksel dalam window W memiliki flow sama → least squares [u;v]=(AᵀA)⁻¹Aᵀb. Solvable jika AᵀA invertible (area bertekstur/corner). Sparse = hanya di keypoints. Pyramidal LK: estimasi di level kasar → propagasi ke halus → refine.

**Slide 5** — Pyramidal Lucas-Kanade OpenCV: p0=cv2.goodFeaturesToTrack(old_gray). p1,status,err=cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, winSize=(15,15), maxLevel=2). Status=1 jika tracked. Trail visualisasi tiap titik.

**Slide 6** — Dense Optical Flow: Horn-Schunck — tambahkan smoothness constraint: E=∫∫[(I_x·u+I_y·v+I_t)²+α²(|∇u|²+|∇v|²)]dxdy. Farnebäck — aproksimasi polinomial kuadratik lokal. cv2.calcOpticalFlowFarneback(). Visualisasi HSV: hue=arah, value=magnitude.

**Slide 7** — Background Subtraction: tujuan memisahkan foreground dari background statis. Metode: Frame Differencing (|I_t−I_{t-1}|>θ), Running Average (B_{t+1}=αI_t+(1−α)B_t), MOG2 (Mixture of Gaussians — model campuran per piksel, shadow detection), KNN. cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True).

**Slide 8** — Object Tracking: cv2.TrackerCSRT_create() (akurasi tinggi), cv2.TrackerKCF_create() (kecepatan tinggi), cv2.TrackerMOSSE_create() (tercepat, akurasi rendah). Multi-object tracking: array trackers. Inisialisasi ROI: cv2.selectROI(). Update per frame: success, bbox = tracker.update(frame).

**Slide 9** — Motion History Image (MHI): akumulasi temporal gerakan. MHI(x,y,t)=τ jika motion terdeteksi, turun seiring waktu. cv2.motempl.updateMotionHistory(), calcMotionGradient(). Segmentasi temporal: identifikasi region gerakan berbeda dalam satu frame.

**Slide 10** — Video Stabilization: estimasi transformasi kamera (affine/homography) per frame. Trajectory kamera (translasi tx,ty + rotasi θ). Smoothing trajectory: moving average atau Kalman filter. Apply inverse transform. Crop untuk menutupi black border.

**Slide 11** — Percobaan 1–4: Sparse Optical Flow LK (trail tracking, re-detection saat tracking hilang). Dense Optical Flow Farnebäck (HSV visualization, flow arrows overlay). Visualisasi Optical Flow (arrow plot, HSV heatmap, magnitude map). Background Subtraction MOG2 (learning rate, shadow detection toggle).

**Slide 12** — Percobaan 5–8: Background Subtraction KNN (perbandingan dengan MOG2: FG coverage, noise). Frame Differencing (gerakan antar frame, threshold sweep, morphology cleaning). Running Average Background (alpha parameter effect, slow vs fast adaptation). Object Tracking CSRT (ROI selection, akurasi tracking objek bergerak cepat).

**Slide 13** — Percobaan 9–12: Object Tracking KCF (speed vs accuracy tradeoff, FPS comparison dengan CSRT). Multi-Object Tracking (multiple ROI, trajectory per objek, collision handling). Motion History Image (MHI temporal, motion segments per region, direction estimation). Video Stabilization (before/after comparison, trajectory visualization, smooth vs raw).

**Slide 14** — Percobaan 13–16: Frame Interpolation Linear (blending antar frame n dan n+1). Frame Interpolation Flow-based (optical flow warp untuk sub-frame synthesis, smoother hasil). Magnitude & Arah Flow (kuantisasi flow: histogram arah, statistik magnitude per region). Feature Trajectory (long-term tracking >50 frame, trail plot, detect track loss).

**Slide 15** — Percobaan 17–20: BGS Comparison (MOG2 vs KNN vs Frame Differencing: tabel kualitas, kecepatan, adaptasi). Deteksi Gerakan Contour (contour area filtering, bounding box, min area threshold). Optical Flow Real-Time (simulasi interaktif webcam, parameter ajustable). Estimasi Kecepatan Objek (kalibrasi piksel/real-world, v=displacement_pixel/frame × fps × scale → km/h). Setup: opencv-python, numpy, matplotlib, video/webcam.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 9, Slide 16–30. Slide 16–25: analisis, rekap, koneksi, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Rekap percobaan 1–10: Sparse LK, Dense Farnebäck, flow visualization, MOG2, KNN BGS, frame differencing, running average, CSRT tracking, KCF tracking, multi-object tracking. Grid thumbnail. Tabel metode, tipe (sparse/dense/BGS/tracker), kecepatan.

**Slide 17** — Rekap percobaan 11–20: MHI, video stabilization, frame interpolation linear, frame interpolation flow, flow magnitude/arah, feature trajectory, BGS comparison, deteksi contour, real-time optical flow, estimasi kecepatan. Tabel aplikasi nyata tiap percobaan.

**Slide 18** — Analisis mendalam: Aperture problem — mengapa satu persamaan tidak cukup? LK berasumsi constant flow dalam window — kapan asumsi ini gagal (object boundary)? HOG+Pyramidal untuk gerakan besar. Tradeoff CSRT (akurasi) vs KCF (~100 FPS) vs MOSSE (kecepatan tertinggi namun drift).

**Slide 19** — Koneksi antar modul: Lucas-Kanade tracking (Modul 9) menggunakan corner detection dari Modul 7. Background subtraction → foreground → input penghitung pedestrian (Modul 6 HOG). Video stabilization menggunakan affine transform (Modul 2). Feature trajectory → visual odometry (Modul 11). Flow-based frame interpolation → computational photography (Modul 10).

**Slide 20** — Best practices: detect ulang features setiap N frame (LK drift). Morphological operations (erode+dilate) untuk membersihkan BGS noise. Adaptive learning rate MOG2 (tinggi saat scene baru, rendah saat stabil). Kalman filter untuk smooth tracking dan predict saat occluded.

**Slide 21** — Aplikasi nyata: Traffic monitoring (CCTV → BGS → counting → kecepatan). Security smart camera (MOG2 → alert saat motion > threshold). GoPro video stabilization (estimasi affine per frame → smooth trajectory). Sports analytics (player trajectory → heatmap lapangan).

**Slide 22** — Perbandingan: Sparse vs Dense optical flow — tabel kecepatan, informasi (titik vs seluruh piksel), use-case. BGS methods: Frame Differencing vs Running Average vs MOG2 vs KNN — adaptasi, shadow handling, memory. Tracker KCF vs CSRT vs MOSSE: FPS, akurasi, robustness oklusi.

**Slide 23** — Checklist kompetensi: sparse LK tracking, dense Farnebäck + HSV visualization, MOG2 + KNN BGS, CSRT/KCF multi-object tracking, MHI, video stabilization, frame interpolation, flow magnitude analysis, speed estimation, real-time webcam. Self-assessment tabel.

**Slide 24** — Kuis: (1) Rumus Brightness Constancy Assumption? (2) Apa Aperture Problem? (3) Perbedaan MOG2 dan KNN BGS? (4) Parameter winSize pada calcOpticalFlowPyrLK? (5) Mengapa trajectory smoothing diperlukan pada stabilization?

**Slide 25** — Diskusi: Kapan sparse lebih cocok dari dense optical flow? Bagaimana mendeteksi kamera bergerak vs objek bergerak dari flow? Tradeoff learning rate MOG2: cepat adapt vs false positive. Deep optical flow (RAFT, FlowNet) vs Lucas-Kanade klasik untuk autonomous driving.

**Slide 26** — Project "Motion Analysis System". 10 soal cerita: Monitoring Lalu Lintas (BGS+tracking+counting+kecepatan), Smart Motion Detection (MOG2+filtering+alert+recording), Video Stabilizer Pro (trajectory+Kalman+before-after), Penghitung Pengunjung (BGS+blob+line crossing), Analisis Olahraga (dense flow+trajectory+slow-motion), Virtual Tripod (stabilize+panorama), Baby Monitor (motion scoring+alert), Drone Flight Stabilizer (dekomposisi pan/tilt/zoom), Slow Motion Creator (flow-based 4×8× slowdown), Smart Whiteboard Recorder (BGS person vs tulisan+timeline). Deliverable: .py, output/, laporan.

**Slide 27** — 15 improvisasi: Speed Estimator (km/h dengan kalibrasi), Crowd Flow Analysis (arah+kepadatan), Gesture Video Player (kontrol play/pause flow), Smart Motion Detection (hanya alert gerakan signifikan), Object Counter Line Crossing, Trajectory Prediction (beberapa frame ke depan), Kalman Filter Tracker, Action Replay Generator (slow-motion otomatis), Heatmap Accumulator (area sering dilalui), SORT Tracker (Hungarian+Kalman), PTZ Estimator, Video Synopsis, Abnormal Motion Detector, Interactive Dense Flow Viewer GUI, Temporal Super Resolution.

**Slide 28** — Rubrik Project: Fungsionalitas 35%, Integrasi Percobaan 20%, Kualitas Kode 15%, Dokumentasi 15%, Kreativitas 15%. Bonus +5 demo real-time webcam live, +3 Kalman filter integration. Format ZIP NIM_Nama_Project09.zip. Deadline 1 minggu.

**Slide 29** — Tugas Video: Demo REAL-TIME dari webcam untuk percobaan 1–10 (lambaikan tangan, bergerak di depan kamera). Demo 20 percobaan LIVE (40–60 mnt). Materi (10–15 mnt): wajib diagram optical flow equation + stabilization pipeline. Demo Project (10–15 mnt) input video → processing → output.

**Slide 30** — Rubrik Video: Pembukaan (5), Materi (10), 20 Percobaan (40 — 2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 tracking + kecepatan real-time fisik, +3 stabilization before-after mengesankan. Penalti −2/percobaan tidak tampil. "Tangkap Gerak, Analisis Dunia yang Bergerak!"
