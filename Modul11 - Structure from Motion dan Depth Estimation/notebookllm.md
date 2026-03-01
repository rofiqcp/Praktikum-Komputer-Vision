# NotebookLM Prompts — Modul 11: Structure from Motion dan Depth Estimation

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 11: Structure from Motion dan Depth Estimation. Referensi Szeliski (2022) Ch.11–12. Tiap slide ~500 kata, sertakan diagram, formula, dan kode OpenCV.

**Slide 1** — Judul "Modul 11: Structure from Motion dan Depth Estimation", subtitle "Dari Gambar 2D ke Pemahaman 3D", ilustrasi point cloud rekonstruksi + depth map, referensi Szeliski Ch.11–12.

**Slide 2** — Motivasi: SfM merekonstruksi geometri 3D scene dan posisi kamera dari gambar 2D. Depth Estimation memperkirakan kedalaman per piksel (stereo atau monocular). Fundamental untuk: AR, robot navigation, SLAM, 3D scanning, autonomous driving. Diagram pipeline: gambar → matches → poses → 3D points.

**Slide 3** — Epipolar Geometry: dua kamera melihat titik 3D yang sama → epipolar constraint. Fundamental Matrix F (pixel coordinates): x'ᵀFx=0. F = matriks 3×3, rank 2, berisi info rotasi+translasi+intrinsik. 8-point algorithm. cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0). Visualisasi epipolar lines.

**Slide 4** — Essential Matrix E: jika intrinsik K diketahui: E=K'ᵀFK. x̂'ᵀEx̂=0 di normalized coordinates. Dekomposisi: E=[t]×R → rotasi R dan translasi t. cv2.findEssentialMat(pts1,pts2,K,cv2.RANSAC,0.999,1.0). _,R,t,mask=cv2.recoverPose(E,pts1,pts2,K). Diagram epipolar geometry.

**Slide 5** — Triangulasi: dua kamera dengan posisi diketahui + korespondensi → posisi 3D. x=PX, x'=P'X. Noise → ray tidak berpotongan → least squares. P1=K@[I|0], P2=K@[R|t]. points4D=cv2.triangulatePoints(P1,P2,pts1.T,pts2.T). Homogeneous→Euclidean: X=points4D[:3]/points4D[3].

**Slide 6** — SfM Pipeline: (1) Feature Detection & Matching, (2) Estimate F/E → recover pose, (3) Triangulate → initial 3D points, (4) Add cameras incrementally → PnP, (5) Bundle Adjustment → optimasi semua kamera + titik 3D. PnP: cv2.solvePnPRansac(obj_pts,img_pts,K,dist). BA: min Σ‖x_ij−π(R_i,t_i,X_j)‖².

**Slide 7** — Visual Odometry: estimasi gerakan kamera frame-by-frame dari video → trajectory. Pipeline: feature detection (t) → track ke (t+1) → essential matrix → recover pose → accumulate transformation. Scale ambiguity pada monocular VO. Aplikasi: SLAM, autonomous vehicle trajectory.

**Slide 8** — Stereo Vision: dua kamera terpisah baseline B. Stereo Calibration: estimasi K₁,K₂,R,T dari checkerboard. Rectification: warpPerspective agar baris koresponden. Block Matching (BM): cv2.StereoBM_create(numDisparities=32,blockSize=15). SGBM (Semi-Global): lebih akurat, cv2.StereoSGBM_create(). Disparity D → Depth Z=f·B/D.

**Slide 9** — Monocular Depth Estimation: depth cues (linear perspective, texture gradient, shadow, occlusion). DNN model: MiDaS (PyTorch)—relative depth dari satu gambar. Limitasi: tidak absolute, skala ambigu. Visualisasi dengan colormap (plasma, inferno). Percobaan 10: apply MiDaS, perbandingan depth cues.

**Slide 10** — Percobaan 1–4: Feature Matching Multi-View (matches antar 5 view, track linking). Fundamental Matrix (epipolar lines visualization antar 2 gambar). Essential Matrix + Pose Recovery (R, t extraction, baseline effects). Epipolar Lines (draw lines, verify constraint x'ᵀFx≈0 per match).

**Slide 11** — Percobaan 5–8: Triangulasi 3D (2 kamera → 3D scatter plot, akurasi vs ground truth). Stereo Calibration (checkerboard fisik, estimasi K+dist+R+T, reprojection RMSE). Stereo Rectification (horizontal alignment verification, epipolar row check). Block Matching (disparity map, numDisparities dan blockSize tuning).

**Slide 12** — Percobaan 9–12: SGBM (BM vs SGBM comparison — coverage, smoothness, waktu). Monocular Depth (depth cues, MiDaS fallback, inverse depth colormap). Disparity to Depth (Z=f*B/d, reprojectImageTo3D, point cloud dari disparity). BM vs SGBM Comparison (multi-scene, timing tabel, coverage map difference).

**Slide 13** — Percobaan 13–16: WLS Filter Disparity (post-processing, left-right consistency, smoothing). Point Cloud from Depth (3D scatter colored by intensity/distance). PnP Pose Estimation (solvePnP methods ITERATIVE vs RANSAC vs P3P comparison). Stereo Matching Realtime (FPS measurement, video stream depth).

**Slide 14** — Percobaan 17–19: Depth Colorization (colormaps plasma/magma/turbo, overlay transparency, contour lines). Baseline Effect (MAE/RMSE vs baseline distance, near/far objects accuracy). Depth Segmentation (threshold depth range → binary mask → connected components → object isolation).

**Slide 15** — Percobaan 20: Multi-View Reconstruction (SfM mini pipeline — 5+ gambar, feature matching, F/E estimation, triangulasi, BA sederhana, visualisasi point cloud). Setup: opencv-python, numpy, matplotlib, open3d (optional); checkerboard fisik untuk stereo calibration. Rekam gambar objek dari sudut berbeda.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 11, Slide 16–30. Slide 16–25: analisis, rekap, koneksi, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Rekap percobaan 1–10: multi-view matching, Fundamental Matrix + epipolar, Essential Matrix + pose, epipolar lines, triangulasi, stereo calibration, rectification, Block Matching, SGBM, monocular depth. Grid thumbnail. Tabel metode, input/output, OpenCV API.

**Slide 17** — Rekap percobaan 11–20: disparity-to-depth, BM vs SGBM comparison, WLS filter, point cloud 3D scatter, PnP pose estimation, stereo real-time, depth colorization, baseline effect, depth segmentation, SfM mini pipeline. Tabel metrik: RMSE, MAE, FPS, coverage.

**Slide 18** — Analisis mendalam: Mengapa F rank 2 (bukan rank 3)? Perbedaan F vs E: F = sembarang kamera, E = kalibrasi diketahui. Mengapa SGBM lebih baik dari BM di area homogen? Limitasi monocular depth: scale ambiguity dan bagaimana mengatasinya.

**Slide 19** — Koneksi antar modul: SfM menggunakan feature matching (Modul 7) + bundle adjustment (Modul 8). Stereo depth → synthetic bokeh (Modul 10). Visual odometry menggunakan optical flow tracking (Modul 9). Depth map → 3D point cloud → surface reconstruction (Modul 12). Kalibrasi kamera (Modul 2) adalah prasyarat semua metode stereo.

**Slide 20** — Best practices: selalu calibrate kamera sebelum stereo matching. Checkerboard ≥10 gambar dari sudut bervariasi. Verifikasi rectification: periksa baris epipolar horizontal. WLS post-processing meningkatkan disparity quality signifikan. Bundle adjustment wajib untuk SfM skala besar.

**Slide 21** — Aplikasi nyata: Autonomous vehicle depth estimation (stereo kamera). SLAM untuk robot navigation. 3D face scanning (structured light + stereo). AR depth occlusion (virtual object tertutup objek nyata). Drone inspection 3D mapping.

**Slide 22** — Perbandingan stereo matching: BM vs SGBM vs SGBM+WLS — tabel coverage, MAE, waktu (ms/frame), kualitas di area homogen (sky, wall). Monocular vs Stereo depth: absolute metric, robustness, hardware requirements. SfM vs SLAM: batch vs incremental, scale drift.

**Slide 23** — Checklist kompetensi: Fundamental+Essential matrix estimation, epipolar lines, triangulasi, stereo calibration, rectification, BM+SGBM disparity, monocular depth, disparity→depth, point cloud visualization, PnP, SfM mini pipeline. Self-assessment tabel.

**Slide 24** — Kuis: (1) Rumus Fundamental Matrix constraint? (2) Minimum korespondensi untuk 8-point algorithm? (3) Rumus Z dari disparity D (stereo)? (4) Perbedaan solvePnP dan solvePnPRansac? (5) Mengapa rectification penting sebelum stereo matching?

**Slide 25** — Diskusi: Kapan monocular depth cukup vs stereo? Bagaimana bundle adjustment mengurangi drift pada SfM besar? Point cloud dari stereo vs dari SfM: kelebihan masing-masing. End-to-end depth DNN (MiDaS, ZoeDepth) vs geometric stereo untuk autonomous driving.

**Slide 26** — Project "3D Vision System". 10 soal cerita: 3D Object Digitizer (15+ foto SfM → PLY export), Navigasi Robot Visual Odometry (video → trajectory bird's-eye), Dual Kamera Depth (stereo simulasi+bokeh+segmentasi), Scene Reconstruction dari CCTV (2 kamera → point cloud ruangan), AR Depth Demo (MiDaS+floor detection+virtual object), Sistem Pengukuran Jarak Stereo (klik 2 titik → 3D distance), Monocular Depth Video Editing (fog/color grading+parallax), Building Facade Reconstruction (5+ foto SfM+plane fitting), Stereo Anaglyph Photo Tool (red-cyan+parallax adjustment), Parking Distance Estimator (SGBM+YOLO+alert <1m). Deliverable: .py, output/, laporan.

**Slide 27** — 15 improvisasi: Full SfM Pipeline (10+ gambar), Dense Reconstruction (stereo+SfM), Stereo Video Depth real-time, Depth-based Obstacle Detection, 3D Object Scanner (turntable), Visual Odometry with Scale, Disparity Refinement (guided filter), Multi-baseline Stereo (3+ kamera), Depth Colorized Video, Egomotion Dashcam, Plane Detection from Depth, Depth-aware Image Editing, Stereo Anaglyph Generator, Loop Closure Detection (SLAM basic), Depth Map Super Resolution.

**Slide 28** — Rubrik Project: Fungsionalitas 35%, Integrasi Percobaan 20%, Kualitas Kode 15%, Dokumentasi 15%, Kreativitas 15%. Bonus +5 checkerboard kalibrasi asli + verifikasi error, +3 3D visualization interaktif Open3D. Format ZIP NIM_Nama_Project11.zip. Deadline 1 minggu.

**Slide 29** — Tugas Video: Tunjukkan checkerboard fisik dan proses kalibrasi. Tunjukkan gambar 3D dari berbagai angle (rotate di Open3D/matplotlib). Demo 20 percobaan LIVE (40–60 mnt). Materi (10–15 mnt): wajib diagram epipolar geometry + SfM pipeline + stereo pipeline. Demo Project (10–15 mnt) — tunjukkan 3D output.

**Slide 30** — Rubrik Video: Pembukaan (5), Materi (10), 20 Percobaan (40 — 2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 kalibrasi hardware + point cloud nyata dari objek fisik, +3 3D visualization rotasi video mengesankan. Penalti −2/percobaan tidak tampil. "Rekonstruksi Dunia 3D dari Foto 2D!"
