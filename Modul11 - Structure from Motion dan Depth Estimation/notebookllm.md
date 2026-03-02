# PROMPT NOTEBOOKLLM
# MODUL 11: STRUCTURE FROM MOTION DAN DEPTH ESTIMATION

---

## PROMPT 1: SLIDE 1–15 (Materi Dasar + Jobsheet Awal)

Buatkan presentasi 15 slide untuk mata kuliah Praktikum Komputer Vision, Modul 11: Structure from Motion dan Depth Estimation.

Slide 1: Judul "MODUL 11: Structure from Motion dan Depth Estimation" dengan daftar topik utama: Epipolar Geometry, Fundamental & Essential Matrix, Triangulasi, Stereo Vision, Disparity Map, Depth Estimation, SfM Pipeline.

Slide 2: Pendahuluan — rekonstruksi 3D dari gambar 2D adalah salah satu problem paling penting di komputer vision. Aplikasi: autonomous driving, AR/VR, pemetaan 3D, robot navigation.

Slide 3: Epipolar Geometry — konsep: dua kamera melihat titik 3D P yang sama. Komponen: epipole (proyeksi pusat kamera lain), epipolar plane (bidang P dan dua pusat kamera), epipolar line (irisan epipolar plane dengan image plane).

Slide 4: Epipolar Constraint — formula x'^T F x = 0. Artinya pencarian korespondensi berkurang dari 2D area menjadi 1D garis. Ini sangat mempercepat stereo matching.

Slide 5: Fundamental Matrix — matriks 3×3, rank 2, 7 DOF. Menghubungkan dua pandangan tanpa memerlukan kalibrasi kamera. 8-point algorithm: minimal 8 korespondensi, susun sistem Af=0, selesaikan dengan SVD.

Slide 6: Essential Matrix — E = K'^T F K, versi terkalibrasi dari F. 5 DOF (3 rotasi + 2 translasi). Dekomposisi E → R, t menggunakan cv2.recoverPose(). 4 solusi, pilih yang semua titik di depan kedua kamera.

Slide 7: Triangulasi — menghitung posisi 3D dari korespondensi 2D pada dua kamera. cv2.triangulatePoints() menggunakan DLT method. Reprojection error mengukur kualitas: e = ||p - π(P,X)||².

Slide 8: Percobaan 1-3 — (1) Epipolar Geometry Visualisasi: diagram epipolar plane, epipolar constraint. (2) Fundamental Matrix: hitung F dengan 8-point, gambar epipolar lines. (3) Essential Matrix: dekomposisi E → R, t, visualisasi pose.

Slide 9: Percobaan 4-6 — (4) Epipolar Lines: feature matching → F → epipolar lines berwarna. (5) Triangulasi Titik 3D: proyeksi → triangulasi → error analysis. (6) Stereo Calibration: checkerboard, intrinsik/ekstrinsik, baseline.

Slide 10: Stereo Vision — konfigurasi dua kamera terkalibrasi. Stereo Calibration menentukan K1, K2, R, T. Stereo Rectification mentransformasi agar epipolar lines horizontal → pencarian 1D.

Slide 11: Disparity Map — disparity d = x_L - x_R. Dekat = disparity besar, jauh = disparity kecil. StereoBM: SAD-based block matching, cepat tapi kasar. StereoSGBM: semi-global optimization, akurat tapi lambat.

Slide 12: Parameter BM & SGBM — numDisparities: range pencarian (kelipatan 16). blockSize: ukuran window (ganjil). SGBM tambahan: P1, P2 (smoothness penalty), disp12MaxDiff, uniquenessRatio, speckleWindowSize.

Slide 13: Percobaan 7-9 — (7) Stereo Rectification: warp dengan H1/H2, verifikasi garis horizontal. (8) Block Matching: variasi numDisp dan blockSize. (9) SGBM: parameter P1/P2, colorbar disparity.

Slide 14: Depth dari Disparity — formula Z = f·B/d. Hubungan invers: disparity kecil → depth besar dan sangat sensitif noise. Resolusi depth: ΔZ ≈ Z²/(f·B). Q matrix untuk reprojectImageTo3D().

Slide 15: Percobaan 10-12 — (10) Monocular Depth: gradient-based + vertical heuristic. (11) Disparity to Depth: konversi Z=fB/d, profil baris. (12) BM vs SGBM: perbandingan waktu, kualitas, FPS.

---

## PROMPT 2: SLIDE 16–30 (Materi Lanjut + Analisis)

Lanjutkan presentasi 15 slide berikutnya (slide 16-30) untuk Modul 11: Structure from Motion dan Depth Estimation.

Slide 16: Disparity Post-Processing — WLS filter: menghaluskan disparity sambil mempertahankan edge. Parameter Lambda (smoothness) dan SigmaColor (edge sensitivity). Speckle filtering untuk menghilangkan noise kecil. Bilateral filter sebagai alternatif.

Slide 17: Point Cloud dari Depth — konversi depth map ke 3D: X = (u-cx)·Z/fx, Y = (v-cy)·Z/fy. Density bergantung pada resolusi gambar. Visualisasi dengan matplotlib 3D scatter atau Open3D.

Slide 18: PnP Pose Estimation — Perspective-n-Point: menentukan pose kamera dari korespondensi 2D-3D. Minimal 4 titik (P3P + 1). cv2.solvePnP() dan solvePnPRansac(). Aplikasi: AR, robot localization, camera tracking.

Slide 19: Percobaan 13-15 — (13) WLS Filter: raw vs filtered disparity, bilateral fallback. (14) Point Cloud: depth sintetis → 3D scatter, distribusi X/Y/Z. (15) PnP: solvePnP, bandingkan GT vs estimated rvec/tvec.

Slide 20: Stereo Matching Real-time — trade-off kecepatan vs kualitas. BM Fast (32, 9) → FPS tinggi tapi kasar. BM Quality (128, 21) → lebih baik tapi lambat. SGBM → terbaik tapi paling lambat. Resolusi gambar sangat mempengaruhi.

Slide 21: Depth Colorization — colormap yang umum: JET, PLASMA, INFERNO, TURBO, HOT, BONE. JET paling populer tapi memiliki artefak persepsi. PLASMA dan TURBO lebih seragam secara perseptual.

Slide 22: Efek Baseline — baseline besar: akurasi tinggi jarak jauh, overlap kecil jarak dekat. Baseline kecil: kebalikannya. ΔZ ∝ Z²/(f·B). Desain sistem stereo harus mempertimbangkan range operasi.

Slide 23: Percobaan 16-18 — (16) Realtime: benchmark 3 konfigurasi, FPS comparison. (17) Colorization: 6 colormap side-by-side. (18) Baseline Effect: grafik depth vs disparity untuk 4 baseline.

Slide 24: Depth Object Segmentation — segmentasi berdasarkan range kedalaman. Keuntungan: tidak terpengaruh iluminasi, efektif untuk warna serupa. Aplikasi: portrait mode (bokeh), robot grasping, obstacle avoidance.

Slide 25: SfM Pipeline — langkah lengkap: Feature Detection → Feature Matching → F Matrix → E Matrix → Pose Recovery → Triangulasi → Bundle Adjustment. Bundle Adjustment: optimasi Levenberg-Marquardt untuk semua parameter.

Slide 26: COLMAP — tool SfM state-of-the-art. Pipeline: SIFT extraction → exhaustive matching → incremental SfM → Multi-View Stereo (MVS). Menghasilkan dense point cloud + mesh dari kumpulan foto.

Slide 27: Percobaan 19-20 — (19) Depth Segmentation: multi-layer segmentasi, bar chart area. (20) SfM Pipeline: ORB → F → E → pose → triangulasi → 3D reconstruction, visualisasi point cloud.

Slide 28: Perbandingan Metode Depth — tabel: Stereo BM (cepat, kasar), SGBM (akurat, lambat), Monocular Heuristic (sangat cepat, tidak akurat), MiDaS (akurat, perlu GPU). Pilihan bergantung pada aplikasi.

Slide 29: Analisis Kritis — (1) Textureless problem: area tanpa tekstur gagal matching. (2) Occlusion: area terhalang tidak memiliki korespondensi. (3) Scale ambiguity: monocular hanya menghasilkan relative depth. (4) Computational cost: trade-off akurasi vs kecepatan.

Slide 30: Ringkasan Materi — Epipolar geometry = fondasi stereo vision. F dan E matrix menghubungkan dua pandangan. Disparity → Depth via Z=fB/d. SfM = rekonstruksi 3D dari multi-view. Post-processing (WLS) meningkatkan kualitas.

---

## PROMPT 3: SLIDE 31–45 (Aplikasi Lanjut + Project + Tugas Video)

Lanjutkan presentasi 15 slide terakhir (slide 31-45) untuk Modul 11: Structure from Motion dan Depth Estimation.

Slide 31: Aplikasi Autonomous Driving — stereo camera untuk depth estimation real-time. Obstacle detection dari depth map. Lane keeping dan collision avoidance. Tesla, Waymo, dan Mobileye menggunakan multi-camera SfM.

Slide 32: Aplikasi AR/VR — PnP untuk camera tracking dan penempatan objek virtual. SLAM (Simultaneous Localization and Mapping) menggunakan visual odometry. Depth estimation untuk realistic occlusion handling.

Slide 33: Aplikasi 3D Mapping — SfM + MVS untuk pemetaan bangunan, kota, terrain. Google Earth 3D menggunakan SfM dari foto satelit dan udara. COLMAP sebagai tool open-source terpopuler.

Slide 34: Monocular vs Stereo — Monocular: 1 kamera, ill-posed, butuh deep learning. Stereo: 2 kamera, well-posed, akurat. Trade-off: hardware cost vs accuracy. Deep stereo (RAFT-Stereo, AANet) mulai menggantikan metode klasik.

Slide 35: Deep Learning untuk Depth — MiDaS: relative depth dari satu gambar. Monodepth2: self-supervised monocular depth. DPT (Dense Prediction Transformer): state-of-the-art. ZoeDepth: metrik depth absolut.

Slide 36: Project Improvisasi 1-5 — (1) Multi-Baseline Stereo: 3+ baseline, compare accuracy. (2) Disparity Fusion BM+SGBM: weighted average. (3) Depth-Guided Segmentation: portrait mode effect. (4) Auto Parameter Tuning: grid search optimal. (5) Visual Odometry: trajectory dari sequence.

Slide 37: Project Improvisasi 6-10 — (6) Depth Super-Resolution: guided filter. (7) SfM 3+ View: incremental reconstruction. (8) Obstacle Detection: depth threshold warning. (9) Depth Histogram Analysis: statistik scene. (10) Stereo Confidence Map: left-right check.

Slide 38: Project Improvisasi 11-15 — (11) 3D Anaglyph: red-cyan stereoscopic. (12) Depth Completion: inpaint holes. (13) Camera Pose Graph: frustum visualization. (14) Realtime Colorization: toggle colormap keyboard. (15) Bundle Adjustment: scipy optimize.

Slide 39: Soal Cerita 1-4 — (1) Robot Warehouse: Z=fB/d, deteksi rak jarak <0.5m. (2) Quality Control: tinggi objek dari depth difference. (3) Auto Parking: 3 dinding pada 2/4/6m. (4) Rekonstruksi Bangunan: pipeline SfM → point cloud 3D.

Slide 40: Soal Cerita 5-7 — (5) Background Removal: depth threshold 1.5m, blur background. (6) Drone Avoidance: 3 zona depth, pilih teraman. (7) System Comparison: BM/SGBM × 2 resolusi, tabel perbandingan.

Slide 41: Soal Cerita 8-10 — (8) AR Furniture: PnP dari marker lantai, visualisasi 3D. (9) Topografi: triangulasi → point cloud → range Z terrain. (10) Evaluasi Depth: GT sintetis, MAE/RMSE, ranking metode.

Slide 42: Tugas Video — Durasi 15-25 menit. Struktur: Pembukaan (1-2 min), Materi (4-6 min), Demo 10+ percobaan (6-10 min), Demo Project (2-4 min), Penutup (1-2 min). Format: Video_Modul11_NIM_Nama.mp4, minimal 720p.

Slide 43: Tips Demo Video — Jelaskan setiap parameter dan pengaruhnya. Tunjukkan side-by-side: BM vs SGBM, raw vs WLS filtered. Gunakan 3D scatter plot untuk point cloud. Jelaskan formula Z=fB/d dengan contoh numerik. Tunjukkan efek baseline pada grafik.

Slide 44: Evaluasi & Rubrik — Improvisasi (30%): 3 dari 15, kreativitas dan implementasi. Soal Cerita (40%): 10 soal, output benar. Laporan (15%): analisis, format. Kode (15%): struktur, komentar. Video: materi 25%, demo 30%, project 20%, presentasi 15%, teknis 10%.

Slide 45: Penutup — SfM dan depth estimation menghubungkan visi 2D dengan dunia 3D. Stereo vision memberikan depth akurat, SfM memungkinkan rekonstruksi dari multi-view. Masa depan: NeRF, 3D Gaussian Splatting, learning-based SfM. Lanjut ke Modul 12: Rekonstruksi 3D dan Image-Based Rendering.
