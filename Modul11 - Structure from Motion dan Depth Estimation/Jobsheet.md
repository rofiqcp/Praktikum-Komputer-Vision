# JOBSHEET PRAKTIKUM
# MODUL 11: STRUCTURE FROM MOTION DAN DEPTH ESTIMATION

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa mampu:
1. Memahami dan memvisualisasikan konsep epipolar geometry (epipole, epipolar lines, epipolar plane).
2. Menghitung Fundamental Matrix dari korespondensi titik menggunakan 8-point algorithm.
3. Menghitung Essential Matrix dan mendekomposisi menjadi rotasi dan translasi kamera.
4. Menggambar dan menganalisis epipolar lines pada pasangan gambar stereo.
5. Melakukan triangulasi untuk menghitung posisi 3D dari korespondensi 2D.
6. Melakukan kalibrasi kamera stereo (intrinsik dan ekstrinsik).
7. Menerapkan stereo rectification agar epipolar lines horizontal.
8. Menghitung disparity map menggunakan StereoBM (Block Matching).
9. Menghitung disparity map menggunakan StereoSGBM (Semi-Global Block Matching).
10. Melakukan estimasi kedalaman monocular dari satu gambar.
11. Mengkonversi disparity map menjadi depth map (Z = f·B/d).
12. Membandingkan kualitas dan kecepatan StereoBM vs StereoSGBM.
13. Meningkatkan kualitas disparity dengan WLS filter.
14. Membuat point cloud 3D dari depth map.
15. Melakukan estimasi pose kamera dengan solvePnP.
16. Mengimplementasikan stereo matching real-time.
17. Memvisualisasikan depth map dengan berbagai colormap.
18. Menganalisis pengaruh baseline terhadap akurasi depth.
19. Melakukan segmentasi objek berdasarkan kedalaman.
20. Membangun pipeline SfM sederhana: feature matching → F → E → pose → triangulasi.

---

## 2. ALAT DAN BAHAN

### Perangkat Keras
- Laptop/PC (webcam opsional)

### Perangkat Lunak
- Python 3.8+, VS Code
- Library: `opencv-python`, `opencv-contrib-python`, `numpy`, `matplotlib`
- Opsional: `open3d` (visualisasi point cloud), `scipy`

### Dataset
- Gambar stereo dari folder `image/` (diunduh via `download_image.py`):
  - `stereo_left.png`, `stereo_right.png` — pasangan stereo
  - `building_stereo.jpg` — bangunan stereo
  - `gambar_fitur.png`, `gambar_fitur_rotasi.png` — matching fitur
  - `indoor_scene.jpg` — scene dalam ruangan
  - `gambar_depth.png` — depth map referensi

---

## 3. LANGKAH KERJA

### Percobaan 1: Epipolar Geometry Visualisasi

**Tujuan**: Memvisualisasikan konsep dasar epipolar geometry.

**Langkah Kerja**:
1. Buat file `01_epipolar_geometry_visualisasi.py`.
2. Buat diagram epipolar geometry: dua kamera, titik 3D, baseline, projection rays.
3. Visualisasikan epipolar plane yang terbentuk oleh dua pusat kamera dan titik 3D.
4. Buat diagram matematis epipolar constraint: $x'^T F x = 0$.
5. Jelaskan sifat Fundamental Matrix: rank 2, 7 DOF, dan hubungan dengan epipole.
6. Simpan diagram ke folder `output/`.

**Analisis**: Epipolar geometry adalah fondasi struktur dua pandangan. Epipolar constraint mengurangi pencarian korespondensi dari 2D menjadi 1D (garis epipolar).

---

### Percobaan 2: Fundamental Matrix

**Tujuan**: Menghitung Fundamental Matrix dari korespondensi titik.

**Langkah Kerja**:
1. Buat file `02_fundamental_matrix.py`.
2. Load pasangan gambar stereo dari `image/`.
3. Buat/gunakan minimal 8 pasangan titik koresponden.
4. Hitung F menggunakan `cv2.findFundamentalMat()` dengan metode 8-point.
5. Verifikasi: cek rank F = 2, det(F) ≈ 0.
6. Hitung epipolar lines: `l' = Fx`, `l = F^Tx'`.
7. Gambar epipolar lines pada kedua gambar.
8. Simpan hasil ke folder `output/`.

**Analisis**: F menghubungkan titik di gambar kiri ke garis di gambar kanan. Kualitas F bergantung pada jumlah dan distribusi korespondensi.

---

### Percobaan 3: Essential Matrix dan Pose Estimation

**Tujuan**: Menghitung Essential Matrix dan mengekstrak pose kamera.

**Langkah Kerja**:
1. Buat file `03_essential_matrix_pose.py`.
2. Definisikan matriks intrinsik K (focal length dan principal point).
3. Buat korespondensi sintetis atau gunakan gambar stereo real.
4. Hitung E menggunakan `cv2.findEssentialMat()`.
5. Dekomposisi E menjadi R dan t menggunakan `cv2.recoverPose()`.
6. Visualisasikan korespondensi titik dan relative pose kamera.
7. Bandingkan dengan ground truth jika tersedia.
8. Simpan hasil ke folder `output/`.

**Analisis**: E = K'^T F K, hanya berlaku jika kamera terkalibrasi. Dekomposisi E menghasilkan 4 solusi, dipilih yang semua titik di depan kedua kamera.

---

### Percobaan 4: Epipolar Lines

**Tujuan**: Menggambar dan menganalisis epipolar lines menggunakan feature matching.

**Langkah Kerja**:
1. Buat file `04_epipolar_lines.py`.
2. Load pasangan stereo, deteksi fitur ORB, match dengan BFMatcher.
3. Hitung F dengan RANSAC.
4. Hitung epipolar lines menggunakan `cv2.computeCorrespondEpilines()`.
5. Gambar epipolar lines berwarna pada gambar kiri, titik koresponden pada gambar kanan.
6. Verifikasi: semua epipolar lines pada gambar yang sudah di-rektifikasi harus horizontal.
7. Simpan hasil ke folder `output/`.

**Analisis**: Epipolar lines yang baik menunjukkan F yang akurat. Setelah rektifikasi, semua epipolar lines menjadi horizontal.

---

### Percobaan 5: Triangulasi Titik 3D

**Tujuan**: Menghitung posisi 3D dari korespondensi 2D menggunakan triangulasi.

**Langkah Kerja**:
1. Buat file `05_triangulasi_titik_3d.py`.
2. Setup dua projection matrix P1, P2 dari K, R, t.
3. Buat titik 3D ground truth, proyeksikan ke kedua kamera.
4. Triangulasi menggunakan `cv2.triangulatePoints()`.
5. Konversi koordinat homogen ke Euclidean.
6. Hitung reprojection error per titik.
7. Visualisasikan 3D: ground truth vs hasil triangulasi.
8. Simpan grafik error dan point cloud ke folder `output/`.

**Analisis**: Triangulasi akurat bila baseline cukup besar dan noise rendah. Error meningkat untuk titik yang jauh dari kamera.

---

### Percobaan 6: Stereo Calibration

**Tujuan**: Melakukan kalibrasi kamera stereo menggunakan checkerboard.

**Langkah Kerja**:
1. Buat file `06_stereo_calibration.py`.
2. Buat gambar checkerboard sintetis.
3. Simulasikan deteksi corners dari beberapa posisi/orientasi.
4. Jelaskan konsep intrinsik (fx, fy, cx, cy) dan ekstrinsik (R, T).
5. Hitung baseline dari translasi vektor T.
6. Visualisasikan checkerboard dan distribusi corner detection.
7. Simpan hasil ke folder `output/`.

**Analisis**: Kalibrasi stereo menentukan hubungan geometris antara dua kamera. Akurasi kalibrasi sangat mempengaruhi kualitas depth estimation.

---

### Percobaan 7: Stereo Rectification

**Tujuan**: Menerapkan rektifikasi stereo agar epipolar lines menjadi horizontal.

**Langkah Kerja**:
1. Buat file `07_stereo_rectification.py`.
2. Load pasangan stereo, deteksi fitur dan match.
3. Hitung F dan gunakan `cv2.stereoRectifyUncalibrated()` untuk mendapat H1, H2.
4. Warp kedua gambar menggunakan `cv2.warpPerspective()`.
5. Gambar garis horizontal untuk verifikasi bahwa baris gambar sejajar.
6. Bandingkan gambar sebelum dan sesudah rektifikasi.
7. Simpan hasil ke folder `output/`.

**Analisis**: Setelah rektifikasi, stereo matching menjadi pencarian 1D (sepanjang baris). Ini mempercepat dan menyederhanakan Block Matching.

---

### Percobaan 8: Block Matching (StereoBM) Disparity

**Tujuan**: Menghitung disparity map menggunakan Block Matching.

**Langkah Kerja**:
1. Buat file `08_block_matching_disparity.py`.
2. Load pasangan stereo, konversi ke grayscale.
3. Buat StereoBM dengan `cv2.StereoBM_create(numDisparities, blockSize)`.
4. Variasikan parameter: numDisparities (32, 64, 128), blockSize (9, 15, 21).
5. Visualisasikan 4 variasi disparity map side-by-side.
6. Analisis: area mana yang menghasilkan disparity baik/buruk.
7. Simpan hasil ke folder `output/`.

**Analisis**: numDisparities menentukan range pencarian, blockSize menentukan Detail vs Noise trade-off. Area homogen tanpa tekstur menghasilkan disparity buruk.

---

### Percobaan 9: SGBM Disparity

**Tujuan**: Menghitung disparity menggunakan Semi-Global Block Matching.

**Langkah Kerja**:
1. Buat file `09_sgbm_disparity.py`.
2. Load pasangan stereo.
3. Buat StereoSGBM dengan parameter P1, P2, disp12MaxDiff, uniquenessRatio, speckleWindowSize.
4. Jelaskan peran masing-masing parameter.
5. Bandingkan disparity SGBM dengan BM secara visual.
6. Tampilkan colorbar untuk skala disparity.
7. Simpan hasil ke folder `output/`.

**Analisis**: SGBM mengoptimasi cost secara semi-global (8 atau 16 arah), menghasilkan disparity lebih halus dan akurat dibanding BM, tetapi lebih lambat.

---

### Percobaan 10: Monocular Depth Estimation

**Tujuan**: Estimasi kedalaman dari satu gambar tanpa stereo.

**Langkah Kerja**:
1. Buat file `10_monocular_depth_estimation.py`.
2. Load gambar single view.
3. Implementasikan estimasi depth berbasis gradien (area tajam = dekat).
4. Implementasikan heuristik vertikal (bawah gambar = dekat).
5. Gabungkan kedua metode.
6. Bandingkan visual ketiga metode.
7. Simpan hasil ke folder `output/`.

**Analisis**: Monocular depth adalah ill-posed problem (satu gambar memiliki banyak solusi depth). Heuristik sederhana terbatas; deep learning (MiDaS) jauh lebih akurat.

---

### Percobaan 11: Disparity to Depth

**Tujuan**: Konversi disparity map menjadi depth map menggunakan formula.

**Langkah Kerja**:
1. Buat file `11_disparity_to_depth.py`.
2. Buat disparity map sintetis dengan variasi nilai.
3. Terapkan formula: $Z = \frac{f \cdot B}{d}$, di mana f = focal, B = baseline, d = disparity.
4. Visualisasikan disparity, depth map, dan profil baris tengah.
5. Analisis: mengapa depth tidak linier terhadap disparity.
6. Simpan hasil ke folder `output/`.

**Analisis**: Hubungan disparity-depth bersifat inversely proportional. Disparity kecil (objek jauh) menghasilkan depth yang sangat sensitif terhadap noise.

---

### Percobaan 12: Perbandingan StereoBM vs StereoSGBM

**Tujuan**: Membandingkan kinerja dan kecepatan BM vs SGBM.

**Langkah Kerja**:
1. Buat file `12_stereo_bm_vs_sgbm.py`.
2. Load pasangan stereo.
3. Hitung disparity dengan BM dan SGBM.
4. Ukur waktu (ms) masing-masing metode.
5. Tampilkan side-by-side: input, BM, SGBM dengan label waktu.
6. Hitung speedup ratio.
7. Simpan hasil ke folder `output/`.

**Analisis**: BM 3-5x lebih cepat, cocok untuk real-time. SGBM lebih akurat di area sulit (oklusi, textureless). Pilihan bergantung pada kebutuhan aplikasi.

---

### Percobaan 13: WLS Filter Disparity

**Tujuan**: Meningkatkan kualitas disparity map menggunakan edge-aware filter.

**Langkah Kerja**:
1. Buat file `13_wls_filter_disparity.py`.
2. Hitung disparity raw menggunakan StereoBM.
3. Terapkan WLS filter (`ximgproc.createDisparityWLSFilter`) atau bilateral filter sebagai fallback.
4. Bandingkan disparity raw vs filtered.
5. Atur parameter Lambda dan SigmaColor.
6. Simpan perbandingan ke folder `output/`.

**Analisis**: WLS filter menghaluskan disparity sambil mempertahankan edge objek. Lambda mengontrol kekuatan smoothing, SigmaColor mengontrol sensitivitas edge.

---

### Percobaan 14: Point Cloud dari Depth Map

**Tujuan**: Membuat point cloud 3D dari depth map dan parameter kamera.

**Langkah Kerja**:
1. Buat file `14_point_cloud_from_depth.py`.
2. Buat depth map sintetis dengan beberapa objek pada kedalaman berbeda.
3. Definisikan matriks intrinsik K (fx, fy, cx, cy).
4. Konversi depth ke 3D: $X = (u - c_x) \cdot Z / f_x$, $Y = (v - c_y) \cdot Z / f_y$.
5. Visualisasikan point cloud 3D dengan matplotlib scatter.
6. Analisis distribusi spasial titik-titik 3D.
7. Simpan visualisasi ke folder `output/`.

**Analisis**: Point cloud memungkinkan pemahaman 3D scene. Density point cloud bergantung pada resolusi gambar dan range depth.

---

### Percobaan 15: PnP Pose Estimation

**Tujuan**: Estimasi pose kamera dari korespondensi 2D-3D.

**Langkah Kerja**:
1. Buat file `15_pnp_pose_estimation.py`.
2. Definisikan titik 3D objek dan proyeksikan ke 2D + noise.
3. Gunakan `cv2.solvePnP()` untuk estimasi rvec, tvec.
4. Bandingkan hasil dengan ground truth: hitung error rotasi dan translasi.
5. Visualisasikan titik 2D dan pose kamera 3D.
6. Simpan hasil ke folder `output/`.

**Analisis**: PnP memerlukan minimal 4 korespondensi 2D-3D untuk solusi unik. RANSAC variant (`solvePnPRansac`) lebih robust terhadap outlier.

---

### Percobaan 16: Stereo Matching Real-time

**Tujuan**: Implementasi stereo matching dengan parameter tuning untuk performa real-time.

**Langkah Kerja**:
1. Buat file `16_stereo_matching_realtime.py`.
2. Bandingkan tiga konfigurasi: BM Fast (numDisp=32, block=9), BM Quality (numDisp=128, block=21), SGBM.
3. Ukur FPS masing-masing konfigurasi (rata-rata dari beberapa frame).
4. Visualisasikan disparity map dengan label FPS.
5. Analisis trade-off kecepatan vs kualitas.
6. Simpan hasil ke folder `output/`.

**Analisis**: Parameter tuning kritis untuk aplikasi real-time. Resolusi gambar sangat mempengaruhi kecepatan.

---

### Percobaan 17: Depth Map Colorization

**Tujuan**: Memvisualisasikan depth map dengan berbagai colormap.

**Langkah Kerja**:
1. Buat file `17_depth_map_colorization.py`.
2. Buat depth map sintetis dengan gradien.
3. Terapkan 6 colormap: JET, PLASMA, INFERNO, TURBO, HOT, BONE.
4. Tampilkan 6 visualisasi side-by-side.
5. Analisis: colormap mana paling intuitif untuk depth.
6. Simpan hasil ke folder `output/`.

**Analisis**: Colormap JET paling umum tetapi memiliki artefak persepsi. PLASMA dan TURBO secara perseptual lebih seragam.

---

### Percobaan 18: Efek Baseline terhadap Depth

**Tujuan**: Menganalisis pengaruh jarak antar kamera (baseline) terhadap akurasi depth.

**Langkah Kerja**:
1. Buat file `18_baseline_effect_depth.py`.
2. Simulasikan formula Z = f·B/d dengan berbagai baseline (0.05, 0.1, 0.2, 0.5 m).
3. Plot Depth vs Disparity untuk setiap baseline.
4. Plot resolusi depth ($\Delta Z \approx Z^2 / (f \cdot B)$) vs jarak.
5. Analisis trade-off: baseline besar vs kecil.
6. Simpan grafik ke folder `output/`.

**Analisis**: Baseline besar → akurasi depth lebih baik di jarak jauh, tetapi area matching overlap berkurang di jarak dekat. Trade-off desain sistem stereo.

---

### Percobaan 19: Depth Object Segmentation

**Tujuan**: Segmentasi objek berdasarkan kedalaman.

**Langkah Kerja**:
1. Buat file `19_depth_object_segmentation.py`.
2. Buat depth map sintetis dengan beberapa objek pada kedalaman berbeda.
3. Segmentasi depth menjadi beberapa layer berdasarkan range nilai.
4. Hitung area piksel per layer.
5. Visualisasikan: depth map, segmentation map, dan bar chart area.
6. Simpan hasil ke folder `output/`.

**Analisis**: Depth-based segmentation efektif untuk memisahkan foreground dan background. Lebih robust terhadap perubahan warna/tekstur dibanding segmentasi berbasis warna.

---

### Percobaan 20: Multiview Reconstruction Pipeline

**Tujuan**: Membangun pipeline Structure from Motion end-to-end.

**Langkah Kerja**:
1. Buat file `20_multiview_reconstruction_pipeline.py`.
2. Load dua gambar, deteksi fitur ORB, match.
3. Hitung F → E → dekomposisi pose (R, t).
4. Triangulasi semua inlier → point cloud 3D.
5. Visualisasikan: kedua view dan 3D reconstruction.
6. Tampilkan jumlah matches, inliers, titik 3D valid.
7. Cetak ringkasan pipeline: Feature → F → E → Pose → Triangulasi → 3D.
8. Simpan hasil ke folder `output/`.

**Analisis**: SfM pipeline menghubungkan semua konsep modul ini. Kualitas output bergantung pada kualitas fitur, jumlah korespondensi, dan baseline antar pandangan.

---

## 4. ANALISIS DAN PEMBAHASAN

Jawab pertanyaan berikut dalam laporan:
1. Mengapa epipolar constraint mengurangi pencarian korespondensi dari 2D menjadi 1D?
2. Apa perbedaan Fundamental Matrix dan Essential Matrix? Kapan menggunakan masing-masing?
3. Jelaskan rumus $Z = f \cdot B / d$ dan konsekuensi hubungan inversnya.
4. Mengapa area tanpa tekstur (textureless) menghasilkan disparity yang buruk?
5. Bagaimana pengaruh baseline terhadap akurasi dan range depth?
6. Apa kelebihan SGBM dibanding BM? Kapan sebaiknya menggunakan BM?
7. Jelaskan langkah-langkah pipeline SfM dan peran masing-masing tahap.
8. Apa itu PnP dan mengapa minimal diperlukan 4 korespondensi 2D-3D?
9. Mengapa WLS filter efektif meningkatkan kualitas disparity?
10. Bagaimana monocular depth estimation berbeda dari stereo-based?

---

## 5. KESIMPULAN

Tuliskan kesimpulan berdasarkan hasil percobaan, meliputi:
- Pemahaman konsep epipolar geometry dan perannya dalam stereo vision.
- Kemampuan menghitung F, E, dan melakukan triangulasi.
- Perbandingan metode disparity (BM vs SGBM) dan post-processing.
- Pemahaman pipeline SfM end-to-end.
- Aplikasi depth estimation dalam segmentasi dan rekonstruksi 3D.
