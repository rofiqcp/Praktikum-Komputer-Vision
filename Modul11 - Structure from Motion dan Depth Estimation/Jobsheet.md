# JOBSHEET MODUL 11: STRUCTURE FROM MOTION DAN DEPTH ESTIMATION

---

## Tujuan Praktikum
1. Memahami epipolar geometry (Fundamental dan Essential matrix).
2. Mengimplementasikan triangulasi 3D dari dua view.
3. Memahami pipeline Structure from Motion (SfM).
4. Mengimplementasikan stereo calibration dan rectification.
5. Mengimplementasikan stereo matching (BM dan SGBM).
6. Mencoba monocular depth estimation.

---

## Alat dan Bahan
- **Hardware**: PC/Laptop, webcam, smartphone, checkerboard pattern (cetak A4).
- **Software**: Python 3.8+, Jupyter Notebook / VS Code.
- **Library**: OpenCV (`opencv-contrib-python`), NumPy, Matplotlib, Open3D (opsional untuk visualisasi 3D).
- **Dataset**: Pasangan gambar stereo, video objek dari berbagai sudut, checkerboard images.

### Persiapan
```text
1. Cetak checkerboard pattern (9×6 inner corners) ukuran A4.
2. Ambil foto objek dari 5+ sudut berbeda (untuk SfM).
3. Ambil pasangan gambar stereo (geser kamera horizontal ~5-10 cm).
```

---

## Percobaan 1: Feature Matching Multi-View

### Tujuan
Membangun korespondensi fitur antar banyak view dari objek yang sama.

### Dasar Teori
SfM dimulai dari feature matching yang robust antar pasangan gambar. Match yang konsisten di banyak view menjadi dasar triangulasi.

### Langkah Kerja
1. Ambil 5 foto objek dari sudut berbeda (putar 15-30° per foto).
2. Deteksi SIFT keypoints + descriptors pada semua gambar.
3. Match fitur antar pasangan bersebelahan: (1,2), (2,3), (3,4), (4,5).
4. Apply Lowe's ratio test (0.75).
5. Visualisasikan matches per pasangan.
6. Cari fitur yang muncul di 3+ gambar (track across views).
7. Buat tabel: pasangan, jumlah matches, jumlah good matches.
8. Uji match (1,3), (1,4) → match antar view yang jauh.
9. Visualisasikan distribusi spasial keypoints per gambar.
10. Identifikasi fitur yang paling sering match (most robust features).

### Analisis Percobaan 1
- Berapa persen fitur yang dapat di-track di 3+ view?
- Apakah match antar view yang jauh (1,4) masih banyak?
- Area mana dari objek yang paling banyak fitur robust?
- Bagaimana sudut pandang mempengaruhi jumlah matches?

---

## Percobaan 2: Fundamental Matrix Estimation

### Tujuan
Mengestimasi Fundamental Matrix dari pasangan gambar dan memvisualisasikan epipolar lines.

### Dasar Teori
Fundamental matrix $F$ mendefinisikan epipolar geometry antara 2 view tanpa memerlukan informasi intrinsik kamera. Setiap titik di satu gambar berkorespondensi dengan garis (epipolar line) di gambar lain.

### Langkah Kerja
1. Load 2 gambar dari scene yang sama (sudut berbeda).
2. Deteksi + match fitur (SIFT + FLANN + ratio test).
3. Estimasi F: `cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)`.
4. Hitung persentase inlier.
5. Hitung epipolar lines: `cv2.computeCorrespondEpilines()`.
6. Gambar epipolar lines pada kedua gambar.
7. Verifikasi: epipolar lines harus melewati titik koresponden.
8. Hitung epipolar error: jarak titik ke epipolar line rata-rata.
9. Bandingkan FM_RANSAC vs FM_LMEDS vs FM_8POINT.
10. Visualisasikan epipole (titik konvergensi epipolar lines).

### Analisis Percobaan 2
- Berapa epipolar error rata-rata (piksel)?
- Metode estimasi mana (RANSAC, LMEDS, 8-point) yang paling akurat?
- Apakah epipolar lines konvergen ke satu titik (epipole)?
- Apa terjadi jika gerakan kamera hanya translasi murni?

---

## Percobaan 3: Essential Matrix dan Pose Recovery

### Tujuan
Mengestimasi Essential Matrix dan me-recover relative pose (R, t) antara 2 kamera.

### Dasar Teori
Essential matrix $E = K'^T F K$ menghubungkan normalized coordinates. Dekomposisi E menghasilkan rotasi dan translasi relatif kamera.

### Langkah Kerja
1. Kalibrasi kamera: ambil 10+ foto checkerboard → `calibrateCamera()` → dapatkan $K$.
2. Load 2 gambar dari scene.
3. Deteksi + match fitur.
4. Estimasi E: `cv2.findEssentialMat(pts1, pts2, K)`.
5. Recover pose: `cv2.recoverPose(E, pts1, pts2, K)`.
6. Print R (rotasi) dan t (translasi) → interpretasikan gerakan kamera.
7. Atau: hitung E dari F → $E = K^T F K$ → bandingkan.
8. Verifikasi: hitung $\mathbf{x'}^T E \mathbf{x}$ untuk setiap match → harus ≈ 0.
9. Visualisasikan arah gerakan kamera (translasi vektor).
10. Uji pada 3 pasangan gambar dengan gerakan berbeda.

### Analisis Percobaan 3
- Apakah R dan t sesuai dengan gerakan kamera sebenarnya?
- Berapa error $\mathbf{x'}^T E \mathbf{x}$ rata-rata?
- Mengapa t hanya menunjukkan arah (bukan magnitud)?
- Pada kondisi apa pose recovery gagal (degenerate case)?

---

## Percobaan 4: Triangulasi 3D

### Tujuan
Merekonstruksi titik 3D dari 2 view menggunakan triangulasi.

### Dasar Teori
Dari R, t antara 2 kamera + korespondensi titik, kita bisa menghitung posisi 3D titik menggunakan triangulasi (interseksi ray).

### Langkah Kerja
1. Gunakan K, R, t dari Percobaan 3.
2. Buat projection matrix: $P_1 = K[I|0]$, $P_2 = K[R|t]$.
3. Triangulasi: `cv2.triangulatePoints(P1, P2, pts1, pts2)`.
4. Konversi homogeneous → Euclidean (bagi dengan w).
5. Filter: hapus titik di belakang kamera (Z < 0).
6. Plot 3D point cloud menggunakan Matplotlib `Axes3D` atau Open3D.
7. Plot posisi kamera di 3D space.
8. Hitung reprojection error: project 3D points kembali → bandingkan.
9. Variasikan baseline (jarak kamera): kecil vs besar → bandingkan akurasi 3D.
10. Colorize point cloud berdasarkan warna piksel dari gambar.

### Analisis Percobaan 4
- Berapa banyak titik 3D yang berhasil direkonstruksi?
- Berapa reprojection error rata-rata?
- Apakah bentuk objek bisa dikenali dari point cloud?
- Bagaimana baseline mempengaruhi akurasi depth?

---

## Percobaan 5: Visual Odometry Sederhana

### Tujuan
Mengimplementasikan visual odometry dari sekuens gambar/video.

### Dasar Teori
Visual odometry mengestimasi gerakan kamera dari frame ke frame, mengakumulasi pose untuk mendapatkan trajectory lengkap.

### Langkah Kerja
1. Rekam video sambil berjalan lurus (atau gunakan dataset seperti KITTI).
2. Baca frame secara berurutan.
3. Deteksi fitur → track antar frame.
4. Estimasi Essential matrix → recover R, t per pasangan frame.
5. Akumulasi pose: $T_{global} = T_{global} \cdot T_{relative}$.
6. Plot trajectory kamera (bird's-eye view: X vs Z).
7. Bandingkan dengan ground truth jika ada (atau intuisi gerakan).
8. Identifikasi scale drift (skala translasi berbeda per frame).
9. Coba absolute scale: gunakan objek dengan ukuran diketahui.
10. Plot trajectory 3D + orientasi kamera per keyframe.

### Analisis Percobaan 5
- Apakah trajectory sesuai dengan gerakan sebenarnya?
- Apa yang menyebabkan drift pada trajectory?
- Berapa frame per detik yang bisa diproses?
- Bagaimana cara mengatasi scale ambiguity pada monocular VO?

---

## Percobaan 6: Stereo Calibration

### Tujuan
Mengkalibrasi pasangan kamera stereo menggunakan checkerboard.

### Dasar Teori
Stereo calibration menentukan intrinsic parameters kedua kamera + extrinsic (R, T) di antara keduanya, diperlukan untuk rectification dan depth estimation.

### Langkah Kerja
1. Setup dua kamera atau gunakan satu kamera yang digeser horizontal.
2. Ambil 15+ pasangan gambar checkerboard (tampilkan di berbagai posisi/orientasi).
3. Deteksi chessboard corners di kedua gambar.
4. Kalibrasi individual: `calibrateCamera()` per kamera.
5. Stereo calibration: `stereoCalibrate()`.
6. Print hasil: K1, K2, dist1, dist2, R, T, E, F.
7. Hitung baseline: $|T|$ dalam unit yang diketahui (cm).
8. Hitung reprojection error per kamera.
9. Visualisasikan detected corners pada beberapa pasangan gambar.
10. Simpan parameter kalibrasi ke file (JSON atau NPZ).

### Analisis Percobaan 6
- Berapa reprojection error per kamera?
- Apakah baseline sesuai dengan jarak fisik kamera?
- Berapa distortion coefficient yang signifikan?
- Apa pengaruh jumlah gambar kalibrasi pada akurasi?

---

## Percobaan 7: Stereo Rectification

### Tujuan
Merektifikasi pasangan gambar stereo agar epipolar lines horizontal.

### Dasar Teori
Rectification mentransformasi kedua gambar sehingga titik koresponden berada di baris piksel yang sama. Ini memplifikasi stereo matching menjadi pencarian 1D.

### Langkah Kerja
1. Load parameter kalibrasi stereo.
2. Hitung rectification: `cv2.stereoRectify()`.
3. Buat remap: `cv2.initUndistortRectifyMap()`.
4. Remap kedua gambar: `cv2.remap()`.
5. Tampilkan: gambar asli vs rectified (side-by-side).
6. Gambar garis horizontal pada gambar rectified → verifikasi fitur sejajar.
7. Overlay kedua gambar rectified → objek yang sama harus geser horizontal.
8. Crop valid ROI dari rectification.
9. Hitung dan visualisasikan disparity awal (hanya horizontal shift).
10. Simpan gambar rectified untuk Percobaan 8.

### Analisis Percobaan 7
- Apakah epipolar lines sudah horizontal setelah rectification?
- Berapa area gambar yang hilang setelah rectification?
- Apakah fitur di kedua gambar benar-benar sejajar horizontal?
- Apa artinya Q matrix dari stereoRectify?

---

## Percobaan 8: Block Matching Disparity

### Tujuan
Menghitung disparity map menggunakan Block Matching (BM).

### Dasar Teori
BM mencari pasangan piksel di gambar kiri dan kanan menggunakan SAD (Sum of Absolute Differences) dalam block kecil. Disparity = perpindahan horizontal match.

### Langkah Kerja
1. Load gambar rectified (dari Percobaan 7 atau dataset Middlebury).
2. Konversi ke grayscale.
3. Buat StereoBM: `cv2.StereoBM_create(numDisparities=64, blockSize=15)`.
4. Compute disparity.
5. Visualisasikan disparity map (colormap JET).
6. Variasikan `numDisparities` (16, 32, 64, 128).
7. Variasikan `blockSize` (5, 9, 15, 21).
8. Post-processing: WLS filter untuk smooth disparity.
9. Konversi disparity ke depth: $Z = fb/d$.
10. Visualisasikan depth map.

### Analisis Percobaan 8
- Parameter mana yang menghasilkan disparity paling akurat?
- Area mana yang memiliki disparity noise (textureless regions)?
- Apakah depth map sesuai dengan jarak fisik objek?
- Apa limitasi Block Matching?

---

## Percobaan 9: Semi-Global Matching (SGBM)

### Tujuan
Menghitung disparity map menggunakan SGBM yang lebih akurat dari BM.

### Dasar Teori
SGBM mengoptimalkan cost function yang memperhitungkan smoothness dari 8 arah, menghasilkan disparity yang lebih konsisten terutama di area low-texture.

### Langkah Kerja
1. Load gambar rectified yang sama.
2. Buat StereoSGBM: set P1 (smoothness penalty kecil), P2 (smoothness penalty besar).
3. Compute disparity.
4. Bandingkan visual: BM vs SGBM.
5. Variasikan P1 dan P2.
6. Variasikan `mode`: SGBM, SGBM_3WAY, HH.
7. Post-processing: WLS filter.
8. Konversi ke depth map.
9. Hitung statistik: disparity coverage (% piksel valid).
10. Buat tabel: BM vs SGBM (parameter, coverage, kualitas visual, waktu).

### Analisis Percobaan 9
- Seberapa besar peningkatan SGBM vs BM?
- Area mana yang paling terpengaruh oleh smoothness penalty?
- Mode SGBM mana yang optimal (kualitas vs waktu)?
- Berapa banyak piksel invalid setelah SGBM?

---

## Percobaan 10: Monocular Depth Estimation

### Tujuan
Mengestimasi depth map dari SATU gambar menggunakan deep learning (MiDaS).

### Dasar Teori
Monocular depth estimation menggunakan CNN yang dilatih pada dataset gambar-depth untuk memprediksi kedalaman relatif dari satu gambar. Hasilnya relatif (bukan metrik absolut).

### Langkah Kerja
1. Download model MiDaS (small atau large) dalam format ONNX.
2. Load model dengan `cv2.dnn.readNet()`.
3. Preprocessing: resize ke 256×256 atau 384×384, normalize.
4. Forward pass → depth map.
5. Post-processing: resize ke ukuran asli, normalize ke 0-255.
6. Visualisasikan depth map (colormap).
7. Uji pada 5 gambar berbeda: indoor, outdoor, close-up, landscape, street.
8. Bandingkan depth MiDaS vs stereo depth (jika ada) pada gambar yang sama.
9. Overlay depth map semi-transparent pada gambar asli.
10. Gunakan depth map untuk synthetic bokeh (dari Modul 10).

### Analisis Percobaan 10
- Apakah depth map sesuai dengan intuisi kedalaman?
- Pada jenis scene apa MiDaS paling akurat?
- Apa perbedaan utama monocular vs stereo depth?
- Bagaimana depth map bisa digunakan untuk aplikasi lain?

---

## Percobaan 11: Konversi Disparity ke Depth Map

### Tujuan
Memahami hubungan matematis antara disparity dan depth menggunakan rumus $Z = f \cdot B / d$ serta membandingkan konversi manual dengan `cv2.reprojectImageTo3D()`.

### Dasar Teori
Depth (kedalaman) suatu titik dapat dihitung dari disparity menggunakan rumus $Z = f \cdot B / d$, di mana $f$ adalah focal length, $B$ adalah baseline, dan $d$ adalah disparity. Matriks Q (4×4) dari stereo rectification memungkinkan konversi disparity ke koordinat 3D secara langsung menggunakan `cv2.reprojectImageTo3D()`. Analisis depth zones (near/mid/far) membantu memahami distribusi kedalaman objek dalam scene.

### Langkah Kerja
1. Buat pasangan stereo sintetis dengan objek pada depth yang diketahui (ground truth).
2. Definisikan parameter kamera sintetis: `focal_length` dan `baseline`.
3. Hitung disparity map menggunakan `cv2.StereoBM_create()`.
4. Konversi disparity ke depth secara manual: $Z = f \cdot B / d$ menggunakan `np.where()`.
5. Bangun matriks Q dan gunakan `cv2.reprojectImageTo3D()` untuk konversi otomatis.
6. Bandingkan hasil depth manual vs `reprojectImageTo3D()` → hitung perbedaan.
7. Normalisasi depth map dengan `cv2.normalize()` dan visualisasikan dengan `cv2.applyColorMap()`.
8. Segmentasi depth zones: near (< 300), mid (300–600), far (> 600) menggunakan `np.clip()`.
9. Buat histogram distribusi depth dan analisis persebaran objek.
10. Bandingkan depth yang diestimasi dengan ground truth → hitung error per objek.

### Analisis Percobaan 11
- Seberapa akurat konversi $Z = f \cdot B / d$ dibandingkan ground truth?
- Apakah hasil `cv2.reprojectImageTo3D()` identik dengan konversi manual?
- Bagaimana distribusi depth zones (near/mid/far) pada scene?
- Pada rentang disparity berapa konversi ke depth paling tidak stabil?

---

## Percobaan 12: Perbandingan Stereo BM vs SGBM

### Tujuan
Membandingkan performa StereoBM dan StereoSGBM secara menyeluruh pada beberapa scene dengan tingkat kesulitan berbeda.

### Dasar Teori
StereoBM menggunakan block matching sederhana (SAD) yang cepat namun kurang akurat pada area low-texture. StereoSGBM mengoptimalkan cost function dari 8 arah (semi-global), menghasilkan disparity lebih konsisten tetapi lebih lambat. Perbandingan meliputi coverage (persentase piksel valid), kualitas visual, dan waktu komputasi.

### Langkah Kerja
1. Buat 3 scene sintetis dengan tingkat kesulitan berbeda: easy, medium, hard.
2. Buat `cv2.StereoBM_create()` dengan parameter `numDisparities` dan `blockSize`.
3. Buat `cv2.StereoSGBM_create()` dengan parameter P1, P2, dan mode.
4. Hitung disparity menggunakan `stereo.compute()` untuk kedua metode pada setiap scene.
5. Ukur waktu komputasi menggunakan `time.time()` untuk setiap metode.
6. Hitung coverage: persentase piksel dengan disparity valid (> 0).
7. Normalisasi dan visualisasikan disparity map dengan `cv2.applyColorMap()`.
8. Buat grid perbandingan visual: BM vs SGBM per scene.
9. Buat tabel metrik: waktu, coverage, kualitas visual untuk setiap kombinasi.
10. Analisis pro/kontra masing-masing metode berdasarkan tipe scene.

### Analisis Percobaan 12
- Scene tipe apa yang paling diuntungkan oleh SGBM dibanding BM?
- Berapa rasio waktu komputasi SGBM/BM pada setiap scene?
- Apakah coverage SGBM selalu lebih tinggi dari BM?
- Pada kondisi apa BM sudah cukup baik sehingga SGBM tidak diperlukan?

---

## Percobaan 13: WLS Filter untuk Post-Processing Disparity

### Tujuan
Mempelajari cara memperhalus disparity map menggunakan WLS Filter, median blur, dan bilateral filter.

### Dasar Teori
Disparity map mentah sering memiliki noise dan lubang (holes) terutama di area textureless. Median blur efektif menghilangkan salt-and-pepper noise, bilateral filter memperhalus sambil menjaga tepi (edge-preserving), dan WLS (Weighted Least Squares) Filter menggunakan informasi gambar asli sebagai panduan untuk menghasilkan disparity yang halus dan edge-aware.

### Langkah Kerja
1. Buat pasangan stereo sintetis dan hitung raw disparity menggunakan `cv2.StereoSGBM_create()`.
2. Buat right matcher: `cv2.ximgproc.createRightMatcher(left_matcher)`.
3. Hitung disparity kiri dan kanan.
4. Buat WLS filter: `cv2.ximgproc.createDisparityWLSFilter(left_matcher)`.
5. Set parameter WLS: `setLambda()` dan `setSigmaColor()`.
6. Filter disparity: `wls_filter.filter(disp_left, img_left, disparity_map_right=disp_right)`.
7. Terapkan `cv2.medianBlur()` pada raw disparity sebagai pembanding.
8. Terapkan `cv2.bilateralFilter()` pada raw disparity.
9. Hitung metrik: coverage dan smoothness untuk setiap metode filtering.
10. Buat grid perbandingan visual: raw vs median vs bilateral vs WLS.

### Analisis Percobaan 13
- Metode filter mana yang menghasilkan disparity paling halus tanpa kehilangan tepi?
- Berapa peningkatan coverage setelah WLS filtering?
- Apakah median blur atau bilateral filter sudah cukup untuk post-processing?
- Bagaimana parameter lambda dan sigma mempengaruhi hasil WLS filter?

---

## Percobaan 14: Membuat Point Cloud dari Depth Map

### Tujuan
Mengkonversi depth map menjadi point cloud 3D dan memvisualisasikannya menggunakan Matplotlib 3D scatter plot.

### Dasar Teori
Point cloud adalah kumpulan titik 3D (X, Y, Z) yang merepresentasikan permukaan objek. Konversi dari depth map ke point cloud dapat dilakukan secara manual ($X = (x - c_x) \cdot Z / f$, $Y = (y - c_y) \cdot Z / f$) atau menggunakan `cv2.reprojectImageTo3D()` dengan matriks Q. Warna titik diambil dari gambar asli untuk visualisasi realistis.

### Langkah Kerja
1. Buat pasangan stereo sintetis dengan objek pada depth yang diketahui.
2. Hitung disparity map menggunakan `cv2.StereoSGBM_create()`.
3. Konversi manual: hitung Z, lalu $X = (x - c_x) \cdot Z / f$ dan $Y = (y - c_y) \cdot Z / f$ menggunakan `np.meshgrid()`.
4. Konversi otomatis menggunakan `cv2.reprojectImageTo3D()` dengan matriks Q.
5. Filter titik invalid (disparity ≤ 0 atau Z di luar rentang).
6. Subsample titik untuk performa visualisasi.
7. Ambil warna dari gambar kiri untuk setiap titik 3D.
8. Visualisasikan point cloud dengan `matplotlib Axes3D scatter` dari berbagai sudut pandang.
9. Bandingkan point cloud manual vs `reprojectImageTo3D()`.
10. Plot posisi objek 3D dan verifikasi terhadap ground truth depth.

### Analisis Percobaan 14
- Apakah bentuk objek dapat dikenali dari point cloud 3D?
- Seberapa besar perbedaan antara metode konversi manual dan `reprojectImageTo3D()`?
- Berapa persentase titik yang berhasil direkonstruksi (valid)?
- Bagaimana densitas titik mempengaruhi kualitas visualisasi 3D?

---

## Percobaan 15: Pose Estimation dengan PnP

### Tujuan
Mengestimasi pose kamera (posisi dan orientasi) dari korespondensi titik 3D-2D menggunakan algoritma PnP (Perspective-N-Point).

### Dasar Teori
PnP mengestimasi pose kamera (rotation vector + translation vector) dari minimal 4 pasangan titik 3D objek dan proyeksi 2D-nya. `cv2.solvePnP()` menyediakan beberapa metode (ITERATIVE, P3P, EPNP, SQPNP), sedangkan `cv2.solvePnPRansac()` lebih robust terhadap outlier. `cv2.Rodrigues()` mengkonversi antara rotation vector dan rotation matrix.

### Langkah Kerja
1. Definisikan titik 3D objek (misal sudut kubus) dan matriks intrinsik kamera K.
2. Tentukan pose kamera ground truth (rvec, tvec) dan koefisien distorsi.
3. Proyeksi titik 3D ke 2D menggunakan `cv2.projectPoints()`.
4. Estimasi pose dengan `cv2.solvePnP()` menggunakan metode ITERATIVE.
5. Bandingkan dengan metode lain: P3P, EPNP, SQPNP.
6. Estimasi pose robust: `cv2.solvePnPRansac()` pada data dengan noise.
7. Konversi rvec ke rotation matrix: `cv2.Rodrigues()` dan bandingkan dengan ground truth.
8. Hitung reprojection error: proyeksi ulang titik 3D → bandingkan dengan titik 2D.
9. Visualisasikan sumbu 3D pada gambar menggunakan `cv2.drawFrameAxes()`.
10. Buat tabel perbandingan: metode, rotation error, translation error, reprojection error.

### Analisis Percobaan 15
- Metode PnP mana yang paling akurat pada data bersih?
- Seberapa besar keuntungan `solvePnPRansac()` pada data dengan noise/outlier?
- Berapa reprojection error rata-rata untuk setiap metode?
- Pada kondisi apa estimasi pose PnP gagal atau tidak stabil?

---

## Percobaan 16: Simulasi Stereo Matching Real-Time

### Tujuan
Melakukan stereo matching secara real-time pada video sintetis dan membandingkan performa FPS antara StereoBM dan StereoSGBM.

### Dasar Teori
Stereo matching real-time memerlukan keseimbangan antara akurasi dan kecepatan komputasi. StereoBM lebih cepat namun kurang akurat, StereoSGBM lebih akurat namun lebih lambat. FPS (Frames Per Second) mengukur kemampuan sistem memproses frame secara kontinu, yang kritis untuk aplikasi seperti autonomous driving dan robotika.

### Langkah Kerja
1. Definisikan parameter simulasi: ukuran frame, jumlah frame, `numDisparities`, `blockSize`.
2. Buat fungsi generator frame stereo sintetis dengan objek bergerak.
3. Inisialisasi `cv2.StereoBM_create()` dan `cv2.StereoSGBM_create()`.
4. Proses setiap frame: generate stereo pair → konversi grayscale → compute disparity.
5. Ukur waktu per frame menggunakan `time.time()` untuk BM dan SGBM.
6. Hitung FPS rata-rata: total frame / total waktu.
7. Normalisasi dan visualisasikan disparity map per frame dengan `cv2.applyColorMap()`.
8. Simpan sample frame ke output menggunakan `cv2.VideoWriter()` atau gambar.
9. Plot grafik FPS per frame untuk BM vs SGBM.
10. Buat ringkasan statistik: min/max/avg FPS, coverage rata-rata per metode.

### Analisis Percobaan 16
- Berapa FPS rata-rata BM vs SGBM pada resolusi yang diuji?
- Apakah FPS stabil sepanjang video atau berfluktuasi?
- Pada resolusi berapa SGBM tidak lagi memenuhi real-time (< 30 FPS)?
- Apakah kualitas disparity SGBM sebanding dengan penurunan FPS-nya?

---

## Percobaan 17: Depth Map Colorization dan Visualisasi

### Tujuan
Mempelajari berbagai teknik visualisasi depth map menggunakan colormap, kontur kedalaman, dan overlay semi-transparan.

### Dasar Teori
Depth map grayscale sulit diinterpretasi tanpa pewarnaan. Colormap memetakan nilai skalar ke warna untuk visualisasi intuitif. Kontur kedalaman menunjukkan batas-batas perubahan depth, sedangkan overlay depth pada gambar asli membantu korelasi spasial antara objek dan jaraknya. Segmentasi zona depth membagi scene menjadi near/mid/far.

### Langkah Kerja
1. Buat pasangan stereo sintetis dan hitung disparity/depth map dengan `cv2.StereoSGBM_create()`.
2. Normalisasi depth map ke 0–255 menggunakan `cv2.normalize()`.
3. Terapkan berbagai colormap: `cv2.applyColorMap()` dengan JET, INFERNO, VIRIDIS, MAGMA, TURBO.
4. Buat overlay semi-transparan: `cv2.addWeighted(img_color, alpha, depth_color, 1-alpha, 0)`.
5. Segmentasi zona depth: `cv2.threshold()` untuk memisahkan near, mid, far.
6. Buat kontur kedalaman menggunakan `cv2.threshold()` pada beberapa level depth.
7. Visualisasikan kontur depth pada gambar asli.
8. Buat histogram distribusi depth menggunakan `np.histogram()` dan plot dengan Matplotlib.
9. Buat grid visualisasi: semua colormap + overlay + kontur dalam satu figure.
10. Analisis colormap mana yang paling informatif untuk interpretasi depth.

### Analisis Percobaan 17
- Colormap mana yang paling efektif membedakan objek dekat dan jauh?
- Apakah overlay semi-transparan membantu memahami konteks spasial?
- Berapa level kontur yang optimal untuk visualisasi depth?
- Bagaimana histogram depth membantu mengidentifikasi jumlah objek pada depth berbeda?

---

## Percobaan 18: Pengaruh Baseline terhadap Akurasi Depth

### Tujuan
Mempelajari bagaimana jarak antar kamera (baseline) mempengaruhi akurasi dan resolusi depth estimation.

### Dasar Teori
Baseline adalah jarak horizontal antara dua kamera stereo. Baseline kecil menghasilkan disparity kecil sehingga resolusi depth rendah, sementara baseline besar meningkatkan resolusi depth tetapi memperbesar area occlusion. Hubungan $Z = f \cdot B / d$ menunjukkan bahwa error depth sensitif terhadap error disparity, terutama untuk objek jauh.

### Langkah Kerja
1. Buat scene 3D sintetis dengan objek pada depth yang diketahui (ground truth).
2. Definisikan beberapa nilai baseline: misal 10, 20, 30, 50, 80 pixel-equiv.
3. Untuk setiap baseline, buat pasangan stereo dengan menggeser gambar.
4. Hitung disparity map menggunakan `cv2.StereoSGBM_create()` untuk setiap baseline.
5. Konversi disparity ke depth: $Z = f \cdot B / d$.
6. Hitung error per objek: MAE = `np.mean(np.abs(depth_est - gt_depth))`.
7. Hitung RMSE = `np.sqrt(np.mean((depth_est - gt_depth)**2))`.
8. Pisahkan analisis: akurasi objek near vs mid vs far untuk setiap baseline.
9. Plot grafik MAE dan RMSE vs baseline.
10. Buat tabel: baseline, MAE near, MAE mid, MAE far, RMSE keseluruhan.

### Analisis Percobaan 18
- Baseline berapa yang menghasilkan MAE terendah secara keseluruhan?
- Apakah objek dekat atau jauh yang lebih terpengaruh oleh perubahan baseline?
- Pada baseline berapa mulai muncul masalah occlusion signifikan?
- Bagaimana trade-off antara akurasi depth dan coverage saat baseline diperbesar?

---

## Percobaan 19: Segmentasi Objek Berbasis Depth

### Tujuan
Menggunakan informasi depth untuk melakukan segmentasi objek berdasarkan jaraknya dari kamera.

### Dasar Teori
Depth map dapat digunakan sebagai dasar segmentasi objek dengan menerapkan thresholding pada nilai kedalaman. Connected components mengidentifikasi objek individual dalam mask biner, dan morphological operations (opening, closing) membersihkan noise pada mask. Kombinasi informasi depth dan spatial memberikan segmentasi yang lebih robust dibanding metode berbasis warna saja.

### Langkah Kerja
1. Buat pasangan stereo sintetis dengan objek pada depth berbeda dan hitung depth map.
2. Normalisasi depth map ke 0–255.
3. Terapkan threshold depth: `cv2.threshold()` untuk memisahkan foreground/background.
4. Buat mask per zona depth: `cv2.inRange()` untuk near, mid, far.
5. Bersihkan mask menggunakan `cv2.morphologyEx()` dengan operasi opening dan closing.
6. Identifikasi objek individual: `cv2.connectedComponentsWithStats()`.
7. Temukan kontur objek: `cv2.findContours()` dan gambar dengan `cv2.drawContours()`.
8. Hitung statistik per objek: area, bounding box, rata-rata depth.
9. Warnai setiap objek berdasarkan zona depth (near=merah, mid=hijau, far=biru).
10. Buat visualisasi final: gambar asli + overlay segmentasi + label depth per objek.

### Analisis Percobaan 19
- Berapa jumlah objek yang berhasil disegmentasi dengan benar?
- Apakah thresholding depth cukup untuk memisahkan objek yang berdekatan?
- Bagaimana morphological operations mempengaruhi kualitas mask segmentasi?
- Pada kondisi apa segmentasi berbasis depth gagal (objek pada depth yang sama)?

---

## Percobaan 20: Pipeline Rekonstruksi 3D Multi-View

### Tujuan
Mengimplementasikan pipeline lengkap rekonstruksi 3D dari multiple views: deteksi fitur → matching → Essential matrix → pose recovery → triangulasi → PnP → point cloud.

### Dasar Teori
Rekonstruksi 3D multi-view memerlukan minimal 2 view dengan overlap yang cukup. Pipeline dimulai dari deteksi fitur SIFT yang robust, dilanjutkan estimasi Essential matrix untuk mendapatkan hubungan geometri antar view, lalu recovery pose (R, t) dan triangulasi titik 3D. Untuk view tambahan, PnP digunakan untuk mengestimasi pose kamera baru terhadap titik 3D yang sudah direkonstruksi.

### Langkah Kerja
1. Buat scene 3D sintetis dengan titik-titik acak dan definisikan 5 pose kamera.
2. Proyeksikan titik 3D ke setiap view menggunakan matriks intrinsik K dan pose kamera.
3. Deteksi fitur SIFT: `cv2.SIFT_create()` pada setiap gambar view.
4. Match fitur antar pasangan view bersebelahan: `cv2.BFMatcher()` + ratio test.
5. Estimasi Essential matrix: `cv2.findEssentialMat()` dari pasangan pertama.
6. Recovery pose: `cv2.recoverPose()` → dapatkan R dan t relatif.
7. Triangulasi titik 3D: `cv2.triangulatePoints()` dari pasangan pertama.
8. Untuk view berikutnya, gunakan `cv2.solvePnPRansac()` untuk estimasi pose.
9. Akumulasi titik 3D dari setiap pasangan view → gabungkan point cloud.
10. Visualisasikan point cloud final dan posisi semua kamera menggunakan Matplotlib 3D scatter.

### Analisis Percobaan 20
- Berapa banyak titik 3D yang berhasil direkonstruksi dari semua view?
- Apakah posisi kamera yang diestimasi sesuai dengan ground truth?
- Bagaimana jumlah view mempengaruhi densitas dan akurasi point cloud?
- Apa langkah pipeline yang paling kritis terhadap kualitas rekonstruksi akhir?

---

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Pemahaman epipolar geometry dan signifikansinya.
2. Kemampuan SfM merekonstruksi 3D dari gambar.
3. Perbandingan BM vs SGBM untuk stereo matching.
4. Monocular vs stereo depth estimation.
5. Konversi disparity ke depth dan akurasi formula $Z = f \cdot B / d$.
6. Efektivitas post-processing (WLS, median, bilateral) pada disparity map.
7. Pembuatan point cloud 3D dari depth map dan visualisasinya.
8. Pose estimation menggunakan PnP dan perbandingan metode.
9. Trade-off akurasi vs kecepatan pada stereo matching real-time.
10. Pengaruh baseline terhadap akurasi depth estimation.
11. Segmentasi objek berbasis depth dan morfologi.
12. Pipeline rekonstruksi 3D multi-view secara end-to-end.
13. Aplikasi dan limitasi masing-masing metode.

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`).
- **Naming**: `NIM_Nama_Modul11.ipynb`
