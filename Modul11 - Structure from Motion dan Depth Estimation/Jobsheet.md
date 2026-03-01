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

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Pemahaman epipolar geometry dan signifikansinya.
2. Kemampuan SfM merekonstruksi 3D dari gambar.
3. Perbandingan BM vs SGBM untuk stereo matching.
4. Monocular vs stereo depth estimation.
5. Aplikasi dan limitasi masing-masing metode.

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`).
- **Naming**: `NIM_Nama_Modul11.ipynb`
