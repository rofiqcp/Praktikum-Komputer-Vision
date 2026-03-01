# JOBSHEET MODUL 9: ESTIMASI GERAK (MOTION ESTIMATION)

---

## Tujuan Praktikum
1. Memahami dan mengimplementasikan sparse optical flow (Lucas-Kanade).
2. Memahami dan mengimplementasikan dense optical flow (Farnebäck).
3. Mengimplementasikan background subtraction (MOG2, KNN).
4. Mengimplementasikan object tracking (KCF, CSRT, dll).
5. Memahami motion history image.
6. Membangun video stabilizer dan frame interpolation sederhana.

---

## Alat dan Bahan
- **Hardware**: PC/Laptop, webcam.
- **Software**: Python 3.8+, Jupyter Notebook / VS Code.
- **Library**: OpenCV (`opencv-contrib-python`), NumPy, Matplotlib.
- **Dataset**: Video (rekam sendiri atau download): (a) kamera statis + objek bergerak, (b) kamera bergerak (handheld), (c) video dengan multiple objects.

---

## Percobaan 1: Sparse Optical Flow — Lucas-Kanade

### Tujuan
Mengimplementasikan Lucas-Kanade optical flow untuk melacak titik-titik fitur antar frame.

### Dasar Teori
Lucas-Kanade menghitung displacement titik-titik fitur dengan asumsi brightness constancy dan spatial coherence dalam window kecil. Pyramidal LK menangani gerakan besar.

### Langkah Kerja
1. Buka video atau webcam.
2. Ambil frame pertama → konversi grayscale.
3. Deteksi fitur: `goodFeaturesToTrack(maxCorners=100, qualityLevel=0.3, minDistance=7)`.
4. Loop frame: `calcOpticalFlowPyrLK()` → track fitur ke frame berikutnya.
5. Filter: hanya simpan titik dengan `status==1`.
6. Gambar trail: garis dari posisi lama ke posisi baru.
7. Visualisasikan titik (lingkaran) + trail (garis) pada frame.
8. Variasikan `maxLevel` (0, 1, 2, 3) → amati kemampuan handle gerakan besar.
9. Variasikan `winSize` (5, 15, 31) → amati akurasi vs smoothness.
10. Re-detect fitur setiap 30 frame untuk menggantikan fitur yang hilang.

### Analisis Percobaan 1
- Berapa persen fitur yang survive setelah 100 frame?
- Pada kecepatan gerakan berapa tracking mulai gagal?
- Bagaimana `maxLevel` mempengaruhi tracking gerakan cepat?
- Apakah re-detection efektif menjaga jumlah fitur?

---

## Percobaan 2: Dense Optical Flow — Farnebäck

### Tujuan
Mengimplementasikan dense optical flow untuk menghitung perpindahan seluruh piksel.

### Dasar Teori
Farnebäck menggunakan ekspansi polinomial kuadratik lokal untuk mengestimasi displacement di setiap piksel, menghasilkan dense flow field.

### Langkah Kerja
1. Buka video atau webcam.
2. Baca 2 frame berturut-turut → konversi grayscale.
3. Hitung dense flow: `calcOpticalFlowFarneback()`.
4. Konversi flow ke HSV: hue = arah, value = magnitude.
5. Konversi HSV → BGR dan tampilkan.
6. Gambar flow arrows (subsample setiap 16 piksel).
7. Overlay flow arrows di atas frame original.
8. Variasikan `pyr_scale` (0.3, 0.5, 0.8) → amati perbedaan.
9. Variasikan `winsize` (5, 15, 31) → amati detail vs smoothness.
10. Hitung statistik flow: magnitude rata-rata, arah dominan per frame.

### Analisis Percobaan 2
- Area mana yang memiliki magnitude flow terbesar?
- Bagaimana `winsize` mempengaruhi resolusi flow?
- Apakah background statis memiliki flow mendekati nol?
- Bagaimana perbandingan dense flow vs sparse flow dari Percobaan 1?

---

## Percobaan 3: Motion Detection — Background Subtraction

### Tujuan
Mendeteksi objek bergerak menggunakan background subtraction.

### Dasar Teori
Background subtraction membuat model statistik dari background statis, lalu mengklasifikasikan setiap piksel sebagai foreground (bergerak) atau background (statis).

### Langkah Kerja
1. Siapkan video dari kamera statis dengan objek bergerak.
2. Implementasikan simple frame differencing: $|I_t - I_{t-1}| > \theta$.
3. Implementasikan running average background model.
4. Gunakan `cv2.createBackgroundSubtractorMOG2()`.
5. Gunakan `cv2.createBackgroundSubtractorKNN()`.
6. Post-process: morphology (opening → closing) untuk clean mask.
7. Cari contours pada foreground mask → bounding box objek bergerak.
8. Variasikan MOG2 `history` (100, 500, 1000) dan `varThreshold`.
9. Bandingkan semua metode: frame diff vs running avg vs MOG2 vs KNN.
10. Tampilkan 4-panel: original, foreground mask, contours, bounding box.

### Analisis Percobaan 3
- Metode mana yang paling bersih (least noise) pada foreground mask?
- Bagaimana MOG2 menangani shadow? Cek `detectShadows=True`.
- Berapa waktu adaptasi background model? (berapa frame?)
- Apa terjadi jika objek berhenti bergerak?

---

## Percobaan 4: Object Tracking — Single Object

### Tujuan
Melacak objek tunggal menggunakan berbagai tracker OpenCV.

### Dasar Teori
Object tracker melacak objek yang ditandai di frame pertama (bounding box) secara otomatis di frame-frame berikutnya menggunakan appearance model.

### Langkah Kerja
1. Buka video dengan satu objek bergerak.
2. Pilih ROI objek di frame pertama: `cv2.selectROI()`.
3. Inisialisasi tracker: CSRT.
4. Loop: `tracker.update()` → gambar bounding box.
5. Tampilkan status (tracking/lost) dan FPS.
6. Ulangi dengan tracker: KCF.
7. Ulangi dengan tracker: MOSSE.
8. Ulangi dengan tracker: MIL.
9. Buat tabel perbandingan: tracker, FPS, success rate (frame tracked / total frame).
10. Uji pada video: (a) objek bergerak lambat, (b) objek bergerak cepat, (c) objek terokclusi.

### Analisis Percobaan 4
- Tracker mana yang paling akurat?
- Tracker mana yang paling cepat (FPS)?
- Bagaimana masing-masing tracker menangani oklusi?
- Pada kecepatan berapa MOSSE mulai kehilangan objek?

---

## Percobaan 5: Multi-Object Tracking

### Tujuan
Melacak beberapa objek secara simultan.

### Dasar Teori
Multi-object tracking (MOT) memerlukan inisialisasi bounding box per objek, tracking independen, dan penanganan saat objek overlap atau keluar frame.

### Langkah Kerja
1. Siapkan video dengan 3+ objek bergerak.
2. Pilih ROI untuk 3 objek: `selectROI()` berulang.
3. Inisialisasi `MultiTracker` dengan CSRT.
4. Loop: `multiTracker.update()` → gambar bounding box per objek (warna berbeda).
5. Tampilkan ID objek di setiap box.
6. Catat kapan masing-masing objek lost.
7. Hitung path (trajectory) setiap objek → gambar trail.
8. Uji saat 2 objek berpotongan.
9. Uji saat objek keluar dan masuk kembali ke frame.
10. Plot trajectory semua objek overlay di satu frame.

### Analisis Percobaan 5
- Apa terjadi saat 2 objek berpotongan (ID swap)?
- Berapa FPS untuk tracking 3 objek?
- Apakah tracker bisa recover setelah objek kembali ke frame?
- Bagaimana trajectory membantu memahami pola gerakan?

---

## Percobaan 6: Motion History Image

### Tujuan
Memvisualisasikan sejarah gerakan menggunakan Motion History Image (MHI).

### Dasar Teori
MHI merepresentasikan temporal motion: piksel yang baru bergerak memiliki nilai tinggi (terang), piksel yang sudah lama tidak bergerak memudar (gelap).

### Langkah Kerja
1. Buka video atau webcam.
2. Implementasikan MHI manual: update berdasarkan foreground mask + decay.
3. Gunakan `cv2.motempl.updateMotionHistory()`.
4. Visualisasikan MHI sebagai grayscale image.
5. Variasikan `duration` (0.5, 1.0, 2.0, 5.0 detik).
6. Hitung motion gradient: `cv2.motempl.calcMotionGradient()`.
7. Hitung global motion orientation: `cv2.motempl.calcGlobalOrientation()`.
8. Segmentasi motion: `cv2.motempl.segmentMotion()`.
9. Tampilkan: frame, foreground, MHI, motion segments.
10. Gunakan MHI untuk membedakan jenis gerakan (cepat vs lambat).

### Analisis Percobaan 6
- Bagaimana `duration` mempengaruhi visualisasi MHI?
- Apakah arah gerakan global sesuai dengan pengamatan visual?
- Berapa segmen motion yang terdeteksi?
- Bagaimana MHI bisa digunakan untuk activity recognition?

---

## Percobaan 7: Video Stabilization

### Tujuan
Mengimplementasikan video stabilizer sederhana untuk menghilangkan goncangan kamera.

### Dasar Teori
Video stabilization mengestimasi transformasi antar frame, mengakumulasi trajectory, smoothing trajectory, lalu koreksi setiap frame agar trajectory halus.

### Langkah Kerja
1. Rekam video handheld (sengaja goyang sedikit).
2. Deteksi fitur di frame pertama → track menggunakan Lucas-Kanade.
3. Estimasi transformasi antar frame: `estimateAffinePartial2D()`.
4. Dekomposisi: dx, dy, da (angle).
5. Akumulasi trajectory kumulatif.
6. Smoothing: moving average (window=30).
7. Hitung koreksi: smooth trajectory - original trajectory.
8. Warp setiap frame dengan koreksi → stabilized video.
9. Tampilkan side-by-side: original vs stabilized.
10. Plot trajectory: original (goyang) vs smoothed (halus).

### Analisis Percobaan 7
- Seberapa efektif stabilization mengurangi goncangan?
- Berapa window moving average yang optimal?
- Apakah ada artifacts (black borders) setelah stabilization?
- Bagaimana handling cropping untuk menghilangkan borders?

---

## Percobaan 8: Translational Alignment / Image Registration

### Tujuan
Mengalignment dua gambar yang bergeser (translasi) menggunakan phase correlation.

### Dasar Teori
Phase correlation mengestimasi translasi antara dua gambar menggunakan cross-power spectrum di domain frekuensi.

### Langkah Kerja
1. Load 2 gambar (sama tapi tergeser beberapa piksel).
2. Konversi ke grayscale float.
3. Hitung phase correlation: `cv2.phaseCorrelate()`.
4. Tampilkan shift (dx, dy) dan confidence.
5. Shift gambar kedua → overlay → verifikasi alignment.
6. Buat gambar geser sintetis (shift 10, 20, 50 piksel) → cek akurasi.
7. Uji pada gambar yang dirotasi sedikit → cek apakah phase correlation menangani.
8. Bandingkan: phase correlation vs feature-based alignment.
9. Uji pada video: alignment frame-to-frame.
10. Aplikasi: deteksi gerakan kamera (pan/tilt) dari frame berturut-turut.

### Analisis Percobaan 8
- Seberapa akurat estimasi translasi pada gambar sintetis?
- Apakah phase correlation menangani rotasi?
- Berapa batas shift (piksel) yang masih akurat?
- Kapan feature-based lebih baik dari phase correlation?

---

## Percobaan 9: Frame Interpolation

### Tujuan
Menghasilkan frame in-between menggunakan optical flow-based interpolation.

### Dasar Teori
Frame interpolation menggunakan optical flow untuk warp frame ke posisi temporal di antara dua frame, menghasilkan slow motion atau smoother video.

### Langkah Kerja
1. Load video pendek (10–30 frame).
2. Ambil 2 frame berturut-turut.
3. Hitung dense flow: frame1 → frame2 (forward) dan frame2 → frame1 (backward).
4. **Linear blend**: rata-rata 2 frame (baseline).
5. **Flow-based**: warp frame1 dengan $0.5 \times$ forward flow.
6. **Bidirectional**: warp kedua frame ke titik tengah → blend.
7. Tampilkan: frame1, interpolated, frame2.
8. Buat video slow-motion: sisipkan 1 frame interpolation antar setiap pasangan.
9. Buat video slow-motion: sisipkan 3 frame interpolation (4× slow).
10. Bandingkan kualitas: linear blend vs flow-based vs bidirectional.

### Analisis Percobaan 9
- Metode mana yang menghasilkan frame interpolation paling natural?
- Pada gerakan seperti apa interpolation gagal (ghosting)?
- Berapa tambahan frame yang reasonable sebelum kualitas menurun?
- Apa perbedaan visual antara linear blend dan flow-based?

---

## Percobaan 10: Activity Recognition dari Motion

### Tujuan
Menggunakan motion features untuk membedakan jenis aktivitas sederhana.

### Dasar Teori
Motion features (optical flow magnitude/direction, MHI) dapat digunakan sebagai fitur untuk mengklasifikasikan jenis aktivitas. Aktivitas berbeda menghasilkan pola motion yang berbeda.

### Langkah Kerja
1. Rekam 3 jenis aktivitas dari webcam: (a) melambaikan tangan, (b) berjalan, (c) diam.
2. Masing-masing 10 clips pendek (2–3 detik).
3. Untuk setiap clip, hitung dense optical flow.
4. Ekstrak fitur: rata-rata magnitude, arah dominan, area bergerak.
5. Atau: gunakan MHI → resize → flatten sebagai fitur.
6. Bagi data: 70% training, 30% testing.
7. Train classifier (SVM atau KNN dari scikit-learn).
8. Predict pada test set.
9. Hitung accuracy dan confusion matrix.
10. Tambahkan 1 aktivitas baru (misal: menulis) → retrain → evaluasi.

### Analisis Percobaan 10
- Berapa akurasi klasifikasi aktivitas?
- Fitur motion mana yang paling diskriminatif?
- Aktivitas mana yang paling sulit dibedakan?
- Bagaimana menambah kelas baru mempengaruhi akurasi?

---

## Percobaan 11: Motion History Image (MHI)

### Tujuan
Memvisualisasikan jejak gerakan temporal menggunakan Motion History Image (MHI) dengan implementasi manual dan colormap.

### Dasar Teori
Motion History Image (MHI) merepresentasikan gerakan secara temporal: piksel yang baru bergerak bernilai terang (bright) dan piksel yang sudah lama bergerak memudar (dark). MHI diupdate setiap frame berdasarkan mask foreground dari frame differencing, dengan mekanisme decay yang mengontrol berapa lama jejak gerakan bertahan.

### Langkah Kerja
1. Buka video `video_bola.avi` dan baca frame pertama sebagai referensi awal.
2. Inisialisasi array MHI `np.zeros((height, width), dtype=np.float32)` dan tetapkan `MHI_DURATION = 30.0`.
3. Konversi setiap frame ke grayscale, hitung perbedaan antar frame dengan `cv2.absdiff()`.
4. Terapkan `cv2.GaussianBlur()` pada diff untuk menghaluskan noise, lalu threshold dengan `cv2.threshold()` (`DIFF_THRESHOLD = 25`).
5. Update MHI secara manual: area bergerak (mask > 0) → set ke timestamp saat ini, area tidak bergerak → decay jika sudah melewati `MHI_DURATION`.
6. Normalisasi MHI ke rentang 0–255 untuk visualisasi grayscale.
7. Terapkan `cv2.applyColorMap()` (misal `COLORMAP_JET`) untuk memberikan warna pada MHI.
8. Variasikan `MHI_DURATION` (10, 30, 60) → amati perbedaan panjang jejak gerakan.
9. Tampilkan panel: frame asli, foreground mask, MHI grayscale, MHI colormap.
10. Simpan hasil visualisasi MHI ke folder output menggunakan Matplotlib.

### Analisis Percobaan 11
- Bagaimana nilai `MHI_DURATION` mempengaruhi panjang jejak gerakan yang terlihat?
- Apakah MHI dapat menunjukkan arah gerakan objek secara visual?
- Bagaimana threshold frame differencing mempengaruhi sensitivitas deteksi gerakan di MHI?
- Apa keuntungan menggunakan colormap dibandingkan grayscale untuk visualisasi MHI?

---

## Percobaan 12: Video Stabilization

### Tujuan
Mengimplementasikan pipeline stabilisasi video berbasis optical flow untuk mengurangi goncangan kamera.

### Dasar Teori
Video stabilization bekerja dengan mengestimasi transformasi geometris (translasi dan rotasi) antar frame berturut-turut, mengakumulasi transformasi menjadi trajectory, lalu menghaluskan trajectory dengan moving average. Selisih antara trajectory asli dan trajectory halus digunakan untuk koreksi setiap frame.

### Langkah Kerja
1. Buka video panning `video_panning.avi` yang direkam secara handheld (bergoyang).
2. Deteksi fitur pada setiap frame menggunakan `cv2.goodFeaturesToTrack(maxCorners=200, qualityLevel=0.01, minDistance=30)`.
3. Tracking fitur antar frame dengan `cv2.calcOpticalFlowPyrLK()` (winSize=15, maxLevel=3).
4. Estimasi transformasi affine parsial menggunakan `cv2.estimateAffinePartial2D()` → ekstrak `dx, dy, da` (translasi dan rotasi).
5. Akumulasi trajectory kumulatif: `trajectory[i] = trajectory[i-1] + transform[i]`.
6. Haluskan trajectory menggunakan moving average dengan window tertentu (misal 30 frame).
7. Hitung koreksi: `correction = smooth_trajectory - original_trajectory`.
8. Warp setiap frame dengan transformasi koreksi menggunakan `cv2.warpAffine()`.
9. Simpan video hasil stabilisasi dan tampilkan perbandingan side-by-side original vs stabilized.
10. Plot grafik trajectory asli vs trajectory halus (dx, dy, da) menggunakan Matplotlib.

### Analisis Percobaan 12
- Seberapa efektif stabilization mengurangi goncangan berdasarkan plot trajectory?
- Bagaimana ukuran window moving average mempengaruhi hasil stabilisasi?
- Apakah ada artifacts (black borders) pada video yang telah distabilkan?
- Bagaimana cara mengatasi border hitam setelah stabilisasi (cropping/padding)?

---

## Percobaan 13: Frame Interpolation Linear

### Tujuan
Menghasilkan frame-frame antara (intermediate frames) menggunakan interpolasi linear berbasis alpha blending.

### Dasar Teori
Interpolasi frame linear menghitung frame antara dengan rumus: $Frame_{interp} = (1 - \alpha) \times Frame_A + \alpha \times Frame_B$. Nilai $\alpha$ bervariasi dari 0 ke 1, di mana $\alpha = 0$ menghasilkan Frame A murni dan $\alpha = 1$ menghasilkan Frame B murni.

### Langkah Kerja
1. Baca dua frame (`frame_t0.png` dan `frame_t1.png`) menggunakan `cv2.imread()`.
2. Pastikan kedua frame memiliki ukuran yang sama; resize jika perlu.
3. Tentukan jumlah frame interpolasi (`NUM_INTERPOLATED = 9`), total 11 frame termasuk frame asli.
4. Buat array nilai alpha dari 0.0 sampai 1.0 menggunakan `np.linspace(0.0, 1.0, 11)`.
5. Untuk setiap nilai alpha, hitung frame interpolasi menggunakan `cv2.addWeighted(frame_a, 1-alpha, frame_b, alpha, 0)`.
6. Simpan semua frame hasil interpolasi ke dalam list.
7. Tampilkan grid semua frame interpolasi menggunakan Matplotlib subplot.
8. Analisis kualitas visual: amati efek ghosting/blur pada alpha sekitar 0.5.
9. Hitung metrik PSNR antara frame interpolasi dengan frame referensi (jika tersedia).
10. Simpan visualisasi dan bandingkan transisi frame-by-frame.

### Analisis Percobaan 13
- Pada nilai alpha berapa efek ghosting paling terlihat?
- Mengapa interpolasi linear menghasilkan blur pada area gerakan?
- Bagaimana jumlah frame interpolasi mempengaruhi kehalusan transisi?
- Apa keterbatasan utama interpolasi linear dibandingkan metode berbasis flow?

---

## Percobaan 14: Frame Interpolation Berbasis Optical Flow

### Tujuan
Menghasilkan frame interpolasi yang lebih natural menggunakan warping berbasis optical flow bidirectional.

### Dasar Teori
Flow-based interpolation menghitung dense optical flow (Farneback) secara bidirectional (A→B dan B→A), lalu melakukan warping kedua frame ke posisi temporal intermediate menggunakan `cv2.remap()`. Hasil warping di-blend untuk menghasilkan frame tanpa ghosting karena perpindahan piksel mengikuti alur gerakan sebenarnya.

### Langkah Kerja
1. Baca dua frame (`frame_t0.png` dan `frame_t1.png`) dan konversi ke grayscale.
2. Hitung forward flow (A→B) dengan `cv2.calcOpticalFlowFarneback(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2)`.
3. Hitung backward flow (B→A) dengan parameter yang sama.
4. Buat fungsi `warp_frame()` yang menggunakan `cv2.remap()` dengan grid koordinat dari `np.meshgrid()`.
5. Untuk setiap nilai alpha (0.0 → 1.0), skala forward flow dengan alpha dan backward flow dengan (1-alpha).
6. Warp Frame A dengan scaled forward flow, warp Frame B dengan scaled backward flow.
7. Blend kedua hasil warping: `cv2.addWeighted(warped_a, 1-alpha, warped_b, alpha, 0)`.
8. Bandingkan hasil interpolasi flow-based vs linear blending dari Percobaan 13.
9. Tampilkan grid perbandingan: linear blend vs flow-based untuk setiap alpha.
10. Simpan visualisasi perbandingan dan analisis perbedaan kualitas.

### Analisis Percobaan 14
- Pada area mana interpolasi flow-based jauh lebih baik dari linear?
- Apakah ada artifacts pada area dengan occlusion (satu objek menutupi yang lain)?
- Bagaimana parameter Farneback (winsize, levels) mempengaruhi kualitas interpolasi?
- Pada gerakan seperti apa metode flow-based masih menghasilkan ghosting?

---

## Percobaan 15: Magnitude dan Arah Optical Flow

### Tujuan
Menganalisis komponen magnitude (kecepatan) dan direction (arah) dari vektor dense optical flow secara detail.

### Dasar Teori
Vektor optical flow $(dx, dy)$ dapat dikonversi ke koordinat polar: magnitude $= \sqrt{dx^2 + dy^2}$ menunjukkan kecepatan gerakan, dan direction $= \arctan2(dy, dx)$ menunjukkan arah gerakan. Analisis kedua komponen ini memungkinkan pemisahan area bergerak vs statis serta identifikasi pola gerakan dominan.

### Langkah Kerja
1. Buka `video_multi_objek.avi`, baca dua frame dengan jarak 10 frame untuk gerakan yang jelas.
2. Konversi kedua frame ke grayscale, hitung dense optical flow dengan `cv2.calcOpticalFlowFarneback()`.
3. Pisahkan komponen horizontal (`flow_x`) dan vertikal (`flow_y`).
4. Konversi ke koordinat polar menggunakan `cv2.cartToPolar()` → magnitude dan angle.
5. Visualisasikan magnitude sebagai heatmap menggunakan `cv2.applyColorMap()`.
6. Threshold magnitude untuk memisahkan area bergerak vs statis.
7. Buat histogram arah (rose diagram) menggunakan `np.histogram()` dan `matplotlib polar plot`.
8. Visualisasikan arah flow menggunakan HSV encoding (hue=arah, value=magnitude).
9. Overlay flow arrows pada frame original (subsample setiap 16 piksel).
10. Simpan semua visualisasi: magnitude map, direction map, polar histogram, flow arrows.

### Analisis Percobaan 15
- Area mana yang memiliki magnitude flow terbesar dan apa interpretasinya?
- Berapa arah dominan gerakan berdasarkan rose diagram (polar histogram)?
- Apakah threshold magnitude efektif memisahkan area bergerak dan statis?
- Bagaimana distribusi arah flow berubah untuk video dengan gerakan berbeda?

---

## Percobaan 16: Feature Trajectory Tracking

### Tujuan
Melacak fitur di banyak frame dan menggambar trajectory lengkap setiap fitur dengan pewarnaan berdasarkan waktu.

### Dasar Teori
Feature trajectory tracking mendeteksi fitur corner di frame pertama kemudian melacaknya secara kontinu di setiap frame berikutnya menggunakan Lucas-Kanade optical flow. Posisi fitur disimpan dari awal hingga akhir sehingga membentuk trajectory (jejak perjalanan) yang dapat divisualisasikan dengan warna gradien (biru→merah) untuk menunjukkan progres temporal.

### Langkah Kerja
1. Buka `video_bola.avi` dan baca frame pertama, konversi ke grayscale.
2. Deteksi fitur menggunakan `cv2.goodFeaturesToTrack(maxCorners=50, qualityLevel=0.3, minDistance=10, blockSize=7)`.
3. Inisialisasi list trajectory untuk setiap fitur: `trajectories[i] = [(x, y), ...]`.
4. Loop frame: tracking fitur menggunakan `cv2.calcOpticalFlowPyrLK()`.
5. Filter fitur berdasarkan `status == 1`, simpan posisi baru ke trajectory.
6. Tandai fitur yang hilang (status == 0) dan hentikan tracking-nya.
7. Gambar polyline trajectory untuk setiap fitur menggunakan `cv2.polylines()` dengan warna dari matplotlib colormap.
8. Gambar lingkaran pada posisi terkini setiap fitur yang masih aktif.
9. Buat visualisasi overlay seluruh trajectory di atas frame terakhir.
10. Plot statistik: jumlah fitur aktif per frame dan panjang rata-rata trajectory.

### Analisis Percobaan 16
- Berapa persen fitur yang bertahan dari frame pertama hingga frame terakhir?
- Fitur pada area mana yang cenderung hilang lebih cepat?
- Apakah trajectory menunjukkan pola gerakan yang konsisten dengan gerakan objek?
- Bagaimana parameter `qualityLevel` dan `minDistance` mempengaruhi jumlah dan distribusi fitur?

---

## Percobaan 17: Perbandingan Background Subtraction

### Tujuan
Membandingkan empat metode background subtraction (Frame Differencing, Running Average, MOG2, KNN) dari segi kualitas mask dan kecepatan pemrosesan.

### Dasar Teori
Setiap metode background subtraction memiliki pendekatan berbeda: Frame Differencing membandingkan frame berturut-turut secara langsung, Running Average memelihara model background adaptif dengan `cv2.accumulateWeighted()`, MOG2 memodelkan setiap piksel sebagai campuran distribusi Gaussian, dan KNN mengklasifikasi piksel menggunakan K-Nearest Neighbors. Trade-off antara kualitas dan kecepatan menjadi pertimbangan utama dalam pemilihan metode.

### Langkah Kerja
1. Buka `video_multi_objek.avi` untuk digunakan sebagai input semua metode.
2. Inisialisasi Frame Differencing dengan `DIFF_THRESHOLD = 30` dan hitung `cv2.absdiff()` antar frame.
3. Inisialisasi Running Average dengan `cv2.accumulateWeighted(alpha=0.05)` untuk adaptasi background.
4. Inisialisasi MOG2: `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)`.
5. Inisialisasi KNN: `cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)`.
6. Proses setiap frame dengan keempat metode, ukur waktu per frame menggunakan `time.perf_counter()`.
7. Hitung jumlah piksel foreground yang terdeteksi oleh setiap metode.
8. Terapkan morphological operations (opening + closing) pada mask setiap metode.
9. Tampilkan grid perbandingan: 4 kolom (satu per metode) × beberapa frame snapshot.
10. Buat grafik perbandingan: jumlah piksel foreground dan waktu pemrosesan per metode.

### Analisis Percobaan 17
- Metode mana yang menghasilkan foreground mask paling bersih (sedikit noise)?
- Metode mana yang paling cepat dan berapa FPS masing-masing?
- Bagaimana MOG2 dan KNN menangani bayangan dibandingkan metode sederhana?
- Pada skenario apa Running Average lebih baik dari Frame Differencing?

---

## Percobaan 18: Deteksi Gerakan Berbasis Contour

### Tujuan
Mendeteksi objek bergerak menggunakan kombinasi background subtraction dan analisis kontur untuk menghasilkan bounding box dan label.

### Dasar Teori
Pipeline deteksi gerakan berbasis kontur dimulai dari background subtraction untuk mendapatkan binary foreground mask, dilanjutkan morphological operations untuk membersihkan noise, lalu ekstraksi kontur dengan `cv2.findContours()`. Kontur difilter berdasarkan area minimum untuk membuang noise kecil, dan bounding box digambar pada setiap objek terdeteksi.

### Langkah Kerja
1. Buka `video_multi_objek.avi` dan inisialisasi `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)`.
2. Definisikan `MIN_CONTOUR_AREA = 500` piksel² sebagai filter noise.
3. Buat kernel morfologi: `cv2.getStructuringElement(MORPH_ELLIPSE, (5,5))` dan `(15,15)` untuk closing.
4. Proses setiap frame: apply background subtractor → dapatkan foreground mask.
5. Terapkan `cv2.morphologyEx()` (opening untuk hapus noise kecil, closing untuk menutup celah).
6. Temukan kontur pada mask: `cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`.
7. Filter kontur: hanya simpan kontur dengan `cv2.contourArea() >= MIN_CONTOUR_AREA`.
8. Gambar bounding box `cv2.boundingRect()` dan label `cv2.putText()` pada setiap objek terdeteksi.
9. Catat jumlah objek terdeteksi per frame, simpan snapshot pada frame target (15, 30, 60, 90).
10. Tampilkan panel: frame asli, foreground mask, frame dengan bounding box, dan grafik jumlah objek per frame.

### Analisis Percobaan 18
- Berapa rata-rata jumlah objek yang berhasil terdeteksi per frame?
- Bagaimana `MIN_CONTOUR_AREA` mempengaruhi jumlah false positive?
- Apakah morphological operations efektif membersihkan noise pada foreground mask?
- Apa terjadi saat dua objek bergerak berdekatan (merged contour)?

---

## Percobaan 19: Simulasi Optical Flow Real-Time

### Tujuan
Mensimulasikan pemrosesan optical flow secara real-time dan membandingkan kecepatan serta kualitas antara sparse (Lucas-Kanade) vs dense (Farneback).

### Dasar Teori
Dalam aplikasi real-time, kecepatan pemrosesan (FPS) menjadi faktor kritis. Sparse optical flow (Lucas-Kanade) hanya menghitung flow pada titik fitur sehingga sangat cepat, sedangkan dense optical flow (Farneback) menghitung flow di setiap piksel sehingga lebih lambat namun memberikan informasi gerakan yang lengkap. Pengukuran timing menggunakan `time.perf_counter()` memberikan akurasi tinggi.

### Langkah Kerja
1. Buka `video_bola.avi` dan baca properti video (resolusi, FPS, total frame).
2. Inisialisasi parameter sparse: `cv2.goodFeaturesToTrack(maxCorners=100, qualityLevel=0.3, minDistance=7)`.
3. Inisialisasi parameter LK: `winSize=(15,15), maxLevel=2`.
4. **Fase 1 — Sparse LK**: proses setiap frame, hitung waktu per frame dengan `time.perf_counter()`, track fitur dengan `cv2.calcOpticalFlowPyrLK()`.
5. Catat FPS sparse dan magnitude flow rata-rata per frame.
6. Reset video, baca ulang frame pertama untuk Fase 2.
7. **Fase 2 — Dense Farneback**: proses setiap frame, hitung dense flow dengan `cv2.calcOpticalFlowFarneback()`, catat waktu per frame.
8. Catat FPS dense dan magnitude flow rata-rata per frame.
9. Buat visualisasi perbandingan: snapshot sparse vs dense pada frame yang sama.
10. Plot grafik: FPS sparse vs dense per frame, dan magnitude flow per frame untuk kedua metode.

### Analisis Percobaan 19
- Berapa rata-rata FPS sparse LK vs dense Farneback pada video ini?
- Berapa kali lipat sparse LK lebih cepat dari dense Farneback?
- Apakah magnitude flow dari kedua metode menunjukkan pola yang serupa?
- Pada resolusi dan jumlah fitur berapa, sparse LK masih memenuhi persyaratan real-time (≥ 30 FPS)?

---

## Percobaan 20: Estimasi Kecepatan Objek

### Tujuan
Mengestimasi kecepatan objek bergerak menggunakan informasi magnitude dan arah dari dense optical flow.

### Dasar Teori
Kecepatan objek dapat diestimasi dari optical flow dengan menghitung rata-rata magnitude vektor flow pada area objek. Magnitude flow dalam satuan piksel/frame dikonversi ke satuan dunia nyata (m/s) menggunakan faktor skala spasial (piksel ke meter) dan temporal (FPS). Pipeline melibatkan thresholding magnitude, deteksi kontur objek, dan perhitungan centroid menggunakan `cv2.moments()`.

### Langkah Kerja
1. Buka `video_multi_objek.avi` dan tetapkan asumsi skala: `ASSUMED_REAL_WIDTH_M = 5.0`, hitung `PIXEL_TO_METER = 5.0 / width`.
2. Baca frame pertama, konversi ke grayscale sebagai referensi.
3. Untuk setiap pasangan frame berturut-turut, hitung dense optical flow `cv2.calcOpticalFlowFarneback()`.
4. Konversi flow ke magnitude dan arah menggunakan `cv2.cartToPolar()`.
5. Threshold magnitude (≥ `MAG_THRESHOLD = 1.5` piksel/frame) untuk membuat binary mask area bergerak.
6. Terapkan morphological operations dengan kernel ellipse (7×7) untuk membersihkan noise.
7. Temukan kontur objek bergerak dengan `cv2.findContours()`, filter area minimum `MIN_CONTOUR_AREA = 300`.
8. Hitung centroid setiap objek (`cv2.moments()`), gambar vektor kecepatan dengan `cv2.arrowedLine()`.
9. Konversi kecepatan ke m/s: `speed_ms = mean_magnitude * PIXEL_TO_METER * fps`.
10. Tampilkan panel: frame dengan vektor kecepatan, magnitude map, dan grafik kecepatan per frame.

### Analisis Percobaan 20
- Berapa kecepatan rata-rata objek dalam satuan piksel/frame dan m/s?
- Apakah estimasi kecepatan konsisten antar frame untuk objek yang sama?
- Bagaimana asumsi skala (PIXEL_TO_METER) mempengaruhi akurasi kecepatan dalam m/s?
- Pada kecepatan berapa estimasi mulai tidak akurat karena keterbatasan optical flow?

---

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Perbandingan sparse vs dense optical flow (Percobaan 1, 2, 15, 19).
2. Perbandingan metode background subtraction: frame differencing, running average, MOG2, KNN (Percobaan 3, 17).
3. Perbandingan tracker: akurasi vs kecepatan pada single dan multi-object tracking (Percobaan 4, 5).
4. Representasi gerakan temporal menggunakan Motion History Image (Percobaan 6, 11).
5. Efektivitas video stabilization berbasis optical flow (Percobaan 7, 12).
6. Perbandingan frame interpolation: linear blending vs flow-based (Percobaan 9, 13, 14).
7. Analisis magnitude dan arah optical flow untuk memahami pola gerakan (Percobaan 15).
8. Feature trajectory tracking dan pola perjalanan fitur (Percobaan 8, 16).
9. Deteksi gerakan berbasis kontur dan estimasi kecepatan objek (Percobaan 18, 20).
10. Potensi motion features untuk activity recognition dan aplikasi real-time (Percobaan 10, 19).

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`) + video data.
- **Naming**: `NIM_Nama_Modul09.ipynb`
