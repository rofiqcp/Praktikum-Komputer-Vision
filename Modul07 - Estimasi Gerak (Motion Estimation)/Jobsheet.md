# JOBSHEET MODUL 7: ESTIMASI GERAK (MOTION ESTIMATION)

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

## Percobaan 3: Visualisasi Optical Flow

### Tujuan
Memvisualisasikan optical flow dalam berbagai representasi: HSV color-coded, quiver plot, dan overlay arrows.

### Dasar Teori
Visualisasi optical flow penting untuk memahami pola gerakan. HSV coding menggunakan hue untuk arah dan value untuk magnitude. Quiver plot menampilkan vektor arah di grid reguler.

### Langkah Kerja
1. Buka video dan baca 2 frame berturut-turut.
2. Hitung dense optical flow dengan Farneback.
3. Konversi flow ke representasi HSV (hue=arah, value=magnitude).
4. Buat quiver plot dari flow (subsample setiap 16 piksel).
5. Overlay flow arrows pada frame original.
6. Hitung histogram arah flow untuk analisis distribusi gerakan.
7. Buat magnitude map dengan threshold untuk area bergerak.
8. Tampilkan panel: frame, HSV flow, quiver, magnitude map.
9. Simpan semua visualisasi ke folder output.
10. Bandingkan visualisasi pada video gerakan cepat vs lambat.

### Analisis Percobaan 3
- Visualisasi mana yang paling informatif untuk analisis gerakan?
- Bagaimana arah gerakan ditampilkan dalam HSV coding?
- Area mana yang memiliki magnitude flow terbesar?

---

## Percobaan 4: Background Subtraction MOG2

### Tujuan
Mendeteksi objek bergerak menggunakan Mixture of Gaussians (MOG2) background subtraction.

### Dasar Teori
MOG2 memodelkan setiap piksel sebagai campuran distribusi Gaussian. Piksel yang tidak cocok dengan model background diklasifikasikan sebagai foreground (objek bergerak).

### Langkah Kerja
1. Buka video dengan objek bergerak pada background statis.
2. Buat background subtractor: `cv2.createBackgroundSubtractorMOG2()`.
3. Terapkan pada setiap frame → dapatkan foreground mask.
4. Variasikan parameter `history` (100, 300, 500).
5. Variasikan `varThreshold` (16, 25, 50).
6. Aktifkan dan nonaktifkan `detectShadows`.
7. Post-process mask dengan morphological operations (opening, closing).
8. Cari contours pada mask → gambar bounding box.
9. Tampilkan: frame asli, foreground mask, mask cleaned, bounding boxes.
10. Simpan hasil ke folder output.

### Analisis Percobaan 4
- Bagaimana `history` mempengaruhi kecepatan adaptasi background?
- Apa efek `detectShadows` pada kualitas mask?
- Parameter mana yang paling berpengaruh pada akurasi deteksi?

---

## Percobaan 5: Background Subtraction KNN

### Tujuan
Mendeteksi objek bergerak menggunakan K-Nearest Neighbors (KNN) background subtractor.

### Dasar Teori
KNN background subtractor menggunakan K tetangga terdekat untuk mengklasifikasikan piksel. Lebih efisien dari MOG2 dan menghasilkan mask yang lebih bersih.

### Langkah Kerja
1. Buka video yang sama dengan percobaan 4.
2. Buat KNN subtractor: `cv2.createBackgroundSubtractorKNN()`.
3. Terapkan pada setiap frame → foreground mask.
4. Variasikan `history` dan `dist2Threshold`.
5. Bandingkan hasil KNN vs MOG2 secara side-by-side.
6. Post-process mask dengan morphological operations.
7. Cari contours → gambar bounding box.
8. Hitung metrik: jumlah piksel foreground per frame.
9. Tampilkan panel perbandingan MOG2 vs KNN.
10. Simpan hasil perbandingan ke folder output.

### Analisis Percobaan 5
- Apa perbedaan visual antara MOG2 dan KNN?
- Metode mana yang lebih bersih (less noise)?
- Bagaimana kecepatan pemrosesan MOG2 vs KNN?

---

## Percobaan 6: Frame Differencing

### Tujuan
Mendeteksi gerakan menggunakan teknik frame differencing sederhana.

### Dasar Teori
Frame differencing menghitung perbedaan absolut antara frame berturut-turut. Piksel dengan perbedaan di atas threshold dianggap bergerak.

### Langkah Kerja
1. Buka video dan baca frame berturut-turut.
2. Hitung absolute difference: `cv2.absdiff(frame_t, frame_t-1)`.
3. Konversi ke grayscale dan terapkan threshold.
4. Variasikan nilai threshold (10, 25, 50, 75).
5. Implementasikan two-frame dan three-frame differencing.
6. Post-process dengan Gaussian blur sebelum thresholding.
7. Terapkan morphological operations pada mask.
8. Cari contours → bounding box objek bergerak.
9. Tampilkan: frame, difference, threshold mask, deteksi.
10. Simpan hasil ke folder output.

### Analisis Percobaan 6
- Bagaimana threshold mempengaruhi sensitivitas deteksi?
- Apa kelebihan three-frame differencing vs two-frame?
- Mengapa frame differencing gagal pada gerakan lambat?

---

## Percobaan 7: Running Average Background

### Tujuan
Membuat model background menggunakan running average (akumulasi rata-rata) dari frame video.

### Dasar Teori
Running average memperbarui model background secara bertahap menggunakan `cv2.accumulateWeighted()` dengan learning rate alpha. Background yang stabil terbentuk seiring waktu.

### Langkah Kerja
1. Buka video dengan background statis.
2. Inisialisasi model background dari frame pertama.
3. Update background: `cv2.accumulateWeighted(frame, bg, alpha)`.
4. Hitung foreground: `cv2.absdiff(frame, background)` → threshold.
5. Variasikan alpha (0.01, 0.05, 0.1, 0.3).
6. Amati kecepatan adaptasi background tiap nilai alpha.
7. Terapkan morphological operations pada mask.
8. Cari contours → gambar bounding box.
9. Tampilkan: frame, model background, foreground, deteksi.
10. Simpan hasil ke folder output.

### Analisis Percobaan 7
- Bagaimana alpha mempengaruhi kecepatan adaptasi?
- Apa terjadi jika objek berhenti bergerak?
- Bandingkan running average vs MOG2 vs KNN.

---

## Percobaan 8: Object Tracking CSRT

### Tujuan
Melacak objek tunggal menggunakan CSRT (Channel and Spatial Reliability Tracking).

### Dasar Teori
CSRT menggunakan channel reliability dan spatial reliability untuk tracking yang lebih akurat dibandingkan tracker lain. Cocok untuk kondisi dengan perubahan bentuk dan skala.

### Langkah Kerja
1. Buka video dengan objek bergerak.
2. Pilih ROI objek di frame pertama: `cv2.selectROI()`.
3. Inisialisasi CSRT tracker: `cv2.TrackerCSRT_create()`.
4. Loop: `tracker.update()` → gambar bounding box.
5. Tampilkan status tracking (success/lost) dan FPS.
6. Hitung trajectory (center point tiap frame).
7. Gambar trail trajectory pada frame.
8. Uji pada objek yang berubah skala.
9. Uji pada objek yang berubah orientasi.
10. Simpan video hasil tracking dan trajectory ke folder output.

### Analisis Percobaan 8
- Berapa FPS rata-rata CSRT tracker?
- Apakah CSRT menangani perubahan skala?
- Pada kondisi apa CSRT kehilangan tracking?

---

## Percobaan 9: Object Tracking KCF

### Tujuan
Melacak objek tunggal menggunakan KCF (Kernelized Correlation Filter) tracker.

### Dasar Teori
KCF menggunakan circulant matrix dan kernel trick untuk efisiensi komputasi. Lebih cepat dari CSRT tetapi kurang robust terhadap perubahan skala.

### Langkah Kerja
1. Buka video yang sama dengan percobaan 8.
2. Pilih ROI objek: `cv2.selectROI()`.
3. Inisialisasi KCF tracker: `cv2.TrackerKCF_create()`.
4. Loop: update dan gambar bounding box.
5. Hitung dan tampilkan FPS.
6. Bandingkan langsung dengan CSRT pada video yang sama.
7. Buat tabel perbandingan: akurasi, FPS, robustness.
8. Uji pada objek bergerak cepat.
9. Uji pada video dengan perubahan iluminasi.
10. Simpan hasil perbandingan ke folder output.

### Analisis Percobaan 9
- Berapa perbedaan FPS KCF vs CSRT?
- Mana yang lebih akurat secara visual?
- Pada kondisi apa KCF lebih unggul?

---

## Percobaan 10: Multi-Object Tracking

### Tujuan
Melacak beberapa objek secara simultan menggunakan MultiTracker.

### Dasar Teori
Multi-object tracking memerlukan inisialisasi bounding box per objek, tracking independen, dan penanganan saat objek overlap.

### Langkah Kerja
1. Siapkan video dengan 3+ objek bergerak.
2. Pilih ROI untuk setiap objek menggunakan `selectROI()`.
3. Inisialisasi MultiTracker dengan CSRT tracker per objek.
4. Loop: `multiTracker.update()` → gambar warna berbeda per objek.
5. Tampilkan ID objek di setiap bounding box.
6. Catat kapan tiap objek lost.
7. Hitung trajectory setiap objek → gambar trail.
8. Uji saat 2 objek berpotongan.
9. Plot semua trajectory overlay di satu frame.
10. Simpan visualisasi multi-tracking ke folder output.

### Analisis Percobaan 10
- Apa terjadi saat 2 objek berpotongan (ID swap)?
- Berapa FPS untuk tracking 3+ objek?
- Bagaimana trajectory membantu memahami pola gerakan?

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
- **Naming**: `NIM_Nama_Modul07.ipynb`
