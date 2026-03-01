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

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Perbandingan sparse vs dense optical flow.
2. Perbandingan metode background subtraction.
3. Perbandingan tracker: akurasi vs kecepatan.
4. Efektivitas video stabilization.
5. Potensi motion features untuk activity recognition.

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`) + video data.
- **Naming**: `NIM_Nama_Modul09.ipynb`
