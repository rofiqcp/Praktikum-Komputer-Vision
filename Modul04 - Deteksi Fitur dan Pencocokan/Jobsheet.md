# JOBSHEET PRAKTIKUM
# Modul 4: Deteksi Fitur dan Pencocokan (Feature Detection and Matching)

**Mata Kuliah:** Praktikum Komputer Vision
**Semester:** Genap
**Waktu:** 4 x 50 menit (2 pertemuan)
**Prasyarat:** Modul 1-3 (Pendahuluan, Pembentukan Citra, Pemrosesan Citra)

---

## TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa diharapkan mampu:

1. Mendeteksi sudut/corner menggunakan metode Harris dan Shi-Tomasi
2. Mendeteksi dan mendeskripsikan fitur menggunakan SIFT (Scale-Invariant Feature Transform)
3. Mendeteksi fitur menggunakan ORB yang cepat dan bebas paten
4. Membandingkan karakteristik AKAZE dan FAST detector
5. Melakukan pencocokan fitur dengan Brute Force Matcher
6. Menerapkan FLANN untuk pencocokan fitur yang efisien dan cepat
7. Menghitung dan menerapkan Homography dengan algoritma RANSAC
8. Menguji invariansi fitur terhadap perubahan rotasi, skala, dan iluminasi
9. Membandingkan performa berbagai deskriptor fitur secara kuantitatif
10. Menerapkan image retrieval sederhana berbasis pencocokan fitur
11. Membangun pipeline lengkap (complete) feature matching dari awal hingga akhir
12. Memahami konsep geometric verification menggunakan RANSAC
13. Mengembangkan aplikasi praktis berbasis feature matching

---

## ALAT DAN BAHAN

### Perangkat Keras
- Komputer/laptop dengan RAM minimal 4 GB
- Kamera webcam (opsional, untuk percobaan real-time)

### Perangkat Lunak
- Python 3.8 atau lebih baru
- OpenCV 4.x (`opencv-contrib-python` untuk SIFT)
- NumPy, Matplotlib, SciPy
- IDE: VS Code / Jupyter Notebook

### Instalasi
```bash
pip install opencv-contrib-python numpy matplotlib scipy scikit-image
```

### Dataset
- Gambar objek statis (buku, kotak, papan, dll.)
- Gambar query dan database untuk percobaan retrieval
- Gambar dengan berbagai tingkat rotasi, skala, dan pencahayaan

---

## PERSIAPAN

### A. Verifikasi Instalasi
```python
import cv2
import numpy as np
print("OpenCV version:", cv2.__version__)
sift = cv2.SIFT_create()
print("SIFT tersedia:", sift is not None)
orb = cv2.ORB_create()
print("ORB tersedia:", orb is not None)
```

### B. Struktur Folder Kerja
```
praktikum/
  images/
    query.jpg        <- gambar query
    template.jpg     <- gambar template
    database/        <- folder gambar database
  hasil/             <- folder output
```

### C. Konsep Dasar yang Harus Dipahami Sebelum Praktikum
- Konsep keypoint, descriptor, dan matching
- Perbedaan corner, blob, dan edge sebagai fitur
- Pengertian invariansi (rotasi, skala, iluminasi)
- Konsep homography dan transformasi perspektif

---

## LANGKAH KERJA

---

### Percobaan 1: Harris Corner Detection

**File:** `01_harris_corner_detection.py`
**Tujuan:** Memahami deteksi sudut menggunakan analisis eigenvalue matriks Harris

**Langkah Kerja:**
1. Muat gambar grayscale dan terapkan `cv2.cornerHarris()` dengan parameter `blockSize`, `ksize`, `k`
2. Normalkan respons Harris dan terapkan threshold untuk memilih titik sudut kuat
3. Tandai lokasi corner pada gambar asli menggunakan lingkaran berwarna
4. Ubah nilai parameter `k` (0.04-0.06) dan amati perubahan jumlah corner yang terdeteksi
5. Terapkan `cv2.dilate()` untuk memperjelas visualisasi respons Harris
6. Bandingkan hasil deteksi pada gambar dengan tekstur halus vs tekstur kuat

**Pertanyaan:**
- Apa pengaruh nilai `k` terhadap sensitivitas detektor Harris?
- Bagian gambar mana yang menghasilkan nilai respons Harris tertinggi?

---

### Percobaan 2: Shi-Tomasi Corner Detection

**File:** `02_shi_tomasi_corner.py`
**Tujuan:** Menerapkan Shi-Tomasi corner detection dan refinement sub-piksel

**Langkah Kerja:**
1. Gunakan `cv2.goodFeaturesToTrack()` untuk mendeteksi N corner terkuat
2. Terapkan `cv2.cornerSubPix()` untuk mendapatkan posisi corner dengan akurasi sub-piksel
3. Visualisasikan corner sebelum dan sesudah refinement untuk membandingkan posisi
4. Ubah parameter `maxCorners`, `qualityLevel`, `minDistance` dan amati perubahan hasil
5. Bandingkan corner yang terdeteksi oleh Shi-Tomasi vs Harris pada gambar yang sama
6. Hitung dan catat waktu eksekusi kedua metode untuk perbandingan performa

**Pertanyaan:**
- Apa perbedaan kriteria seleksi corner antara Harris dan Shi-Tomasi?
- Kapan penggunaan `cornerSubPix` diperlukan dalam aplikasi nyata?

---

### Percobaan 3: SIFT Feature Detection dan Description

**File:** `03_sift_feature_detection.py`
**Tujuan:** Memahami deteksi fitur SIFT dan properti deskriptor 128 dimensi

**Langkah Kerja:**
1. Buat objek SIFT dengan `cv2.SIFT_create()` dan jalankan `detectAndCompute()`
2. Visualisasikan keypoint dengan ukuran dan orientasi menggunakan `cv2.drawKeypoints()` dengan flag `DRAW_RICH_KEYPOINTS`
3. Cetak informasi keypoint: posisi (x,y), skala (size), orientasi (angle), dan respons
4. Periksa shape array deskriptor dan konfirmasi dimensi 128
5. Terapkan SIFT pada gambar yang sama dengan berbagai tingkat blur Gaussian
6. Hitung jumlah keypoint yang konsisten terdeteksi pada berbagai kondisi blur

**Pertanyaan:**
- Mengapa dimensi deskriptor SIFT adalah 128?
- Apa hubungan antara nilai `size` keypoint dan skala pada scale-space?

---

### Percobaan 4: ORB Feature Detection

**File:** `04_orb_feature_detection.py`
**Tujuan:** Menerapkan ORB dan memahami binary descriptor rBRIEF

**Langkah Kerja:**
1. Buat objek ORB dengan `cv2.ORB_create(nfeatures=500)` dan jalankan `detectAndCompute()`
2. Visualisasikan keypoint ORB dan bandingkan distribusinya dengan SIFT
3. Periksa tipe data deskriptor ORB (uint8, bukan float32) dan ukuran 32 bytes = 256 bit
4. Ukur waktu eksekusi ORB vs SIFT pada gambar yang sama untuk perbandingan kecepatan
5. Ubah parameter ORB (`scaleFactor`, `nlevels`, `edgeThreshold`) dan amati dampaknya
6. Demonstrasikan pencocokan dua gambar menggunakan Hamming distance untuk ORB

**Pertanyaan:**
- Mengapa ORB menggunakan Hamming distance, bukan Euclidean distance?
- Apa keuntungan binary descriptor dibanding float descriptor?

---

### Percobaan 5: AKAZE dan FAST Detectors

**File:** `05_akaze_dan_fast.py`
**Tujuan:** Membandingkan AKAZE (nonlinear scale space) dan FAST (circle test)

**Langkah Kerja:**
1. Terapkan AKAZE dengan `cv2.AKAZE_create()` dan visualisasikan keypoint hasilnya
2. Terapkan FAST dengan `cv2.FastFeatureDetector_create()` dan tampilkan hasil deteksi
3. Aktifkan dan nonaktifkan Non-Maximum Suppression (NMS) pada FAST dan amati perbedaannya
4. Bandingkan jumlah keypoint, tipe deskriptor, dan waktu deteksi ketiga metode (AKAZE, FAST, ORB)
5. Uji ketahanan AKAZE pada gambar yang dirotasi 45 derajat dibanding FAST
6. Buat tabel perbandingan: AKAZE vs FAST vs ORB vs SIFT dalam hal kecepatan dan kualitas

**Pertanyaan:**
- Apa perbedaan prinsip "nonlinear scale space" AKAZE vs "Gaussian scale space" SIFT?
- Mengapa FAST sangat cepat dibanding detector berbasis skala?

---

### Percobaan 6: Brute Force Matching

**File:** `06_brute_force_matching.py`
**Tujuan:** Menerapkan pencocokan fitur Brute Force dengan ratio test

**Langkah Kerja:**
1. Deteksi fitur SIFT pada dua gambar berbeda dan lakukan matching dengan `BFMatcher`
2. Gunakan `match()` untuk mendapatkan satu match terbaik per keypoint dan visualisasikan
3. Gunakan `knnMatch(k=2)` dan terapkan Lowe ratio test (d1/d2 < 0.75) untuk filter match buruk
4. Aktifkan `crossCheck=True` pada BFMatcher dan bandingkan hasilnya dengan ratio test
5. Visualisasikan match menggunakan `cv2.drawMatches()` dan `cv2.drawMatchesKnn()`
6. Hitung persentase match yang bertahan setelah ratio test pada berbagai threshold (0.6, 0.7, 0.8)

**Pertanyaan:**
- Apa yang dimaksud dengan "ratio test" dalam konteks feature matching?
- Kapan sebaiknya menggunakan `crossCheck` vs ratio test?

---

### Percobaan 7: FLANN Matching dengan Ratio Test

**File:** `07_flann_matching_ratio_test.py`
**Tujuan:** Menggunakan FLANN untuk pencocokan cepat dengan KDTree dan LSH

**Langkah Kerja:**
1. Konfigurasi FLANN untuk deskriptor SIFT menggunakan KDTree (FLANN_INDEX_KDTREE=1)
2. Lakukan `knnMatch(k=2)` dengan FLANN dan terapkan Lowe ratio test
3. Konfigurasi FLANN untuk deskriptor ORB menggunakan LSH (FLANN_INDEX_LSH=6)
4. Bandingkan hasil dan kecepatan FLANN vs BFMatcher pada dataset yang sama
5. Ubah parameter FLANN (`trees`, `checks`) dan catat pengaruhnya terhadap kecepatan dan akurasi
6. Visualisasikan hasil matching terbaik dari kedua konfigurasi FLANN

**Pertanyaan:**
- Mengapa SIFT menggunakan KDTree sementara ORB menggunakan LSH di FLANN?
- Apa trade-off antara nilai `checks` yang tinggi vs rendah?

---

### Percobaan 8: Homography dan RANSAC

**File:** `08_homography_ransac.py`
**Tujuan:** Menghitung Homography dan menerapkan RANSAC untuk eliminasi outlier

**Langkah Kerja:**
1. Dapatkan match yang baik antara dua gambar (gunakan SIFT + FLANN + ratio test)
2. Ekstrak titik sumber dan tujuan dari match, lalu hitung Homography dengan `cv2.findHomography()`
3. Gunakan `method=cv2.RANSAC` dan threshold reproyeksi `ransacReprojThreshold=5.0`
4. Gunakan `mask` dari `findHomography` untuk memisahkan inlier dan outlier
5. Visualisasikan inlier (hijau) dan outlier (merah) pada match result
6. Terapkan `cv2.warpPerspective()` menggunakan H untuk mentransformasi gambar query

**Pertanyaan:**
- Berapa pasang titik minimum yang dibutuhkan untuk menghitung Homography?
- Apa yang terjadi jika persentase outlier sangat tinggi (>80%) pada RANSAC?

---

### Percobaan 9: Object Detection dengan Feature Matching

**File:** `09_object_detection_feature.py`
**Tujuan:** Mendeteksi objek dalam scene menggunakan template matching berbasis fitur

**Langkah Kerja:**
1. Siapkan gambar template objek dan gambar scene yang mengandung objek tersebut
2. Deteksi dan deskripsikan fitur SIFT pada keduanya, lakukan matching dengan FLANN
3. Hitung Homography dari match yang baik untuk menemukan lokasi objek dalam scene
4. Transformasikan pojok-pojok template menggunakan H untuk mendapatkan bounding polygon
5. Gambar bounding polygon pada scene untuk menunjukkan lokasi objek terdeteksi
6. Uji robustness dengan gambar scene yang mengandung objek dalam orientasi berbeda

**Pertanyaan:**
- Apa keunggulan feature-based object detection dibanding template matching pixel-based?
- Berapa minimum match yang dibutuhkan agar Homography valid dan reliable?

---

### Percobaan 10: Feature Invariansi Rotasi

**File:** `10_feature_invariance_rotasi.py`
**Tujuan:** Menguji ketahanan detektor fitur terhadap perubahan rotasi

**Langkah Kerja:**
1. Muat gambar referensi dan buat versi rotasi dengan sudut 0, 30, 60, 90, 120, 150, 180 derajat
2. Deteksi fitur menggunakan SIFT, ORB, dan AKAZE pada setiap versi rotasi
3. Lakukan matching antara gambar referensi dan setiap gambar rotasi
4. Hitung jumlah match yang baik (setelah ratio test) untuk setiap sudut dan detektor
5. Plot grafik: jumlah match vs sudut rotasi untuk SIFT, ORB, AKAZE pada grafik yang sama
6. Analisis dan simpulkan detektor mana yang paling tahan terhadap rotasi

**Pertanyaan:**
- Mengapa SIFT lebih tahan terhadap rotasi dibanding FAST?
- Pada sudut rotasi berapa performa detektor mulai menurun secara signifikan?

---

### Percobaan 11: Feature Invariansi Skala

**File:** `11_feature_invariance_skala.py`
**Tujuan:** Menguji ketahanan detektor fitur terhadap perubahan skala/ukuran

**Langkah Kerja:**
1. Buat gambar dengan berbagai skala: 0.5x, 0.75x, 1.0x, 1.5x, 2.0x dari gambar referensi
2. Deteksi fitur menggunakan SIFT, ORB, dan AKAZE pada setiap skala
3. Lakukan matching antara gambar referensi (1.0x) dan setiap skala gambar
4. Hitung jumlah match yang berhasil dan hitung match rate (%) untuk setiap skala
5. Plot grafik: match rate vs scale factor untuk ketiga detektor
6. Diskusikan mengapa metode berbasis scale-space lebih tahan terhadap perubahan skala

**Pertanyaan:**
- Apa peran "scale-space" dalam memastikan invariansi skala?
- Bagaimana pyramid level pada ORB mempengaruhi invariansi skalanya?

---

### Percobaan 12: Feature Invariansi Iluminasi

**File:** `12_feature_invariance_iluminasi.py`
**Tujuan:** Menguji ketahanan deskriptor fitur terhadap perubahan kondisi pencahayaan

**Langkah Kerja:**
1. Buat variasi gambar: brightness +50, +100, -50, contrast 0.5x, contrast 2.0x, noise Gaussian
2. Terapkan SIFT, ORB, dan AKAZE pada gambar referensi dan setiap variasinya
3. Lakukan matching dan hitung jumlah inlier setelah RANSAC untuk setiap kondisi
4. Buat bar chart yang membandingkan ketahanan ketiga detektor pada berbagai kondisi iluminasi
5. Demonstrasikan normalisasi histogram sebagai preprocessing untuk meningkatkan robustness
6. Analisis apakah normalisasi HOG dalam deskriptor SIFT membantu invariansi iluminasi

**Pertanyaan:**
- Bagaimana normalisasi dalam deskriptor SIFT membantu invariansi terhadap pencahayaan?
- Kondisi iluminasi mana yang paling merusak performa matching?

---

### Percobaan 13: Perbandingan Deskriptor

**File:** `13_deskriptor_perbandingan.py`
**Tujuan:** Membandingkan SIFT, ORB, AKAZE, dan BRISK secara komprehensif

**Langkah Kerja:**
1. Deteksi dan deskripsikan fitur menggunakan SIFT, ORB, AKAZE, dan BRISK pada gambar yang sama
2. Ukur waktu eksekusi `detectAndCompute()` untuk masing-masing deskriptor (rata-rata 10 iterasi)
3. Catat dimensi deskriptor, tipe data (float32 / uint8), dan jarak yang digunakan (L2 / Hamming)
4. Lakukan matching antara dua gambar dan hitung precision (inlier/total match) setelah RANSAC
5. Buat tabel dan visualisasi bar chart untuk kecepatan dan precision masing-masing deskriptor
6. Simpulkan rekomendasi penggunaan berdasarkan trade-off kecepatan vs akurasi

**Pertanyaan:**
- Deskriptor mana yang paling cocok untuk aplikasi mobile real-time? Mengapa?
- Apa implikasi penggunaan binary descriptor vs float descriptor terhadap kebutuhan memori?

---

### Percobaan 14: Geometric Verification Detail

**File:** `14_geometric_verification_detail.py`
**Tujuan:** Memahami RANSAC secara mendalam untuk verifikasi geometri

**Langkah Kerja:**
1. Deteksi fitur dan dapatkan semua match (termasuk outlier) antara dua gambar
2. Implementasikan visualisasi sebelum RANSAC (semua match, termasuk yang salah)
3. Jalankan `cv2.findHomography()` dengan RANSAC dan dapatkan mask inlier/outlier
4. Hitung inlier ratio = jumlah_inlier / total_match dan tampilkan statistik
5. Ubah nilai `ransacReprojThreshold` (1, 3, 5, 10) dan amati perubahan inlier ratio
6. Bandingkan hasil Homography RANSAC vs estimasi tanpa RANSAC (least squares)

**Pertanyaan:**
- Bagaimana nilai threshold reproyeksi RANSAC mempengaruhi keseimbangan precision-recall?
- Apa yang membedakan RANSAC dengan metode estimasi least squares biasa?

---

### Percobaan 15: Image Retrieval

**File:** `15_image_retrieval.py`
**Tujuan:** Membangun sistem pencarian gambar sederhana berbasis feature matching

**Langkah Kerja:**
1. Siapkan database minimal 5-10 gambar berbeda dan ekstrak fitur SIFT dari semuanya
2. Muat gambar query dan ekstrak fiturnya, lalu lakukan matching terhadap setiap gambar database
3. Hitung skor kemiripan berdasarkan jumlah good match (setelah ratio test + RANSAC)
4. Urutkan gambar database berdasarkan skor kemiripan (ranking dari tertinggi ke terendah)
5. Tampilkan top-3 gambar database yang paling mirip dengan gambar query
6. Uji dengan berbagai gambar query dan evaluasi apakah gambar relevan masuk top-3

**Pertanyaan:**
- Apa kelebihan dan kekurangan metode ranking berbasis jumlah match dibanding cosine similarity?
- Bagaimana cara meningkatkan efisiensi retrieval untuk database yang sangat besar?

---

### Percobaan 16: AR Marker Detection

**File:** `16_ar_marker_detection.py`
**Tujuan:** Mendeteksi dan melacak marker menggunakan feature matching

**Langkah Kerja:**
1. Siapkan gambar marker (dapat berupa logo atau pola kustom) sebagai referensi
2. Deteksi fitur SIFT pada marker referensi dan simpan keypoint serta deskriptornya
3. Buka kamera atau video dan untuk setiap frame deteksi fitur dan cocokkan dengan marker
4. Hitung Homography dari match yang berhasil untuk menentukan pose marker dalam frame
5. Proyeksikan bounding box 3D sederhana atau teks "AR DETECTED" di atas marker yang terdeteksi
6. Tambahkan penanganan kasus ketika marker tidak terdeteksi (jumlah match < threshold)

**Pertanyaan:**
- Apa saja tantangan utama AR marker detection berbasis fitur dibanding ArUco marker?
- Bagaimana cara membuat tracking lebih stabil antara frame yang berurutan?

---

### Percobaan 17: Keypoint Repeatability

**File:** `17_keypoint_repeatability.py`
**Tujuan:** Mengukur repeatability keypoint sebagai metrik kualitas detektor

**Langkah Kerja:**
1. Transformasikan gambar referensi dengan homography yang diketahui (rotasi + skala tertentu)
2. Deteksi keypoint pada gambar asli dan gambar transformasi menggunakan SIFT, ORB, AKAZE
3. Proyeksikan keypoint gambar asli menggunakan H yang diketahui untuk mendapat posisi prediksi
4. Hitung overlap rate: keypoint yang posisinya berdekatan (< epsilon piksel) antara prediksi dan aktual
5. Hitung repeatability rate = jumlah_repeated / min(N1, N2) untuk setiap detektor
6. Plot bar chart perbandingan repeatability score ketiga detektor

**Pertanyaan:**
- Apa hubungan antara repeatability score dan kualitas matching secara keseluruhan?
- Bagaimana nilai epsilon (overlap threshold) mempengaruhi perhitungan repeatability?

---

### Percobaan 18: Multi-Image Matching

**File:** `18_multi_image_matching.py`
**Tujuan:** Melakukan matching antara N gambar secara pairwise dan membangun similarity graph

**Langkah Kerja:**
1. Muat sekumpulan gambar (5-8 gambar) dengan konten yang saling berkaitan
2. Ekstrak fitur dari semua gambar dan simpan dalam list/dictionary
3. Lakukan matching pairwise untuk semua kombinasi gambar (N*(N-1)/2 pasang)
4. Bangun matriks similaritas NxN berdasarkan jumlah good match setiap pasang
5. Visualisasikan matriks similaritas sebagai heatmap menggunakan Matplotlib
6. Identifikasi kelompok gambar yang saling berkaitan berdasarkan ambang batas similarity

**Pertanyaan:**
- Kompleksitas waktu dari pendekatan pairwise matching adalah O(N^2). Bagaimana menguranginya?
- Apa manfaat similarity graph dalam aplikasi image stitching atau 3D reconstruction?

---

### Percobaan 19: Complete Feature Matching Pipeline

**File:** `19_feature_matching_pipeline.py`
**Tujuan:** Membangun pipeline feature matching end-to-end yang lengkap dan modular

**Langkah Kerja:**
1. Rancang dan implementasikan class `FeatureMatcher` dengan metode: `detect()`, `match()`, `verify()`
2. Parameter pipeline dapat dikonfigurasi: jenis detektor, matcher, threshold ratio, threshold RANSAC
3. Tambahkan logging yang informatif: jumlah keypoint, match awal, match setelah filter, inlier
4. Uji pipeline dengan kombinasi: SIFT+BF, SIFT+FLANN, ORB+BF, ORB+FLANN
5. Tambahkan visualisasi otomatis yang menampilkan setiap tahap pipeline
6. Ukur waktu total pipeline dan breakdown per tahap (deteksi / matching / verifikasi)

**Pertanyaan:**
- Bagaimana desain modular pipeline mempermudah eksperimen dan debugging?
- Dapatkah pipeline ini dengan mudah diperluas untuk mendukung detektor baru?

---

### Percobaan 20: Proyek Feature Matching App

**File:** `20_proyek_feature_matching_app.py`
**Tujuan:** Mengembangkan aplikasi feature matching praktis yang terintegrasi

**Langkah Kerja:**
1. Implementasikan GUI sederhana menggunakan OpenCV window dengan event handling mouse/keyboard
2. Fitur: pilih gambar query, pilih detektor (S=SIFT/O=ORB/A=AKAZE), tampilkan match real-time
3. Tambahkan mode "object detection" yang menghitung dan menampilkan homography + bounding polygon
4. Implementasikan mode "image retrieval" yang mencari top-3 gambar paling mirip dari database folder
5. Tambahkan fitur simpan hasil (screenshot dengan overlay match) ke file JPG
6. Dokumentasikan semua shortcut keyboard dan pastikan aplikasi berjalan stabil tanpa crash

**Pertanyaan:**
- Apa saja design pattern yang berguna untuk membangun aplikasi computer vision interaktif?
- Bagaimana cara mengoptimalkan aplikasi agar responsif ketika database gambar sangat besar?

---

## TUGAS AKHIR

Jawab pertanyaan berikut dalam laporan praktikum:

1. **Comparative Analysis:** Buatlah tabel perbandingan komprehensif antara SIFT, ORB, AKAZE, dan BRISK yang mencakup: dimensi deskriptor, tipe data, jarak yang digunakan, invariansi (rotasi/skala/iluminasi), kecepatan relatif, ketersediaan lisensi, dan rekomendasi penggunaan.

2. **RANSAC Analysis:** Jelaskan formula iterasi RANSAC $N = \frac{\log(1-p)}{\log(1-(1-\epsilon)^s)}$ dimana $N$ = jumlah iterasi, $p$ = probabilitas sukses (0.99), $\epsilon$ = rasio outlier, $s$ = ukuran model (4 untuk Homography). Berapa iterasi yang dibutuhkan jika outlier 50% dengan $p=0.99$?

3. **Pipeline Design:** Rancang pipeline feature matching yang optimal untuk skenario berikut: (a) aplikasi mobile real-time dengan keterbatasan CPU, (b) sistem inspeksi industri yang butuh akurasi tinggi, (c) sistem SLAM untuk robot autonomus. Jelaskan pilihan detektor, deskriptor, matcher, dan parameter yang digunakan.

4. **Invariance Testing:** Berdasarkan percobaan 10-12, fitur apa yang paling robust secara keseluruhan? Apakah ada trade-off antara kecepatan dan ketahanan (robustness)? Sertakan grafik dari hasil percobaan sebagai bukti.

5. **Application Design:** Rancang arsitektur sistem image retrieval berbasis fitur untuk katalog produk e-commerce dengan 10.000 gambar. Pertimbangkan: (a) strategi indexing untuk efisiensi query, (b) metode ranking, (c) cara menangani query gambar dari sudut pandang yang berbeda.

---

## KRITERIA PENILAIAN

| No | Komponen | Bobot | Indikator |
|----|----------|-------|-----------|
| 1 | Kehadiran dan Partisipasi | 10% | Hadir tepat waktu, aktif mengikuti praktikum |
| 2 | Kelengkapan Percobaan | 25% | Semua 20 percobaan dijalankan dan menghasilkan output |
| 3 | Output Visual | 15% | Screenshot/gambar hasil setiap percobaan tersimpan |
| 4 | Analisis dan Diskusi | 25% | Jawaban pertanyaan per percobaan tepat dan mendalam |
| 5 | Tugas Akhir | 20% | 5 pertanyaan dijawab lengkap dengan analisis kuantitatif |
| 6 | Kreativitas dan Inisiatif | 5% | Modifikasi tambahan atau eksperimen mandiri dilakukan |

**Total: 100%**

### Skala Nilai
- **A (90-100):** Semua percobaan berhasil, analisis mendalam, teori dipahami dengan baik
- **B (75-89):** Sebagian besar percobaan berhasil, analisis cukup baik
- **C (60-74):** Percobaan minimal berhasil, analisis terbatas
- **D (50-59):** Banyak percobaan gagal, analisis sangat kurang
- **E (<50):** Tidak menyelesaikan praktikum

---

## REFERENSI

1. Lowe, D.G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *IJCV*, 60(2), 91-110.
2. Harris, C., & Stephens, M. (1988). A Combined Corner and Edge Detector. *Alvey Vision Conference*.
3. Rublee, E., et al. (2011). ORB: An Efficient Alternative to SIFT or SURF. *ICCV 2011*.
4. Alcantarilla, P.F., et al. (2012). KAZE Features. *ECCV 2012*.
5. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed. Springer. Chapter 7.
6. OpenCV Documentation: https://docs.opencv.org/4.x/
7. Fischler, M.A., & Bolles, R.C. (1981). Random Sample Consensus. *Communications of the ACM*, 24(6).
