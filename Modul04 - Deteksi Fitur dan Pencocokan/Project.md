# PROJECT MODUL 4: Deteksi Fitur dan Pencocokan

## Aplikasi Pencocokan dan Pengenalan Gambar Berbasis Fitur

---

## Deskripsi Proyek

Proyek ini bertujuan membangun aplikasi komprehensif untuk pencocokan dan pengenalan gambar menggunakan berbagai algoritma deteksi fitur lokal. Aplikasi terdiri dari 4 modul terintegrasi yang mencakup benchmark detektor, robust matching, object recognition, dan image retrieval.

**Nama Proyek:** FeatureMatch Vision App
**Durasi Pengerjaan:** 2 minggu
**Mode:** Kelompok (2-3 orang) atau individu

---

## Fase Pengembangan

### FASE 1: Modul Perbandingan Detektor

**Tujuan:** Mengimplementasikan dan membandingkan berbagai detektor fitur secara sistematis.

**Yang Harus Diimplementasikan:**

1. Kelas `DetectorBenchmark` yang menerima daftar detektor sebagai parameter
2. Metode `run_detector(name, image)` yang menjalankan detektor dan mencatat waktu eksekusi
3. Metode `compare_all(image)` yang menjalankan semua detektor dan mengembalikan tabel hasil
4. Visualisasi side-by-side keypoint dari semua detektor pada gambar yang sama
5. Bar chart perbandingan: jumlah keypoint, waktu eksekusi, rata-rata response strength

**Detektor yang Dibandingkan:**
- Harris Corner (via `cv2.cornerHarris`)
- Shi-Tomasi (via `cv2.goodFeaturesToTrack`)
- SIFT (`cv2.SIFT_create`)
- ORB (`cv2.ORB_create`)
- AKAZE (`cv2.AKAZE_create`)
- FAST (`cv2.FastFeatureDetector_create`)
- BRISK (`cv2.BRISK_create`)

**Contoh Output:**
```
Detektor  | N Keypoints | Waktu (ms) | Deskriptor
----------|-------------|------------|----------
Harris    |         312 |        2.1 | -
Shi-Tomasi|         100 |        1.8 | -
SIFT      |         847 |       45.2 | 128-D float
ORB       |         500 |        4.3 | 32-B binary
AKAZE     |         634 |       18.7 | 61-B binary
FAST      |        2341 |        0.9 | -
BRISK     |         512 |        7.6 | 64-B binary
```

---

### FASE 2: Modul Robust Matcher

**Tujuan:** Mengimplementasikan pipeline robust matching yang menggabungkan FLANN, ratio test, dan RANSAC.

**Yang Harus Diimplementasikan:**

1. Kelas `RobustMatcher` dengan konfigurasi modular:
   ```python
   class RobustMatcher:
       def __init__(self, detector='SIFT', matcher='FLANN',
                    ratio=0.75, ransac_thresh=5.0):
           ...
       def detect_and_compute(self, img):
           ...
       def match(self, des1, des2):
           ...
       def verify_geometry(self, kp1, kp2, matches):
           # Returns: H, mask, inliers
           ...
       def visualize(self, img1, img2, kp1, kp2, matches, mask):
           ...
   ```

2. Support untuk kombinasi: SIFT+BF, SIFT+FLANN, ORB+BF, ORB+FLANN, AKAZE+BF, AKAZE+FLANN
3. Logging detail: jumlah keypoint tiap gambar, match sebelum/sesudah ratio test, inlier setelah RANSAC
4. Visualisasi tiga panel: gambar 1 dengan keypoint | gambar 2 dengan keypoint | hasil matching dengan warna inlier/outlier
5. Kemampuan menyimpan hasil ke file PNG secara otomatis

---

### FASE 3: Modul Object Recognition

**Tujuan:** Mendeteksi dan melokalisasi objek dalam scene gambar menggunakan feature-based matching.

**Yang Harus Diimplementasikan:**

1. Kelas `ObjectRecognizer` yang memuat template objek dari folder database
2. Metode `add_template(name, image_path)` untuk menambahkan template ke database
3. Metode `recognize(scene_image)` yang mengembalikan daftar objek terdeteksi beserta homography
4. Untuk setiap objek terdeteksi: gambar bounding polygon (transformed corners) pada scene
5. Hitung dan tampilkan confidence score berdasarkan inlier ratio
6. Mode demo interaktif: pengguna dapat memilih scene image dan melihat semua objek terdeteksi

**Format Output:**
```
Objek Terdeteksi:
  [1] buku_matematika.jpg - Score: 0.82 - Inliers: 45/52
  [2] kotak_merah.jpg     - Score: 0.67 - Inliers: 28/41
  [3] botol_air.jpg       - Score: 0.12 - Inliers: 6/49 (REJECTED)
```

---

### FASE 4: Modul Image Retrieval

**Tujuan:** Membangun sistem pencarian gambar yang dapat menerima query dan mengembalikan hasil terurut.

**Yang Harus Diimplementasikan:**

1. Kelas `ImageRetriever` dengan metode:
   - `build_index(database_folder)`: ekstrak dan simpan fitur semua gambar
   - `query(query_image, top_k=5)`: kembalikan top-k gambar paling mirip
   - `save_index(path)` / `load_index(path)`: simpan/muat index ke file (menggunakan pickle atau numpy)

2. Interface visual untuk menampilkan hasil query:
   - Baris atas: gambar query
   - Baris bawah: top-k gambar hasil beserta skor kemiripan

3. Evaluasi sistem: jika dataset memiliki label, hitung Precision@3 dan Precision@5
4. Benchmark kecepatan: ukur waktu indexing dan waktu per query

---

## Spesifikasi Teknis

1. **Bahasa:** Python 3.8+
2. **Library utama:** OpenCV 4.x (opencv-contrib-python), NumPy, Matplotlib
3. **Struktur folder:**
   ```
   feature_match_app/
     main.py              <- Entry point utama
     detector_benchmark.py
     robust_matcher.py
     object_recognizer.py
     image_retriever.py
     utils/
       visualization.py
       io_utils.py
     data/
       templates/         <- Gambar template untuk object recognition
       database/          <- Gambar untuk image retrieval database
       query/             <- Gambar query
     results/             <- Output gambar dan log
     tests/               <- Unit test (opsional)
     README.md
     requirements.txt
   ```
4. **Konfigurasi:** semua parameter (threshold, detector type) dapat diubah melalui dictionary konfigurasi di `main.py`
5. **Error handling:** setiap fungsi memiliki penanganan exception yang informatif (misalnya jika gambar tidak dapat dimuat)
6. **Dokumentasi:** setiap kelas dan fungsi utama memiliki docstring yang jelas
7. **Performance logging:** setiap operasi komputasi berat mencatat waktu eksekusinya
8. **Reproducibility:** set `np.random.seed()` dan `cv2.setRNGSeed()` untuk hasil yang konsisten
9. **Portabilitas:** tidak menggunakan path hardcoded, gunakan `os.path` atau `pathlib`
10. **Test images:** sertakan minimal 10 gambar dataset yang beragam dalam folder `data/`

---

## Ide Improvisasi (Nilai Tambah)

Implementasikan satu atau lebih improvisasi berikut untuk mendapatkan nilai ekstra:

1. **Video Real-Time:** Tambahkan mode deteksi objek real-time dari webcam menggunakan FAST + FLANN untuk memastikan frame rate cukup tinggi (>15 fps)

2. **Multi-Scale Template Matching:** Tambahkan deteksi objek yang tahan terhadap perubahan skala dengan mencoba matching pada berbagai skala gambar query (0.5x, 0.75x, 1.0x, 1.5x)

3. **Mini SLAM Demo:** Implementasikan visual odometry sederhana: track keypoint antar frame video dan estimasikan gerakan kamera (translasi 2D) menggunakan komposisi homography

4. **3D Pose Estimation:** Gunakan `cv2.solvePnP()` dan marker dengan titik 3D yang diketahui untuk menghitung pose kamera (rotasi dan translasi dalam 3D)

5. **Panorama Stitching Mini:** Implementasikan image stitching sederhana untuk 2-3 gambar yang tumpang tindih menggunakan homography chain dan alpha blending

6. **Bag of Visual Words Sederhana:** Implementasikan BoVW menggunakan k-means pada deskriptor dari seluruh database untuk representasi gambar yang lebih efisien

7. **GUI Interaktif:** Bangun antarmuka pengguna menggunakan Tkinter atau PyQt yang memungkinkan drag-and-drop gambar query dan menampilkan hasil dengan UI yang rapi

8. **Keypoint Heatmap:** Buat visualisasi densitas keypoint sebagai heatmap yang menunjukkan bagian gambar mana yang paling "kaya fitur"

---

## Format Pengumpulan

### Berkas yang Dikumpulkan
1. **Kode sumber** (folder `feature_match_app/` dikompres menjadi `.zip`)
2. **Laporan PDF** (10-15 halaman) berisi:
   - Penjelasan arsitektur sistem
   - Screenshot hasil setiap modul
   - Tabel perbandingan performa detektor
   - Analisis dan diskusi temuan
   - Kesimpulan dan saran pengembangan
3. **Video demo** (3-5 menit) menunjukkan aplikasi berjalan

### Penamaan File
```
Praktikum_KV_Modul4_[NIM]_[Nama].zip
```

### Batas Waktu
Dikumpulkan maksimal 1 minggu setelah pertemuan Modul 4.

---

## Kriteria Penilaian Proyek

| Komponen | Bobot | Indikator |
|----------|-------|-----------|
| Implementasi Fase 1 (Benchmark) | 15% | Semua detektor berjalan, tabel dan visualisasi lengkap |
| Implementasi Fase 2 (Matcher) | 20% | Kelas modular, ratio test + RANSAC bekerja benar |
| Implementasi Fase 3 (Recognition) | 20% | Deteksi objek berhasil pada minimal 3 template |
| Implementasi Fase 4 (Retrieval) | 20% | Index berhasil dibuat, top-k query akurat |
| Kualitas Kode | 10% | Bersih, terdokumentasi, modular, ada error handling |
| Laporan | 10% | Lengkap, analisis mendalam, didukung data |
| Improvisasi | 5% | Implementasi minimal 1 improvisasi dengan kualitas baik |
| **Total** | **100%** | |

### Bonus
- Implementasi 3+ improvisasi: +10 poin
- Kualitas demo video luar biasa: +5 poin
- Unit test dengan coverage >60%: +5 poin
