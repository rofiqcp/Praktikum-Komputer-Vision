# PROJECT PRAKTIKUM
# MODUL 11: STRUCTURE FROM MOTION DAN DEPTH ESTIMATION

---

## FORMAT PENGUMPULAN

- **Nama file**: `NIM_Nama_Project11.zip`
- **Isi**: source code (.py), gambar input, folder output, laporan (PDF)
- **Deadline**: Sesuai jadwal yang ditentukan dosen

---

## BAGIAN A: IMPROVISASI (Pilih minimal 3 dari 15)

Kembangkan percobaan yang telah dilakukan dengan improvisasi berikut:

### 1. Multi-Baseline Stereo
Buat sistem stereo dengan 3+ baseline berbeda, bandingkan akurasi depth pada jarak dekat, menengah, dan jauh. Visualisasikan depth error vs baseline.

### 2. Disparity Fusion BM + SGBM
Gabungkan disparity dari BM dan SGBM menggunakan weighted average (bobot berdasarkan confidence). Bandingkan dengan masing-masing metode individual.

### 3. Depth-Guided Image Segmentation
Gunakan depth map untuk segmentasi multi-objek: foreground, middleground, background. Terapkan efek blur pada background (simulasi portrait mode).

### 4. Automated Stereo Parameter Tuning
Buat optimizer yang mencari parameter StereoBM/SGBM optimal menggunakan grid search atau random search. Metrik evaluasi: smoothness + sparsity.

### 5. Visual Odometry 2-Frame
Implementasikan visual odometry sederhana: essential matrix → pose → trajectory. Buat lintasan kamera dari sequence gambar (minimal 5 frame).

### 6. Depth Map Super-Resolution
Tingkatkan resolusi depth map rendah menggunakan guided filter dari gambar RGB resolusi tinggi. Bandingkan bicubic vs guided upsampling.

### 7. SfM dari 3+ View
Perluas pipeline SfM untuk 3+ gambar: incremental reconstruction. Tambahkan view baru secara bertahap (PnP + triangulasi).

### 8. Obstacle Detection dari Depth
Buat sistem deteksi halangan berdasarkan depth map: identifikasi area yang terlalu dekat (threshold). Visualisasikan warning zone.

### 9. Depth Histogram Analysis
Analisis distribusi depth pada berbagai scene (indoor, outdoor, cluttered). Buat histogram depth dan bandingkan statistik (mean, std, modality).

### 10. Stereo Confidence Map
Hitung confidence map dari disparity: piksel dengan confidence rendah ditandai invalid. Gunakan left-right consistency check.

### 11. 3D Anaglyph dari Stereo
Buat gambar anaglyph (red-cyan 3D) dari pasangan stereo. Variasikan shift untuk efek kedalaman yang berbeda.

### 12. Depth Completion
Isi hole pada depth map menggunakan inpainting (`cv2.inpaint`) atau interpolasi. Bandingkan metode: nearest neighbor vs bilinear vs inpainting.

### 13. Camera Pose Graph
Visualisasikan pose beberapa kamera dalam ruang 3D (frustum visualization). Tunjukkan baseline dan orientasi masing-masing kamera.

### 14. Real-time Depth Colorization
Buat aplikasi yang membaca gambar depth, terapkan colormap, dan tampilkan histogram depth secara bersamaan. Toggle antar colormap dengan keyboard.

### 15. Bundle Adjustment Sederhana
Implementasikan bundle adjustment sederhana: optimasi posisi 3D titik dan pose kamera menggunakan scipy.optimize.least_squares.

---

## BAGIAN B: SOAL CERITA (Kerjakan semua)

### Soal 1: Sistem Navigasi Robot
Sebuah robot warehouse menggunakan kamera stereo (baseline 10 cm, focal 500 px) untuk menghindari rak. Jika disparity rak terdekat = 80 piksel, berapa jarak rak tersebut? Implementasikan deteksi objek dekat (< 0.5 m) dari depth map sintetis.

### Soal 2: Pengecekan Kualitas Produk
Pabrik menggunakan stereo vision untuk mengukur ketinggian produk di conveyor belt. Buat program yang menghitung tinggi objek (dalam satuan relatif) dari perbedaan depth di atas dan di bawah objek. Gunakan disparity sintetis.

### Soal 3: Autonomous Parking
Sistem parkir otomatis menggunakan kamera stereo untuk mengukur jarak ke dinding. Simulasikan scene dengan 3 dinding pada jarak berbeda (2m, 4m, 6m). Hitung dan visualisasikan depth dari disparity map.

### Soal 4: Rekonstruksi Bangunan
Arsitek ingin membuat model 3D fasad bangunan dari 2 foto dengan posisi kamera berbeda. Implementasikan pipeline: feature matching → F matrix → pose → triangulasi. Visualisasikan point cloud 3D.

### Soal 5: Depth-based Background Removal
Aplikasi video conference perlu memisahkan presenter (foreground) dari background. Gunakan depth map untuk segmentasi: objek dalam 1.5m = foreground, sisanya = background. Terapkan blur pada background.

### Soal 6: Drone Obstacle Avoidance
Drone menggunakan kamera stereo untuk mendeteksi halangan. Buat program yang:
1. Menghitung depth map dari stereo
2. Membagi gambar menjadi 3 zona: kiri, tengah, kanan
3. Tentukan zona mana yang paling aman (depth rata-rata terbesar)

### Soal 7: Stereo System Comparison
Tim engineering harus memilih sistem stereo untuk proyek. Bandingkan 4 konfigurasi (BM/SGBM × 2 resolusi) dari aspek: waktu komputasi, kelengkapan disparity (% piksel valid), dan smoothness. Buat tabel perbandingan.

### Soal 8: AR Furniture Placement
Aplikasi AR menempatkan furniture virtual di ruangan. Gunakan PnP untuk menentukan pose kamera relatif terhadap marker (titik 3D lantai). Visualisasikan posisi kamera dan marker dalam 3D.

### Soal 9: Survei Topografi
Surveyor menggunakan SfM untuk membuat model terrain dari foto drone. Implementasikan triangulasi dari dua pandangan, konversi ke point cloud, dan hitung beda tinggi (range Z) daerah survei.

### Soal 10: Evaluasi Sistem Depth
Perusahaan mengevaluasi 3 metode depth estimation: BM, SGBM, dan monocular (heuristik). Buat ground truth depth sintetis, hitung metrik per metode (MAE, RMSE, % piksel valid), dan buat laporan ranking.

---

## BAGIAN C: RUBRIK PENILAIAN

| Komponen | Bobot | Kriteria |
|----------|-------|----------|
| Improvisasi (3 dari 15) | 30% | Kreativitas, implementasi benar, visualisasi lengkap |
| Soal Cerita (10) | 40% | Solusi benar, program berjalan, output sesuai |
| Laporan | 15% | Analisis mendalam, penjelasan jelas, format rapi |
| Kode | 15% | Struktur bersih, komentar lengkap, error handling |

### Skala Penilaian

| Nilai | Deskripsi |
|-------|-----------|
| A (85-100) | Semua soal benar, 3+ improvisasi kreatif, analisis mendalam |
| B (70-84) | Sebagian besar benar, 3 improvisasi standar, analisis cukup |
| C (55-69) | Setengah benar, 2 improvisasi, analisis minimal |
| D (< 55) | Banyak error, improvisasi kurang, tidak ada analisis |
