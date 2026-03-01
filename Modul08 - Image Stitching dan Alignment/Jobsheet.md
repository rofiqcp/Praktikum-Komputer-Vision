# JOBSHEET MODUL 8: IMAGE STITCHING DAN ALIGNMENT

---

## Tujuan Praktikum
1. Memahami dan mengimplementasikan pipeline image stitching dari awal.
2. Menggunakan OpenCV Stitcher API untuk panorama otomatis.
3. Membandingkan teknik blending (feather, multi-band, Poisson).
4. Mengimplementasikan cylindrical dan spherical projection.
5. Memahami exposure compensation dan seam finding.
6. Membangun panorama multi-image.

---

## Alat dan Bahan
- **Hardware**: PC/Laptop, webcam, smartphone (untuk mengambil foto panorama).
- **Software**: Python 3.8+, Jupyter Notebook / VS Code.
- **Library**: OpenCV (`opencv-contrib-python`), NumPy, Matplotlib.
- **Dataset**: Minimal 3 set foto panorama (masing-masing 3–5 foto overlapping). Ambil sendiri menggunakan smartphone/kamera.

### Persiapan Dataset
```text
Ambil foto dengan overlap 30–50% menggunakan keterangan berikut:
Set 1: Scene outdoor (taman, gedung, jalan) — 3 foto horizontal.
Set 2: Scene indoor (ruangan, lab) — 4 foto horizontal.
Set 3: Scene lebar (panorama alam, stadion) — 5+ foto horizontal.
```

---

## Percobaan 1: Image Stitching Sederhana (Manual Pipeline)

### Tujuan
Membangun pipeline stitching dari awal: detect → match → homography → warp → combine.

### Dasar Teori
Dua gambar yang overlapping dapat digabungkan dengan mengestimasi homography dari fitur yang cocok, lalu mewarp satu gambar ke perspektif gambar lainnya.

### Langkah Kerja
1. Load 2 gambar overlapping (Set 1: foto 1 dan 2).
2. Deteksi SIFT fitur pada kedua gambar.
3. Match menggunakan FLANN + Lowe's ratio test.
4. Estimasi homography dengan RANSAC.
5. Tentukan ukuran canvas output (cukup besar untuk menampung kedua gambar).
6. Warp gambar pertama ke canvas dengan `warpPerspective`.
7. Tempatkan gambar kedua pada canvas.
8. Gabungkan — tampilkan raw stitching (tanpa blending).
9. Identifikasi area overlap (mask intersection).
10. Tampilkan: gambar 1, gambar 2, matches, hasil stitching.

### Analisis Percobaan 1
- Berapa good matches dan berapa inlier setelah RANSAC?
- Apakah ada seam yang terlihat di area overlap?
- Apa yang terjadi jika overlap kurang dari 20%?
- Bagaimana kualitas stitching di area tepi vs tengah?

---

## Percobaan 2: OpenCV Stitcher API

### Tujuan
Menggunakan `cv2.Stitcher` untuk membuat panorama otomatis dengan pipeline lengkap.

### Dasar Teori
OpenCV Stitcher mengimplementasikan pipeline lengkap: feature detection → matching → homography → bundle adjustment → warp → exposure compensation → seam finding → blending.

### Langkah Kerja
1. Load 3–5 gambar overlapping (Set 1 atau Set 2).
2. Buat Stitcher: `cv2.Stitcher_create(cv2.Stitcher_PANORAMA)`.
3. Jalankan `stitcher.stitch(images)` dan cek status.
4. Tampilkan panorama hasil.
5. Crop black borders dari hasil.
6. Ulangi dengan mode `SCANS` (untuk flat documents).
7. Uji dengan gambar yang diacak urutannya → Stitcher otomatis menyusun.
8. Uji dengan gambar yang TIDAK overlapping → cek error handling.
9. Ukur waktu stitching untuk 3, 4, 5 gambar.
10. Bandingkan hasil Stitcher API vs manual pipeline (Percobaan 1).

### Analisis Percobaan 2
- Apakah Stitcher menghasilkan panorama yang lebih baik dari manual?
- Apa perbedaan mode PANORAMA vs SCANS?
- Bagaimana Stitcher menangani gambar yang tidak overlapping?
- Berapa waktu stitching per gambar tambahan?

---

## Percobaan 3: Teknik Blending — Perbandingan

### Tujuan
Membandingkan teknik blending: no blending, feather, multi-band (Laplacian pyramid).

### Dasar Teori
Blending menghilangkan seam di area overlap. Feather blending menggunakan linear alpha, sementara multi-band blending menggunakan Laplacian pyramid untuk memadukan frekuensi berbeda secara terpisah.

### Langkah Kerja
1. Siapkan 2 gambar yang sudah di-warp dan aligned (dari Percobaan 1).
2. Buat mask untuk area overlap.
3. **No blending**: Langsung overlay (gambar 2 menutupi gambar 1 di overlap).
4. **Feather blending**: Buat alpha gradient di area overlap, blend linear.
5. **Multi-band blending**: Buat Laplacian pyramid (3 level) → blend per level.
6. **50-50 blend**: Rata-rata sederhana di area overlap.
7. Tampilkan 4 hasil side-by-side.
8. Zoom ke area seam → bandingkan kualitas detail.
9. Variasikan jumlah level pyramid (2, 3, 4, 5) pada multi-band.
10. Buat tabel perbandingan: metode, kualitas seam (subjektif 1–5), waktu.

### Analisis Percobaan 3
- Metode blending mana yang menghasilkan seam paling invisible?
- Berapa level pyramid optimal untuk multi-band blending?
- Pada kasus apa feather blending sudah cukup?
- Mengapa multi-band blending lebih baik secara teori?

---

## Percobaan 4: Multi-Image Panorama

### Tujuan
Membangun panorama dari 4+ gambar menggunakan chain homography.

### Dasar Teori
Untuk N gambar, kita mengestimasi homography antar pasangan bersebelahan, lalu chain mereka ke satu gambar referensi (tengah) untuk meminimalkan distorsi.

### Langkah Kerja
1. Load 5 gambar overlapping berurutan (Set 3).
2. Pilih gambar tengah sebagai referensi (misal gambar ke-3).
3. Estimasi homography antar pasangan bersebelahan: $H_{12}, H_{23}, H_{34}, H_{45}$.
4. Hitung homography ke referensi: $H_{1 \to 3} = H_{23}^{-1} \cdot H_{12}^{-1}$, dst.
5. Tentukan ukuran canvas yang cukup besar.
6. Warp semua gambar ke canvas.
7. Blend menggunakan multi-band.
8. Crop dan tampilkan panorama.
9. Bandingkan: referensi gambar kiri vs referensi gambar tengah vs referensi gambar kanan.
10. Analisis distorsi di tepi panorama.

### Analisis Percobaan 4
- Mengapa memilih gambar tengah sebagai referensi mengurangi distorsi?
- Berapa FOV total panorama yang dihasilkan?
- Apakah ada akumulasi error yang terlihat?
- Bagaimana perbandingan dengan hasil OpenCV Stitcher pada gambar yang sama?

---

## Percobaan 5: Cylindrical Projection

### Tujuan
Mengimplementasikan cylindrical warping untuk panorama wide-angle.

### Dasar Teori
Cylindrical projection memetakan gambar ke permukaan silinder, mengurangi distorsi pada panorama dengan FOV > 90°. Memerlukan estimasi focal length.

### Langkah Kerja
1. Implementasikan fungsi `cylindrical_warp(img, focal_length)`.
2. Estimasi focal length dari EXIF atau manual (coba beberapa nilai).
3. Warp setiap gambar ke cylindrical coordinates.
4. Stitch gambar cylindrical menggunakan translasi (bukan homography penuh).
5. Tampilkan hasil: planar projection vs cylindrical projection.
6. Variasikan focal length: terlalu kecil, optimal, terlalu besar.
7. Hitung translasi menggunakan feature matching dari gambar cylindrical.
8. Blend dan crop hasil panorama cylindrical.
9. Uji pada 5+ gambar → full cylindrical panorama.
10. Bandingkan distorsi di tepi: planar vs cylindrical.

### Analisis Percobaan 5
- Pada FOV berapa cylindrical lebih baik dari planar?
- Bagaimana focal length mempengaruhi hasil cylindrical warp?
- Apakah model translasi cukup untuk gambar cylindrical?
- Apa limitasi cylindrical projection?

---

## Percobaan 6: Spherical Projection

### Tujuan
Mengimplementasikan spherical warping untuk panorama full-view.

### Dasar Teori
Spherical projection memetakan gambar ke permukaan bola menggunakan $(\theta, \phi)$ coordinates. Lebih akurat daripada cylindrical untuk panorama yang juga menyertakan komponen vertikal.

### Langkah Kerja
1. Implementasikan fungsi `spherical_warp(img, focal_length)`.
2. Warp gambar test ke spherical coordinates.
3. Bandingkan visual: original vs cylindrical vs spherical.
4. Stitch 3 gambar menggunakan spherical projection.
5. Hitung translasi dari spherical-warped features.
6. Blend dan tampilkan hasil.
7. Equirectangular format: buat panorama 360° (jika punya cukup gambar).
8. Visualisasikan grid lines sebelum dan sesudah spherical warp.
9. Ukur distorsi: bandingkan garis lurus sebelum/sesudah warp.
10. Bandingkan kualitas: planar vs cylindrical vs spherical pada dataset yang sama.

### Analisis Percobaan 6
- Kapan spherical projection diperlukan vs cylindrical sudah cukup?
- Bagaimana distorsi garis lurus pada masing-masing proyeksi?
- Apa tambahan computational cost spherical vs cylindrical?
- Mengapa 360° panorama memerlukan spherical projection?

---

## Percobaan 7: Bundle Adjustment

### Tujuan
Memahami efek bundle adjustment pada kualitas panorama multi-image.

### Dasar Teori
Bundle adjustment mengoptimalkan semua parameter kamera secara simultan untuk meminimalkan reprojection error global, mengurangi akumulasi error dari chain homography.

### Langkah Kerja
1. Load 5+ gambar panorama.
2. Stitch menggunakan chain homography TANPA bundle adjustment (Percobaan 4).
3. Gunakan OpenCV `detail` module untuk bundle adjustment:
   - `cv2.detail.BundleAdjusterRay()` atau `BundleAdjusterReproj()`.
4. Bandingkan hasil: dengan vs tanpa bundle adjustment.
5. Zoom ke area seam dan titik-titik yang seharusnya lurus.
6. Hitung reprojection error sebelum dan sesudah BA.
7. Variasikan jumlah gambar (3, 5, 7) → amati efek BA.
8. Uji BA pada panorama loop (gambar pertama dan terakhir overlap).
9. Visualisasikan posisi kamera sebelum dan sesudah BA.
10. Dokumentasikan perbedaan kualitas dalam tabel + screenshot.

### Analisis Percobaan 7
- Seberapa besar penurunan reprojection error setelah BA?
- Pada berapa gambar efek BA mulai signifikan?
- Apakah BA berhasil menutup loop pada panorama 360°?
- Apa trade-off waktu komputasi BA?

---

## Percobaan 8: Exposure Compensation

### Tujuan
Mengatasi perbedaan exposure antar gambar dalam panorama.

### Dasar Teori
Gambar yang diambil secara manual sering memiliki exposure berbeda. Gain compensation menyesuaikan brightness setiap gambar agar konsisten di area overlap.

### Langkah Kerja
1. Ambil 3 foto dengan exposure sengaja berbeda (manual exposure control, atau edit brightness).
2. Stitch tanpa compensation → lihat seam exposure.
3. Implementasikan simple gain compensation: hitung rata-rata brightness di overlap, equalize.
4. Gunakan OpenCV: `cv2.detail.ExposureCompensator_createDefault(GAIN)`.
5. Gunakan `GAIN_BLOCKS` compensator.
6. Bandingkan: no compensation vs GAIN vs GAIN_BLOCKS.
7. Tampilkan gambar before/after compensation.
8. Stitch dengan compensation + blending → bandingkan dengan tanpa compensation.
9. Uji pada gambar indoor vs outdoor.
10. Buat tabel perbandingan metode compensation + screenshot seam area.

### Analisis Percobaan 8
- Seberapa efektif gain compensation mengurangi exposure seam?
- Apakah GAIN_BLOCKS lebih baik dari GAIN global?
- Pada perbedaan exposure berapa compensation mulai gagal?
- Mengapa exposure manual saat fotografi lebih baik daripada post-compensation?

---

## Percobaan 9: Seam Finding

### Tujuan
Memahami dan membandingkan metode seam finding untuk panorama.

### Dasar Teori
Seam finding menentukan garis potong optimal di area overlap yang meminimalkan perbedaan visual. Optimal seam mengikuti area dengan perbedaan kecil antara kedua gambar.

### Langkah Kerja
1. Siapkan 2 gambar yang sudah di-warp dan aligned.
2. Implementasikan Voronoi seam (batas di tengah overlap).
3. Gunakan OpenCV: `cv2.detail.GraphCutSeamFinder('COST_COLOR')`.
4. Gunakan `DpSeamFinder` (dynamic programming).
5. Visualisasikan garis seam di atas gambar overlap.
6. Stitch dengan masing-masing seam → bandingkan kualitas.
7. Uji pada gambar dengan objek di area overlap vs area kosong.
8. Hitung waktu per metode seam finding.
9. Kombinasikan seam finding + multi-band blending → hasil optimal.
10. Dokumentasikan visual comparison + tabel evaluation.

### Analisis Percobaan 9
- Seam finder mana yang menghasilkan seam paling invisible?
- Bagaimana GraphCut menangani objek di area overlap?
- Apakah seam finding + blending secara signifikan lebih baik dari blending saja?
- Apa trade-off komputasi antar metode?

---

## Percobaan 10: Real-time / Interactive Stitching

### Tujuan
Membangun aplikasi stitching interaktif atau real-time dari webcam/video.

### Dasar Teori
Real-time stitching memerlukan pipeline yang dioptimalkan: fitur cepat (ORB), limited matches, simple blending, dan reuse homography jika scene statis.

### Langkah Kerja
1. Setup 2 webcam atau 1 webcam + 1 gambar referensi.
2. Capture frame dari webcam.
3. Implementasikan fast stitching pipeline (ORB + BF + homography).
4. Stitch frame webcam dengan referensi secara real-time.
5. Tampilkan FPS dan jumlah matches.
6. Optimasi: cache homography jika jumlah inlier stabil.
7. Uji dengan kamera bergerak → dynamic homography update.
8. Tambahkan simple blending (feather) pada hasil real-time.
9. Jika hanya 1 webcam: tangkap 3 frame sambil memutar kamera → stitch.
10. Buat demo lengkap: capture → stitch → save panorama.

### Analisis Percobaan 10
- Berapa FPS yang dicapai?
- Apakah homography caching efektif?
- Apa bottleneck utama dalam real-time stitching?
- Bagaimana kualitas real-time stitching vs offline?

---

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Pipeline stitching: dari manual ke automatic (OpenCV Stitcher).
2. Perbandingan blending: feather vs multi-band.
3. Perbandingan projection: planar vs cylindrical vs spherical.
4. Efek bundle adjustment dan exposure compensation.
5. Rekomendasi pipeline optimal untuk berbagai skenario.

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`).
- **Naming**: `NIM_Nama_Modul08.ipynb`
