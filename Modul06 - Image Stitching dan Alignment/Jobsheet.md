# JOBSHEET MODUL 6: IMAGE STITCHING DAN ALIGNMENT

---

## Tujuan Praktikum
1. Memahami dan mengimplementasikan pipeline image stitching dari awal.
2. Menggunakan OpenCV Stitcher API untuk panorama otomatis.
3. Membandingkan teknik blending (feather, multi-band, Poisson).
4. Mengimplementasikan cylindrical dan spherical projection.
5. Memahami exposure compensation dan seam finding.
6. Membangun panorama multi-image dan real-time stitching interaktif.
7. Mengimplementasikan auto-cropping panorama untuk menghilangkan border hitam.
8. Memahami homography estimation secara mendalam (RANSAC, dekomposisi, analisis error).
9. Mengimplementasikan image registration menggunakan ECC dan feature-based.
10. Mendalami Laplacian pyramid blending per level secara detail.
11. Mengimplementasikan gain compensation manual dan histogram matching.
12. Mengevaluasi kualitas stitching menggunakan metrik PSNR dan SSIM.
13. Memahami panorama loop closure dan distribusi error.
14. Mengimplementasikan document alignment dan stitching.
15. Membangun pipeline panorama lengkap berbasis class.
16. Mengembangkan aplikasi panorama maker dengan berbagai mode.

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

## Percobaan 11: Panorama Cropping dan Auto-crop

### Tujuan
Mengimplementasikan auto-cropping untuk menghilangkan border hitam pada hasil panorama.

### Dasar Teori
Hasil stitching sering memiliki border hitam tidak beraturan. Auto-crop menggunakan thresholding, deteksi kontur, operasi morfologi, dan pendekatan maximum inscribed rectangle untuk mendapatkan panorama bersih tanpa area hitam.

### Langkah Kerja
1. Load hasil panorama dari percobaan sebelumnya (yang memiliki border hitam).
2. Konversi ke grayscale dan terapkan threshold untuk memisahkan konten dari border hitam.
3. Terapkan operasi morfologi (closing) untuk menutup gap kecil pada mask.
4. Temukan kontur terbesar dari mask hasil threshold.
5. Hitung bounding rectangle dari kontur → crop sederhana.
6. Implementasikan maximum inscribed rectangle (largest rect tanpa piksel hitam di dalamnya).
7. Bandingkan hasil: bounding rect crop vs max inscribed rect crop.
8. Uji pada beberapa panorama dengan shape border berbeda.
9. Tambahkan padding opsional setelah crop (margin putih/hitam).
10. Tampilkan before/after crop + persentase area yang terpotong.

### Analisis Percobaan 11
- Seberapa efektif threshold-based cropping menghilangkan border hitam?
- Apa perbedaan visual antara bounding rect vs max inscribed rect?
- Berapa persen area gambar yang hilang akibat cropping?
- Pada kasus apa max inscribed rect jauh lebih kecil dari bounding rect?

---

## Percobaan 12: Homography Estimation dan Visualisasi

### Tujuan
Memahami homography estimation secara mendalam: 4-point DLT, RANSAC, dekomposisi, dan analisis error.

### Dasar Teori
Homography $H$ (matriks 3×3, 8 DOF) memetakan titik dari satu bidang ke bidang lain. Estimasi membutuhkan minimal 4 korespondensi titik (DLT). RANSAC memfilter outlier secara robust. Dekomposisi $H$ mengungkapkan rotasi, translasi, dan normal bidang.

### Langkah Kerja
1. Load 2 gambar overlapping dan deteksi fitur (SIFT) + match (FLANN + ratio test).
2. Estimasi homography menggunakan DLT (Direct Linear Transform) 4-point minimal.
3. Estimasi homography menggunakan RANSAC → bandingkan dengan DLT.
4. Visualisasikan inlier (hijau) dan outlier (merah) pada gambar matches.
5. Hitung reprojection error per titik korespondensi.
6. Plot distribusi reprojection error (histogram).
7. Dekomposisi homography menggunakan `cv2.decomposeHomographyMat()`.
8. Visualisasikan efek homography: warp grid dan lihat deformasi.
9. Uji robustness: tambahkan noise pada matches → amati perubahan H.
10. Bandingkan homography dari SIFT vs ORB vs AKAZE matches.

### Analisis Percobaan 12
- Berapa reprojection error rata-rata untuk DLT vs RANSAC?
- Berapa persen outlier yang berhasil difilter RANSAC?
- Apa informasi yang didapat dari dekomposisi homography?
- Bagaimana noise pada matches mempengaruhi stabilitas homography?

---

## Percobaan 13: Image Registration (Penyelarasan Gambar)

### Tujuan
Mengimplementasikan image registration menggunakan ECC (Enhanced Correlation Coefficient) dan membandingkan dengan feature-based.

### Dasar Teori
ECC algorithm menemukan transformasi optimal yang memaksimalkan korelasi antara template dan gambar target. Mendukung mode translasi, euclidean, affine, dan homography. Berbeda dengan feature-based yang diskrit, ECC bekerja pada level piksel secara iteratif.

### Langkah Kerja
1. Load 2 gambar yang sedikit bergeser (translasi kecil).
2. Implementasikan ECC registration dengan mode `MOTION_TRANSLATION`.
3. Uji mode `MOTION_EUCLIDEAN` (translasi + rotasi).
4. Uji mode `MOTION_AFFINE` (6 DOF).
5. Uji mode `MOTION_HOMOGRAPHY` (8 DOF).
6. Bandingkan warp matrix hasil ECC dari setiap mode.
7. Implementasikan feature-based registration (SIFT + homography) pada gambar yang sama.
8. Bandingkan akurasi ECC vs feature-based: overlay registered images.
9. Uji pada gambar dengan perubahan brightness → amati robustness.
10. Ukur waktu komputasi per mode dan bandingkan.

### Analisis Percobaan 13
- Mode ECC mana yang paling sesuai untuk translasi murni?
- Bagaimana perbandingan akurasi ECC vs feature-based?
- Apakah ECC robust terhadap perubahan brightness?
- Kapan ECC lebih tepat digunakan dibanding feature-based dan sebaliknya?

---

## Percobaan 14: Laplacian Pyramid Blending Detail

### Tujuan
Mengimplementasikan Laplacian pyramid blending secara detail langkah demi langkah.

### Dasar Teori
Multi-band blending memisahkan gambar menjadi band frekuensi melalui Laplacian pyramid. Setiap level diblend menggunakan mask Gaussian pyramid. Rekonstruksi menghasilkan transisi halus karena frekuensi rendah dan tinggi diblend secara terpisah.

### Langkah Kerja
1. Load 2 gambar overlapping dan buat mask binary (kiri/kanan).
2. Bangun Gaussian pyramid (5 level) dari kedua gambar.
3. Bangun Laplacian pyramid dari Gaussian pyramid → visualisasikan setiap level.
4. Bangun Gaussian pyramid dari mask.
5. Blend Laplacian di setiap level: `L_blend = mask * L1 + (1-mask) * L2`.
6. Rekonstruksi gambar dari blended Laplacian pyramid.
7. Tampilkan hasil per tahap: Gaussian levels, Laplacian levels, blended levels, final.
8. Variasikan jumlah level (2, 3, 5, 7) → bandingkan kualitas blending.
9. Bandingkan dengan simple alpha blending dan feather blending.
10. Uji pada gambar dengan perbedaan exposure besar di area overlap.

### Analisis Percobaan 14
- Bagaimana setiap level Laplacian pyramid menyimpan informasi frekuensi?
- Berapa level optimal untuk blending?
- Mengapa Laplacian blending lebih baik dari alpha blending biasa?
- Apa efek jumlah level terlalu sedikit vs terlalu banyak?

---

## Percobaan 15: Gain Compensation Manual

### Tujuan
Mengimplementasikan gain compensation secara manual untuk menyesuaikan exposure antar gambar.

### Dasar Teori
Gain compensation menyesuaikan brightness setiap gambar agar konsisten. Dapat dilakukan secara global (satu gain per gambar), per-channel (R, G, B terpisah), atau di ruang warna LAB. CLAHE dan histogram matching memberikan pendekatan adaptif.

### Langkah Kerja
1. Siapkan 3 gambar overlapping dengan exposure sengaja berbeda.
2. Implementasikan global gain compensation: hitung ratio brightness di overlap.
3. Implementasikan per-channel gain: kompensasi R, G, B secara independen.
4. Implementasikan kompensasi di ruang warna LAB (hanya channel L).
5. Terapkan CLAHE pada setiap gambar sebelum stitching.
6. Implementasikan histogram matching: sesuaikan histogram gambar ke gambar referensi.
7. Stitch dengan masing-masing metode compensation → bandingkan visual.
8. Zoom ke area overlap → evaluasi konsistensi warna.
9. Ukur standard deviation brightness di area overlap untuk setiap metode.
10. Buat tabel perbandingan: metode, visual quality (1–5), color consistency, waktu.

### Analisis Percobaan 15
- Metode compensation mana yang menghasilkan warna paling konsisten?
- Apakah per-channel lebih baik dari global gain?
- Kapan CLAHE lebih tepat dibanding gain compensation?
- Bagaimana histogram matching menangani perbedaan white balance?

---

## Percobaan 16: Evaluasi Kualitas Stitching (PSNR, SSIM)

### Tujuan
Mengimplementasikan metrik evaluasi kualitas stitching: PSNR, SSIM, difference map, dan edge alignment.

### Dasar Teori
Kualitas stitching dapat dievaluasi secara kuantitatif. PSNR mengukur rasio signal-to-noise, SSIM mengukur kemiripan struktural. Difference map menunjukkan area dengan perbedaan besar, dan edge alignment mengukur kesinambungan tepi di seam.

### Langkah Kerja
1. Siapkan area overlap dari 2 gambar yang sudah di-warp (ground truth overlap).
2. Hitung PSNR antara area overlap gambar 1 dan gambar 2.
3. Hitung SSIM menggunakan `skimage.metrics.structural_similarity`.
4. Buat difference map (absolute difference) dan visualisasikan dengan colormap.
5. Deteksi edge pada kedua gambar di area seam → hitung edge alignment score.
6. Bandingkan metrik untuk berbagai metode blending: no blend, feather, multi-band.
7. Bandingkan metrik untuk panorama dengan vs tanpa exposure compensation.
8. Buat fungsi evaluasi lengkap yang menghitung semua metrik sekaligus.
9. Uji pada 3 dataset berbeda → rata-rata metrik per metode.
10. Visualisasikan hasil: bar chart perbandingan metrik per metode.

### Analisis Percobaan 16
- Metode blending mana yang menghasilkan PSNR dan SSIM tertinggi?
- Apakah PSNR dan SSIM selalu berkorelasi dengan persepsi visual?
- Seberapa besar pengaruh exposure compensation terhadap metrik kualitas?
- Area mana pada difference map yang menunjukkan perbedaan terbesar?

---

## Percobaan 17: Panorama Loop Closure

### Tujuan
Memahami dan mengimplementasikan loop closure untuk panorama 360° untuk mengurangi drift akumulatif.

### Dasar Teori
Pada panorama 360°, chain homography mengakumulasi error sehingga gambar pertama dan terakhir tidak align sempurna (drift). Loop closure mendeteksi overlap antara gambar awal dan akhir, kemudian mendistribusikan error secara merata ke seluruh chain.

### Langkah Kerja
1. Load 6+ gambar yang membentuk loop (gambar pertama dan terakhir overlap).
2. Stitch tanpa loop closure → amati drift antara ujung pertama dan terakhir.
3. Deteksi overlap antara gambar pertama dan terakhir (feature matching).
4. Hitung homography "penutup" loop: $H_{N \to 1}$.
5. Hitung accumulated drift: $H_{drift} = H_{N \to 1} \cdot H_{(N-1) \to N} \cdot \ldots \cdot H_{1 \to 2}$.
6. Distribusikan error secara merata: interpolasi koreksi ke setiap homography.
7. Stitch ulang dengan homography yang sudah dikoreksi.
8. Bandingkan visual: tanpa vs dengan loop closure.
9. Ukur reprojection error sebelum dan sesudah loop closure.
10. Visualisasikan posisi kamera sebelum dan sesudah koreksi.

### Analisis Percobaan 17
- Seberapa besar drift yang terjadi tanpa loop closure?
- Apakah distribusi error merata efektif mengurangi drift?
- Berapa banyak gambar minimal agar loop closure signifikan?
- Apa limitasi pendekatan distribusi error linier?

---

## Percobaan 18: Document Alignment dan Stitching

### Tujuan
Mengimplementasikan alignment dan stitching khusus untuk dokumen (buku, whiteboard, receipt panjang).

### Dasar Teori
Document stitching berbeda dari panorama alam karena memerlukan perspective correction (dokumen harus rectangular), alignment yang presisi, dan biasanya stacking vertikal. Feature-based matching dikombinasikan dengan edge detection untuk alignment optimal.

### Langkah Kerja
1. Ambil 2–3 foto bagian dokumen/whiteboard yang overlapping.
2. Terapkan perspective correction pada setiap foto (4-point transform).
3. Konversi ke grayscale dan enhance contrast.
4. Deteksi fitur dan match antar bagian dokumen.
5. Estimasi transformasi (biasanya translasi + sedikit rotasi).
6. Stitch secara vertikal (top-to-bottom) untuk dokumen panjang.
7. Stitch secara horizontal untuk whiteboard lebar.
8. Terapkan binarization (Otsu/adaptive) pada hasil stitching.
9. Bandingkan hasil: feature-based vs template matching untuk alignment.
10. Export hasil sebagai gambar high-res dan simulasikan output PDF.

### Analisis Percobaan 18
- Apa tantangan khusus stitching dokumen vs panorama alam?
- Seberapa penting perspective correction untuk hasil stitching dokumen?
- Metode alignment mana yang lebih akurat untuk teks/dokumen?
- Bagaimana kualitas teks di area overlap setelah stitching?

---

## Percobaan 19: Panorama Pipeline Lengkap

### Tujuan
Membangun pipeline panorama lengkap berbasis class yang mengintegrasikan seluruh teknik.

### Dasar Teori
Pipeline panorama lengkap menggabungkan: feature detection → matching → homography estimation → exposure compensation → warping → seam finding → blending → cropping. Implementasi berbasis class memungkinkan modularitas dan konfigurasi fleksibel.

### Langkah Kerja
1. Buat class `PanoramaPipeline` dengan method untuk setiap tahap.
2. Implementasikan `detect_and_match()`: SIFT/ORB + ratio test.
3. Implementasikan `estimate_homography()`: RANSAC + validasi.
4. Implementasikan `compensate_exposure()`: gain compensation.
5. Implementasikan `warp_images()`: cylindrical/planar, tentukan canvas.
6. Implementasikan `find_seams()`: GraphCut seam finding.
7. Implementasikan `blend_images()`: multi-band blending.
8. Implementasikan `crop_result()`: auto-crop border hitam.
9. Jalankan pipeline end-to-end pada 5 gambar → panorama final.
10. Tambahkan logging: waktu per tahap, jumlah fitur, error metrics.

### Analisis Percobaan 19
- Tahap mana yang paling memakan waktu dalam pipeline?
- Apakah urutan compensate → warp → seam → blend optimal?
- Bagaimana modularitas class membantu debugging?
- Apa perbedaan hasil pipeline lengkap vs OpenCV Stitcher API?

---

## Percobaan 20: Proyek Panorama Maker App

### Tujuan
Mengembangkan aplikasi panorama maker lengkap dengan mode auto, manual, fast, batch processing, dan quality report.

### Dasar Teori
Aplikasi panorama yang robust memerlukan beberapa mode untuk menangani berbagai skenario: mode auto (full pipeline), mode manual (user memilih pasangan dan parameter), mode fast (pipeline minimal untuk kecepatan). Batch processing dan quality report melengkapi fitur produksi.

### Langkah Kerja
1. Buat struktur aplikasi: main menu dengan pilihan mode (auto/manual/fast).
2. **Mode Auto**: Implementasikan full pipeline otomatis (deteksi urutan, stitch semua).
3. **Mode Manual**: User memilih pasangan gambar, preview matches, adjust parameter.
4. **Mode Fast**: Pipeline minimal (ORB + simple blend, tanpa seam finding).
5. Implementasikan batch processing: input folder → stitch semua set → output folder.
6. Implementasikan quality report: PSNR, SSIM, waktu, jumlah fitur per panorama.
7. Tambahkan preview sebelum save (resize untuk display).
8. Implementasikan export: save panorama dengan metadata (resolusi, metode, waktu).
9. Uji aplikasi pada 3 dataset berbeda dengan ketiga mode.
10. Dokumentasikan: screenshot setiap mode, tabel perbandingan kecepatan vs kualitas.

### Analisis Percobaan 20
- Mode mana yang memberikan trade-off terbaik antara kualitas dan kecepatan?
- Apakah batch processing berhasil menangani dataset dengan variasi?
- Seberapa informatif quality report untuk menilai hasil?
- Fitur apa yang paling penting untuk user experience aplikasi panorama?

---

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Pipeline stitching: dari manual ke automatic (OpenCV Stitcher).
2. Perbandingan blending: feather vs multi-band vs Laplacian pyramid detail.
3. Perbandingan projection: planar vs cylindrical vs spherical.
4. Efek bundle adjustment dan exposure compensation.
5. Teknik auto-cropping dan post-processing panorama.
6. Homography estimation: metode, dekomposisi, dan analisis error.
7. Image registration: ECC vs feature-based.
8. Gain compensation manual dan teknik histogram matching.
9. Evaluasi kualitas stitching: PSNR, SSIM, dan metrik lainnya.
10. Panorama loop closure dan distribusi error.
11. Document alignment dan stitching untuk kasus khusus.
12. Pipeline panorama lengkap dan aplikasi panorama maker.
13. Rekomendasi pipeline optimal untuk berbagai skenario.

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`).
- **Naming**: `NIM_Nama_Modul06.ipynb`
