# JOBSHEET PRAKTIKUM
# MODUL 2: PEMBENTUKAN CITRA (IMAGE FORMATION)

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa mampu:
1. Memahami dan mengimplementasikan transformasi geometri 2D pada gambar.
2. Melakukan translasi, rotasi, dan scaling gambar menggunakan OpenCV.
3. Memahami dan menerapkan transformasi affine dan perspektif.
4. Melakukan kalibrasi kamera menggunakan pola checkerboard.
5. Memahami proyeksi perspektif dari 3D ke 2D.
6. Mendeteksi dan mengoreksi distorsi lensa.
7. Memahami efek sampling dan aliasing pada gambar.
8. Menerapkan koreksi gamma dan manipulasi photometric.
9. Mengimplementasikan simulasi Color Filter Array dan demosaicing.
10. Menganalisis artefak kompresi JPEG.

---

## 2. ALAT DAN BAHAN

### A. Perangkat Keras
- Laptop/PC (min. Intel Core i3, RAM 4 GB)
- Webcam (diperlukan untuk kalibrasi kamera)
- Pola checkerboard cetak (9×6 inner corners, tersedia PDF di folder data)

### B. Perangkat Lunak
- Python 3.8+, VS Code
- Library: `opencv-python`, `numpy`, `matplotlib`, `scipy`

### C. Dataset
- Gambar sample di folder `data/images/`.
- Gambar checkerboard untuk kalibrasi.
- Jalankan `download_data.py` untuk menyiapkan data.

---

## 3. LANGKAH KERJA

### Persiapan
1. Buat folder `Modul02/praktikum/`.
2. Siapkan gambar sample dan data kalibrasi.

---

### Percobaan 1: Translasi Gambar

**Tujuan**: Memahami translasi sebagai pergeseran posisi gambar menggunakan matriks transformasi.

**Langkah Kerja**:
1. Buat file `01_translasi.py`.
2. Baca gambar berwarna.
3. Definisikan matriks translasi 2×3:
   ```
   M = [[1, 0, tx],
        [0, 1, ty]]
   ```
4. Terapkan translasi menggunakan `cv2.warpAffine(img, M, (w, h))`.
5. Variasikan nilai `tx` dan `ty` (positif dan negatif).
6. Amati bagian gambar yang hilang dan area hitam yang muncul.
7. Implementasikan translasi interaktif dengan trackbar OpenCV.
8. Buat animasi translasi (geser gambar dari kiri ke kanan).
9. Terapkan translasi pada ROI tertentu.
10. Simpan hasil ke folder output.

---

### Percobaan 2: Rotasi Gambar

**Tujuan**: Memutar gambar dengan sudut tertentu menggunakan matriks rotasi.

**Langkah Kerja**:
1. Buat file `02_rotasi.py`.
2. Baca gambar.
3. Tentukan center of rotation (pusat gambar).
4. Buat matriks rotasi: `M = cv2.getRotationMatrix2D(center, angle, scale)`.
5. Terapkan: `cv2.warpAffine(img, M, (w, h))`.
6. Rotasi dengan sudut 45°, 90°, 180°, 270°.
7. Rotasi dengan skala berbeda (`scale` = 0.5, 1.0, 1.5).
8. Rotasi dengan pusat yang bukan di tengah gambar.
9. Implementasikan rotasi yang memperbesar canvas agar gambar tidak terpotong.
10. Buat animasi rotasi 360° (sequence gambar).

---

### Percobaan 3: Scaling dan Interpolasi

**Tujuan**: Memahami proses scaling dan pengaruh metode interpolasi terhadap kualitas.

**Langkah Kerja**:
1. Buat file `03_scaling.py`.
2. Baca gambar.
3. Upscale gambar 2× dan 4× dengan berbagai interpolasi:
   - `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC`, `INTER_LANCZOS4`.
4. Downscale gambar 0.5× dan 0.25× dengan `INTER_AREA`.
5. Bandingkan kualitas visual: crop area detail, tampilkan side-by-side.
6. Hitung waktu eksekusi setiap metode interpolasi.
7. Scale gambar dengan mempertahankan aspect ratio.
8. Buat piramida gambar (downscale bertahap 50% × 4 level).
9. Implementasikan resize ke ukuran target tertentu dengan letterbox/pillarbox.
10. Simpan perbandingan visual ke folder output.

---

### Percobaan 4: Transformasi Affine

**Tujuan**: Menerapkan transformasi affine menggunakan 3 pasang titik korespondensi.

**Langkah Kerja**:
1. Buat file `04_affine_transform.py`.
2. Baca gambar.
3. Definisikan 3 titik sumber dan 3 titik tujuan.
4. Hitung matriks affine: `M = cv2.getAffineTransform(src_pts, dst_pts)`.
5. Terapkan: `cv2.warpAffine(img, M, (w, h))`.
6. Buat contoh: shearing (memiringkan gambar).
7. Buat contoh: mirroring (pencerminan).
8. Kombinasikan: rotasi + scaling + translasi dalam satu matriks affine.
9. Terapkan transformasi affine pada wajah dalam foto (simulasi face alignment).
10. Visualisasikan titik-titik korespondensi sebelum dan sesudah transformasi.

---

### Percobaan 5: Transformasi Perspektif (Homography)

**Tujuan**: Memahami transformasi perspektif menggunakan 4 pasang titik dan matriks homografi.

**Langkah Kerja**:
1. Buat file `05_perspektif_transform.py`.
2. Baca gambar dokumen/papan yang difoto dari sudut miring.
3. Tentukan 4 titik sudut objek di gambar (sumber).
4. Tentukan 4 titik sudut tujuan (persegi panjang tegak lurus).
5. Hitung homografi: `M = cv2.getPerspectiveTransform(src, dst)`.
6. Terapkan: `cv2.warpPerspective(img, M, (w, h))`.
7. Implementasikan document scanner sederhana: koreksi perspektif dokumen.
8. Gunakan `cv2.findHomography()` dengan lebih dari 4 titik (least squares).
9. Terapkan inverse perspective transform (bird's eye view untuk jalan).
10. Buat overlay gambar ke permukaan perspektif (billboard replacement).

---

### Percobaan 6: Kalibrasi Kamera

**Tujuan**: Melakukan kalibrasi kamera untuk mendapatkan parameter intrinsik dan koefisien distorsi.

**Langkah Kerja**:
1. Buat file `06_kalibrasi_kamera.py`.
2. Cetak atau tampilkan pola checkerboard di layar.
3. Ambil minimal 10 foto checkerboard dari berbagai sudut.
4. Deteksi corners: `cv2.findChessboardCorners()`.
5. Refine corners: `cv2.cornerSubPix()`.
6. Kalibrasi: `ret, K, dist, rvecs, tvecs = cv2.calibrateCamera()`.
7. Cetak matriks intrinsik K dan koefisien distorsi.
8. Hitung reprojection error.
9. Simpan hasil kalibrasi ke file YAML/JSON.
10. Visualisasikan posisi kamera relatif terhadap pattern (opsional).

---

### Percobaan 7: Proyeksi 3D ke 2D

**Tujuan**: Memahami proses proyeksi titik 3D ke koordinat gambar 2D.

**Langkah Kerja**:
1. Buat file `07_proyeksi_3d.py`.
2. Load hasil kalibrasi (K, dist).
3. Definisikan titik-titik 3D (kubus, piramida, sumbu XYZ).
4. Proyeksikan ke 2D: `cv2.projectPoints(objPoints, rvec, tvec, K, dist)`.
5. Gambar objek 3D yang terproyeksi di atas gambar checkerboard.
6. Gambar sumbu koordinat 3D (merah=X, hijau=Y, biru=Z) pada pattern.
7. Variasikan rvec dan tvec untuk mengubah pose objek virtual.
8. Buat kubus 3D wireframe yang ter-overlay di atas checkerboard.
9. Simulasikan efek perubahan focal length terhadap perspektif.
10. Simpan visualisasi ke folder output.

---

### Percobaan 8: Koreksi Distorsi Lensa

**Tujuan**: Mendeteksi dan mengoreksi distorsi radial dan tangensial.

**Langkah Kerja**:
1. Buat file `08_distorsi_lensa.py`.
2. Load hasil kalibrasi (K, dist coefficients).
3. Undistort gambar: `cv2.undistort(img, K, dist)`.
4. Gunakan metode alternatif: `cv2.initUndistortRectifyMap()` + `cv2.remap()`.
5. Bandingkan gambar sebelum dan sesudah koreksi (overlay grid).
6. Simulasikan distorsi barrel: buat gambar grid dan terapkan distorsi buatan.
7. Simulasikan distorsi pincushion.
8. Visualisasikan efek setiap koefisien distorsi (k1, k2, p1, p2) secara terpisah.
9. Hitung dan plot radial distortion profile.
10. Bandingkan undistort dengan crop vs tanpa crop (alpha parameter).

---

### Percobaan 9: Sampling dan Aliasing

**Tujuan**: Memahami efek sampling, aliasing, dan anti-aliasing pada gambar.

**Langkah Kerja**:
1. Buat file `09_sampling_aliasing.py`.
2. Buat gambar sintetis: garis-garis dengan frekuensi tinggi (pola grid halus).
3. Downscale tanpa filter → amati aliasing (pola moiré).
4. Downscale dengan Gaussian blur terlebih dahulu → amati perbedaan.
5. Bandingkan `INTER_NEAREST` vs `INTER_AREA` pada downscaling.
6. Buat zona plate (pola lingkaran konsentris) dan demonstrasikan aliasing.
7. Implementasikan anti-aliasing manual: blur → downsample.
8. Variasikan sigma Gaussian dan amati efeknya terhadap detail.
9. Demonstrasikan aliasing pada rotasi gambar (jagged edges).
10. Buat laporan visual perbandingan dengan dan tanpa anti-aliasing.

---

### Percobaan 10: Koreksi Gamma dan Photometric

**Tujuan**: Memahami gamma correction, photometric shading, dan manipulasi pencahayaan.

**Langkah Kerja**:
1. Buat file `10_gamma_photometric.py`.
2. Terapkan gamma correction: $I_{out} = I_{in}^{\gamma}$ untuk berbagai nilai γ (0.5, 1.0, 1.5, 2.2).
3. Buat Look-Up Table (LUT) untuk gamma correction cepat.
4. Terapkan gamma adjustment menggunakan `cv2.LUT()`.
5. Simulasikan pencahayaan Lambertian pada permukaan datar.
6. Buat efek vignette (gelap di pinggir, terang di tengah).
7. Implementasikan white balance sederhana (gray world assumption).
8. Simulasikan Bayer filter dan demosaicing sederhana.
9. Bandingkan gambar sebelum dan sesudah gamma correction dengan histogram.
10. Analisis artefak kompresi JPEG pada berbagai quality factor.

---

## 4. ANALISIS

### Analisis Percobaan 1 — Translasi
- Jelaskan mengapa muncul area hitam setelah translasi.
- Apa yang terjadi jika `tx` atau `ty` lebih besar dari dimensi gambar?
- Bagaimana translasi digunakan dalam image registration?

### Analisis Percobaan 2 — Rotasi
- Mengapa gambar terpotong saat dirotasi 45°? Bagaimana solusinya?
- Apa efek parameter `scale` pada rotasi?
- Jelaskan konsep center of rotation dan implikasinya.

### Analisis Percobaan 3 — Scaling
- Bandingkan kualitas visual interpolasi nearest vs bilinear vs bicubic.
- Mengapa `INTER_AREA` lebih baik untuk downscaling?
- Berapa speedup/slowdown dari tiap metode interpolasi?

### Analisis Percobaan 4 — Affine
- Berapa DOF (Degrees of Freedom) transformasi affine? Mengapa perlu 3 pasang titik?
- Properti apa yang dipertahankan oleh transformasi affine?
- Berikan contoh aplikasi shearing di dunia nyata.

### Analisis Percobaan 5 — Perspektif
- Apa beda transformasi affine dan perspektif?
- Mengapa diperlukan 4 pasang titik untuk homografi?
- Jelaskan mengapa document scanner menggunakan transformasi perspektif.

### Analisis Percobaan 6 — Kalibrasi
- Berapa reprojection error yang didapat? Apakah sudah baik?
- Apa pengaruh jumlah gambar kalibrasi terhadap akurasi?
- Jelaskan arti fisik dari setiap elemen matriks intrinsik K.

### Analisis Percobaan 7 — Proyeksi 3D
- Jelaskan hubungan antara focal length dan field of view.
- Apa yang terjadi pada objek 3D saat mendekat/menjauh dari kamera?
- Mengapa objek yang jauh terlihat lebih kecil (perspektif)?

### Analisis Percobaan 8 — Distorsi Lensa
- Bandingkan visual gambar sebelum dan sesudah undistort.
- Jenis distorsi apa yang dominan pada kamera Anda (barrel/pincushion)?
- Mengapa koreksi distorsi penting untuk pengukuran yang akurat?

### Analisis Percobaan 9 — Sampling dan Aliasing
- Jelaskan teorema Nyquist dan hubungannya dengan aliasing pada gambar.
- Mengapa pola moiré muncul? Bagaimana mencegahnya?
- Trade-off apa yang ada antara anti-aliasing dan ketajaman gambar?

### Analisis Percobaan 10 — Gamma dan Photometric
- Jelaskan mengapa gamma correction diperlukan.
- Bandingkan histogram sebelum dan sesudah gamma correction.
- Bagaimana white balance mempengaruhi hasil gambar?

---

## 5. KESIMPULAN

Buatlah kesimpulan yang mencakup:
1. Pemahaman tentang proses pembentukan citra dari scene 3D ke gambar 2D.
2. Perbedaan dan hierarki transformasi geometri (translasi → affine → perspektif).
3. Pentingnya kalibrasi kamera dan koreksi distorsi untuk aplikasi yang presisi.
4. Hubungan antara fotometri, gamma, dan kualitas visual gambar.
5. Konsep sampling, aliasing, dan solusi anti-aliasing.
