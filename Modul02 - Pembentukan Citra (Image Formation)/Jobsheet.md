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
9. Mengimplementasikan berbagai metode interpolasi gambar.
10. Memahami konversi koordinat Cartesian ke polar.
11. Menerapkan shearing dan refleksi sebagai transformasi geometri.
12. Mengkomposisikan transformasi dalam matriks homogen.
13. Membangun image pyramid untuk representasi multi-skala.
14. Mensimulasi dan mengoreksi distorsi barrel dan pincushion.
15. Menggunakan remapping untuk transformasi gambar fleksibel.
16. Menerapkan transformasi intensitas logaritmik dan power-law.
17. Membuat citra sintetis untuk keperluan pengujian.

---

## 2. ALAT DAN BAHAN

### A. Perangkat Keras
- Laptop/PC (min. Intel Core i3, RAM 4 GB)
- Webcam (diperlukan untuk kalibrasi kamera)
- Pola checkerboard cetak (9×6 inner corners)

### B. Perangkat Lunak
- Python 3.8+, VS Code
- Library: `opencv-python`, `numpy`, `matplotlib`, `scipy`

### C. Dataset
- Jalankan `download_image.py` di folder `praktikum/` untuk menyiapkan gambar sample.
- Gambar akan tersimpan di folder `praktikum/image/`.

---

## 3. PERSIAPAN

1. Masuk ke folder `praktikum/`.
2. Jalankan: `python download_image.py`
3. Pastikan folder `image/` berisi gambar sample yang dibutuhkan.
4. Folder `output/` akan dibuat otomatis oleh setiap program.

---

## 4. LANGKAH KERJA

### Percobaan 1: Translasi Gambar
**File**: `01_translasi_gambar.py`

**Tujuan**: Memahami translasi sebagai pergeseran posisi gambar menggunakan matriks transformasi 2×3.

**Langkah Kerja**:
1. Buka dan pelajari file `01_translasi_gambar.py`.
2. Jalankan program: `python 01_translasi_gambar.py`.
3. Amati translasi dengan berbagai nilai `tx` dan `ty` (positif dan negatif).
4. Perhatikan area hitam yang muncul dan bagian gambar yang hilang.
5. Amati implementasi translasi interaktif dan animasi translasi.
6. Periksa output di folder `output/`.

---

### Percobaan 2: Rotasi Sudut Bebas
**File**: `02_rotasi_sudut_bebas.py`

**Tujuan**: Memutar gambar dengan sudut bebas menggunakan `cv2.getRotationMatrix2D` dan `cv2.warpAffine`.

**Langkah Kerja**:
1. Buka dan pelajari file `02_rotasi_sudut_bebas.py`.
2. Jalankan program: `python 02_rotasi_sudut_bebas.py`.
3. Amati rotasi dengan berbagai sudut (45°, 90°, 180°, 270°).
4. Amati efek parameter scale pada rotasi.
5. Perhatikan perbedaan rotasi dengan dan tanpa expand canvas.
6. Periksa output di folder `output/`.

---

### Percobaan 3: Scaling dan Zoom
**File**: `03_scaling_zoom.py`

**Tujuan**: Memahami proses scaling (upscale/downscale) dan pengaruh metode interpolasi terhadap kualitas.

**Langkah Kerja**:
1. Buka dan pelajari file `03_scaling_zoom.py`.
2. Jalankan program: `python 03_scaling_zoom.py`.
3. Bandingkan kualitas visual berbagai metode interpolasi (NEAREST, LINEAR, CUBIC, LANCZOS4, AREA).
4. Amati perbedaan antara upscale dan downscale.
5. Perhatikan piramida gambar (downscale bertahap).
6. Periksa output di folder `output/`.

---

### Percobaan 4: Transformasi Affine
**File**: `04_transformasi_affine.py`

**Tujuan**: Menerapkan transformasi affine menggunakan 3 pasang titik korespondensi.

**Langkah Kerja**:
1. Buka dan pelajari file `04_transformasi_affine.py`.
2. Jalankan program: `python 04_transformasi_affine.py`.
3. Amati transformasi affine dari 3 pasang titik sumber dan tujuan.
4. Perhatikan kombinasi rotasi + scaling + translasi dalam satu matriks affine.
5. Amati contoh shearing dan pencerminan via affine.
6. Periksa output di folder `output/`.

---

### Percobaan 5: Transformasi Perspektif
**File**: `05_transformasi_perspektif.py`

**Tujuan**: Memahami transformasi perspektif (homografi) menggunakan 4 pasang titik.

**Langkah Kerja**:
1. Buka dan pelajari file `05_transformasi_perspektif.py`.
2. Jalankan program: `python 05_transformasi_perspektif.py`.
3. Amati koreksi perspektif dokumen (document scanner).
4. Perhatikan perbedaan `cv2.getPerspectiveTransform` dan `cv2.findHomography`.
5. Amati bird's eye view transform dan billboard replacement.
6. Periksa output di folder `output/`.

---

### Percobaan 6: Kalibrasi Kamera Checkerboard
**File**: `06_kalibrasi_kamera_checkerboard.py`

**Tujuan**: Melakukan kalibrasi kamera untuk mendapatkan parameter intrinsik dan koefisien distorsi.

**Langkah Kerja**:
1. Buka dan pelajari file `06_kalibrasi_kamera_checkerboard.py`.
2. Jalankan program: `python 06_kalibrasi_kamera_checkerboard.py`.
3. Amati proses deteksi corners dan refinement.
4. Periksa matriks intrinsik K dan koefisien distorsi yang dihasilkan.
5. Amati reprojection error.
6. Periksa output di folder `output/`.

---

### Percobaan 7: Proyeksi 3D ke 2D
**File**: `07_proyeksi_3d_ke_2d.py`

**Tujuan**: Memahami proses proyeksi titik 3D ke koordinat gambar 2D menggunakan `cv2.projectPoints`.

**Langkah Kerja**:
1. Buka dan pelajari file `07_proyeksi_3d_ke_2d.py`.
2. Jalankan program: `python 07_proyeksi_3d_ke_2d.py`.
3. Amati objek 3D (kubus, sumbu XYZ) yang diproyeksikan ke gambar 2D.
4. Perhatikan efek perubahan rvec/tvec dan focal length.
5. Amati wireframe kubus 3D yang ter-overlay.
6. Periksa output di folder `output/`.

---

### Percobaan 8: Shearing dan Refleksi
**File**: `08_shearing_dan_refleksi.py`

**Tujuan**: Menerapkan transformasi shearing (geser) dan refleksi (cermin) menggunakan matriks affine dan cv2.flip.

**Langkah Kerja**:
1. Buka dan pelajari file `08_shearing_dan_refleksi.py`.
2. Jalankan program: `python 08_shearing_dan_refleksi.py`.
3. Amati shearing horizontal dan vertikal dengan berbagai faktor shear.
4. Bandingkan refleksi horizontal, vertikal, dan gabungan menggunakan cv2.flip.
5. Perhatikan efek artistik seperti water reflection.
6. Periksa output di folder `output/`.

---

### Percobaan 9: Komposisi Transformasi
**File**: `09_komposisi_transformasi.py`

**Tujuan**: Memahami komposisi transformasi geometri menggunakan perkalian matriks homogen 3×3.

**Langkah Kerja**:
1. Buka dan pelajari file `09_komposisi_transformasi.py`.
2. Jalankan program: `python 09_komposisi_transformasi.py`.
3. Amati komposisi translasi + rotasi + scaling dalam satu matriks homogen.
4. Perhatikan perbedaan urutan transformasi (non-komutatif).
5. Bandingkan beberapa urutan transformasi dan visualisasikan hasilnya.
6. Periksa output di folder `output/`.

---

### Percobaan 10: Simulasi Distorsi Lensa
**File**: `10_distorsi_lensa.py`

**Tujuan**: Mensimulasi distorsi radial barrel dan pincushion serta distorsi tangensial pada gambar.

**Langkah Kerja**:
1. Buka dan pelajari file `10_distorsi_lensa.py`.
2. Jalankan program: `python 10_distorsi_lensa.py`.
3. Amati simulasi distorsi barrel (wide-angle) dan pincushion (telephoto).
4. Perhatikan distorsi tangensial dan pengaruhnya pada garis lurus.
5. Bandingkan berbagai tingkat koefisien distorsi radial k1.
6. Periksa output di folder `output/`.

---

### Percobaan 11: Koreksi Distorsi Kamera
**File**: `11_koreksi_distorsi.py`

**Tujuan**: Mengoreksi distorsi lensa menggunakan cv2.undistort dan initUndistortRectifyMap.

**Langkah Kerja**:
1. Buka dan pelajari file `11_koreksi_distorsi.py`.
2. Jalankan program: `python 11_koreksi_distorsi.py`.
3. Amati perbedaan cv2.undistort vs remap dua-langkah.
4. Perhatikan alpha parameter pada getOptimalNewCameraMatrix.
5. Visualisasikan vektor displacement distorsi menggunakan quiver plot.
6. Periksa output di folder `output/`.

---

### Percobaan 12: Sampling dan Aliasing
**File**: `12_sampling_dan_aliasing.py`

**Tujuan**: Memahami efek sampling, aliasing (pola moiré), dan teknik anti-aliasing pada gambar.

**Langkah Kerja**:
1. Buka dan pelajari file `12_sampling_dan_aliasing.py`.
2. Jalankan program: `python 12_sampling_dan_aliasing.py`.
3. Amati aliasing saat downscale tanpa filter vs dengan Gaussian blur.
4. Bandingkan INTER_NEAREST vs INTER_AREA pada downscaling.
5. Perhatikan zona plate dan pola moiré yang muncul.
6. Periksa output di folder `output/`.

---

### Percobaan 13: Interpolasi Gambar
**File**: `13_interpolasi_gambar.py`

**Tujuan**: Membandingkan lima metode interpolasi gambar (NEAREST, LINEAR, CUBIC, LANCZOS4, AREA) secara kualitas dan kecepatan.

**Langkah Kerja**:
1. Buka dan pelajari file `13_interpolasi_gambar.py`.
2. Jalankan program: `python 13_interpolasi_gambar.py`.
3. Bandingkan kualitas visual NEAREST, LINEAR, CUBIC, LANCZOS4, AREA.
4. Amati perbedaan interpolasi pada upscale vs downscale.
5. Ukur PSNR dan waktu eksekusi setiap metode.
6. Periksa output di folder `output/`.

---

### Percobaan 14: Image Pyramid
**File**: `14_image_pyramid.py`

**Tujuan**: Membangun Gaussian dan Laplacian pyramid untuk representasi multi-skala dan pyramid blending.

**Langkah Kerja**:
1. Buka dan pelajari file `14_image_pyramid.py`.
2. Jalankan program: `python 14_image_pyramid.py`.
3. Amati Gaussian pyramid (pyrDown) dan rekonstruksi (pyrUp).
4. Perhatikan Laplacian pyramid sebagai selisih antar-level.
5. Bandingkan pyramid blending vs penggabungan langsung (direct).
6. Periksa output di folder `output/`.

---

### Percobaan 15: Konversi Koordinat Polar
**File**: `15_koordinat_polar.py`

**Tujuan**: Memahami transformasi antara koordinat Cartesian dan polar (linear dan log-polar) menggunakan cv2.warpPolar.

**Langkah Kerja**:
1. Buka dan pelajari file `15_koordinat_polar.py`.
2. Jalankan program: `python 15_koordinat_polar.py`.
3. Amati transformasi linear polar dan log-polar.
4. Perhatikan demonstrasi rotasi = translasi vertikal dalam ruang log-polar.
5. Amati aplikasi polar unwrapping untuk iris sintetis.
6. Periksa output di folder `output/`.

---

### Percobaan 16: Remapping dan Efek Custom
**File**: `16_remapping_efek_custom.py`

**Tujuan**: Membuat transformasi gambar fleksibel menggunakan cv2.remap() dengan custom lookup map.

**Langkah Kerja**:
1. Buka dan pelajari file `16_remapping_efek_custom.py`.
2. Jalankan program: `python 16_remapping_efek_custom.py`.
3. Amati efek gelombang horizontal dan vertikal (wave distortion).
4. Perhatikan efek pusaran (swirl), kaca pembesar, dan fisheye.
5. Bandingkan kecepatan remap vs warpAffine/warpPerspective.
6. Periksa output di folder `output/`.

---

### Percobaan 17: Koreksi Gamma dan Power-Law
**File**: `17_gamma_correction.py`

**Tujuan**: Memahami gamma correction, transformasi power-law, dan pengaruhnya terhadap kecerahan gambar.

**Langkah Kerja**:
1. Buka dan pelajari file `17_gamma_correction.py`.
2. Jalankan program: `python 17_gamma_correction.py`.
3. Amati efek berbagai nilai gamma (0.25, 0.5, 1.0, 2.0, 3.0).
4. Perhatikan penggunaan LUT untuk koreksi gamma yang efisien.
5. Bandingkan histogram sebelum dan sesudah koreksi gamma.
6. Periksa output di folder `output/`.

---

### Percobaan 18: Transformasi Intensitas (Point Operations)
**File**: `18_transformasi_intensitas.py`

**Tujuan**: Mempelajari transformasi intensitas piksel (negatif, log, power-law, piecewise) dan ekualisasi histogram.

**Langkah Kerja**:
1. Buka dan pelajari file `18_transformasi_intensitas.py`.
2. Jalankan program: `python 18_transformasi_intensitas.py`.
3. Bandingkan transformasi negatif, log, power-law, dan piecewise linear.
4. Amati histogram equalization global vs CLAHE (adaptif).
5. Plot kurva transfer T(r) untuk setiap transformasi.
6. Periksa output di folder `output/`.

---

### Percobaan 19: Citra Sintetis untuk Pengujian Algoritma
**File**: `19_citra_sintetis.py`

**Tujuan**: Membuat pola gambar standar (checkerboard, Siemens star, zone plate, noise) sebagai test target algoritma.

**Langkah Kerja**:
1. Buka dan pelajari file `19_citra_sintetis.py`.
2. Jalankan program: `python 19_citra_sintetis.py`.
3. Amati checkerboard, Siemens star, dan zona plate yang dibuat secara matematis.
4. Bandingkan noise Gaussian vs salt-and-pepper pada berbagai intensitas.
5. Perhatikan slanted edge target untuk pengukuran MTF (resolusi).
6. Periksa output di folder `output/`.

---

### Percobaan 20: Deteksi dan Estimasi Pose Marker ArUco
**File**: `20_aruco_marker.py`

**Tujuan**: Mendeteksi marker ArUco, mendekode ID, dan mengestimasi pose kamera (Rvec, Tvec) untuk AR sederhana.

**Langkah Kerja**:
1. Buka dan pelajari file `20_aruco_marker.py`.
2. Jalankan program: `python 20_aruco_marker.py`.
3. Amati pembuatan marker ArUco DICT_4X4_50 dengan berbagai ID.
4. Perhatikan proses deteksi corners dan dekoding ID marker.
5. Amati overlay sumbu 3D (merah=X, hijau=Y, biru=Z) pada setiap marker.
6. Periksa output di folder `output/`.

---

## 5. ANALISIS

### Analisis Percobaan 1–2: Translasi dan Rotasi
- Jelaskan mengapa muncul area hitam setelah translasi dan rotasi.
- Apa solusi agar gambar tidak terpotong saat dirotasi 45°?
- Bagaimana center of rotation mempengaruhi hasil?

### Analisis Percobaan 3–4: Scaling dan Affine
- Bandingkan kualitas visual interpolasi nearest vs bilinear vs bicubic.
- Mengapa `INTER_AREA` lebih baik untuk downscaling?
- Berapa DOF transformasi affine dan mengapa perlu 3 pasang titik?

### Analisis Percobaan 5–6: Perspektif dan Kalibrasi
- Apa beda transformasi affine dan perspektif?
- Berapa reprojection error yang didapat? Apakah sudah baik?
- Jelaskan arti fisik setiap elemen matriks intrinsik K.

### Analisis Percobaan 7–8: Proyeksi 3D dan Distorsi
- Jelaskan hubungan antara focal length dan field of view.
- Jenis distorsi apa yang dominan pada kamera Anda (barrel/pincushion)?
- Mengapa koreksi distorsi penting untuk pengukuran akurat?

### Analisis Percobaan 9–10: Sampling dan Gamma
- Jelaskan teorema Nyquist dan hubungannya dengan aliasing.
- Mengapa gamma correction diperlukan?
- Bandingkan histogram sebelum dan sesudah gamma correction.

### Analisis Percobaan 11–12: Interpolasi dan Koordinat Polar
- Kapan sebaiknya menggunakan NEAREST vs CUBIC vs LANCZOS4?
- Jelaskan aplikasi konversi polar dalam computer vision (misalnya untuk iris recognition).
- Apa perbedaan linear polar dan log-polar?

### Analisis Percobaan 13–14: Shearing dan Refleksi
- Berikan contoh aplikasi shearing di dunia nyata.
- Bagaimana refleksi digunakan untuk data augmentation dalam deep learning?
- Apakah refleksi termasuk rigid transformation? Mengapa?

### Analisis Percobaan 15–16: Matriks Homogen dan Image Pyramid
- Mengapa urutan transformasi penting (non-komutatif)?
- Jelaskan perbedaan Gaussian dan Laplacian pyramid.
- Bagaimana pyramid blending mengatasi seam yang terlihat?

### Analisis Percobaan 17–18: Distorsi dan Remapping
- Bandingkan distorsi barrel dan pincushion secara visual.
- Apa keunggulan `cv2.remap` dibandingkan `cv2.warpAffine`?
- Sebutkan 3 efek kreatif yang bisa dibuat dengan remap.

### Analisis Percobaan 19–20: Transformasi Intensitas dan Citra Sintetis
- Kapan transformasi log lebih cocok digunakan daripada gamma?
- Mengapa citra sintetis penting untuk pengujian algoritma?
- Jenis noise apa yang paling sulit dihilangkan? Mengapa?

---

## 6. KESIMPULAN

Buatlah kesimpulan yang mencakup:
1. Pemahaman tentang proses pembentukan citra dari scene 3D ke gambar 2D.
2. Perbedaan dan hierarki transformasi geometri (translasi → affine → perspektif).
3. Pentingnya kalibrasi kamera dan koreksi distorsi untuk aplikasi presisi.
4. Hubungan antara fotometri, gamma, dan kualitas visual gambar.
5. Konsep sampling, aliasing, dan solusi anti-aliasing.
6. Peran interpolasi, image pyramid, dan remapping dalam pengolahan citra.
7. Manfaat transformasi intensitas (log, power) untuk manipulasi kontras.
8. Pentingnya citra sintetis untuk pengujian dan validasi algoritma.
