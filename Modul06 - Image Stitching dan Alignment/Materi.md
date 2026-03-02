# MATERI MODUL 6: IMAGE STITCHING DAN ALIGNMENT

---

## 1. Pendahuluan

Image stitching adalah proses menggabungkan beberapa gambar yang overlapping menjadi satu gambar panorama. Area ini menggabungkan konsep dari feature detection, geometric transformation, blending, dan optimasi global untuk menghasilkan panorama yang seamless.

**Referensi utama**: Szeliski, *Computer Vision: Algorithms and Applications*, 2nd Edition, **Chapter 8 — Image Alignment and Stitching**.

---

## 2. Pipeline Image Stitching

### 2.1 Tahapan Umum
1. **Feature Detection & Matching** — Temukan korespondensi antar gambar.
2. **Homography Estimation** — Estimasi transformasi geometris.
3. **Image Warping** — Transformasi gambar ke koordinat bersama.
4. **Blending** — Gabungkan gambar yang di-warp agar seamless.

### 2.2 Motion Models
- **Translation** (2 DOF): $\mathbf{x}' = \mathbf{x} + \mathbf{t}$
- **Affine** (6 DOF): $\mathbf{x}' = A\mathbf{x} + \mathbf{t}$
- **Homography/Projective** (8 DOF): $\tilde{\mathbf{x}}' = H\tilde{\mathbf{x}}$

Untuk panorama dari kamera yang berotasi di satu titik, **homography** adalah model yang tepat.

---

## 3. Feature-based Alignment

### 3.1 Proses
1. Deteksi fitur (SIFT, ORB, AKAZE) pada semua gambar.
2. Match fitur antar pasangan gambar.
3. Estimasi homography per pasangan menggunakan RANSAC.
4. Verifikasi: cek jumlah inlier dan konsistensi geometris.

### 3.2 Image Registration
$$
H_{i \to ref} = H_{i \to i+1} \cdot H_{i+1 \to i+2} \cdot \ldots \cdot H_{n-1 \to ref}
$$

Semua gambar ditransformasi ke koordinat gambar referensi (biasanya gambar tengah).

---

## 4. Image Warping

### 4.1 Forward Warping vs Inverse Warping
- **Forward**: Untuk setiap piksel sumber, petakan ke destinasi → bisa ada gaps.
- **Inverse**: Untuk setiap piksel destinasi, cari sumber → lebih baik, tanpa gaps.

OpenCV menggunakan inverse warping:
```python
result = cv2.warpPerspective(img, H, (width, height))
```

### 4.2 Proyeksi untuk Panorama

#### Planar Projection
- Langsung gunakan homography.
- Cocok untuk FOV kecil (< 90°).
- Distorsi besar pada tepi untuk FOV lebar.

#### Cylindrical Projection
Gambar dipetakan ke permukaan silinder:
$$
x' = f \cdot \tan^{-1}\left(\frac{x - c_x}{f}\right), \quad y' = f \cdot \frac{y - c_y}{\sqrt{(x-c_x)^2 + f^2}}
$$

- Cocok untuk panorama horizontal 360°.
- Mengurangi distorsi pada tepi.

#### Spherical Projection
Gambar dipetakan ke permukaan bola:
$$
\theta = \tan^{-1}\left(\frac{x - c_x}{f}\right), \quad \phi = \tan^{-1}\left(\frac{y - c_y}{\sqrt{(x-c_x)^2 + f^2}}\right)
$$

- Paling akurat untuk full 360° panorama.

```python
def cylindrical_warp(img, f):
    h, w = img.shape[:2]
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    # ... cylindrical mapping
```

---

## 5. Bundle Adjustment

### 5.1 Motivasi
Chain homography ($H_{12} \cdot H_{23} \cdot \ldots$) mengakumulasi error. Bundle adjustment mengoptimalkan SEMUA parameter kamera secara simultan.

### 5.2 Formulasi
Minimasi total reprojection error:

$$
\min_{\{R_i, f_i\}} \sum_{i,j} \sum_{k} d(x_{ij}^k, \pi(R_i, f_i, X_j^k))^2
$$

di mana $R_i, f_i$ adalah rotasi dan focal length kamera ke-$i$, dan $X_j^k$ adalah titik 3D.

### 5.3 OpenCV Stitcher
`cv2.Stitcher` mengimplementasikan bundle adjustment secara otomatis:
```python
stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA)
status, panorama = stitcher.stitch(images)
```

---

## 6. Image Blending

### 6.1 Masalah Seam
Perbedaan exposure, white balance, dan vignetting menyebabkan seam yang terlihat di batas antar gambar.

### 6.2 Teknik Blending

#### Feather Blending (Alpha Blending)
Linearly blend di area overlap berdasarkan jarak dari tepi:
$$
I_{blend}(x,y) = \alpha(x,y) \cdot I_1(x,y) + (1 - \alpha(x,y)) \cdot I_2(x,y)
$$

#### Multi-band Blending (Laplacian Pyramid)
Gabungkan frekuensi rendah dari satu gambar dengan frekuensi tinggi dari gambar lain:
1. Buat Laplacian pyramid untuk setiap gambar.
2. Buat Gaussian pyramid untuk mask.
3. Blend per level pyramid.
4. Rekonstruksi dari blended pyramid.

```python
# Multi-band blending concept
for level in range(num_levels):
    blended_lap = mask_gauss[level] * lap1[level] + (1 - mask_gauss[level]) * lap2[level]
```

#### Gradient-domain Blending (Poisson)
- Minimasi gradien difference alih-alih intensitas.
- Menghasilkan transisi paling halus.

---

## 7. Exposure Compensation

### 7.1 Gain Compensation
Menyesuaikan gain (brightness) setiap gambar agar konsisten:
$$
I'_i = g_i \cdot I_i
$$

Gain $g_i$ diestimasi dari area overlapping.

### 7.2 Block-based Compensation
Bagi overlap area menjadi blocks, sesuaikan gain per block untuk variasi lokal.

```python
compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
```

---

## 8. Seam Finding

### 6.1 Pendekatan
Temukan garis seam optimal di area overlap yang meminimalkan perbedaan visual.

### 6.2 Metode
- **Voronoi**: Batas di tengah overlap (sederhana tapi tidak optimal).
- **Graph Cut**: Minimasi energy function yang mempertimbangkan perbedaan warna di seam.
- **Dynamic Programming**: Cari path minimum cost dari atas ke bawah.

$$
E_{seam} = \sum_{(p,q) \in seam} |I_1(p) - I_2(p)| + |I_1(q) - I_2(q)|
$$

```python
seam_finder = cv2.detail.GraphCutSeamFinder('COST_COLOR')
```

---

## 9. Panorama Photography Tips

### 9.1 Pengambilan Gambar
- Overlap 30–50% antar gambar.
- Rotasi dari satu titik (gunakan tripod jika mungkin).
- Exposure manual dan white balance manual → konsisten.
- Avoid moving objects di area overlap.

### 9.2 Panorama 360°
- Minimal 8–12 gambar untuk 360° horizontal.
- Cylindrical/spherical projection wajib.
- Bundle adjustment penting untuk menutup loop (gambar pertama dan terakhir).

---

## 10. Auto-Cropping Panorama

### 10.1 Masalah Border Hitam
Hasil stitching selalu memiliki area hitam (piksel tanpa data) karena warping menghasilkan bentuk non-rectangular. Border ini harus dihilangkan untuk panorama yang presentable.

### 10.2 Teknik Auto-Crop

#### Threshold + Bounding Rectangle
1. Konversi panorama ke grayscale.
2. Threshold untuk membuat binary mask (konten = putih, border = hitam).
3. Temukan kontur terbesar → `cv2.boundingRect()` → crop.

#### Maximum Inscribed Rectangle
Cari rectangle terbesar yang sepenuhnya berada di dalam area konten (tanpa piksel hitam):
- Pendekatan berbasis dynamic programming.
- Menghasilkan crop yang lebih ketat tapi tanpa border hitam sama sekali.

#### Morfologi untuk Refinement
Operasi closing (`cv2.morphologyEx`) menutup gap kecil pada mask sebelum kontur detection, meningkatkan akurasi cropping.

```python
gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
cropped = panorama[y:y+h, x:x+w]
```

---

## 11. Homography: Dekomposisi dan Analisis

### 11.1 Dekomposisi Homography
Homography $H$ dapat didekomposisi menjadi komponen rotasi $R$, translasi $\mathbf{t}$, dan normal bidang $\mathbf{n}$:

$$
H = R + \mathbf{t} \cdot \mathbf{n}^T
$$

OpenCV menyediakan `cv2.decomposeHomographyMat(H, K)` yang mengembalikan hingga 4 solusi.

### 11.2 Reprojection Error
Kualitas homography diukur dengan reprojection error:
$$
e = \frac{1}{N} \sum_{i=1}^{N} \|x'_i - H \cdot x_i\|^2
$$

Error rendah menunjukkan estimasi yang akurat. Distribusi error membantu mengidentifikasi outlier.

### 11.3 Analisis Outlier
- Inlier: titik dengan error < threshold (biasanya 3–5 piksel).
- RANSAC secara iteratif memilih subset random, estimasi $H$, dan menghitung inlier.
- Rasio inlier/total menunjukkan kualitas matches.

---

## 12. Image Registration (ECC Algorithm)

### 12.1 Konsep
Image registration menyelaraskan dua gambar menggunakan optimasi intensitas piksel, bukan fitur diskrit. ECC (Enhanced Correlation Coefficient) memaksimalkan korelasi antara template dan warped image.

### 12.2 Model Transformasi
ECC mendukung beberapa motion model:
- **Translation** (2 DOF): Pergeseran $dx, dy$.
- **Euclidean** (3 DOF): Translasi + rotasi.
- **Affine** (6 DOF): Termasuk scaling dan shearing.
- **Homography** (8 DOF): Transformasi perspektif penuh.

### 12.3 Implementasi
```python
warp_mode = cv2.MOTION_EUCLIDEAN
warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
cc, warp_matrix = cv2.findTransformECC(template_gray, target_gray, warp_matrix, warp_mode, criteria)
```

### 12.4 ECC vs Feature-based
| Aspek | ECC | Feature-based |
|-------|-----|---------------|
| Input | Piksel (intensitas) | Keypoints + descriptors |
| Akurasi | Sub-pixel | Bergantung pada fitur |
| Kecepatan | Lambat (iteratif) | Cepat |
| Robustness | Sensitif inisialisasi | Robust terhadap perubahan besar |
| Use case | Perubahan kecil | Perubahan perspektif besar |

---

## 13. Laplacian Pyramid Blending (Detail)

### 13.1 Gaussian Pyramid
Gaussian pyramid dibangun dengan repeated downsampling:
$$
G_l = \text{reduce}(G_{l-1}) = \text{downsample}(\text{blur}(G_{l-1}))
$$

### 13.2 Laplacian Pyramid
Laplacian pyramid menyimpan detail (perbedaan antara level Gaussian):
$$
L_l = G_l - \text{expand}(G_{l+1})
$$

Setiap level menyimpan band frekuensi tertentu: level rendah = frekuensi tinggi (detail), level tinggi = frekuensi rendah (struktur).

### 13.3 Proses Blending
1. Bangun Laplacian pyramid $LA$ dan $LB$ dari gambar A dan B.
2. Bangun Gaussian pyramid $GM$ dari mask.
3. Blend per level: $LC_l = GM_l \cdot LA_l + (1 - GM_l) \cdot LB_l$.
4. Rekonstruksi: $R_l = LC_l + \text{expand}(R_{l+1})$.

### 13.4 Keunggulan
- Frekuensi rendah (warna, brightness) diblend secara luas → transisi halus.
- Frekuensi tinggi (tepi, tekstur) diblend secara lokal → detail tajam.
- Hasil lebih alami dibanding alpha blending sederhana.

---

## 14. Metrik Kualitas Stitching

### 14.1 PSNR (Peak Signal-to-Noise Ratio)
$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)
$$

PSNR tinggi menunjukkan perbedaan kecil. Untuk area overlap, PSNR > 30 dB dianggap baik.

### 14.2 SSIM (Structural Similarity Index)
$$
\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

SSIM mengukur kemiripan struktural (luminance, contrast, structure). Rentang [0, 1], dimana 1 = identik.

### 14.3 Difference Map
Visualisasi absolute difference antara area overlap sebelum dan sesudah blending:
```python
diff = cv2.absdiff(overlap1, overlap2)
heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
```

### 14.4 Edge Alignment
Mengukur kesinambungan tepi di seam line menggunakan Canny edge detection. Tepi yang terputus di seam menunjukkan alignment buruk.

---

## 15. Loop Closure dalam Panorama

### 15.1 Masalah Drift
Pada panorama 360°, chain homography mengakumulasi error:
$$
H_{1 \to N} = H_{(N-1) \to N} \cdot \ldots \cdot H_{2 \to 3} \cdot H_{1 \to 2}
$$

Idealnya, setelah lingkaran penuh kembali ke gambar 1, transformasi akumulasi harus identitas. Perbedaan dari identitas = drift.

### 15.2 Deteksi Loop
Feature matching antara gambar pertama dan terakhir mendeteksi loop. Jika ditemukan cukup banyak matches, loop closure dapat dilakukan.

### 15.3 Distribusi Error
Drift didistribusikan secara merata ke setiap pasangan homography menggunakan interpolasi:
$$
H'_i = H_i \cdot \Delta H_i, \quad \Delta H_i = H_{drift}^{i/N}
$$

Pendekatan ini analog dengan bundle adjustment sederhana khusus untuk loop.

---

## 16. Document Stitching

### 16.1 Perbedaan dengan Panorama Alam
- Dokumen memerlukan perspektif yang benar (rectangular).
- Transformasi biasanya translasi + sedikit rotasi (bukan homography penuh).
- Stitching biasanya vertikal (halaman panjang) atau horizontal (whiteboard lebar).

### 16.2 Pipeline Document Stitching
1. **Perspective Correction**: 4-point transform untuk membuat dokumen rectangular.
2. **Enhancement**: Contrast enhancement, sharpening.
3. **Feature Matching**: Pada bagian overlap teks/konten.
4. **Alignment**: Estimasi translasi vertikal/horizontal.
5. **Stacking**: Gabungkan secara vertikal atau horizontal.
6. **Post-processing**: Binarization, cropping, cleanup.

### 16.3 Perspective Correction
```python
pts = np.float32([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
dst = np.float32([[0,0], [w,0], [w,h], [0,h]])
M = cv2.getPerspectiveTransform(pts, dst)
warped = cv2.warpPerspective(img, M, (w, h))
```

---

## 16b. Phase Correlation

### Konsep
Phase correlation mengestimasi translasi antar dua gambar menggunakan cross-power spectrum di domain frekuensi. Lebih robust terhadap noise dan perubahan illuminasi dibanding metode spatial.

### Prinsip Kerja
$$\delta = \arg\max \mathcal{F}^{-1}\left(\frac{F_1 \cdot F_2^*}{|F_1 \cdot F_2^*|}\right)$$

Peak pada inverse FFT menunjukkan translasi (dx, dy) antar gambar.

### Implementasi OpenCV
```python
shift, response = cv2.phaseCorrelate(img1_float, img2_float)
# shift = (dx, dy), response = confidence (0-1)
```

### Keunggulan vs Feature-based
| Aspek | Phase Correlation | Feature-based |
|-------|-------------------|---------------|
| Kecepatan | Sangat cepat (FFT) | Lambat (detect+match) |
| Transformasi | Hanya translasi | Homography penuh |
| Robustness | Noise tolerant | Outlier handling (RANSAC) |
| Texture requirement | Butuh tekstur global | Butuh fitur lokal |

### Aplikasi
- Video stabilization (estimasi translasi antar frame).
- Panorama stitching untuk gerakan kamera murni translasi.
- Image registration cepat untuk medical imaging.

---

## 16c. Wave Correction

### Konsep
Wave correction memperbaiki distorsi gelombang pada panorama yang dihasilkan oleh rotasi kamera yang tidak sempurna (tilting). Efek ini terlihat sebagai garis horizon yang bergelombang.

### Implementasi di Stitcher
```python
stitcher = cv2.Stitcher_create()
# Wave correction diaktifkan secara default
# Untuk kontrol manual:
stitcher.setWaveCorrection(True)  # atau False
stitcher.setWaveCorrectKind(cv2.detail.WAVE_CORRECT_HORIZ)  # horizontal correction
```

### Tipe Wave Correction
- **WAVE_CORRECT_HORIZ**: Koreksi untuk panorama horizontal (paling umum).
- **WAVE_CORRECT_VERT**: Koreksi untuk panorama vertikal.

---

## 17. Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed., Chapter 8.
2. Brown, M. & Lowe, D. (2007). *Automatic Panoramic Image Stitching using Invariant Features*. IJCV.
3. Burt, P. & Adelson, E. (1983). *A Multiresolution Spline with Application to Image Mosaics*. ACM TOG.
4. Agarwala, A. et al. (2004). *Interactive Digital Photomontage*. SIGGRAPH.
5. Szeliski, R. (2006). *Image Alignment and Stitching: A Tutorial*. Foundations and Trends in Computer Graphics and Vision.
6. Kwatra, V. et al. (2003). *Graphcut Textures: Image and Video Synthesis Using Graph Cuts*. SIGGRAPH.
7. Pérez, P. et al. (2003). *Poisson Image Editing*. SIGGRAPH.
8. Shum, H. & Szeliski, R. (2000). *Construction of Panoramic Image Mosaics with Global and Local Alignment*. IJCV.
