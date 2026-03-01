# MODUL 2: PEMBENTUKAN CITRA (IMAGE FORMATION)

## Referensi Utama
**Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. Chapter 2: Image Formation.**

---

## 2.1 Pendahuluan Pembentukan Citra

Pembentukan citra (*image formation*) adalah proses dimana scene 3D di dunia nyata diproyeksikan menjadi gambar 2D oleh sebuah kamera. Memahami proses ini sangat krusial karena seluruh algoritma computer vision bergantung pada asumsi tentang bagaimana gambar terbentuk.

Menurut Szeliski (2022, Ch. 2):
> "Understanding how images are formed and how they relate to the 3D world is fundamental to computer vision."

---

## 2.2 Primitif Geometri dan Transformasi 2D

### Titik, Garis, dan Bidang
- **Titik 2D**: $\mathbf{x} = (x, y)$ dalam koordinat Cartesian, atau $\tilde{\mathbf{x}} = (x, y, 1)$ dalam koordinat homogen.
- **Garis 2D**: $\tilde{\mathbf{l}} = (a, b, c)$ sehingga $ax + by + c = 0$.
- **Titik 3D**: $\mathbf{X} = (X, Y, Z)$ atau $\tilde{\mathbf{X}} = (X, Y, Z, 1)$ dalam koordinat homogen.

### Koordinat Homogen
Koordinat homogen menambahkan satu dimensi ekstra untuk memungkinkan representasi transformasi proyektif sebagai perkalian matriks:
$$\tilde{\mathbf{x}} = \begin{pmatrix} x \\ y \\ w \end{pmatrix} \quad \Rightarrow \quad \mathbf{x} = \left(\frac{x}{w}, \frac{y}{w}\right)$$

### Hierarki Transformasi 2D

| Transformasi | DOF | Matriks | Preserve |
|-------------|-----|---------|----------|
| **Translasi** | 2 | $\begin{pmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{pmatrix}$ | Orientasi, panjang, sudut |
| **Rotasi** | 1 | $\begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$ | Panjang, sudut |
| **Rigid (Euclidean)** | 3 | Rotasi + Translasi | Panjang, sudut |
| **Similarity** | 4 | $s \cdot$ Rotasi + Translasi | Sudut, rasio |
| **Affine** | 6 | $\begin{pmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{pmatrix}$ | Paralelisme |
| **Projective (Homography)** | 8 | Matriks 3×3, 8 DOF | Garis lurus |

### Translasi
Pergeseran posisi gambar:
```python
M = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img, M, (w, h))
```

### Rotasi
Putar gambar di sekitar titik pusat:
```python
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(img, M, (w, h))
```

### Scaling
Perubahan ukuran gambar dengan interpolasi:
```python
resized = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
```

### Shearing
Transformasi geser menggunakan matriks affine:
$$M_{shear} = \begin{pmatrix} 1 & s_x & 0 \\ s_y & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

### Refleksi
Pencerminan gambar terhadap sumbu:
```python
flipped_h = cv2.flip(img, 1)   # Horizontal
flipped_v = cv2.flip(img, 0)   # Vertikal
```

### Transformasi Affine
Memerlukan 3 pasang titik korespondensi:
```python
M = cv2.getAffineTransform(src_pts, dst_pts)
result = cv2.warpAffine(img, M, (w, h))
```

### Transformasi Perspektif (Homography)
Memerlukan 4 pasang titik korespondensi:
```python
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
result = cv2.warpPerspective(img, M, (w, h))
```

### Komposisi Transformasi
Transformasi dapat dikomposisikan melalui perkalian matriks homogen:
$$M_{total} = M_n \cdot M_{n-1} \cdots M_2 \cdot M_1$$

**Penting**: Urutan perkalian matriks transformasi **tidak komutatif** — $M_1 \cdot M_2 \neq M_2 \cdot M_1$.

---

## 2.3 Transformasi 3D

### Rotasi 3D
Rotasi 3D dapat direpresentasikan dengan:
1. **Matriks Rotasi** $\mathbf{R}$ (3×3, ortogonal, det = 1).
2. **Euler Angles**: Roll, Pitch, Yaw ($\phi, \theta, \psi$).
3. **Axis-Angle**: Sumbu $\hat{\mathbf{n}}$ dan sudut $\theta$.
4. **Quaternion**: $\mathbf{q} = (q_w, q_x, q_y, q_z)$ — menghindari gimbal lock.

Matriks rotasi dasar:
$$R_z(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

### Rigid Body Transformation
$$\mathbf{X'} = \mathbf{R}\mathbf{X} + \mathbf{t}$$

Dalam koordinat homogen:
$$\tilde{\mathbf{X'}} = \begin{pmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix} \tilde{\mathbf{X}}$$

---

## 2.4 Model Kamera

### Pinhole Camera Model
Model kamera paling sederhana — cahaya melewati satu titik (aperture) dan memproyeksikan scene ke bidang gambar.

$$\begin{pmatrix} x \\ y \\ 1 \end{pmatrix} = \frac{1}{Z} \begin{pmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} X \\ Y \\ Z \end{pmatrix}$$

### Matriks Intrinsik Kamera
$$\mathbf{K} = \begin{pmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$$

- $f_x, f_y$: Focal length dalam piksel.
- $c_x, c_y$: Principal point (pusat optik).
- $s$: Skew (biasanya 0).

### Proyeksi Perspektif
Proyeksi titik 3D ke 2D:
$$\mathbf{x} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \mathbf{X}$$

di mana $[\mathbf{R} | \mathbf{t}]$ adalah matriks ekstrinsik (pose kamera).

---

## 2.5 Distorsi Lensa

### Distorsi Radial
$$x_{distorted} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_{distorted} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

- **Barrel distortion**: $k_1 < 0$ — garis melengkung ke luar (lensa wide-angle).
- **Pincushion distortion**: $k_1 > 0$ — garis melengkung ke dalam (lensa telephoto).

### Distorsi Tangensial
$$x_{distorted} = x + [2p_1 xy + p_2(r^2 + 2x^2)]$$
$$y_{distorted} = y + [p_1(r^2 + 2y^2) + 2p_2 xy]$$

### Koreksi Distorsi
```python
undistorted = cv2.undistort(img, K, distCoeffs)
# Atau menggunakan remap:
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_32FC1)
result = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
```

---

## 2.6 Kalibrasi Kamera

### Proses Kalibrasi
1. Siapkan pattern kalibrasi (checkerboard).
2. Ambil beberapa foto pattern dari berbagai sudut.
3. Deteksi corner: `cv2.findChessboardCorners()`.
4. Refine corner: `cv2.cornerSubPix()`.
5. Kalibrasi: `cv2.calibrateCamera()`.

### Hasil Kalibrasi
- Matriks intrinsik $\mathbf{K}$.
- Koefisien distorsi $(k_1, k_2, p_1, p_2, k_3)$.
- Rotation dan translation vectors per gambar.
- Reprojection error (idealnya < 0.5 piksel).

---

## 2.7 Photometry dan Iluminasi

### Model Pencahayaan
Intensitas gambar dipengaruhi oleh:
1. **Sumber cahaya** — posisi, intensitas, arah.
2. **Refleksi permukaan** — BRDF (Bidirectional Reflectance Distribution Function).
3. **Geometri** — orientasi permukaan relatif terhadap cahaya dan kamera.

### Lambertian Reflectance
$$I = I_{\text{light}} \cdot k_d \cdot \max(0, \mathbf{n} \cdot \mathbf{l})$$

### Gamma Correction
Sensor kamera dan display tidak linear. Koreksi gamma:
$$I_{corrected} = I^{1/\gamma}$$

Umumnya $\gamma = 2.2$ untuk standar sRGB. Implementasi efisien menggunakan LUT:
```python
table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype('uint8')
corrected = cv2.LUT(img, table)
```

### Transformasi Log dan Power-law
**Transformasi Logaritmik** — memperluas range gelap, mengompresi range terang:
$$s = c \cdot \log(1 + r)$$

**Transformasi Power-law (Gamma)** — kontrol kontras fleksibel:
$$s = c \cdot r^{\gamma}$$

- $\gamma < 1$: Mencerahkan (ekspansi range gelap).
- $\gamma > 1$: Menggelapkan (ekspansi range terang).

---

## 2.8 Sampling dan Aliasing

### Teorema Nyquist-Shannon
Untuk merekonstruksi sinyal tanpa aliasing, frekuensi sampling harus minimal 2× frekuensi tertinggi sinyal:
$$f_s \geq 2 \cdot f_{max}$$

### Aliasing pada Gambar
- Muncul sebagai pola moiré, jagged edges, atau artefak saat downsampling tanpa filter anti-aliasing.
- **Solusi**: Terapkan low-pass filter (Gaussian blur) sebelum downsampling.

```python
blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
downscaled = cv2.resize(blurred, (w//2, h//2), interpolation=cv2.INTER_AREA)
```

---

## 2.9 Interpolasi Gambar

Interpolasi diperlukan saat gambar di-resize, dirotasi, atau ditransformasi:

| Metode | Deskripsi | Kecepatan | Kualitas |
|--------|-----------|-----------|----------|
| **Nearest Neighbor** | Ambil piksel terdekat | Tercepat | Rendah (blocky) |
| **Bilinear** | Rata-rata 4 piksel terdekat | Cepat | Baik |
| **Bicubic** | Konvolusi 4×4 piksel | Sedang | Sangat baik |
| **Lanczos** | Konvolusi 8×8 piksel | Lambat | Terbaik |
| **Area** | Rata-rata piksel area | Cepat | Terbaik untuk downscale |

```python
cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
```

---

## 2.10 Image Pyramid

### Gaussian Pyramid
Representasi multi-skala: setiap level di-blur dan di-downsample 2×:
```python
lower = cv2.pyrDown(img)    # Downscale
higher = cv2.pyrUp(lower)   # Upscale (bukan inverse!)
```

### Laplacian Pyramid
Menyimpan detail (perbedaan antar level Gaussian):
$$L_i = G_i - \text{expand}(G_{i+1})$$

Digunakan untuk:
- **Image blending** tanpa seam yang terlihat.
- **Kompresi** gambar.
- **Multi-scale analysis**.

---

## 2.11 Konversi Koordinat Polar

Konversi dari Cartesian $(x, y)$ ke polar $(r, \theta)$:
$$r = \sqrt{x^2 + y^2}, \quad \theta = \arctan\left(\frac{y}{x}\right)$$

```python
polar = cv2.linearPolar(img, center, maxRadius, cv2.WARP_FILL_OUTLIERS)
log_polar = cv2.logPolar(img, center, M, cv2.WARP_FILL_OUTLIERS)
```

**Aplikasi**: Iris recognition, analisis objek radial, rotation-invariant matching.

---

## 2.12 Remapping

`cv2.remap` memungkinkan transformasi gambar fleksibel menggunakan custom map:
```python
dst = cv2.remap(src, map_x, map_y, interpolation)
```

Dimana `map_x[y,x]` dan `map_y[y,x]` menentukan dari mana setiap piksel output diambil. Digunakan untuk:
- Koreksi distorsi lensa
- Efek artistik (wave, swirl, fisheye)
- Transformasi non-linear kustom

---

## 2.13 Pembuatan Citra Sintetis

Citra sintetis berguna untuk pengujian algoritma karena memiliki ground truth yang diketahui:

### Jenis Citra Sintetis
- **Gradien** — linear, radial
- **Pola geometri** — checkerboard, grid, Siemens star
- **Pola frekuensi** — zona plate, sinusoidal grating
- **Noise** — Gaussian, salt-and-pepper, Poisson, speckle

```python
# Gaussian noise
noise = np.random.normal(0, sigma, img.shape).astype(np.uint8)
noisy = cv2.add(img, noise)
```

---

## 2.14 Ringkasan

| Konsep | Penjelasan |
|--------|------------|
| Image Formation | Proyeksi scene 3D ke gambar 2D |
| Koordinat Homogen | Representasi untuk transformasi proyektif |
| Translasi, Rotasi, Scaling | Transformasi geometri dasar |
| Shearing & Refleksi | Transformasi affine khusus |
| Transformasi Affine | 6 DOF, mempertahankan paralelisme |
| Homography | 8 DOF, transformasi perspektif |
| Komposisi Transformasi | Perkalian matriks (non-komutatif) |
| Pinhole Model | Model kamera ideal tanpa lensa |
| Matriks Intrinsik | Parameter kamera: focal length, principal point |
| Distorsi Lensa | Barrel, pincushion — koreksi dengan kalibrasi |
| Kalibrasi Kamera | Estimasi parameter intrinsik dan ekstrinsik |
| Gamma Correction | Koreksi non-linearitas sensor/display |
| Log/Power Transform | Manipulasi kontras non-linear |
| Sampling/Aliasing | Artefak under-sampling, solusi: anti-aliasing |
| Interpolasi | Nearest, bilinear, bicubic, Lanczos |
| Image Pyramid | Representasi multi-skala (Gaussian, Laplacian) |
| Koordinat Polar | Konversi Cartesian ↔ polar/log-polar |
| Remapping | Transformasi fleksibel via custom map |
| Citra Sintetis | Gambar buatan untuk pengujian algoritma |

---

## Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition, Springer. **Chapter 2: Image Formation** (pp. 29–102).
2. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd Edition, Cambridge University Press.
3. OpenCV Camera Calibration Documentation. https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
4. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing*, 4th Edition, Pearson. Chapter 2.
5. Forsyth, D. A., & Ponce, J. (2012). *Computer Vision: A Modern Approach*, 2nd Edition, Pearson.
