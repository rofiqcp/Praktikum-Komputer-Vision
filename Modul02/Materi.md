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

### Transformasi 2D

| Transformasi | DOF | Matriks | Preserve |
|-------------|-----|---------|----------|
| **Translasi** | 2 | $\begin{pmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{pmatrix}$ | Orientasi, panjang, sudut |
| **Rotasi** | 1 | $\begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$ | Panjang, sudut |
| **Rigid (Euclidean)** | 3 | Rotasi + Translasi | Panjang, sudut |
| **Similarity** | 4 | $s \cdot$ Rotasi + Translasi | Sudut, rasio |
| **Affine** | 6 | $\begin{pmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{pmatrix}$ | Paralelisme |
| **Projective (Homography)** | 8 | Matriks 3×3, 8 DOF | Garis lurus |

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
Transformasi rigid (rotation + translation):
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

Lensa nyata menyebabkan distorsi pada gambar:

### Distorsi Radial
$$x_{distorted} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_{distorted} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

di mana $r^2 = x^2 + y^2$.

- **Barrel distortion**: $k_1 < 0$ — garis melengkung ke luar.
- **Pincushion distortion**: $k_1 > 0$ — garis melengkung ke dalam.

### Distorsi Tangensial
$$x_{distorted} = x + [2p_1 xy + p_2(r^2 + 2x^2)]$$
$$y_{distorted} = y + [p_1(r^2 + 2y^2) + 2p_2 xy]$$

### Koreksi Distorsi
OpenCV: `cv2.undistort(img, K, distCoeffs)` atau `cv2.initUndistortRectifyMap()`.

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
- Reprojection error (semakin kecil semakin baik, idealnya < 0.5 piksel).

---

## 2.7 Photometry dan Iluminasi

### Model Pencahayaan
Intensitas gambar dipengaruhi oleh:
1. **Sumber cahaya** — posisi, intensitas, arah.
2. **Refleksi permukaan** — BRDF (Bidirectional Reflectance Distribution Function).
3. **Geometri** — orientasi permukaan relatif terhadap cahaya dan kamera.

### Lambertian Reflectance
$$I = I_{\text{light}} \cdot k_d \cdot \max(0, \mathbf{n} \cdot \mathbf{l})$$

- $I_{\text{light}}$: Intensitas sumber cahaya.
- $k_d$: Koefisien diffuse reflectance.
- $\mathbf{n}$: Normal permukaan.
- $\mathbf{l}$: Arah cahaya.

### Gamma Correction
Sensor kamera dan display tidak linear. Koreksi gamma:
$$I_{corrected} = I^{1/\gamma}$$

Umumnya $\gamma = 2.2$ untuk standar sRGB.

---

## 2.8 Sampling dan Aliasing

### Teorema Nyquist-Shannon
Untuk merekonstruksi sinyal tanpa aliasing, frekuensi sampling harus minimal 2× frekuensi tertinggi sinyal:
$$f_s \geq 2 \cdot f_{max}$$

### Aliasing pada Gambar
- Muncul sebagai pola moiré, jagged edges, atau artefak saat gambar didownsample tanpa filter anti-aliasing.
- **Solusi**: Terapkan low-pass filter (Gaussian blur) sebelum downsampling.

### Anti-aliasing
```python
# Downscale dengan anti-aliasing
blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
downscaled = cv2.resize(blurred, (w//2, h//2), interpolation=cv2.INTER_AREA)
```

---

## 2.9 Ruang Warna dan Pembentukan Warna

### Color Filter Array (CFA)
- Kebanyakan sensor kamera menggunakan **Bayer filter** — pola RGGB.
- Setiap piksel sensor hanya merekam satu warna.
- **Demosaicing** menginterpolasi warna yang hilang.

### White Balance
Menyesuaikan warna agar objek putih terlihat putih di berbagai kondisi pencahayaan:
$$R_{corrected} = R \cdot \frac{G_{avg}}{R_{avg}}, \quad B_{corrected} = B \cdot \frac{G_{avg}}{B_{avg}}$$

---

## 2.10 Artefak Kompresi

### JPEG Compression
1. Konversi RGB → YCbCr.
2. Block splitting (8×8).
3. DCT (Discrete Cosine Transform).
4. Quantization (lossy step).
5. Entropy coding (Huffman).

Artefak umum: blocking, ringing, blurring pada kualitas rendah.

---

## 2.11 Ringkasan

| Konsep | Penjelasan |
|--------|------------|
| Image Formation | Proyeksi scene 3D ke gambar 2D |
| Koordinat Homogen | Representasi yang memungkinkan transformasi proyektif |
| Transformasi 2D | Translasi, rotasi, affine, homography |
| Pinhole Model | Model kamera ideal tanpa lensa |
| Matriks Intrinsik | Parameter internal kamera (focal length, principal point) |
| Distorsi Lensa | Barrel, pincushion — koreksi dengan koefisien distorsi |
| Kalibrasi Kamera | Estimasi parameter intrinsik dan ekstrinsik |
| Photometry | Hubungan antara cahaya, permukaan, dan intensitas gambar |
| Gamma Correction | Koreksi non-linearitas sensor/display |
| Sampling/Aliasing | Artefak akibat under-sampling, solusi: anti-aliasing |

---

## Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition, Springer. **Chapter 2: Image Formation** (pp. 29–102).
2. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd Edition, Cambridge University Press.
3. OpenCV Camera Calibration Documentation. https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
4. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing*, 4th Edition, Pearson. Chapter 2.
5. Forsyth, D. A., & Ponce, J. (2012). *Computer Vision: A Modern Approach*, 2nd Edition, Pearson.
