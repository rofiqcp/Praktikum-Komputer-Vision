# MODUL 3: PEMROSESAN CITRA (IMAGE PROCESSING)

## Referensi Utama
**Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. Chapter 3: Image Processing.**

---

## 3.1 Pendahuluan

Pemrosesan citra adalah fondasi dari computer vision. Pada tahapan ini, gambar ditingkatkan kualitasnya, ditransformasi, atau diekstrak informasinya melalui operasi-operasi matematika pada level piksel maupun frekuensi. Output dari image processing adalah gambar yang dimodifikasi atau fitur yang diekstrak.

---

## 3.2 Point Operators (Operasi Titik)

### Brightness dan Contrast
Operasi titik mengubah setiap piksel secara independen:
$$g(x, y) = \alpha \cdot f(x, y) + \beta$$

- $\alpha$ (gain): mengontrol **kontras** (α > 1 → lebih kontras).
- $\beta$ (bias): mengontrol **brightness** (β > 0 → lebih terang).

### Histogram Equalization
Menyebarkan distribusi intensitas piksel secara merata untuk meningkatkan kontras:

$$s_k = T(r_k) = (L-1) \sum_{j=0}^{k} p_r(r_j)$$

di mana $p_r(r_j) = n_j / N$ adalah probabilitas piksel dengan intensitas $r_j$.

OpenCV: `cv2.equalizeHist(gray)`, CLAHE: `cv2.createCLAHE()`.

### Gamma Correction
$$g = f^{\gamma}$$
- $\gamma < 1$: mencerahkan area gelap.
- $\gamma > 1$: menggelapkan area terang.

---

## 3.3 Thresholding

Mengkonversi gambar grayscale menjadi biner berdasarkan threshold.

### Global Thresholding
$$g(x,y) = \begin{cases} 255, & \text{if } f(x,y) > T \\ 0, & \text{otherwise} \end{cases}$$

### Metode Otsu
Menentukan threshold optimal secara otomatis dengan meminimalkan intra-class variance:
$$T^* = \arg\min_T \sigma^2_w(T)$$

### Adaptive Thresholding
Threshold bervariasi per lokasi berdasarkan mean atau Gaussian-weighted mean dari neighborhood.

---

## 3.4 Spatial Filtering (Linear)

### Konvolusi
Operasi konvolusi 2D antara gambar $f$ dan kernel $h$:
$$(f * h)(x, y) = \sum_{i} \sum_{j} f(x-i, y-j) \cdot h(i, j)$$

### Jenis-jenis Filter

| Filter | Fungsi | Kernel Contoh |
|--------|--------|---------------|
| **Box blur** | Averaging / smoothing | Semua elemen = 1/n² |
| **Gaussian blur** | Smoothing tanpa ringing | Distribusi Gaussian |
| **Sharpening** | Mempertajam detail | Center tinggi, sekitar negatif |
| **Emboss** | Efek relief/timbul | Asimetris |

### Gaussian Filter
$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

OpenCV: `cv2.GaussianBlur(img, (ksize, ksize), sigma)`.

### Bilateral Filter
Filter non-linear yang mempertahankan edge sambil menghaluskan area flat:
$$BF[I]_p = \frac{1}{W_p} \sum_{q \in S} G_{\sigma_s}(\|p-q\|) \cdot G_{\sigma_r}(|I_p - I_q|) \cdot I_q$$

---

## 3.5 Deteksi Tepi (Edge Detection)

### Operator Sobel
Filter derivatif untuk mendeteksi edge:
$$G_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}, \quad G_y = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{pmatrix}$$

Magnitude gradient: $G = \sqrt{G_x^2 + G_y^2}$

Arah gradient: $\theta = \arctan(G_y / G_x)$

### Canny Edge Detector
Pipeline:
1. Gaussian smoothing.
2. Hitung gradient (Sobel).
3. Non-maximum suppression.
4. Double thresholding.
5. Edge tracking by hysteresis.

OpenCV: `cv2.Canny(img, threshold1, threshold2)`.

### Laplacian
Deteksi edge menggunakan turunan kedua:
$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

---

## 3.6 Operasi Morfologi

Operasi berbasis bentuk (structuring element) pada gambar biner/grayscale.

### Operasi Dasar
- **Erosi**: Mengecilkan foreground, menghilangkan noise kecil.
$$E = A \ominus B$$
- **Dilasi**: Memperbesar foreground, mengisi lubang kecil.
$$D = A \oplus B$$
- **Opening**: Erosi → Dilasi. Menghilangkan noise kecil.
- **Closing**: Dilasi → Erosi. Menutup lubang kecil.
- **Gradient**: Dilasi − Erosi. Menghasilkan outline.
- **Top Hat**: Gambar − Opening. Mendeteksi fitur terang kecil.
- **Black Hat**: Closing − Gambar. Mendeteksi fitur gelap kecil.

---

## 3.7 Transformasi Fourier

### Discrete Fourier Transform (DFT)
Mengubah gambar dari domain spasial ke domain frekuensi:
$$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \cdot e^{-j2\pi(ux/M + vy/N)}$$

### Aplikasi
- **Low-pass filter** di domain frekuensi → menghilangkan noise (blur).
- **High-pass filter** → mendeteksi tepi.
- **Band-pass filter** → mengisolasi frekuensi tertentu.
- **Analisis periodisitas** → deteksi pola berulang (texture).

---

## 3.8 Image Pyramids

### Gaussian Pyramid
Sequence gambar yang mengecil berturut-turut (downscale + smooth):
```
Level 0: Original (512×512)
Level 1: 256×256
Level 2: 128×128
Level 3: 64×64
```
OpenCV: `cv2.pyrDown()`, `cv2.pyrUp()`.

### Laplacian Pyramid
Selisih antara level Gaussian pyramid yang berurutan — menyimpan detail per level:
$$L_i = G_i - \text{expand}(G_{i+1})$$

Berguna untuk: multi-scale blending, texture synthesis, compression.

---

## 3.9 Compositing dan Alpha Blending

### Alpha Compositing
Menggabungkan dua gambar dengan transparansi:
$$C_{out} = \alpha_f \cdot C_f + (1 - \alpha_f) \cdot \alpha_b \cdot C_b$$

### Laplacian Pyramid Blending
Blending halus dua gambar menggunakan Laplacian pyramid — menghilangkan seam yang terlihat.

### Matting
Ekstraksi foreground (α matte) dari gambar:
$$I = \alpha F + (1 - \alpha) B$$

---

## 3.10 Geometric Transformations dan Warping

### Forward Warping vs Inverse Warping
- **Forward**: Untuk setiap piksel sumber, hitung posisi tujuan → bisa ada lubang.
- **Inverse**: Untuk setiap piksel tujuan, hitung posisi sumber → lebih baik (tidak ada lubang).

### Mesh Warping
Membagi gambar menjadi grid, warp setiap cell secara independen menggunakan interpolasi.

### Aplikasi
- Image morphing (face morphing).
- Panorama blending.
- Image registration.

---

## 3.11 Ringkasan

| Konsep | Penjelasan |
|--------|------------|
| Point Operators | Brightness, contrast, gamma, histogram equalization |
| Thresholding | Global, Otsu, adaptive — konversi ke biner |
| Spatial Filtering | Konvolusi: blur, sharpen, edge detection |
| Edge Detection | Sobel, Canny, Laplacian |
| Morfologi | Erosi, dilasi, opening, closing |
| Fourier Transform | Analisis frekuensi, filtering di domain frekuensi |
| Pyramids | Gaussian pyramid, Laplacian pyramid |
| Compositing | Alpha blending, Laplacian blending, matting |

---

## Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition, Springer. **Chapter 3: Image Processing** (pp. 103–188).
2. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing*, 4th Edition, Pearson.
3. OpenCV Image Processing Tutorials. https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
4. Fisher, R., Perkins, S., Walker, A., & Wolfart, E. *Image Processing Learning Resources*. https://homepages.inf.ed.ac.uk/rbf/HIPR2/
5. Burt, P. J., & Adelson, E. H. (1983). "A multiresolution spline with application to image mosaics." *ACM TOG*.
