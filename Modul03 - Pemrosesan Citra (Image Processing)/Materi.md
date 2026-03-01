# MODUL 3: PEMROSESAN CITRA (IMAGE PROCESSING)

## Referensi Utama
**Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. Chapter 3: Image Processing.**

---

## 3.1 Pendahuluan Pemrosesan Citra

Pemrosesan citra (*image processing*) mencakup operasi-operasi yang menerima gambar sebagai input dan menghasilkan gambar yang telah diperbaiki atau informasi yang diekstrak. Menurut Szeliski (2022, Ch. 3), pemrosesan citra merupakan tahap *pre-processing* yang krusial sebelum analisis tingkat tinggi.

### Kategori Operasi

| Kategori | Contoh | Karakteristik |
|----------|--------|---------------|
| **Point Operators** | Brightness, contrast, gamma, thresholding | Setiap piksel diproses independen |
| **Neighborhood Operators** | Blur, sharpen, edge detection | Bergantung piksel tetangga (kernel) |
| **Global Operators** | Histogram equalization, Fourier transform | Bergantung seluruh gambar |

---

## 3.2 Brightness dan Contrast

### Transformasi Linear
$$g(x,y) = \alpha \cdot f(x,y) + \beta$$

- $\alpha$ — **contrast** (gain). $\alpha > 1$: kontras naik; $\alpha < 1$: kontras turun.
- $\beta$ — **brightness** (bias). $\beta > 0$: lebih terang; $\beta < 0$: lebih gelap.

```python
adjusted = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
```

**Saturasi (Clipping)**: Nilai piksel di-clip ke range [0, 255] sehingga detail bisa hilang.

### Auto-Contrast (Linear Stretch)
Memetakan intensitas minimum ke 0 dan maksimum ke 255:
$$g = \frac{f - f_{min}}{f_{max} - f_{min}} \times 255$$

---

## 3.3 Histogram dan Equalization

### Histogram Gambar
Histogram menggambarkan distribusi intensitas piksel:
$$h(r_k) = n_k, \quad k = 0, 1, \ldots, L-1$$

```python
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
```

### Histogram Equalization
Meratakan histogram untuk meningkatkan kontras:
$$s = T(r) = (L-1) \sum_{j=0}^{r} p(r_j)$$

```python
equalized = cv2.equalizeHist(gray)
```

**Limitasi**: Cenderung over-amplify noise dan menghasilkan gambar yang terlihat tidak natural.

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
Mengatasi limitasi global equalization dengan:
1. Membagi gambar menjadi **tile** (region kecil).
2. Menerapkan equalization per tile.
3. **Clip limit** membatasi amplifikasi noise.
4. Interpolasi bilinear antar tile untuk menghindari artefak.

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
result = clahe.apply(gray)
```

---

## 3.4 Gamma Correction

$$I_{out} = \left(\frac{I_{in}}{255}\right)^{\gamma} \times 255$$

- $\gamma < 1$: Mencerahkan (ekspansi range gelap).
- $\gamma = 1$: Tidak ada perubahan.
- $\gamma > 1$: Menggelapkan (ekspansi range terang).

### Implementasi Efisien dengan LUT
```python
table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype('uint8')
corrected = cv2.LUT(img, table)
```

---

## 3.5 Thresholding

### Thresholding Global
Mengkonversi gambar ke biner berdasarkan nilai threshold $T$:
$$g(x,y) = \begin{cases} 255 & \text{jika } f(x,y) > T \\ 0 & \text{lainnya} \end{cases}$$

```python
ret, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
```

Tipe threshold OpenCV: `BINARY`, `BINARY_INV`, `TRUNC`, `TOZERO`, `TOZERO_INV`.

### Otsu's Method
Menentukan threshold optimal secara otomatis dengan meminimalkan **intra-class variance**:
$$\sigma^2_w(T) = q_1(T)\sigma^2_1(T) + q_2(T)\sigma^2_2(T)$$

```python
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Triangle Method
Untuk histogram unimodal — menarik garis dari puncak histogram ke ujung, dan mencari jarak maksimum:
```python
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
```

### Adaptive Thresholding
Untuk gambar dengan pencahayaan tidak merata — threshold dihitung per region:
```python
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, blockSize=11, C=2)
```

- **Mean**: threshold = mean piksel dalam block.
- **Gaussian**: threshold = weighted mean (Gaussian) dalam block.
- **blockSize**: ukuran neighborhood (ganjil).
- **C**: konstanta yang dikurangkan dari mean.

---

## 3.6 Konvolusi dan Spatial Filtering

### Prinsip Konvolusi
Konvolusi menggeser kernel $h$ atas gambar $f$ dan menghitung weighted sum:
$$(f * h)[x,y] = \sum_i \sum_j f[x-i, y-j] \cdot h[i,j]$$

```python
result = cv2.filter2D(img, -1, kernel)
```

### Gaussian Blur
Filter smoothing menggunakan kernel Gaussian:
$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

```python
blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
```

Parameter penting:
- **ksize**: Ukuran kernel (ganjil). Lebih besar = lebih blur.
- **sigma**: Standard deviation. Lebih besar = lebih smooth.

### Box Filter
Rata-rata sederhana semua piksel dalam kernel:
```python
blurred = cv2.blur(img, (ksize, ksize))
```

### Median Filter
Mengambil nilai median dalam neighborhood — efektif untuk **salt-and-pepper noise**:
```python
denoised = cv2.medianBlur(img, ksize)
```

### Bilateral Filter
Edge-preserving smoothing — smoothing berdasarkan kedekatan spasial DAN kedekatan intensitas:
```python
smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
```

- **sigmaColor**: Piksel dengan perbedaan intensitas besar tidak di-smooth bersama.
- **sigmaSpace**: Radius spasial pengaruh.

---

## 3.7 Sharpening

### Sharpening Kernel
```python
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])
sharpened = cv2.filter2D(img, -1, kernel)
```

### Unsharp Masking
Menambahkan detail (high-frequency) yang hilang saat blur:
$$\text{sharpened} = \text{original} + \alpha \cdot (\text{original} - \text{blurred})$$

```python
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
sharpened = cv2.addWeighted(img, 1.0 + alpha, blurred, -alpha, 0)
```

**Halo artifact**: Terjadi jika parameter terlalu agresif — garis terang/gelap di sekitar edge.

---

## 3.8 Deteksi Tepi (Edge Detection)

### Sobel Operator
Menghitung gradien menggunakan kernel 3×3:
$$G_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}, \quad G_y = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{pmatrix}$$

Magnitude: $G = \sqrt{G_x^2 + G_y^2}$

```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
```

### Canny Edge Detection
Algoritma multi-stage:
1. **Gaussian smoothing** — mengurangi noise.
2. **Gradient computation** — Sobel.
3. **Non-maximum suppression** — edge thinning.
4. **Hysteresis thresholding** — double threshold.

```python
edges = cv2.Canny(gray, low_threshold, high_threshold)
```

### Laplacian
Derivatif orde kedua — mendeteksi edge di semua arah:
$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

```python
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
```

**Catatan**: Sangat sensitif terhadap noise — selalu gunakan dengan Gaussian blur (LoG = Laplacian of Gaussian).

---

## 3.9 Operasi Morfologi

### Structuring Element
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Bentuk: MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS
```

### Operasi Dasar
| Operasi | Formula | Efek |
|---------|---------|------|
| **Erosi** | $A \ominus B$ | Mengecilkan objek, menghilangkan noise kecil |
| **Dilasi** | $A \oplus B$ | Memperbesar objek, menutup gap kecil |

```python
eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)
```

### Operasi Lanjut
| Operasi | Formula | Kegunaan |
|---------|---------|----------|
| **Opening** | $(A \ominus B) \oplus B$ | Menghilangkan noise kecil |
| **Closing** | $(A \oplus B) \ominus B$ | Menutup lubang kecil |
| **Gradient** | $(A \oplus B) - (A \ominus B)$ | Mengekstrak outline objek |

```python
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
```

### Top Hat dan Black Hat
- **Top Hat** = original − opening → mengekstrak fitur **terang** kecil di latar gelap.
- **Black Hat** = closing − original → mengekstrak fitur **gelap** kecil di latar terang.

```python
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
```

**Aplikasi**: Mengoreksi pencahayaan tidak merata (subtract Top Hat dari original).

---

## 3.10 Transformasi Fourier

### Discrete Fourier Transform (DFT)
Mengubah gambar dari domain spasial ke domain frekuensi:
$$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \cdot e^{-j2\pi(ux/M + vy/N)}$$

```python
dft = np.fft.fft2(gray)
dft_shift = np.fft.fftshift(dft)
magnitude = 20 * np.log(np.abs(dft_shift) + 1)
```

### Interpretasi Spectrum
- **Frekuensi rendah** (tengah spectrum): area gambar yang smooth/uniform.
- **Frekuensi tinggi** (pinggir spectrum): edge, detail, noise.

---

## 3.11 Filter Frekuensi

### Low-Pass Filter
Meloloskan frekuensi rendah (smoothing effect):

| Tipe | Karakteristik |
|------|---------------|
| **Ideal** | Cutoff tajam → artefak ringing |
| **Gaussian** | Transisi smooth → tanpa ringing |
| **Butterworth** | Kontrol kecuraman transisi via orde n |

### High-Pass Filter
Meloloskan frekuensi tinggi (edge enhancement):
$$H_{HP}(u,v) = 1 - H_{LP}(u,v)$$

### Band-Pass Filter
Meloloskan band frekuensi tertentu. Notch filter untuk menghilangkan pola periodik (moiré).

### Proses Filtering
```python
# 1. FFT
dft = np.fft.fft2(gray)
dft_shift = np.fft.fftshift(dft)
# 2. Terapkan filter (multiply)
filtered = dft_shift * H
# 3. Inverse FFT
result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
```

---

## 3.12 Alpha Blending dan Compositing

### Alpha Blending
$$I_{out} = \alpha \cdot I_1 + (1 - \alpha) \cdot I_2$$

```python
blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
```

### Laplacian Pyramid Blending
Blending yang lebih halus menggunakan multi-skala:
1. Build Laplacian pyramid untuk kedua gambar.
2. Build Gaussian pyramid untuk mask.
3. Blend setiap level pyramid.
4. Reconstruct dari blended pyramid.

Menghasilkan transisi yang **seamless** tanpa edge artifacts.

### Chroma Keying (Green Screen)
Mengganti background hijau dengan gambar lain:
1. Konversi ke HSV.
2. Threshold range hijau → buat mask.
3. Morfologi untuk clean-up mask.
4. Composite foreground + new background menggunakan mask.

---

## 3.13 Ringkasan

| Konsep | Penjelasan |
|--------|------------|
| Brightness/Contrast | Transformasi linear: $\alpha \cdot f + \beta$ |
| Histogram Equalization | Meratakan distribusi intensitas |
| CLAHE | Equalization adaptif per tile dengan clip limit |
| Gamma Correction | Transformasi non-linear: $I^{\gamma}$ |
| Thresholding | Global, Otsu, Triangle, Adaptive |
| Konvolusi | Operasi kernel × region gambar |
| Gaussian Blur | Smoothing menggunakan kernel Gaussian |
| Median/Bilateral Filter | Denoising edge-preserving |
| Sharpening | Unsharp masking, kernel sharpening |
| Sobel | Gradien orde pertama (edge) |
| Canny | Multi-stage edge detection (NMS + hysteresis) |
| Laplacian | Derivatif orde kedua (edge multidirectional) |
| Erosi/Dilasi | Operasi morfologi dasar |
| Opening/Closing | Morfologi lanjut: hapus noise / tutup lubang |
| Top Hat/Black Hat | Ekstraksi fitur kecil terang/gelap |
| Fourier Transform | Analisis domain frekuensi |
| Filter Frekuensi | Low-pass, high-pass, band-pass, notch |
| Alpha Blending | Penggabungan gambar dengan bobot |
| Laplacian Blending | Multi-scale seamless blending |

---

## Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition, Springer. **Chapter 3: Image Processing** (pp. 103–188).
2. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing*, 4th Edition, Pearson. Chapters 3–5.
3. OpenCV Documentation — Image Processing. https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
4. Burger, W., & Burge, M. J. (2016). *Digital Image Processing: An Algorithmic Introduction Using Java*, 2nd Edition, Springer.
5. Fisher, R., et al. HIPR2 — Hypermedia Image Processing Reference. https://homepages.inf.ed.ac.uk/rbf/HIPR2/
