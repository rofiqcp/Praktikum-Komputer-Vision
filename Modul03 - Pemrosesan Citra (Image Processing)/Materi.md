# MODUL 3: PEMROSESAN CITRA (IMAGE PROCESSING)

## Referensi Utama
**Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. Chapter 3: Image Processing.**

---

## 3.1 Pendahuluan Pemrosesan Citra

Pemrosesan citra (*image processing*) mencakup operasi yang menerima gambar sebagai input dan menghasilkan gambar
yang telah diperbaiki atau informasi yang diekstrak. Menurut Szeliski (2022, Ch. 3), operasi pemrosesan citra
dapat dikategorikan menjadi tiga kelompok utama:

1. **Point Operators (Operator Titik):** Setiap piksel output bergantung hanya pada piksel input yang bersesuaian.
   Contoh: thresholding, histogram equalization, konversi warna.
2. **Neighborhood Operators (Operator Ketetanggaan):** Setiap piksel output bergantung pada sekumpulan piksel
   di sekitar piksel input. Contoh: konvolusi, blur, deteksi tepi, morfologi.
3. **Global Operators:** Piksel output bergantung pada keseluruhan gambar.
   Contoh: transformasi Fourier, histogram global.

Modul ini mencakup 20 topik praktikum yang tersebar di ketiga kategori:
- **Kategori Point:** Histogram equalization (02), CLAHE (03), thresholding global (05), Otsu/Triangle (06)
- **Kategori Neighborhood:** Segmentasi HSV (04), adaptive threshold (07), konvolusi (08), Gaussian blur (09),
  median/bilateral (10), sharpening (11), Sobel (12), Canny (13), Laplacian (14), erosi/dilasi (15),
  morfologi lanjut (16), top-hat/black-hat (17), filter frekuensi (19)
- **Kategori Global:** Deteksi kontur (01), transformasi Fourier (18), connected components (20)

---

## 3.2 Histogram dan Equalization

### 3.2.1 Histogram Gambar

Histogram gambar $h(r_k)$ menyatakan jumlah piksel dengan intensitas $r_k$:

$$h(r_k) = n_k, \quad k = 0, 1, 2, \ldots, 255$$

di mana $n_k$ adalah jumlah piksel dengan nilai intensitas $r_k$ pada gambar berukuran $M \times N$.

Histogram ternormalisasi (probability density function):

$$p(r_k) = \frac{n_k}{MN}$$

### 3.2.2 Histogram Equalization

Histogram equalization memetakan intensitas input ke intensitas output agar distribusi histogramnya merata.
Fungsi transformasi kumulatif:

$$s_k = T(r_k) = (L-1) \sum_{j=0}^{k} p(r_j)$$

di mana $L = 256$ untuk gambar 8-bit. Transformasi ini menyamakan CDF (Cumulative Distribution Function)
dengan distribusi seragam.

```python
import cv2
import numpy as np

img = cv2.imread('gambar.jpg', cv2.IMREAD_GRAYSCALE)

# Histogram equalization dengan OpenCV
img_eq = cv2.equalizeHist(img)

# Histogram stretching manual (min-max normalization)
img_min = img.min()
img_max = img.max()
img_stretched = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)

# Tampilkan histogram
hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
```

### 3.2.3 CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE membagi gambar menjadi tile (blok) kecil dan menerapkan histogram equalization per tile, dengan
pembatasan penguatan kontras menggunakan `clipLimit`. Piksel histogram yang melebihi clip limit
didistribusikan ulang ke seluruh bins sebelum equalization dilakukan.

Proses CLAHE:
1. Bagikan gambar menjadi $T \times T$ tile (misalnya 8x8).
2. Hitung histogram per tile.
3. Batasi histogram: jika $h(k) > clipLimit$ maka distribusikan kelebihan ke seluruh bins.
4. Terapkan equalization per tile.
5. Interpolasi bilinear antar tile untuk menghilangkan batas.

```python
# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)

# Pengaruh clipLimit
for clip in [1.0, 2.0, 5.0, 10.0]:
    clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    result = clahe_obj.apply(img)
```

---

## 3.3 Thresholding

### 3.3.1 Thresholding Global

Thresholding global mengklasifikasikan setiap piksel menjadi foreground atau background berdasarkan
nilai threshold tunggal $T$:

$$g(x,y) = \begin{cases} 255 & \text{jika } f(x,y) > T \\ 0 & \text{jika } f(x,y) \leq T \end{cases}$$

Tipe-tipe thresholding dalam OpenCV:
- `THRESH_BINARY`: $g = 255$ jika $f > T$, else $0$
- `THRESH_BINARY_INV`: $g = 0$ jika $f > T$, else $255$
- `THRESH_TRUNC`: $g = T$ jika $f > T$, else $f$
- `THRESH_TOZERO`: $g = f$ jika $f > T$, else $0$
- `THRESH_TOZERO_INV`: $g = 0$ jika $f > T$, else $f$

```python
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, trunc   = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, tozero  = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
```

### 3.3.2 Metode Otsu

Metode Otsu secara otomatis mencari nilai $T$ yang memaksimalkan *between-class variance* (sama dengan
meminimalkan *within-class variance*):

$$\sigma_B^2(T) = w_0(T) \cdot w_1(T) \cdot [\mu_0(T) - \mu_1(T)]^2$$

di mana $w_0, w_1$ adalah probabilitas piksel di kelas 0 dan 1, serta $\mu_0, \mu_1$ adalah rata-rata
intensitas masing-masing kelas.

Nilai Otsu optimal:

$$T^* = \arg\max_T \sigma_B^2(T)$$

```python
# Otsu thresholding
otsu_val, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Threshold Otsu: {otsu_val}")

# Triangle thresholding
tri_val, img_tri = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
print(f"Threshold Triangle: {tri_val}")
```

### 3.3.3 Adaptive Thresholding

Adaptive thresholding menghitung threshold $T(x,y)$ secara lokal untuk setiap piksel berdasarkan
ketetanggaan berukuran $blockSize \times blockSize$:

$$T(x,y) = \text{mean}(\text{neighborhood}) - C$$

atau menggunakan bobot Gaussian:

$$T(x,y) = \text{GaussianWeightedMean}(\text{neighborhood}) - C$$

```python
# Adaptive Mean
adapt_mean = cv2.adaptiveThreshold(img, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Adaptive Gaussian
adapt_gauss = cv2.adaptiveThreshold(img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
```

---

## 3.4 Deteksi Kontur

Kontur adalah kurva yang menghubungkan titik-titik dengan intensitas atau warna yang sama pada batas
objek. Deteksi kontur pada gambar biner dilakukan dengan `cv2.findContours()`.

### 3.4.1 Hierarki Kontur

Mode retrieval kontur:
- `RETR_EXTERNAL`: hanya kontur paling luar
- `RETR_LIST`: semua kontur tanpa hirarki
- `RETR_CCOMP`: dua level hirarki (kontur luar dan lubang)
- `RETR_TREE`: hirarki lengkap (nested contours)

### 3.4.2 Metode Aproksimasi

- `CHAIN_APPROX_NONE`: menyimpan semua titik kontur
- `CHAIN_APPROX_SIMPLE`: menghapus titik redundan pada segmen lurus

```python
import cv2
import numpy as np

img = cv2.imread('gambar.jpg', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Temukan kontur
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Gambar kontur
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)

# Analisis setiap kontur
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
```

### 3.4.3 Momen Kontur

Momen spasial orde ke-$(p+q)$ dari kontur:

$$m_{pq} = \sum_x \sum_y x^p y^q I(x,y)$$

Centroid dari momen:

$$\bar{x} = \frac{m_{10}}{m_{00}}, \quad \bar{y} = \frac{m_{01}}{m_{00}}$$

---

## 3.5 Segmentasi Warna HSV

### 3.5.1 Ruang Warna HSV

Ruang warna HSV (Hue, Saturation, Value) lebih intuitif untuk segmentasi warna dibandingkan RGB/BGR
karena komponen warna (Hue) terpisah dari informasi kecerahan (Value).

Dalam OpenCV, rentang nilai HSV adalah:
- Hue: 0-179 (merepresentasikan sudut 0-360 derajat dibagi 2)
- Saturation: 0-255
- Value: 0-255

### 3.5.2 Rentang Warna Umum di HSV (OpenCV)

| Warna | H (min) | H (max) | S (min) | S (max) | V (min) | V (max) |
|-------|---------|---------|---------|---------|---------|---------|
| Merah (1) | 0 | 10 | 100 | 255 | 100 | 255 |
| Merah (2) | 160 | 179 | 100 | 255 | 100 | 255 |
| Hijau | 40 | 80 | 40 | 255 | 40 | 255 |
| Biru | 100 | 130 | 100 | 255 | 100 | 255 |
| Kuning | 20 | 35 | 100 | 255 | 100 | 255 |
| Oranye | 10 | 25 | 100 | 255 | 100 | 255 |
| Putih | 0 | 179 | 0 | 30 | 200 | 255 |
| Hitam | 0 | 179 | 0 | 255 | 0 | 50 |

```python
import cv2
import numpy as np

img_bgr = cv2.imread('buah.jpg')
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Segmentasi warna hijau
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])
mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

# Warna merah memerlukan dua range (melewati 0/180 derajat)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])
mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Terapkan mask
result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_green)
```

---

## 3.6 Konvolusi dan Spatial Filtering

### 3.6.1 Prinsip Konvolusi 2D

Konvolusi gambar $f$ dengan kernel $h$ menghasilkan gambar output $g$:

$$g(x,y) = (f * h)(x,y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} h(s,t) \cdot f(x-s, y-t)$$

di mana kernel $h$ berukuran $(2a+1) \times (2b+1)$.

Dalam praktiknya, OpenCV menggunakan korelasi (bukan konvolusi sejati), tetapi untuk kernel simetris
hasilnya identik.

### 3.6.2 Kernel Umum

**Kernel Identitas (tidak mengubah gambar):**
$$K_{identity} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

**Kernel Averaging (blur rata-rata):**
$$K_{avg} = \frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

**Kernel Sharpening:**
$$K_{sharp} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

**Kernel Emboss:**
$$K_{emboss} = \begin{bmatrix} -2 & -1 & 0 \\ -1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix}$$

```python
import cv2
import numpy as np

img = cv2.imread('gambar.jpg')

# Kernel identitas
kernel_identity = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]], dtype=np.float32)

# Kernel averaging
kernel_avg = np.ones((3, 3), dtype=np.float32) / 9.0

# Kernel sharpening
kernel_sharp = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float32)

# Aplikasikan kernel dengan filter2D
img_identity = cv2.filter2D(img, -1, kernel_identity)
img_blurred  = cv2.filter2D(img, -1, kernel_avg)
img_sharp    = cv2.filter2D(img, -1, kernel_sharp)
```

### 3.6.3 Gaussian Blur

Kernel Gaussian 2D diturunkan dari fungsi Gaussian 2D:

$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

Kernel Gaussian terpisahkan (separable): $G_{2D} = G_{1D}^x \otimes G_{1D}^y$, yang lebih efisien secara komputasional.

```python
# Gaussian blur
img_gauss = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)

# Averaging blur (box filter)
img_avg = cv2.blur(img, (5, 5))

# Box filter
img_box = cv2.boxFilter(img, -1, (5, 5))
```

### 3.6.4 Median dan Bilateral Filter

**Median Filter:** Mengganti nilai piksel dengan nilai median dari piksel di ketetanggaannya.
Tidak linier - sangat efektif untuk salt-and-pepper noise.

**Bilateral Filter:** Filter yang mempertahankan tepi berdasarkan dua komponen:
- Gaussian spasial (domain filter): berdasarkan jarak spasial
- Gaussian range (range filter): berdasarkan perbedaan intensitas

$$BF[I]_p = \frac{1}{W_p} \sum_{q \in S} G_{\sigma_s}(||p-q||) \cdot G_{\sigma_r}(|I_p - I_q|) \cdot I_q$$

di mana $\sigma_s$ mengontrol smoothing spasial dan $\sigma_r$ mengontrol sensitivitas tepi.

```python
# Median filter
img_median = cv2.medianBlur(img, ksize=5)

# Bilateral filter (d=9, sigmaColor=75, sigmaSpace=75)
img_bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
```

---

## 3.7 Sharpening

### 3.7.1 Unsharp Masking

Unsharp masking meningkatkan ketajaman dengan menambahkan detail frekuensi tinggi ke gambar asli:

$$\text{sharpened} = \text{orig} + \alpha \cdot (\text{orig} - \text{blur})$$

di mana $\alpha$ adalah faktor penguatan sharpening (biasanya 0.5 sampai 2.0).
- $(\text{orig} - \text{blur})$ disebut *unsharp mask* atau *high-frequency detail*
- Nilai $\alpha$ yang terlalu besar menyebabkan oversharpening dan ringing artifacts

```python
import cv2
import numpy as np

img = cv2.imread('gambar.jpg')
img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)

# Unsharp masking
alpha = 1.5
img_sharpened = cv2.addWeighted(img, 1 + alpha, img_blur, -alpha, 0)

# High-boost filtering (alpha > 1)
alpha_boost = 2.0
unsharp_mask = img.astype(np.float32) - img_blur.astype(np.float32)
img_boost = np.clip(img.astype(np.float32) + alpha_boost * unsharp_mask, 0, 255).astype(np.uint8)
```

### 3.7.2 Laplacian Sharpening

Laplacian mengukur turunan kedua gambar dan dapat digunakan untuk sharpening:

$$\text{sharpened} = f(x,y) - \nabla^2 f(x,y)$$

Kernel Laplacian untuk sharpening (varian 4-neighbor):

$$K_{LoG\_sharp} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

---

## 3.8 Deteksi Tepi

### 3.8.1 Operator Sobel

Operator Sobel menghitung gradien gambar menggunakan kernel 3x3:

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * f(x,y)$$

$$G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * f(x,y)$$

Magnitude gradien dan arah:

$$|G| = \sqrt{G_x^2 + G_y^2} \approx |G_x| + |G_y|$$

$$\theta = \arctan\left(\frac{G_y}{G_x}\right)$$

```python
import cv2
import numpy as np

img_gray = cv2.imread('gambar.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel X dan Y
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude dan arah
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
direction = np.arctan2(sobel_y, sobel_x)

# Scharr (ksize=1) - lebih akurat dari Sobel 3x3
scharr_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=-1)  # ksize=-1 = Scharr
scharr_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=-1)
```

### 3.8.2 Algoritma Canny

Canny adalah algoritma deteksi tepi optimal yang terdiri dari 4 tahap:

**Tahap 1 - Gaussian Smoothing:**
$$f_{smooth} = G_{\sigma} * f$$

**Tahap 2 - Komputasi Gradien Sobel:**
Hitung $G_x$, $G_y$, magnitude $|G|$, dan arah $\theta$.

**Tahap 3 - Non-Maximum Suppression (NMS):**
Pada setiap piksel, pertahankan hanya piksel yang merupakan maksimum lokal di sepanjang arah gradien.
Hasilnya adalah tepi setipis satu piksel.

**Tahap 4 - Hysteresis Thresholding:**
Gunakan dua threshold $T_{low}$ dan $T_{high}$:
- Piksel dengan $|G| > T_{high}$: pasti tepi (strong edge)
- Piksel dengan $T_{low} < |G| < T_{high}$: tepi jika terhubung dengan strong edge (weak edge)
- Piksel dengan $|G| < T_{low}$: bukan tepi

```python
# Canny edge detection
edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)

# Dengan Gaussian blur pra-proses
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
edges_smooth = cv2.Canny(img_blur, 50, 150)
```

### 3.8.3 Operator Laplacian

Laplacian adalah operator turunan kedua isotropik:

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

Kernel Laplacian diskrit (4-neighbor):

$$K_{Lap} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

Laplacian of Gaussian (LoG) menggabungkan smoothing dan deteksi tepi:

$$LoG(x,y) = -\frac{1}{\pi\sigma^4}\left(1 - \frac{x^2+y^2}{2\sigma^2}\right)e^{-\frac{x^2+y^2}{2\sigma^2}}$$

Difference of Gaussians (DoG) sebagai aproksimasi LoG:

$$DoG(x,y) = G_{\sigma_1}(x,y) - G_{\sigma_2}(x,y)$$

```python
# Laplacian
lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
lap_abs = np.abs(lap).astype(np.uint8)

# LoG: blur dulu, lalu Laplacian
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
log = cv2.Laplacian(img_blur, cv2.CV_64F)

# DoG (aproksimasi LoG)
blur1 = cv2.GaussianBlur(img_gray, (3, 3), 1.0)
blur2 = cv2.GaussianBlur(img_gray, (5, 5), 2.0)
dog = blur1.astype(np.float32) - blur2.astype(np.float32)
```

---

## 3.9 Morfologi

### 3.9.1 Erosi dan Dilasi

Operasi morfologi bekerja pada gambar biner menggunakan elemen penyusunan (structuring element) $B$.

**Erosi:** piksel output bernilai 1 hanya jika semua piksel di bawah structuring element bernilai 1.

$$A \ominus B = \{z \mid (B)_z \subseteq A\}$$

**Dilasi:** piksel output bernilai 1 jika setidaknya satu piksel di bawah structuring element bernilai 1.

$$A \oplus B = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$$

```python
import cv2
import numpy as np

img_bin = cv2.imread('biner.jpg', cv2.IMREAD_GRAYSCALE)
_, img_bin = cv2.threshold(img_bin, 127, 255, cv2.THRESH_BINARY)

# Structuring elements
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# Erosi dan dilasi
img_eroded  = cv2.erode(img_bin, kernel_rect, iterations=1)
img_dilated = cv2.dilate(img_bin, kernel_rect, iterations=1)
```

### 3.9.2 Opening dan Closing

**Opening** = erosi diikuti dilasi. Menghilangkan noise kecil (spot kecil) tanpa mengubah bentuk objek besar:

$$A \circ B = (A \ominus B) \oplus B$$

**Closing** = dilasi diikuti erosi. Mengisi lubang kecil dalam objek tanpa mengubah bentuk luar:

$$A \bullet B = (A \oplus B) \ominus B$$

**Morphological Gradient** = perbedaan dilasi dan erosi, menghasilkan tepi objek:

$$\text{gradient} = (A \oplus B) - (A \ominus B)$$

```python
# Opening (noise removal)
opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_rect)

# Closing (hole filling)
closing = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_rect)

# Gradient
gradient = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel_rect)
```

---

## 3.10 Top-Hat dan Black-Hat Transform

### 3.10.1 Top-Hat Transform

Top-Hat transform adalah selisih antara gambar asli dan opening-nya:

$$\text{TopHat}(f) = f - (f \circ B)$$

Hasilnya menonjolkan struktur yang lebih kecil dari structuring element dan lebih terang dari latar belakang.
Aplikasi: mendeteksi teks terang, granula sel darah, atau spot terang pada latar tidak merata.

### 3.10.2 Black-Hat Transform

Black-Hat transform adalah selisih antara closing dan gambar asli:

$$\text{BlackHat}(f) = (f \bullet B) - f$$

Hasilnya menonjolkan struktur yang lebih kecil dari structuring element dan lebih gelap dari latar belakang.
Aplikasi: mendeteksi teks gelap pada latar terang, lubang kecil, atau noda gelap.

```python
# Top-Hat
top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_rect)

# Black-Hat
black_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_rect)

# Aplikasi: koreksi pencahayaan menggunakan Top-Hat
corrected = cv2.add(img, top_hat)
```

---

## 3.11 Transformasi Fourier

### 3.11.1 Discrete Fourier Transform (DFT)

DFT 2D mengubah gambar dari domain spasial ke domain frekuensi:

$$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \cdot e^{-j2\pi(ux/M + vy/N)}$$

Inverse DFT (IDFT) untuk mengembalikan ke domain spasial:

$$f(x,y) = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u,v) \cdot e^{j2\pi(ux/M + vy/N)}$$

### 3.11.2 Magnitude dan Phase Spectrum

$F(u,v)$ adalah bilangan kompleks: $F(u,v) = R(u,v) + jI(u,v)$

**Magnitude spectrum:** merepresentasikan amplitudo setiap komponen frekuensi:

$$|F(u,v)| = \sqrt{R(u,v)^2 + I(u,v)^2}$$

**Phase spectrum:** merepresentasikan fase setiap komponen frekuensi:

$$\angle F(u,v) = \arctan\left(\frac{I(u,v)}{R(u,v)}\right)$$

Dalam visualisasi, digunakan skala logaritmik: $\log(1 + |F(u,v)|)$

```python
import cv2
import numpy as np

img_gray = cv2.imread('gambar.jpg', cv2.IMREAD_GRAYSCALE)

# DFT menggunakan NumPy
f = np.fft.fft2(img_gray)
fshift = np.fft.fftshift(f)  # Pindahkan DC ke tengah
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# DFT menggunakan OpenCV
dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_cv = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)

# IDFT (invers)
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
```

---

## 3.12 Filter Frekuensi

### 3.12.1 Ideal Low-Pass Filter (ILPF)

ILPF mempertahankan frekuensi rendah (dalam radius $D_0$ dari DC) dan memotong frekuensi tinggi:

$$H_{ILPF}(u,v) = \begin{cases} 1 & \text{jika } D(u,v) \leq D_0 \\ 0 & \text{jika } D(u,v) > D_0 \end{cases}$$

di mana $D(u,v) = \sqrt{(u - M/2)^2 + (v - N/2)^2}$ adalah jarak dari DC.

### 3.12.2 Ideal High-Pass Filter (IHPF)

$$H_{IHPF}(u,v) = 1 - H_{ILPF}(u,v)$$

### 3.12.3 Butterworth Low-Pass Filter (BLPF)

BLPF menghindari ringing artifacts dengan transisi yang lebih halus:

$$H_{BLPF}(u,v) = \frac{1}{1 + [D(u,v)/D_0]^{2n}}$$

di mana $n$ adalah orde filter (semakin besar = semakin tajam transisi).

### 3.12.4 Gaussian Low-Pass Filter (GLPF)

$$H_{GLPF}(u,v) = e^{-D^2(u,v) / 2D_0^2}$$

```python
import cv2
import numpy as np

def apply_frequency_filter(img, mask):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    filtered = fshift * mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back).astype(np.uint8)

def ideal_lpf_mask(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow)**2 + (j - ccol)**2) <= cutoff:
                mask[i, j] = 1
    return mask

def gaussian_lpf_mask(shape, sigma):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    mask = np.exp(-D**2 / (2 * sigma**2))
    return mask

img_gray = cv2.imread('gambar.jpg', cv2.IMREAD_GRAYSCALE)
mask_lpf = ideal_lpf_mask(img_gray.shape, cutoff=50)
mask_hpf = 1 - mask_lpf
img_lpf = apply_frequency_filter(img_gray, mask_lpf)
img_hpf = apply_frequency_filter(img_gray, mask_hpf)
```

---

## 3.13 Connected Components

### 3.13.1 Definisi

Connected components labeling mengidentifikasi region (blob) yang terhubung dalam gambar biner.
Dua piksel disebut terhubung jika keduanya bernilai 1 dan terhubung secara fisik:

- **4-connectivity:** piksel terhubung jika berbagi sisi (atas, bawah, kiri, kanan)
- **8-connectivity:** piksel terhubung jika berbagi sisi atau sudut (8 tetangga)

### 3.13.2 Algoritma Labeling

Algoritma two-pass:
1. **Pass Pertama:** Scanline kiri ke kanan, atas ke bawah. Tetapkan label sementara.
2. **Pass Kedua:** Selesaikan equivalence classes menggunakan Union-Find.

### 3.13.3 Implementasi OpenCV

```python
import cv2
import numpy as np

img_bin = cv2.imread('objek_biner.jpg', cv2.IMREAD_GRAYSCALE)
_, img_bin = cv2.threshold(img_bin, 127, 255, cv2.THRESH_BINARY)

# Connected components dengan statistik
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    img_bin, connectivity=8)

print(f"Jumlah komponen (termasuk background): {num_labels}")

# Label 0 adalah background
for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    cx, cy = centroids[i]
    print(f"  Komponen {i}: area={area}, bbox=({x},{y},{w},{h}), centroid=({cx:.1f},{cy:.1f})")

# Visualisasi dengan warna berbeda per komponen
labeled_color = np.zeros((*img_bin.shape, 3), dtype=np.uint8)
colors = np.random.randint(0, 255, size=(num_labels, 3))
colors[0] = [0, 0, 0]  # Background hitam
for label in range(1, num_labels):
    labeled_color[labels == label] = colors[label]
```

### 3.13.4 Filter Berdasarkan Properti Blob

```python
# Filter berdasarkan area
min_area = 500
max_area = 50000
filtered_labels = []

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if min_area <= area <= max_area:
        filtered_labels.append(i)

print(f"Komponen yang lolos filter: {len(filtered_labels)}")
```

---

## Rangkuman

| No | Topik | Fungsi OpenCV Utama | Kategori |
|----|-------|---------------------|----------|
| 01 | Deteksi Kontur | `findContours`, `drawContours` | Global |
| 02 | Histogram Equalization | `equalizeHist`, `calcHist` | Point |
| 03 | CLAHE | `createCLAHE`, `clahe.apply()` | Point/Local |
| 04 | Segmentasi HSV | `cvtColor`, `inRange`, `bitwise_and` | Neighborhood |
| 05 | Thresholding Global | `threshold` | Point |
| 06 | Otsu / Triangle | `threshold + THRESH_OTSU` | Point |
| 07 | Adaptive Threshold | `adaptiveThreshold` | Neighborhood |
| 08 | Konvolusi / Filter2D | `filter2D` | Neighborhood |
| 09 | Gaussian Blur | `GaussianBlur`, `blur`, `boxFilter` | Neighborhood |
| 10 | Median / Bilateral | `medianBlur`, `bilateralFilter` | Neighborhood |
| 11 | Sharpening | `addWeighted`, `filter2D` | Neighborhood |
| 12 | Deteksi Tepi Sobel | `Sobel` | Neighborhood |
| 13 | Deteksi Tepi Canny | `Canny` | Neighborhood |
| 14 | Deteksi Tepi Laplacian | `Laplacian` | Neighborhood |
| 15 | Erosi / Dilasi | `erode`, `dilate` | Neighborhood |
| 16 | Morfologi Lanjut | `morphologyEx` | Neighborhood |
| 17 | Top-Hat / Black-Hat | `morphologyEx` | Neighborhood |
| 18 | Transformasi Fourier | `np.fft.fft2`, `cv2.dft` | Global |
| 19 | Filter Frekuensi | `np.fft.*` + mask | Global |
| 20 | Connected Components | `connectedComponentsWithStats` | Global |

---

## Referensi Tambahan

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*. 2nd ed. Springer.
2. Gonzalez, R.C., & Woods, R.E. (2018). *Digital Image Processing*. 4th ed. Pearson.
3. OpenCV Documentation: https://docs.opencv.org/4.x/
4. NumPy FFT Documentation: https://numpy.org/doc/stable/reference/routines.fft.html
