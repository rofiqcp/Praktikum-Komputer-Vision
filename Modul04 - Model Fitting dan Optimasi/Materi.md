# MODUL 4: MODEL FITTING DAN OPTIMASI

## Referensi Utama
**Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. Chapter 4: Model Fitting and Optimization.**

---

## 4.1 Pendahuluan

Model fitting adalah proses mencocokkan model matematika pada data observasi. Dalam computer vision, ini mencakup fitting garis/lingkaran pada edge points, estimasi homografi dari korespondensi fitur, dan estimasi optical flow antar frame.

Tantangan utama:
- **Noise**: Data real selalu mengandung noise.
- **Outlier**: Sebagian data mungkin tidak mengikuti model.
- **Overfitting**: Model terlalu kompleks → fit noise.
- **Underfitting**: Model terlalu sederhana → tidak capture pattern.

---

## 4.2 Least Squares

### Ordinary Least Squares (OLS)
Meminimalkan jumlah kuadrat residual vertikal:
$$\hat{\beta} = \arg\min_{\beta} \|y - X\beta\|^2$$

Solusi (normal equation):
$$\hat{\beta} = (X^T X)^{-1} X^T y$$

### Weighted Least Squares (WLS)
Memberikan bobot berbeda pada setiap observasi:
$$\hat{\beta} = \arg\min_{\beta} \sum_i w_i (y_i - x_i^T \beta)^2$$

Solusi:
$$\hat{\beta} = (X^T W X)^{-1} X^T W y$$

Dimana $W = \text{diag}(w_1, \ldots, w_n)$.

### Total Least Squares (TLS)
Meminimalkan jarak orthogonal (tegak lurus) ke model — tepat ketika kedua variabel (x dan y) memiliki noise:
$$\min \sum_i d_{\perp}(\mathbf{p}_i, \text{model})^2$$

Solusi via SVD: eigenvector dari eigenvalue terkecil pada matriks kovarians data.

---

## 4.3 RANSAC (Random Sample Consensus)

### Algoritma
1. **Random sampling**: Ambil s titik minimal untuk fit model.
2. **Model fitting**: Fit model dari s titik.
3. **Inlier counting**: Hitung titik dengan residual < threshold ε.
4. **Repeat**: Ulangi N kali, simpan model dengan inlier terbanyak.
5. **Refine**: Re-fit model menggunakan semua inlier dari model terbaik.

### Jumlah Iterasi
Untuk probabilitas $p$ menemukan model bebas outlier dengan rasio inlier $w$:
$$N = \frac{\log(1 - p)}{\log(1 - w^s)}$$

### Implementasi OpenCV
```python
# RANSAC untuk homography
H, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, ransacReprojThreshold=5.0)
```

---

## 4.4 IRLS (Iteratively Reweighted Least Squares)

Robust fitting iteratif:
1. Fit model dengan OLS (iterasi 0).
2. Hitung residual $r_i$.
3. Update bobot: $w_i = \psi(r_i)$ (fungsi weight: Huber, Tukey, Cauchy).
4. Re-fit dengan WLS menggunakan bobot baru.
5. Repeat hingga konvergen.

Fungsi weight populer:
- **Huber**: $w_i = \min(1, c/|r_i|)$
- **Tukey bisquare**: $w_i = (1 - (r_i/c)^2)^2$ jika $|r_i| < c$, else 0.

---

## 4.5 Regularisasi

### Ridge Regression (L2)
$$\hat{\beta} = \arg\min_{\beta} \|y - X\beta\|^2 + \lambda \|\beta\|^2$$

Efek: Menyusutkan koefisien menuju 0 tetapi tidak pernah tepat 0.

### Lasso Regression (L1)
$$\hat{\beta} = \arg\min_{\beta} \|y - X\beta\|^2 + \lambda \|\beta\|_1$$

Efek: Membuat beberapa koefisien tepat = 0 → **feature selection** (sparsity).

### Trade-off Bias-Variance
- $\lambda$ kecil → low bias, high variance (overfit).
- $\lambda$ besar → high bias, low variance (underfit).

---

## 4.6 Hough Transform

### Hough Line Transform
Setiap titik $(x, y)$ dalam gambar memiliki banyak kemungkinan garis yang melaluinya. Dalam parameterisasi $(\rho, \theta)$:
$$\rho = x \cos\theta + y \sin\theta$$

Setiap titik memberikan "vote" di Hough space. Puncak akumulator = garis terdeteksi.

```python
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                           minLineLength=50, maxLineGap=10)
```

### Hough Circle Transform
Menggunakan Hough gradient method:
```python
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=100, param2=30, minRadius=10, maxRadius=100)
```

---

## 4.7 Homography

Homografi adalah transformasi projective 3×3 yang memetakan titik-titik pada satu bidang ke bidang lain:
$$\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} \sim \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

8 DOF → memerlukan minimal 4 pasang titik korespondensi.

```python
H, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)
warped = cv2.warpPerspective(img, H, (w, h))
```

### Aplikasi
- Document scanner (koreksi perspektif)
- Panorama stitching
- Augmented reality (overlay pada planar surface)

---

## 4.8 Fitting Kontur dan Template Matching

### Fitting Ellips
```python
ellipse = cv2.fitEllipse(contour)       # Standard
ellipse = cv2.fitEllipseAMS(contour)    # Algebraic method
ellipse = cv2.fitEllipseDirect(contour) # Direct method
```

### Template Matching
$$R(x,y) = \text{similarity}(I(x:x+w, y:y+h), T)$$

```python
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
```

---

## 4.9 Graph Cut dan Segmentasi

### GrabCut
Segmentasi interaktif berbasis graph cut:
```python
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
```

### Watershed
Segmentasi berbasis topographic surface:
```python
markers = cv2.watershed(img, markers)
```

### MRF Energy Minimization
Minimasi energi:
$$E = \sum_i D(x_i, l_i) + \lambda \sum_{(i,j)} V(l_i, l_j)$$

- $D$: Data term (kecocokan label dengan observasi).
- $V$: Smoothness term (konsistensi antar piksel tetangga).

---

## 4.10 Optical Flow

### Brightness Constancy Assumption
$$I(x, y, t) = I(x + u, y + v, t + 1)$$

Taylor expansion → **optical flow constraint equation**:
$$I_x u + I_y v + I_t = 0$$

### Lucas-Kanade (Sparse)
Asumsi: flow konstan dalam window kecil. Sistem overdetermined → least squares:
```python
p1, st, err = cv2.calcOpticalFlowPyrLK(prev, next, p0, None,
                                         winSize=(15,15), maxLevel=3)
```

### Farneback (Dense)
Estimasi flow untuk setiap piksel menggunakan polynomial expansion:
```python
flow = cv2.calcOpticalFlowFarneback(prev, next, None,
                                     pyr_scale=0.5, levels=3, winsize=15,
                                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
```

### Visualisasi Flow (HSV)
- **Hue**: Arah flow (0–360°).
- **Saturation**: Magnitude flow.

---

## 4.11 Cross-Validation

### K-Fold Cross-Validation
1. Bagi data menjadi K fold.
2. Untuk setiap fold: gunakan sebagai test, sisanya sebagai train.
3. Rata-rata performance di semua fold.

### Leave-One-Out (LOO)
K-Fold dengan K = n (jumlah data). Setiap observasi dijadikan test set sekali.

### Model Selection
- Pilih model/parameter dengan **CV error terendah**.
- Plot train error vs CV error → deteksi overfitting.
- Learning curves: evaluasi apakah perlu lebih banyak data.

---

## 4.12 Denoising via Optimasi

### Non-Local Means (NLM)
Averaging piksel yang mirip (bukan hanya tetangga):
```python
denoised = cv2.fastNlMeansDenoising(noisy, None, h=10)
```

### Total Variation Denoising
Minimasi energi:
$$\min_u \|u - f\|^2 + \lambda \|\nabla u\|_1$$

- Data fidelity term: $u$ dekat dengan observasi $f$.
- Regularity term: gradien kecil (gambar smooth) tetapi edge dipertahankan (L1 norm).

### Bilateral Filter
Edge-preserving smoothing:
```python
denoised = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)
```

---

## 4.13 Ringkasan

| Konsep | Penjelasan |
|--------|------------|
| OLS | Least squares klasik — minimasi residual vertikal |
| WLS | Least squares dengan bobot per observasi |
| TLS | Minimasi jarak orthogonal — noise di x dan y |
| RANSAC | Estimasi robust: random sampling + voting |
| IRLS | Robust iteratif: update bobot berdasarkan residual |
| Ridge (L2) | Regularisasi: susutkan koefisien |
| Lasso (L1) | Regularisasi: sparsity (koefisien = 0) |
| Hough Transform | Voting-based detection: garis, lingkaran |
| Homography | Transformasi projective 3×3, 8 DOF |
| Fitting Kontur | fitEllipse, minAreaRect, convexHull |
| Template Matching | Sliding-window correlation |
| GrabCut/Watershed | Graph cut dan topographic segmentation |
| Lucas-Kanade | Sparse optical flow (window-based least squares) |
| Farneback | Dense optical flow (polynomial expansion) |
| Cross-Validation | Model selection via K-Fold, LOO |
| NLM / TV Denoising | Denoising berbasis optimasi |

---

## Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition, Springer. **Chapter 4: Model Fitting and Optimization** (pp. 189–260).
2. Fischler, M. A., & Bolles, R. C. (1981). Random Sample Consensus: A Paradigm for Model Fitting. *Communications of the ACM*, 24(6), 381–395.
3. Duda, R. O., & Hart, P. E. (1972). Use of the Hough Transformation to Detect Lines and Curves in Pictures. *Communications of the ACM*, 15(1), 11–15.
4. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd Edition, Cambridge University Press.
5. Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear Total Variation Based Noise Removal Algorithms. *Physica D*, 60, 259–268.
6. Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. *IJCAI*, 674–679.
7. OpenCV Documentation. https://docs.opencv.org/4.x/
