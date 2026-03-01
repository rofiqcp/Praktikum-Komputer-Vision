# MODUL 4: MODEL FITTING DAN OPTIMASI

## Referensi Utama
**Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. Chapter 4: Model Fitting and Optimization.**

---

## 4.1 Pendahuluan

Model fitting adalah proses menemukan parameter model matematika yang paling cocok menjelaskan data observasi (misalnya sekumpulan titik, fitur, atau piksel). Optimasi adalah teknik numerik untuk menemukan parameter tersebut secara efisien. Bab ini membahas teknik-teknik yang sangat fundamental dan digunakan di hampir seluruh algoritma computer vision tingkat lanjut.

---

## 4.2 Least Squares

### Ordinary Least Squares (OLS)
Minimalisasi jumlah kuadrat residual:
$$\min_{\mathbf{x}} \| \mathbf{A}\mathbf{x} - \mathbf{b} \|^2$$

Solusi: $\mathbf{x} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$

### Weighted Least Squares
$$\min_{\mathbf{x}} \| \mathbf{W}^{1/2}(\mathbf{A}\mathbf{x} - \mathbf{b}) \|^2$$

Memberikan bobot berbeda pada tiap observasi berdasarkan kepercayaan.

### Total Least Squares
Meminimalkan jarak orthogonal (tegak lurus) ke model, bukan hanya residual vertikal. Diselesaikan dengan SVD.

---

## 4.3 RANSAC (Random Sample Consensus)

Algoritma robust untuk model fitting yang tahan terhadap outlier.

### Langkah RANSAC:
1. Pilih subset minimal titik secara acak (misal 2 titik untuk garis).
2. Fit model menggunakan subset tersebut.
3. Hitung jumlah inlier (titik yang jaraknya ke model < threshold ε).
4. Ulangi N kali, simpan model dengan inlier terbanyak.
5. Re-fit model menggunakan seluruh inlier dari model terbaik.

### Jumlah Iterasi
$$N = \frac{\log(1 - p)}{\log(1 - w^n)}$$
- $p$: probabilitas keberhasilan (misal 0.99).
- $w$: rasio inlier.
- $n$: jumlah titik minimal.

### Varian RANSAC
- **MSAC**: Skor berdasarkan distance, bukan hanya count inlier.
- **PROSAC**: Sampling berdasarkan kualitas match.
- **LO-RANSAC**: Local optimization setelah menemukan model awal.

---

## 4.4 Hough Transform

### Hough Transform untuk Garis
Setiap titik $(x, y)$ di image space dipetakan ke kurva di parameter space $(\rho, \theta)$:
$$\rho = x \cos\theta + y \sin\theta$$

Garis dideteksi sebagai peak di accumulator space.

### Hough Transform untuk Lingkaran
Parameter: $(a, b, r)$ — center $(a, b)$ dan radius $r$.
$$\sqrt{(x-a)^2 + (y-b)^2} = r$$

OpenCV: `cv2.HoughCircles()`.

### Generalized Hough Transform
Mendeteksi bentuk arbitrary menggunakan tabel R (edge direction → displacement ke center).

---

## 4.5 Homografi dan Estimasi Model

### Homography Estimation
Matriks 3×3 yang memetakan titik dari satu bidang ke bidang lain:
$$\tilde{\mathbf{x'}} = \mathbf{H} \tilde{\mathbf{x}}$$

Memerlukan minimal 4 pasang titik korespondensi. Diselesaikan dengan DLT (Direct Linear Transform).

### Dekomposisi Homografi
Homografi antar dua view scene planar:
$$\mathbf{H} = \mathbf{K'} (\mathbf{R} - \frac{\mathbf{t}\mathbf{n}^T}{d}) \mathbf{K}^{-1}$$

---

## 4.6 Iteratively Reweighted Least Squares (IRLS)

Robust regression yang iteratif mengurangi bobot outlier:
1. Mulai dengan OLS.
2. Hitung residual.
3. Update weight berdasarkan fungsi robust (Huber, Tukey bisquare).
4. Ulangi hingga konvergen.

---

## 4.7 Regularisasi

### Tujuan
Mencegah overfitting dengan menambahkan term penalti:

$$\min_{\mathbf{x}} \| \mathbf{A}\mathbf{x} - \mathbf{b} \|^2 + \lambda \| \mathbf{x} \|^2$$

### Jenis Regularisasi
- **L2 (Ridge/Tikhonov)**: $\lambda \|\mathbf{x}\|_2^2$ → solusi smooth.
- **L1 (Lasso)**: $\lambda \|\mathbf{x}\|_1$ → solusi sparse.
- **Elastic Net**: Kombinasi L1 + L2.

---

## 4.8 Markov Random Fields (MRF)

### Konsep
MRF memodelkan hubungan spasial antar piksel menggunakan graph. Energy function:
$$E(\mathbf{x}) = \sum_i D_i(x_i) + \sum_{(i,j)} V_{ij}(x_i, x_j)$$

- $D_i$: Data term (seberapa cocok label $x_i$ dengan observasi).
- $V_{ij}$: Smoothness term (penalti jika label tetangga berbeda).

### Optimasi MRF
- **ICM (Iterated Conditional Modes)**: Greedy, update satu piksel pada satu waktu.
- **Graph Cut**: Optimal untuk binary labels.
- **Belief Propagation**: Message passing pada graph.

---

## 4.9 Optical Flow Estimation

### Lucas-Kanade (Sparse)
Asumsi: brightness constancy dan motion konstan dalam neighborhood kecil.
$$\mathbf{A}^T\mathbf{A} \mathbf{v} = -\mathbf{A}^T\mathbf{b}$$

### Horn-Schunck (Dense)
Minimalisasi energy dengan data term + smoothness term:
$$E = \iint \left[ (I_x u + I_y v + I_t)^2 + \alpha^2(\|\nabla u\|^2 + \|\nabla v\|^2) \right] dx\, dy$$

---

## 4.10 Ringkasan

| Konsep | Penjelasan |
|--------|------------|
| Least Squares | Minimalisasi kuadrat error — dasar model fitting |
| RANSAC | Model fitting robust terhadap outlier |
| Hough Transform | Deteksi garis dan lingkaran di parameter space |
| Homography | Transformasi proyektif antar bidang |
| IRLS | Iterative robust fitting |
| Regularisasi | Mencegah overfitting (L1, L2) |
| MRF | Model spasial antar piksel (labeling problem) |
| Optical Flow | Estimasi gerakan piksel antar frame |

---

## Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition, Springer. **Chapter 4: Model Fitting and Optimization** (pp. 189–266).
2. Fischler, M. A., & Bolles, R. C. (1981). "Random sample consensus: a paradigm for model fitting." *Communications of the ACM*.
3. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd Edition.
4. Boykov, Y., Veksler, O., & Zabih, R. (2001). "Fast approximate energy minimization via graph cuts." *IEEE TPAMI*.
5. Lucas, B. D., & Kanade, T. (1981). "An iterative image registration technique with an application to stereo vision." *IJCAI*.
