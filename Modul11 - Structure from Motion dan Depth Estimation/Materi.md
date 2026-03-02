# MATERI PRAKTIKUM
# MODUL 11: STRUCTURE FROM MOTION DAN DEPTH ESTIMATION

---

## 11.1 Pendahuluan

Structure from Motion (SfM) dan Depth Estimation merupakan area fundamental dalam komputer vision yang bertujuan merekonstruksi informasi 3D dari gambar 2D. Modul ini mencakup:
- **Epipolar Geometry**: hubungan geometris antara dua pandangan sebuah scene 3D.
- **Stereo Vision**: estimasi kedalaman dari pasangan kamera.
- **Structure from Motion**: rekonstruksi 3D dari beberapa gambar.

Aplikasi meliputi autonomous driving, augmented reality, pemetaan 3D, dan robot navigation.

---

## 11.2 Epipolar Geometry

### Konsep Dasar

Dua kamera yang melihat titik 3D $P$ yang sama menghasilkan dua titik proyeksi $p$ dan $p'$. Hubungan geometris antara kedua proyeksi ini disebut **epipolar geometry**.

Komponen utama:
- **Epipole**: titik dimana baseline (garis antara dua pusat kamera) memotong bidang gambar.
- **Epipolar Plane**: bidang yang dibentuk oleh titik 3D dan dua pusat kamera.
- **Epipolar Line**: irisan epipolar plane dengan bidang gambar.

### Epipolar Constraint

Untuk setiap pasangan titik koresponden $p$ dan $p'$:

$$p'^T F p = 0$$

di mana $F$ adalah **Fundamental Matrix** (3×3, rank 2). Constraint ini mengurangi pencarian korespondensi dari 2D area menjadi 1D garis (epipolar line).

---

## 11.3 Fundamental Matrix

### Definisi

Fundamental Matrix $F$ adalah matriks 3×3 rank 2 yang mengenkode hubungan epipolar antara dua pandangan **tanpa memerlukan kalibrasi kamera**.

Sifat penting:
- $\text{rank}(F) = 2$, $\det(F) = 0$
- 7 derajat kebebasan (9 elemen - 1 skala - 1 rank constraint)
- $Fe = 0$ dan $F^T e' = 0$ (epipoles)

### 8-Point Algorithm

Diberikan $n \geq 8$ korespondensi $(p_i, p_i')$, setiap pasangan memberikan satu persamaan linier. Disusun menjadi sistem:

$$Af = 0$$

di mana $f$ adalah vektor 9 elemen dari $F$. Solusi diperoleh via SVD.

### Implementasi OpenCV

```python
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
# atau dengan RANSAC:
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
```

---

## 11.4 Essential Matrix

### Definisi

Essential Matrix $E$ adalah versi terkalibrasi dari Fundamental Matrix:

$$E = K'^T F K$$

$E$ memiliki 5 derajat kebebasan (3 rotasi + 2 translasi, tanpa skala) dan dua singular value yang sama.

### Dekomposisi

$E$ dapat didekomposisi menjadi rotasi $R$ dan translasi $t$:

$$E = [t]_\times R$$

di mana $[t]_\times$ adalah skew-symmetric matrix dari $t$. Dekomposisi menghasilkan 4 solusi $(R_1, t)$, $(R_1, -t)$, $(R_2, t)$, $(R_2, -t)$, dipilih yang semua titik berada di depan kedua kamera.

```python
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
```

---

## 11.5 Triangulasi

### Prinsip

Triangulasi menghitung posisi 3D $P$ dari proyeksi $p$ dan $p'$ pada dua kamera dengan projection matrix $P_1$ dan $P_2$.

$$\lambda_1 p = P_1 P, \quad \lambda_2 p' = P_2 P$$

Sistem overdetermined diselesaikan menggunakan SVD (DLT method).

### Implementasi

```python
pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
pts3d = pts4d[:3] / pts4d[3]  # homogeneous → Euclidean
```

### Error Analysis

Reprojection error mengukur kualitas triangulasi:

$$e = \|p - \pi(P_i, X)\|^2$$

di mana $\pi$ adalah fungsi proyeksi.

---

## 11.6 Stereo Vision

### Konfigurasi Stereo

Dua kamera yang terkalibrasi dan telah diketahui posisi relatifnya (baseline $B$). Setelah **rektifikasi**, pencarian korespondensi menjadi 1D sepanjang baris gambar yang sama.

### Stereo Calibration

```python
# Kalibrasi masing-masing kamera
ret1, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(...)
ret2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(...)

# Kalibrasi stereo
ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, K1, dist1, K2, dist2, imageSize)
```

### Stereo Rectification

Rektifikasi mentransformasi kedua gambar sehingga:
- Epipolar lines horizontal dan sejajar
- Baris yang sama di kedua gambar berkorespondensi

```python
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, d1, K2, d2, imageSize, R, T)
# Uncalibrated:
_, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imageSize)
```

---

## 11.7 Disparity Map

### Definisi

Disparity $d$ adalah perbedaan posisi horizontal titik koresponden antara gambar kiri dan kanan:

$$d = x_L - x_R$$

Objek dekat memiliki disparity besar, objek jauh memiliki disparity kecil.

### StereoBM (Block Matching)

Algoritma sederhana dan cepat:
1. Untuk setiap piksel di gambar kiri, cari match di gambar kanan.
2. Gunakan SAD (Sum of Absolute Differences) dalam window block.
3. Disparity = posisi match terbaik.

```python
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(grayL, grayR)
```

Parameter penting:
- `numDisparities`: range pencarian (kelipatan 16)
- `blockSize`: ukuran window matching (ganjil, 5-21)

### StereoSGBM (Semi-Global Block Matching)

Lebih akurat karena meminimalkan energy function secara semi-global:

$$E(D) = \sum_p C(p, D_p) + \sum_{q \in N_p} P_1 T[|D_p - D_q| = 1] + \sum_{q \in N_p} P_2 T[|D_p - D_q| > 1]$$

- $C$: matching cost
- $P_1$: penalty untuk perubahan disparity kecil (±1)
- $P_2$: penalty untuk perubahan disparity besar

```python
sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=5,
    P1=8*3*blockSize**2, P2=32*3*blockSize**2)
```

---

## 11.8 Depth dari Disparity

### Formula

$$Z = \frac{f \cdot B}{d}$$

di mana:
- $Z$: depth (meter)
- $f$: focal length (piksel)
- $B$: baseline (meter)
- $d$: disparity (piksel)

### Q Matrix

OpenCV menyediakan matriks reprojeksi $Q$ dari `stereoRectify()`:

```python
points3D = cv2.reprojectImageTo3D(disparity, Q)
```

### Resolusi Depth

Resolusi depth (kemampuan membedakan dua kedalaman berbeda):

$$\Delta Z \approx \frac{Z^2}{f \cdot B}$$

Semakin jauh objek, semakin buruk resolusinya.

---

## 11.9 Disparity Post-Processing

### WLS Filter

Weighted Least Squares filter menghaluskan disparity sambil mempertahankan edge:

```python
wls = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
wls.setLambda(8000)      # smoothness
wls.setSigmaColor(1.5)   # edge sensitivity
filtered = wls.filter(disparityL, imgL)
```

### Speckle Filtering

Menghilangkan noise speckle (area kecil dengan disparity error):

```python
cv2.filterSpeckles(disparity, 0, maxSpeckleSize, maxDiff)
```

### Bilateral Filter

Alternatif jika `ximgproc` tidak tersedia:

```python
filtered = cv2.bilateralFilter(disparity_norm, d=9, sigmaColor=75, sigmaSpace=75)
```

---

## 11.10 Monocular Depth Estimation

### Tantangan

Estimasi depth dari satu gambar adalah masalah **ill-posed**: satu gambar 2D memiliki infinite kemungkinan scene 3D. Diperlukan **prior knowledge**:
- Gradien tajam → objek dekat (depth from focus)
- Objek di bawah gambar → lebih dekat (perspektif)
- Ukuran objek yang diketahui → estimasi jarak

### Deep Learning: MiDaS

MiDaS (2020) menghasilkan relative depth map dari gambar tunggal:

```python
model = cv2.dnn.readNet("midas_v21.onnx")
blob = cv2.dnn.blobFromImage(img, 1/255.0, (384,384))
model.setInput(blob)
depth = model.forward()
```

---

## 11.11 PnP (Perspective-n-Point)

### Prinsip

PnP menentukan pose kamera (rotasi $R$ dan translasi $t$) dari $n$ korespondensi antara titik 3D objek dan proyeksi 2D-nya pada gambar.

Minimal diperlukan:
- **4 titik** (P3P + 1 untuk disambiguasi)
- **6+ titik** untuk DLT (Direct Linear Transform)

```python
success, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, K, distCoeffs)
# Dengan RANSAC untuk robustness:
success, rvec, tvec, inliers = cv2.solvePnPRansac(...)
```

### Aplikasi

- Augmented Reality (penempatan objek virtual)
- Robot localization
- Camera tracking

---

## 11.12 Structure from Motion (SfM)

### Pipeline

1. **Feature Detection & Matching**: ORB, SIFT, SuperPoint → korespondensi antar gambar.
2. **Fundamental/Essential Matrix**: menghitung hubungan geometris antar pandangan.
3. **Camera Pose Recovery**: dekomposisi E → R, t.
4. **Triangulasi**: menghitung posisi 3D dari korespondensi 2D.
5. **Bundle Adjustment**: optimasi simultan semua parameter kamera dan titik 3D untuk meminimalkan reprojection error.
6. **Dense Reconstruction**: dari sparse point cloud ke dense mesh.

### Bundle Adjustment

Minimisasi:

$$\min_{R_i, t_i, X_j} \sum_{i,j} \rho\left(\|p_{ij} - \pi(R_i, t_i, X_j)\|^2\right)$$

menggunakan Levenberg-Marquardt optimizer. $\rho$ adalah robust loss function (Huber, Cauchy).

### COLMAP

COLMAP adalah tool SfM state-of-the-art yang mengotomasi seluruh pipeline:
1. Feature extraction (SIFT)
2. Feature matching (exhaustive/sequential)
3. Incremental SfM
4. Multi-View Stereo (MVS)

---

## 11.13 Efek Baseline

### Trade-off Baseline

| Baseline | Kelebihan | Kekurangan |
|----------|-----------|------------|
| Besar | Akurasi depth tinggi di jarak jauh | Overlap area kecil di jarak dekat |
| Kecil | Overlap besar, mudah matching dekat | Akurasi depth buruk di jarak jauh |

Resolusi depth proporsional terhadap $B$:

$$\Delta Z \propto \frac{Z^2}{f \cdot B}$$

---

## 11.14 Depth-based Segmentation

### Prinsip

Segmentasi berdasarkan depth memisahkan objek menurut jaraknya dari kamera. Keuntungan dibanding segmentasi berbasis warna:
- Tidak terpengaruh perubahan iluminasi
- Efektif untuk objek dengan warna serupa background
- Cocok untuk aplikasi robot grasping

### Implementasi

```python
# Segmentasi berdasarkan range depth
near_mask = (depth > 0) & (depth < threshold_near)
mid_mask = (depth >= threshold_near) & (depth < threshold_far)
far_mask = depth >= threshold_far
```

---

## 11.15 Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition, Chapter 11-12.
2. Hartley, R. & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*.
3. Bradski, G. & Kaehler, A. (2008). *Learning OpenCV*.
4. OpenCV Documentation: Stereo Vision, Camera Calibration.
5. Ranftl, R. et al. (2020). *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer* (MiDaS).
6. Schönberger, J.L. & Frahm, J.M. (2016). *Structure-from-Motion Revisited* (COLMAP).
