# MATERI MODUL 11: STRUCTURE FROM MOTION DAN DEPTH ESTIMATION

---

## 1. Pendahuluan

Structure from Motion (SfM) merekonstruksi geometri 3D scene dan posisi kamera dari sekumpulan gambar 2D. Depth Estimation memperkirakan kedalaman setiap piksel dari satu atau dua gambar (stereo). Keduanya merupakan fondasi 3D computer vision.

**Referensi utama**: Szeliski, *Computer Vision: Algorithms and Applications*, 2nd Edition, **Chapter 11 — Structure from Motion** dan **Chapter 12 — Depth Estimation**.

---

## 2. Epipolar Geometry

### 2.1 Konsep Dasar
Dua kamera melihat titik 3D yang sama menghasilkan pasangan titik gambar terkait oleh **epipolar constraint**. Titik di gambar kiri berkorespondensi dengan garis (epipolar line) di gambar kanan.

### 2.2 Fundamental Matrix (F)
Menghubungkan titik koresponden dalam pixel coordinates:

$$
\mathbf{x'}^T F \mathbf{x} = 0
$$

- $F$ adalah matriks 3×3, rank 2.
- Berisi informasi rotasi, translasi, dan intrinsik kamera.
- Dihitung dari minimal 8 korespondensi titik (**8-point algorithm**).

```python
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
```

### 2.3 Essential Matrix (E)
Jika intrinsik kamera diketahui ($K$):

$$
E = K'^T F K
$$

$$
\mathbf{\hat{x}'}^T E \mathbf{\hat{x}} = 0
$$

di mana $\hat{\mathbf{x}} = K^{-1}\mathbf{x}$ adalah normalized coordinates.

**Dekomposisi**: $E = [t]_\times R$ → menghasilkan rotasi ($R$) dan translasi ($t$) kamera.

```python
E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
```

### 2.4 Epipolar Lines
```python
# Gambar epipolar lines
lines = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
for r, pt in zip(lines.reshape(-1, 3), pts1):
    x0, y0 = map(int, [0, -r[2]/r[1]])
    x1, y1 = map(int, [w, -(r[2]+r[0]*w)/r[1]])
    cv2.line(img1, (x0,y0), (x1,y1), (0,255,0), 1)
```

---

## 3. Triangulasi

### 3.1 Prinsip
Diberikan dua kamera dengan posisi diketahui dan titik koresponden, posisi 3D titik dapat dihitung dari interseksi dua ray.

$$
\mathbf{x} = P \mathbf{X}, \quad \mathbf{x}' = P' \mathbf{X}
$$

Karena noise, ray tidak selalu berpotongan → gunakan least squares (linear triangulation atau optimal Sampson).

```python
P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = K @ np.hstack((R, t))
points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points3D = points4D[:3] / points4D[3]  # Homogeneous → Euclidean
```

---

## 4. Structure from Motion (SfM)

### 4.1 Pipeline
1. **Feature Detection & Matching** antar pasangan gambar.
2. **Estimate Fundamental/Essential Matrix** → recover relative pose.
3. **Triangulate** → initial 3D points.
4. **Add cameras incrementally** → PnP (Perspective-n-Point).
5. **Bundle Adjustment** → optimasi simultan semua kamera + titik 3D.

### 4.2 PnP (Perspective-n-Point)
Mengestimasi pose kamera baru dari korespondensi 3D-2D:

```python
success, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, K, dist_coeffs)
R, _ = cv2.Rodrigues(rvec)
```

### 4.3 Bundle Adjustment
Minimasi reprojection error secara global:

$$
\min_{R_i, t_i, X_j} \sum_{i,j} \| \mathbf{x}_{ij} - \pi(R_i, t_i, X_j) \|^2
$$

Menggunakan sparse Levenberg-Marquardt (karena Jacobian sparse).

---

## 5. Visual Odometry

### 5.1 Konsep
Estimasi gerakan kamera frame-by-frame dari video, menghasilkan trajectory kamera.

### 5.2 Pipeline
1. Feature detection di frame $t$.
2. Track features ke frame $t+1$ (Lucas-Kanade atau matching).
3. Essential matrix → recover $R, t$.
4. Akumulasi pose: $T_{world} = T_{world} \cdot T_{relative}$.

---

## 6. Stereo Vision

### 6.1 Stereo Setup
Dua kamera paralel dipisahkan oleh baseline $b$. Disparity $d$ terkait dengan depth $Z$:

$$
Z = \frac{f \cdot b}{d}
$$

di mana $f$ = focal length (piksel), $b$ = baseline (meter), $d$ = disparity (piksel).

### 6.2 Stereo Calibration
Kalibrasi intrinsic + extrinsic dua kamera secara bersamaan:

```python
retval, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, K1, dist1, K2, dist2, imageSize)
```

### 6.3 Stereo Rectification
Transformasi kedua gambar agar epipolar lines horizontal (paralel):

```python
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, dist1, K2, dist2, imageSize, R, T)
map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, imageSize, cv2.CV_32FC1)
rectified1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
```

---

## 7. Stereo Matching

### 7.1 Block Matching (BM)
Cari best match menggunakan Sum of Absolute Differences (SAD) dalam window:

```python
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left_gray, right_gray)
```

### 7.2 Semi-Global Matching (SGM / SGBM)
Optimasi global energi cost + smoothness penalty dari 8 arah:

$$
E(D) = \sum_p \left[ C(p, D_p) + \sum_{q \in N_p} P_1 \cdot T[|D_p - D_q|=1] + P_2 \cdot T[|D_p - D_q|>1] \right]
$$

```python
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=5,
    P1=8*3*5**2, P2=32*3*5**2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disparity = stereo.compute(left, right)
```

### 7.3 Disparity ke Depth
```python
# Menggunakan Q matrix dari stereoRectify
points_3D = cv2.reprojectImageTo3D(disparity, Q)
```

---

## 8. Monocular Depth Estimation

### 8.1 Konsep
Estimasi depth map dari satu gambar menggunakan deep learning. Model dilatih pada pasangan gambar-depth.

### 8.2 MiDaS (Intel)
Model pre-trained untuk monocular depth estimation:

```python
model = cv2.dnn.readNet("model-small.onnx")
blob = cv2.dnn.blobFromImage(img, 1/255.0, (256, 256), (0.485, 0.456, 0.406), True, False)
model.setInput(blob)
depth = model.forward()
```

### 8.3 Limitasi
- Depth relatif (bukan metrik absolut).
- Akurasi bergantung pada scene similarity dengan training data.
- Struggle di area repetitive textures dan refleksi.

---

## 9. Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed., Chapter 11-12.
2. Hartley, R. & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*. Cambridge University Press.
3. Nistér, D. (2004). *An Efficient Solution to the Five-Point Relative Pose Problem*. TPAMI.
4. Hirschmüller, H. (2005). *Accurate and Efficient Stereo Processing by Semi-Global Matching*. CVPR.
5. Scharstein, D. & Szeliski, R. (2002). *A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms*. IJCV.
6. Ranftl, R. et al. (2020). *Towards Robust Monocular Depth Estimation*. TPAMI (MiDaS).
7. Mur-Artal, R. et al. (2015). *ORB-SLAM: A Versatile and Accurate Monocular SLAM System*. TRO.
8. Longuet-Higgins, H. (1981). *A Computer Algorithm for Reconstructing a Scene from Two Projections*. Nature.
