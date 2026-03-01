# MATERI MODUL 9: ESTIMASI GERAK (MOTION ESTIMATION)

---

## 1. Pendahuluan

Motion estimation adalah proses menghitung pergerakan piksel, objek, atau kamera antar frame dalam sekuens video. Ini adalah fondasi dari video analysis, object tracking, video stabilization, action recognition, dan autonomous navigation.

**Referensi utama**: Szeliski, *Computer Vision: Algorithms and Applications*, 2nd Edition, **Chapter 9 — Motion Estimation**.

---

## 2. Optical Flow

### 2.1 Definisi
Optical flow adalah bidang vektor 2D $(u, v)$ yang menunjukkan perpindahan apparent setiap piksel antara dua frame berturut-turut.

### 2.2 Brightness Constancy Assumption
Asumsi dasar: intensitas piksel tidak berubah saat berpindah.

$$
I(x, y, t) = I(x + u, y + v, t + 1)
$$

Taylor expansion menghasilkan **Optical Flow Constraint Equation**:

$$
I_x u + I_y v + I_t = 0 \quad \Leftrightarrow \quad \nabla I \cdot \mathbf{v} + I_t = 0
$$

di mana $I_x, I_y$ adalah gradien spasial dan $I_t$ adalah gradien temporal.

### 2.3 Aperture Problem
Satu persamaan, dua unknown ($u, v$) → hanya komponen normal flow yang dapat ditentukan. Perlu constraint tambahan.

---

## 3. Lucas-Kanade Optical Flow

### 3.1 Formulasi
Asumsikan seluruh piksel dalam window $W$ memiliki flow yang sama. Solvable sebagai least squares:

$$
\begin{bmatrix} u \\ v \end{bmatrix} = (A^T A)^{-1} A^T \mathbf{b}
$$

di mana:
$$
A = \begin{bmatrix} I_{x_1} & I_{y_1} \\ \vdots & \vdots \\ I_{x_n} & I_{y_n} \end{bmatrix}, \quad \mathbf{b} = -\begin{bmatrix} I_{t_1} \\ \vdots \\ I_{t_n} \end{bmatrix}
$$

### 3.2 Properti
- **Sparse**: Hanya di titik-titik fitur (corners).
- **Akurat** untuk gerakan kecil.
- Memerlukan $A^T A$ invertible → fitur harus di area dengan tekstur (corners).

### 3.3 Pyramidal Lucas-Kanade
Untuk mengatasi gerakan besar, gunakan image pyramid:
1. Estimasi flow di level kasar.
2. Propagasi ke level lebih halus.
3. Refine di setiap level.

```python
# Sparse Optical Flow (Lucas-Kanade)
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                            winSize=(15,15), maxLevel=2)
```

---

## 4. Dense Optical Flow

### 4.1 Horn-Schunck
Menambahkan smoothness constraint (asumsi flow tetangga mirip):

$$
E = \int \int \left[ (I_x u + I_y v + I_t)^2 + \alpha^2 (|\nabla u|^2 + |\nabla v|^2) \right] dx \, dy
$$

Menghasilkan dense flow field untuk seluruh gambar.

### 4.2 Farnebäck
Aproksimasi polinomial kuadratik lokal → dense flow yang cepat.

```python
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                      pyr_scale=0.5, levels=3, winsize=15,
                                      iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
```

### 4.3 Visualisasi Dense Flow
Menggunakan HSV: hue = arah, value = magnitude.

```python
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

---

## 5. Background Subtraction

### 5.1 Tujuan
Memisahkan foreground (objek bergerak) dari background statis.

### 5.2 Metode
- **Frame Differencing**: $|I_t - I_{t-1}| > \theta$
- **Running Average**: $B_{t+1} = \alpha I_t + (1-\alpha) B_t$
- **MOG2 (Mixture of Gaussians)**: Model setiap piksel sebagai campuran Gaussian.
- **KNN**: K-Nearest Neighbors background model.

```python
# MOG2
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
fgmask = fgbg.apply(frame)

# KNN
fgbg_knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400)
fgmask_knn = fgbg_knn.apply(frame)
```

---

## 6. Object Tracking

### 6.1 Tracking Algorithms di OpenCV
| Tracker | Algoritma | Kecepatan | Akurasi |
|---------|-----------|-----------|---------|
| BOOSTING | AdaBoost | Lambat | Rendah |
| MIL | Multiple Instance Learning | Sedang | Sedang |
| KCF | Kernelized Correlation Filter | Cepat | Baik |
| CSRT | Channel & Spatial Reliability | Sedang | Sangat baik |
| MOSSE | Minimum Output Sum of Squared Error | Sangat cepat | Rendah |
| MedianFlow | Median point tracking | Cepat | Baik (gerakan kecil) |

```python
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)
success, bbox = tracker.update(frame)
```

### 6.2 Multi-Object Tracking
```python
multiTracker = cv2.legacy.MultiTracker_create()
for bbox in bboxes:
    multiTracker.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)
success, boxes = multiTracker.update(frame)
```

---

## 7. Motion History Image (MHI)

Representasi temporal dari gerakan: piksel yang baru bergerak lebih terang.

$$
H_\tau(x, y, t) = \begin{cases} \tau & \text{jika } D(x, y, t) = 1 \\ \max(0, H_\tau(x, y, t-1) - \delta) & \text{otherwise} \end{cases}
$$

```python
cv2.motempl.updateMotionHistory(fgmask, mhi, timestamp, duration)
```

---

## 8. Video Stabilization

### 8.1 Pipeline
1. Deteksi fitur antar frame berturut-turut.
2. Estimasi transformasi (translasi atau affine).
3. Akumulasi transformasi kumulatif.
4. Smoothing trajectory (moving average atau Kalman filter).
5. Warp frame berdasarkan koreksi = smooth - kumulatif.

### 8.2 Transformasi
```python
# Estimasi affine antar frame
transform = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
# Dekomposisi: dx, dy, da (translasi + rotasi)
dx = transform[0, 2]
dy = transform[1, 2]
da = np.arctan2(transform[1, 0], transform[0, 0])
```

---

## 9. Frame Interpolation

### 9.1 Tujuan
Menghasilkan frame antara (in-between) untuk slow motion atau frame rate conversion.

### 9.2 Metode
- **Linear interpolation**: Blend 2 frame (ghosting).
- **Flow-based interpolation**: Warp frame berdasarkan optical flow.
- **Bidirectional flow**: Flow forward + backward → warp keduanya ke waktu target.

$$
I_{t+\alpha} = (1-\alpha) \cdot \text{warp}(I_t, \alpha \cdot F_{t \to t+1}) + \alpha \cdot \text{warp}(I_{t+1}, (1-\alpha) \cdot F_{t+1 \to t})
$$

---

## 10. Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed., Chapter 9.
2. Lucas, B. & Kanade, T. (1981). *An Iterative Image Registration Technique with an Application to Stereo Vision*. IJCAI.
3. Horn, B. & Schunck, B. (1981). *Determining Optical Flow*. Artificial Intelligence.
4. Farnebäck, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion*. SCIA.
5. Zivkovic, Z. (2004). *Improved Adaptive Gaussian Mixture Model for Background Subtraction*. ICPR.
6. Bolme, D. et al. (2010). *Visual Object Tracking using Adaptive Correlation Filters*. CVPR.
7. Lukezic, A. et al. (2017). *Discriminative Correlation Filter with Channel and Spatial Reliability*. CVPR.
8. Davis, J. & Bobick, A. (1997). *The Representation and Recognition of Human Movement Using Temporal Templates*. CVPR.
