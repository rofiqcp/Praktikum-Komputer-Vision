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

## 9. Konversi Disparity ke Depth

### 9.1 Hubungan Matematis
Dari geometri stereo, depth $Z$ diperoleh dari disparity $d$:

$$
Z = \frac{f \cdot B}{d}
$$

di mana:
- $f$ = focal length dalam piksel
- $B$ = baseline (jarak antar kamera) dalam meter
- $d$ = disparity dalam piksel

**Catatan penting**: Saat $d = 0$ (tidak ada korespondensi ditemukan), depth menjadi **tak hingga**. Nilai ini harus ditangani secara eksplisit untuk menghindari division by zero.

### 9.2 Konversi Manual

```python
import numpy as np

def disparity_to_depth(disparity, focal_length, baseline):
    """Konversi disparity map ke depth map."""
    # Hindari division by zero
    disparity_safe = disparity.copy().astype(np.float32)
    disparity_safe[disparity_safe == 0] = 0.1  # set minimum disparity
    
    depth = (focal_length * baseline) / disparity_safe
    
    # Clip depth ke range yang wajar (misal 0.1m - 100m)
    depth = np.clip(depth, 0.1, 100.0)
    return depth

# Contoh penggunaan
focal_length = 700  # piksel
baseline = 0.12     # 12 cm = 0.12 meter
depth_map = disparity_to_depth(disparity, focal_length, baseline)
```

### 9.3 Menggunakan reprojectImageTo3D()

OpenCV menyediakan fungsi yang memanfaatkan **Q matrix** dari `stereoRectify()`:

```python
# Q matrix diperoleh dari stereoRectify
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, dist1, K2, dist2, imageSize, R, T)

# Reproject disparity ke 3D (menghasilkan X, Y, Z per piksel)
points_3D = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)

# Ekstrak depth (komponen Z)
depth_map = points_3D[:, :, 2]
```

Matriks $Q$ menggabungkan focal length, baseline, dan principal point:

$$
Q = \begin{bmatrix}
1 & 0 & 0 & -c_x \\
0 & 1 & 0 & -c_y \\
0 & 0 & 0 & f \\
0 & 0 & -\frac{1}{B} & \frac{c_x - c_x'}{B}
\end{bmatrix}
$$

### 9.4 Handling Invalid Disparities

```python
# StereoBM/SGBM menghasilkan disparity * 16 (fixed-point)
disparity_float = disparity.astype(np.float32) / 16.0

# Tandai piksel invalid (disparity <= 0 atau di luar range)
invalid_mask = disparity_float <= 0

# Set invalid ke NaN atau nilai khusus
depth_map[invalid_mask] = 0  # atau np.nan
```

### 9.5 Klasifikasi Zona Depth

Depth map dapat dibagi menjadi zona untuk analisis scene:

```python
def classify_depth_zones(depth_map, near_thresh=1.0, far_thresh=5.0):
    """Klasifikasi piksel ke zona near/mid/far."""
    zones = np.zeros_like(depth_map, dtype=np.uint8)
    
    zones[depth_map < near_thresh] = 1      # Near (merah)
    zones[(depth_map >= near_thresh) & 
          (depth_map < far_thresh)] = 2       # Mid (hijau)
    zones[depth_map >= far_thresh] = 3        # Far (biru)
    
    # Visualisasi
    color_zones = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    color_zones[zones == 1] = [0, 0, 255]    # Near = merah
    color_zones[zones == 2] = [0, 255, 0]    # Mid = hijau
    color_zones[zones == 3] = [255, 0, 0]    # Far = biru
    
    return zones, color_zones
```

---

## 10. Post-Processing Disparity (WLS Filter)

### 10.1 Masalah pada Raw Disparity
Disparity map mentah dari StereoBM/SGBM sering mengandung:
- **Noise** (piksel acak dengan nilai salah)
- **Holes** (area tanpa korespondensi)
- **Streaking artifacts** (garis-garis horizontal)
- **Edge bleeding** (depth merembes di batas objek)

### 10.2 WLS (Weighted Least Squares) Filter

WLS filter memperhalus disparity sambil **mempertahankan tepi** (edge-preserving). Filter ini menggabungkan hasil left disparity dan right disparity untuk konsistensi.

Formulasi:

$$
\min_d \sum_p \left[ (d_p - d_p^{raw})^2 + \lambda \sum_{q \in N_p} w_{pq} (d_p - d_q)^2 \right]
$$

di mana $w_{pq}$ adalah bobot berdasarkan kemiripan warna/intensitas piksel, dan $\lambda$ mengontrol smoothness.

### 10.3 Implementasi dengan cv2.ximgproc

```python
import cv2

# Buat left matcher (SGBM)
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=160, blockSize=5,
    P1=8*3*5**2, P2=32*3*5**2,
    disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

# Buat right matcher (otomatis dari left matcher)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# Hitung disparity kiri dan kanan
left_disp = left_matcher.compute(left_img, right_img)
right_disp = right_matcher.compute(right_img, left_img)

# Buat WLS filter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)    # Smoothness (semakin besar = semakin halus)
wls_filter.setSigmaColor(1.5) # Sensitivity terhadap edge

# Terapkan filter
filtered_disp = wls_filter.filter(left_disp, left_img, None, right_disp)
```

### 10.4 Left-Right Consistency Check

Right matcher menghitung disparity dari perspektif kanan. WLS filter membandingkan kedua hasil:
- Piksel **konsisten** (selisih $\leq$ threshold): disparity valid.
- Piksel **inkonsisten**: ditandai sebagai occluded, diisi melalui interpolasi.

```python
# Confidence map dari WLS filter
conf_map = wls_filter.getConfidenceMap()

# Confidence tinggi = disparity reliable
reliable_mask = conf_map > 200  # range 0-255
```

### 10.5 Filter Alternatif

```python
# 1. Median Filter - menghilangkan salt-and-pepper noise
disp_median = cv2.medianBlur(disparity_uint8, 5)

# 2. Bilateral Filter - edge-preserving smoothing
disp_bilateral = cv2.bilateralFilter(disparity_uint8, 9, 75, 75)

# 3. Guided Filter - menggunakan gambar asli sebagai guide
disp_guided = cv2.ximgproc.guidedFilter(
    guide=left_gray, src=disparity_float, radius=9, eps=1e-2)
```

### 10.6 Perbandingan Metode

| Metode | Edge Preservation | Noise Removal | Hole Filling | Kecepatan |
|--------|:-:|:-:|:-:|:-:|
| Median | Rendah | Tinggi | Tidak | Cepat |
| Bilateral | Sedang | Sedang | Tidak | Sedang |
| Guided | Tinggi | Sedang | Tidak | Sedang |
| WLS | Tinggi | Tinggi | Ya | Lambat |

---

## 11. PnP (Perspective-n-Point)

### 11.1 Definisi Masalah
PnP mengestimasi **pose kamera** (rotasi $R$ dan translasi $t$) dari korespondensi antara $n$ titik 3D yang diketahui dan proyeksi 2D-nya di gambar:

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K [R | t] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

### 11.2 Jumlah Titik Minimum
- **P3P**: Membutuhkan tepat 3 titik → menghasilkan hingga 4 solusi. Titik ke-4 untuk disambiguasi.
- **EPnP**: Efisien untuk $n \geq 4$, menggunakan 4 virtual control points.
- **DLS/SQPnP**: Metode modern dengan akurasi tinggi.
- Semakin banyak titik → semakin robust dan akurat.

### 11.3 Metode dalam OpenCV

| Metode | Flag OpenCV | Min Points | Catatan |
|--------|:-:|:-:|:-:|
| Iterative | `cv2.SOLVEPNP_ITERATIVE` | 4 | Levenberg-Marquardt, default |
| P3P | `cv2.SOLVEPNP_P3P` | 4 (3+1) | Cepat, cocok untuk RANSAC |
| EPnP | `cv2.SOLVEPNP_EPNP` | 4 | Efficient, akurat |
| SQPnP | `cv2.SOLVEPNP_SQPNP` | 4 | State-of-the-art akurasi |

### 11.4 Implementasi cv2.solvePnP()

```python
import numpy as np
import cv2

# Titik 3D di world coordinate (misal dari triangulasi sebelumnya)
object_points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.5, 0.5, 0.5]
], dtype=np.float64)

# Titik 2D yang terobservasi di gambar
image_points = np.array([
    [320, 240],
    [420, 240],
    [320, 340],
    [420, 340],
    [370, 290]
], dtype=np.float64)

# Intrinsic kamera
K = np.array([[700, 0, 320],
              [0, 700, 240],
              [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.zeros(4)

# Solve PnP
success, rvec, tvec = cv2.solvePnP(
    object_points, image_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

print(f"Rotation vector: {rvec.flatten()}")
print(f"Translation vector: {tvec.flatten()}")
```

### 11.5 PnP dengan RANSAC

Untuk menangani outlier pada korespondensi:

```python
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    object_points, image_points, K, dist_coeffs,
    iterationsCount=1000,
    reprojectionError=3.0,   # threshold piksel
    confidence=0.99,
    flags=cv2.SOLVEPNP_EPNP)

print(f"Inliers: {len(inliers)} / {len(object_points)}")
```

### 11.6 Konversi Rodrigues

OpenCV menggunakan **rotation vector** (3×1, Rodrigues representation). Konversi ke rotation matrix:

$$
R = I + \sin\theta \cdot [\hat{r}]_\times + (1 - \cos\theta) \cdot [\hat{r}]_\times^2
$$

di mana $\theta = \|r\|$ dan $\hat{r} = r / \theta$.

```python
# Rotation vector → Rotation matrix
R, _ = cv2.Rodrigues(rvec)

# Rotation matrix → Rotation vector
rvec_back, _ = cv2.Rodrigues(R)
```

### 11.7 Reprojection Error

Metrik kualitas: seberapa dekat titik 3D yang diproyeksikan ulang ke titik 2D observasi:

$$
\text{error} = \frac{1}{n} \sum_{i=1}^{n} \| \mathbf{x}_i - \pi(R, t, \mathbf{X}_i) \|_2
$$

```python
# Proyeksikan titik 3D ke gambar menggunakan pose yang ditemukan
projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)

# Hitung reprojection error
errors = np.linalg.norm(image_points - projected.reshape(-1, 2), axis=1)
mean_error = np.mean(errors)
print(f"Mean reprojection error: {mean_error:.2f} piksel")
```

### 11.8 Aplikasi dalam SfM

PnP digunakan untuk **menambahkan kamera baru** ke rekonstruksi SfM yang sudah ada:
1. Dari kamera-kamera sebelumnya, titik 3D telah ditriangulasi.
2. Kamera baru mendeteksi fitur yang cocok dengan titik 3D tersebut.
3. `solvePnPRansac()` mengestimasi pose kamera baru.
4. Titik 3D baru ditriangulasi dari kamera baru.
5. Bundle adjustment memperbaiki semua parameter.

---

## 12. Point Cloud dari Depth Map

### 12.1 Backprojection 2D ke 3D

Setiap piksel $(u, v)$ dengan depth $Z$ dapat di-backproject ke koordinat 3D:

$$
X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}
$$

di mana $(c_x, c_y)$ adalah principal point dan $(f_x, f_y)$ adalah focal length.

### 12.2 Implementasi Manual

```python
import numpy as np

def depth_to_pointcloud(depth_map, K):
    """Konversi depth map ke point cloud menggunakan intrinsic matrix K."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    Z = depth_map.astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack menjadi (N, 3) point cloud
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return points

K = np.array([[700, 0, 320],
              [0, 700, 240],
              [0, 0, 1]], dtype=np.float32)

pointcloud = depth_to_pointcloud(depth_map, K)
```

### 12.3 Menggunakan reprojectImageTo3D()

```python
# Dari disparity + Q matrix
points_3D = cv2.reprojectImageTo3D(disparity_float, Q, handleMissingValues=True)

# Reshape menjadi (N, 3)
pointcloud = points_3D.reshape(-1, 3)
```

### 12.4 Menambahkan Warna dari Gambar Asli

```python
def create_colored_pointcloud(depth_map, color_image, K):
    """Buat point cloud berwarna."""
    points = depth_to_pointcloud(depth_map, K)
    
    # Ambil warna dari gambar (BGR → RGB)
    colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    colors = colors.reshape(-1, 3) / 255.0  # Normalisasi ke [0, 1]
    
    # Filter titik invalid (depth = 0 atau terlalu jauh)
    valid = (points[:, 2] > 0.1) & (points[:, 2] < 50.0)
    points = points[valid]
    colors = colors[valid]
    
    return points, colors
```

### 12.5 Filtering Outlier

```python
def filter_pointcloud(points, colors=None, z_min=0.1, z_max=50.0, 
                       statistical=True, nb_neighbors=20, std_ratio=2.0):
    """Filter outlier dari point cloud."""
    # 1. Depth range filter
    valid = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    points = points[valid]
    if colors is not None:
        colors = colors[valid]
    
    # 2. Statistical outlier removal (manual)
    if statistical and len(points) > nb_neighbors:
        from scipy.spatial import KDTree
        tree = KDTree(points)
        distances, _ = tree.query(points, k=nb_neighbors + 1)
        mean_dist = np.mean(distances[:, 1:], axis=1)
        threshold = np.mean(mean_dist) + std_ratio * np.std(mean_dist)
        inlier_mask = mean_dist < threshold
        points = points[inlier_mask]
        if colors is not None:
            colors = colors[inlier_mask]
    
    return points, colors
```

### 12.6 Visualisasi dengan Matplotlib

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pointcloud(points, colors=None, max_points=50000):
    """Visualisasi point cloud dengan matplotlib 3D scatter."""
    # Subsample jika terlalu banyak titik
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors if colors is not None else points[:, 2],
               cmap='viridis' if colors is None else None,
               s=0.5, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')
    
    # Set equal aspect ratio
    max_range = np.ptp(points, axis=0).max() / 2
    mid = np.mean(points, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    plt.tight_layout()
    plt.show()

# Penggunaan
points, colors = create_colored_pointcloud(depth_map, left_img, K)
points, colors = filter_pointcloud(points, colors)
visualize_pointcloud(points, colors)
```

---

## 13. Depth Map Visualization

### 13.1 Colormaps untuk Depth

Colormap memetakan nilai skalar (depth) ke warna untuk visualisasi intuitif:

| Colormap | Flag OpenCV | Karakteristik |
|----------|:-:|:-:|
| JET | `cv2.COLORMAP_JET` | Klasik, biru (dekat) → merah (jauh) |
| INFERNO | `cv2.COLORMAP_INFERNO` | Gelap → terang, perceptually uniform |
| VIRIDIS | `cv2.COLORMAP_VIRIDIS` | Hijau-biru, baik untuk colorblind |
| PLASMA | `cv2.COLORMAP_PLASMA` | Ungu → kuning, vivid |
| HOT | `cv2.COLORMAP_HOT` | Hitam → merah → kuning → putih |
| BONE | `cv2.COLORMAP_BONE` | Grayscale + sedikit biru |

### 13.2 Aplikasi Colormap

```python
import cv2
import numpy as np

def apply_depth_colormap(depth_map, colormap=cv2.COLORMAP_JET):
    """Terapkan colormap pada depth map."""
    # Normalisasi depth ke 0-255
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    
    # Terapkan colormap
    depth_colored = cv2.applyColorMap(depth_uint8, colormap)
    return depth_colored

# Contoh berbagai colormap
colormaps = {
    'JET': cv2.COLORMAP_JET,
    'INFERNO': cv2.COLORMAP_INFERNO,
    'VIRIDIS': cv2.COLORMAP_VIRIDIS,
    'PLASMA': cv2.COLORMAP_PLASMA,
    'HOT': cv2.COLORMAP_HOT,
    'BONE': cv2.COLORMAP_BONE
}

for name, cmap in colormaps.items():
    colored = apply_depth_colormap(depth_map, cmap)
    cv2.imshow(f'Depth - {name}', colored)
```

### 13.3 Depth Overlay (Semi-Transparent)

Overlay depth berwarna di atas gambar asli untuk konteks spasial:

```python
def depth_overlay(image, depth_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay depth map semi-transparan pada gambar asli."""
    depth_colored = apply_depth_colormap(depth_map, colormap)
    
    # Pastikan ukuran sama
    if depth_colored.shape[:2] != image.shape[:2]:
        depth_colored = cv2.resize(depth_colored, 
                                    (image.shape[1], image.shape[0]))
    
    # Blending
    overlay = cv2.addWeighted(image, 1 - alpha, depth_colored, alpha, 0)
    return overlay

overlay = depth_overlay(left_img, depth_map, alpha=0.4)
cv2.imshow('Depth Overlay', overlay)
```

### 13.4 Depth Contours (Iso-Depth Lines)

Menggambar kontur pada kedalaman tertentu (seperti peta topografi):

```python
def draw_depth_contours(image, depth_map, num_levels=10):
    """Gambar iso-depth contours pada gambar."""
    result = image.copy()
    
    # Normalisasi depth
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    
    # Tentukan level depth
    d_min, d_max = depth_map[depth_map > 0].min(), depth_map.max()
    levels = np.linspace(d_min, d_max, num_levels + 2)[1:-1]
    
    colors = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0),
              (255, 255, 0), (255, 128, 0), (255, 0, 0), (255, 0, 128),
              (128, 0, 255), (255, 255, 255)]
    
    for i, level in enumerate(levels):
        # Threshold pada level tertentu
        mask = (depth_map > level - 0.2) & (depth_map < level + 0.2)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        color = colors[i % len(colors)]
        cv2.drawContours(result, contours, -1, color, 1)
        
        # Label depth
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result, f"{level:.1f}m", (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return result
```

### 13.5 Depth Histogram

Analisis distribusi depth dalam scene:

```python
import matplotlib.pyplot as plt

def depth_histogram(depth_map, bins=100):
    """Analisis dan tampilkan histogram depth."""
    # Ambil hanya depth valid
    valid_depth = depth_map[depth_map > 0].flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(valid_depth, bins=bins, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Depth (meter)')
    axes[0].set_ylabel('Jumlah Piksel')
    axes[0].set_title('Distribusi Depth')
    axes[0].axvline(np.median(valid_depth), color='red', linestyle='--',
                     label=f'Median: {np.median(valid_depth):.2f}m')
    axes[0].legend()
    
    # Cumulative distribution
    axes[1].hist(valid_depth, bins=bins, cumulative=True, density=True,
                  color='darkorange', edgecolor='black')
    axes[1].set_xlabel('Depth (meter)')
    axes[1].set_ylabel('Proporsi Kumulatif')
    axes[1].set_title('CDF Depth')
    
    plt.tight_layout()
    plt.show()
    
    # Statistik
    print(f"Depth min: {valid_depth.min():.2f}m")
    print(f"Depth max: {valid_depth.max():.2f}m")
    print(f"Depth mean: {valid_depth.mean():.2f}m")
    print(f"Depth median: {np.median(valid_depth):.2f}m")
    print(f"Depth std: {valid_depth.std():.2f}m")
```

---

## 14. Pengaruh Baseline pada Depth Estimation

### 14.1 Baseline dan Resolusi Depth

Baseline $B$ memiliki pengaruh krusial terhadap kualitas depth estimation. Dari persamaan $Z = f \cdot B / d$, perubahan depth per satu piksel disparity:

$$
\Delta Z = \frac{f \cdot B}{d^2} \cdot \Delta d
$$

**Baseline terlalu kecil** ($B$ kecil):
- Range disparity sempit → resolusi depth rendah.
- Sulit membedakan objek pada depth berbeda.
- Noise disparity (±1 piksel) menyebabkan error depth besar.

**Baseline terlalu besar** ($B$ besar):
- Area **overlap** antara kamera kiri-kanan mengecil.
- Banyak area **occluded** (terlihat di satu kamera, tidak di kamera lain).
- Stereo matching lebih sulit karena perbedaan perspektif besar.

### 14.2 Sweet Spot Baseline

Aturan praktis:

$$
B \approx \frac{1}{30} \times Z_{target}
$$

di mana $Z_{target}$ adalah jarak ke scene yang menjadi fokus. Contoh:
- Scene berjarak 3 meter → baseline ≈ 10 cm
- Scene berjarak 30 meter → baseline ≈ 1 meter

### 14.3 Error Depth vs Jarak

Error depth tumbuh **kuadratik** terhadap jarak:

$$
\sigma_Z = \frac{Z^2}{f \cdot B} \cdot \sigma_d
$$

di mana $\sigma_d$ adalah error disparity (biasanya 0.5–1 piksel).

| Jarak $Z$ | Baseline $B$ | $f$ (piksel) | $\sigma_d$ | $\sigma_Z$ |
|:-:|:-:|:-:|:-:|:-:|
| 1 m | 0.12 m | 700 | 0.5 px | 0.006 m |
| 5 m | 0.12 m | 700 | 0.5 px | 0.149 m |
| 10 m | 0.12 m | 700 | 0.5 px | 0.595 m |
| 20 m | 0.12 m | 700 | 0.5 px | 2.381 m |

### 14.4 Simulasi Pengaruh Baseline

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_baseline_effect():
    """Simulasi pengaruh baseline terhadap depth error."""
    focal = 700          # piksel
    sigma_d = 0.5        # error disparity (piksel)
    Z = np.linspace(0.5, 30, 100)  # jarak 0.5 - 30 meter
    
    baselines = [0.05, 0.12, 0.25, 0.5, 1.0]  # meter
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Depth error vs distance
    plt.subplot(1, 2, 1)
    for B in baselines:
        sigma_Z = (Z ** 2) / (focal * B) * sigma_d
        plt.plot(Z, sigma_Z, label=f'B = {B*100:.0f} cm')
    
    plt.xlabel('Jarak Z (meter)')
    plt.ylabel('Depth Error σ_Z (meter)')
    plt.title('Depth Error vs Jarak (untuk berbagai baseline)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 5)
    
    # Plot 2: Disparity range vs baseline
    plt.subplot(1, 2, 2)
    Z_range = np.array([1, 2, 5, 10, 20])
    for B in baselines:
        disparities = focal * B / Z_range
        plt.plot(Z_range, disparities, 'o-', label=f'B = {B*100:.0f} cm')
    
    plt.xlabel('Jarak Z (meter)')
    plt.ylabel('Disparity (piksel)')
    plt.title('Disparity vs Jarak (untuk berbagai baseline)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

simulate_baseline_effect()
```

### 14.5 Implikasi Praktis

```python
def recommend_baseline(target_distance, focal_length=700, 
                        max_depth_error_percent=5, disparity_error=0.5):
    """Rekomendasikan baseline berdasarkan target distance."""
    max_Z_error = target_distance * (max_depth_error_percent / 100)
    
    # Dari sigma_Z = Z^2 / (f * B) * sigma_d
    # B_min = Z^2 * sigma_d / (f * max_Z_error)
    B_min = (target_distance ** 2 * disparity_error) / (focal_length * max_Z_error)
    
    # Rule of thumb
    B_rule = target_distance / 30
    
    B_recommended = max(B_min, B_rule)
    
    print(f"Target distance: {target_distance:.1f} m")
    print(f"Max depth error ({max_depth_error_percent}%): {max_Z_error:.3f} m")
    print(f"Minimum baseline: {B_min*100:.1f} cm")
    print(f"Rule-of-thumb baseline: {B_rule*100:.1f} cm")
    print(f"Recommended baseline: {B_recommended*100:.1f} cm")
    
    return B_recommended

# Contoh
recommend_baseline(3.0)   # Indoor scene
recommend_baseline(20.0)  # Outdoor scene
```

---

## 15. Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed., Chapter 11-12.
2. Hartley, R. & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*. Cambridge University Press.
3. Nistér, D. (2004). *An Efficient Solution to the Five-Point Relative Pose Problem*. TPAMI.
4. Hirschmüller, H. (2005). *Accurate and Efficient Stereo Processing by Semi-Global Matching*. CVPR.
5. Scharstein, D. & Szeliski, R. (2002). *A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms*. IJCV.
6. Ranftl, R. et al. (2020). *Towards Robust Monocular Depth Estimation*. TPAMI (MiDaS).
7. Mur-Artal, R. et al. (2015). *ORB-SLAM: A Versatile and Accurate Monocular SLAM System*. TRO.
8. Longuet-Higgins, H. (1981). *A Computer Algorithm for Reconstructing a Scene from Two Projections*. Nature.
