# MATERI MODUL 7: ESTIMASI GERAK (MOTION ESTIMATION)

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

### 7.1 Tujuan
Menghasilkan frame antara (in-between) untuk slow motion atau frame rate conversion.

### 7.2 Metode
- **Linear interpolation**: Blend 2 frame (ghosting).
- **Flow-based interpolation**: Warp frame berdasarkan optical flow.
- **Bidirectional flow**: Flow forward + backward → warp keduanya ke waktu target.

$$
I_{t+\alpha} = (1-\alpha) \cdot \text{warp}(I_t, \alpha \cdot F_{t \to t+1}) + \alpha \cdot \text{warp}(I_{t+1}, (1-\alpha) \cdot F_{t+1 \to t})
$$

---

## 10. Magnitude dan Arah Optical Flow

### 10.1 Analisis Kuantitatif Vektor Flow
Setiap piksel pada dense optical flow memiliki komponen horizontal $u$ dan vertikal $v$. Dari dua komponen ini, kita dapat menghitung **magnitude** (kecepatan) dan **arah** (direction) pergerakan.

### 10.2 Perhitungan Magnitude dan Arah
**Magnitude** menunjukkan seberapa besar perpindahan piksel:

$$
\text{magnitude} = \sqrt{u^2 + v^2}
$$

**Arah** menunjukkan ke mana piksel bergerak:

$$
\theta = \arctan2(v, u)
$$

Di mana $\theta$ dalam radian, dengan rentang $[-\pi, \pi]$.

### 10.3 Visualisasi HSV
Representasi HSV sangat efektif untuk menampilkan magnitude dan arah secara bersamaan:
- **Hue** = arah gerakan (warna menunjukkan ke mana objek bergerak).
- **Saturation** = 255 (penuh, agar warna terlihat jelas).
- **Value** = magnitude (area yang bergerak cepat lebih terang).

```python
# Hitung dense optical flow
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                      0.5, 3, 15, 3, 5, 1.2, 0)

# Konversi ke magnitude dan arah
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

# Buat visualisasi HSV
hsv = np.zeros((*prev_gray.shape, 3), dtype=np.uint8)
hsv[..., 0] = ang * 180 / np.pi / 2      # Hue: arah (0-180 untuk OpenCV)
hsv[..., 1] = 255                          # Saturation: penuh
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude

rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

### 10.4 Statistik Flow
Analisis statistik dari magnitude dan arah memberikan informasi kuantitatif tentang gerakan dalam scene:

- **Histogram magnitude**: Distribusi kecepatan piksel — berguna untuk menentukan threshold gerakan.
- **Rata-rata magnitude**: Indikator umum seberapa banyak gerakan dalam frame.
- **Arah dominan**: Arah gerakan utama, dihitung dari histogram arah atau circular mean.

```python
# Statistik magnitude
mean_mag = np.mean(mag)
max_mag = np.max(mag)

# Threshold: hanya piksel yang bergerak signifikan
motion_mask = mag > 2.0  # threshold magnitude

# Arah dominan dari piksel yang bergerak
dominant_angles = ang[motion_mask]
hist_ang, bins = np.histogram(dominant_angles, bins=36, range=(0, 2*np.pi))
dominant_direction = bins[np.argmax(hist_ang)]
```

---

## 11. Feature Trajectory Tracking

### 11.1 Tracking Jangka Panjang
Berbeda dengan optical flow antar dua frame, **feature trajectory tracking** melacak titik-titik fitur secara kontinu melintasi banyak frame. Ini menghasilkan lintasan (trajectory) yang merepresentasikan jalur pergerakan objek dari waktu ke waktu.

### 11.2 Siklus Hidup Track
Setiap fitur yang dilacak memiliki siklus hidup:

1. **Detection**: Fitur baru dideteksi menggunakan corner detector (e.g., `goodFeaturesToTrack`).
2. **Tracking**: Posisi fitur diestimasi di frame berikutnya menggunakan Lucas-Kanade.
3. **Loss**: Fitur hilang jika status tracking gagal atau keluar dari frame.
4. **Re-detection**: Fitur baru dideteksi untuk menggantikan yang hilang.

### 11.3 Akumulasi Trajectory
```python
# Inisialisasi
tracks = []  # list of trajectory, setiap trajectory = list of (x, y)
detect_interval = 5  # re-detect setiap N frame

p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)

# Inisialisasi tracks dari titik awal
for p in p0:
    tracks.append([tuple(p.ravel())])

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(tracks) > 0:
        # Ambil titik terakhir dari setiap track
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None,
                                                     winSize=(15, 15), maxLevel=2)

        # Update tracks yang berhasil
        new_tracks = []
        for tr, (x, y), st in zip(tracks, p1.reshape(-1, 2), status.ravel()):
            if st == 1:
                tr.append((x, y))
                new_tracks.append(tr)
        tracks = new_tracks

    # Re-detect fitur secara periodik
    if frame_idx % detect_interval == 0:
        p_new = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if p_new is not None:
            for p in p_new:
                tracks.append([tuple(p.ravel())])

    old_gray = gray.copy()
    frame_idx += 1
```

### 11.4 Analisis Trajectory
Dari trajectory yang terkumpul, berbagai metrik dapat dihitung:

- **Panjang lintasan (path length)**: Total jarak yang ditempuh fitur.

$$
L = \sum_{i=1}^{N-1} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}
$$

- **Smoothness**: Seberapa halus lintasan, diukur dari variasi arah antar segmen.
- **Kecepatan rata-rata**: $\bar{v} = L / N$ (piksel per frame).

```python
def trajectory_length(traj):
    length = 0
    for i in range(1, len(traj)):
        dx = traj[i][0] - traj[i-1][0]
        dy = traj[i][1] - traj[i-1][1]
        length += np.sqrt(dx**2 + dy**2)
    return length
```

---

## 12. Perbandingan Metode Background Subtraction

### 12.1 Overview Metode
Terdapat beberapa metode background subtraction yang umum digunakan, masing-masing dengan kelebihan dan kekurangan.

### 12.2 Frame Differencing
Metode paling sederhana: menghitung perbedaan absolut antar frame berturut-turut.

$$
M(x, y) = |I_t(x, y) - I_{t-1}(x, y)| > \theta
$$

- **Kelebihan**: Sangat cepat, mudah diimplementasi.
- **Kekurangan**: Tidak menangkap objek diam, sensitif terhadap noise.

### 12.3 Running Average
Background diestimasi sebagai rata-rata bergerak eksponensial:

$$
B_{t+1}(x, y) = \alpha \cdot I_t(x, y) + (1 - \alpha) \cdot B_t(x, y)
$$

- **Kelebihan**: Adaptif terhadap perubahan pencahayaan gradual.
- **Kekurangan**: Parameter $\alpha$ harus di-tuning; objek diam lama akan masuk ke background.

### 12.4 MOG2 (Mixture of Gaussians)
Setiap piksel dimodelkan sebagai campuran $K$ distribusi Gaussian. Distribusi yang paling stabil dianggap background.

- **Kelebihan**: Menangani scene dinamis (daun bergoyang, air), deteksi shadow.
- **Kekurangan**: Lebih lambat, memerlukan lebih banyak memori.

### 12.5 KNN (K-Nearest Neighbors)
Background model berbasis sampel historis. Piksel diklasifikasi berdasarkan jarak ke $K$ sampel terdekat.

- **Kelebihan**: Baik untuk scene dengan background bergerak, shadow detection.
- **Kekurangan**: Konsumsi memori tinggi.

### 12.6 Tabel Perbandingan

| Kriteria | Frame Diff | Running Avg | MOG2 | KNN |
|----------|-----------|-------------|------|-----|
| **Kecepatan** | Sangat cepat | Cepat | Sedang | Sedang |
| **Akurasi** | Rendah | Sedang | Tinggi | Tinggi |
| **Robustness noise** | Rendah | Sedang | Tinggi | Tinggi |
| **Shadow handling** | Tidak | Tidak | Ya | Ya |
| **Background dinamis** | Tidak | Sebagian | Ya | Ya |
| **Memori** | Minimal | Rendah | Sedang | Tinggi |

### 12.7 Kapan Menggunakan Setiap Metode
- **Frame Differencing**: Prototipe cepat, deteksi gerakan kasar, resource terbatas.
- **Running Average**: Scene dengan pencahayaan berubah gradual, kamera statis.
- **MOG2**: Aplikasi umum surveillance, scene outdoor dengan background dinamis.
- **KNN**: Mirip MOG2, alternatif saat MOG2 kurang baik pada scene tertentu.

```python
# Perbandingan dalam kode
methods = {
    'MOG2': cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True),
    'KNN': cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True),
}

# Frame Differencing manual
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
diff_mask = cv2.absdiff(prev_gray, curr_gray)
_, frame_diff = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY)

# Running Average
alpha = 0.05
bg_model = np.float32(prev_gray)
cv2.accumulateWeighted(curr_gray, bg_model, alpha)
running_avg_mask = cv2.absdiff(curr_gray, cv2.convertScaleAbs(bg_model))
_, running_avg_mask = cv2.threshold(running_avg_mask, 30, 255, cv2.THRESH_BINARY)
```

---

## 13. Deteksi Gerakan Berbasis Contour

### 13.1 Dari Motion Mask ke Deteksi Objek
Background subtraction menghasilkan **motion mask** (binary image). Langkah selanjutnya adalah mengidentifikasi objek individual dari mask tersebut menggunakan analisis contour.

### 13.2 Pipeline Deteksi
Pipeline lengkap dari background subtraction hingga deteksi objek:

1. **Background Subtraction**: Hasilkan foreground mask.
2. **Morphological Operations**: Bersihkan noise dan isi lubang.
3. **Find Contours**: Temukan kontur objek.
4. **Bounding Rectangle**: Gambar kotak pembatas di sekitar objek.

```python
# 1. Background subtraction
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
fgmask = fgbg.apply(frame)

# Hilangkan shadow (shadow bernilai 127 di MOG2)
_, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

# 2. Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)   # hilangkan noise kecil
fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)  # isi lubang

# 3. Find contours
contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Filter dan gambar bounding box
min_area = 500  # minimum area dalam piksel
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

### 13.3 Filtering Objek
Tidak semua contour adalah objek yang relevan. Beberapa teknik filtering:

- **Area filtering**: Abaikan contour dengan area terlalu kecil (noise) atau terlalu besar (seluruh frame).
- **Aspect ratio filtering**: Rasio $w/h$ — misalnya manusia memiliki rasio tertentu ($0.3 < w/h < 0.8$).
- **Solidity**: Rasio area contour terhadap convex hull area → menghilangkan bentuk tidak wajar.

```python
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500 or area > 50000:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    if aspect_ratio < 0.2 or aspect_ratio > 5.0:
        continue

    # Solidity check
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < 0.3:
        continue

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

### 13.4 Counting dan Centroid Tracking
Untuk menghitung jumlah objek bergerak dan melacak posisinya:

```python
object_count = 0
centroids = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > min_area:
        object_count += 1
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

print(f"Jumlah objek terdeteksi: {object_count}")
```

---

## 14. Estimasi Kecepatan Objek

### 14.1 Dari Perpindahan Piksel ke Kecepatan Nyata
Optical flow dan object tracking menghasilkan perpindahan dalam satuan **piksel per frame**. Untuk mendapatkan kecepatan dalam satuan dunia nyata (m/s, km/h), diperlukan **kalibrasi** yang menghubungkan koordinat piksel dengan koordinat dunia nyata.

### 14.2 Kalibrasi: Pixels per Meter
Kalibrasi dilakukan dengan mengukur jarak yang diketahui dalam scene:

1. Tentukan dua titik dalam gambar yang jaraknya diketahui di dunia nyata.
2. Hitung jarak dalam piksel antara kedua titik tersebut.
3. Rasio memberikan **pixels per meter (ppm)**.

$$
\text{ppm} = \frac{\text{jarak piksel antara dua titik}}{\text{jarak dunia nyata (meter)}}
$$

### 14.3 Formula Kecepatan
Dengan kalibrasi dan frame rate yang diketahui:

$$
v = \frac{d_{\text{pixel}}}{\text{ppm}} \times \text{fps}
$$

di mana:
- $d_{\text{pixel}}$ = perpindahan objek dalam piksel antar frame
- $\text{ppm}$ = pixels per meter (dari kalibrasi)
- $\text{fps}$ = frame rate video
- $v$ = kecepatan dalam meter per detik

Untuk konversi ke km/h: $v_{\text{km/h}} = v \times 3.6$

```python
# Parameter kalibrasi
known_distance_meters = 5.0      # jarak referensi di dunia nyata
known_distance_pixels = 200.0    # jarak referensi dalam piksel
ppm = known_distance_pixels / known_distance_meters  # pixels per meter

fps = cap.get(cv2.CAP_PROP_FPS)

# Hitung kecepatan dari perpindahan centroid
prev_centroid = (100, 200)
curr_centroid = (115, 205)

dx = curr_centroid[0] - prev_centroid[0]
dy = curr_centroid[1] - prev_centroid[1]
displacement_pixels = np.sqrt(dx**2 + dy**2)

# Kecepatan dalam m/s
speed_mps = (displacement_pixels / ppm) * fps

# Konversi ke km/h
speed_kmh = speed_mps * 3.6
print(f"Kecepatan: {speed_mps:.2f} m/s ({speed_kmh:.2f} km/h)")
```

### 14.4 Smoothing Kecepatan
Estimasi kecepatan per-frame bisa noisy. Gunakan **moving average** untuk smoothing:

```python
from collections import deque

speed_buffer = deque(maxlen=10)  # buffer 10 frame terakhir

speed_buffer.append(speed_mps)
smoothed_speed = np.mean(speed_buffer)
```

### 14.5 Limitasi dan Asumsi
Estimasi kecepatan berbasis piksel memiliki beberapa keterbatasan penting:

- **Perspektif**: Objek yang lebih jauh dari kamera tampak bergerak lebih lambat — kalibrasi hanya valid pada satu bidang (plane).
- **Gerakan kamera**: Jika kamera bergerak, perlu kompensasi ego-motion terlebih dahulu.
- **Akurasi tracking**: Error dalam deteksi/tracking menghasilkan noise pada estimasi kecepatan.
- **Single-plane assumption**: Kalibrasi pixels-per-meter hanya akurat jika semua objek bergerak pada bidang yang sama (e.g., permukaan jalan).
- **Occlusion**: Objek yang terhalang dapat menyebabkan lompatan posisi dan kecepatan palsu.

---

## 14b. Kalman Filter untuk Tracking

### Konsep
Kalman filter memprediksi posisi objek ketika deteksi gagal (occlusion, noise). Terdiri dari dua tahap: **predict** (extrapolasi state berdasarkan model gerak) dan **correct** (update dengan pengukuran baru).

### Model Gerak (Constant Velocity)
State vector: $[x, y, v_x, v_y]^T$

$$\mathbf{x}_k = \begin{bmatrix} 1 & 0 & dt & 0 \\ 0 & 1 & 0 & dt \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \mathbf{x}_{k-1} + \mathbf{w}_k$$

### Implementasi OpenCV
```python
kf = cv2.KalmanFilter(4, 2)  # 4 state vars (x,y,vx,vy), 2 measurements (x,y)
kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Dalam loop tracking:
predicted = kf.predict()   # prediksi posisi berikutnya
if detection_available:
    estimated = kf.correct(measurement)  # koreksi dengan deteksi
```

### Tuning Parameter
- **processNoiseCov**: Besar → Kalman lebih percaya pengukuran. Kecil → lebih percaya model.
- **measurementNoiseCov**: Besar → Kalman lebih percaya model (smooth trajectory).

---

## 14c. Mean Shift dan CAMShift

### Mean Shift Tracking
Mean Shift iteratif mencari mode (puncak) distribusi histogram warna dalam ROI. Pada setiap frame, ROI digeser ke posisi di mana distribusi warna paling cocok.

```python
# Setup initial ROI dan histogram
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Tracking criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Dalam loop video:
dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
ret, track_window = cv2.meanShift(dst, track_window, criteria)
```

### CAMShift (Continuously Adaptive Mean Shift)
Ekstensi Mean Shift yang **auto-resize** window tracking. Cocok untuk objek yang berubah ukuran (mendekat/menjauh dari kamera).

```python
ret, track_window = cv2.CamShift(dst, track_window, criteria)
pts = cv2.boxPoints(ret)  # rotated rectangle
```

### Perbandingan
| Aspek | Mean Shift | CAMShift | CSRT/KCF |
|-------|-----------|----------|----------|
| Adaptasi ukuran | Tidak | Ya | Ya |
| Rotasi | Tidak | Ya | Tidak |
| Kecepatan | Sangat cepat | Cepat | Sedang |
| Akurasi | Rendah | Sedang | Tinggi |
| Basis | Histogram warna | Histogram warna | Correlation filter |

---

## 14d. SORT dan DeepSORT

### SORT (Simple Online and Realtime Tracking)
SORT mengkombinasikan Kalman filter untuk prediksi dan Hungarian algorithm untuk asosiasi deteksi-track.

**Pipeline:**
1. Deteksi objek per frame (YOLO, SSD, dll.).
2. Prediksi posisi track yang ada (Kalman predict).
3. Hitung IoU antara deteksi baru dan prediksi.
4. Asosiasi optimal menggunakan Hungarian algorithm.
5. Update track yang cocok (Kalman correct).
6. Buat track baru untuk deteksi tanpa pasangan.
7. Hapus track yang tidak di-update selama N frame.

### DeepSORT
Menambahkan **appearance descriptor** (deep learning feature) untuk mengatasi ID switch saat occlusion.

- **Mahalanobis distance**: Gating berdasarkan Kalman state.
- **Cosine distance**: Similarity berdasarkan appearance embedding (Re-ID network).

### Kapan Menggunakan
- **Mean Shift/CAMShift**: Single object, histogram-based, sangat cepat.
- **CSRT/KCF**: Single object, akurasi tinggi, tanpa detektor.
- **SORT**: Multi-object, butuh detektor, real-time.
- **DeepSORT**: Multi-object, occlusion berat, ID consistency.

---

## 15. Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed., Chapter 9.
2. Lucas, B. & Kanade, T. (1981). *An Iterative Image Registration Technique with an Application to Stereo Vision*. IJCAI.
3. Horn, B. & Schunck, B. (1981). *Determining Optical Flow*. Artificial Intelligence.
4. Farnebäck, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion*. SCIA.
5. Zivkovic, Z. (2004). *Improved Adaptive Gaussian Mixture Model for Background Subtraction*. ICPR.
6. Bolme, D. et al. (2010). *Visual Object Tracking using Adaptive Correlation Filters*. CVPR.
7. Lukezic, A. et al. (2017). *Discriminative Correlation Filter with Channel and Spatial Reliability*. CVPR.
8. Davis, J. & Bobick, A. (1997). *The Representation and Recognition of Human Movement Using Temporal Templates*. CVPR.
