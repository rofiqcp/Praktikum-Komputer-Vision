# MATERI MODUL 7: DETEKSI FITUR DAN PENCOCOKAN

---

## 1. Pendahuluan

Deteksi fitur (feature detection) dan pencocokan (feature matching) adalah fondasi banyak aplikasi computer vision: image stitching, 3D reconstruction, object recognition, visual SLAM, dan augmented reality. Proses ini menjawab pertanyaan: "Titik-titik mana di dua gambar yang berkorespondensi?"

**Referensi utama**: Szeliski, *Computer Vision: Algorithms and Applications*, 2nd Edition, **Chapter 7 — Feature Detection and Matching**.

---

## 2. Feature Detection Pipeline

### 2.1 Tahapan Umum
1. **Feature Detection** — Temukan lokasi titik-titik menarik (keypoints).
2. **Feature Description** — Buat deskriptor (vektor) yang merepresentasikan lingkungan lokal di sekitar keypoint.
3. **Feature Matching** — Cocokkan deskriptor antar gambar.
4. **Geometric Verification** — Validasi kecocokan menggunakan model geometris (RANSAC).

### 2.2 Properti Fitur yang Baik
- **Repeatability**: Terdeteksi kembali pada gambar yang sama dari sudut/skala berbeda.
- **Distinctiveness**: Deskriptor unik, tidak mudah tertukar.
- **Locality**: Dihitung dari area kecil → robust terhadap oklusi.
- **Efficiency**: Cepat dihitung dan dicocokkan.

---

## 3. Corner Detection

### 3.1 Harris Corner Detector
Mendeteksi titik di mana intensitas berubah signifikan di semua arah.

**Auto-correlation Matrix (Second Moment Matrix)**:

$$
M = \sum_{(x,y) \in W} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
$$

**Corner Response Function**:

$$
R = \det(M) - k \cdot (\text{trace}(M))^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2
$$

Interpretasi:
- $R > 0$ (kedua eigenvalue besar) → **Corner**
- $R < 0$ (satu eigenvalue besar) → **Edge**
- $|R|$ kecil (kedua eigenvalue kecil) → **Flat region**

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
img[harris > 0.01 * harris.max()] = [0, 0, 255]
```

### 3.2 Shi-Tomasi (Good Features to Track)
Menggunakan minimum eigenvalue sebagai response function:

$$
R = \min(\lambda_1, \lambda_2)
$$

Lebih stabil daripada Harris karena langsung memilih corner berdasarkan eigenvalue terkecil.

```python
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
```

---

## 4. Blob Detection dan Multi-Scale Features

### 4.1 Laplacian of Gaussian (LoG)
Mendeteksi blob pada berbagai skala menggunakan Laplacian yang dinormalisasi skala:

$$
\nabla^2_{norm} G = \sigma^2 (\frac{\partial^2 G}{\partial x^2} + \frac{\partial^2 G}{\partial y^2})
$$

Blob terdeteksi di maxima/minima pada scale-space 3D (x, y, σ).

### 4.2 Difference of Gaussians (DoG)
Aproksimasi efisien dari LoG, digunakan oleh SIFT:

$$
D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y)
$$

---

## 5. Detektor dan Deskriptor Modern

### 5.1 SIFT (Scale-Invariant Feature Transform)
1. **Scale-space extrema detection**: DoG pada octave pyramid.
2. **Keypoint localization**: Sub-pixel refinement + reject low contrast/edge.
3. **Orientation assignment**: Histogram orientasi gradien → 128-D descriptor.
4. **Descriptor generation**: 4×4 grid × 8 orientasi = 128 dimensi.

**Properti**: Invariant terhadap skala, rotasi, dan sebagian terhadap iluminasi dan viewpoint.

```python
sift = cv2.SIFT_create(nfeatures=500)
keypoints, descriptors = sift.detectAndCompute(gray, None)
```

### 5.2 SURF (Speeded-Up Robust Features)
- Gunakan Hessian matrix + integral image → lebih cepat dari SIFT.
- Deskriptor 64-D (atau 128-D extended).
- *Catatan: SURF dipatenkan dan tidak tersedia di opencv-contrib versi terbaru.*

### 5.3 ORB (Oriented FAST and Rotated BRIEF)
- **FAST** keypoint detector + orientasi.
- **rBRIEF** descriptor: binary descriptor 256-bit.
- Free, fast, good alternative to SIFT/SURF.

```python
orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(gray, None)
```

### 5.4 AKAZE
- Nonlinear scale space (berbeda dari SIFT yang linear Gaussian).
- Modified-Local Difference Binary (M-LDB) descriptor.
- Lebih baik dalam boundary preservation.

```python
akaze = cv2.AKAZE_create()
keypoints, descriptors = akaze.detectAndCompute(gray, None)
```

### 5.5 FAST (Features from Accelerated Segment Test)
- Deteksi corner sangat cepat.
- Cek 16 piksel pada circle radius 3 → jika N piksel berturut-turut lebih terang/gelap → corner.
- Tidak menghasilkan deskriptor (hanya detector).

```python
fast = cv2.FastFeatureDetector_create(threshold=25)
keypoints = fast.detect(gray, None)
```

### 5.6 Perbandingan Detektor

| Detektor | Invariansi | Deskriptor | Kecepatan | Lisensi |
|----------|-----------|------------|-----------|---------|
| SIFT | Scale + Rotation | 128-D float | Lambat | Free (OpenCV 4.4+) |
| SURF | Scale + Rotation | 64-D float | Sedang | Paten |
| ORB | Rotation | 256-bit binary | Cepat | Free |
| AKAZE | Scale + Rotation | Binary | Sedang | Free |
| FAST | - | Tidak ada | Sangat cepat | Free |

---

## 6. Feature Matching

### 6.1 Brute-Force Matcher
Mencocokkan setiap deskriptor dari gambar 1 ke semua deskriptor di gambar 2.

- **Norm**: L2 untuk float descriptor (SIFT), Hamming untuk binary (ORB).
- **Cross-Check**: Match (i→j) ∧ match (j→i) = robust match.

```python
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
```

### 6.2 FLANN (Fast Library for Approximate Nearest Neighbors)
Menggunakan algoritma approximate nearest neighbor untuk matching lebih cepat pada dataset besar.

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)
```

### 6.3 Ratio Test (Lowe's)
Menghilangkan ambiguous matches:

$$
\frac{d_1}{d_2} < \tau \quad (\text{biasanya } \tau = 0.75)
$$

Jika match terbaik jauh lebih baik dari match kedua → match dianggap reliable.

```python
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

---

## 7. Geometric Verification

### 7.1 Homography
Transformasi perspektif 3×3 yang memetakan titik-titik dari satu plane ke plane lain:

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

### 7.2 RANSAC (Random Sample Consensus)
Algoritma robust estimation untuk mengestimasi model dari data dengan outlier:

1. Random sample minimal points (4 untuk homography).
2. Fit model dari sample.
3. Hitung inlier (titik yang sesuai model).
4. Ulangi N iterasi → pilih model dengan inlier terbanyak.

```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

---

## 8. Visualisasi Fitur dan Kecocokan

```python
# Visualisasi keypoints
img_kp = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Visualisasi matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

---

## 9. Aplikasi Feature Detection dan Matching

1. **Image Stitching / Panorama**: Match features antar gambar tumpang tindih.
2. **Object Recognition**: Match features objek query ke database.
3. **Visual SLAM**: Track features antar frame untuk estimasi pose.
4. **Augmented Reality**: Deteksi marker/target → overlay konten virtual.
5. **Image Retrieval**: Cari gambar mirip berdasarkan deskriptor.

---

## 10. Analisis Invariansi Fitur

### 10.1 Invariansi terhadap Rotasi
Detektor yang baik harus mendeteksi keypoints pada lokasi yang sama meskipun gambar dirotasi. SIFT dan ORB dirancang dengan orientation assignment sehingga deskriptornya invariant terhadap rotasi. Pengujian dilakukan dengan merotasi gambar pada berbagai sudut (0°–360°) dan mengukur konsistensi deteksi.

### 10.2 Invariansi terhadap Skala
Scale-space based detectors (SIFT, AKAZE) mendeteksi fitur pada berbagai skala melalui pyramid/octave. Keypoints yang terdeteksi pada scale tertentu akan terdeteksi kembali meskipun gambar di-resize (50%, 100%, 200%, 300%). ORB menggunakan scale pyramid sederhana sehingga kurang robust.

### 10.3 Invariansi terhadap Iluminasi
Perubahan brightness dan contrast secara global mempengaruhi intensitas piksel. Deskriptor yang menggunakan gradien (SIFT) atau perbandingan biner (ORB, AKAZE) cenderung lebih robust karena tidak bergantung pada nilai absolut intensitas.

Strategi preprocessing untuk meningkatkan ketahanan:
- Normalisasi histogram atau CLAHE.
- Gamma correction.
- Konversi ke ruang warna yang memisahkan luminance (LAB, HSV).

---

## 11. Properti dan Perbandingan Deskriptor

### 11.1 Dimensionalitas dan Tipe Data

| Deskriptor | Dimensi | Tipe Data | Ukuran per Keypoint |
|------------|---------|-----------|---------------------|
| SIFT | 128 | float32 | 512 bytes |
| ORB | 32 | uint8 (binary) | 32 bytes |
| AKAZE | Variabel | uint8 (binary) | ~61 bytes |

### 11.2 Metrik Jarak
- **L2 Norm**: Digunakan untuk float descriptor (SIFT). $d = \sqrt{\sum (a_i - b_i)^2}$
- **Hamming Distance**: Digunakan untuk binary descriptor (ORB, AKAZE). Menghitung jumlah bit yang berbeda via XOR.

### 11.3 Metrik Evaluasi Deskriptor
- **Matching Score**: Jumlah correct matches / total matches.
- **Recall at Precision X**: Berapa banyak true matches ditemukan pada precision tertentu.
- **Trade-off Speed vs Accuracy**: Binary descriptors jauh lebih cepat namun bisa kurang diskriminatif pada scene kompleks.

---

## 12. Content-Based Image Retrieval (CBIR)

### 12.1 Pendekatan Direct Matching
Cocokkan deskriptor query terhadap setiap gambar di database. Similarity score = jumlah good matches. Sederhana tetapi lambat untuk database besar ($O(N)$ per query).

### 12.2 Bag of Visual Words (BoVW)
1. Ekstrak deskriptor dari semua gambar training.
2. Cluster menggunakan K-Means → visual vocabulary (codebook).
3. Representasikan setiap gambar sebagai histogram frekuensi visual words.
4. Cocokkan berdasarkan jarak histogram (chi-squared, cosine similarity).

### 12.3 Inverted Index
Struktur data yang memetakan setiap visual word ke daftar gambar yang mengandungnya → mempercepat retrieval secara signifikan, mirip search engine teks.

```python
# Contoh sederhana image retrieval
sift = cv2.SIFT_create()
for query in queries:
    kp_q, desc_q = sift.detectAndCompute(query, None)
    scores = {}
    for name, desc_db in database.items():
        matches = bf.knnMatch(desc_q, desc_db, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        scores[name] = len(good)
    ranking = sorted(scores.items(), key=lambda x: -x[1])
```

---

## 13. Augmented Reality dengan Marker Detection

### 13.1 Pipeline AR Berbasis Fitur
1. **Deteksi**: Temukan fitur pada marker referensi dan frame kamera.
2. **Matching**: Cocokkan fitur marker ↔ frame.
3. **Homography**: Estimasi transformasi perspektif marker → frame.
4. **Warping**: Transformasi konten overlay menggunakan homography.
5. **Blending**: Gabungkan overlay dengan frame asli (alpha blending).

### 13.2 Stabilisasi Overlay
Jitter pada estimasi homography frame-by-frame bisa dikurangi dengan:
- Temporal smoothing (rata-rata homography beberapa frame terakhir).
- Kalman filtering pada parameter homography.
- Minimum inlier threshold sebelum update homography.

```python
# Warp overlay ke posisi marker pada scene
h, w = marker.shape[:2]
pts_marker = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
pts_scene = cv2.perspectiveTransform(pts_marker, H)
overlay_warped = cv2.warpPerspective(overlay_img, H, (scene.shape[1], scene.shape[0]))
```

---

## 14. Keypoint Repeatability Metrics

### 14.1 Definisi Repeatability

$$
\text{Repeatability Rate} = \frac{|\{(k_1, k_2) : \|T(k_1) - k_2\| < \epsilon\}|}{\min(|K_1|, |K_2|)}
$$

Di mana $K_1, K_2$ adalah set keypoints dari gambar asli dan transformasi, $T$ adalah transformasi geometris, dan $\epsilon$ adalah toleransi lokasi (biasanya 3–5 piksel).

### 14.2 Faktor yang Mempengaruhi Repeatability
- **Tipe transformasi**: Rotasi, skala, perubahan viewpoint, blur, noise.
- **Intensitas transformasi**: Repeatability umumnya menurun seiring meningkatnya derajat transformasi.
- **Threshold detektor**: Threshold yang lebih rendah menghasilkan lebih banyak keypoints tetapi repeatability bisa menurun.
- **Resolusi gambar**: Gambar resolusi tinggi cenderung memiliki repeatability lebih baik.

---

## 15. Strategi Multi-Image Matching

### 15.1 Pairwise Matching
Cocokkan setiap pasangan gambar secara independen. Kompleksitas $O(n^2)$ untuk $n$ gambar.

### 15.2 Feature Tracks
Rangkaian fitur yang berkorespondensi melintasi beberapa gambar:
- Fitur $f$ di gambar 1 cocok dengan $f'$ di gambar 2 dan $f''$ di gambar 3.
- Track yang panjang dan konsisten menunjukkan fitur yang reliable.
- Digunakan dalam Structure from Motion (SfM) dan visual SLAM.

### 15.3 Graph-Based Matching
- Setiap gambar = node, setiap pasangan dengan matches yang cukup = edge.
- Bobot edge = jumlah inlier matches.
- Connected components menunjukkan kelompok gambar yang terhubung.

### 15.4 Vocabulary Tree
Untuk matching skala besar, vocabulary tree memungkinkan retrieval cepat pasangan gambar yang berpotensi overlap sebelum melakukan matching detail — mengurangi kompleksitas dari $O(n^2)$ menjadi mendekati $O(n \log n)$.

---

## 16. Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed., Chapter 7.
2. Harris, C. & Stephens, M. (1988). *A Combined Corner and Edge Detector*. Alvey Vision Conference.
3. Lowe, D. (2004). *Distinctive Image Features from Scale-Invariant Keypoints*. IJCV.
4. Rublee, E. et al. (2011). *ORB: An Efficient Alternative to SIFT or SURF*. ICCV.
5. Alcantarilla, P. et al. (2013). *Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces*. BMVC.
6. Rosten, E. & Drummond, T. (2006). *Machine Learning for High-Speed Corner Detection*. ECCV.
7. Fischler, M. & Bolles, R. (1981). *Random Sample Consensus*. CACM.
8. Bay, H. et al. (2008). *Speeded-Up Robust Features (SURF)*. CVIU.
9. Shi, J. & Tomasi, C. (1994). *Good Features to Track*. CVPR.
10. Muja, M. & Lowe, D. (2009). *Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration*. VISAPP.
