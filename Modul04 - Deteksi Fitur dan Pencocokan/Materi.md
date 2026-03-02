# Materi Modul 4: Deteksi Fitur dan Pencocokan
# (Feature Detection and Matching)

---

## 4.1 Pendahuluan Deteksi Fitur

### Apa Itu Fitur Lokal?

Fitur lokal (*local feature*) adalah titik, wilayah, atau struktur khas pada gambar yang dapat dideteksi secara andal dan dideskripsikan secara diskriminatif. Tidak seperti pendekatan berbasis piksel global (misalnya histogram warna seluruh gambar), fitur lokal bekerja pada skala kecil namun mengandung informasi geometri dan tekstur yang kaya.

Fitur lokal yang baik memiliki sifat:
- **Repeatability:** terdeteksi pada gambar yang sama meski dalam kondisi berbeda
- **Distinctiveness:** deskriptor cukup unik untuk membedakan satu fitur dari yang lain
- **Locality:** hanya bergantung pada wilayah kecil, tahan terhadap oklusi parsial
- **Efficiency:** dapat dihitung dengan cepat dalam jumlah banyak

### Aplikasi Utama

| Aplikasi | Deskripsi |
|----------|-----------|
| Image Stitching | Menggabungkan beberapa foto menjadi panorama |
| SLAM | Simultaneous Localization and Mapping untuk robot/drone |
| Augmented Reality | Melacak marker dan menumpangkan objek virtual |
| Object Recognition | Mengenali objek berdasarkan fitur lokal |
| Image Retrieval | Mencari gambar serupa dalam database besar |
| 3D Reconstruction | Merekonstruksi struktur 3D dari banyak foto |
| Visual Odometry | Estimasi gerakan kamera dari perubahan fitur |

### Pipeline Umum Feature Matching

```
Gambar A ─► Deteksi Keypoint ─► Komputasi Deskriptor ─┐
                                                        ├─► Matching ─► Geometric Verification ─► Hasil
Gambar B ─► Deteksi Keypoint ─► Komputasi Deskriptor ─┘
```

---

## 4.2 Corner Detection

### Konsep Dasar

Sudut (*corner*) adalah titik di mana gradien gambar memiliki dua arah dominan yang berbeda secara signifikan. Sudut ideal untuk dijadikan keypoint karena:
- Dapat dilokalisasi dengan tepat dalam dua dimensi
- Terdapat di perpotongan dua tepi yang berbeda

Secara formal, untuk setiap titik $(x, y)$ kita hitung matriks struktur (*structure tensor*) atau matriks *second moment*:

$$M = \sum_{(x,y) \in W} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

di mana $I_x$ dan $I_y$ adalah turunan parsial gambar terhadap sumbu $x$ dan $y$, dan $w(x,y)$ adalah fungsi pembobotan Gaussian.

### Harris Corner Detector

Harris (1988) mengusulkan menggunakan *eigenvalue* dari matriks $M$ untuk mengklasifikasikan setiap titik. Namun sebagai aproksimasi yang efisien, Harris mendefinisikan fungsi respons:

$$R = \det(M) - k \cdot \text{trace}(M)^2$$

dengan:
- $\det(M) = \lambda_1 \lambda_2$
- $\text{trace}(M) = \lambda_1 + \lambda_2$
- $k$ adalah konstanta empiris (biasanya 0.04-0.06)

**Interpretasi nilai R:**

| Kondisi | Interpretasi |
|---------|-------------|
| $R \gg 0$ | Corner (eigenvalue besar keduanya) |
| $R \ll 0$ | Edge (satu eigenvalue besar, satu kecil) |
| $|R| \approx 0$ | Flat region (kedua eigenvalue kecil) |

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_float = np.float32(gray)

# blockSize=2, ksize=3 (Sobel kernel), k=0.04
harris = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
harris = cv2.dilate(harris, None)

# Tandai corner di atas threshold
img[harris > 0.01 * harris.max()] = [0, 0, 255]
```

### Shi-Tomasi Corner Detector

Shi dan Tomasi (1994) menyederhanakan kriteria dengan menggunakan eigenvalue terkecil secara langsung:

$$R_{ST} = \min(\lambda_1, \lambda_2)$$

Sebuah titik adalah corner jika $\min(\lambda_1, \lambda_2) > \lambda_{threshold}$.

Kriteria ini terbukti lebih baik untuk pelacakan optical flow karena menghasilkan corner yang lebih stabil.

```python
corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=100,    # Maksimum jumlah corner
    qualityLevel=0.01, # Threshold relatif terhadap corner terkuat
    minDistance=10     # Jarak minimum antar corner (piksel)
)

# Refinement sub-piksel
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_refined = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
```

### Sub-Pixel Corner Refinement

`cornerSubPix` meningkatkan presisi lokasi corner dari level piksel ke level sub-piksel dengan meminimalkan dot product antara gradient dan vektor dari corner ke piksel tetangga:

$$\sum_i \nabla I(q_i) \cdot (q_i - \hat{q}) = 0$$

di mana $\hat{q}$ adalah posisi corner yang disempurnakan.

---

## 4.3 SIFT (Scale-Invariant Feature Transform)

### Gambaran Umum

SIFT (Lowe, 1999, 2004) adalah algoritma deteksi dan deskripsi fitur paling berpengaruh dalam computer vision. SIFT dirancang untuk invarian terhadap skala dan rotasi, serta robust terhadap perubahan iluminasi dan distorsi geometri sederhana.

Pipeline SIFT terdiri dari 4 tahap utama:

### Tahap 1: Scale-Space Extrema Detection

SIFT membangun *scale-space* $L(x,y,\sigma)$ dengan melakukan konvolusi gambar $I(x,y)$ dengan Gaussian pada berbagai $\sigma$:

$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$

Kemudian menghitung *Difference of Gaussian* (DoG):

$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)$$

DoG adalah aproksimasi efisien dari *Laplacian of Gaussian* (LoG), yang optimal untuk deteksi blob multi-skala.

### Tahap 2: Keypoint Localization

Extrema lokal dicari dalam 3D space (x, y, skala). Keypoint lemah dieliminasi menggunakan *Taylor expansion* dan ambang batas kontras. Keypoint di sepanjang tepi dieliminasi menggunakan rasio principal curvature:

$$\frac{\text{trace}(H)^2}{\det(H)} < \frac{(r+1)^2}{r}$$

di mana $H$ adalah Hessian 2x2 dan $r=10$ adalah threshold.

### Tahap 3: Orientation Assignment

Untuk setiap keypoint, orientasi dominan dihitung dari histogram gradien pada lingkungan keypoint (dalam 36 bin, masing-masing 10 derajat). Puncak histogram menjadi orientasi keypoint. Ini memberikan orientasi referensi yang membuat deskriptor invarian terhadap rotasi.

### Tahap 4: Keypoint Descriptor (128-D)

Wilayah 16x16 piksel di sekitar keypoint dibagi menjadi 4x4 sel. Untuk setiap sel, dibuat histogram gradien dengan 8 bin arah. Total: 4x4x8 = **128 dimensi**.

$$d = [h_{1,1}, h_{1,2}, ..., h_{4,4}] \in \mathbb{R}^{128}$$

Deskriptor dinormalisasi untuk invariansi terhadap perubahan pencahayaan linier.

```python
sift = cv2.SIFT_create(
    nfeatures=0,        # 0 = tidak dibatasi
    nOctaveLayers=3,    # Layer per oktaf
    contrastThreshold=0.04,
    edgeThreshold=10,
    sigma=1.6
)

kp, des = sift.detectAndCompute(gray, None)
# des.shape = (N, 128), dtype=float32

img_kp = cv2.drawKeypoints(
    img, kp, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
```

### Sifat Invariansi SIFT

| Transformasi | Invariansi |
|-------------|-----------|
| Rotasi | Penuh (karena assignment orientasi) |
| Skala | Penuh (karena scale-space) |
| Translasi | Penuh (deteksi lokal) |
| Pencahayaan linier | Sebagian (normalisasi descriptor) |
| Pencahayaan non-linear | Terbatas |
| Perubahan perspektif | Terbatas (hingga ~60 derajat) |

---

## 4.4 ORB (Oriented FAST and Rotated BRIEF)

### Latar Belakang

ORB (Rublee et al., 2011) dikembangkan sebagai alternatif bebas paten yang cepat untuk SIFT dan SURF. ORB menggabungkan:
- **FAST** (Features from Accelerated Segment Test) untuk deteksi keypoint
- **rBRIEF** (rotation-aware BRIEF) untuk deskripsi fitur biner

### FAST Corner Detector

FAST menguji apakah suatu piksel $p$ adalah corner dengan membandingkannya terhadap 16 piksel pada lingkaran Bresenham berjejari 3 di sekitarnya. Piksel $p$ dianggap corner jika terdapat $n$ piksel berurutan yang semuanya lebih terang atau lebih gelap dari $p+t$ atau $p-t$:

$$\text{FAST corner}: \exists \text{ arc dari } n \text{ piksel berurutan} |I(p_i) - I(p)| > t$$

Biasanya $n=9$ (FAST-9) atau $n=12$ (FAST-12). NMS diterapkan untuk menghilangkan respons yang berdekatan.

### rBRIEF Descriptor

BRIEF menghasilkan deskriptor biner dengan membandingkan $n_d$ pasang piksel dalam patch:

$$f_{nd}(p) = \sum_{1 \le i \le n_d} 2^{i-1} \cdot \tau(p; x_i, y_i)$$

di mana $\tau(p; x, y) = 1$ jika $p(x) < p(y)$, dan 0 sebaliknya.

ORB menggunakan $n_d = 256$, menghasilkan deskriptor 256-bit (32 bytes). Pasangan piksel dipilih menggunakan *steered test* untuk mengurangi korelasi dan *rBRIEF* untuk invariansi rotasi.

### Hamming Distance

Karena deskriptor biner, jaraknya dihitung menggunakan Hamming distance (jumlah bit yang berbeda):

$$d_H(a, b) = \text{popcount}(a \oplus b)$$

Operasi XOR diimplementasikan secara efisien dengan instruksi CPU `POPCNT`.

```python
orb = cv2.ORB_create(
    nfeatures=500,
    scaleFactor=1.2,   # Scale pyramid factor
    nlevels=8,         # Number of pyramid levels
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,           # Points per test (2=BRIEF, 3/4=richer)
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31
)

kp, des = orb.detectAndCompute(gray, None)
# des.shape = (N, 32), dtype=uint8 (32 bytes = 256 bits)
```

### Perbandingan SIFT vs ORB

| Aspek | SIFT | ORB |
|-------|------|-----|
| Deskriptor | 128 float (512 bytes) | 32 uint8 (32 bytes) |
| Jarak | L2 (Euclidean) | Hamming |
| Kecepatan | Lambat | ~100x lebih cepat |
| Ketahanan rotasi | Sangat baik | Baik |
| Ketahanan skala | Sangat baik | Baik |
| Lisensi | Bebas (sejak 2020) | Bebas |
| Akurasi matching | Tinggi | Menengah |

---

## 4.5 AKAZE dan FAST

### AKAZE (Accelerated-KAZE)

AKAZE (Alcantarilla et al., 2012) merupakan penyempurnaan dari KAZE yang menggunakan *nonlinear scale space* alih-alih Gaussian scale space. Keunggulan utamanya adalah kemampuan mempertahankan batas objek lebih baik karena difusi nonlinear tidak melintasi batas-batas tajam.

*Nonlinear diffusion* dimodelkan dengan persamaan:

$$\frac{\partial L}{\partial t} = \text{div}(c(x, y, t) \nabla L)$$

di mana $c(x,y,t)$ adalah fungsi konduktivitas yang berkurang di dekat tepi gambar.

AKAZE menggunakan deskriptor M-LDB (Modified Local Difference Binary), yang merupakan deskriptor biner sehingga lebih cepat dari SIFT namun lebih akurat dari BRIEF standar.

```python
akaze = cv2.AKAZE_create(
    descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
    descriptor_size=0,      # 0 = ukuran penuh
    descriptor_channels=3,
    threshold=0.001,
    nOctaves=4,
    nOctaveLayers=4,
    diffusivity=cv2.KAZE_DIFF_PM_G2
)
kp, des = akaze.detectAndCompute(gray, None)
```

### FAST (Features from Accelerated Segment Test)

FAST adalah detektor corner yang sangat cepat, dirancang untuk aplikasi real-time. Prinsip kerjanya:

1. Untuk setiap kandidat piksel $p$, pilih 16 piksel pada lingkaran Bresenham jari-jari 3
2. Periksa 4 piksel diagonal terlebih dahulu (barat, timur, utara, selatan) sebagai pre-filter
3. Jika $\ge 3$ dari 4 piksel ini memenuhi syarat, lanjutkan pengecekan penuh
4. Terapkan machine learning (decision tree) untuk mempercepat klasifikasi

```python
fast = cv2.FastFeatureDetector_create(
    threshold=10,
    nonmaxSuppression=True,
    type=cv2.FastFeatureDetector_TYPE_9_16  # FAST-9
)
kp = fast.detect(gray, None)
# FAST hanya mendeteksi, tidak mendeskripsikan
# Gunakan BRIEF/SIFT/ORB untuk deskripsi
```

---

## 4.6 Feature Matching

### BFMatcher (Brute Force Matcher)

BFMatcher membandingkan deskriptor setiap keypoint dari gambar pertama dengan **semua** deskriptor dari gambar kedua dan menemukan yang terdekat. Kompleksitas: $O(N \cdot M)$ dengan $N$ dan $M$ jumlah keypoint di kedua gambar.

```python
# Untuk SIFT (float) - gunakan NORM_L2
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Untuk ORB/AKAZE (binary) - gunakan NORM_HAMMING
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# match() - satu match terbaik per keypoint
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# knnMatch() - k match terbaik per keypoint
knn_matches = bf.knnMatch(des1, des2, k=2)
```

### Lowe Ratio Test

Ratio test yang diusulkan oleh Lowe (2004) adalah cara efektif untuk menyaring match yang ambigu. Untuk setiap keypoint, dua match terbaik ditemukan dan match diterima hanya jika:

$$\frac{d_1}{d_2} < r$$

di mana $d_1$ adalah jarak ke match terbaik, $d_2$ ke match kedua terbaik, dan $r$ biasanya 0.7-0.8.

Ide: jika jarak terkecil jauh lebih kecil dari jarak kedua, match tersebut "distinctive" dan lebih mungkin benar.

```python
good_matches = []
for m, n in knn_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

### Cross-Check Matching

Mode alternatif di mana match $A \to B$ diterima hanya jika match $B \to A$ juga menghasilkan pasangan yang sama ("consistent match"). Ini ekuivalen dengan ratio test yang sangat ketat tetapi lebih lambat.

```python
bf_cross = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf_cross.match(des1, des2)
```

### FLANN (Fast Library for Approximate Nearest Neighbors)

FLANN menggunakan struktur data khusus untuk pencarian *approximate nearest neighbor* yang jauh lebih cepat dari brute force untuk dataset besar.

**Untuk deskriptor SIFT (float):** KD-Tree

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # Presisi vs kecepatan

flann = cv2.FlannBasedMatcher(index_params, search_params)
knn_matches = flann.knnMatch(des1, des2, k=2)
```

**Untuk deskriptor ORB (binary):** LSH (Locality-Sensitive Hashing)

```python
FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,
    key_size=12,
    multi_probe_level=1
)
flann = cv2.FlannBasedMatcher(index_params, search_params)
```

---

## 4.7 Homography dan RANSAC

### Homography

Homography adalah transformasi proyektif 2D yang memetakan titik dari satu bidang ke bidang lain. Representasinya adalah matriks $3 \times 3$ dengan 8 derajat kebebasan:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Karena skala tidak relevan, $H$ memiliki 8 derajat kebebasan dan membutuhkan **minimal 4 pasang titik** untuk diselesaikan.

### Direct Linear Transform (DLT)

Untuk setiap pasang titik $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$, persamaan linier homogen ditulis:

$$\begin{bmatrix} -x_i & -y_i & -1 & 0 & 0 & 0 & x_i x'_i & y_i x'_i & x'_i \\ 0 & 0 & 0 & -x_i & -y_i & -1 & x_i y'_i & y_i y'_i & y'_i \end{bmatrix} \mathbf{h} = 0$$

Dengan 4 pasang titik, sistem ini menghasilkan matriks $8 \times 9$ yang diselesaikan menggunakan SVD: solusi adalah eigenvector terkecil.

### RANSAC (Random Sample Consensus)

RANSAC adalah algoritma estimasi model yang tahan terhadap outlier. Algoritma ini secara berulang:
1. Pilih sampel minimal secara acak (4 titik untuk Homography)
2. Hitung model dari sampel tersebut
3. Hitung inlier (titik yang konsisten dengan model dalam batas error $\epsilon$)
4. Simpan model terbaik (inlier terbanyak)

Jumlah iterasi yang dibutuhkan:

$$N = \frac{\log(1-p)}{\log(1-(1-\varepsilon)^s)}$$

- $p = 0.99$: probabilitas mendapatkan sekali sampel bebas outlier
- $\varepsilon$: rasio outlier (0 = semua inlier, 1 = semua outlier)
- $s = 4$: ukuran sampel minimal (untuk Homography)

**Contoh:** Jika 50% outlier, $\varepsilon=0.5$, $s=4$, $p=0.99$:
$$N = \frac{\log(0.01)}{\log(1-0.0625)} \approx 72 \text{ iterasi}$$

```python
src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

H, mask = cv2.findHomography(
    src_pts, dst_pts,
    cv2.RANSAC,
    ransacReprojThreshold=5.0  # Maksimum error reproyeksi dalam piksel
)

inliers = good[mask.ravel() == 1]
outliers = good[mask.ravel() == 0]

# Warp gambar sumber menggunakan H
h, w = img1.shape[:2]
warped = cv2.warpPerspective(img1, H, (w, h))
```

---

## 4.8 Invariansi Fitur

### Invariansi Rotasi

SIFT mencapai invariansi rotasi dengan menetapkan orientasi dominan gradient sebagai orientasi referensi keypoint. Semua komputasi deskriptor dilakukan dalam sistem koordinat yang dirotasi sesuai orientasi ini.

ORB mencapainya dengan *intensity centroid method*: menghitung momen piksel dalam patch untuk mendapatkan arah referensi.

### Invariansi Skala

Detektor berbasis scale-space (SIFT, AKAZE) menemukan extrema dalam 3D (posisi x, y, dan skala). Setiap keypoint memiliki skala karakteristik sehingga sampling patch selalu dilakukan pada skala yang sama relatif terhadap struktur yang dideteksi.

ORB menggunakan image pyramid dengan $L$ level dan faktor $s$: gambar disampling ulang pada resolusi berbeda, kemudian FAST dijalankan pada setiap level.

### Invariansi Iluminasi

SIFT menormalisasi deskriptor gradien sehingga perubahan kontras global tidak mempengaruhi nilai deskriptor. Nilai yang melebihi 0.2 setelah normalisasi dikliping untuk mengurangi pengaruh saturasi.

ORB menggunakan perbandingan intensitas piksel (bukan nilai absolut) sehingga secara inheren lebih tahan terhadap perubahan brightness. Namun kurang tahan terhadap perubahan kontras.

---

## 4.9 Perbandingan Deskriptor

### Tabel Perbandingan Komprehensif

| Deskriptor | Dim | Tipe | Jarak | Det. Skala | Det. Rot | Kecepatan | Akurasi |
|-----------|-----|------|-------|-----------|---------|-----------|---------|
| SIFT | 128 | float32 | L2 | Sangat Baik | Sangat Baik | Lambat | Tinggi |
| ORB | 32 byte | uint8 | Hamming | Baik | Baik | Sangat Cepat | Menengah |
| AKAZE | 61 byte | uint8 | Hamming | Baik | Baik | Menengah | Baik |
| BRISK | 64 byte | uint8 | Hamming | Baik | Baik | Cepat | Menengah |
| KAZE | 64/128 | float32 | L2 | Baik | Baik | Lambat | Baik |
| SURF | 64/128 | float32 | L2 | Sangat Baik | Baik | Menengah | Tinggi |

### Rekomendasi Pemilihan

- **Akurasi tinggi, waktu tidak kritis:** SIFT
- **Real-time, mobile:** ORB
- **Trade-off terbaik:** AKAZE
- **Alternatif cepat dengan akurasi lebih baik dari ORB:** BRISK

### Memori dan Bandwidth

Untuk $N = 1000$ keypoint:
- SIFT: $1000 \times 128 \times 4 = 512$ KB
- ORB: $1000 \times 32 \times 1 = 32$ KB  (16x lebih hemat)
- AKAZE: $1000 \times 61 \times 1 \approx 61$ KB

---

## 4.10 Image Retrieval Berbasis Fitur

### Prinsip Dasar

Image retrieval berbasis fitur bekerja dengan:
1. **Indexing:** ekstrak fitur dari semua gambar database, simpan dalam struktur yang dapat dicari
2. **Query:** ekstrak fitur dari gambar query
3. **Matching:** cocokkan fitur query terhadap database
4. **Ranking:** urutkan hasil berdasarkan jumlah match atau skor kemiripan

### Skor Kemiripan Sederhana

Pendekatan paling dasar menggunakan jumlah good match setelah filtering:

$$\text{score}(q, d_i) = |\text{GoodMatches}(q, d_i)|$$

atau dinormalisasi:

$$\text{score\_norm}(q, d_i) = \frac{|\text{GoodMatches}(q, d_i)|}{\min(|KP_q|, |KP_{d_i}|)}$$

### Bag of Visual Words (BoVW) - Sekilas

Untuk skala besar, Bag of Visual Words mengelompokkan deskriptor menjadi "visual words" menggunakan k-means. Setiap gambar direpresentasikan sebagai histogram frekuensi visual words. Retrieval dilakukan menggunakan kemiripan histogram (cosine similarity) yang jauh lebih cepat dari feature matching langsung.

```python
# Retrieval sederhana berbasis feature matching
def retrieve_top_k(query_des, database, k=3):
    scores = {}
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5},
                                   {'checks': 50})
    for img_name, db_des in database.items():
        if db_des is None or len(db_des) < 2:
            scores[img_name] = 0
            continue
        knn = flann.knnMatch(query_des, db_des, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        scores[img_name] = len(good)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]
```

### Evaluasi Retrieval

Metrik evaluasi yang umum digunakan:
- **Precision@k:** proporsi hasil relevan dalam top-k
- **Recall@k:** proporsi semua dokumen relevan yang muncul dalam top-k
- **Mean Average Precision (mAP):** rata-rata area di bawah kurva precision-recall

$$\text{Precision@k} = \frac{|\text{relevant items in top-k}|}{k}$$

---

## Ringkasan Modul 4

Modul ini mencakup fondasi lengkap deteksi fitur dan pencocokan:

1. **Corner Detection** (Harris, Shi-Tomasi): dasar untuk memahami keypoint
2. **SIFT**: standar emas, invarian skala dan rotasi, deskriptor 128-D
3. **ORB**: alternatif cepat gratis, biner 256-bit, Hamming distance
4. **AKAZE/FAST**: trade-off antara kualitas dan kecepatan
5. **Matching** (BF, FLANN, ratio test): menyaring match yang berkualitas
6. **Homography + RANSAC**: verifikasi geometri dan eliminasi outlier
7. **Invariansi**: analisis teoritis dan empiris
8. **Image Retrieval**: aplikasi praktis pencarian gambar

Pemahaman komprehensif terhadap semua konsep ini memungkinkan mahasiswa membangun sistem computer vision nyata seperti AR, SLAM, dan image search.
