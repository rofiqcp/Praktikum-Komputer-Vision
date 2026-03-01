# MATERI MODUL 8: IMAGE STITCHING DAN ALIGNMENT

---

## 1. Pendahuluan

Image stitching adalah proses menggabungkan beberapa gambar yang overlapping menjadi satu gambar panorama. Area ini menggabungkan konsep dari feature detection, geometric transformation, blending, dan optimasi global untuk menghasilkan panorama yang seamless.

**Referensi utama**: Szeliski, *Computer Vision: Algorithms and Applications*, 2nd Edition, **Chapter 8 — Image Alignment and Stitching**.

---

## 2. Pipeline Image Stitching

### 2.1 Tahapan Umum
1. **Feature Detection & Matching** — Temukan korespondensi antar gambar.
2. **Homography Estimation** — Estimasi transformasi geometris.
3. **Image Warping** — Transformasi gambar ke koordinat bersama.
4. **Blending** — Gabungkan gambar yang di-warp agar seamless.

### 2.2 Motion Models
- **Translation** (2 DOF): $\mathbf{x}' = \mathbf{x} + \mathbf{t}$
- **Affine** (6 DOF): $\mathbf{x}' = A\mathbf{x} + \mathbf{t}$
- **Homography/Projective** (8 DOF): $\tilde{\mathbf{x}}' = H\tilde{\mathbf{x}}$

Untuk panorama dari kamera yang berotasi di satu titik, **homography** adalah model yang tepat.

---

## 3. Feature-based Alignment

### 3.1 Proses
1. Deteksi fitur (SIFT, ORB, AKAZE) pada semua gambar.
2. Match fitur antar pasangan gambar.
3. Estimasi homography per pasangan menggunakan RANSAC.
4. Verifikasi: cek jumlah inlier dan konsistensi geometris.

### 3.2 Image Registration
$$
H_{i \to ref} = H_{i \to i+1} \cdot H_{i+1 \to i+2} \cdot \ldots \cdot H_{n-1 \to ref}
$$

Semua gambar ditransformasi ke koordinat gambar referensi (biasanya gambar tengah).

---

## 4. Image Warping

### 4.1 Forward Warping vs Inverse Warping
- **Forward**: Untuk setiap piksel sumber, petakan ke destinasi → bisa ada gaps.
- **Inverse**: Untuk setiap piksel destinasi, cari sumber → lebih baik, tanpa gaps.

OpenCV menggunakan inverse warping:
```python
result = cv2.warpPerspective(img, H, (width, height))
```

### 4.2 Proyeksi untuk Panorama

#### Planar Projection
- Langsung gunakan homography.
- Cocok untuk FOV kecil (< 90°).
- Distorsi besar pada tepi untuk FOV lebar.

#### Cylindrical Projection
Gambar dipetakan ke permukaan silinder:
$$
x' = f \cdot \tan^{-1}\left(\frac{x - c_x}{f}\right), \quad y' = f \cdot \frac{y - c_y}{\sqrt{(x-c_x)^2 + f^2}}
$$

- Cocok untuk panorama horizontal 360°.
- Mengurangi distorsi pada tepi.

#### Spherical Projection
Gambar dipetakan ke permukaan bola:
$$
\theta = \tan^{-1}\left(\frac{x - c_x}{f}\right), \quad \phi = \tan^{-1}\left(\frac{y - c_y}{\sqrt{(x-c_x)^2 + f^2}}\right)
$$

- Paling akurat untuk full 360° panorama.

```python
def cylindrical_warp(img, f):
    h, w = img.shape[:2]
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    # ... cylindrical mapping
```

---

## 5. Bundle Adjustment

### 5.1 Motivasi
Chain homography ($H_{12} \cdot H_{23} \cdot \ldots$) mengakumulasi error. Bundle adjustment mengoptimalkan SEMUA parameter kamera secara simultan.

### 5.2 Formulasi
Minimasi total reprojection error:

$$
\min_{\{R_i, f_i\}} \sum_{i,j} \sum_{k} d(x_{ij}^k, \pi(R_i, f_i, X_j^k))^2
$$

di mana $R_i, f_i$ adalah rotasi dan focal length kamera ke-$i$, dan $X_j^k$ adalah titik 3D.

### 5.3 OpenCV Stitcher
`cv2.Stitcher` mengimplementasikan bundle adjustment secara otomatis:
```python
stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA)
status, panorama = stitcher.stitch(images)
```

---

## 6. Image Blending

### 6.1 Masalah Seam
Perbedaan exposure, white balance, dan vignetting menyebabkan seam yang terlihat di batas antar gambar.

### 6.2 Teknik Blending

#### Feather Blending (Alpha Blending)
Linearly blend di area overlap berdasarkan jarak dari tepi:
$$
I_{blend}(x,y) = \alpha(x,y) \cdot I_1(x,y) + (1 - \alpha(x,y)) \cdot I_2(x,y)
$$

#### Multi-band Blending (Laplacian Pyramid)
Gabungkan frekuensi rendah dari satu gambar dengan frekuensi tinggi dari gambar lain:
1. Buat Laplacian pyramid untuk setiap gambar.
2. Buat Gaussian pyramid untuk mask.
3. Blend per level pyramid.
4. Rekonstruksi dari blended pyramid.

```python
# Multi-band blending concept
for level in range(num_levels):
    blended_lap = mask_gauss[level] * lap1[level] + (1 - mask_gauss[level]) * lap2[level]
```

#### Gradient-domain Blending (Poisson)
- Minimasi gradien difference alih-alih intensitas.
- Menghasilkan transisi paling halus.

---

## 7. Exposure Compensation

### 7.1 Gain Compensation
Menyesuaikan gain (brightness) setiap gambar agar konsisten:
$$
I'_i = g_i \cdot I_i
$$

Gain $g_i$ diestimasi dari area overlapping.

### 7.2 Block-based Compensation
Bagi overlap area menjadi blocks, sesuaikan gain per block untuk variasi lokal.

```python
compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
```

---

## 8. Seam Finding

### 8.1 Pendekatan
Temukan garis seam optimal di area overlap yang meminimalkan perbedaan visual.

### 8.2 Metode
- **Voronoi**: Batas di tengah overlap (sederhana tapi tidak optimal).
- **Graph Cut**: Minimasi energy function yang mempertimbangkan perbedaan warna di seam.
- **Dynamic Programming**: Cari path minimum cost dari atas ke bawah.

$$
E_{seam} = \sum_{(p,q) \in seam} |I_1(p) - I_2(p)| + |I_1(q) - I_2(q)|
$$

```python
seam_finder = cv2.detail.GraphCutSeamFinder('COST_COLOR')
```

---

## 9. Panorama Photography Tips

### 9.1 Pengambilan Gambar
- Overlap 30–50% antar gambar.
- Rotasi dari satu titik (gunakan tripod jika mungkin).
- Exposure manual dan white balance manual → konsisten.
- Avoid moving objects di area overlap.

### 9.2 Panorama 360°
- Minimal 8–12 gambar untuk 360° horizontal.
- Cylindrical/spherical projection wajib.
- Bundle adjustment penting untuk menutup loop (gambar pertama dan terakhir).

---

## 10. Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed., Chapter 8.
2. Brown, M. & Lowe, D. (2007). *Automatic Panoramic Image Stitching using Invariant Features*. IJCV.
3. Burt, P. & Adelson, E. (1983). *A Multiresolution Spline with Application to Image Mosaics*. ACM TOG.
4. Agarwala, A. et al. (2004). *Interactive Digital Photomontage*. SIGGRAPH.
5. Szeliski, R. (2006). *Image Alignment and Stitching: A Tutorial*. Foundations and Trends in Computer Graphics and Vision.
6. Kwatra, V. et al. (2003). *Graphcut Textures: Image and Video Synthesis Using Graph Cuts*. SIGGRAPH.
7. Pérez, P. et al. (2003). *Poisson Image Editing*. SIGGRAPH.
8. Shum, H. & Szeliski, R. (2000). *Construction of Panoramic Image Mosaics with Global and Local Alignment*. IJCV.
