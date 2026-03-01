# MODUL 1: PENDAHULUAN KOMPUTER VISION

## Referensi Utama
**Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. Chapter 1: Introduction.**

---

## 1.1 Apa Itu Computer Vision?

**Computer Vision** adalah bidang ilmu interdisipliner yang mempelajari bagaimana komputer dapat memperoleh pemahaman tingkat tinggi dari gambar atau video digital. Bidang ini berusaha mengotomatisasi tugas-tugas yang dapat dilakukan oleh sistem visual manusia — mulai dari mengenali objek, memahami scene, hingga merekonstruksi dunia 3D dari citra 2D.

Menurut Szeliski (2022), computer vision berupaya untuk:
> "Describe the world that we see in one or more images and to reconstruct its properties, such as shape, illumination, and color distributions."

### Perbedaan dengan Bidang Terkait

| Bidang | Input | Output | Fokus |
|--------|-------|--------|-------|
| **Image Processing** | Gambar | Gambar (lebih baik) | Meningkatkan kualitas visual |
| **Computer Vision** | Gambar/Video | Informasi / Keputusan | Memahami konten visual |
| **Computer Graphics** | Model 3D / Data | Gambar | Membuat gambar dari data |
| **Machine Learning** | Data (termasuk gambar) | Prediksi / Klasifikasi | Pola dan generalisasi |

### Sejarah Singkat
- **1960-an**: Larry Roberts — extraksi struktur 3D dari gambar 2D (block world).
- **1970-an**: David Marr — teori representasi visual (primal sketch, 2.5D sketch, 3D model).
- **1980-an**: Pendekatan berbasis fitur (edge, corner, texture).
- **1990-an**: Pendekatan statistik dan probabilistik, stereo vision.
- **2000-an**: Feature descriptors (SIFT, SURF), bag-of-words.
- **2010-an**: Deep learning revolution — AlexNet (2012), ResNet, YOLO.
- **2020-an**: Vision Transformers (ViT), Diffusion Models, NeRF, Foundation Models.

---

## 1.2 Pipeline Computer Vision

Sebuah sistem computer vision umumnya mengikuti pipeline berikut:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Akuisisi    │───▶│  Pre-        │───▶│  Ekstraksi   │───▶│  Analisis &  │───▶│  Keputusan / │
│  Citra       │    │  Processing  │    │  Fitur       │    │  Interpretasi│    │  Aksi        │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
     Kamera,          Filtering,          Edge, Corner,       Klasifikasi,       Self-driving,
     Scanner,         Resize,             SIFT, CNN           Deteksi,           Robotika,
     Sensor           Normalisasi         Features            Segmentasi         Surveillance
```

### Level Pemrosesan Visual (David Marr)

1. **Low-level Vision** — Operasi pada piksel: filtering, edge detection, thresholding
2. **Mid-level Vision** — Grouping dan fitur: segmentasi, feature matching, optical flow
3. **High-level Vision** — Pemahaman semantik: object recognition, scene understanding, activity recognition

---

## 1.3 Representasi Gambar Digital

### Gambar sebagai Fungsi Matematika
Sebuah gambar digital dapat direpresentasikan sebagai fungsi:

$$I(x, y) : \mathbb{R}^2 \rightarrow \mathbb{R}$$

di mana $(x, y)$ adalah koordinat spasial dan $I$ adalah intensitas (brightness).

Untuk gambar berwarna (RGB):

$$I(x, y) = [R(x, y), G(x, y), B(x, y)]$$

### Representasi Matriks
Dalam komputer, gambar disimpan sebagai array multidimensi:
- **Grayscale**: Matriks 2D berukuran $H \times W$, setiap elemen bernilai 0–255 (8-bit).
- **RGB/BGR**: Tensor 3D berukuran $H \times W \times 3$.
- **RGBA**: Tensor 3D berukuran $H \times W \times 4$ (dengan alpha channel).

```
Contoh gambar grayscale 4×4:
┌─────┬─────┬─────┬─────┐
│  45 │  89 │ 120 │ 200 │
├─────┼─────┼─────┼─────┤
│ 100 │ 150 │ 200 │  50 │
├─────┼─────┼─────┼─────┤
│  30 │  75 │ 180 │ 220 │
├─────┼─────┼─────┼─────┤
│ 250 │ 130 │  60 │  10 │
└─────┴─────┴─────┴─────┘
```

### Ukuran File Gambar
$$\text{File Size (bytes)} = W \times H \times C \times \frac{D}{8}$$

di mana $W$ = width, $H$ = height, $C$ = channels, $D$ = bit depth.

Contoh: Gambar RGB 1920×1080, 8-bit:
$$1920 \times 1080 \times 3 \times 1 = 6{,}220{,}800 \text{ bytes} \approx 5.93 \text{ MB}$$

---

## 1.4 Ruang Warna (Color Spaces)

### RGB (Red, Green, Blue)
- Model warna aditif, standar untuk display monitor.
- Setiap channel bernilai 0–255 (8-bit per channel).
- OpenCV menggunakan format **BGR** (Blue-Green-Red), bukan RGB.

### HSV (Hue, Saturation, Value)
- **Hue**: Warna dominan (0°–360° atau 0–179 di OpenCV).
- **Saturation**: Kemurnian warna (0–255).
- **Value**: Kecerahan (0–255).
- Lebih intuitif untuk deteksi warna objek.

### Grayscale
- Satu channel intensitas.
- Rumus konversi dari RGB:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
- Mengurangi kompleksitas komputasi.

### YCbCr
- Digunakan dalam kompresi video (JPEG, MPEG).
- Y = luminance, Cb = blue-difference chroma, Cr = red-difference chroma.

### Lab (CIELAB)
- Dirancang agar jarak Euclidean berkorelasi dengan persepsi visual manusia.
- L = lightness, a = green-red, b = blue-yellow.

---

## 1.5 Library dan Tools

### OpenCV (Open Source Computer Vision Library)
- Library utama untuk computer vision.
- Mendukung C++, Python, Java.
- Lebih dari 2500 algoritma: filtering, feature detection, object detection, DNN, dll.

### NumPy
- Library komputasi numerik Python.
- Gambar di OpenCV direpresentasikan sebagai `numpy.ndarray`.
- Operasi matriks yang efisien untuk manipulasi piksel.

### Matplotlib
- Library visualisasi Python.
- Berguna untuk menampilkan gambar dan plot histogram.
- Menggunakan format RGB (berbeda dengan BGR di OpenCV).

### Pillow (PIL)
- Library pemrosesan gambar Python yang lebih sederhana.
- Cocok untuk operasi dasar: open, save, resize, crop.

---

## 1.6 Operasi Dasar pada Gambar

### Loading dan Saving
```python
img = cv2.imread('path/to/image.jpg')      # Load gambar
cv2.imwrite('output.png', img)              # Simpan gambar
```

### Akses Piksel
```python
pixel = img[y, x]         # Akses piksel pada (x,y) → [B, G, R]
img[y, x] = [255, 0, 0]   # Set piksel ke biru (BGR)
roi = img[y1:y2, x1:x2]   # Region of Interest (slicing)
```

### Konversi Ruang Warna
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

### Operasi Aritmatika
```python
added = cv2.add(img1, img2)                 # Penjumlahan (saturated)
blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)  # Blending
```

### Menggambar pada Gambar
```python
cv2.line(img, (0,0), (100,100), (255,0,0), 2)
cv2.rectangle(img, (50,50), (200,200), (0,255,0), 3)
cv2.circle(img, (150,150), 50, (0,0,255), -1)
cv2.putText(img, "Hello", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
```

### Resize dan Scaling
```python
resized = cv2.resize(img, (width, height))                         # Ukuran eksplisit
resized = cv2.resize(img, None, fx=0.5, fy=0.5)                   # Faktor skala
resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)  # Dengan interpolasi
```

Metode interpolasi:
| Metode | Kegunaan |
|--------|----------|
| `INTER_NEAREST` | Tercepat, untuk upscale piksel art |
| `INTER_LINEAR` | Default, baik untuk downscale moderat |
| `INTER_CUBIC` | Lebih halus, lebih lambat |
| `INTER_AREA` | Terbaik untuk downscale (mengurangi aliasing) |
| `INTER_LANCZOS4` | Kualitas tinggi untuk upscale |

### Cropping
Cropping dilakukan dengan slicing array NumPy:
```python
cropped = img[y:y+h, x:x+w]          # Crop region (y, x, h, w)
center_crop = img[cy-r:cy+r, cx-r:cx+r]  # Center crop
```

### Rotasi Gambar
```python
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(img, M, (w, h))
```

Matriks rotasi 2D:
$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

### Flip (Pencerminan)
```python
flipped_h = cv2.flip(img, 1)    # Horizontal (cermin kiri-kanan)
flipped_v = cv2.flip(img, 0)    # Vertikal (cermin atas-bawah)
flipped_b = cv2.flip(img, -1)   # Keduanya (rotasi 180°)
```

### Padding dan Border
```python
bordered = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
```

| Border Type | Deskripsi |
|-------------|-----------|
| `BORDER_CONSTANT` | Warna konstan (default hitam) |
| `BORDER_REPLICATE` | Replikasi piksel tepi |
| `BORDER_REFLECT` | Refleksi piksel tepi |
| `BORDER_WRAP` | Wrapping (tiling) |

### Splitting dan Merging Channel
```python
b, g, r = cv2.split(img)          # Pisahkan channel BGR
merged = cv2.merge([b, g, r])     # Gabungkan kembali
blue_only = np.zeros_like(img)
blue_only[:,:,0] = b               # Visualisasi channel biru saja
```

### Blending Dua Gambar
Alpha blending menggabungkan dua gambar dengan bobot:
$$I_{blend} = \alpha \cdot I_1 + (1 - \alpha) \cdot I_2 + \gamma$$

```python
blended = cv2.addWeighted(img1, alpha, img2, 1-alpha, gamma)
```

### Brightness dan Contrast
Penyesuaian brightness dan contrast mengikuti transformasi linear:
$$g(x,y) = \alpha \cdot f(x,y) + \beta$$

di mana $\alpha$ mengontrol **contrast** (gain) dan $\beta$ mengontrol **brightness** (bias).

```python
adjusted = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
```

### Histogram
Histogram gambar menunjukkan distribusi intensitas piksel. Berguna untuk analisis kontras dan exposure:
```python
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
```

Histogram equalization meratakan distribusi intensitas untuk meningkatkan kontras:
```python
equalized = cv2.equalizeHist(gray)
```

### Masking
Mask adalah gambar biner yang digunakan untuk membatasi operasi pada region tertentu:
```python
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(mask, (cx, cy), radius, 255, -1)   # Mask lingkaran
masked = cv2.bitwise_and(img, img, mask=mask)  # Terapkan mask
```

### Format Gambar dan Penyimpanan
```python
cv2.imwrite('output.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite('output.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
cv2.imwrite('output.bmp', img)
cv2.imwrite('output.tiff', img)
```

| Format | Kompresi | Kelebihan | Kekurangan |
|--------|----------|-----------|------------|
| JPEG | Lossy | Ukuran kecil, cocok foto | Artefak, tidak ada alpha |
| PNG | Lossless | Preservasi kualitas, alpha channel | Ukuran lebih besar |
| BMP | Tidak ada | Tidak ada loss | Ukuran sangat besar |
| TIFF | Opsional | Fleksibel, high quality | Less portable |

---

## 1.7 Aplikasi Computer Vision di Dunia Nyata

### Industri Manufaktur
- **Quality Control**: Deteksi cacat produk secara otomatis pada lini produksi.
- **Pick-and-Place Robotics**: Robot yang dapat mengenali dan mengambil objek.

### Kesehatan / Medis
- **Medical Imaging**: Deteksi tumor dari CT scan, analisis retina, segmentasi organ.
- **Surgical Assistance**: Panduan visual untuk operasi minimal invasif.

### Otomotif
- **Autonomous Driving**: Deteksi pejalan kaki, rambu lalu lintas, jalur jalan.
- **ADAS (Advanced Driver-Assistance Systems)**: Lane departure warning, collision avoidance.

### Keamanan
- **Surveillance**: Face recognition, anomaly detection, license plate recognition (ANPR).
- **Biometric Authentication**: Pengenalan wajah, iris, sidik jari.

### Retail & E-Commerce
- **Visual Search**: Cari produk berdasarkan foto.
- **Automated Checkout**: Sistem tanpa kasir (Amazon Go).

### Pertanian
- **Precision Agriculture**: Deteksi hama, estimasi kematangan buah, monitoring drone.

### Smartphone
- **Computational Photography**: Portrait mode, night mode, HDR.
- **AR Filters**: Snapchat, Instagram face filters.

---

## 1.8 Tantangan dalam Computer Vision

1. **Variasi Viewpoint**: Objek yang sama terlihat berbeda dari sudut pandang berbeda.
2. **Variasi Iluminasi**: Perubahan pencahayaan mengubah tampilan objek secara drastis.
3. **Oklusi**: Objek terhalang sebagian oleh objek lain.
4. **Skala**: Objek dapat muncul dalam berbagai ukuran.
5. **Deformasi**: Objek non-rigid berubah bentuk (tubuh manusia, kain).
6. **Background Clutter**: Objek mirip dengan latar belakang.
7. **Intra-class Variation**: Variasi besar dalam satu kategori (berbagai jenis kursi).

---

## 1.9 Konsep Matematika Dasar

### Koordinat Gambar
Dalam OpenCV, sistem koordinat gambar:
- Sumbu-x: dari kiri ke kanan (kolom).
- Sumbu-y: dari atas ke bawah (baris).
- Origin (0,0) di pojok kiri atas.

### Operasi Matriks Dasar
Gambar adalah matriks NumPy, sehingga operasi matriks berlaku:
- **Transpose**: `img.T` (menukar baris dan kolom channel-wise).
- **Reshape**: `img.reshape(new_shape)`.
- **Statistik**: `np.mean(img)`, `np.std(img)`, `np.min(img)`, `np.max(img)`.

### Histogram
Histogram gambar menggambarkan distribusi intensitas piksel:
$$h(r_k) = n_k, \quad k = 0, 1, 2, \ldots, L-1$$
di mana $r_k$ adalah nilai intensitas dan $n_k$ adalah jumlah piksel dengan intensitas $r_k$.

---

## 1.10 Ringkasan

| Konsep | Penjelasan Singkat |
|--------|--------------------|
| Computer Vision | Membuat komputer "melihat" dan memahami gambar |
| Gambar Digital | Matriks piksel dengan nilai intensitas |
| RGB/BGR | Ruang warna 3-channel untuk representasi warna |
| HSV | Ruang warna intuitif untuk deteksi warna |
| OpenCV | Library utama untuk computer vision di Python |
| Pipeline CV | Akuisisi → Pre-processing → Fitur → Analisis → Keputusan |
| Low/Mid/High Level | Piksel → Fitur → Semantik |

---

## Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition, Springer. **Chapter 1: Introduction** (pp. 1–28).
2. OpenCV Documentation. https://docs.opencv.org/4.x/
3. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing*, 4th Edition, Pearson.
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, MIT Press.
5. Marr, D. (1982). *Vision: A Computational Investigation*, MIT Press.
6. NumPy Documentation. https://numpy.org/doc/
7. Matplotlib Documentation. https://matplotlib.org/stable/
