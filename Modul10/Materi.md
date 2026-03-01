# MATERI MODUL 10: COMPUTATIONAL PHOTOGRAPHY

---

## 1. Pendahuluan

Computational photography menggabungkan teknik fotografi tradisional dengan pemrosesan citra dan computer vision untuk menghasilkan foto yang melampaui kemampuan kamera konvensional. Topik ini meliputi HDR imaging, denoising, inpainting, super resolution, style transfer, dan teknik enhancement lainnya.

**Referensi utama**: Szeliski, *Computer Vision: Algorithms and Applications*, 2nd Edition, **Chapter 10 — Computational Photography**.

---

## 2. High Dynamic Range (HDR) Imaging

### 2.1 Motivasi
Kamera konvensional memiliki dynamic range terbatas (~8 stop). Scene dunia nyata bisa memiliki range hingga 20+ stop. HDR menggabungkan beberapa exposure untuk merepresentasikan full dynamic range.

### 2.2 HDR Pipeline
1. **Capture**: Ambil beberapa foto dengan exposure berbeda (bracketing).
2. **Alignment**: Sejajarkan foto (jika handheld).
3. **Merge**: Gabungkan menjadi HDR radiance map.
4. **Tone Mapping**: Kompres HDR ke displayable LDR.

### 2.3 Camera Response Function (CRF)
Hubungan antara irradiance ($E$) dan pixel value ($Z$):

$$
Z = f(E \cdot \Delta t)
$$

Inverse CRF ($g = f^{-1}$) diperlukan untuk mengrecovery radiance dari pixel values.

**Debevec's Method**:
$$
g(Z_{ij}) = \ln E_i + \ln \Delta t_j
$$

```python
# OpenCV HDR
calibrate = cv2.createCalibrateDebevec()
response = calibrate.process(images, times)
merge = cv2.createMergeDebevec()
hdr = merge.process(images, times, response)
```

### 2.4 Tone Mapping
Mengompres HDR (float32, 10^5 range) ke LDR (uint8, 0-255):

- **Drago**: Logarithmic mapping adaptif.
- **Reinhard**: Global/local operator berdasarkan luminance.
- **Mantiuk**: Contrast preserving.

$$
L_d = \frac{L_w}{1 + L_w} \quad \text{(Reinhard global)}
$$

```python
tonemap = cv2.createTonemapReinhard(gamma=2.2, intensity=0, light_adapt=0, color_adapt=0)
ldr = tonemap.process(hdr)
```

---

## 3. Exposure Fusion

### 3.1 Alternatif HDR
Langsung fuse bracketed images tanpa membuat HDR map terlebih dahulu. Lebih robust dan tanpa artifact tone mapping.

### 3.2 Mertens Fusion
Menggabungkan berdasarkan quality measures per piksel:
- **Contrast**: Laplacian magnitude.
- **Saturation**: Standar deviasi warna.
- **Well-exposedness**: Kedekatan ke 0.5 (tengah range).

$$
W_{ij} = C_{ij}^{w_c} \cdot S_{ij}^{w_s} \cdot E_{ij}^{w_e}
$$

```python
merge_mertens = cv2.createMergeMertens()
fusion = merge_mertens.process(images)
```

---

## 4. Image Denoising

### 4.1 Sumber Noise
- **Shot noise** (Poisson): Fluktuasi acak foton.
- **Read noise** (Gaussian): Elektronik sensor.
- **Dark current**: Termal.

### 4.2 Metode Denoising

#### Gaussian Blur
Simple tapi menghilangkan detail:
```python
denoised = cv2.GaussianBlur(noisy, (5,5), sigma)
```

#### Bilateral Filter
Edge-preserving: blur hanya piksel dengan intensitas serupa:
$$
I'(p) = \frac{1}{W_p} \sum_{q \in N(p)} G_{\sigma_s}(\|p-q\|) \cdot G_{\sigma_r}(|I(p)-I(q)|) \cdot I(q)
$$

```python
denoised = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)
```

#### Non-Local Means (NLM)
Rata-ratakan patch yang mirip dari seluruh gambar:
$$
I'(p) = \frac{1}{Z(p)} \sum_{q \in I} w(p,q) \cdot I(q), \quad w(p,q) = e^{-\frac{\|P(p)-P(q)\|^2}{h^2}}
$$

```python
denoised = cv2.fastNlMeansDenoisingColored(noisy, None, h=10, hForColorComponents=10)
```

---

## 5. Image Inpainting

### 5.1 Tujuan
Mengisi area yang hilang/rusak pada gambar secara otomatis berdasarkan informasi sekitar.

### 5.2 Metode
- **Navier-Stokes (NS)**: Propagasi isophote menggunakan PDE fluid dynamics.
- **Telea (FMM)**: Fast Marching Method — isi dari boundary ke dalam.
- **Deep Learning**: Generative inpainting (Context Encoder, DeepFill).

```python
# Buat mask area yang ingin di-inpaint
mask = np.zeros(img.shape[:2], np.uint8)
cv2.circle(mask, (100, 100), 30, 255, -1)

# Inpainting
result_ns = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
result_telea = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
```

---

## 6. Super Resolution

### 6.1 Konsep
Menghasilkan gambar resolusi tinggi dari input resolusi rendah.

### 6.2 Metode Klasik
- **Interpolasi**: Bicubic, Lanczos (upscale tanpa detail baru).
- **Multi-frame SR**: Gabungkan beberapa frame low-res yang sedikit shifted.

### 6.3 Deep Learning SR
- **SRCNN** (2014): CNN sederhana 3-layer.
- **ESPCN**: Efficient sub-pixel convolution.
- **EDSR**: Enhanced Deep Residual.
- **Real-ESRGAN**: State-of-the-art, handle real-world degradation.

```python
# OpenCV DNN Super Resolution
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("EDSR_x4.pb")
sr.setModel("edsr", 4)
result = sr.upsample(img)
```

---

## 7. Synthetic Bokeh / Depth of Field

### 7.1 Konsep
Simulasi shallow depth of field (bokeh) secara software menggunakan depth map.

### 7.2 Pipeline
1. Estimasi depth map (monocular depth estimation atau dual camera).
2. Pilih fokus area (klik atau auto-detect wajah).
3. Blur piksel berdasarkan jarak dari fokus: semakin jauh = semakin blur.

$$
\sigma_{blur}(x,y) = k \cdot |D(x,y) - D_{focus}|
$$

```python
# Simple synthetic bokeh
depth_diff = np.abs(depth_map - focus_depth)
for s in range(max_blur):
    mask = (depth_diff >= s) & (depth_diff < s+1)
    blurred = cv2.GaussianBlur(img, (2*s+1, 2*s+1), 0)
    result[mask] = blurred[mask]
```

---

## 8. Image Enhancement Pipeline

### 8.1 Comprehensive Enhancement
Kombinasi teknik untuk meningkatkan kualitas foto:
1. White balance correction.
2. Denoising (NLM atau bilateral).
3. Contrast enhancement (CLAHE).
4. Sharpening (unsharp mask).
5. Color boost (saturation adjustment).

### 8.2 CLAHE (Contrast Limited Adaptive Histogram Equalization)
```python
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lab[:,:,0] = clahe.apply(lab[:,:,0])
result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

---

## 9. Style Transfer

### 9.1 Neural Style Transfer
Mengambil konten dari satu gambar dan style (tekstur, warna) dari gambar lain.

$$
\mathcal{L}_{total} = \alpha \mathcal{L}_{content} + \beta \mathcal{L}_{style}
$$

- **Content Loss**: Perbedaan fitur CNN (layer conv4) antara output dan content image.
- **Style Loss**: Perbedaan Gram matrix fitur CNN (multi-layer) antara output dan style image.

```python
# OpenCV DNN Style Transfer (pre-trained models)
net = cv2.dnn.readNetFromTorch("starry_night.t7")
blob = cv2.dnn.blobFromImage(img, 1.0, (width, height), (103.939, 116.779, 123.68), swapRB=False)
net.setInput(blob)
output = net.forward()
```

---

## 10. Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed., Chapter 10.
2. Debevec, P. & Malik, J. (1997). *Recovering High Dynamic Range Radiance Maps from Photographs*. SIGGRAPH.
3. Reinhard, E. et al. (2002). *Photographic Tone Reproduction for Digital Images*. SIGGRAPH.
4. Mertens, T. et al. (2007). *Exposure Fusion*. Pacific Graphics.
5. Buades, A. et al. (2005). *A Non-Local Algorithm for Image Denoising*. CVPR.
6. Bertalmio, M. et al. (2000). *Image Inpainting*. SIGGRAPH.
7. Dong, C. et al. (2014). *Learning a Deep Convolutional Network for Image Super-Resolution*. ECCV.
8. Gatys, L. et al. (2016). *Image Style Transfer Using Convolutional Neural Networks*. CVPR.
9. Wang, X. et al. (2021). *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*. ICCV Workshop.
10. Tomasi, C. & Manduchi, R. (1998). *Bilateral Filtering for Gray and Color Images*. ICCV.
