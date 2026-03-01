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

## 10. CLAHE (Contrast Limited Adaptive Histogram Equalization)

### 10.1 Perbedaan dengan Global Histogram Equalization
Histogram equalization global menerapkan transformasi yang sama ke seluruh piksel gambar. Hal ini sering menghasilkan **over-enhancement** pada area tertentu dan kehilangan detail lokal. CLAHE mengatasi masalah ini dengan membagi gambar menjadi **tile** kecil dan menerapkan equalization secara lokal pada setiap tile.

### 10.2 Prinsip Kerja CLAHE
1. **Pembagian Tile**: Gambar dibagi menjadi grid tile berukuran `tileGridSize` (default 8×8).
2. **Histogram Lokal**: Histogram dihitung untuk setiap tile secara independen.
3. **Clip Limit**: Histogram di-clip pada batas `clipLimit` untuk mencegah amplifikasi noise berlebih.
4. **Redistribusi**: Piksel yang melebihi clip limit didistribusikan ulang secara merata ke seluruh bin histogram.
5. **Interpolasi Bilinear**: Hasil antar-tile diinterpolasi untuk menghilangkan batas antar-tile yang terlihat.

$$
\text{clipLimit} = \frac{M \times N}{n_{bins}} \times \alpha
$$

di mana $M \times N$ adalah ukuran tile, $n_{bins}$ jumlah bin histogram, dan $\alpha$ adalah faktor clip yang dikontrol pengguna.

### 10.3 Penerapan pada LAB Color Space
CLAHE sebaiknya diterapkan pada **channel L (Luminance)** dari color space LAB, bukan langsung pada BGR. Ini memastikan hanya brightness yang di-enhance tanpa mengubah warna.

```python
import cv2
import numpy as np

img = cv2.imread("input.jpg")

# Konversi ke LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Buat CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Terapkan CLAHE pada L channel
lab[:, :, 0] = clahe.apply(lab[:, :, 0])

# Konversi kembali ke BGR
result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imshow("Original", img)
cv2.imshow("CLAHE Enhanced", result)
cv2.waitKey(0)
```

### 10.4 Parameter Tuning
| Parameter | Nilai Kecil | Nilai Besar |
|-----------|-------------|-------------|
| `clipLimit` | Enhancement halus, sedikit perubahan | Kontras tinggi, potensi noise |
| `tileGridSize` | Tile besar → mendekati global | Tile kecil → sangat lokal, detail tinggi |

---

## 11. Unsharp Mask Sharpening

### 11.1 Konsep Dasar
Unsharp masking adalah teknik klasik sharpening yang bekerja dengan **memperkuat perbedaan antara gambar asli dan versi blur-nya**. Secara paradoks, kita menggunakan versi "unsharp" (blur) untuk membuat gambar lebih tajam.

### 11.2 Formula Unsharp Mask

$$
I_{sharp} = I_{original} + \text{amount} \times (I_{original} - I_{blurred})
$$

di mana:
- $I_{original}$: gambar asli
- $I_{blurred}$: gambar setelah Gaussian blur (low-pass filter)
- $\text{amount}$: faktor penguatan (biasanya 0.5–2.0)
- $(I_{original} - I_{blurred})$: high-frequency detail (edge dan tekstur)

### 11.3 Parameter Penting
- **sigma ($\sigma$)**: Standar deviasi Gaussian blur. Sigma besar → sharpening area lebih luas (coarser detail).
- **amount**: Kekuatan enhancement. Nilai tinggi → lebih tajam tapi lebih rentan terhadap artifact.
- **threshold**: Nilai minimum perbedaan untuk di-sharpen. Mencegah sharpening pada area flat (noise).

### 11.4 Implementasi Simple Unsharp Mask

```python
import cv2
import numpy as np

def unsharp_mask(image, sigma=1.0, amount=1.5, threshold=0):
    """
    Simple Unsharp Mask Sharpening
    """
    # Buat versi blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Hitung detail (high-frequency)
    detail = image.astype(np.float64) - blurred.astype(np.float64)

    # Terapkan threshold
    if threshold > 0:
        mask = np.abs(detail) >= threshold
        detail = detail * mask

    # Tambahkan detail yang diperkuat ke gambar asli
    sharpened = image.astype(np.float64) + amount * detail
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened

img = cv2.imread("input.jpg")
result = unsharp_mask(img, sigma=1.0, amount=1.5, threshold=5)

cv2.imshow("Original", img)
cv2.imshow("Sharpened", result)
cv2.waitKey(0)
```

### 11.5 Edge-Aware Unsharp Mask
Untuk hasil yang lebih baik, gunakan **bilateral filter** atau **edge-preserving filter** sebagai pengganti Gaussian blur. Ini mencegah over-sharpening pada edge dan halo artifact.

```python
def edge_aware_unsharp(image, sigma_s=60, sigma_r=0.4, amount=1.5):
    """
    Edge-aware unsharp mask menggunakan edge-preserving filter
    """
    # Edge-preserving smoothing
    smoothed = cv2.edgePreservingFilter(image, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)

    # Hitung detail
    detail = image.astype(np.float64) - smoothed.astype(np.float64)

    # Sharpen
    sharpened = image.astype(np.float64) + amount * detail
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened
```

---

## 12. White Balance Correction

### 12.1 Motivasi
Warna dalam foto dipengaruhi oleh warna cahaya (color temperature). Lampu tungsten menghasilkan cast kuning, neon menghasilkan cast hijau, dan bayangan menghasilkan cast biru. White balance bertujuan mengoreksi color cast agar objek putih terlihat netral.

### 12.2 Gray World Assumption
Asumsi: rata-rata warna seluruh gambar seharusnya adalah abu-abu netral. Setiap channel dikoreksi berdasarkan rasio rata-rata terhadap target netral.

$$
\text{gain}_c = \frac{\mu_{target}}{\mu_c}, \quad c \in \{B, G, R\}
$$

di mana $\mu_{target}$ biasanya $\frac{\mu_R + \mu_G + \mu_B}{3}$ atau 128.

```python
import cv2
import numpy as np

def gray_world_white_balance(img):
    """
    Gray World Assumption: rata-rata tiap channel harus sama
    """
    result = img.astype(np.float64)

    # Hitung rata-rata tiap channel
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    # Target: rata-rata keseluruhan
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    # Koreksi setiap channel
    result[:, :, 0] *= avg_all / avg_b
    result[:, :, 1] *= avg_all / avg_g
    result[:, :, 2] *= avg_all / avg_r

    return np.clip(result, 0, 255).astype(np.uint8)

img = cv2.imread("input.jpg")
corrected = gray_world_white_balance(img)

cv2.imshow("Original", img)
cv2.imshow("Gray World WB", corrected)
cv2.waitKey(0)
```

### 12.3 White Patch Algorithm (Max-RGB)
Asumsi: piksel paling terang dalam gambar seharusnya berwarna putih. Setiap channel di-scale sehingga nilai maksimumnya menjadi 255.

$$
\text{gain}_c = \frac{255}{\max(I_c)}
$$

```python
def white_patch_white_balance(img):
    """
    White Patch Algorithm: piksel terterang harus putih
    """
    result = img.astype(np.float64)

    # Gunakan persentil 99 (lebih robust dari max murni)
    max_b = np.percentile(result[:, :, 0], 99)
    max_g = np.percentile(result[:, :, 1], 99)
    max_r = np.percentile(result[:, :, 2], 99)

    # Scale setiap channel
    result[:, :, 0] *= 255.0 / max_b
    result[:, :, 1] *= 255.0 / max_g
    result[:, :, 2] *= 255.0 / max_r

    return np.clip(result, 0, 255).astype(np.uint8)
```

### 12.4 Manual Gain Adjustment
Kadang diperlukan koreksi manual per channel untuk fine-tuning:

```python
def manual_white_balance(img, r_gain=1.0, g_gain=1.0, b_gain=1.0):
    """
    Manual white balance dengan gain per channel
    """
    result = img.astype(np.float64)
    result[:, :, 0] *= b_gain  # Blue channel
    result[:, :, 1] *= g_gain  # Green channel
    result[:, :, 2] *= r_gain  # Red channel
    return np.clip(result, 0, 255).astype(np.uint8)

# Contoh: mengurangi cast kuning (boost blue, reduce red)
corrected = manual_white_balance(img, r_gain=0.9, g_gain=1.0, b_gain=1.2)
```

---

## 13. Synthetic Bokeh / Depth of Field

### 13.1 Bokeh pada Fotografi
Bokeh adalah efek blur estetis pada area out-of-focus. Pada kamera nyata, bokeh dipengaruhi oleh:
- **Aperture** (f-number): Aperture besar (f/1.4) → bokeh kuat
- **Focal length**: Telephoto → bokeh lebih kuat
- **Jarak subjek**: Subjek dekat → background lebih blur

### 13.2 Circle of Confusion (CoC)
Diameter blur disk (circle of confusion) untuk titik pada jarak $d$ ketika fokus pada jarak $d_f$:

$$
\text{CoC} = \frac{|d - d_f|}{d} \cdot \frac{f^2}{N \cdot (d_f - f)}
$$

di mana $f$ = focal length, $N$ = f-number. Titik yang lebih jauh dari $d_f$ mendapat blur lebih besar.

### 13.3 Synthetic Bokeh dengan Depth Map
Untuk membuat bokeh sintetis, kita memerlukan **depth map** — estimasi kedalaman setiap piksel. Jumlah blur ditentukan oleh perbedaan kedalaman dari titik fokus.

```python
import cv2
import numpy as np

def synthetic_bokeh(image, depth_map, focus_depth, blur_strength=15):
    """
    Membuat efek bokeh sintetis berdasarkan depth map.

    Parameters:
        image: gambar input (BGR)
        depth_map: depth map (grayscale, 0=dekat, 255=jauh)
        focus_depth: kedalaman fokus (0-255)
        blur_strength: kekuatan blur maksimum
    """
    h, w = image.shape[:2]
    result = np.zeros_like(image, dtype=np.float64)
    weight_sum = np.zeros((h, w, 1), dtype=np.float64)

    # Normalisasi depth map
    depth_norm = depth_map.astype(np.float64) / 255.0
    focus_norm = focus_depth / 255.0

    # Hitung blur amount berdasarkan jarak dari focus depth
    blur_amount = np.abs(depth_norm - focus_norm)

    # Buat beberapa level blur
    num_levels = 10
    for i in range(num_levels):
        level = i / (num_levels - 1)  # 0.0 — 1.0
        ksize = int(blur_strength * level) * 2 + 1

        if ksize <= 1:
            blurred = image.astype(np.float64)
        else:
            blurred = cv2.GaussianBlur(image, (ksize, ksize), 0).astype(np.float64)

        # Weight: seberapa cocok level blur ini dengan blur_amount piksel
        sigma_level = 0.1
        w_map = np.exp(-0.5 * ((blur_amount - level) / sigma_level) ** 2)
        w_map = w_map[:, :, np.newaxis]

        result += blurred * w_map
        weight_sum += w_map

    result /= (weight_sum + 1e-8)
    return np.clip(result, 0, 255).astype(np.uint8)

# Contoh penggunaan
img = cv2.imread("input.jpg")
depth = cv2.imread("depth_map.png", cv2.IMREAD_GRAYSCALE)
bokeh = synthetic_bokeh(img, depth, focus_depth=100, blur_strength=20)

cv2.imshow("Synthetic Bokeh", bokeh)
cv2.waitKey(0)
```

### 13.4 Pendekatan Sederhana (Tanpa Depth Map)
Jika depth map tidak tersedia, bisa menggunakan **gradient mask** sederhana (blur bertambah dari tengah/bawah ke atas):

```python
def simple_tilt_shift(image, focus_y=0.5, blur_range=0.2, max_blur=21):
    """
    Tilt-shift sederhana: area sekitar focus_y tajam, sisanya blur
    """
    h, w = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (max_blur, max_blur), 0)

    # Buat gradient mask
    mask = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        dist = abs(y / h - focus_y)
        if dist > blur_range:
            mask[y, :] = min((dist - blur_range) / blur_range, 1.0)

    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_3ch = mask[:, :, np.newaxis]

    result = image.astype(np.float64) * (1 - mask_3ch) + blurred.astype(np.float64) * mask_3ch
    return np.clip(result, 0, 255).astype(np.uint8)
```

---

## 14. Color Enhancement dan Saturation

### 14.1 Manipulasi Saturasi via HSV
Color space HSV memisahkan **Hue** (warna), **Saturation** (kekuatan warna), dan **Value** (kecerahan). Kita bisa memanipulasi saturasi tanpa mengubah hue atau brightness.

```python
import cv2
import numpy as np

def adjust_saturation(image, factor=1.5):
    """
    Boost atau reduce saturasi.
    factor > 1: saturasi naik, factor < 1: saturasi turun
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] *= factor  # S channel
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

img = cv2.imread("input.jpg")
saturated = adjust_saturation(img, factor=1.5)
desaturated = adjust_saturation(img, factor=0.5)
```

### 14.2 Vibrance (Smart Saturation)
Berbeda dengan saturasi biasa, **vibrance** memperkuat warna yang less-saturated lebih banyak, sambil menjaga warna yang sudah saturated agar tidak over-saturated. Ini menghasilkan peningkatan warna yang lebih natural, terutama pada skin tones.

$$
\text{boost}(s) = \text{amount} \times (1 - s / 255)
$$

Warna dengan saturasi rendah mendapat boost lebih besar.

```python
def vibrance(image, amount=50):
    """
    Vibrance: selective saturation boost.
    Warna kurang saturated mendapat boost lebih besar.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    s = hsv[:, :, 1]

    # Boost terbalik proporsional terhadap saturasi yang ada
    boost = amount * (1.0 - s / 255.0)
    hsv[:, :, 1] = np.clip(s + boost, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

### 14.3 Color Grading: Split Toning
Split toning memberikan warna berbeda pada highlights dan shadows. Misalnya, shadows berwarna biru dan highlights berwarna orange (teal-orange look populer di film).

```python
def split_toning(image, shadow_hue=120, highlight_hue=30, strength=0.3):
    """
    Split toning: warna berbeda untuk shadow dan highlight
    shadow_hue, highlight_hue: 0-179 (OpenCV hue range)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    v = hsv[:, :, 2]

    # Mask shadow dan highlight
    shadow_mask = 1.0 - v / 255.0  # Tinggi di area gelap
    highlight_mask = v / 255.0      # Tinggi di area terang

    # Terapkan hue berdasarkan mask
    hsv[:, :, 0] = (hsv[:, :, 0] * (1 - strength)
                     + shadow_hue * shadow_mask * strength
                     + highlight_hue * highlight_mask * strength)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)

    # Sedikit boost saturasi
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + strength * 0.3), 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

### 14.4 LUT (Look-Up Table) untuk Color Grading
LUT memetakan setiap nilai input ke nilai output yang telah ditentukan. Digunakan secara luas di industri film dan fotografi.

```python
def apply_lut_curve(image, lut):
    """
    Menerapkan 1D LUT (256 entries) ke gambar.
    """
    return cv2.LUT(image, lut)

# Contoh: membuat LUT S-curve untuk kontras
def create_s_curve_lut(strength=0.5):
    lut = np.arange(256, dtype=np.float64) / 255.0
    # S-curve menggunakan sigmoid-like transformation
    lut = lut + strength * lut * (1 - lut) * (0.5 - lut) * 4
    lut = np.clip(lut * 255, 0, 255).astype(np.uint8)
    return lut

lut = create_s_curve_lut(strength=0.5)
result = apply_lut_curve(img, lut)
```

---

## 15. Pencil Sketch dan Cartoon Effects

### 15.1 OpenCV Pencil Sketch
OpenCV menyediakan fungsi `cv2.pencilSketch()` yang menghasilkan efek sketsa pensil (grayscale dan color).

```python
import cv2

img = cv2.imread("input.jpg")

# pencilSketch menghasilkan dua output:
# sketch_gray: sketsa hitam-putih
# sketch_color: sketsa berwarna
sketch_gray, sketch_color = cv2.pencilSketch(
    img,
    sigma_s=60,   # Spatial sigma: ukuran neighborhood (0-200)
    sigma_r=0.07, # Range sigma: sensitivitas perbedaan warna (0-1)
    shade_factor=0.05  # Intensitas shading (0-0.1)
)

cv2.imshow("Pencil Sketch (Gray)", sketch_gray)
cv2.imshow("Pencil Sketch (Color)", sketch_color)
cv2.waitKey(0)
```

Parameter:
- **sigma_s**: Semakin besar → smoothing lebih kuat, detail hilang.
- **sigma_r**: Semakin besar → range warna yang di-smooth lebih luas.
- **shade_factor**: Intensitas efek shading (0.0 = gelap, 0.1 = terang).

### 15.2 Stylization (Efek Lukisan)
`cv2.stylization()` menghasilkan efek lukisan artistik yang menyerupai cat air atau cat minyak.

```python
stylized = cv2.stylization(img, sigma_s=150, sigma_r=0.25)
cv2.imshow("Stylization", stylized)
cv2.waitKey(0)
```

### 15.3 Cartoon Effect Pipeline
Efek kartun memerlukan dua komponen: **warna flat** (tanpa gradien halus) dan **edge yang tegas**.

```python
def cartoonize(image, num_bilateral=7, ksize=9):
    """
    Membuat efek kartun:
    1. Bilateral filter untuk warna flat
    2. Edge detection untuk garis tepi
    3. Gabungkan keduanya
    """
    # Step 1: Deteksi edge dari grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=ksize,
        C=2
    )

    # Step 2: Bilateral filter berulang untuk smooth flat colors
    color = image.copy()
    for _ in range(num_bilateral):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=9, sigmaSpace=7)

    # Step 3: Gabungkan edges dengan warna flat
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_3ch)

    return cartoon

img = cv2.imread("input.jpg")
cartoon = cartoonize(img)

cv2.imshow("Original", img)
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
```

### 15.4 Edge-Preserving Filter untuk Smooth Colors
Alternatif selain bilateral filter, OpenCV menyediakan `cv2.edgePreservingFilter()`:

```python
# flags=1: Recursive filter (lebih cepat)
# flags=2: Normalized convolution filter (lebih halus)
smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
cv2.imshow("Edge-Preserving Smooth", smooth)
cv2.waitKey(0)
```

---

## 16. Pseudo-HDR dari Single Image

### 16.1 Motivasi
HDR sejati membutuhkan beberapa foto dengan exposure berbeda. Namun, dalam banyak situasi kita hanya memiliki **satu foto**. Pseudo-HDR mensimulasikan efek HDR dengan membuat "exposure bracket sintetis" dari satu gambar menggunakan transformasi gamma.

### 16.2 Membuat Synthetic Exposure Brackets
Transformasi gamma mengubah brightness secara non-linear:

$$
I_{out} = I_{in}^{\gamma}
$$

- $\gamma < 1$: gambar lebih terang (simulated over-exposure)
- $\gamma = 1$: tidak berubah (normal exposure)
- $\gamma > 1$: gambar lebih gelap (simulated under-exposure)

```python
import cv2
import numpy as np

def gamma_transform(image, gamma):
    """Terapkan transformasi gamma ke gambar."""
    normalized = image.astype(np.float64) / 255.0
    corrected = np.power(normalized, gamma)
    return (corrected * 255).astype(np.uint8)

img = cv2.imread("input.jpg")

# Buat synthetic brackets
under_exposed = gamma_transform(img, gamma=2.0)   # Gelap
normal = img.copy()                                 # Normal
over_exposed = gamma_transform(img, gamma=0.5)     # Terang

cv2.imshow("Under-exposed (gamma=2.0)", under_exposed)
cv2.imshow("Normal", normal)
cv2.imshow("Over-exposed (gamma=0.5)", over_exposed)
cv2.waitKey(0)
```

### 16.3 Merge Brackets dan Tone Mapping
Setelah membuat synthetic brackets, proses selanjutnya sama dengan HDR konvensional:

```python
def pseudo_hdr_from_single(image, gammas=[2.5, 1.0, 0.4]):
    """
    Membuat pseudo-HDR dari satu gambar.

    Parameters:
        image: gambar input (BGR, uint8)
        gammas: list gamma values untuk synthetic brackets
    """
    # 1. Generate synthetic exposure brackets
    brackets = [gamma_transform(image, g) for g in gammas]

    # Exposure times (relatif, sesuai gamma)
    # gamma kecil = over-exposed = exposure time panjang
    times = np.array([1.0 / g for g in gammas], dtype=np.float32)

    # 2. Merge ke HDR
    merge = cv2.createMergeDebevec()
    hdr = merge.process(brackets, times)

    # 3. Tone mapping
    tonemap = cv2.createTonemapReinhard(gamma=1.5, intensity=0, light_adapt=0.8, color_adapt=0)
    ldr = tonemap.process(hdr)
    ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)

    return ldr

img = cv2.imread("input.jpg")
pseudo_hdr = pseudo_hdr_from_single(img)

cv2.imshow("Original", img)
cv2.imshow("Pseudo-HDR", pseudo_hdr)
cv2.waitKey(0)
```

### 16.4 Alternatif: Exposure Fusion untuk Pseudo-HDR
Exposure fusion (Mertens) sering memberikan hasil yang lebih baik untuk pseudo-HDR karena tidak memerlukan estimasi camera response function:

```python
def pseudo_hdr_fusion(image, gammas=[2.5, 1.0, 0.4]):
    """
    Pseudo-HDR menggunakan exposure fusion (Mertens).
    """
    brackets = [gamma_transform(image, g) for g in gammas]

    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0
    )
    fusion = merge_mertens.process(brackets)
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)

    return fusion
```

### 16.5 Perbandingan dan Limitasi

| Aspek | Real HDR | Pseudo-HDR |
|-------|----------|------------|
| Input | Multiple exposure photos | Single image |
| Dynamic range recovery | Informasi asli dari tiap exposure | Hanya "stretch" dari satu exposure |
| Detail di shadow/highlight | Detail asli terrecovery | Detail yang hilang (clipped) tidak bisa dikembalikan |
| Noise | Lebih baik (averaging) | Bisa memperburuk noise di shadow |
| Kualitas | Superior | Cukup baik untuk efek visual |

**Limitasi utama**: Pseudo-HDR **tidak menambah informasi baru**. Jika detail shadow sudah clipped (hitam total) atau highlight sudah blown out (putih total) di gambar asli, pseudo-HDR tidak bisa mengembalikannya. Teknik ini hanya mengoptimalkan distribusi tonal dari informasi yang sudah ada.

---

## 17. Referensi

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
