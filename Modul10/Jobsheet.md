# JOBSHEET MODUL 10: COMPUTATIONAL PHOTOGRAPHY

---

## Tujuan Praktikum
1. Memahami dan mengimplementasikan HDR imaging dan tone mapping.
2. Mengimplementasikan exposure fusion (Mertens).
3. Membandingkan metode denoising (Gaussian, bilateral, NLM).
4. Mengimplementasikan image inpainting.
5. Mengimplementasikan super resolution.
6. Membangun synthetic bokeh dan image enhancement pipeline.
7. Mencoba style transfer menggunakan pre-trained neural network.

---

## Alat dan Bahan
- **Hardware**: PC/Laptop, smartphone (untuk foto bracketing).
- **Software**: Python 3.8+, Jupyter Notebook / VS Code.
- **Library**: OpenCV (`opencv-contrib-python`), NumPy, Matplotlib.
- **Dataset**: Exposure bracketed photos (3–5 exposure per scene), gambar noisy, gambar untuk inpainting.
- **Model**: DNN super resolution (EDSR/ESPCN), style transfer models.

### Persiapan Dataset
```text
1. Foto bracketing: Buka kamera HP → mode Pro/Manual → ambil 3 foto scene yang sama:
   - Underexposed (EV -2)
   - Normal (EV 0)
   - Overexposed (EV +2)
2. Foto noisy: Ambil foto di tempat gelap dengan ISO tinggi.
3. Foto untuk inpainting: Gambar yang ingin dihapus objeknya.
```

---

## Percobaan 1: HDR Imaging

### Tujuan
Membuat gambar HDR dari exposure-bracketed photos dan mencoba berbagai tone mapping.

### Dasar Teori
HDR menggabungkan beberapa exposure menjadi radiance map float32 yang merepresentasikan full dynamic range scene. Tone mapping kemudian mengompres ke displayable range.

### Langkah Kerja
1. Load 3+ gambar bracketed (under, normal, over).
2. Definisikan exposure times (dalam detik): misal `[1/30, 1/8, 1]`.
3. Kalibrasi CRF: `cv2.createCalibrateDebevec()`.
4. Plot response CRF (log scale).
5. Merge HDR: `cv2.createMergeDebevec()`.
6. Tone mapping Drago: `cv2.createTonemapDrago(gamma=2.2)`.
7. Tone mapping Reinhard: `cv2.createTonemapReinhard()`.
8. Tone mapping Mantiuk: `cv2.createTonemapMantiuk()`.
9. Tampilkan 4 hasil side-by-side: Drago, Reinhard, Mantiuk, + original (normal exposure).
10. Variasikan gamma (1.0, 1.5, 2.2, 3.0) pada Reinhard.

### Analisis Percobaan 1
- Tone mapper mana yang menghasilkan foto paling natural?
- Bagaimana gamma mempengaruhi brightness keseluruhan?
- Area mana yang paling terpengaruh oleh HDR (shadow recovery, highlight detail)?
- Apakah ada ghosting artifact (jika ada objek bergerak antar exposure)?

---

## Percobaan 2: Exposure Fusion (Mertens)

### Tujuan
Menggabungkan bracketed photos tanpa HDR pipeline (langsung fuse ke LDR).

### Dasar Teori
Mertens fusion menggunakan quality measures (contrast, saturation, well-exposedness) sebagai weight per piksel untuk menggabungkan gambar.

### Langkah Kerja
1. Load gambar bracketed yang sama dengan Percobaan 1.
2. Merge Mertens: `cv2.createMergeMertens()`.
3. Process dan clip ke [0, 1].
4. Konversi ke uint8 dan tampilkan.
5. Variasikan `contrast_weight` (0, 0.5, 1.0).
6. Variasikan `saturation_weight` (0, 0.5, 1.0).
7. Variasikan `exposure_weight` (0, 0.5, 1.0).
8. Bandingkan hasil Mertens vs HDR + Reinhard.
9. Buat gambar weight map per exposure (visualisasi).
10. Uji pada scene lain dan bandingkan.

### Analisis Percobaan 2
- Apakah Mertens fusion lebih natural dari HDR + tone mapping?
- Weight mana yang paling berpengaruh?
- Pada kasus apa Mertens lebih baik dari HDR? Sebaliknya?
- Apakah ada banding artifact pada Mertens?

---

## Percobaan 3: Image Denoising — Perbandingan Metode

### Tujuan
Membandingkan berbagai metode denoising: Gaussian blur, bilateral filter, NLM.

### Dasar Teori
Denoising menghilangkan noise sambil mempertahankan detail. Metode berbeda memiliki trade-off antara noise removal dan detail preservation.

### Langkah Kerja
1. Load gambar bersih → tambahkan Gaussian noise (σ=25, 50).
2. Denoising dengan Gaussian blur: kernel 3, 5, 7.
3. Denoising dengan bilateral filter: variasi `sigmaColor` dan `sigmaSpace`.
4. Denoising dengan NLM: `fastNlMeansDenoisingColored()`, variasi `h` (5, 10, 20, 30).
5. Hitung PSNR setiap hasil terhadap gambar asli (bersih).
6. Hitung SSIM (dari scikit-image atau manual).
7. Tampilkan grid: noisy, Gaussian, bilateral, NLM + PSNR label.
8. Zoom ke detail (crop area kecil) → bandingkan sharpness.
9. Uji pada gambar noisy asli (bukan sintetis) → foto ISO tinggi.
10. Plot: PSNR vs parameter (h untuk NLM, sigma untuk blur) → temukan optimal.

### Analisis Percobaan 3
- Metode mana dengan PSNR tertinggi?
- Metode mana yang paling preserve edges?
- Pada noise level berapa NLM mulai gagal?
- Apa trade-off waktu per metode?

---

## Percobaan 4: Synthetic Bokeh (Depth of Field)

### Tujuan
Membuat efek bokeh (background blur) secara software.

### Dasar Teori
Synthetic bokeh menggunakan depth map untuk menentukan berapa banyak blur per piksel. Area dekat fokus tetap tajam, area jauh di-blur.

### Langkah Kerja
1. Load gambar portrait (orang di depan background).
2. Buat depth map sederhana (manual mask: foreground = dekat, background = jauh).
3. Atau: gunakan monocular depth estimator (MiDaS OpenCV) jika tersedia.
4. Pilih focus depth (depth area wajah/tubuh).
5. Hitung blur amount per piksel: `sigma = k * |depth - focus_depth|`.
6. Apply variable blur per region.
7. Variasikan `k` (1, 3, 5, 10) → intensity bokeh.
8. Bandingkan Gaussian blur vs disc blur (uniform kernel).
9. Tampilkan: original, depth map, bokeh result.
10. Uji pada 3 gambar berbeda: portrait, still life, landscape.

### Analisis Percobaan 4
- Seberapa realistis bokeh yang dihasilkan?
- Apa artifact paling umum di boundary foreground/background?
- Bagaimana kualitas depth map mempengaruhi hasil bokeh?
- Apakah disc blur lebih natural dari Gaussian? Mengapa?

---

## Percobaan 5: Image Enhancement Pipeline

### Tujuan
Membangun pipeline enhancement yang menggabungkan multiple teknik.

### Dasar Teori
Enhancement menyeluruh memerlukan kombinasi: white balance, denoising, contrast, sharpening, dan color correction yang diterapkan secara berurutan.

### Langkah Kerja
1. Load gambar kualitas rendah (gelap, noisy, low contrast).
2. **White balance**: Gray world assumption atau manual correction.
3. **Denoising**: NLM.
4. **Contrast**: CLAHE pada channel L (LAB).
5. **Sharpening**: Unsharp mask (original + α × (original - blurred)).
6. **Color boost**: Increase saturation di HSV.
7. Tampilkan pipeline step-by-step: setiap tahap.
8. Variasikan urutan: denoising dulu vs contrast dulu → bandingkan.
9. Buat parameter tuning: slider untuk setiap parameter (GUI opsional).
10. Uji pipeline pada 5 gambar dengan masalah berbeda.

### Analisis Percobaan 5
- Apakah urutan processing mempengaruhi hasil akhir?
- Step mana yang memberikan perbaikan paling signifikan?
- Pada gambar mana pipeline gagal menghasilkan perbaikan?
- Bagaimana cara mengenali secara otomatis step mana yang diperlukan?

---

## Percobaan 6: Multi-Frame Enhancement

### Tujuan
Meningkatkan kualitas gambar dengan merata-ratakan beberapa frame (noise reduction).

### Dasar Teori
Jika noise bersifat random (zero-mean), merata-ratakan N frame mengurangi noise sebesar $\sqrt{N}$:

$$
\text{SNR}_{N \ frames} = \sqrt{N} \cdot \text{SNR}_{1 \ frame}
$$

### Langkah Kerja
1. Pasang kamera statis (tripod/meja), ambil 20 frame dari scene statis.
2. Atau: load video statis, ambil 20 frame berturut-turut.
3. Ratakan 2 frame → tampilkan.
4. Ratakan 5 frame → tampilkan.
5. Ratakan 10 frame → tampilkan.
6. Ratakan 20 frame → tampilkan.
7. Bandingkan dengan single frame: zoom ke detail noisy.
8. Hitung PSNR jika ada ground truth (atau estimasi noise reduction).
9. Alignment: tambahkan feature-based alignment sebelum averaging (jika kamera sedikit goyang).
10. Plot: jumlah frame vs noise level (estimated dari std dev area flat).

### Analisis Percobaan 6
- Berapa frame diperlukan untuk noise reduction yang signifikan?
- Apakah alignment diperlukan bahkan dengan tripod?
- Pada berapa frame averaging saturates (tidak ada perbaikan lagi)?
- Bagaimana metode ini dibandingkan NLM denoising?

---

## Percobaan 7: Image Inpainting

### Tujuan
Mengisi area yang hilang/rusak pada gambar secara otomatis.

### Dasar Teori
Inpainting menggunakan informasi di sekitar area rusak untuk menghasilkan fill yang plausible. Navier-Stokes menggunakan PDE, sementara Telea menggunakan Fast Marching.

### Langkah Kerja
1. Load gambar yang ingin di-inpaint.
2. Buat mask: tandai area yang ingin dihapus (gambar manual atau threshold).
3. Inpainting Navier-Stokes: `cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)`.
4. Inpainting Telea: `cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)`.
5. Bandingkan visual: NS vs Telea.
6. Variasikan radius inpainting (1, 3, 5, 10, 20).
7. Uji: hapus teks/watermark dari gambar.
8. Uji: hapus objek kecil (kabel, noda).
9. Uji: hapus objek besar (orang, kendaraan) → evaluasi kualitas.
10. Buat interactive inpainting: user gambar mask dengan mouse → inpaint.

### Analisis Percobaan 7
- Metode mana (NS vs Telea) yang lebih baik secara visual?
- Pada radius berapa inpainting mulai ter-blurry?
- Berapa ukuran maksimum area yang bisa di-inpaint dengan baik?
- Apa limitasi inpainting klasik dibanding deep learning?

---

## Percobaan 8: Super Resolution

### Tujuan
Meningkatkan resolusi gambar menggunakan interpolasi klasik dan deep learning.

### Dasar Teori
Interpolasi hanya menambah piksel tanpa detail baru. Deep learning SR menggunakan model yang belajar dari pasangan low-res/high-res untuk menghasilkan detail yang plausible.

### Langkah Kerja
1. Load gambar high-res → downscale 4× → ini menjadi input low-res.
2. Upscale 4× dengan `cv2.INTER_NEAREST` (baseline).
3. Upscale 4× dengan `cv2.INTER_LINEAR`.
4. Upscale 4× dengan `cv2.INTER_CUBIC`.
5. Upscale 4× dengan `cv2.INTER_LANCZOS4`.
6. Download model EDSR atau ESPCN dari OpenCV DNN Super Resolution.
7. Upscale 4× dengan DNN SR.
8. Hitung PSNR dan SSIM per metode vs gambar asli high-res.
9. Zoom ke detail area → bandingkan sharpness.
10. Buat tabel + visual comparison: metode, PSNR, SSIM, waktu.

### Analisis Percobaan 8
- Metode mana yang memiliki PSNR/SSIM tertinggi?
- Apakah deep learning SR menghasilkan detail yang "benar" atau "hallucinated"?
- Berapa trade-off waktu DNN SR vs interpolasi?
- Pada jenis gambar apa SR bekerja paling baik?

---

## Percobaan 9: Style Transfer

### Tujuan
Menerapkan style satu gambar (mis. lukisan) ke konten gambar lain.

### Dasar Teori
Neural style transfer mengoptimalkan gambar output agar memiliki content dari gambar konten (fitur CNN layer tinggi) dan style dari gambar style (Gram matrix fitur CNN).

### Langkah Kerja
1. Download 3 pre-trained style transfer models (misal: starry_night, mosaic, candy).
2. Load gambar konten (foto biasa).
3. Apply style "Starry Night": load model → forward pass.
4. Apply style "Mosaic".
5. Apply style "Candy" (atau model lainnya).
6. Tampilkan: original, style image, hasil per style.
7. Variasikan ukuran input (256, 512, 1024) → perhatikan detail.
8. Uji pada 3 gambar konten berbeda: portrait, landscape, urban.
9. Ukur waktu processing per style per resolusi.
10. Buat collage: 3 konten × 3 style = 9 kombinasi.

### Analisis Percobaan 9
- Style mana yang paling "menyatu" dengan konten?
- Bagaimana resolusi mempengaruhi kualitas style transfer?
- Apakah ada konten yang tidak cocok untuk style tertentu?
- Berapa waktu processing yang reasonable?

---

## Percobaan 10: Panoramic HDR Photography

### Tujuan
Menggabungkan HDR imaging dengan image stitching untuk panorama HDR.

### Dasar Teori
Panoramic HDR menggabungkan 2 teknik: exposure bracketing per posisi → HDR per posisi → stitch semua HDR menjadi panorama.

### Langkah Kerja
1. Ambil foto: 3 posisi × 3 exposure = 9 foto total.
2. Per posisi: merge HDR atau exposure fusion.
3. Tone map per posisi → 3 gambar LDR.
4. Stitch 3 gambar LDR → panorama.
5. Atau: stitch semua gambar medium exposure dulu → bandingkan.
6. Bandingkan: panorama HDR vs panorama single-exposure.
7. Zoom ke area contrasty → bandingkan detail shadow/highlight.
8. Uji exposure compensation pada stitcher.
9. Crop dan finalize panorama.
10. Buat comparison: input photos, per-position HDR, final panorama.

### Analisis Percobaan 10
- Apakah panorama HDR lebih baik dari panorama + exposure compensation?
- Pada scene apa HDR panorama paling bermanfaat?
- Apa tantangan alignment pada exposure-bracketed panorama?
- Bagaimana urutan optimal: HDR dulu lalu stitch, atau stitch dulu lalu HDR?

---

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Perbandingan HDR vs exposure fusion.
2. Perbandingan metode denoising: speed vs quality.
3. Efektivitas inpainting dan limitasinya.
4. Deep learning SR vs interpolasi klasik.
5. Integrasi multiple techniques untuk foto berkualitas tinggi.

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`) + dataset foto.
- **Naming**: `NIM_Nama_Modul10.ipynb`
