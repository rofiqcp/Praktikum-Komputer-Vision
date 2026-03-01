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

## Percobaan 11: CLAHE Enhancement

### Tujuan
Meningkatkan kontras gambar secara lokal menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization).

### Dasar Teori
CLAHE membagi gambar menjadi tile kecil dan menerapkan histogram equalization per tile dengan pembatasan kontras (clip limit) untuk mencegah amplifikasi noise. Dibandingkan equalizeHist global, CLAHE menghasilkan peningkatan kontras yang lebih natural dan merata.

### Langkah Kerja
1. Load gambar low contrast (gelap atau berkabut).
2. Konversi ke grayscale → terapkan `cv2.equalizeHist()` sebagai baseline.
3. Buat CLAHE object: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`.
4. Terapkan CLAHE pada grayscale → bandingkan dengan equalizeHist.
5. Untuk gambar berwarna: konversi ke LAB → CLAHE pada channel L → merge kembali.
6. Variasikan `clipLimit` (1.0, 2.0, 4.0, 8.0, 16.0) → amati efek pada kontras.
7. Variasikan `tileGridSize` ((4,4), (8,8), (16,16), (32,32)) → amati perbedaan.
8. Hitung histogram sebelum dan sesudah CLAHE → plot perbandingan.
9. Uji pada 3 jenis gambar: foto gelap, foto berkabut, foto medis (X-ray jika ada).
10. Bandingkan visual: original, equalizeHist, CLAHE dengan parameter terbaik.

### Analisis Percobaan 11
- Apa perbedaan visual antara equalizeHist global dan CLAHE?
- Pada clipLimit berapa CLAHE mulai menghasilkan artifact noise?
- Bagaimana tileGridSize mempengaruhi detail lokal vs global?
- Pada jenis gambar apa CLAHE memberikan perbaikan paling signifikan?

---

## Percobaan 12: Unsharp Mask Sharpening

### Tujuan
Mempertajam detail gambar menggunakan teknik unsharp masking dengan variasi parameter.

### Dasar Teori
Unsharp masking bekerja dengan mengurangi versi blurred dari gambar asli untuk mendapatkan detail (high-frequency), kemudian menambahkannya kembali ke gambar asli dengan faktor penguatan. Formula: `sharpened = original + amount × (original - blurred)`.

### Langkah Kerja
1. Load gambar yang sedikit blur atau kurang tajam.
2. Buat blurred version menggunakan `cv2.GaussianBlur()` dengan sigma=1.0.
3. Hitung detail mask: `detail = original - blurred`.
4. Terapkan unsharp mask: `sharpened = original + amount * detail`.
5. Variasikan sigma Gaussian (0.5, 1.0, 2.0, 5.0) → amati perbedaan.
6. Variasikan amount (0.5, 1.0, 1.5, 2.0, 3.0) → amati tingkat ketajaman.
7. Clip hasil ke range [0, 255] untuk menghindari overflow.
8. Bandingkan dengan `cv2.filter2D()` menggunakan kernel sharpening standar.
9. Zoom ke area detail (teks, edge, tekstur) → bandingkan ketajaman.
10. Uji pada gambar berbeda: portrait, landscape, dokumen teks.

### Analisis Percobaan 12
- Pada sigma dan amount berapa sharpening terlihat natural tanpa halo artifact?
- Apa perbedaan antara unsharp mask dan kernel sharpening filter2D?
- Bagaimana sigma mempengaruhi skala detail yang dipertajam?
- Pada jenis gambar apa unsharp mask paling efektif?

---

## Percobaan 13: White Balance Correction

### Tujuan
Mengkoreksi white balance gambar menggunakan metode gray world, white patch, dan simple scaling.

### Dasar Teori
White balance mengkoreksi color cast yang disebabkan oleh pencahayaan. Gray world assumption mengasumsikan rata-rata warna scene adalah abu-abu netral. White patch assumption menggunakan piksel paling terang sebagai referensi putih. Simple scaling menyeimbangkan mean per channel.

### Langkah Kerja
1. Load gambar dengan color cast yang jelas (terlalu kuning, biru, atau hijau).
2. Implementasikan gray world: hitung mean per channel → scale agar mean seimbang.
3. Implementasikan white patch: cari piksel paling terang → scale channel berdasarkan max.
4. Implementasikan simple scaling: normalisasi setiap channel ke range penuh.
5. Tampilkan histogram per channel sebelum dan sesudah koreksi.
6. Bandingkan ketiga metode secara visual side-by-side.
7. Uji pada gambar indoor dengan lampu tungsten (yellowish cast).
8. Uji pada gambar outdoor dengan shade (bluish cast).
9. Uji pada gambar dengan mixed lighting → evaluasi keterbatasan.
10. Hitung color temperature estimasi sebelum dan sesudah koreksi (opsional).

### Analisis Percobaan 13
- Metode white balance mana yang memberikan warna paling natural?
- Pada kondisi pencahayaan apa gray world assumption gagal?
- Mengapa white patch bisa gagal jika tidak ada objek putih di scene?
- Bagaimana cara menentukan metode terbaik secara otomatis?

---

## Percobaan 14: Synthetic Bokeh Effect

### Tujuan
Membuat efek bokeh sintetis menggunakan depth map dan variable Gaussian blur untuk simulasi depth of field.

### Dasar Teori
Efek bokeh mensimulasikan shallow depth of field kamera dengan aperture besar. Dengan depth map, setiap piksel mendapat blur proporsional terhadap jaraknya dari focal plane. Semakin jauh dari fokus, semakin besar blur radius yang diterapkan.

### Langkah Kerja
1. Load gambar portrait atau scene dengan foreground/background yang jelas.
2. Buat depth map sederhana menggunakan manual segmentation (mask foreground).
3. Definisikan focal depth (kedalaman objek yang ingin tetap tajam).
4. Hitung blur amount per piksel: `blur_sigma = k * abs(depth - focal_depth)`.
5. Implementasikan variable blur: bagi gambar ke beberapa depth layer → blur per layer.
6. Gabungkan layer menggunakan depth map sebagai alpha blending weight.
7. Variasikan konstanta `k` (1, 3, 5, 10) untuk intensitas bokeh berbeda.
8. Bandingkan Gaussian blur vs disc (uniform circular) blur kernel.
9. Tambahkan feathering di boundary foreground/background untuk transisi halus.
10. Tampilkan: original, depth map, bokeh results dengan berbagai intensitas.

### Analisis Percobaan 14
- Seberapa realistis bokeh sintetis dibandingkan bokeh optik asli?
- Apa artifact yang muncul di boundary antara foreground dan background?
- Bagaimana kualitas depth map mempengaruhi realisme bokeh?
- Apakah disc blur menghasilkan bokeh lebih natural dibanding Gaussian blur?

---

## Percobaan 15: Color Enhancement dan Saturation

### Tujuan
Meningkatkan dan memanipulasi warna gambar melalui saturasi di ruang warna HSV.

### Dasar Teori
Manipulasi channel Saturation di ruang warna HSV memungkinkan peningkatan atau pengurangan intensitas warna tanpa mengubah hue atau brightness. Selective color enhancement memungkinkan penguatan warna tertentu saja berdasarkan range hue.

### Langkah Kerja
1. Load gambar dengan warna yang kurang vibrant.
2. Konversi ke HSV menggunakan `cv2.cvtColor()`.
3. Boost saturation: kalikan channel S dengan faktor (1.2, 1.5, 2.0).
4. Reduce saturation: kalikan channel S dengan faktor (0.5, 0.3, 0.0 → grayscale).
5. Clip channel S ke range [0, 255] → konversi kembali ke BGR.
6. Implementasikan selective color: boost hanya warna tertentu (misal: hijau saja).
7. Buat mask berdasarkan range hue → boost saturation hanya di area mask.
8. Bandingkan dengan manipulasi di ruang warna LAB (channel a, b).
9. Tampilkan grid: original, saturated, desaturated, selective color.
10. Uji pada gambar landscape, food photography, dan portrait.

### Analisis Percobaan 15
- Pada faktor berapa saturation boost mulai terlihat tidak natural?
- Apa perbedaan manipulasi warna di HSV vs LAB?
- Bagaimana selective color enhancement bisa digunakan untuk efek kreatif?
- Mengapa clipping penting saat meningkatkan saturasi?

---

## Percobaan 16: Image Enhancement Pipeline Lengkap

### Tujuan
Membangun pipeline enhancement lengkap yang menggabungkan denoise, white balance, CLAHE, sharpening, dan color enhancement.

### Dasar Teori
Pipeline enhancement yang komprehensif menerapkan serangkaian operasi secara berurutan. Urutan operasi mempengaruhi hasil akhir: denoising sebaiknya dilakukan lebih awal agar noise tidak diperkuat oleh langkah berikutnya, sementara sharpening diterapkan di akhir.

### Langkah Kerja
1. Load gambar berkualitas rendah (gelap, noisy, color cast, low contrast).
2. **Step 1 — Denoise**: terapkan `cv2.fastNlMeansDenoisingColored()` dengan h=10.
3. **Step 2 — White Balance**: terapkan gray world correction pada hasil denoise.
4. **Step 3 — CLAHE**: konversi ke LAB → CLAHE pada channel L → merge kembali.
5. **Step 4 — Sharpen**: terapkan unsharp mask (sigma=1.0, amount=1.5).
6. **Step 5 — Color Enhance**: boost saturation di HSV (faktor 1.3).
7. Tampilkan hasil setiap step secara berurutan (6 gambar).
8. Ubah urutan pipeline (misal: CLAHE dulu, lalu denoise) → bandingkan.
9. Buat fungsi reusable `enhance_image(img, params)` dengan parameter adjustable.
10. Uji pipeline pada 5 gambar dengan masalah berbeda → evaluasi robustness.

### Analisis Percobaan 16
- Bagaimana urutan pipeline mempengaruhi kualitas hasil akhir?
- Step mana dalam pipeline yang memberikan perbaikan visual terbesar?
- Apakah ada kasus di mana salah satu step justru memperburuk gambar?
- Bagaimana cara membuat pipeline yang adaptif terhadap kondisi gambar?

---

## Percobaan 17: Pencil Sketch Effect

### Tujuan
Mengubah foto menjadi efek sketsa pensil menggunakan fungsi bawaan OpenCV dan teknik edge detection.

### Dasar Teori
Efek pencil sketch dapat dibuat menggunakan `cv2.pencilSketch()` yang menghasilkan sketch grayscale dan berwarna, atau secara manual dengan mendeteksi edge menggunakan Canny/Laplacian lalu membalik intensitas. `cv2.stylization()` memberikan efek lukisan dari foto.

### Langkah Kerja
1. Load gambar portrait atau scene dengan detail yang jelas.
2. Terapkan `cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)`.
3. Tampilkan hasil grayscale sketch dan color sketch.
4. Variasikan `sigma_s` (10, 30, 60, 100) → amati efek pada smoothness.
5. Variasikan `sigma_r` (0.01, 0.05, 0.1, 0.2) → amati efek pada edge sensitivity.
6. Implementasikan sketch manual: grayscale → invert → GaussianBlur → divide blend.
7. Terapkan `cv2.stylization(img, sigma_s=60, sigma_r=0.45)` untuk efek lukisan.
8. Terapkan edge detection (Canny) → invert untuk efek sketch alternatif.
9. Bandingkan semua metode sketch side-by-side.
10. Uji pada 3 jenis gambar: portrait, arsitektur, pemandangan alam.

### Analisis Percobaan 17
- Metode sketch mana yang menghasilkan efek paling realistis?
- Bagaimana sigma_s dan sigma_r mempengaruhi kualitas sketch?
- Apa kelebihan pencilSketch bawaan vs implementasi manual?
- Pada jenis gambar apa efek sketch paling menarik secara visual?

---

## Percobaan 18: Cartoon Effect dan Stylization

### Tujuan
Mengubah foto menjadi efek kartun menggunakan edge-preserving filter dan overlay edge detection.

### Dasar Teori
Efek kartun dibuat dengan mengkombinasikan area warna yang di-smooth (menggunakan bilateral filter atau edge-preserving filter) dengan garis tepi yang tegas. `cv2.edgePreservingFilter()` menghaluskan warna sambil mempertahankan edge, dan edge dari Canny/adaptive threshold ditambahkan sebagai outline.

### Langkah Kerja
1. Load gambar portrait atau scene berwarna.
2. Terapkan `cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)`.
3. Terapkan bilateral filter berulang (3-5 iterasi) untuk efek smooth lebih kuat.
4. Deteksi edge menggunakan adaptive threshold pada grayscale.
5. Gabungkan: area smooth di-AND dengan edge mask → efek kartun.
6. Variasikan jumlah iterasi bilateral (1, 3, 5, 7) → amati smoothness.
7. Variasikan parameter adaptive threshold (blockSize, C) → amati ketebalan edge.
8. Bandingkan `cv2.edgePreservingFilter()` flags=1 (Recursive) vs flags=2 (Normalized Convolution).
9. Tambahkan quantisasi warna (reduce jumlah warna ke 8-16) untuk efek kartun lebih kuat.
10. Tampilkan: original, smooth, edges, cartoon final, dan variasi parameter.

### Analisis Percobaan 18
- Apa perbedaan visual antara bilateral filter iteratif dan edgePreservingFilter?
- Berapa iterasi bilateral yang optimal untuk efek kartun natural?
- Bagaimana quantisasi warna mempengaruhi tampilan kartun?
- Pada jenis gambar apa efek kartun paling menarik?

---

## Percobaan 19: HDR dari Single Image (Pseudo-HDR)

### Tujuan
Membuat efek HDR dari satu gambar tunggal dengan mensimulasikan exposure bracketing dan menerapkan tone mapping.

### Dasar Teori
Pseudo-HDR mensimulasikan multiple exposure dari satu gambar dengan mengubah brightness/gamma secara sintetis. Gambar-gambar sintetis ini kemudian di-merge seperti HDR biasa, menghasilkan efek HDR tanpa memerlukan foto bracketed asli.

### Langkah Kerja
1. Load satu gambar LDR (foto normal, bukan HDR).
2. Buat versi underexposed: `under = np.clip(img * 0.4, 0, 255).astype(np.uint8)`.
3. Buat versi overexposed: `over = np.clip(img * 2.5, 0, 255).astype(np.uint8)`.
4. Buat 2 versi tambahan dengan gamma correction berbeda (gamma=0.5, gamma=2.0).
5. Definisikan fake exposure times untuk setiap versi sintetis.
6. Merge menggunakan `cv2.createMergeDebevec()` atau `cv2.createMergeMertens()`.
7. Terapkan tone mapping (Reinhard, Drago) pada hasil merge.
8. Bandingkan hasil pseudo-HDR dengan gambar original.
9. Variasikan range exposure sintetis (±1 EV, ±2 EV, ±3 EV) → amati efek.
10. Bandingkan pseudo-HDR dengan CLAHE → metode mana yang lebih efektif?

### Analisis Percobaan 19
- Apakah pseudo-HDR dari single image bisa mendekati kualitas HDR asli?
- Apa limitasi utama dari pendekatan single-image HDR?
- Metode merge mana (Debevec vs Mertens) yang lebih baik untuk pseudo-HDR?
- Pada jenis scene apa pseudo-HDR memberikan perbaikan paling signifikan?

---

## Percobaan 20: Style Transfer Manual

### Tujuan
Mengimplementasikan style transfer menggunakan konsep Gram matrix dan model DNN pre-trained di OpenCV.

### Dasar Teori
Style transfer memisahkan konten (struktur high-level) dan style (tekstur, warna, pola) dari gambar. Gram matrix menangkap korelasi antar feature map CNN yang merepresentasikan style. OpenCV `cv2.dnn` dapat menjalankan model style transfer yang sudah di-training untuk menerapkan style secara real-time.

### Langkah Kerja
1. Download model style transfer pre-trained (format .t7: eccv16 atau instance_norm).
2. Load model menggunakan `cv2.dnn.readNetFromTorch(model_path)`.
3. Load gambar konten dan preprocess: resize, blobFromImage (mean subtraction).
4. Jalankan forward pass: `net.setInput(blob)` → `output = net.forward()`.
5. Post-process output: denormalize, clip, transpose ke format BGR.
6. Uji dengan 3 model style berbeda (starry_night, la_muse, composition).
7. Variasikan ukuran input (256×256, 512×512, 720×720) → perhatikan detail style.
8. Implementasikan Gram matrix secara manual: `G = F^T × F` dari feature map.
9. Visualisasikan Gram matrix untuk memahami representasi style.
10. Bandingkan kecepatan dan kualitas model eccv16 vs instance_norm.

### Analisis Percobaan 20
- Bagaimana ukuran input mempengaruhi kualitas dan kecepatan style transfer?
- Apa perbedaan visual antara model eccv16 dan instance_norm?
- Bagaimana Gram matrix merepresentasikan "style" dari sebuah gambar?
- Pada jenis konten foto apa style transfer menghasilkan output terbaik?

---

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Perbandingan HDR vs exposure fusion dan pseudo-HDR dari single image.
2. Perbandingan metode denoising: speed vs quality.
3. Efektivitas inpainting dan limitasinya.
4. Deep learning SR vs interpolasi klasik.
5. Teknik enhancement lokal: CLAHE vs equalizeHist vs unsharp mask.
6. Koreksi white balance dan pengaruhnya terhadap kualitas warna.
7. Efek artistik: pencil sketch, cartoon, dan stylization.
8. Image enhancement pipeline: urutan optimal dan parameter tuning.
9. Color enhancement dan manipulasi saturasi di berbagai ruang warna.
10. Style transfer: pendekatan DNN vs manual, serta konsep Gram matrix.
11. Integrasi multiple techniques untuk computational photography berkualitas tinggi.

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`) + dataset foto.
- **Naming**: `NIM_Nama_Modul10.ipynb`
