# JOBSHEET MODUL 8: COMPUTATIONAL PHOTOGRAPHY

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

## Percobaan 2: Tone Mapping Reinhard

### Tujuan
Menerapkan tone mapping Reinhard untuk mengkonversi HDR ke gambar LDR yang bisa ditampilkan.

### Dasar Teori
Tone mapping Reinhard terinspirasi dari fotografi. Menggunakan operator global dan local untuk mengompres dynamic range sambil mempertahankan detail kontras.

### Langkah Kerja
1. Load HDR image atau buat dari bracketed exposure.
2. Terapkan tone mapping Reinhard: `cv2.createTonemapReinhard()`.
3. Variasikan parameter `gamma` (0.5, 1.0, 1.5, 2.2).
4. Variasikan `intensity` (-1.0, 0.0, 1.0).
5. Variasikan `light_adapt` (0.0, 0.5, 1.0).
6. Variasikan `color_adapt` (0.0, 0.5, 1.0).
7. Bandingkan dengan tone mapping global sederhana `cv2.createTonemap()`.
8. Tampilkan grid parameter variasi.
9. Simpan semua hasil ke folder output.
10. Analisis histogram distribusi intensitas sebelum dan sesudah.

### Analisis Percobaan 2
- Bagaimana gamma mempengaruhi brightness keseluruhan?
- Parameter mana yang paling berpengaruh pada naturalness?
- Bandingkan Reinhard dengan tonemap global sederhana.

---

## Percobaan 3: Tone Mapping Drago

### Tujuan
Menerapkan tone mapping Drago yang menggunakan skala logaritmik untuk kompresi dynamic range.

### Dasar Teori
Drago tone mapping menggunakan operator logaritmik adaptif. Parameter bias mengontrol seberapa kuat kompresi dilakukan.

### Langkah Kerja
1. Load HDR image yang sama dengan percobaan 2.
2. Terapkan Drago: `cv2.createTonemapDrago()`.
3. Variasikan parameter `gamma` (0.5, 1.0, 2.2).
4. Variasikan `saturation` (0.5, 1.0, 1.5).
5. Variasikan `bias` (0.7, 0.85, 1.0).
6. Bandingkan hasil Drago vs Reinhard secara side-by-side.
7. Bandingkan dengan Mantiuk: `cv2.createTonemapMantiuk()`.
8. Tampilkan grid perbandingan 3 tone mapper.
9. Simpan semua ke folder output.
10. Plot histogram untuk setiap metode tone mapping.

### Analisis Percobaan 3
- Operator tone mapping mana yang paling natural?
- Bagaimana bias mempengaruhi shadow dan highlight?
- Pada scene apa Drago lebih unggul dari Reinhard?

---

## Percobaan 4: Exposure Fusion Mertens

### Tujuan
Menggabungkan bracketed photos menggunakan exposure fusion tanpa HDR pipeline.

### Dasar Teori
Mertens fusion menggunakan quality measures (contrast, saturation, well-exposedness) sebagai weight per piksel untuk menghasilkan fused image langsung dalam LDR.

### Langkah Kerja
1. Load gambar bracketed yang sama.
2. Merge Mertens: `cv2.createMergeMertens()`.
3. Variasikan `contrast_weight` (0, 0.5, 1.0).
4. Variasikan `saturation_weight` (0, 0.5, 1.0).
5. Variasikan `exposure_weight` (0, 0.5, 1.0).
6. Bandingkan dengan HDR + Reinhard tone mapping.
7. Buat visualisasi weight map per exposure.
8. Clip output ke [0, 1] dan konversi ke uint8.
9. Tampilkan: exposure inputs, weight maps, fused result.
10. Simpan semua ke folder output.

### Analisis Percobaan 4
- Apakah Mertens fusion lebih natural dari HDR?
- Weight mana yang paling berpengaruh pada kualitas?
- Kapan Mertens lebih baik dari HDR dan sebaliknya?

---

## Percobaan 5: Seamless Cloning

### Tujuan
Menggabungkan objek dari satu gambar ke gambar lain menggunakan Poisson blending (seamless cloning).

### Dasar Teori
Seamless cloning menggunakan Poisson equation untuk menyamakan gradient di boundary, sehingga objek yang di-paste menyatu natural dengan background baru tanpa terlihat batas yang jelas.

### Langkah Kerja
1. Load gambar sumber (source) yang berisi objek yang ingin dipindahkan.
2. Load gambar tujuan (destination) sebagai background.
3. Buat mask untuk objek yang akan di-clone.
4. Terapkan NORMAL_CLONE: `cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)`.
5. Terapkan MIXED_CLONE: `cv2.seamlessClone(..., cv2.MIXED_CLONE)`.
6. Terapkan MONOCHROME_TRANSFER: `cv2.seamlessClone(..., cv2.MONOCHROME_TRANSFER)`.
7. Bandingkan ketiga mode cloning secara visual.
8. Variasikan posisi center → amati perbedaan hasil.
9. Uji pada 3 kombinasi gambar berbeda.
10. Simpan semua hasil ke folder output.

### Analisis Percobaan 5
- Mode cloning mana yang paling natural?
- Kapan MIXED_CLONE lebih baik dari NORMAL_CLONE?
- Apa limitasi seamless cloning?

---

## Percobaan 6: Denoising Bilateral Filter

### Tujuan
Menghilangkan noise sambil menjaga edge menggunakan bilateral filter.

### Dasar Teori
Bilateral filter menggabungkan spatial proximity dan intensity similarity. Piksel yang dekat secara spasial DAN mirip intensitasnya mendapat bobot tinggi, sehingga edge tetap tajam.

### Langkah Kerja
1. Load gambar bersih → tambahkan Gaussian noise (sigma=25, 50).
2. Terapkan bilateral filter: `cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)`.
3. Variasikan `d` (5, 9, 15).
4. Variasikan `sigmaColor` (25, 50, 75, 100).
5. Variasikan `sigmaSpace` (25, 50, 75, 100).
6. Hitung PSNR setiap hasil vs gambar asli.
7. Bandingkan dengan Gaussian blur pada sigma setara.
8. Zoom ke detail edge → bandingkan sharpness.
9. Tampilkan grid parameter comparison.
10. Simpan semua ke folder output.

### Analisis Percobaan 6
- Parameter mana yang paling berpengaruh pada kualitas?
- Bagaimana bilateral filter menjaga edge?
- Bandingkan bilateral vs Gaussian dalam hal PSNR dan visual.

---

## Percobaan 7: Denoising Non-Local Means

### Tujuan
Menghilangkan noise menggunakan Non-Local Means (NLM) yang mencari patch serupa di seluruh gambar.

### Dasar Teori
NLM menghitung similarity antar patch (bukan piksel individual). Filter strength h mengontrol trade-off antara noise removal dan detail preservation.

### Langkah Kerja
1. Load gambar bersih → tambahkan noise Gaussian.
2. Terapkan NLM: `cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)`.
3. Variasikan `h` (5, 10, 15, 20, 30).
4. Variasikan `templateWindowSize` (5, 7, 11).
5. Variasikan `searchWindowSize` (11, 21, 31).
6. Hitung PSNR dan SSIM untuk setiap konfigurasi.
7. Bandingkan NLM vs bilateral filter.
8. Plot kurva: h vs PSNR → temukan h optimal.
9. Tampilkan comparison grid.
10. Simpan semua ke folder output.

### Analisis Percobaan 7
- Pada h berapa NLM optimal (PSNR tertinggi)?
- Bagaimana searchWindowSize mempengaruhi waktu pemrosesan?
- Bandingkan NLM vs bilateral: mana yang lebih preserve detail?

---

## Percobaan 8: Image Inpainting Navier-Stokes

### Tujuan
Mengisi area gambar yang rusak/hilang menggunakan metode Navier-Stokes.

### Dasar Teori
Inpainting Navier-Stokes menggunakan persamaan diferensial parsial (PDE) untuk mempropagasi informasi dari sekitar area rusak ke dalam area yang perlu diisi.

### Langkah Kerja
1. Load gambar dan buat mask area yang ingin dihapus.
2. Terapkan NS inpainting: `cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)`.
3. Variasikan radius (1, 3, 5, 10, 20).
4. Uji menghapus teks/watermark dari gambar.
5. Uji menghapus objek kecil (noda, kabel).
6. Uji menghapus objek sedang.
7. Tampilkan: original, mask, hasil inpainting.
8. Zoom ke area inpainted → evaluasi kualitas.
9. Simpan semua ke folder output.
10. Bandingkan visual kualitas pada radius berbeda.

### Analisis Percobaan 8
- Pada radius berapa inpainting NS optimal?
- Apa limitasi NS untuk area besar?
- Bagaimana kualitas mask mempengaruhi hasil?

---

## Percobaan 9: Image Inpainting Telea

### Tujuan
Mengisi area gambar menggunakan metode Fast Marching Telea.

### Dasar Teori
Telea menggunakan Fast Marching Method yang mengisi dari boundary ke center area, menggunakan weighted average dari piksel terdekat yang diketahui.

### Langkah Kerja
1. Load gambar dan mask yang sama dengan Percobaan 8.
2. Terapkan Telea: `cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)`.
3. Variasikan radius (1, 3, 5, 10, 20).
4. Bandingkan NS vs Telea pada mask yang sama.
5. Uji pada gambar berbeda.
6. Hitung metrik kualitas (PSNR jika ada ground truth).
7. Buat comparison side-by-side NS vs Telea.
8. Buat interactive inpainting dengan mouse drawing.
9. Simpan semua ke folder output.
10. Tampilkan tabel perbandingan visual kedua metode.

### Analisis Percobaan 9
- Mana yang lebih baik: NS atau Telea pada area kecil?
- Mana yang lebih baik pada area besar?
- Berapa waktu pemrosesan masing-masing?

---

## Percobaan 10: Super Resolution Interpolasi

### Tujuan
Meningkatkan resolusi gambar menggunakan berbagai metode interpolasi klasik.

### Dasar Teori
Interpolasi menambah piksel baru berdasarkan estimasi dari piksel tetangga. Metode berbeda (nearest, linear, cubic, Lanczos) menghasilkan kualitas yang berbeda.

### Langkah Kerja
1. Load gambar high-res → downscale 4× sebagai input low-res.
2. Upscale 4× dengan INTER_NEAREST (baseline).
3. Upscale 4× dengan INTER_LINEAR.
4. Upscale 4× dengan INTER_CUBIC.
5. Upscale 4× dengan INTER_LANCZOS4.
6. Hitung PSNR dan SSIM vs gambar asli high-res.
7. Zoom ke detail → bandingkan sharpness.
8. Buat tabel perbandingan: metode, PSNR, SSIM, waktu.
9. Tampilkan grid visual comparison.
10. Simpan semua ke folder output.

### Analisis Percobaan 10
- Metode mana dengan PSNR/SSIM tertinggi?
- Perbedaan visual apa antara nearest dan Lanczos?
- Berapa trade-off kecepatan setiap metode?

---

## Percobaan 11: Neural Style Transfer

### Tujuan
Menerapkan gaya artistik dari lukisan ke foto menggunakan deep neural network.

### Dasar Teori
Neural style transfer mengoptimalkan gambar output agar memiliki content dari gambar konten (fitur CNN layer tinggi) dan style dari gambar referensi (Gram matrix).

### Langkah Kerja
1. Download pre-trained style transfer models (starry_night, mosaic, candy).
2. Load gambar konten (foto biasa).
3. Apply style transfer untuk setiap model.
4. Variasikan ukuran input (256, 512).
5. Uji pada 3 gambar konten berbeda.
6. Buat collage: konten × style kombinasi.
7. Ukur waktu processing per model.
8. Bandingkan kualitas setiap style model.
9. Tampilkan: style image, konten, hasil.
10. Simpan semua ke folder output.

### Analisis Percobaan 11
- Style mana yang paling natural menyatu dengan konten?
- Bagaimana resolusi mempengaruhi kualitas?
- Berapa waktu processing yang reasonable?

---

## Percobaan 12: Image Colorization DNN

### Tujuan
Mewarnai gambar grayscale secara otomatis menggunakan deep learning model.

### Dasar Teori
Model colorization belajar mapping dari luminance ke chrominance channel menggunakan CNN yang di-train pada dataset besar gambar berwarna. Model memprediksi channel ab di ruang warna LAB.

### Langkah Kerja
1. Download pre-trained colorization model (Zhang et al. ECCV 2016).
2. Load gambar grayscale (atau konversi gambar berwarna ke gray).
3. Preprocessing: resize, normalisasi ke ruang LAB.
4. Forward pass melalui model DNN.
5. Postprocessing: gabungkan channel L asli dengan ab prediksi.
6. Konversi kembali ke BGR.
7. Bandingkan hasil colorization dengan gambar asli berwarna.
8. Uji pada gambar hitam-putih lama.
9. Uji pada berbagai jenis scene: portrait, landscape, urban.
10. Simpan semua ke folder output.

### Analisis Percobaan 12
- Seberapa akurat prediksi warna dibandingkan ground truth?
- Pada jenis gambar apa colorization bekerja paling baik?
- Apa limitasi model colorization ini?

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
- **Naming**: `NIM_Nama_Modul08.ipynb`
