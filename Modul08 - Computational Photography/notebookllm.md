# NotebookLM Prompts  Modul 8: Computational Photography

---

## PROMPT 1  Slide 115 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 8: Computational Photography. Referensi Szeliski (2022) Ch.10. Tiap slide ~300 kata, sertakan formula dan kode OpenCV.

**Slide 1**  Judul "Modul 8: Computational Photography", subtitle "Meningkatkan Foto dengan Algoritma: HDR, Denoising, Inpainting, dan Artistic Effects", ilustrasi pipeline enhancement.

**Slide 2**  HDR Imaging: dynamic range scene > sensor. Pipeline: multiple exposures  Camera Response Function (Debevec)  merge HDR radiance map. cv2.createCalibrateDebevec(), cv2.createMergeDebevec().

**Slide 3**  Tone Mapping: HDR  LDR displayable. Reinhard (cv2.createTonemapReinhard(), parameter gamma/intensity/light_adapt/color_adapt). Drago (logaritmik, bias parameter). Mantiuk (contrast-based). Perbandingan visual.

**Slide 4**  Exposure Fusion (Mertens): langsung gabung bracketed ke LDR tanpa HDR pipeline. Quality measures: contrast, saturation, well-exposedness sebagai weight. cv2.createMergeMertens(). Lebih natural, tanpa tone mapping.

**Slide 5**  Seamless Cloning: Poisson blending  samakan gradient di boundary. cv2.seamlessClone(src, dst, mask, center, flag). Mode: NORMAL_CLONE, MIXED_CLONE, MONOCHROME_TRANSFER. Aplikasi: object compositing natural.

**Slide 6**  Image Denoising: Bilateral filter (edge-preserving, cv2.bilateralFilter()). Non-Local Means (patch similarity, cv2.fastNlMeansDenoisingColored()). Parameter h mengontrol strength. PSNR/SSIM sebagai metrik.

**Slide 7**  Image Inpainting: mengisi area rusak/hilang. Navier-Stokes (cv2.INPAINT_NS)  PDE propagasi. Telea (cv2.INPAINT_TELEA)  Fast Marching. cv2.inpaint(img, mask, radius, method). Hapus teks/objek.

**Slide 8**  Super Resolution: upscale gambar. Interpolasi klasik: NEAREST, LINEAR, CUBIC, LANCZOS4. cv2.resize(img, None, fx=4, fy=4, interpolation). PSNR/SSIM comparison antar metode.

**Slide 9**  Neural Style Transfer: gabungkan content satu foto dengan style lukisan. DNN model pre-trained. cv2.dnn.readNetFromTorch(). Forward pass  stylized output. Percobaan 11.

**Slide 10**  Image Colorization DNN: grayscale  color via CNN. LAB color space: L input, predict ab. Zhang et al. 2016 model. cv2.dnn.readNetFromCaffe(). Percobaan 12.

**Slide 11**  Percobaan 14: HDR imaging pipeline (3 exposures, CRF, merge, 3 tone mappers), Tone mapping Reinhard (parameter sweep), Tone mapping Drago (saturation, bias), Exposure fusion Mertens (weight comparison).

**Slide 12**  Percobaan 58: Seamless cloning (3 modes comparison), Denoising bilateral (sigmaColor/Space tuning), Denoising NLM (h optimal, PSNR curve), Inpainting NS (radius variasi, text/object removal).

**Slide 13**  Percobaan 912: Inpainting Telea (NS vs Telea comparison), Super resolution interpolasi (4 metode, PSNR/SSIM table), Neural style transfer (3 styles  3 contents), Image colorization DNN (grayscalecolor evaluation).

**Slide 14**  Percobaan 1316: White balance correction (gray world, white patch), Synthetic bokeh (depth-based blur, DOF), Color enhancement saturation (HSV boost, selective color), Enhancement pipeline (denoiseWBCLAHEsharpencolor).

**Slide 15**  Percobaan 1720: Pencil sketch effect (pencilSketch, stylization), Cartoon effect (bilateral+edge overlay), HDR dari single image (pseudo-HDR, synthetic brackets), Style transfer manual (Gram matrix, DNN torch model).

---

## PROMPT 2  Slide 1630 (Materi Lanjutan + Analisis)

Lanjutkan Modul 8, Slide 1630. Analisis mendalam, rekap, teknik advanced. Tiap slide ~300 kata.

**Slide 16**  CLAHE: Contrast Limited Adaptive Histogram Equalization. Per-tile equalization + clip limit. cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)). Lebih natural dari equalizeHist global. LAB channel L application.

**Slide 17**  Unsharp Mask: sharpened = original + amount * (original - blurred). Parameter: sigma (scale detail), amount (intensity). Edge-aware: bilateral sebagai blur. Perbandingan dengan kernel sharpening filter2D.

**Slide 18**  Photo Dehazing: Dark Channel Prior (He 2009). Model: $I(x)=J(x)t(x)+A(1-t(x))$. Estimasi atmospheric light A, transmission t(x). Guided filter untuk refine t. Recovery scene radiance J.

**Slide 19**  Rekap Percobaan 15: HDR pipeline (CRF+merge+tone map), Reinhard (gamma+light adapt), Drago (saturation+bias), Exposure fusion (Mertens weights), Seamless cloning (Poisson blending 3 modes).

**Slide 20**  Rekap Percobaan 610: Bilateral denoising (edge-preserving), NLM denoising (patch similarity), Inpainting NS (PDE), Inpainting Telea (Fast Marching), Super resolution (4 interpolasi).

**Slide 21**  Rekap Percobaan 1115: Neural style transfer (3 models), Colorization DNN (LAB prediction), White balance (3 methods), Synthetic bokeh (depth blur), Color enhancement (HSV manipulation).

**Slide 22**  Rekap Percobaan 1620: Enhancement pipeline (5-step chain), Pencil sketch (3 methods), Cartoon (bilateral+edge), Pseudo-HDR (single image), Style transfer manual (Gram matrix).

**Slide 23**  Analisis HDR: Reinhard vs Drago vs Mantiuk trade-off. Exposure fusion vs HDR: kapan masing-masing lebih baik. Pseudo-HDR limitasi (tidak menambah informasi baru).

**Slide 24**  Analisis Denoising: Bilateral vs NLM: PSNR comparison. h parameter optimal. Template window vs search window trade-off. Bilateral untuk video (fast), NLM untuk foto (quality).

**Slide 25**  Analisis Inpainting: NS vs Telea: area kecil (Telea sedikit lebih baik), area besar (keduanya gagal). Deep learning inpainting sebagai future. Radius optimal tergantung mask size.

**Slide 26**  Koneksi antar modul: Feature matching (M4)  panoramic HDR. Homography (M5)  multi-view HDR. Stitching (M6)  panoramic HDR. Motion estimation (M7)  video denoising (temporal).

**Slide 27**  Best practices: Shoot RAW untuk HDR. Exposure bracket 2 EV. Tripod untuk multi-frame. Denoising sebelum enhancement. CLAHE pada LAB channel L. Pipeline: denoiseWBcontrastsharpencolor.

**Slide 28**  Aplikasi nyata: Smartphone HDR (auto-bracket), Portrait mode (synthetic bokeh), Night mode (multi-frame denoising), Photo restoration (inpainting), Instagram filters (style transfer+color enhancement).

**Slide 29**  Checklist kompetensi: HDR pipeline, 3 tone mappers, Mertens fusion, seamless cloning, bilateral+NLM denoising, NS+Telea inpainting, super resolution, style transfer, colorization, CLAHE, unsharp mask, white balance, enhancement pipeline.

**Slide 30**  Kuis: (1) Perbedaan HDR dan exposure fusion? (2) Parameter h di NLM mengontrol apa? (3) Seamless cloning menggunakan persamaan apa? (4) CLAHE vs equalizeHist? (5) Formula unsharp mask?

---

## PROMPT 3  Slide 3145 (Analisis + Project + Tugas Video)

Lanjutkan Modul 8, Slide 3145. Slide 3135: analisis lanjutan. Slide 3641: Project. Slide 4245: Tugas Video. Tiap slide ~300 kata.

**Slide 31**  Enhancement pipeline design: urutan matters (denoise dulu agar noise tidak diperkuat). Adaptive pipeline: deteksi masalah gambar  pilih step. Modular: setiap step punya parameter independen.

**Slide 32**  Artistic effects: pencilSketch (sigma_s, sigma_r, shade_factor), stylization, cartoon (bilateralN + adaptive threshold edge). Color quantization K-Means untuk efek poster.

**Slide 33**  Deep learning untuk computational photography: Real-ESRGAN (super resolution), neural inpainting (LaMa), image-to-image translation (pix2pix). Fondasi: CNN feature extraction.

**Slide 34**  Perbandingan metode enhancement tabel: CLAHE (contrast), unsharp mask (sharpness), bilateral (denoise+edge), NLM (denoise quality), white balance (color). Parameter optimal per kasus.

**Slide 35**  Ringkasan: 20 percobaan dari HDR imaging ke artistic effects. Computational photography = melampaui batas hardware dengan software. Integration: capture  enhance  artistic  output.

**Slide 36**  Project "Computational Photography": 10 soal cerita. Integrasikan min 10 konsep. Contoh: Aplikasi Edit Foto (auto-enhance+denoise+HDR+inpainting+SR+style transfer), Restorasi Foto Lama (denoise+inpainting+CLAHE+SR+colorization).

**Slide 37**  Soal cerita 35: HDR Real Estate Photography (bracketed HDR, tone mapping, exposure fusion), Tool Penghapus Objek (YOLO+inpainting+undo), Night Photography Enhancement (burst averaging+NLM+synthetic long exposure).

**Slide 38**  Soal cerita 610: Portrait Enhancement Suite (bokeh+white balance+skin smooth), Artistic Photo Filter App (sketch+cartoon+style transfer+color effects), Batch Photo Processor, Document Enhancement Tool, Photo Comparison Dashboard.

**Slide 39**  15 improvisasi: HDR Video, Real-time Denoising, Interactive Inpainting, StyleGAN Explorer, Auto-colorize Batch, Face Beautifier, Dehazing App, dll.

**Slide 40**  Rubrik: Fungsionalitas 35%, Integrasi 20%, Kode 15%, Dokumentasi 15%, Kreativitas 15%. Bonus +5 foto sendiri semua percobaan, +5 tabel PSNR/SSIM lengkap. Format: NIM_Nama_Project08.zip.

**Slide 41**  Tips project: Foto sendiri lebih bernilai. Before/after comparison untuk setiap teknik. Pipeline modular dengan parameter tuning. Save semua intermediate results.

**Slide 42**  Tugas Video: Demo 20 percobaan (4060 menit). Highlights: HDR 3 tone mappers, seamless cloning, denoising comparison, inpainting demo, style transfer + colorization. Before/after close-up.

**Slide 43**  Tugas Video Materi: 1015 menit. Wajib: diagram HDR pipeline, denoising comparison chart, Poisson blending concept. Jelaskan kapan menggunakan teknik mana.

**Slide 44**  Tugas Video Project: Demo 1015 menit. Gunakan foto sendiri (selfie/landscape). Tunjukkan pipeline enhancement end-to-end. PSNR/SSIM metrics jika applicable.

**Slide 45**  Rubrik Video: Pembukaan 5%, Materi 15%, 20 Percobaan 40%, Project 20%, Analisis 10%, Kualitas 10%. Bonus +5 foto sendiri, +5 tabel PSNR/SSIM. "Enhance Every Pixel, Create Every Possibility!"
