# NotebookLM Prompts — Modul 10: Computational Photography

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 10: Computational Photography. Referensi Szeliski (2022) Ch.10. Tiap slide ~500 kata, sertakan diagram, formula, dan kode OpenCV/Python.

**Slide 1** — Judul "Modul 10: Computational Photography", subtitle "Melampaui Batas Kamera Konvensional dengan Algoritma", ilustrasi HDR + style transfer + inpainting pipeline, referensi Szeliski Ch.10.

**Slide 2** — Pendahuluan: Computational photography = teknik fotografi + image processing + CV untuk menghasilkan gambar melampaui kemampuan kamera konvensional. Topik: HDR imaging, exposure fusion, denoising, inpainting, super resolution, enhancement, style transfer. Diagram overview.

**Slide 3** — HDR Imaging: kamera konvensional ~8 stop dynamic range, scene nyata 20+ stop. Pipeline: (1) Capture bracketed (3+ exposure), (2) Alignment, (3) Merge → HDR radiance map float32, (4) Tone Mapping → LDR uint8. Camera Response Function (CRF): Z=f(E·Δt). Debevec's Method: g(Z_ij)=ln E_i+ln Δt_j.

**Slide 4** — HDR OpenCV: calibrate=cv2.createCalibrateDebevec(); response=calibrate.process(images,times). merge=cv2.createMergeDebevec(); hdr=merge.process(images,times,response). Tone Mapping operators: Drago (logarithmic adaptif), Reinhard L_d=L_w/(1+L_w), Mantiuk (contrast preserving). cv2.createTonemapReinhard(gamma=2.2).

**Slide 5** — Exposure Fusion (Mertens): alternatif HDR — langsung fuse tanpa radiance map, lebih robust. Quality measures per piksel: Contrast (Laplacian magnitude), Saturation (std deviation warna), Well-exposedness (kedekatan ke 0.5). W_ij=C_ij^wc · S_ij^ws · E_ij^we. cv2.createMergeMertens().process(images).

**Slide 6** — Image Denoising: Shot noise (Poisson), Read noise (Gaussian), Dark current. Gaussian Blur (sederhana, hilangkan detail). Bilateral Filter: I'(p)=(1/W_p)Σ G_σs(‖p−q‖)·G_σr(|I(p)−I(q)|)·I(q) — edge-preserving. NLM: rata-ratakan patch mirip dari seluruh gambar: w(p,q)=e^{−‖P(p)−P(q)‖²/h²}.

**Slide 7** — NLM OpenCV: cv2.fastNlMeansDenoisingColored(noisy, None, h=10, hForColorComponents=10). Perbandingan: Gaussian (fast, blurry), Bilateral (edge-preserving, slow), NLM (best quality, slowest). Inpainting: NS (Navier-Stokes, diffusion dari batas), Telea (Fast Marching Method). cv2.inpaint(src, mask, 3, cv2.INPAINT_NS).

**Slide 8** — Super Resolution: Interpolasi (bilinear, bicubic, Lanczos4 — fast, limited quality). DNN-based: ESRGAN, EDSR, RealESRGAN. Upscaling 4× dengan detail recovery. Evaluasi: PSNR, SSIM. Image Enhancement pipeline urutan optimal: denoise → white balance → CLAHE → unsharp mask → color enhance.

**Slide 9** — Style Transfer: content image + style image → stylized output. Content Loss: L_content=‖F_CNN(content)−F_CNN(generated)‖². Style Loss: Gram matrix G=FFᵀ, L_style=Σ_l ‖G_l^gen−G_l^style‖². Optimasi iteratif. DNN style transfer model (pre-trained untuk real-time).

**Slide 10** — Percobaan 1–4: HDR Imaging (3 exposure → HDR → 3 tone mappers: Reinhard/Drago/Mantiuk side-by-side). Tone Mapping Reinhard (parameter gamma, intensity, light_adapt). Tone Mapping Drago (saturation, bias). Exposure Fusion Mertens vs HDR (output comparison, artifact analysis).

**Slide 11** — Percobaan 5–8: Denoising Gaussian (kernel size 3/5/7/11, sigma effect pada edge). Denoising Bilateral (d=9, sigmaColor sweep 25/50/75/100, edge-preserving demo). Denoising NLM (patch size, search window, h parameter). Inpainting NS (Navier-Stokes, gambar mask manual dengan trackbar).

**Slide 12** — Percobaan 9–12: Inpainting Telea (Fast Marching, NS vs Telea side-by-side: tekstur, kecepatan). Super Resolution (bilinear vs bicubic vs Lanczos4, DNN jika tersedia). CLAHE Enhancement (clipLimit 1/2/4/8, tileGridSize effect, compare global EQ). Unsharp Mask (sigma, amount, threshold — perbandingan sharpening methods).

**Slide 13** — Percobaan 13–15: White Balance (Gray World algorithm vs White Patch vs manual). Synthetic Bokeh (depth-based blur, DOF effect, gaussian kernel radius = f(simulated depth)). Color Enhancement (saturation boost HSV, vibrance adjustment, selective color).

**Slide 14** — Percobaan 16–18: Enhancement Pipeline Full Chain (denoise → WB → CLAHE → unsharp → color — visualisasi tiap langkah). Pencil Sketch (cv2.pencilSketch() sigma_s, sigma_r, shade_factor; atau grayscale + divide by blur). Cartoon Effect (bilateral filter 5× + edge overlay Canny).

**Slide 15** — Percobaan 19–20: HDR Single Image/Pseudo-HDR (local tone mapping dari 1 foto untuk meningkatkan DR). Style Transfer Manual (Gram matrix computation, loss optimization, atau DNN model pre-trained). Setup: opencv-python, numpy, matplotlib, scipy; folder image/ output/. Foto asli sendiri untuk percobaan demo.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 10, Slide 16–30. Slide 16–25: analisis, rekap, koneksi, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Rekap percobaan 1–10: HDR Debevec, Reinhard tone map, Drago tone map, Mertens fusion, Gaussian denoise, bilateral filter, NLM, inpainting NS, inpainting Telea, super resolution. Grid thumbnail. Tabel metode, kategori, OpenCV API.

**Slide 17** — Rekap percobaan 11–20: CLAHE, unsharp mask, white balance, synthetic bokeh, color enhancement, enhancement pipeline, pencil sketch, cartoon effect, pseudo-HDR, style transfer. Tabel parameter kunci tiap percobaan dan efek visual.

**Slide 18** — Analisis mendalam: Mengapa Mertens fusion menghindari artifact tone mapping? Kapan NLM lebih baik dari bilateral (tekstur halus vs tepi tajam). CRF (Camera Response Function) dan mengapa perlu di-calibrate untuk HDR akurat. Style Loss vs Content Loss: tradeoff artistik vs konten semantik.

**Slide 19** — Koneksi antar modul: CLAHE berasal dari histogram equalization Modul 1. Laplacian pyramid Mertens menggunakan piramid dari Modul 2. Inpainting untuk menghapus seam arena stitching (Modul 8). Denoising sebelum deteksi fitur (Modul 7) meningkatkan repeatability keypoints. Super resolution preprocessing input DL (Modul 5).

**Slide 20** — Best practices: urutan pipeline enhancement penting (denoise sebelum sharpen). Jangan over-enhance (CLAHE clipLimit terlalu tinggi → artefak halo). Evaluasi kuantitatif PSNR/SSIM untuk denoising ground truth. Style transfer: balance content vs style weight untuk hasil optimal.

**Slide 21** — Aplikasi nyata: Night Mode smartphone = NLM + multi-frame averaging + super resolution. Portrait mode = synthetic bokeh dari depth estimation. Instagram/VSCO filter = saturation + color grading + vignette. Professional HDR photography = Debevec + Reinhard real estate, landscape.

**Slide 22** — Perbandingan denoising: tabel Gaussian/Bilateral/NLM — PSNR pada Gaussian noise σ=25, kecepatan (ms per megapixel), edge preservation, implementasi. Tone mapping: Reinhard vs Drago vs Mantiuk — naturalisme, color saturation, contrast, kecepatan. Super resolution: interpolasi vs DNN — PSNR/SSIM, waktu.

**Slide 23** — Checklist kompetensi: HDR merge + tone mapping, Mertens fusion, denoising 3 metode, inpainting NS+Telea, super resolution, CLAHE, white balance, synthetic bokeh, pencil/cartoon effect, style transfer, enhancement pipeline. Self-assessment tabel.

**Slide 24** — Kuis: (1) Dynamic range kamera vs scene nyata (stop)? (2) Quality measure apa saja yang digunakan Mertens? (3) Perbedaan NLM vs Bilateral filter? (4) INPAINT_NS vs INPAINT_TELEA: algoritma dasarnya? (5) Apa yang dioptimalkan oleh style transfer (2 loss)?

**Slide 25** — Diskusi: HDR vs Exposure Fusion — kapan masing-masing digunakan? Keterbatasan inpainting untuk area besar. Super resolution DNN vs interpolasi biasa: kapan worth it? Style transfer Gram matrix vs feed-forward model — tradeoff kualitas vs kecepatan real-time.

**Slide 26** — Project "Computational Photography Suite". 10 soal cerita: Aplikasi Edit Foto (auto enhance+denoise+HDR+inpaint+SR+style), Restorasi Foto Lama (denoise+inpaint+CLAHE+SR+colorization), HDR Real Estate (bracketed+fusion+batch), Tool Penghapus Objek Otomatis (YOLO mask+multi-pass inpaint), Night Photo Enhancement (burst averaging+NLM+SR), Social Media Filter App (5 preset+style+bokeh), Panoramic HDR Generator (multi-exposure+stitch), Dokumen/Whiteboard Enhancer (perspective+binarize+SR), Microscopy Image Enhancer (stack+NLM+CLAHE+measurement), AI Art Generator (style transfer 5 painting+artistic effects). Deliverable: .py, output/, laporan.

**Slide 27** — 15 improvisasi: Auto-HDR dari Video, Night Photo Enhancer pipeline, Batch Photo Enhancer, Before/After Slider interaktif, AI Background Remover+Replace, Photo Restoration Tool, Text/Watermark Remover, HDR Video per-frame, Dual Camera Bokeh, Custom Style Transfer (train sendiri jika GPU), Photo Colorization DNN, Image Forensics (deteksi manipulasi), Dynamic Range Analyzer, EXIF-aware Enhancement, Artistic Filter Gallery (10+ kombinasi filter).

**Slide 28** — Rubrik Project: Fungsionalitas 35%, Integrasi Percobaan 20%, Kualitas Kode 15%, Dokumentasi 15%, Kreativitas 15%. Bonus +5 foto asli sendiri sebagai input demonstrasi, +3 evaluasi kuantitatif PSNR/SSIM. Format ZIP NIM_Nama_Project10.zip. Deadline 1 minggu.

**Slide 29** — Tugas Video: Tunjukkan foto asli yang Anda ambil sendiri (selfie, landscape, objek). Before/after comparison harus jelas dan close-up. Demo 20 percobaan LIVE (40–60 mnt). Materi (10–15 mnt): wajib diagram HDR pipeline + denoising comparison. Demo Project (10–15 mnt). Analisis & Penutup (5 mnt).

**Slide 30** — Rubrik Video: Pembukaan (5), Materi (10), 20 Percobaan (40 — 2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 HDR foto outdoor asli mengesankan, +3 style transfer artistik. Penalti −2/percobaan tidak tampil. "Ubah Foto Biasa Menjadi Karya Luar Biasa dengan Algoritma!"
