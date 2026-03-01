# NotebookLM Prompts — Modul 3: Pemrosesan Citra (Image Processing)

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 3: Pemrosesan Citra. Referensi Szeliski (2022) Ch.3. Tiap slide ~500 kata, sertakan diagram, formula, dan kode OpenCV.

**Slide 1** — Judul "Modul 3: Pemrosesan Citra", subtitle "Filtering, Segmentasi, Frekuensi", ilustrasi citra input→filter→output, nama MK, semester, referensi Szeliski Ch.3.

**Slide 2** — Kategori operasi: transformasi intensitas, filter spasial, filter frekuensi, morfologi, compositing. Tujuan: tingkatkan kualitas, ekstrak informasi, siapkan input analisis.

**Slide 3** — Brightness/Contrast: output=α·input+β. α>1 naik kontras, β>0 cerah. np.clip(). Auto-contrast: scale min→0 max→255. Percobaan 1: 6 kombinasi α dan β, histogram sebelum-sesudah.

**Slide 4** — Histogram: calcHist(), distribusi intensitas. Equalization: equalizeHist() via CDF. CLAHE: createCLAHE(clipLimit, tileGridSize), adaptive lokal. Percobaan 2–3: global vs CLAHE pada wajah.

**Slide 5** — Gamma: s=c·rᵞ, γ<1 brightening, γ>1 darkening. γ=2.2 monitor. LUT efisiensi O(1). Percobaan 4: kurva 0.4/0.7/1.0/1.5/2.5. Thresholding global: BINARY/BINARY_INV/TRUNC/TOZERO. Percobaan 5.

**Slide 6** — Otsu (THRESH_OTSU): threshold optimal histogram bimodal. Triangle: histogram unimodal. Adaptive: MEAN_C atau GAUSSIAN_C, threshold lokal per block. Percobaan 6–7: perbandingan pada pencahayaan tidak merata.

**Slide 7** — Konvolusi: cv2.filter2D(img,-1,kernel). Kernel: averaging, identity, Laplacian. Padding: BORDER_REFLECT/REPLICATE. Percobaan 8 (konvolusi_manual): implementasi tanpa filter2D, bandingkan hasil.

**Slide 8** — Gaussian Blur: GaussianBlur(img,(k,k),sigma). Box filter: boxFilter() lebih cepat, kurang smooth. Percobaan 9: efek sigma=1/3/5/7, perbandingan Gaussian vs box filter.

**Slide 9** — Median: medianBlur(), ampuh salt-and-pepper, preservasi edge. Bilateral: bilateralFilter(d,sigmaColor,sigmaSpace), edge-preserving. Percobaan 10–11: perbandingan 4 filter pada gambar bising.

**Slide 10** — Sharpening: kernel [[0,−1,0],[−1,5,−1],[0,−1,0]]. Unsharp Masking: sharp=original+λ·(original−blurred). Percobaan 12: λ=0.5/1.0/2.0. Sobel: Gx+Gy, magnitude=√(Gx²+Gy²). Percobaan 13.

**Slide 11** — Canny: Gaussian→Sobel→NMS→double thresholding→hysteresis. cv2.Canny(img,threshold1,threshold2). Laplacian: turunan kedua. Percobaan 14: Sobel vs Canny vs Laplacian.

**Slide 12** — Morfologi: SE via getStructuringElement(RECT/ELLIPSE/CROSS,ksize). Erosi: susutkan bright region. Dilasi: perluas. Opening=Erosi→Dilasi: hilangkan noise. Closing=Dilasi→Erosi: tutup lubang. Percobaan 15–16.

**Slide 13** — Morfologi lanjutan: Gradient=Dilasi−Erosi (kontur). Top-hat=Original−Opening. Black-hat=Closing−Original. morphologyEx(MORPH_GRADIENT/TOPHAT/BLACKHAT). Percobaan 17.

**Slide 14** — Fourier: numpy.fft.fft2()+fftshift(). Magnitude: 20·log(|F|+1). Pusat=low freq, pinggir=high freq. Percobaan 18. Filter frekuensi: Low-pass, High-pass, Band-pass, Notch. Percobaan 19.

**Slide 15** — Alpha Blending: addWeighted(). Laplacian pyramid blending: seamless boundary. Chroma keying: inRange() HSV→mask→composite. Percobaan 20. Setup: env, download_image.py, folder image/output/.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 3, Slide 16–30. Slide 16–25: rekap, koneksi, analisis, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Rekap percobaan 1–8: brightness, histogram EQ, CLAHE, gamma, threshold global, Otsu/Triangle, adaptive, konvolusi. Grid thumbnail. Tabel nama file dan fungsi utama.

**Slide 17** — Rekap percobaan 9–14: Gaussian (sigma), bilateral (edge), median (s&p), sharpening, Sobel edge, Canny (threshold). Grid thumbnail output.

**Slide 18** — Rekap percobaan 15–20: erosi/dilasi, opening/closing, morfologi lanjutan, Fourier spectrum, filter frekuensi, alpha compositing. Grid thumbnail output.

**Slide 19** — Filter spasial (kernel kecil, lokal, intuitif) vs frekuensi (global, presisi band, efisien kernel besar). Convolution theorem: konvolusi spasial = perkalian frekuensi. Panduan pemilihan metode.

**Slide 20** — Mengapa Canny lebih baik dari Sobel? clipLimit CLAHE cegah over-amplification? Erosi vs opening perbedaan semantik? Bilateral filter preserve edge? Laplacian pyramid hilangkan jahitan?

**Slide 21** — Aplikasi: median/bilateral→medical imaging preprocessing. Otsu→dokumen OCR. Canny+Hough→deteksi garis ADAS. Morfologi→counting sel darah. Fourier→analisis tekstur. Peta ke Modul 4 dan 7.

**Slide 22** — Tips: ksize ganjil selalu. GaussianBlur sebelum Canny. CLAHE bukan equalizeHist untuk gambar real. Normalisasi ke [0,1] sebelum aritmatika float. Template pipeline: load→filter→segment→save.

**Slide 23** — Checklist: pilih filter sesuai noise type, bedakan Otsu vs adaptive, jelaskan opening vs closing, baca magnitude spectrum, terapkan Canny dengan threshold tepat. Tabel self-assessment.

**Slide 24** — Kuis: (1) Otsu vs adaptive thresholding? (2) Kenapa bilateral lebih lambat? (3) Opening—apa yang dihilangkan? (4) Low-pass frekuensi = spasial apa? (5) Canny: berapa threshold, fungsinya?

**Slide 25** — Diskusi: sigma=1 vs sigma=5 Gaussian blur. Mengapa median lebih baik dari Gaussian untuk salt-and-pepper? Sobel threshold terlalu rendah vs tinggi? Morphological gradient vs Canny?

**Slide 26** — Project: "Pipeline Pemrosesan Citra Komprehensif". Min. 10 dari 20 konsep. Tema: Medical Enhancer (CLAHE+threshold+morfologi), Document Binarizer, Noise Pipeline, Edge Segmentation, HDR Tone Mapper. Deliverable: .py, output/, laporan PDF.

**Slide 27** — 20 opsi improvisasi (pilih min. 10): auto brightness, CLAHE grid sweep, gamma LUT, Otsu multi-level, adaptive block-size, emboss kernel, Gaussian sigma sweep, box vs Gaussian, median vs Gaussian noise, bilateral sweep, unsharp λ variasi, Sobel diagonal, Canny sweep, erosi/dilasi iterasi, opening/closing SE shape, top-hat bintik, DFT normalisasi, notch filter, Laplacian blending, chroma keying.

**Slide 28** — Rubrik: Integrasi 0–40 (≥15=40, 12–14=35, 10–11=30). Fungsionalitas 0–30. Kreativitas 0–20. Dokumentasi 0–10. Total 100. Bonus +5 pipeline real-time, +3 multi-panel interaktif.

**Slide 29** — Video: 15–20 menit, screen+face-cam. Struktur: pembukaan (2 mnt), teori 5 konsep (3–4 mnt), demo 20 percobaan live (2 poin/percobaan=40 poin), project (5 mnt), penutup (1 mnt). Submit link LMS.

**Slide 30** — Rubrik video: Pembukaan (5), Teori (10), 20 percobaan (40—2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 real-time webcam, +3 animasi filter. Penalti −2/percobaan tidak tampil. "Kuasai Seni Memproses Citra Digital!"
