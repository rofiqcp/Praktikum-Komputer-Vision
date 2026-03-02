# NotebookLM Prompts — Modul 3: Pemrosesan Citra (Image Processing)

---

## PROMPT 1 — Slide 1-15 (Materi Dasar + Percobaan 1-8)

Buat 15 slide presentasi akademik Modul 3: Pemrosesan Citra. Referensi Szeliski (2022) Ch.3. Tiap slide informatif, sertakan formula dan kode OpenCV Python.

**Slide 1** — Judul Modul 3: Image processing mengubah piksel menjadi representasi bermakna. Tiga kategori: point operation (tiap piksel independen), neighborhood operation (area lokal), global/frequency operation. Pipeline: akuisisi, preprocessing, segmentasi, analisis, output informasi.

**Slide 2** — Histogram Gambar: $h(r_k)=n_k$ hitung distribusi intensitas. cv2.calcHist(img,[0],None,[256],[0,256]) kembalikan array 256 nilai. Visualisasi mengungkap karakter gambar: gelap, terang, kontras rendah, atau bimodal untuk pemilihan metode preprocessing yang tepat.

**Slide 3** — Histogram Equalization: CDF $s=T(r)=(L-1)\sum p(r_j)$ perluas rentang dinamis. cv2.equalizeHist beroperasi global. Kelebihan: sederhana cepat. Kekurangan: kurangi kontras lokal, munculkan artefak noise di area seragam intensitas.

**Slide 4** — CLAHE: Contrast Limited Adaptive HE bagi citra ke tile kecil, equalize tiap tile, interpolasi batas antar tile. Parameter: clipLimit (default 2.0) dan tileGridSize (default 8x8). Lebih baik dari equalizeHist global untuk gambar medis dan pemandangan berkontrasi berbeda.

**Slide 5** — Thresholding Global: Piksel $> T$ jadi 255, lainnya 0. Lima tipe cv2.threshold: THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV. Pilih T manual atau otomatis. Cocok untuk citra pencahayaan seragam dengan histogram bimodal jelas.

**Slide 6** — Thresholding Otsu dan Triangle: Otsu minimasi $\sigma^2_w(T) = w_0\sigma^2_0 + w_1\sigma^2_1$, optimal histogram bimodal. Triangle untuk histogram unimodal. Gunakan flag cv2.THRESH_OTSU gabung cv2.THRESH_BINARY; nilai T dikembalikan sebagai elemen pertama tuple output.

**Slide 7** — Adaptive Thresholding: Threshold tiap piksel berbeda berdasarkan tetangga lokal (blockSize x blockSize). ADAPTIVE_THRESH_MEAN_C (rata-rata blok - C) atau ADAPTIVE_THRESH_GAUSSIAN_C (Gaussian weighted - C). Sangat efektif dokumen scan dengan pencahayaan tidak merata.

**Slide 8** — Segmentasi Warna HSV: HSV pisahkan hue dari brightness, lebih robust perubahan cahaya dibanding BGR. cv2.cvtColor ke COLOR_BGR2HSV, lalu cv2.inRange masking. Rentang: merah H 0-10 dan 170-180, hijau H 40-80, biru H 100-130. Gabung mask merah dengan bitwise_or.

**Slide 9** — Deteksi Kontur: cv2.findContours kembalikan kontur dan hierarchy 4 level (RETR_TREE). CHAIN_APPROX_SIMPLE hemat memori. Hitung: cv2.moments, cv2.contourArea, cv2.arcLength, cv2.boundingRect, dan cv2.minEnclosingCircle untuk analisis bentuk dan posisi objek.

**Slide 10** — Konvolusi 2D: $(f*h)[x,y]=\sum_i\sum_j f[x-i,y-j]\cdot h[i,j]$ terapkan kernel $h$ ke tiap posisi $f$. cv2.filter2D(src,-1,kernel). Contoh kernel: identity, average blur 1/9, sharpen [[0,-1,0],[-1,5,-1],[0,-1,0]], emboss.

**Slide 11** — Gaussian Blur: Kernel $G(x,y)=\frac{1}{2\pi\sigma^2}e^{-(x^2+y^2)/2\sigma^2}$ bobot piksel dekat lebih besar. cv2.GaussianBlur(src,(ksize,ksize),sigmaX); ksize harus ganjil. Sigma besar blur lebih kuat. Digunakan preprocessing sebelum edge detection untuk mengurangi noise.

**Slide 12** — Median Filter: Gantikan tiap piksel dengan median tetangga; tidak buat nilai baru sehingga efektif noise salt-and-pepper. cv2.medianBlur(src,ksize); ksize harus ganjil. Kurang efektif noise Gaussian. Lebih lambat dari Gaussian untuk kernel besar karena butuh sorting.

**Slide 13** — Bilateral Filter: Edge-preserving dengan bobot $w = \exp(-d^2/2\sigma_s^2 - \Delta I^2/2\sigma_c^2)$. cv2.bilateralFilter(src,d,sigmaColor,sigmaSpace). sigmaColor besar lebih banyak warna dicampur. Jauh lebih lambat dari Gaussian namun mempertahankan tepi objek dengan baik.

**Slide 14** — Ringkasan Percobaan 1-8: 01 histogram+equalization, 02 CLAHE, 03 thresholding global+Otsu+Triangle, 04 adaptive thresholding, 05 segmentasi HSV+inRange, 06 deteksi kontur, 07 filter2D+konvolusi, 08 Gaussian+Median+Bilateral. Konsep kunci, fungsi OpenCV, parameter penting tiap percobaan.

**Slide 15** — Perbandingan Filter Spasial: Gaussian: smooth merata, cepat, hilangkan detail. Median: efektif salt-and-pepper, non-linear, lebih lambat. Bilateral: pertahankan tepi, paling lambat. Panduan: Gaussian preprocessing cepat, Median noise impulsif, Bilateral untuk segmentasi berkualitas.

---

## PROMPT 2 — Slide 16-30 (Materi Lanjut + Analisis + Tips)

Lanjutkan presentasi Modul 3: Pemrosesan Citra, slide 16-30. Fokus materi lanjut, analisis mendalam, tips praktis, dan kuis.

**Slide 16** — Sharpening: Unsharp masking $s = f + \alpha(f-\text{blur}(f))$; alpha kontrol kekuatan. Kernel Laplacian [[0,-1,0],[-1,5,-1],[0,-1,0]] perkuat tepi via filter2D. Artefak halo muncul jika alpha terlalu besar. Gunakan np.clip(result,0,255) setelah operasi aritmatika.

**Slide 17** — Deteksi Tepi Sobel: Kernel Gx dan Gy estimasi gradien horizontal/vertikal. Magnitude $M=\sqrt{G_x^2+G_y^2}$ dan arah $\theta=\arctan(G_y/G_x)$. cv2.Sobel(src,cv2.CV_64F,dx,dy,ksize). Scharr (ksize=-1) lebih akurat untuk gradien kecil. Konversi ke uint8 dengan np.absolute.

**Slide 18** — Deteksi Tepi Canny: 4 tahap: (1) Gaussian blur noise reduction, (2) Sobel gradient magnitude + arah, (3) Non-maximum suppression pertahankan lokal maksimum, (4) Hysteresis dua threshold: di atas high = tepi pasti; antara low-high = tepi jika terhubung ke tepi pasti.

**Slide 19** — Deteksi Tepi Laplacian: $\nabla^2 = \partial^2/\partial x^2 + \partial^2/\partial y^2$ deteksi perubahan gradien. LoG gabungkan smoothing Gaussian + Laplacian. DoG mendekati LoG lebih efisien. Deteksi tepi via zero-crossing output. cv2.Laplacian(gray,cv2.CV_64F,ksize).

**Slide 20** — Morfologi Erosi dan Dilasi: Erosi $A\ominus B$ mengecilkan objek putih. Dilasi $A\oplus B$ memperbesar. Structuring element: cv2.getStructuringElement dengan MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS. cv2.erode dan cv2.dilate mendukung parameter iterasi bertingkat.

**Slide 21** — Morfologi Lanjut: Opening = erosi+dilasi: hapus noise kecil tanpa ubah bentuk objek besar. Closing = dilasi+erosi: tutup lubang kecil dalam objek. Gradient = dilasi-erosi: hasilkan kontur. Semua via cv2.morphologyEx dengan MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT.

**Slide 22** — Top-Hat dan Black-Hat: Top-Hat = $I - \text{opening}(I)$: bright detail kecil, deteksi teks terang di latar gelap. Black-Hat = $\text{closing}(I) - I$: dark detail kecil, deteksi teks gelap. Keduanya via cv2.morphologyEx dengan MORPH_TOPHAT dan MORPH_BLACKHAT.

**Slide 23** — Transformasi Fourier: DFT $F(u,v)=\sum_x\sum_y f(x,y)e^{-j2\pi(ux/M+vy/N)}$ ubah ke domain frekuensi. np.fft.fft2 + np.fft.fftshift komponen DC ke tengah. Visualisasi $20\log(1+|F|)$. Frekuensi rendah di tengah = global; tinggi di tepi = detail dan tepi tajam.

**Slide 24** — Filter Frekuensi: LPF melingkar pertahankan frekuensi rendah. HPF pertahankan tinggi. Butterworth $H=1/(1+(D/D_0)^{2n})$ tanpa ringing. Alur: fft2, fftshift, kalikan masker, ifftshift, ifft2, ambil real. Gaussian LPF frekuensi ekuivalen GaussianBlur spasial.

**Slide 25** — Connected Components: cv2.connectedComponentsWithStats kembalikan label, stats (x,y,w,h,area), centroids. Label 0 = background. Visualisasi warna acak per label. Filter berdasarkan area minimal, ekstrak bounding box tiap blob. Berguna menghitung jumlah objek secara otomatis.

**Slide 26** — Analisis Perbandingan Edge Detector: Sobel: cepat, arah gradien, sensitif noise. Canny: robust, hysteresis kontinyu, tuning dua threshold. Laplacian: peka noise, zero-crossing lokalisasi tepat, dipasangkan LoG. Panduan: Canny umum, Sobel visualisasi arah, LoG analisis frekuensi.

**Slide 27** — Domain Spasial vs Domain Frekuensi: LPF spasial (Gaussian) ekuivalen LPF frekuensi (masker Gaussian/melingkar). Keunggulan frekuensi: bandpass/band-reject kompleks, hapus periodik noise dengan notch filter, analisis tekstur via spectrum. Kelemahan: komputasi lebih berat dan kurang intuitif.

**Slide 28** — Tips dan Troubleshooting: Urutan preprocessing: resize lalu CLAHE lalu bilateralFilter lalu threshold. Gunakan float32 untuk operasi aritmatika hindari overflow uint8. Setelah filter2D kernel negatif nilai bisa negatif; normalkan dengan np.clip(result,0,255).astype(np.uint8).

**Slide 29** — Kuis 5 Soal: (1) Otsu pilih T berdasarkan minimasi intra-class variance atau CDF? (2) CLAHE clipLimit=2 batasi amplifikasi 2x. (3) Opening hapus noise kecil atau tutup lubang? (4) Canny high/low rasio ideal 2-3x untuk apa? (5) Top-Hat menonjolkan bright atau dark detail kecil?

**Slide 30** — Ringkasan Semua Teknik: Pipeline: histogram+CLAHE, thresholding global/Otsu/adaptive, HSV+kontur+connected components, filter spasial (Gaussian, Median, Bilateral, filter2D), sharpening+edge (Sobel, Canny, Laplacian), morfologi (erosi, dilasi, opening, closing, top-hat), Fourier domain.

---

## PROMPT 3 — Slide 31-45 (Materi Lanjut + Project + TugasVideo)

Lanjutkan presentasi Modul 3: Pemrosesan Citra, slide 31-45. Slide 31-35 materi lanjutan, slide 36-41 project, slide 42-45 tugas video.

**Slide 31** — Watershed Segmentation: Flood-fill dari seed markers menuju watershed lines sebagai batas cekungan. Alur: threshold+morphology buat sure/unsure region, cv2.distanceTransform untuk sure foreground, markers integer, cv2.watershed(src,markers). Aplikasi: sel mikroskop bersentuhan, koin overlapping.

**Slide 32** — Distance Transform: cv2.distanceTransform(binary,distanceType,maskSize) hitung jarak tiap piksel putih ke piksel hitam terdekat. DIST_L2 = Euclidean; DIST_L1 = Manhattan. Output float32: puncak lokal = pusat objek, digunakan seed watershed. Juga dipakai morphological skeleton iteratif.

**Slide 33** — Inpainting: cv2.inpaint(src,mask,inpaintRadius,flags) perbaiki region rusak (mask=255) dari piksel sekitar. INPAINT_TELEA (fast marching dari batas ke dalam). INPAINT_NS (Navier-Stokes model fluida). Aplikasi: restorasi foto lama tergores, hapus watermark, hilangkan teks overlay.

**Slide 34** — Frequency Domain Applications: Periodik noise tampil sebagai titik simetris di luar pusat magnitude spectrum. Notch filter: tutup titik noise spesifik lalu IFFT. Analisis tekstur via pola spectrum: frekuensi dominan (kasar vs halus). Aplikasi: deteksi defek berulang permukaan produk industri.

**Slide 35** — Real-Time Image Processing: Optimasi untuk 30 fps: proses hanya ROI, resize dulu ke resolusi kecil sebelum filter berat, gunakan cv2.imshow (hindari pyplot GUI), operasi in-place dengan parameter dst. Profiling dengan time.perf_counter() per tahap. Target: setiap frame di bawah 33 ms.

**Slide 36** — Project Overview: "Pipeline Analisis Citra Lengkap" gabungkan minimal 7 teknik Modul 3. Kelompok 2-3 mahasiswa, pilih domain (pertanian/medis/industri/dokumen). Output: program Python dengan binary segmentation + statistik objek + laporan PDF minimal 5 halaman metodologi-hasil-diskusi-referensi.

**Slide 37** — Project Fase 1 - Preprocessing: cv2.imread, resize ke 640x480 (INTER_AREA). CLAHE(clipLimit=2.0, tileGridSize=(8,8)) pada channel L (konversi LAB). bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75) smooth pertahankan tepi. Simpan intermediate result untuk debugging dan laporan.

**Slide 38** — Project Fase 2 - Segmentasi: Konversi HSV, cv2.inRange masking warna target terkalibrasi. Opening (erode+dilate) hapus noise kecil, closing tutup lubang. Multi-warna: gabungkan mask dengan cv2.bitwise_or. Output: binary mask bersih siap analisis kontur dan statistik objek.

**Slide 39** — Project Fase 3 - Analisis Objek: findContours pada mask, hitung per kontur: contourArea, arcLength, boundingRect (x,y,w,h), aspect ratio = w/h, extent = area/rect_area. connectedComponentsWithStats untuk statistik semua blob. Tampilkan bounding box dan label teks pada output image.

**Slide 40** — Project Fase 4 - Domain Frekuensi: fft2 + fftshift, buat masker Gaussian LPF, kalikan spectrum, ifftshift + ifft2 + ambil real. Bandingkan dengan cv2.GaussianBlur ekuivalen menggunakan PSNR dan waktu komputasi. Analisis performa kedua pendekatan secara kuantitatif dalam laporan.

**Slide 41** — Project Kriteria Penilaian: Implementasi Teknis 40% (7+ teknik, kode bersih, parameter terdokumentasi). Analisis Hasil 25% (kuantitatif: jumlah objek, rata-rata area, distribusi). Kreativitas 15% (GUI slider, batch processing, domain unik). Laporan 20% (PDF 5+ halaman terstruktur dan jelas).

**Slide 42** — TugasVideo Overview: Video 10-15 menit demo minimal 15 dari 20 teknik Modul 3. Struktur: intro 1 menit, filter+histogram 4 menit, edge+morfologi 4 menit, aplikasi 3 menit, kesimpulan 1 menit. Screen recorder (OBS/Bandicam), narasi bahasa Indonesia jelas, kode dan output terlihat bersamaan.

**Slide 43** — TugasVideo Bagian 1 dan 2: Bagian 1: histogram sebelum/sesudah equalization, CLAHE vs equalizeHist, 5 tipe threshold, adaptive thresholding dokumen, HSV segmentasi objek nyata. Bagian 2: Gaussian vs Median vs Bilateral pada gambar bernoise, tunjukkan trade-off edge preservation vs noise removal secara visual naratif.

**Slide 44** — TugasVideo Bagian 3 dan 4: Bagian 3: Canny tuning low/high parameter, Sobel Gx-Gy-magnitude terpisah, opening hapus noise vs closing tutup lubang, top-hat pada teks. Bagian 4: Fourier magnitude spectrum, LPF melingkar vs GaussianBlur, connected components bounding box berwarna per label.

**Slide 45** — Penutup dan Referensi: Szeliski (2022) Computer Vision Ch.3, Bradski dan Kaehler (2008) Learning OpenCV, Gonzalez dan Woods Digital Image Processing 4th Ed. Selanjutnya: Modul 4 Deteksi Fitur. Seluruh 13 tujuan Modul 3 tercakup: dari histogram dasar hingga watershed dan Fourier domain filtering.

---
