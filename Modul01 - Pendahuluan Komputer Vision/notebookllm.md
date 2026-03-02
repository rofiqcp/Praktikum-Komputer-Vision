# NotebookLM Prompts — Modul 1: Pendahuluan Komputer Vision

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet Bagian 1)

Buat 15 slide presentasi akademik Modul 1: Pendahuluan Komputer Vision. Referensi Szeliski (2022) Ch.1. Tiap slide informatif, sertakan diagram dan kode OpenCV Python.

**Slide 1** — Judul: "Modul 1 — Pendahuluan Komputer Vision". Subtitle "Dari Piksel ke Persepsi". Referensi Szeliski Ch.1. Ilustrasi pipeline: kamera → piksel → informasi.

**Slide 2** — Definisi CV menurut Szeliski: komputer memahami konten visual. Analogi mata manusia vs kamera. Lima aplikasi nyata: self-driving car, face unlock, CT scan analysis, AR filter, quality control pabrik.

**Slide 3** — Tabel 4 bidang terkait: Image Processing (gambar→gambar lebih baik), Computer Vision (gambar→informasi/keputusan), Computer Graphics (model 3D→gambar), Machine Learning (data→prediksi). Jelaskan perbedaan dan irisan.

**Slide 4** — Timeline sejarah CV: 1960s Roberts block world, 1970s Marr representasi visual, 1980s fitur edge/corner, 1990s pendekatan statistik, 2000s SIFT/SURF/BoW, 2010s deep learning AlexNet/ResNet/YOLO, 2020s ViT/NeRF/Diffusion Models.

**Slide 5** — Pipeline CV lengkap: Akuisisi→Pre-processing→Ekstraksi Fitur→Analisis→Keputusan. Tiga level: Low-level (piksel: filtering, edge), Mid-level (fitur: segmentasi, matching), High-level (semantik: recognition, understanding).

**Slide 6** — Representasi gambar digital: fungsi I(x,y), matriks grayscale H×W, tensor BGR H×W×3, uint8 (0–255). Rumus ukuran file: W×H×C×(D/8). Contoh: 1920×1080 RGB = 5.93 MB. Sistem koordinat OpenCV: origin kiri-atas.

**Slide 7** — Enam ruang warna: BGR (default OpenCV), RGB (display), Grayscale (Y=0.299R+0.587G+0.114B), HSV (intuitif untuk deteksi warna), Lab (persepsi manusia), YCrCb (kompresi video). cv2.cvtColor(). Tabel kegunaan masing-masing.

**Slide 8** — Percobaan 1-3: P1 imread() tiga mode (COLOR/GRAYSCALE/UNCHANGED), shape, dtype. P2 properti gambar: shape, size, ndim, nbytes, statistik min/max/mean/std. P3 konversi 6 ruang warna + visualisasi channel.

**Slide 9** — Percobaan 4-6: P4 akses piksel img[y,x], modifikasi area, copy vs reference. P5 aritmatika cv2.add (saturasi) vs numpy+ (overflow), subtract, addWeighted. P6 bitwise AND/OR/XOR/NOT, masking dengan bitwise_and.

**Slide 10** — Percobaan 7-9: P7 menggambar line, rectangle, circle, ellipse, polylines, fillPoly, arrowedLine. P8 putText 5 font, getTextSize, teks dengan background box. P9 ROI extract, highlight, paste resized ROI.

**Slide 11** — Percobaan 10-12: P10 resize absolut dan skala, 5 metode interpolasi (NEAREST/LINEAR/CUBIC/AREA/LANCZOS4). P11 manual crop, center crop, grid crop 2×3. P12 rotasi getRotationMatrix2D+warpAffine, multi-sudut, tanpa crop.

**Slide 12** — Percobaan 13-15: P13 flip horizontal(1)/vertikal(0)/keduanya(-1). P14 padding copyMakeBorder: CONSTANT/REFLECT/REPLICATE/WRAP. P15 split channel cv2.split(), merge, visualisasi channel berwarna, swap channel.

**Slide 13** — Percobaan 16-18: P16 blending addWeighted alpha 0-1, transisi fade. P17 brightness (β) dan contrast (α): g(x)=α·f(x)+β, convertScaleAbs. P18 histogram calcHist, equalizeHist, CDF, histogram warna BGR.

**Slide 14** — Percobaan 19-20: P19 masking — mask persegi/lingkaran/poligon, bitwise_and, invert. P20 simpan JPEG (quality 0-100), PNG (compression 0-9), BMP, TIFF. Perbandingan ukuran file antar format.

**Slide 15** — Ruang warna mendalam: konversi BGR→HSV untuk deteksi warna, BGR→Lab untuk perbandingan warna perseptual. Operasi matriks NumPy: transpose, reshape, statistik. Histogram sebagai distribusi h(rk)=nk. Ringkasan 20 percobaan.

---

## PROMPT 2 — Slide 16–30 (Materi + Jobsheet Bagian 2)

Lanjutkan presentasi Modul 1: Pendahuluan Komputer Vision, slide 16–30. Fokus analisis mendalam, tips praktis, perbandingan, koneksi antar modul, dan kuis.

**Slide 16** — Rekap percobaan 1–10: tabel nama file, konsep kunci, fungsi OpenCV utama. P1 imread, P2 shape/dtype, P3 cvtColor, P4 akses piksel, P5 add/subtract, P6 bitwise, P7 drawing, P8 putText, P9 ROI, P10 resize.

**Slide 17** — Rekap percobaan 11–20: P11 crop, P12 rotasi, P13 flip, P14 padding, P15 split/merge, P16 blending, P17 brightness/contrast, P18 histogram, P19 masking, P20 format. 20 percobaan = fondasi lengkap manipulasi citra.

**Slide 18** — Analisis percobaan 1-5: perbedaan shape COLOR vs GRAYSCALE vs UNCHANGED. Hubungan resolusi×channel×bitdepth = memori. Mengapa OpenCV BGR bukan RGB (warisan kamera). Saturasi vs overflow.

**Slide 19** — Analisis percobaan 6-10: bitwise AND untuk masking logo. Anti-aliasing LINE_AA pada drawing. ROI sebagai view (bukan copy) — hati-hati. Perbandingan kualitas 5 interpolasi saat enlarge 4×.

**Slide 20** — Analisis percobaan 11-15: efisiensi crop via slicing. Rotasi tanpa crop: hitung canvas baru. Flip untuk augmentasi data ML. Border REFLECT vs REPLICATE use case. Channel split untuk analisis warna.

**Slide 21** — Analisis percobaan 16-20: alpha blending gradual. Auto brightness/contrast via histogram. equalizeHist meratakan CDF menjadi linear. JPEG sweet spot quality 85-90 untuk web. PNG untuk grafis/transparansi.

**Slide 22** — Tips praktis coding: selalu copy() agar original aman. Cek img is not None setelah imread. Assert shape sebelum operasi. Normalisasi ke float [0,1] untuk aritmatika. Gunakan matplotlib Agg backend untuk non-GUI.

**Slide 23** — Perbandingan interpolasi zoom 4×: NEAREST (cepat, blocky, pixel art), LINEAR (default, smooth), CUBIC (lebih halus), AREA (terbaik downscale), LANCZOS4 (terbaik upscale). Tabel kualitas vs kecepatan.

**Slide 24** — Aplikasi dunia nyata: HSV untuk deteksi warna robotika. Masking untuk privacy blur video. Blending untuk watermark. Histogram equalization untuk foto underexposed. Format WebP modern untuk web performance.

**Slide 25** — Koneksi antar modul: ruang warna (M1) → thresholding (M3). Histogram → CLAHE (M3). Transformasi geometri → model kamera (M2). Fitur dasar → feature detection (M4). Peta dependensi modul.

**Slide 26** — Checklist kompetensi: load/save gambar, konversi 6 ruang warna, pilih interpolasi tepat, buat mask berbagai bentuk, analisis histogram, simpan format optimal. Self-assessment per percobaan.

**Slide 27** — Kuis 5 soal: (1) format output imread BGR/RGB? (2) interpolasi terbaik downscale? (3) fungsi penjumlahan tanpa overflow? (4) ruang warna pemisah luminance-chrominance? (5) addWeighted alpha=0.7 artinya apa?

**Slide 28** — Diskusi: saturasi cv2.add vs overflow numpy — mana lebih aman? equalizeHist pada gambar sudah kontras tinggi — apa efeknya? Kapan masking lebih baik dari ROI slicing? Format terbaik untuk arsip medis?

**Slide 29** — Tantangan CV: variasi viewpoint, iluminasi, oklusi, skala, deformasi, background clutter, intra-class variation. Bagaimana operasi dasar M1 membantu mengatasi tantangan ini?

**Slide 30** — Ringkasan besar: 20 percobaan menguasai operasi fundamental citra. Dari akuisisi sampai penyimpanan. Library: OpenCV, NumPy, Matplotlib. Fondasi kuat untuk modul selanjutnya.

---

## PROMPT 3 — Slide 31–45 (Materi Lanjut + Project + Tugas Video)

Lanjutkan presentasi Modul 1: Pendahuluan Komputer Vision, slide 31–45. Slide 31-35: materi lanjutan dan pendalaman. Slide 36-41: Project. Slide 42-45: Tugas Video.

**Slide 31** — Pendalaman format file: JPEG lossy — DCT, quantization, quality 0-100. PNG lossless — DEFLATE, alpha channel. BMP tanpa kompresi. TIFF fleksibel. Grafik ukuran vs kualitas JPEG (quality 10/50/90).

**Slide 32** — Histogram mendalam: distribusi intensitas sebagai "sidik jari" gambar. CDF dan hubungannya dengan equalizeHist. Histogram warna per channel BGR. Perbandingan histogram gambar terang vs gelap vs normal.

**Slide 33** — Transformasi geometri mendalam: matriks rotasi 2D R(θ). warpAffine untuk translasi, rotasi, scaling. Perhitungan canvas baru agar rotasi tidak terpotong. Flip sebagai refleksi matriks. Augmentasi data dasar.

**Slide 34** — Pattern pemrograman praktikum: def/function style, docstring Indonesia, import standar, BASE_DIR/IMAGE_DIR/OUTPUT_DIR, os.makedirs, matplotlib Agg backend, tampilkan_dan_simpan(), main(), if __name__=="__main__".

**Slide 35** — Persiapan environment: Python 3.8+, pip install opencv-python numpy matplotlib. Virtual environment. download_image.py → 19 gambar sample. Struktur folder: praktikum/image/, praktikum/output/. Verifikasi cv2.__version__.

**Slide 36** — Project: "Aplikasi Pengolahan Citra Terpadu". Integrasikan minimal 10 dari 20 konsep percobaan. 20 opsi improvisasi: batch loader, metadata analyzer, color explorer, pixel art, image calculator, watermark, drawing tool, annotation, smart ROI, batch resizer.

**Slide 37** — 10 opsi improvisasi lanjutan: auto crop, panorama rotator, augmentasi data, photo frame, channel mixer, image blending transition, auto brightness, histogram matcher, color filter simulator, image quality comparator. Pilih min. 1 soal cerita.

**Slide 38** — Soal cerita 1-5: (1) Katalog Produk UMKM — standarisasi 30 foto kue, watermark, collage. (2) Presensi Kartu Mahasiswa — crop ROI kartu, timestamp. (3) Analisis Citra Medis — X-ray histogram, pseudocolor. (4) Photo Booth Festival — filter, frame, strip. (5) QC Tekstil — Lab color compare.

**Slide 39** — Soal cerita 6-10: (6) Arsip Dokumen Digital — scan, auto-crop, standar A4. (7) Instagram Filter — 6 filter, grid preview, slider. (8) Monitoring Tanaman — HSV green detection, progress. (9) Perbandingan Resolusi E-Commerce — PSNR, interpolasi. (10) ID Card Otomatis — template, CSV data, batch.

**Slide 40** — Rubrik project: Fungsionalitas 35%, Integrasi (min 10 percobaan) 20%, Kualitas Kode 15%, Dokumentasi 15%, Kreativitas 15%. Total 100. Deadline 1 minggu. Format ZIP: NIM_Nama_Project01.zip.

**Slide 41** — Tips project: mulai dari soal cerita, identifikasi 10+ percobaan relevan, buat fungsi modular, simpan semua output, buat README.md dengan screenshot. Penalti: terlambat −10%/hari, plagiat = 0.

**Slide 42** — Tugas Video: rekam video 30-50 menit, MP4 720p+. Wajib screen recording + webcam. Struktur: Pembukaan (2-3 mnt), Materi (5-8 mnt), Demo 20 Percobaan (15-25 mnt), Project (5-10 mnt), Penutup (3-5 mnt).

**Slide 43** — Detail tugas video: Pembukaan — nama, NIM, judul modul, tujuan. Materi — definisi CV bahasa sendiri, pipeline, 3 level vision, 3 contoh aplikasi, representasi matriks, 3 ruang warna. Demo — setiap percobaan tunjukkan kode, eksekusi, output, penjelasan.

**Slide 44** — Rubrik video: Pembukaan 5%, Materi 15%, Demo 20 Percobaan 40% (2 poin/percobaan), Project 20%, Penutup 10%, Kualitas A/V 10%. Bonus: editing profesional +3, visual aids +3, demo hardware +2, percobaan tambahan +2.

**Slide 45** — Penalti video: durasi <25 menit −10, >55 menit −5, tanpa webcam −10, membaca teks −10, terlambat −5/hari, plagiat = 0. Submit YouTube Unlisted/GDrive. Penamaan: NIM_Nama_Video_Modul01. "Selamat Praktikum — Fondasi CV Anda Dimulai!"
