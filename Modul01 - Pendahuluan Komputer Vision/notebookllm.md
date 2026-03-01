# NotebookLM Prompts — Modul 1: Pendahuluan Komputer Vision

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 1: Pendahuluan Komputer Vision. Referensi Szeliski (2022) Ch.1. Tiap slide ~500 kata, sertakan diagram dan kode OpenCV.

**Slide 1** — Judul modul, subtitle "Dari Piksel ke Persepsi", nama MK, semester, referensi Szeliski Ch.1, ilustrasi pipeline visi komputer.

**Slide 2** — Definisi CV menurut Szeliski. Analogi mata manusia vs kamera. Lima aplikasi: autonomous vehicle, face recognition, medical imaging, AR, industrial inspection.

**Slide 3** — Tabel 4 bidang: Image Processing (gambar→gambar), Computer Vision (gambar→informasi), Computer Graphics (model 3D→gambar), Machine Learning (data→prediksi).

**Slide 4** — Timeline: 1960s Roberts, 1970s Marr, 1980s edge/corner, 1990s statistik, 2000s SIFT/SURF, 2010s AlexNet/YOLO, 2020s ViT/NeRF/Diffusion.

**Slide 5** — Pipeline CV: Akuisisi→Pre-processing→Ekstraksi Fitur→Analisis→Keputusan. Level: Low-level, Mid-level, High-level Vision dengan contoh tiap tahap.

**Slide 6** — Gambar sebagai fungsi I(x,y), matriks grayscale H×W, tensor RGB H×W×3, uint8 0–255. Visualisasi matriks piksel dengan nilai numerik.

**Slide 7** — Lima ruang warna: BGR, RGB, Grayscale (Y=0.114B+0.587G+0.299R), HSV, LAB. cv2.cvtColor(). Tabel kapan menggunakan tiap ruang warna.

**Slide 8** — Percobaan 1: imread() tiga mode, cetak shape+dtype. Percobaan 2: properti gambar (shape,size,ndim,nbytes,min/max/mean/std). Percobaan 3: grayscale weighted vs averaging.

**Slide 9** — Percobaan 4: konversi BGR→RGB→HSV→LAB, visualisasi tiap channel. Percobaan 5: akses piksel img[y,x], slicing ROI img[y1:y2,x1:x2], copy vs in-place.

**Slide 10** — Aritmatika: cv2.add/subtract (saturasi) vs numpy (overflow). Bitwise: AND/OR/XOR/NOT. Percobaan 6: blending additive. Percobaan 7: ekstraksi ROI via masking.

**Slide 11** — Drawing: line/rectangle/circle/ellipse/fillPoly, parameter koordinat+warna+ketebalan+LINE_AA. Teks: putText/getTextSize. Percobaan 8–9.

**Slide 12** — ROI: slicing dan selectROI. Masking biner, circular mask via cv2.circle(). Alpha blending: dst=α·src1+(1−α)·src2, addWeighted(). Percobaan 10, 17.

**Slide 13** — Resize: 5 interpolasi (NEAREST/LINEAR/CUBIC/AREA/LANCZOS4). Rotasi: getRotationMatrix2D+warpAffine. Flip: flipCode 0/1/-1. Percobaan 11–13.

**Slide 14** — Padding: copyMakeBorder (CONSTANT/REFLECT/REPLICATE/WRAP). Splitting: cv2.split(). Merging: cv2.merge(). Brightness/contrast: α·input+β. Percobaan 14–18.

**Slide 15** — Histogram: calcHist, equalizeHist via CDF. Format: JPEG (lossy), PNG (lossless), WebP (modern). Persiapan: setup env, download_image.py, folder image/ output/. Percobaan 19–20.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 1, Slide 16–30. Slide 16–25: analisis, rekap, koneksi, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Rekap percobaan 1–10: grid thumbnail loading/properti/grayscale/ruang warna/piksel/aritmatika/bitwise/drawing/teks/ROI. Tabel nama file, konsep kunci, fungsi utama.

**Slide 17** — Rekap percobaan 11–20: resize/rotasi/flip/padding/split/merge/blending/brightness/histogram/format. Kesimpulan: 20 percobaan membangun fondasi manipulasi citra OpenCV.

**Slide 18** — Mengapa OpenCV BGR bukan RGB? Kapan INTER_AREA vs INTER_CUBIC? Bagaimana bitwise AND untuk segmentasi? Tradeoff JPEG vs PNG di medical imaging?

**Slide 19** — Koneksi: ruang warna (Modul 1) → thresholding (Modul 3). Histogram → CLAHE (Modul 3). Transformasi geometri → model kamera (Modul 2). Peta ketergantungan antar modul.

**Slide 20** — Tips: selalu copy() agar original tidak berubah. Cek imread() is not None. Assert shape sebelum operasi. Normalisasi ke [0,1] untuk aritmatika float.

**Slide 21** — Aplikasi nyata: konversi HSV untuk deteksi warna robotik. Masking untuk privacy blur. Blending untuk watermark. Histogram EQ untuk foto underexposed. WebP untuk web performance.

**Slide 22** — Perbandingan interpolasi zoom 4×: NEAREST (blocky), LINEAR (smooth), CUBIC (smoother), AREA, LANCZOS4 (sharpest). Tabel kualitas, kecepatan, use-case.

**Slide 23** — Checklist kompetensi: load/save, konversi ruang warna, pilih interpolasi, buat mask, analisis histogram, simpan format tepat. Tabel self-assessment.

**Slide 24** — Kuis: (1) imread() format output? (2) Interpolasi terbaik downscale? (3) Fungsi gabungkan piksel tanpa overflow? (4) Ruang warna pemisah luminance? (5) addWeighted alpha=0.7 artinya?

**Slide 25** — Diskusi: saturasi cv2.add vs overflow numpy. EqualizeHist pada gambar kontras tinggi. Median filter untuk salt-and-pepper (preview Modul 3). Pemilihan format ekspor berdasarkan use-case.

**Slide 26** — Project: "Aplikasi Pengolahan Citra Terpadu". Integrasikan min. 10 dari 20 konsep. Tema: face preprocessing pipeline, document enhancer, art filter, smart watermarking, batch processor. Deliverable: .py, output/, laporan PDF.

**Slide 27** — 20 opsi improvisasi (pilih min. 10): loading multi-format, konversi HSV, blending multi-layer, masking kompleks, ROI interaktif, resize aspect-ratio, flip augmentasi, channel swap, auto-contrast, histogram CLAHE, multi-format export, pipeline batch, annotasi otomatis, fade animasi, blending crossfade, statistik per-region, padding dinamis, grayscale artistik, watermark semi-transparan, metadata logging.

**Slide 28** — Rubrik: Integrasi 0–40 (≥15=40, 12–14=35, 10–11=30). Fungsionalitas 0–30. Kreativitas 0–20. Dokumentasi 0–10. Total 100. Bonus +5 demo interaktif, +3 visualisasi before/after.

**Slide 29** — Tugas Video: 10–20 menit, screen+face-cam. Struktur: pembukaan (2 mnt), teori 5 konsep (3–4 mnt), demo 20 percobaan live (2 poin/percobaan=40 poin), project (5 mnt), penutup (1 mnt). Submit link YouTube/Drive ke LMS.

**Slide 30** — Rubrik video: Pembukaan (5), Teori (10), 20 percobaan (40—2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 webcam live, +3 animasi. Penalti −2/percobaan tidak tampil. "Selamat Praktikum!"
