# NotebookLM Prompts  Modul 6: Image Stitching dan Alignment

---

## PROMPT 1  Slide 115 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 6: Image Stitching dan Alignment. Referensi Szeliski (2022) Ch.8. Tiap slide ~300 kata, sertakan diagram dan kode OpenCV.

**Slide 1**  Judul "Modul 6: Image Stitching dan Alignment", subtitle "Dari Multiple Images ke Panorama Seamless", ilustrasi pipeline stitching: detect  match  warp  blend.

**Slide 2**  Pipeline stitching: (1) Feature detection+matching, (2) Homography estimation (RANSAC), (3) Image warping, (4) Blending+seam finding. Motion models: translation, affine, homography. Diagram pipeline end-to-end.

**Slide 3**  Feature-based Alignment: SIFT/ORB  BF/FLANN matching  ratio test  RANSAC homography. Image registration: menyelaraskan dua gambar ke koordinat yang sama. cv2.findHomography().

**Slide 4**  Image Warping: Forward vs inverse warping. Proyeksi: planar (perspektif), cylindrical (*atan(x/f)), spherical. cv2.warpPerspective(). Kapan menggunakan proyeksi mana.

**Slide 5**  Bundle Adjustment: optimasi global seluruh parameter kamera sekaligus. Minimasi reprojection error total. OpenCV Stitcher class: cv2.Stitcher_create(), mode PANORAMA vs SCANS.

**Slide 6**  Image Blending: masalah seam visibility. Teknik: no blend (hard cut), feathering (linear alpha), multi-band blending (Laplacian pyramid). cv2.detail.MultiBandBlender. Perbandingan visual.

**Slide 7**  Exposure Compensation dan Seam Finding: gain compensation menyelaraskan brightness. Seam finding: Voronoi, GraphCut (optimal seam). cv2.detail.VoronoiSeamFinder(), cv2.detail.GraphCutSeamFinder().

**Slide 8**  Percobaan 13: Manual stitching pipeline (featurehomographywarpblend), OpenCV Stitcher API (automatic vs manual comparison), Blending comparison (no blend vs feather vs multi-band).

**Slide 9**  Percobaan 46: Multi-image panorama (5+ gambar, referensi tengah), Cylindrical projection (planar vs cylindrical), Spherical projection (3 proyeksi side-by-side).

**Slide 10**  Percobaan 79: Bundle adjustment (before/after BA comparison), Exposure compensation (different exposure  compensated), Seam finding (Voronoi vs GraphCut visual).

**Slide 11**  Percobaan 1012: Real-time interactive stitching (live demo, FPS), Panorama auto-cropping (threshold, contour, max inscribed rect), Homography estimation visualisasi (RANSAC, reprojection error).

**Slide 12**  Percobaan 1315: Image registration (ECC vs feature-based), Laplacian pyramid blending detail (build pyramid, blend per level), Gain compensation manual (global, per-channel, LAB).

**Slide 13**  Percobaan 1618: Seam quality evaluation (PSNR, SSIM, difference map), Panorama loop closure (drift detection, error distribution), Document stitching (perspective correction, vertical stacking).

**Slide 14**  Percobaan 1920: Panorama pipeline lengkap (class-based full pipeline), Proyek panorama maker app (auto/manual/fast modes, batch processing, quality report).

**Slide 15**  Setup dan tools: opencv-contrib-python, numpy, matplotlib. Folder output/ untuk semua hasil. Fungsi download_image.py untuk dataset. Tips: ambil foto dengan overlap 30-50%.

---

## PROMPT 2  Slide 1630 (Materi Lanjutan + Analisis)

Lanjutkan slide Modul 6, Slide 1630. Analisis mendalam, rekap percobaan, koneksi antar modul. Tiap slide ~300 kata.

**Slide 16**  ECC Algorithm: cv2.findTransformECC()  iterative alignment berbasis korelasi di frekuensi. Model: translation, euclidean, affine, homography. Lebih robust vs feature-based untuk low-texture scenes.

**Slide 17**  Phase Correlation: cv2.phaseCorrelate()  estimasi translasi via cross-power spectrum FFT. Sangat cepat, noise tolerant. Limitasi: hanya translasi murni. Aplikasi: video stabilization, medical imaging.

**Slide 18**  Laplacian Pyramid Blending detail: (1) Build Gaussian pyramid dua gambar, (2) Build Laplacian pyramid, (3) Blend per level dengan mask, (4) Reconstruct dari blended pyramid. Visualisasi per level.

**Slide 19**  Wave Correction: memperbaiki horizon bergelombang pada panorama. stitcher.setWaveCorrectKind(WAVE_CORRECT_HORIZ). Penyebab: rotasi kamera tidak sempurna saat panning.

**Slide 20**  Rekap Percobaan 110: Manual pipeline, Stitcher API, blending, multi-image, cylindrical, spherical, bundle adjustment, exposure compensation, seam finding, interactive stitching. Tabel metrik.

**Slide 21**  Rekap Percobaan 1120: Auto-crop, homography visualisasi, registration, pyramid blending, gain compensation, seam quality, loop closure, document stitch, full pipeline, app. Grid visual.

**Slide 22**  Analisis: Mengapa multi-band blending lebih baik dari feathering? Kapan cylindrical projection wajib? Bagaimana bundle adjustment meningkatkan akurasi? Trade-off kualitas vs kecepatan stitching.

**Slide 23**  Metrik kualitas stitching: PSNR (pixel accuracy), SSIM (structural similarity), difference map (error localization), edge alignment (seam visibility). Interpretasi dan threshold acceptable.

**Slide 24**  Loop closure: deteksi drift saat panorama kembali ke posisi awal. Distribusi error ke seluruh chain. Pentingnya untuk panorama 360.

**Slide 25**  Panorama 360: equirectangular projection. Spherical warping semua gambar. Seam finding circular. Auto-crop dan fill. Tips pengambilan: tripod, overlap konsisten.

**Slide 26**  Koneksi antar modul: Feature matching (M4)  stitching alignment. Homography (M5)  warping. HDR+exposure (M8)  exposure compensation. Motion estimation (M7)  video panorama.

**Slide 27**  Best practices: Overlap 30-50% antar foto. Tripod untuk konsistensi. Exposure lock saat bracketing. Referensi center image. RANSAC threshold 5.0 untuk homography.

**Slide 28**  Aplikasi nyata: Google Street View, virtual tour, drone mapping, satellite imagery mosaicking, whiteboard capture, gigapixel photography, medical image registration.

**Slide 29**  Checklist kompetensi: Manual stitching, Stitcher API, 3 blending methods, 3 projections, bundle adjustment, exposure compensation, seam finding, ECC registration, quality evaluation, loop closure.

**Slide 30**  Kuis: (1) Overlap minimal untuk stitching? (2) Perbedaan planar vs cylindrical projection? (3) Fungsi seam finding? (4) ECC vs feature-based: kapan masing-masing? (5) Apa itu bundle adjustment?

---

## PROMPT 3  Slide 3145 (Analisis + Project + Tugas Video)

Lanjutkan Modul 6, Slide 3145. Slide 3135: analisis lanjutan. Slide 3641: Project. Slide 4245: Tugas Video. Tiap slide ~300 kata.

**Slide 31**  Advanced stitching: video stitching (real-time, homography caching). HDR panorama (bracket per posisi  HDR  stitch). Content-aware fill untuk area hitam setelah stitching.

**Slide 32**  Document stitching: tantangan  low texture, repetitive patterns. Pipeline: edge detection  line detection  alignment  merge. Perspective correction sebelum stitching.

**Slide 33**  Perbandingan metode blending tabel lengkap: No blend (cepat, seam visible), Feathering (medium, ghosting), Multi-band (terbaik, lambat). Parameter optimal per kasus.

**Slide 34**  Troubleshooting: few matches  lower ratio test. Wrong homography  check epipolar geometry. Ghosting  moving objects, gunakan seam finding. Exposure beda  exposure compensation.

**Slide 35**  Ringkasan: dari 2 gambar sederhana ke panorama 360 lengkap. 20 percobaan mencakup seluruh pipeline dan variasi. Integration: featurematchwarpblendevaluate.

**Slide 36**  Project "Aplikasi Stitching": 10 soal cerita. Integrasikan min 8 dari 20 konsep. Contoh: Virtual Tour Generator (8+ foto/ruangan, multi-band), Whiteboard Documenter (perspective+stitch+enhance).

**Slide 37**  Soal cerita 35: Drone Mapping (aerial stitch, area calculation), Panoramic Security Camera (3-kamera, real-time), Gigapixel Art Scanner (33 grid, zoom viewer).

**Slide 38**  Soal cerita 610: Medical Image Registration (multi-modal alignment), Satellite Mosaic (large-scale, geo-referenced), Street View Creator, Time-lapse Panorama (temporal changes), Interactive Stitching Tool.

**Slide 39**  15 improvisasi project: Real-time Stitcher, Quality Benchmark, Stitcher CLI Tool, Panorama Viewer VR, Auto-Perspective Corrector, Multi-row Stitcher, dll.

**Slide 40**  Rubrik: Fungsionalitas 35%, Integrasi 20%, Kode 15%, Dokumentasi 15%, Kreativitas 15%. Bonus +5 panorama dari foto sendiri, +3 real-time demo. Format: NIM_Nama_Project06.zip.

**Slide 41**  Tips project: Siapkan dataset foto sendiri (minimal 5 scene). Gunakan class-based architecture. Simpan intermediate results. Buat README dengan contoh output.

**Slide 42**  Tugas Video: Demo 20 percobaan (4055 menit). Highlights: manual vs auto stitching, blending comparison visual, multi-image panorama, seam finding.

**Slide 43**  Tugas Video Materi: 1015 menit penjelasan. Wajib: diagram pipeline stitching, perbandingan blending methods, proyeksi planar vs cylindrical vs spherical.

**Slide 44**  Tugas Video Project: Demo 1015 menit. Tunjukkan panorama dari foto sendiri. Before/after blending. Quality metrics. Analisis limitasi.

**Slide 45**  Rubrik Video: Pembukaan 5%, Materi 15%, 20 Percobaan 40%, Project 20%, Analisis 10%, Kualitas 10%. Bonus +5 foto sendiri semua percobaan. "Stitch Perfectly, See the Bigger Picture!"
