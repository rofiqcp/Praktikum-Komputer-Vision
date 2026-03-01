# NotebookLM Prompts — Modul 8: Image Stitching dan Alignment

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 8: Image Stitching dan Alignment. Referensi Szeliski (2022) Ch.8. Tiap slide ~500 kata, sertakan diagram, formula, dan kode OpenCV.

**Slide 1** — Judul "Modul 8: Image Stitching dan Alignment", subtitle "Dari Foto Terpisah ke Panorama Seamless", ilustrasi pipeline stitching panorama, referensi Szeliski Ch.8.

**Slide 2** — Motivasi: menggabungkan gambar overlapping → panorama. Pipeline: Feature Detection & Matching → Homography Estimation → Image Warping → Blending. Motion Models: Translation (2 DOF), Affine (6 DOF), Homography/Projective (8 DOF: x̃'=Hx̃). Tabel DOF dan aplikasi masing-masing.

**Slide 3** — Feature-based Alignment: Deteksi SIFT/ORB/AKAZE → matching → RANSAC homography per pasangan. Image registration: H_{i→ref}=H_{i→i+1}·H_{i+1→i+2}·…·H_{n−1→ref}. Semua gambar ke koordinat referensi (gambar tengah). Diagram chain homography.

**Slide 4** — Image Warping: Forward (src→dst, bisa gaps) vs Inverse warping (dst←src, tanpa gaps, lebih baik). cv2.warpPerspective(img, H, (width, height)). Proyeksi Planar (FOV kecil <90°), Cylindrical (panorama horizontal 360°): x'=f·arctan((x−cx)/f), Spherical (full 360°): θ,ϕ mapping. Diagram 3 proyeksi.

**Slide 5** — Bundle Adjustment: chain homography akumulasi error. Minimasi total reprojection error: min_{R_i,f_i} Σ_{i,j,k} d(x_ij^k, π(R_i,f_i,X_j^k))². cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA), stitcher.stitch(images). Otomatis: feature→match→BA→blend.

**Slide 6** — Image Blending: Masalah seam (perbedaan exposure, white balance, vignetting). Teknik: No blending (hard cut), Feathering/Linear (alpha weight dari jarak seam), Multi-band Blending (Laplacian pyramid per band → blend → reconstruct). Seam Finding: Voronoi vs GraphCut (optimal minimum cost path).

**Slide 7** — Exposure Compensation: Global gain per gambar (mean matching). Per-channel gain (BGR independent). LAB color space (luminance + color). CLAHE (adaptive histogram equalization). Histogram matching. Percobaan 8: exposure berbeda → compensated seamless blend.

**Slide 8** — Percobaan 1–2: Manual Stitching Pipeline langkah demi langkah (feature match → compute homography → warp → paste). OpenCV Stitcher API (automatic mode vs manual comparison, error handling status codes).

**Slide 9** — Percobaan 3–4: Blending Comparison (no blend vs feather vs multi-band — side-by-side seam visibility). Multi-Image Panorama (5+ gambar, gambar referensi tengah, chain warp).

**Slide 10** — Percobaan 5–7: Cylindrical Projection (planar vs cylindrical comparison, distorsi tepi). Spherical Projection (3 proyeksi side-by-side, equirectangular output). Bundle Adjustment (before/after BA: drift correction, straight lines).

**Slide 11** — Percobaan 8–10: Exposure Compensation (berbagai metode global/channel/LAB/CLAHE/histogram matching perbandingan). Seam Finding (Voronoi vs GraphCut visualization, minimum cost path). Real-time/Interactive Stitching (homography caching, FPS display, live demo).

**Slide 12** — Percobaan 11–13: Panorama Cropping & Auto-crop (threshold, contour, max inscribed rectangle). Homography Estimation & Visualisasi (RANSAC tuning, dekomposisi H, reprojection error plot). Image Registration (ECC — Enhanced Correlation Coefficient vs feature-based comparison, precision).

**Slide 13** — Percobaan 14–16: Laplacian Pyramid Blending detail (build per-level, blend per level, reconstruct — visualisasi tiap level). Gain Compensation Manual (5 metode, perbandingan seam quality). Evaluasi Kualitas Stitching (PSNR, SSIM di overlap region, difference map colorized).

**Slide 14** — Percobaan 17–19: Panorama Loop Closure (drift detection, error distribution per gambar, correction). Document Alignment & Stitching (perspective correction, vertical stacking, enhanced output). Panorama Pipeline Lengkap (class-based full pipeline: PanoramaStitcher dengan auto/manual/fast mode).

**Slide 15** — Percobaan 20: Proyek Panorama Maker App (3 mode, batch processing, quality report). Setup: opencv-python, numpy, matplotlib, scipy; folder image/ output/. Rekomendasi: gunakan overlap 30–50% antar foto, hindari motion blur, jaga focal length konsisten.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 8, Slide 16–30. Slide 16–25: analisis, rekap, koneksi, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Rekap percobaan 1–10: manual pipeline, OpenCV Stitcher API, blending comparison, multi-image panorama, cylindrical, spherical, bundle adjustment, exposure compensation, seam finding, real-time stitching. Grid thumbnail. Tabel metode blending, kualitas, kecepatan.

**Slide 17** — Rekap percobaan 11–20: auto-crop, homography detail, image registration ECC, Laplacian pyramid blending, gain compensation manual, evaluasi PSNR/SSIM, loop closure, document stitching, full pipeline class, proyek app. Tabel metrik kualitas per percobaan.

**Slide 18** — Analisis mendalam: Mengapa multi-band blending lebih baik dari feathering untuk tekstur tinggi? Kapan cylindrical vs spherical projection? Akumulasi error chain homography dan pentingnya BA. PSNR vs SSIM: mana lebih relevan untuk evaluasi stitching?

**Slide 19** — Koneksi antar modul: Feature matching (Modul 7) sebagai input stitching. Kalibrasi kamera (Modul 2) → undistort sebelum stitch. Laplacian pyramid (Modul 2) → multi-band blending. Panorama output → optical flow analysis (Modul 9). SfM (Modul 11) mewarisi konsep bundle adjustment.

**Slide 20** — Best practices: undistort gambar sebelum stitching. Pilih gambar referensi di tengah untuk minimize warping. Caching homography untuk real-time. Test overlap minimal 30%. Evaluasi reprojection error < 2 pixel. Handle Stitcher status codes (OK, ERR_NEED_MORE_IMGS, ERR_CAMERA_PARAMETERS_ADJUSTMENT_FAILED).

**Slide 21** — Aplikasi nyata: Virtual tour properti (panorama per ruangan + navigation). Drone mapping/mosaicing pertanian (top-down aerial). Document/whiteboard scanner panorama. Security panoramic camera (3 feed di-merge). Gigapixel art documentation (museum lukisan besar).

**Slide 22** — Perbandingan: No blend vs Feather vs Multi-band — tabel kualitas seam, waktu proses, kompleksitas implementasi. Planar vs Cylindrical vs Spherical — tabel FOV optimal, distorsi, use-case. SIFT vs ORB sebagai feature untuk stitching — akurasi homography, kecepatan.

**Slide 23** — Checklist kompetensi: manual stitching pipeline, Stitcher API, 3 proyeksi, multi-band blending, exposure compensation, seam finding, evaluasi PSNR/SSIM, loop closure, class-based pipeline. Self-assessment tabel per percobaan.

**Slide 24** — Kuis: (1) DOF homography? (2) Mengapa gambar referensi di tengah lebih baik? (3) GraphCut seam finding meminimalkan apa? (4) SSIM mengukur apa dibanding PSNR? (5) Proyeksi apa yang cocok untuk panorama 360° horizontal?

**Slide 25** — Diskusi: Kapan Stitcher API cukup vs manual pipeline? Tradeoff bundle adjustment: akurasi vs waktu komputasi. Masalah moving objects dalam area overlap dan solusinya. Kapan exposure compensation diperlukan (indoor, outdoor, golden hour)?

**Slide 26** — Project "Panorama Maker Application". 10 soal cerita: Virtual Tour Generator (8+ foto per ruangan), Dokumentasi Whiteboard (OCR integration), Drone Mapping (6+ top-down), Panoramic Security Camera (3 feed real-time), Gigapixel Art Scanner (9+ foto grid 3×3), Peta Lantai Overhead, Street View Mini (12+ foto 360°), Dokumentasi Scene Kecelakaan, Satellite Image Compositor, Photo Booth Panorama (auto-capture 3 foto). Deliverable: .py, output/, laporan.

**Slide 27** — 15 improvisasi: 360° Panorama (12+ gambar), Automatic Panorama Ordering, HDR Panorama, Video Panorama, Drone Image Mosaic, Document Scanner Panorama, Multi-row Panorama (2D), Gigapixel Mosaic, Moving Object Removal, Custom Blending Mask, Panorama with GPS tagging, Quality Assessment Tool (PSNR/SSIM di overlap), Interactive Stitch Editor GUI, Exposure Bracketing Panorama, Cylindrical Video Loop.

**Slide 28** — Rubrik Project: Fungsionalitas 35%, Integrasi Percobaan 20%, Kualitas Kode 15%, Dokumentasi 15%, Kreativitas 15%. Bonus +5 real outdoor panorama, +3 evaluasi kuantitatif PSNR/SSIM. Format ZIP NIM_Nama_Project08.zip. Deadline 1 minggu.

**Slide 29** — Tugas Video: PENTING — tunjukkan pengambilan foto asli (video diri sendiri mengambil foto outdoor/indoor). Demo 20 percobaan LIVE (60–80 mnt). Materi (10–15 mnt): wajib diagram pipeline + perbandingan proyeksi. Tunjukkan input→proses→output panorama. Demo Project (10–15 mnt). Analisis (5 mnt).

**Slide 30** — Rubrik Video: Pembukaan (5), Materi (10), 20 Percobaan (40 — 2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 panorama outdoor asli, +3 animasi fly-through virtual tour. Penalti −2/percobaan tidak tampil. "Sambungkan Dunia dalam Satu Pandangan Panorama!"
