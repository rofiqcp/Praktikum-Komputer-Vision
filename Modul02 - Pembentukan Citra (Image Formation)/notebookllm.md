# NotebookLM Prompts — Modul 2: Pembentukan Citra (Image Formation)

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 2: Pembentukan Citra. Referensi Szeliski (2022) Ch.2. Tiap slide ~500 kata, sertakan diagram, formula, dan kode OpenCV.

**Slide 1** — Judul "Modul 2: Pembentukan Citra", subtitle "Dari Cahaya ke Piksel", ilustrasi pinhole camera, nama MK, semester, referensi Szeliski Ch.2.

**Slide 2** — Tiga aspek: Geometri (proyeksi 3D→2D), Photometry (pengukuran cahaya), Digitalisasi (sampling+kuantisasi). Aplikasi: kalibrasi kamera, koreksi distorsi, AR, rekonstruksi 3D.

**Slide 3** — Translasi 2D: matriks [[1,0,tx],[0,1,ty]], warpAffine(). Percobaan 1: geser 100px kanan dan 50px bawah, eksplorasi negatif, canvas expansion vs clipping.

**Slide 4** — Rotasi: matriks [[cosθ,−sinθ],[sinθ,cosθ]], getRotationMatrix2D(center,angle,scale). Percobaan 2: demo 30°/45°/90°/180°. Shearing: [[1,shx],[shy,1]]. Percobaan 4: shear H dan V.

**Slide 5** — Scaling: [[sx,0],[0,sy]], uniform vs anisotropic. Percobaan 3. Refleksi sumbu X/Y/origin. Komposisi: M_total=M3@M2@M1 (tidak komutatif). Percobaan 5–6.

**Slide 6** — Affine: 6 DOF, preservasi kesejajaran, 3 pasang titik, getAffineTransform(). Perspektif: 8 DOF, 4 pasang titik, getPerspectiveTransform()+warpPerspective(). Percobaan 7–8: koreksi dokumen miring.

**Slide 7** — Pinhole: x=f·X/Z, y=f·Y/Z. Matriks intrinsik K=[[fx,0,cx],[0,fy,cy],[0,0,1]]. Parameter: focal length (fx,fy), principal point (cx,cy). Diagram geometri pinhole.

**Slide 8** — Kalibrasi: estimasi K+distCoeffs via checkerboard. cv2.findChessboardCorners()→calibrateCamera(). Butuh ≥10 gambar. Percobaan 9: output camera matrix + reprojection RMSE.

**Slide 9** — Distorsi: radial (barrel k1<0, pincushion k1>0), tangensial (p1,p2). cv2.undistort() dan remap(). Percobaan 10: visualisasi barrel pada grid, sebelum-sesudah undistort.

**Slide 10** — Interpolasi: NEAREST (cepat/blocky), LINEAR (default), CUBIC (4×4 kernel), AREA (downscale), LANCZOS4 (upscale). Percobaan 11: zoom 4× dengan 5 metode, perbandingan edge sharpness.

**Slide 11** — Gaussian Pyramid: pyrDown()=blur+downscale 2×, pyrUp()=expand. Percobaan 12: 4-level pyramid, visualisasi semua level. Kegunaan: multi-scale detection, SIFT, image compression.

**Slide 12** — Laplacian Pyramid: L[i]=G[i]−expand(G[i+1]). Rekonstruksi sempurna: G[0] dari semua L[i]. Percobaan 13: bangun pyramid, rekonstruksi, verifikasi error≈0. Aplikasi: seamless blending.

**Slide 13** — Koordinat polar: warpPolar() Cartesian↔Polar. Percobaan 14. Remapping: remap(src, map_x, map_y)—fisheye dewarp, swirl, cylindrical. Percobaan 15: ripple dan fisheye correction.

**Slide 14** — Log transform: s=c·log(1+r). Gamma: s=c·rᵞ, γ<1 brightening, γ>1 darkening. Monitor γ=2.2. LUT untuk efisiensi. Percobaan 16–17: kurva dan efek visual.

**Slide 15** — Sampling: Nyquist fs≥2·f_max. Aliasing→Moiré. Kuantisasi bit-depth. Citra sintetis: gradient, checkerboard, noise. Percobaan 18–20. Setup: env, download_image.py, folder image/output/.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 2, Slide 16–30. Slide 16–25: photometry, rekap, koneksi, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Lambertian: intensitas=ρ·(n·l). Phong: Ambient+Diffuse+Specular. I(x,y)=L(x,y)·R(x,y). Percobaan 19: simulasi pencahayaan arah berbeda. Aplikasi: shape-from-shading.

**Slide 17** — Histogram EQ sintetis: CDF mapping. Global vs CLAHE. Percobaan 18: apply pada gradient+checkerboard, plot histogram sebelum-sesudah. Histogram flat = kontras maksimum.

**Slide 18** — Rekap percobaan 1–8: translasi, rotasi, scaling, shearing, refleksi, komposisi, affine, perspektif. Grid thumbnail. Tabel nama percobaan, file Python, fungsi utama.

**Slide 19** — Rekap percobaan 9–15: kalibrasi, koreksi distorsi, interpolasi (5-way), Gaussian pyramid, Laplacian pyramid, polar, remapping. Grid thumbnail output.

**Slide 20** — Rekap percobaan 16–20: log, gamma, histogram EQ, pencahayaan, citra sintetis (6 jenis). Photometry = kunci memahami exposure dan color balance kamera.

**Slide 21** — Koneksi: Pinhole→3D proyeksi→distorsi→kalibrasi→AR. Transformasi geometri→perspektif correction. Pyramid→multi-scale→SIFT (Modul 7)→stitching (Modul 8). Photometry→normalisasi→CNN (Modul 5).

**Slide 22** — Mengapa 4 titik untuk perspektif tapi 3 untuk affine? Akibat fx≠fy? Bagaimana Laplacian pyramid seamless blending? Mengapa gamma penting sebelum training CNN?

**Slide 23** — Tips: undistort sebelum transformasi geometri lanjut. Gunakan remap untuk pipeline efisien. Cek reprojection error<1 pixel untuk kalibrasi valid. Template pipeline: load→undistort→warp→analyze.

**Slide 24** — Kuis: (1) DOF affine vs perspektif? (2) Principal point di matriks K? (3) Operasi pyrDown()? (4) Interpolasi terbaik upscale? (5) Fungsi distCoeffs pada undistort()?

**Slide 25** — Diskusi: affine vs perspektif untuk koreksi dokumen. Kapan Gaussian vs Laplacian pyramid? Mengapa kalibrasi perlu banyak sudut (tidak semua frontal)? Keuntungan warpPolar untuk iris mata?

**Slide 26** — Project: "Sistem Citra Berbasis Transformasi dan Kamera". Min. 10 dari 20 konsep. Tema: Document Scanner, Multi-scale Analyzer, Camera Calibration Tool, Photo Warping Studio, Virtual Mirror. Deliverable: .py, output/, laporan PDF.

**Slide 27** — 20 opsi improvisasi (pilih min. 10): translasi interaktif, animasi rotasi, scaling adaptif, shear artistik, refleksi watermark, komposisi multi-step, affine dokumen, perspektif auto, kalibrasi asli, undistort webcam, 5-interpolasi grid, Gaussian pyramid kompresi, Laplacian blending, polar iris, fisheye dewarp, gamma batch, Lambertian 3D, CLAHE sintetis, noise synthesis, citra prosedural.

**Slide 28** — Rubrik: Integrasi 0–40 (≥15=40, 12–14=35, 10–11=30). Fungsionalitas 0–30. Kreativitas 0–20. Dokumentasi 0–10. Total 100. Bonus +5 kalibrasi hardware, +5 AR overlay.

**Slide 29** — Video: 15–20 menit, screen+face-cam. Struktur: pembukaan (2 mnt), teori 5 konsep (3–4 mnt), demo 20 percobaan live (2 poin/percobaan=40 poin), project (5 mnt), penutup (1 mnt). Submit link LMS.

**Slide 30** — Rubrik video: Pembukaan (5), Teori (10), 20 percobaan (40—2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 kalibrasi hardware, +3 AR fisik. Penalti −2/percobaan tidak tampil. "Pahami Bagaimana Kamera Membentuk Realitas Digital!"
