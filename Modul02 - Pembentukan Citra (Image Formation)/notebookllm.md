# NotebookLM Prompts — Modul 2: Pembentukan Citra (Image Formation)

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet Bagian 1)

Buat 15 slide presentasi akademik Modul 2: Pembentukan Citra (Image Formation). Referensi Szeliski (2022) Ch.2. Tiap slide informatif, sertakan diagram dan kode OpenCV Python.

**Slide 1** — Judul: "Modul 2 — Pembentukan Citra (Image Formation)". Subtitle "Dari Dunia 3D ke Piksel 2D". Referensi Szeliski Ch.2. Ilustrasi pipeline: objek 3D → lensa → sensor → matriks piksel.

**Slide 2** — Definisi image formation: proses proyeksi scene 3D ke citra 2D oleh kamera. Tiga komponen: geometri (bagaimana posisi diproyeksikan), fotometri (bagaimana cahaya direkam), sensor (bagaimana analog didigitalkan). Pentingnya memahami fondasi ini untuk seluruh pipeline CV.

**Slide 3** — Koordinat homogen: titik 2D x̃=(x,y,1)^T, titik 3D X̃=(X,Y,Z,1)^T. Keunggulan: translasi jadi perkalian matriks, titik tak hingga dapat direpresentasikan. Konversi balik: bagi komponen w. Perbandingan Cartesian vs homogen secara visual.

**Slide 4** — Hierarki transformasi 2D: Tabel DOF-matriks-invariant. Translasi 2 DOF (preserves all), Rotasi 1 DOF, Rigid 3 DOF (length+angle), Similarity 4 DOF (ratio), Affine 6 DOF (parallelism), Projective/Homography 8 DOF (hanya garis lurus). Ilustrasi efek setiap level pada persegi.

**Slide 5** — Percobaan 1-3: P1 translasi warpAffine M=[[1,0,tx],[0,1,ty]], 4 arah+diagonal. P2 rotasi getRotationMatrix2D(center,angle,scale), canvas extend agar tidak terpotong, overlay multi-sudut. P3 scaling resize() piksel absolut & faktor relatif, perbandingan 5 interpolasi (NEAREST/LINEAR/CUBIC/AREA/LANCZOS4) kualitas vs kecepatan.

**Slide 6** — Percobaan 4-5: P4 transformasi affine getAffineTransform(src3pts,dst3pts), demo shearing Sx/Sy, refleksi vertikal/horizontal/diagonal via matriks affine. P5 transformasi perspektif getPerspectiveTransform(src4pts,dst4pts)+warpPerspective, koreksi dokumen miring ke frontal, bird-eye view. Perbandingan visual affine vs perspektif.

**Slide 7** — Model kamera pinhole: λ·x_img = K·[R|t]·X_world. Matriks intrinsik K: fx/fy (focal length dalam piksel), cx/cy (principal point ≈ pusat frame), s (skew≈0 untuk kamera modern). Proyeksi: x'=fx·X/Z+cx, y'=fy·Y/Z+cy. Diagram lengkap: titik 3D → ray → bidang imej → piksel.

**Slide 8** — Percobaan 6: kalibrasi kamera checkerboard. Pipeline lengkap: cetak/tampilkan checkerboard → foto dari 15-20 sudut berbeda → findChessboardCorners() → cornerSubPix() presisi sub-piksel → calibrateCamera() → undistort(). Output: K (3×3), vektor distorsi 5 koef, Rvecs, Tvecs. Reprojection error <1px = kalibrasi baik.

**Slide 9** — Percobaan 7: proyeksi 3D ke 2D dan pose estimation. cv2.solvePnP(objPoints, imgPoints, K, distCoef): estimasi [R|t] dari ≥4 korespondensi. cv2.projectPoints(): render titik 3D ke piksel. cv2.drawFrameAxes(): overlay sumbu XYZ. Aplikasi: AR sumbu koordinat di atas checkerboard/marker. Rodrigues vector ke matriks rotasi.

**Slide 10** — Distorsi lensa: radial (k1,k2,k3) — barrel (wide-angle, k1<0) vs pincushion (telephoto, k1>0). Tangensial (p1,p2) — sensor tidak sejajar lensa. Rumus: x''=x'(1+k1r²+k2r⁴+k3r⁶). Percobaan 10: simulasi distorsi pada grid kotak menggunakan remap, visualisasi barrel vs pincushion berdampingan.

**Slide 11** — Percobaan 11: koreksi distorsi mendalam. cv2.undistort() vs cv2.remap() (cache map lebih cepat untuk banyak gambar). cv2.initUndistortRectifyMap(): precompute map_x, map_y sekali. cv2.getOptimalNewCameraMatrix(K, dist, size, alpha): alpha=0 no black border, alpha=1 keep all pixels. Verifikasi dengan foto garis arsitektur.

**Slide 12** — Percobaan 8-9: P8 shearing & refleksi. Shearing-x: M=[[1,Sx,0],[0,1,0]], shearing-y: M=[[1,0,0],[Sy,1,0]]. Refleksi diagonal = transpose+rotate. Demo 4 kombinasi. P9 komposisi transformasi: M_total=M3·M2·M1 (urutan: kanan ke kiri). Demonstrasi T·R ≠ R·T secara visual dengan grid.

**Slide 13** — Percobaan 17-18: P17 gamma correction I_out=I_in^γ. γ=0.5 (cerahkan shadow, HDR), γ=1.0 (linear), γ=2.0 (gelapkan). LUT = array 256 nilai, O(1) per piksel vs O(n) manual. np.array([((i/255)^γ*255) for i in range(256)]). P18 transformasi intensitas: log c·log(1+I), exponential, sigmoid S-curve, piecewise linear 2-titik.

**Slide 14** — Percobaan 12-13: P12 sampling & aliasing. Nyquist theorem fs≥2·fmax. Aliasing pada downscale tanpa low-pass filter (blur) = frekuensi tinggi di-alias menjadi frekuensi rendah palsu. Moiré pada tekstur periodik. Solusi: INTER_AREA (averaging) untuk downsample aman. P13 perbandingan kualitas 5 interpolasi pada zoom 4× (detail) dan shrink 1/4 (smooth).

**Slide 15** — Percobaan 14-15: P14 image pyramid Gaussian (pyrDown: blur+halve, pyrUp: double+smooth) dan Laplacian (L_i = G_i − pyrUp(G_{i+1}), band frekuensi, lossless reconstruct). Aplikasi: coarse-to-fine, efficient template matching. P15 polar coordinates: cv2.warpPolar(src, size, center, maxR, WARP_POLAR_LINEAR). Log-polar: WARP_POLAR_LOG. Unroll label botol, simetri analisis.

---

## PROMPT 2 — Slide 16–30 (Materi + Jobsheet Bagian 2)

Lanjutkan presentasi Modul 2: Pembentukan Citra, slide 16–30. Fokus analisis mendalam, perbandingan, tips praktis, koneksi antar modul, dan kuis interaktif.

**Slide 16** — Rekap percobaan 1-10: tabel nama_file | konsep_utama | fungsi_OpenCV. P1 warpAffine, P2 getRotationMatrix2D+canvas extend, P3 resize+5interpolasi, P4 getAffineTransform+shear, P5 getPerspectiveTransform+warpPerspective, P6 findChessboardCorners+calibrateCamera, P7 solvePnP+projectPoints, P8 shear+refleksi, P9 komposisi M=M3·M2·M1, P10 remap simulasi distorsi.

**Slide 17** — Rekap percobaan 11-20: P11 initUndistortRectifyMap+remap, P12 pyrDown+aliasing demo, P13 5-interpolasi perbandingan, P14 Gaussian+Laplacian pyramid, P15 warpPolar+logpolar, P16 remap efek kustom, P17 gamma correction+LUT, P18 log/exp/sigmoid intensitas, P19 citra sintetis NumPy, P20 ArUco detectMarkers+estimatePoseSingleMarkers. Fondasi image formation lengkap.

**Slide 18** — Analisis transformasi 2D: kenapa koordinat homogen vs matriks 2×2 terpisah? Unified representasi komposisi M=M_n·...·M_1. Grup matematika: tutup komposisi, ada identitas, ada invers. SVD dekomposisi affine: M=U·Σ·V^T → rotasi+skala+rotasi. Non-komutatif demo: rotasi(45)·translasi(10,0) ≠ translasi(10,0)·rotasi(45).

**Slide 19** — Analisis model kamera: FoV = 2·arctan(sensor_width/(2·f)). Fisik vs piksel: f_px = f_mm·(sensor_px/sensor_mm). Depth ambiguity: satu piksel = satu ray tak terbatas — perlu dua kamera atau informasi tambahan untuk recover Z. Point at infinity: X=(X,Y,Z,0)^T → arah saja, bukan posisi.

**Slide 20** — Analisis kalibrasi: kenapa butuh banyak foto? N foto → sistem over-constrained → least-squares robust. Coverage frame: sudut gambar paling terdistorsi, wajib tercakup. cornerSubPix: refinement iteratif (centroid, saddle-point) ke presisi 0.1px. Trade-off: lebih banyak foto = lebih akurat tapi lebih lama proses.

**Slide 21** — Analisis pyramid dan sampling: Gaussian pyramid ≡ cascade low-pass filter. Laplacian pyramid ≡ band-pass filter bank (mirip wavelet Haar). Rekonstruksi: Σ pyrUp(L_i) = original tanpa loss numerik. Use case: multiresolution template matching (coarse → fine), image blending Burt-Adelson, SIFT scale space analogy.

**Slide 22** — Percobaan 16 — remap efek kustom: cv2.remap(src, map_x, map_y, INTER_LINEAR, borderMode). Formula efek: (1) Fisheye: r_new=r·k, (2) Twirl: θ_new=θ+α·exp(-r²/2σ²), (3) Gelombang: x_new=x+A·sin(2π·y/λ), (4) Tunnel: scale=1/(1+d·r), (5) Squeeze anamorphic: x_new=x·(1+0.3·abs(y/H-0.5)). Implementasi: buat meshgrid map_x/map_y NumPy, remap sekali.

**Slide 23** — Percobaan 19 — citra sintetis: checkerboard np.indices()%2 XOR. Gradient radial: d=np.hypot(X-cx, Y-cy)/R·255. Sinusoidal: I=128+127·np.sin(2π·f·x). Noise Gaussian: I+=np.random.randn·σ. Salt-pepper: mask = rand<p, set max/min. Perlin noise approximation. Citra sintetis: ground truth untuk unit test algoritma CV.

**Slide 24** — Percobaan 20 — ArUco marker: cv2.aruco.getPredefinedDictionary(DICT_6X6_250). detectMarkers(frame, dict, params) → corners, ids, rejected. estimatePoseSingleMarkers(corners, markerSize, K, dist) → rvecs, tvecs. drawFrameAxes(frame, K, dist, rvec, tvec, length). Jarak = np.linalg.norm(tvec). Heading = cv2.Rodrigues(rvec)[0]. Aplikasi: robot gripper, AGV.

**Slide 25** — Tips praktis transformasi: INTER_AREA untuk downscale (averaging, tidak aliasing), INTER_CUBIC/LANCZOS4 untuk upscale (smooth). borderMode CONSTANT(255) untuk masking putih. WARP_INVERSE_MAP: skip inversi manual. Tambahkan margin pada canvas untuk rotasi: new_w=int(h|sin θ|+w|cos θ|), new_h=int(h|cos θ|+w|sin θ|).

**Slide 26** — Koneksi antar modul: perspektif warp (M2) → panorama stitching homography (M6). K+distorsi (M2) → stereo epipolar (M11). Image pyramid (M2) → Lucas-Kanade optical flow coarse-to-fine (M7). ArUco pose (M2) → mixed reality overlay (M6+M8). Gamma/LUT (M2) → HDR tonemapping (M8). Sampling theorem (M2) → frequency domain analysis (M3).

**Slide 27** — Kuis 5 soal: (1) Berapa DOF transformasi affine 2D? (2) Nama fungsi OpenCV untuk proyeksi titik 3D ke piksel? (3) Koefisien distorsi mana yang sebabkan barrel? (4) Apa perbedaan cv2.undistort() vs cv2.remap() performa? (5) Titik (5,3) dalam koordinat homogen 2D adalah?

**Slide 28** — Diskusi: kenapa kalibrasi perlu diulang setiap kamera berbeda unit, bukan pakai spec sheet? Dampak suhu pada focal length (expansion lensa). Wide-angle vs standard vs telephoto — mana distorsi terbesar? Mengapa fisheye tidak cukup model radial k1-k3 (perlu model equidistant atau equiangular terpisah)?

**Slide 29** — Aplikasi industri: CamScanner/Adobe Scan (perspektif warp), ADAS kalibrasi kamera+LIDAR (extrinsic matrix), medical robot kalibrasi eye-in-hand, drone orthorectified map, AR manufacturing (ArUco guidance), 360° car surround view (fisheye stitch), smartphone computational photography (lens correction).

**Slide 30** — Ringkasan modul 2: 20 percobaan solid = fondasi image formation. Jalur belajar: transformasi 2D (5 percobaan) → model kamera+kalibrasi (6 percobaan) → sampling+interpolasi (3 percobaan) → remapping+fotometri (4 percobaan) → sintetis+marker (2 percobaan). Siap untuk modul deteksi fitur, stitching, dan 3D reconstruction.

---

## PROMPT 3 — Slide 31–45 (Materi Lanjut + Project + Tugas Video)

Lanjutkan presentasi Modul 2: Pembentukan Citra, slide 31–45. Slide 31-35: pendalaman lanjutan. Slide 36-41: Project. Slide 42-45: Tugas Video.

**Slide 31** — Pendalaman geometri epipolar: Essential matrix E=t_x·R (5 DOF, kamera terkalibrasi). Fundamental matrix F (7 DOF, tanpa kalibrasi). Epipolar constraint: x'^T·F·x=0 — titik di kamera kiri berkorespondensi dengan satu garis (bukan semua piksel) di kamera kanan. cv2.findEssentialMat(), cv2.findFundamentalMat(), cv2.computeCorrespondEpilines(). Fondasi stereo dan SfM.

**Slide 32** — Pendalaman Laplacian pyramid blending (Burt & Adelson 1983): bangun L_A dan L_B dari gambar A,B. Bangun pyramid mask M. Blend: B_i = L_A_i·M_i + L_B_i·(1-M_i). Rekonstruksi dari L_blend. Hasil: blend seamless tanpa visible seam. Detail halus di high-freq band, warna smooth di low-freq band. Kode lengkap 20 baris.

**Slide 33** — Pendalaman kalibrasi advanced: Zhang's method teori ringkas — setiap foto planar memberikan dua persamaan untuk K → sistem linear → DLT → refinement Levenberg-Marquardt minimize reprojection error. ChArUco: kombinasi checkerboard+ArUco robust terhadap oklusi parsial. Stereo: cv2.stereoCalibrate → R_rl, T_rl → stereoRectify → disparity → depth = f·baseline/disparity.

**Slide 34** — Pattern kode praktikum M2: docstring Indonesian header, SCRIPT_DIR/IMAGE_DIR/OUTPUT_DIR, def demo_*(image_path=None): dengan synthetic fallback, plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,name)); plt.show(), if __name__=='__main__': panggil semua demo. Guard: if img is None: img = create_synthetic(). Konsisten di semua 20 file.

**Slide 35** — Setup dan verifikasi environment: pip install opencv-contrib-python numpy matplotlib (contrib untuk ArUco+extra). python -c "import cv2; print(cv2.__version__); import cv2.aruco; print('ArUco OK')". Jalankan download_image.py. Folder output/ dibuat otomatis os.makedirs(OUTPUT_DIR, exist_ok=True). Cek gambar: haveImageReader() sebelum imread().

**Slide 36** — Project: "Aplikasi Image Formation Terpadu". Integrasikan minimal 10 dari 20 percobaan dalam satu aplikasi bertema. 10 opsi fitur utama: (1) Interactive Perspective Crop (klik 4 titik). (2) Kalibrasi webcam nyata simpan YAML. (3) AR Frame 3D di ArUco. (4) Batch Document Scanner. (5) Lens Distortion Simulator trackbar. (6) Gamma Photo Batch. (7) Pyramid Blend Seamless. (8) Polar Unroll Label Botol. (9) Custom Remap Art Generator. (10) Virtual Ruler via solvePnP.

**Slide 37** — Opsi fitur tambahan (pilih bebas kombinasi): (11) Moiré Demo interaktif. (12) Multi-Marker AR Scene. (13) Aerial Orthorectification Simulator dengan GPS reference. (14) Sport Field Homography (lapangan ke top-view). (15) Stereo Depth Sederhana. (16) Synthetic Checkerboard Generator untuk kalibrasi. (17) Animated Transformation Viewer. (18) Multi-Scale Pyramid Explorer. (19) Log-Polar Rotation Invariance Test. (20) FoV Calculator GUI per lensa.

**Slide 38** — Soal cerita 1-5: (1) Scanner Dokumen Portable — perspektif koreksi 20 dokumen, output PNG A4 standar. (2) AR Museum — ArUco di setiap exhibit, overlay nama+deskripsi 3D realtime. (3) QC Optik Lini Produksi — kalibrasi kamera, distorsi residual <0.5px, laporan error. (4) Orthorectify Kebun — homografi aerial ke koordinat GPS referensi. (5) Virtual Ruler Ukur Objek — solvePnP + kamera terkalibrasi ukur objek nyata.

**Slide 39** — Soal cerita 6-10: (6) Bird-Eye View Lapangan Basket — 4 titik corner ke plan view 28×15m, overlay posisi pemain. (7) Photo Auto-Enhancer Gelap — gamma auto-detect histogram, batch 100 foto. (8) Lens Comparison Lab — simulasi 5 karakter lensa berbeda (k1 berbeda) pada gambar arsitektur dan evaluasi distorsi. (9) Panorama Barrel Corrector — undistort foto panorama 180°. (10) Diagnostic Loupe — remap zoom area ROI citra histologi interaktif.

**Slide 40** — Rubrik project: Fungsionalitas 35% (running tanpa crash, output sesuai deskripsi), Integrasi ≥10 percobaan 20% (tidak sekedar import, digunakan bermakna), Kualitas Kode 15% (docstring, fungsi modular, SCRIPT_DIR pattern, handle error), Dokumentasi 15% (README.md, screenshot setiap fitur, penjelasan parameter), Kreativitas 15% (UI, aplikasi nyata, keunikan). Total 100+bonus.

**Slide 41** — Tips project M2: mulai dari kalibrasi kamera nyata menggunakan webcam laptop dan checkerboard dicetak A4. Simpan K+distorsi ke YAML: cv2.FileStorage('calib.yaml','w').write('K',K). Load kembali saat project jalan. Test undistort pada foto berbeda setelah kalibrasi. Buat GUI cv2.namedWindow+trackbar untuk interaksi. Penalti: telat −10%/hari, plagiat = 0.

**Slide 42** — Tugas Video M2: rekam 30-50 menit, MP4 720p+, screen recording + webcam corner. Struktur wajib: (1) Pembukaan 2-3 menit, (2) Teori Image Formation 5-8 menit, (3) Demo 20 Percobaan 15-25 menit, (4) Project 5-10 menit, (5) Penutup 2-3 menit. Penjelasan verbal tiap demo — bukan hanya scroll kode.

**Slide 43** — Detail konten video: Pembukaan: nama/NIM/modul/tujuan. Teori: jelaskan secara lisan koordinat homogen, 6 level transformasi, pinhole model, distorsi radial+tangensial, proses kalibrasi. Demo tiap percobaan: tampilkan kode fungsi utama → jalankan → tunjukkan input+output berdampingan → jelaskan apa yang terlihat dan mengapa.

**Slide 44** — Rubrik video: Pembukaan 5% (nama jelas, tujuan tersampaikan), Teori 15% (akurasi konsep, bahasa sendiri), Demo 20 Percobaan 40% (2 poin/percobaan: kode+eksekusi+penjelasan), Project 20% (demo fungsi+jelaskan desain), Penutup 5%, Kualitas A/V 15% (suara jelas, layar terbaca). Bonus: kalibrasi fisik nyata +5, ArUco tracking live +3, diagram animasi +3, stereo depth demo +4.

**Slide 45** — Penalti & submission: durasi <25 mnt −10, >55 mnt −5, tanpa webcam −10, membaca skrip verbatim −10, terlambat −5/hari, plagiat = nilai 0. Nama file: NIM_Nama_Video_Modul02.mp4. Submit YouTube Unlisted atau Google Drive dengan link aktif minimal 6 bulan. "Selamat — Anda kini paham bagaimana kamera mengubah dunia 3D menjadi gambar 2D!"
