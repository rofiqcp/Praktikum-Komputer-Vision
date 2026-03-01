# PROJECT MODUL 9: ESTIMASI GERAK (MOTION ESTIMATION)

---

## Deskripsi Umum
Project mengintegrasikan konsep motion estimation: optical flow, background subtraction, tracking, stabilization, dan motion analysis. Pilih minimal 1 soal cerita.

---

## Daftar Improvisasi Percobaan (15 Pengembangan)

1. **Speed Estimator** — Estimasi kecepatan objek (km/h) dari optical flow + kalibrasi.
2. **Crowd Flow Analysis** — Visualisasikan arah dan kepadatan pergerakan kerumunan.
3. **Gesture-based Video Player** — Kontrol play/pause/seek berdasarkan gerakan tangan.
4. **Smart Motion Detection** — BGS + filtering: hanya alert pada motion signifikan (bukan daun, cahaya).
5. **Object Counter (Line Crossing)** — Hitung objek yang melewati garis virtual.
6. **Trajectory Prediction** — Prediksi posisi objek beberapa frame ke depan.
7. **Kalman Filter Tracker** — Implementasi Kalman filter untuk smooth tracking.
8. **Action Replay Generator** — Auto-detect motion peaks → generate slow-motion replay.
9. **Heatmap Accumulator** — Akumulasi posisi objek → heatmap area sering dilalui.
10. **SORT Tracker** — Simple Online Realtime Tracking (Hungarian + Kalman).
11. **Pan-Tilt-Zoom Estimator** — Dekomposisi gerakan kamera (pan, tilt, zoom) dari optical flow.
12. **Video Synopsis** — Rangkum video panjang dengan menempatkan events dalam waktu bersamaan.
13. **Abnormal Motion Detector** — Deteksi gerakan abnormal (kecepatan/arah tidak biasa).
14. **Interactive Dense Flow Viewer** — GUI: klik piksel → tampilkan flow vector + magnitude.
15. **Temporal Super Resolution** — Frame interpolation advanced untuk video frame rate sangat rendah.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Monitoring Lalu Lintas
Dinas perhubungan ingin menganalisis lalu lintas dari CCTV. Buatlah: (a) deteksi kendaraan bergerak (BGS), (b) tracking per kendaraan (CSRT/KCF), (c) counting kendaraan melewati garis virtual, (d) estimasi kecepatan dari perpindahan piksel, (e) klasifikasi arah (kiri/kanan), (f) statistik per jam, (g) alert jika kecepatan > threshold.

### Soal 2: Sistem Keamanan Motion Detection
Pengelola gedung ingin CCTV pintar. Buatlah: (a) background subtraction (MOG2), (b) filtering noise (morphology, area threshold), (c) bounding box objek bergerak, (d) alert hanya jika objek cukup besar dan bergerak cukup lama, (e) recording otomatis saat motion detected, (f) motion history visualization, (g) log event + screenshot.

### Soal 3: Video Stabilizer Pro
YouTuber memerlukan tool stabilisasi video gratis. Buatlah: (a) deteksi gerakan kamera per frame, (b) estimasi transformasi, (c) trajectory smoothing (coba moving average dan Kalman), (d) stabilize + crop, (e) handle rotation + translation, (f) before/after comparison, (g) batch mode (drag-drop video).

### Soal 4: Sistem Penghitung Pengunjung
Mall ingin menghitung pengunjung dari CCTV overhead. Buatlah: (a) background subtraction, (b) deteksi blob pengunjung, (c) tracking per orang, (d) counting masuk/keluar (line crossing), (e) trajectory tracing, (f) statistik per interval (per 10 menit), (g) heatmap area popular.

### Soal 5: Analisis Gerakan Olahraga
Pelatih tenis ingin menganalisis gerakan serve pemain. Buatlah: (a) dense optical flow pada video serve, (b) segmentasi pemain dari background, (c) motion magnitude per body region, (d) trajectory tangan/raket, (e) perbandingan 2 serve berbeda, (f) frame interpolation untuk slow-motion replay, (g) visualisasi MHI.

### Soal 6: Virtual Tripod (Stabilizer + Panorama)
Fotografer ingin menggabungkan stabilization + panorama dari video handheld. Buatlah: (a) stabilize video, (b) detect keyframes (saat kamera berhenti), (c) stitch keyframes menjadi panorama, (d) atau: seluruh frame di-stabilize → mosaic, (e) crop + blend, (f) export panorama, (g) export stabilized video.

### Soal 7: Baby Monitor Pintar
Orang tua ingin monitor bayi yang mendeteksi gerakan berlebihan. Buatlah: (a) background subtraction (area tempat tidur bayi), (b) motion level scoring (0-100), (c) alert jika gerakan > threshold (bayi menangis), (d) alert jika TIDAK ada gerakan > 30 detik (bayi terlalu diam), (e) recording saat alert, (f) MHI visualization, (g) daily motion log.

### Soal 8: Drone Flight Stabilizer
Operator drone ingin stabilisasi video dari drone. Buatlah: (a) estimasi global motion dari optical flow, (b) dekomposisi: translasi + rotasi + zoom, (c) stabilize translasi + rotasi, (d) preservasi gerakan intentional (panning), (e) crop management, (f) before/after comparison, (g) export stabilized.

### Soal 9: Interactive Slow Motion Creator
Content creator ingin tool slow-motion dari video standar (30fps). Buatlah: (a) select region of interest di timeline, (b) frame interpolation (flow-based) untuk 2×, 4×, 8× slowdown, (c) smooth transisi normal→slow→normal, (d) kualitas comparison per metode, (e) export clip, (f) FPS display, (g) side-by-side: linear vs flow-based.

### Soal 10: Smart Whiteboard Recorder
Dosen ingin merekam presentasi whiteboard tanpa menangkap gerakan tangan/tubuh. Buatlah: (a) background subtraction (person vs whiteboard), (b) hanya update area tulisan baru (bukan area person), (c) akumulasi tulisan dari waktu ke waktu, (d) generate gambar whiteboard bersih per interval, (e) timeline: board state per menit, (f) auto-screenshot saat tulisan baru, (g) export PDF semua board states.

---

## Rubrik Penilaian Project

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Fungsionalitas | 35% | Semua fitur berjalan |
| Integrasi Percobaan | 20% | Menggunakan konsep ≥10 percobaan |
| Kualitas Kode | 15% | Clean, modular |
| Dokumentasi | 15% | README, screenshot, demo video |
| Kreativitas | 15% | Fitur tambahan, optimasi |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project09.zip`
- **Isi**: Source code, sample video, README.md, demo output.
