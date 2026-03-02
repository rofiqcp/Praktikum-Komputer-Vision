# TUGAS VIDEO PRAKTIKUM
# MODUL 11: STRUCTURE FROM MOTION DAN DEPTH ESTIMATION

---

## FORMAT PENGUMPULAN

- **Nama file**: `Video_Modul11_NIM_Nama.mp4`
- **Durasi**: 15–25 menit
- **Resolusi**: Minimal 720p
- **Audio**: Narasi jelas dalam Bahasa Indonesia

---

## 1. PEMBUKAAN (1-2 menit)

- Perkenalan: nama, NIM, mata kuliah
- Judul modul: "Structure from Motion dan Depth Estimation"
- Overview singkat: apa itu SfM, stereo vision, depth estimation
- Sebutkan tujuan pembelajaran modul ini

---

## 2. PENJELASAN MATERI (4-6 menit)

Jelaskan konsep-konsep utama:

1. **Epipolar Geometry**: epipole, epipolar plane, epipolar line, epipolar constraint ($x'^T Fx = 0$)
2. **Fundamental & Essential Matrix**: perbedaan F dan E, dekomposisi pose dari E
3. **Triangulasi**: dari korespondensi 2D ke posisi 3D
4. **Stereo Vision**: kalibrasi, rektifikasi, baseline, disparity
5. **Disparity → Depth**: formula $Z = f \cdot B / d$, hubungan invers
6. **SfM Pipeline**: feature → F → E → pose → triangulasi → bundle adjustment
7. **Monocular Depth**: tantangan dan pendekatan (heuristik vs deep learning)

---

## 3. DEMO PERCOBAAN (6-10 menit)

Demonstrasikan dan jelaskan minimal 10 dari 20 percobaan:

| No | File | Topik Demo |
|----|------|-----------|
| 1 | `01_epipolar_geometry_visualisasi.py` | Diagram epipolar geometry dan constraint |
| 2 | `02_fundamental_matrix.py` | Hitung F matrix dan gambar epipolar lines |
| 3 | `03_essential_matrix_pose.py` | Dekomposisi E → R, t |
| 4 | `04_epipolar_lines.py` | Epipolar lines dari feature matching |
| 5 | `05_triangulasi_titik_3d.py` | Triangulasi dan reprojection error |
| 6 | `06_stereo_calibration.py` | Kalibrasi stereo dengan checkerboard |
| 7 | `07_stereo_rectification.py` | Rektifikasi dan verifikasi horizontal |
| 8 | `08_block_matching_disparity.py` | Variasi parameter BM |
| 9 | `09_sgbm_disparity.py` | SGBM dan parameter P1, P2 |
| 10 | `10_monocular_depth_estimation.py` | Depth dari gradien dan heuristik |
| 11 | `11_disparity_to_depth.py` | Konversi disparity ke depth |
| 12 | `12_stereo_bm_vs_sgbm.py` | Perbandingan BM vs SGBM |
| 13 | `13_wls_filter_disparity.py` | Post-processing disparity |
| 14 | `14_point_cloud_from_depth.py` | Depth → point cloud 3D |
| 15 | `15_pnp_pose_estimation.py` | PnP pose estimation |
| 16 | `16_stereo_matching_realtime.py` | Speed benchmarking |
| 17 | `17_depth_map_colorization.py` | 6 colormap visualization |
| 18 | `18_baseline_effect_depth.py` | Analisis efek baseline |
| 19 | `19_depth_object_segmentation.py` | Segmentasi berbasis depth |
| 20 | `20_multiview_reconstruction_pipeline.py` | Pipeline SfM end-to-end |

Untuk setiap demo, jelaskan:
- Tujuan program
- Parameter / metode yang digunakan
- Interpretasi output dan visualisasi

---

## 4. DEMO PROJECT (2-4 menit)

Tunjukkan minimal 1 improvisasi dan 2 soal cerita:
- Jelaskan pendekatan dan modifikasi yang dilakukan
- Tunjukkan output program
- Bandingkan hasil jika ada variasi

---

## 5. PENUTUP (1-2 menit)

- Rangkuman konsep kunci: epipolar geometry, stereo → disparity → depth, SfM pipeline
- Kesulitan yang dihadapi dan solusinya
- Aplikasi dunia nyata: autonomous driving, 3D mapping, AR/VR
- Saran pengembangan: deep stereo, NeRF, COLMAP

---

## RUBRIK PENILAIAN VIDEO

| Komponen | Bobot | Kriteria |
|----------|-------|----------|
| Penjelasan Materi | 25% | Konsep jelas, formula benar, contoh relevan |
| Demo Percobaan | 30% | Minimal 10 demo, output ditampilkan, analisis ada |
| Demo Project | 20% | Improvisasi kreatif, soal cerita benar |
| Presentasi | 15% | Narasi jelas, pace baik, transisi smooth |
| Teknis Video | 10% | Resolusi baik, audio jelas, durasi sesuai |
